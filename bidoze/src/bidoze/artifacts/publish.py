"""Build a release bundle from a finished training phase, and ship it.

A training run leaves self-describing config snapshots and a ``.pisces``
checkpoint inside a phase directory (``<run>/training/final``). A *release* is a
different, smaller thing: one checkpoint in a documented non-executable format
plus the explicit config, metrics, provenance and model card a consumer needs to
load and trust it, laid out as :data:`BUNDLE_FILES`.

This module builds that bundle deterministically. Network upload is a separate,
optional step (:func:`publish_to_hub`), so a bundle can be built, inspected and
approved offline.

The ``.pisces`` format stays the training/checkpoint container; the public
release format is ``model.safetensors`` with a flat, slash-separated tensor map.
"""

from __future__ import annotations

import base64
import json
import math
import re
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

from bidoze.artifacts.provenance import code_digest, run_provenance, sha256_file

#: Hand-edited Markdown that ships with the package (``quickstart.md``).
TEMPLATES_DIR = Path(__file__).parent / "templates"


def _template_path(name: str) -> Path:
    return TEMPLATES_DIR / f"{name}.md"


BUNDLE_FILES = (
    "model.safetensors",
    "config.json",
    "preprocessing.json",
    "metrics.json",
    "provenance.json",
    "checksums.sha256",
    "README.md",
)

#: Where images referenced by an authored card are copied so the bundle stays
#: self-contained and the Hub upload carries them.
ASSETS_DIR = "assets"

#: Markdown image syntax. The authored card is the only place images come from;
#: a relative ``src`` is a local file to bundle, an absolute URL is left alone.
_MD_IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)")

#: Leading YAML frontmatter, stripped before a preview renders (it is Hub
#: metadata, not page content).
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

#: ``src`` attribute of an ``<img>`` in rendered HTML, for preview inlining.
_HTML_IMG_SRC = re.compile(r'(?P<pre><img\b[^>]*?\bsrc=")(?P<src>[^"]+)(?P<post>")')

#: The card's metrics-grid panels, in reading order, three to a row: the
#: sleep/wake agreement metrics, the total-sleep and WASO errors with balanced
#: accuracy, then the F1 scores.
CARD_GRID_METRICS = (
    "sleep_sensitivity",
    "wake_specificity",
    "kappa_sleep_wake",
    "tst_error_minutes",
    "waso_error_minutes",
    "balanced_accuracy",
    "f1_sleep",
    "f1_wake",
    "f1_macro",
)
CARD_GRID_COLUMNS = 3
#: Bounded scores drawn on their full range, so bar heights compare honestly.
CARD_GRID_YLIMS = {
    "sleep_sensitivity": (0.0, 100.0),
    "wake_specificity": (0.0, 100.0),
    "kappa_sleep_wake": (0.0, 1.0),
    "balanced_accuracy": (0.0, 100.0),
    "f1_sleep": (0.0, 100.0),
    "f1_wake": (0.0, 100.0),
    "f1_macro": (0.0, 100.0),
}

#: The config snapshots a phase writes, and what each becomes in the bundle.
_CONFIG_TO_BUNDLE = {
    "model.json": "config.json",
    "inputs.json": "preprocessing.json",
}

#: Human-review fields the card must carry before a release is approved. A
#: generated card that still says PENDING is a bundle that must not ship.
CARD_REVIEW_PENDING = "PENDING"


@dataclass(frozen=True)
class ReleaseInfo:
    """What distinguishes one model variant's release, read from ``--info``.

    One TOML file per variant (one Hub repo per variant, versions as tags), so
    nothing about an audience is baked into the code. Every key is optional:

    ``repo``           Hub repository the card is titled for, e.g. ``org/name``
    ``card``           authored Markdown, relative to the TOML (what ``--card`` takes)
    ``audience``       the glance table's "For" row
    ``protocol``       paragraph describing how the reported numbers were produced
    ``require_cv``     the glance score must come from the pooled ``cv`` cohort
    ``[cohorts]``      cohort name -> one-line description for the performance section
    """

    repo: str | None = None
    card: Path | None = None
    audience: str | None = None
    protocol: str | None = None
    require_cv: bool = False
    cohorts: Mapping[str, str] = field(default_factory=dict)


_INFO_KEYS = frozenset({"repo", "card", "audience", "protocol", "require_cv", "cohorts"})


def load_release_info(path: Path | str) -> ReleaseInfo:
    """Parse a release-info TOML; unknown keys are an error so typos surface."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such release info file: {path}")
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    unknown = sorted(set(raw) - _INFO_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; expected {sorted(_INFO_KEYS)}")
    return ReleaseInfo(
        repo=raw.get("repo"),
        card=path.parent / raw["card"] if "card" in raw else None,
        audience=raw.get("audience"),
        protocol=raw.get("protocol"),
        require_cv=bool(raw.get("require_cv", False)),
        cohorts=dict(raw.get("cohorts", {})),
    )


@dataclass(frozen=True)
class ReleaseBundle:
    """A built bundle on disk, with the facts its card and manifest record."""

    directory: Path
    version: str
    checkpoint_sha256: str
    input_shape: tuple[int, ...]
    files: tuple[str, ...]
    card_review_status: str

    @property
    def approved(self) -> bool:
        return self.card_review_status != CARD_REVIEW_PENDING


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _config_dir(training_dir: Path) -> Path:
    config_dir = training_dir / "config"
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"{training_dir} is not a training phase directory; no config/ under it"
        )
    return config_dir


def _resolve_checkpoint(training_dir: Path, checkpoint: Path | str | None) -> Path:
    """The phase's ``.pisces`` checkpoint, or the one named explicitly.

    The canonical location is ``<phase>/model_outputs/*.pisces``.
    """
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"no such checkpoint: {path}")
        return path
    if not training_dir.is_dir():
        # A relative -t resolved against another working directory is a likely
        # cause of the phase appearing empty, so report where it looked.
        raise FileNotFoundError(
            f"training phase not found: {training_dir} (resolved to {training_dir.resolve()})"
        )
    model_outputs = training_dir / "model_outputs"
    found = sorted(model_outputs.glob("*.pisces"))
    if not found:
        # Fall back to a checkpoint elsewhere in the phase, in case the
        # canonical model_outputs/ is absent.
        found = sorted(training_dir.rglob("*.pisces"))
    if len(found) != 1:
        where = model_outputs if model_outputs.is_dir() else training_dir
        detail = "none" if not found else f"{len(found)}"
        raise FileNotFoundError(
            f"expected one .pisces under {where}, found {detail}; name one with --checkpoint"
        )
    return found[0]


def _load_phase_configs(training_dir: Path) -> dict[str, dict]:
    config_dir = _config_dir(training_dir)
    return {name: _read_json(config_dir / name) for name in _CONFIG_TO_BUNDLE}


def _config_hashes(training_dir: Path) -> dict[str, str]:
    config_dir = _config_dir(training_dir)
    hashes: dict[str, str] = {}
    for path in sorted(config_dir.glob("*.json")):
        hashes[path.name] = sha256_file(path)
    return hashes


def _split_manifest_sha256(training_dir: Path) -> str | None:
    path = _config_dir(training_dir) / "split-manifest.json"
    return sha256_file(path) if path.is_file() else None


def _evaluation_for(training_dir: Path, evaluation: Path | str | None) -> dict | None:
    """The evaluation.json to fold into provenance, or one named explicitly.

    A phase writes its evaluation.json next to its own artifacts, so the default
    is ``<phase>/evaluation.json``. When the scoring was run elsewhere, name it
    with ``--evaluation``. Returns ``None`` when neither exists.
    """
    path = Path(evaluation) if evaluation is not None else training_dir / "evaluation.json"
    if evaluation is not None and not path.is_file():
        raise FileNotFoundError(f"no such evaluation file: {path}")
    return _read_json(path) if path.is_file() else None


def _summary_metrics(metrics_summary: Path | str | None) -> Mapping[str, Any] | None:
    """The macro-averaged metrics a summary CSV holds, keyed by cohort."""
    if metrics_summary is None:
        return None
    import csv

    path = Path(metrics_summary)
    if not path.is_file():
        raise FileNotFoundError(f"no such metrics summary: {path}")

    by_cohort: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cohort = row["cohort"]
            metric = row["metric"]
            entry = {
                "n": _as_number(row.get("n")),
                "mean": _as_number(row.get("mean")),
                "median": _as_number(row.get("median")),
                "ci95_lo": _as_number(row.get("ci95_lo")),
                "ci95_hi": _as_number(row.get("ci95_hi")),
            }
            by_cohort.setdefault(cohort, {})[metric] = entry
    return by_cohort


def _as_number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _grid_table(metrics_summary: Path | str) -> Any:
    """The metrics summary as a DataFrame, for drawing the comparison grid.

    Read through :func:`load_summaries`, so a file that is not a metrics summary
    (missing the mean/CI columns the figure needs) is refused up front rather
    than failing later inside matplotlib.
    """
    try:
        from bidoze.reporting.compare import load_summaries
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "drawing the model card's metrics grid needs the optional "
            "dependency: pip install 'arcasleep-bidoze[publish]'",
            name=exc.name,
        ) from exc

    path = Path(metrics_summary)
    if not path.is_file():
        raise FileNotFoundError(f"no such metrics summary: {path}")
    return load_summaries([path])[0].table


def _label_map(config: Mapping[str, Any]) -> dict[str, Any]:
    num_classes = int(config.get("num_classes", 4))
    names = ["wake", "light", "deep", "rem", "gap"][: num_classes + 1]
    return {"num_classes": num_classes, "class_names": names, "gap_class": "gap"}


def _discovered_metrics(
    training_dir: Path,
) -> tuple[Any | None, tuple[str, ...]]:
    """Pool the phase's own CV results into a summary table, or ``(None, ())``.

    A phase directory may hold its own ``cv_results_<n>_stages.csv`` and sit
    beside the ``fold<i>`` directories of the cross-validation. The phase's
    datasets become cohorts of their own and every sibling fold is pooled into a
    single ``cv`` cohort, so the card reports the held-out score without the
    user having to run ``summarize`` and ``compare`` first.
    """
    try:
        from bidoze.reporting.summary import (
            FOLD_DIR,
            build_cohorts,
            discover_phases,
            summary_table,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scoring the model card from the phase's results needs the optional "
            "dependency: pip install 'arcasleep-bidoze[publish]'",
            name=exc.name,
        ) from exc

    training_dir = Path(training_dir)
    roots = [training_dir]
    parent = training_dir.parent
    if parent.is_dir() and parent != training_dir:
        roots += [
            path for path in sorted(parent.iterdir()) if path.is_dir() and FOLD_DIR.match(path.name)
        ]
    try:
        phases = discover_phases(roots)
    except ValueError:
        return None, ()
    table = summary_table(build_cohorts(phases))
    if table.empty:
        return None, ()
    return table, tuple(sorted({str(phase.csv_path) for phase in phases}))


def _summary_by_cohort(table: Any) -> dict[str, dict[str, dict[str, Any]]]:
    """Reshape a summary table into the card's ``cohort -> metric -> stats``."""
    by_cohort: dict[str, dict[str, dict[str, Any]]] = {}
    for row in table.to_dict("records"):
        by_cohort.setdefault(str(row["cohort"]), {})[str(row["metric"])] = {
            "n": row.get("n"),
            "mean": row.get("mean"),
            "median": row.get("median"),
            "ci95_lo": row.get("ci95_lo"),
            "ci95_hi": row.get("ci95_hi"),
        }
    return by_cohort


def _draw_metrics_grid(table: Any, output_dir: Path) -> str:
    """Draw the summary's headline metrics into a PNG under :data:`ASSETS_DIR`.

    One panel per metric, cohorts across the x-axis, so the card carries the
    same figure ``bidoze summarize`` and ``bidoze compare`` would produce.
    Returns the bundle-relative asset path. Imported lazily: drawing needs the
    ``publish`` extra's matplotlib/pandas, which the base install does not carry.
    """
    try:
        from bidoze.reporting.compare import RunSummary, draw_comparison
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "drawing the model card's metrics grid needs the optional "
            "dependency: pip install 'arcasleep-bidoze[publish]'",
            name=exc.name,
        ) from exc

    assets_dir = output_dir / ASSETS_DIR
    assets_dir.mkdir(parents=True, exist_ok=True)
    cohorts = {str(c) for c in table["cohort"].unique()}
    label = "5-fold CV + hold-out" if "cv" in cohorts else "hold-out"
    run = RunSummary(path=output_dir, label=label, table=table)
    written = draw_comparison(
        [run],
        output_dir=assets_dir,
        name="metrics_grid",
        metrics=CARD_GRID_METRICS,
        ncols=CARD_GRID_COLUMNS,
        ylims=CARD_GRID_YLIMS,
    )
    png = assets_dir / "metrics_grid.png"
    for path in written:
        if path != png and path.is_file():
            path.unlink()
    return f"{ASSETS_DIR}/{png.name}"


def _describe_architecture(config: Mapping[str, Any], parameter_count: int | None) -> list[str]:
    """A few lines on what the network is, every number read from its config."""
    d_model = config.get("d_model", 256)
    n_heads = config.get("n_heads", 8)
    n_layers = config.get("n_transformer_layers", 4)
    kernels = ", ".join(str(k) for k in config.get("conv_kernels", [3, 7, 31, 63]))
    direction = "bi-causal" if config.get("use_bicausal", True) else "causal"
    size = f" ({parameter_count / 1e6:.2f} M parameters)" if parameter_count else ""
    return [
        f"A convolution bank (kernels `{kernels}` epochs) captures local movement "
        f"patterns; a {direction} transformer (`{n_layers}` layers per direction, "
        f"width `{d_model}`, `{n_heads}` heads) carries context across the whole "
        f"recording; a learned gate fuses the two before a per-epoch readout{size}.",
    ]


def _spectral_bins(preprocessing: Mapping[str, Any]) -> int | None:
    """How many frequency bins the spectrogram keeps: ``1 / window`` steps in band."""
    try:
        window = float(preprocessing["window_seconds"])
        fmin, fmax = float(preprocessing["fmin"]), float(preprocessing["fmax"])
    except (KeyError, TypeError, ValueError):
        return None
    eps = 1e-9
    return int(math.floor(fmax * window + eps) - math.ceil(fmin * window - eps) + 1)


def _describe_preprocessing(
    preprocessing: Mapping[str, Any], input_shape: Sequence[int]
) -> list[str]:
    """One line on the feature surface, from the phase's ``inputs.json``."""
    names = preprocessing.get("spectral_channels", ["mag", "jerk"])
    channels = " and ".join(f"`{c}`" for c in names)
    shape = f" The model input for one pass is `{tuple(input_shape)}`." if input_shape else ""
    bins = _spectral_bins(preprocessing)
    bins_text = f"{bins} bins, " if bins is not None else ""
    return [
        f"`{preprocessing.get('spectrogram_kind', 'log_psd')}` spectrograms of the "
        f"{channels} channels over `{preprocessing.get('window_seconds', '?')}`-s "
        f"windows every `{preprocessing.get('window_step_seconds', '?')}` s "
        f"({bins_text}`{preprocessing.get('fmin', '?')}`–`{preprocessing.get('fmax', '?')}` Hz), "
        "averaged to 30-s epochs, plus one row giving the fraction of each epoch "
        "with no data." + shape + " `bidoze.predict` builds it from "
        "`preprocessing.json`.",
    ]


#: How the card names and formats each metric: (label, unit, higher_is_better).
#: Mirrors ``bidoze.reporting.compare.PANELS`` (the grid figure), kept here so
#: writing the card does not need the plotting extra. Order is reading order:
#: these rows make the headline table; anything else lands in the detail tables.
_METRIC_LABELS: dict[str, tuple[str, str, bool | None]] = {
    "kappa_sleep_wake": ("Cohen's κ, sleep/wake", "", True),
    "sleep_sensitivity": ("Sleep sensitivity", "%", True),
    "wake_specificity": ("Wake specificity", "%", True),
    "balanced_accuracy": ("Balanced accuracy, sleep/wake", "%", True),
    "f1_sleep": ("Sleep F1", "%", True),
    "f1_wake": ("Wake F1", "%", True),
    "f1_macro": ("Macro F1, sleep/wake mean", "%", True),
    "tst_error_minutes": ("TST error", "min", None),
    "waso_error_minutes": ("WASO error", "min", None),
    "tst_mape": ("TST error", "%", False),
    "waso_mape": ("WASO error", "%", False),
    "kappa_4stage": ("Cohen's κ, 4-stage", "", True),
    "light_mape": ("Light-sleep error", "%", False),
    "deep_mape": ("Deep-sleep error", "%", False),
    "rem_mape": ("REM-sleep error", "%", False),
}

#: Metrics left out of the card's performance tables.
CARD_TABLE_EXCLUDED_METRICS = frozenset({"kappa_4stage"})

#: The pooled cross-validation cohort :func:`_discovered_metrics` builds.
_CV_COHORT = "cv"

#: Epoch length the readout scores, in seconds.
_EPOCH_SECONDS = 30

#: What each readout index means, for the labels table.
_CLASS_MEANINGS = {
    "wake": "Wake",
    "light": "Light sleep",
    "deep": "Deep sleep",
    "rem": "REM sleep",
    "gap": "No score: too little accelerometer coverage. Not a sleep stage.",
}

#: One line per bundle file, for the card's file table.
_FILE_DESCRIPTIONS = {
    "model.safetensors": "Weights, as a flat slash-separated tensor map",
    "config.json": "Architecture settings; rebuilds the network",
    "preprocessing.json": "Spectrogram settings the inputs must match",
    "metrics.json": "The scoring protocol behind the evaluation numbers",
    "provenance.json": "Code, config, split and checkpoint hashes",
    "checksums.sha256": "SHA-256 of every file in this release",
}


def _metric_label(metric: str) -> str:
    """``Wake specificity (%) — higher is better``: name, unit and direction."""
    if metric in _METRIC_LABELS:
        label, unit, higher = _METRIC_LABELS[metric]
    else:
        text = metric.replace("_", " ")
        label, unit, higher = text[:1].upper() + text[1:], "", None
    text = f"{label} ({unit})" if unit else label
    return text + {True: " — higher is better", False: " — lower is better"}.get(higher, "")


def _fmt_metric(metric: str, value: Any) -> str:
    """A metric value at the precision it is read at, not a fixed 3 decimals."""
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    value = float(value)
    if metric.startswith("kappa"):
        return f"{value:.2f}"
    if metric in ("tst_error_minutes", "waso_error_minutes"):
        return f"{value:+.1f}".replace("-", "−")
    if metric.endswith("_count"):
        return f"{value:.0f}" if abs(value) >= 10 or value == 0 else f"{value:.2f}"
    return f"{value:.1f}"


def _fmt_ci(metric: str, entry: Mapping[str, Any]) -> str | None:
    """``0.50 to 0.53``, or ``None`` when the summary carries no interval."""
    lo, hi = entry.get("ci95_lo"), entry.get("ci95_hi")
    if lo is None or hi is None:
        return None
    return f"{_fmt_metric(metric, lo)} to {_fmt_metric(metric, hi)}"


def _cohort_title(cohort: str) -> str:
    return "Cross-validation" if cohort == _CV_COHORT else cohort


def _ordered_cohorts(summary: Mapping[str, Any]) -> list[str]:
    """Pooled CV first -- it is the headline -- then hold-outs by name."""
    return sorted(summary, key=lambda c: (c != _CV_COHORT, c))


def _cohort_n(values: Mapping[str, Mapping[str, Any]]) -> str:
    counts = [entry.get("n") for entry in values.values() if entry.get("n") is not None]
    return _fmt_n(max(counts)) if counts else "?"


def _ordered_metrics(values: Mapping[str, Any], declared: Sequence[str]) -> list[str]:
    """Known metrics in reading order, then declared ones, then the rest."""
    ordered = [m for m in _METRIC_LABELS if m in values]
    ordered += [m for m in declared if m in values and m not in ordered]
    ordered += [m for m in sorted(values) if m not in ordered]
    return ordered


def _headline_table(summary: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Headline metrics down, cohorts across: mean with its 95% CI beneath."""
    cohorts = _ordered_cohorts(summary)
    metrics = [
        m
        for m in _METRIC_LABELS
        if m not in CARD_TABLE_EXCLUDED_METRICS and any(m in summary[c] for c in cohorts)
    ]
    if not metrics:
        return []
    header = (
        "| Metric | "
        + " | ".join(
            f"{_cohort_title(c)}<br><sub>n = {_cohort_n(summary[c])}</sub>" for c in cohorts
        )
        + " |"
    )
    lines = [header, "|:---|" + "---:|" * len(cohorts)]
    for metric in metrics:
        cells = []
        for cohort in cohorts:
            entry = summary[cohort].get(metric)
            if entry is None:
                cells.append("n/a")
                continue
            cell = f"**{_fmt_metric(metric, entry.get('mean'))}**"
            ci = _fmt_ci(metric, entry)
            if ci is not None:
                cell += f"<br><sub>{ci}</sub>"
            cells.append(cell)
        lines.append(f"| {_metric_label(metric)} | " + " | ".join(cells) + " |")
    return lines


def _glance_score(
    summary: Mapping[str, Mapping[str, Any]] | None, *, require_cv: bool = False
) -> str | None:
    """One line for the at-a-glance table: the headline κ on its headline cohort.

    With ``require_cv`` the card must not silently substitute an external
    holdout when the pooled cross-validation results are missing.
    """
    if not summary:
        return None
    if require_cv and _CV_COHORT not in summary:
        return None
    cohort = _CV_COHORT if _CV_COHORT in summary else _ordered_cohorts(summary)[0]
    values = summary[cohort]
    parts = []
    for metric, name in (("kappa_sleep_wake", "sleep/wake"), ("kappa_4stage", "4-stage")):
        entry = values.get(metric)
        if entry and entry.get("mean") is not None:
            parts.append(f"**{_fmt_metric(metric, entry['mean'])}** {name}")
    if not parts:
        return None
    return (
        "Cohen's κ "
        + " · ".join(parts)
        + f" ({_cohort_title(cohort).lower()}, {_cohort_n(values)} subjects)"
    )


def _quickstart(repo_id: str, local_dir: str, stage_names: Sequence[str]) -> list[str]:
    """Raw accelerometer file in, per-epoch stages out: the card's main use case.

    The prose lives in ``templates/quickstart.md`` so it can be edited as
    Markdown; ``$repo_id``, ``$local_dir``, ``$columns`` and ``$n_stages`` are
    filled from the bundle.
    """
    text = Template(_template_path("quickstart").read_text()).safe_substitute(
        repo_id=repo_id,
        local_dir=local_dir,
        columns=", ".join(stage_names),
        n_stages=len(stage_names),
    )
    return [*text.rstrip("\n").split("\n"), ""]


def _input_requirements(preprocessing: Mapping[str, Any], input_shape: Sequence[int]) -> list[str]:
    """What a recording must look like; the sampling floor is read from the config."""
    fs = preprocessing.get("fs")
    lines = [
        "## Input requirements",
        "",
        "- **Columns.** Timestamp in seconds, then x, y, z acceleration in g "
        "(divide m/s² by 9.81), from a wrist-worn triaxial accelerometer.",
    ]
    if fs is not None:
        lines.append(
            f"- **Sampling rate.** At least {float(fs):g} Hz; slower recordings are "
            "rejected. Irregular sampling is fine."
        )
    lines.append(
        "- **Missing data.** Leave gaps as gaps: never fill them with zeros or "
        "synthetic rows. Epochs with no samples come back with `is_gap` set; "
        "leave them out of sleep durations."
    )
    if input_shape:
        hours = input_shape[0] * _EPOCH_SECONDS / 3600
        lines.append(
            f"- **Length.** At least 2 hours recommended. Up to {hours:g} hours are "
            "scored in one pass; longer recordings are scored in consecutive passes."
        )
    lines.append("")
    return lines


def _outputs(stage_names: Sequence[str]) -> list[str]:
    lines = [
        "## Outputs",
        "",
        "Per 30-second epoch, one probability per stage, in this order:",
        "",
        "| Index | Stage | Meaning |",
        "|---:|:---|:---|",
    ]
    lines += [
        f"| {i} | `{name}` | {_CLASS_MEANINGS.get(name, name)} |"
        for i, name in enumerate(stage_names)
    ]
    lines += [
        "",
        "`is_gap` is not a model output: it marks epochs with no accelerometer samples.",
        "",
    ]
    return lines


def _performance(
    info: ReleaseInfo,
    metrics: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    grid_asset: str | None,
) -> list[str]:
    """The metrics grid first; full tables collapsed beneath it."""
    lines = ["## Performance", ""]
    if not summary:
        if evaluation:
            return lines + [
                f"Scored {evaluation.get('subject_count', '?')} subjects; see "
                "`metrics.json` for the protocol.",
                "",
            ]
        return lines + [
            "No results CSV (`cv_results_*_stages.csv`) accompanied this phase, so "
            "no scores are reported here. The scoring protocol itself is pinned in "
            "`metrics.json`.",
            "",
        ]
    if grid_asset:
        lines += [f"![Metric comparison]({grid_asset})", ""]
    lines += [
        "Metrics compare model predictions with reference PSG scoring, macro-"
        "averaged over subjects, with 95% confidence intervals of the mean.",
        "",
    ]
    described = [c for c in _ordered_cohorts(summary) if c in info.cohorts]
    lines += [f"- **{_cohort_title(c)}:** {info.cohorts[c]}." for c in described]
    if described:
        lines.append("")
    if info.protocol:
        lines += [f"**Protocol.** {info.protocol.strip()}", ""]

    declared = list(metrics.get("metrics", []))
    lines += [
        "<details>",
        "<summary>All metrics, per cohort</summary>",
        "",
        "'Higher is better' and 'lower is better' describe how to read each "
        "metric; they do not indicate improvement over a control or baseline "
        "model or a comparison between cohorts.",
        "",
    ]
    lines += _headline_table(summary)
    lines.append("")
    for cohort in _ordered_cohorts(summary):
        values = {
            m: v
            for m, v in summary[cohort].items()
            if not m.endswith("_count") and m not in CARD_TABLE_EXCLUDED_METRICS
        }
        lines += [
            f"### {_cohort_title(cohort)}",
            "",
            "| Metric | n | Mean | 95% CI | Median |",
            "|:---|---:|---:|---:|---:|",
        ]
        for metric in _ordered_metrics(values, declared):
            entry = values[metric]
            lines.append(
                f"| {_metric_label(metric)}<br><sub>`{metric}`</sub> "
                f"| {_fmt_n(entry.get('n'))} "
                f"| {_fmt_metric(metric, entry.get('mean'))} "
                f"| {_fmt_ci(metric, entry) or 'n/a'} "
                f"| {_fmt_metric(metric, entry.get('median'))} |"
            )
        lines.append("")
    lines += ["</details>", ""]
    return lines


def _model_card(
    *,
    repo_id: str,
    version: str,
    config: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    authored_card: str | None,
    info: ReleaseInfo = ReleaseInfo(),
    grid_asset: str | None = None,
    parameter_count: int | None = None,
    assets: Sequence[str] = (),
) -> str:
    """Assemble the model card: how to run it, how well it does, then run facts.

    The card opens with an at-a-glance table, then ``authored_card`` -- the
    human-written Markdown supplied via ``publish --card``, inserted verbatim --
    then generated sections: quickstart, performance, input requirements,
    outputs, and collapsed model, file and provenance details. Every generated
    fact is read from the bundle, so it cannot drift from the release.

    ``grid_asset`` is the bundle-relative path of the metrics-grid figure (or
    ``None`` when no results CSV was found). The title and metadata follow the
    destination ``repo_id``, so a variant in its own repository is labelled as
    that variant.
    """
    reviewed = authored_card is not None
    labels = _label_map(config)
    stage_names = labels["class_names"][: labels["num_classes"]]
    input_shape = tuple(provenance.get("input_shape") or ())
    local_dir = repo_id.rsplit("/", 1)[-1] or "bidoze"

    lines: list[str] = []
    # Hub YAML frontmatter: without it the license and task tags would be absent
    # from the published model page, and replacing the repository README with a
    # frameless one would drop the license metadata the Hub already carries.
    lines += [
        "---",
        "license: mit",
        "tags:",
        "- sleep-staging",
        "- actigraphy",
        "- accelerometer",
        "- time-series",
        "---",
        "",
        f"# {repo_id or 'bidoze'} · {version}",
        "",
        "**Sleep stages from triaxial accelerometer recordings.**  ",
        "4-class sleep stage prediction, metric reports, and hypnograms. ",
        "Built with a focus on sleep-wake.",
    ]

    fs = preprocessing.get("fs")
    glance = [
        ("Predicts", ", ".join(f"`{c}`" for c in stage_names) + " per 30-s epoch"),
        (
            "Input",
            "Timestamped triaxial accelerometer"
            + (f", {float(fs):g} Hz or faster" if fs is not None else ""),
        ),
    ]
    if info.audience:
        glance.insert(1, ("For", info.audience))
    score = _glance_score(summary, require_cv=info.require_cv)
    if score:
        glance.append(("Score", score))
    glance.append(
        (
            "Review",
            "✅ Reviewed" if reviewed else f"⚠️ **{CARD_REVIEW_PENDING}**: not approved to ship",
        )
    )
    glance.append(("License", "MIT"))
    # The first row doubles as the header, so the table has no empty header
    # band when the Hub renders it.
    lines.append(f"| **{glance[0][0]}** | {glance[0][1]} |")
    lines.append("|:---|:---|")
    lines += [f"| **{key}** | {value} |" for key, value in glance[1:]]
    lines.append("")
    lines += _quickstart(repo_id, local_dir, stage_names)
    lines += _performance(info, metrics, evaluation, summary, grid_asset)

    if reviewed:
        lines += [authored_card.strip(), ""]
    else:
        lines += [
            "> ⚠️ **Review status: "
            + CARD_REVIEW_PENDING
            + ".** A reviewer must approve this card with `publish --card` before "
            "the bundle may ship.",
            "",
        ]

    lines += _input_requirements(preprocessing, input_shape)
    lines += _outputs(stage_names)

    lines += ["## Model", ""]
    lines += _describe_architecture(config, parameter_count)
    lines.append("")
    lines += _describe_preprocessing(preprocessing, input_shape)
    lines.append("")

    lines += [
        "<details>",
        "<summary>Files and provenance</summary>",
        "",
        "| File | Contents |",
        "|:---|:---|",
    ]
    for name in BUNDLE_FILES:
        if name in _FILE_DESCRIPTIONS:
            lines.append(f"| `{name}` | {_FILE_DESCRIPTIONS[name]} |")
    for name in assets:
        lines.append(f"| `{name}` | Figure shown on this card |")
    lines += ["", "| Field | Value |", "|:---|:---|"]
    for key in (
        "package_version",
        "git_commit",
        "container_image_digest",
        "seed",
        "run_id",
        "fold",
        "epoch",
        "trained_on",
        "checkpoint_sha256",
        "split_manifest_sha256",
        "model_config_sha256",
        "processing_config_sha256",
        "training_config_sha256",
    ):
        if key in provenance and provenance[key] is not None:
            lines.append(f"| {key} | `{provenance[key]}` |")
    lines += [
        "",
        "Check a download with `sha256sum -c checksums.sha256`.",
        "",
        "</details>",
        "",
        f"<sub>Generated by `bidoze publish` from `{provenance.get('git_commit', '?')}`.</sub>",
        "",
    ]
    return "\n".join(lines)


def _fmt_n(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _checksums(directory: Path, files: Sequence[str]) -> str:
    lines = []
    for name in sorted(files):
        if name == "checksums.sha256":
            continue
        lines.append(f"{sha256_file(directory / name)}  {name}")
    return "\n".join(lines) + "\n"


def _is_remote_image(src: str) -> bool:
    """True for an image the bundle does not need to carry.

    A URL, an inline ``data:`` URI or a bare ``#fragment`` is self-contained or
    hosted elsewhere. A plain path (relative or absolute) is a local file the
    bundle must copy.
    """
    return bool(re.match(r"^[a-z][a-z0-9+.-]*://", src)) or src.startswith(("data:", "#"))


def _collect_assets(
    card_text: str, card_dir: Path, output_dir: Path
) -> tuple[str, tuple[str, ...]]:
    """Copy images the authored card references into :data:`ASSETS_DIR`.

    A relative ``src`` is read from next to the card file, copied into the
    bundle's :data:`ASSETS_DIR`, and the link is rewritten to point there, so the
    published card resolves on the Hub without the author's machine. A remote URL
    (``https://``, ``data:`` or an already-bundled :data:`ASSETS_DIR` path) is left
    alone. A relative path that does not resolve is an error: silently dropping a
    figure the author asked for would ship a card with broken images.
    """
    assets: list[str] = []
    #: destination basename -> source, to detect and disambiguate name clashes.
    used: dict[str, Path] = {}

    def rewrite(match: re.Match[str]) -> str:
        src = match.group("src")
        if _is_remote_image(src) or src.startswith(f"{ASSETS_DIR}/"):
            return match.group(0)
        source = (card_dir / src).resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"card image {src!r} not found; expected it next to the card at {(card_dir / src)}"
            )
        name = source.name
        if name in used and used[name] != source:
            digest = sha256_file(source)[:8]
            name = f"{source.stem}-{digest}{source.suffix}"
        used[name] = source
        dest = output_dir / ASSETS_DIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file():
            shutil.copyfile(source, dest)
        rel = f"{ASSETS_DIR}/{name}"
        if rel not in assets:
            assets.append(rel)
        return match.group(0).replace(src, rel, 1)

    return _MD_IMAGE.sub(rewrite, card_text), tuple(assets)


def _bundle_relpaths(directory: Path) -> list[str]:
    """The files :func:`verify_bundle` accepted, as slash-relative paths.

    Read from ``checksums.sha256`` rather than a directory walk: the upload must
    carry exactly the verified release, not a local preview or scratch file a
    user happened to leave in the bundle directory.
    """
    checksums = directory / "checksums.sha256"
    names = [line.partition("  ")[2] for line in checksums.read_text().splitlines() if line.strip()]
    # checksums.sha256 does not list itself, but it ships with the release.
    names.append("checksums.sha256")
    return sorted(names)


def _inline_assets(html: str, bundle_dir: Path) -> str:
    """Replace bundled-asset image srcs with data URIs so a preview is portable.

    Operates on the rendered HTML, so it matches the ``src`` attribute rather
    than Markdown image syntax.
    """

    def inline(match: re.Match[str]) -> str:
        src = match.group("src")
        path = bundle_dir / src
        if _is_remote_image(src) or not path.is_file():
            return match.group(0)
        suffix = path.suffix.lower()
        mime = {".png": "image/png", ".svg": "image/svg+xml"}.get(suffix, "image/jpeg")
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return match.group(0).replace(src, f"data:{mime};base64,{payload}", 1)

    return _HTML_IMG_SRC.sub(inline, html)


def render_card_html(bundle_dir: Path | str) -> str:
    """Render a built bundle's ``README.md`` to a self-contained HTML page.

    This is a local preview only: it is not part of the released bundle. Images
    referenced from :data:`ASSETS_DIR` are inlined as data URIs so the page opens
    standalone and matches what the Hub will render. Requires the ``preview``
    extra (``markdown-it-py``); imports it lazily so the base install is
    unaffected.
    """
    bundle_dir = Path(bundle_dir)
    card_path = bundle_dir / "README.md"
    if not card_path.is_file():
        raise FileNotFoundError(f"{bundle_dir} has no README.md to preview")

    try:
        from markdown_it import MarkdownIt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rendering a card preview needs the optional dependency: "
            "pip install 'arcasleep-bidoze[preview]'",
            name="markdown_it",
        ) from exc

    body = _FRONTMATTER.sub("", card_path.read_text(), count=1)
    parser = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    rendered = parser.render(body)
    rendered = _inline_assets(rendered, bundle_dir)
    heading = re.search(r"<h1[^>]*>(.*?)</h1>", rendered, re.DOTALL)
    title = re.sub(r"<[^>]+>", "", heading.group(1)).strip() if heading else ""
    # Title first: the body is free text and may itself contain "{title}".
    page = _PREVIEW_TEMPLATE.replace("{title}", title or "Model card preview")
    return page.replace("{body}", rendered)


#: Self-contained page around the rendered card. The content is exactly what
#: the Hub renders; the styling only makes the local read pleasant -- a spec
#: sheet for the at-a-glance table, tabular figures for scores, and light/dark
#: themes -- without adding anything the published card lacks.
_PREVIEW_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f6f5f1; --surface: #fcfcfb; --ink: #1a1a19; --muted: #6b6a65;
    --rule: #e1e0d9; --rule-strong: #c3c2b7; --tint: #f1f0ea;
    --accent: #2a78d6; --accent-soft: #e6effa;
    --warn: #8a5a00; --warn-soft: #fdf3dc; --warn-rule: #eda100;
    --code-bg: #f1f0ec; --pre-bg: #1f1f1e; --pre-ink: #ecebe6;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg: #141413; --surface: #1c1c1b; --ink: #e8e7e2; --muted: #9d9c96;
      --rule: #33332f; --rule-strong: #4a4a46; --tint: #232321;
      --accent: #6aa6ec; --accent-soft: #1d2a3a;
      --warn: #f2c261; --warn-soft: #2d2412; --warn-rule: #b67d00;
      --code-bg: #2a2a28; --pre-bg: #0f0f0e; --pre-ink: #e8e7e2;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #141413; --surface: #1c1c1b; --ink: #e8e7e2; --muted: #9d9c96;
    --rule: #33332f; --rule-strong: #4a4a46; --tint: #232321;
    --accent: #6aa6ec; --accent-soft: #1d2a3a;
    --warn: #f2c261; --warn-soft: #2d2412; --warn-rule: #b67d00;
    --code-bg: #2a2a28; --pre-bg: #0f0f0e; --pre-ink: #e8e7e2;
  }
  * { box-sizing: border-box; }
  html { background: var(--bg); }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
               "Helvetica Neue", sans-serif;
         -webkit-font-smoothing: antialiased; }
  main { max-width: 60rem; margin: 2.5rem auto 5rem; padding: 2.5rem 3rem 3rem;
         background: var(--surface); border: 1px solid var(--rule); border-radius: 14px; }
  @media (max-width: 720px) {
    main { margin: 0; padding: 1.5rem 16px 3rem; border: 0; border-radius: 0; }
  }

  h1 { font-size: 1.9rem; line-height: 1.2; letter-spacing: -0.01em; margin: 0 0 .6rem;
       overflow-wrap: anywhere; }
  h1 + p { font-size: 1.08rem; color: var(--muted); margin: 0 0 1.6rem; }
  h1 + p strong { color: var(--ink); font-weight: 600; }
  h2 { font-size: 1.3rem; letter-spacing: -0.005em; margin: 2.8rem 0 .9rem;
       padding-top: 1.4rem; border-top: 1px solid var(--rule); }
  h3 { font-size: 1.02rem; margin: 1.8rem 0 .5rem; }
  p, ul, ol { margin: .7rem 0; }
  li { margin: .3rem 0; }
  li::marker { color: var(--muted); }
  a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
  sub { font-size: .78em; color: var(--muted); vertical-align: baseline; }

  code { font-family: var(--mono); font-size: .86em; background: var(--code-bg);
         padding: .12rem .35rem; border-radius: 4px; overflow-wrap: anywhere; }
  pre { background: var(--pre-bg); color: var(--pre-ink); padding: 1rem 1.2rem;
        border-radius: 10px; overflow-x: auto; line-height: 1.5; margin: 1rem 0; }
  pre code { background: none; padding: 0; font-size: .86rem; color: inherit;
             overflow-wrap: normal; }

  .table-wrap { overflow-x: auto; margin: 1rem 0; }
  table { border-collapse: collapse; width: 100%; font-size: .9rem;
          font-variant-numeric: tabular-nums; }
  th, td { padding: .5rem .75rem; border-bottom: 1px solid var(--rule);
           text-align: left; vertical-align: top; }
  thead th { font-weight: 600; font-size: .8rem; color: var(--muted);
             border-bottom: 1px solid var(--rule-strong); white-space: nowrap; }
  thead th sub { display: block; }
  tbody tr:hover { background: var(--tint); }
  td sub { display: block; line-height: 1.3; }

  /* The at-a-glance table: its first row is a header by Markdown's rules,
     but it reads as a spec sheet, so it is styled as one card. */
  .glance { border: 1px solid var(--rule); border-radius: 10px; overflow: hidden;
            margin: 0 0 2rem; }
  .glance table { font-size: .93rem; }
  .glance th, .glance td { padding: .6rem 1rem; border-bottom: 1px solid var(--rule);
                           color: var(--ink); font-size: inherit; white-space: normal; }
  .glance tr:last-child td { border-bottom: 0; }
  .glance th:first-child, .glance td:first-child { width: 8.5rem; background: var(--tint);
                                                  font-weight: 600; }
  .glance th strong, .glance td:first-child strong { font-weight: 600; }
  .glance tbody tr:hover { background: none; }

  blockquote { margin: 1rem 0; padding: .85rem 1.1rem; background: var(--warn-soft);
               border-left: 4px solid var(--warn-rule); border-radius: 0 8px 8px 0;
               color: var(--ink); }
  blockquote p { margin: 0; }
  blockquote strong { color: var(--warn); }

  img { display: block; max-width: 100%; height: auto; margin: 1.4rem 0;
        border: 1px solid var(--rule); border-radius: 10px; background: #fcfcfb; }

  details { margin: 1.2rem 0; border: 1px solid var(--rule); border-radius: 10px;
            padding: 0 1rem; }
  details[open] { padding-bottom: .6rem; }
  summary { cursor: pointer; padding: .8rem 0; font-weight: 600; color: var(--accent);
            list-style-position: outside; margin-left: 1rem; }
  details h3:first-of-type { margin-top: .6rem; }
</style>
</head>
<body>
<main>
{body}
</main>
<script>
  // Wrap tables so wide ones scroll inside the card instead of the page, and
  // mark the first one after the title as the at-a-glance spec sheet.
  document.querySelectorAll("main table").forEach(function (table) {
    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    var prev = table.previousElementSibling;
    if (prev && prev.tagName === "P" && prev.previousElementSibling
        && prev.previousElementSibling.tagName === "H1") {
      wrap.className += " glance";
    }
    table.parentNode.insertBefore(wrap, table);
    wrap.appendChild(table);
  });
</script>
</body>
</html>
"""


def build_release_bundle(
    training_dir: Path | str,
    output_dir: Path | str,
    *,
    version: str,
    checkpoint: Path | str | None = None,
    evaluation: Path | str | None = None,
    metrics_summary: Path | str | None = None,
    card: Path | str | None = None,
    input_shape: tuple[int, ...] | None = None,
    repo_id: str | None = None,
    info: Path | str | None = None,
    overwrite: bool = False,
) -> ReleaseBundle:
    """Build the release bundle from ``training_dir`` into ``output_dir``.

    The checkpoint is loaded and re-exported as ``model.safetensors`` so the
    bundle never depends on the internal ``.pisces`` container. Everything else
    is derived from the phase's own config snapshots, the run provenance
    primitives, and any evaluation or summary artifact supplied.

    ``info`` is a release-info TOML (:class:`ReleaseInfo`) describing the
    variant: its repo, audience, protocol, cohort descriptions and authored
    card. ``repo_id`` and ``card`` override what it names. ``repo_id`` titles
    the card, so a variant in its own repository is labelled as that variant.

    When ``metrics_summary`` is omitted, the phase's own
    ``cv_results_<n>_stages.csv`` and any sibling ``fold<i>`` directories are
    pooled into a summary automatically, so a downloaded phase directory is
    enough to build a scored card with a comparison figure.

    ``card`` is an authored Markdown file: intended use, limitations, how to
    apply, comparisons, figures. Its text is placed after the title and the
    generated model/input-pipeline/labels/evaluation/provenance sections follow
    it. Relative images it references are copied into :data:`ASSETS_DIR` and rewritten.
    Without ``card`` the card is generated with a :data:`CARD_REVIEW_PENDING` review marker and
    :attr:`ReleaseBundle.approved` is false.
    """
    training_dir = Path(training_dir)
    output_dir = Path(output_dir)
    release_info = load_release_info(info) if info is not None else ReleaseInfo()
    repo_id = repo_id or release_info.repo or ""
    if card is None:
        card = release_info.card
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already holds files; pass overwrite=True to replace them"
            )
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint(training_dir, checkpoint)
    phase_configs = _load_phase_configs(training_dir)
    model_config = phase_configs["model.json"]
    preprocessing = phase_configs["inputs.json"]

    from bidoze.model import BiDozeNetModel

    model = BiDozeNetModel.loadz(checkpoint_path, input_shape=input_shape)
    resolved_shape = tuple(model.input_shape or ())
    _names, arrays = model._flatten_params(model.params)
    parameter_count = sum(int(array.size) for array in arrays)
    model.save_weights_safetensors(output_dir / "model.safetensors")

    _write_json(output_dir / "config.json", model_config)
    _write_json(output_dir / "preprocessing.json", preprocessing)
    # metrics_effective.json is the resolved protocol the run actually scored
    # with; metrics.json is the request. Prefer the resolved one when present.
    config_dir = _config_dir(training_dir)
    metrics_source = config_dir / "metrics_effective.json"
    if not metrics_source.is_file():
        metrics_source = config_dir / "metrics.json"
    shutil.copyfile(metrics_source, output_dir / "metrics.json")

    phase_eval = _evaluation_for(training_dir, evaluation)
    summary = _summary_metrics(metrics_summary)
    # The table the card's comparison grid is drawn from: the supplied summary
    # when given, otherwise the phase's own results CSV (and any sibling fold<i>
    # directories), so a downloaded phase directory alone yields a scored card.
    summary_table = _grid_table(metrics_summary) if metrics_summary else None
    if summary is None:
        discovered, _csv_paths = _discovered_metrics(training_dir)
        if discovered is not None:
            summary = _summary_by_cohort(discovered)
            summary_table = discovered
    identity = (phase_eval or {}).get("identity", {})

    provenance = run_provenance(
        package_version=version,
        raw_manifest_sha256=None,
        split_manifest_sha256=_split_manifest_sha256(training_dir),
        derived_manifest_sha256=None,
        processing_config_sha256=_config_hashes(training_dir).get("inputs.json"),
        training_config_sha256=_config_hashes(training_dir).get("training.json"),
        model_config_sha256=_config_hashes(training_dir).get("model.json"),
        seed=int((_read_json(config_dir / "training.json")).get("run_seed", 0)),
        git_commit=code_digest(),
        image_digest=None,
    )
    provenance.update(
        {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "input_shape": list(resolved_shape),
            "run_id": identity.get("run_id"),
            "fold": identity.get("fold"),
            "epoch": identity.get("epoch"),
            "trained_on": identity.get("training_dataset"),
            "training_dir": str(training_dir),
            "metrics_summary": str(metrics_summary) if metrics_summary else None,
        }
    )
    _write_json(output_dir / "provenance.json", provenance)

    authored_text: str | None = None
    asset_list: list[str] = []
    if card is not None:
        card_path = Path(card)
        if not card_path.is_file():
            raise FileNotFoundError(f"no such card file: {card_path}")
        authored_text, authored_assets = _collect_assets(
            card_path.read_text(), card_path.parent, output_dir
        )
        asset_list.extend(authored_assets)

    # The comparison grid is generated from the summary, whether it was supplied
    # as a metrics_summary.csv or discovered from the phase's own results CSV.
    grid_asset: str | None = None
    if summary_table is not None:
        grid_asset = _draw_metrics_grid(summary_table, output_dir)
        asset_list.append(grid_asset)
    assets: tuple[str, ...] = tuple(dict.fromkeys(asset_list))

    card_text = _model_card(
        repo_id=repo_id,
        version=version,
        config=model_config,
        preprocessing=preprocessing,
        metrics=_read_json(output_dir / "metrics.json"),
        provenance=provenance,
        evaluation=phase_eval,
        summary=summary,
        authored_card=authored_text,
        info=release_info,
        grid_asset=grid_asset,
        parameter_count=parameter_count,
        assets=assets,
    )
    (output_dir / "README.md").write_text(card_text)

    # Assets are checksummed alongside the fixed bundle files so a tampered
    # figure is caught by verify_bundle and travels in the same upload commit.
    checksummed = [name for name in BUNDLE_FILES if name != "checksums.sha256"] + list(assets)
    (output_dir / "checksums.sha256").write_text(_checksums(output_dir, checksummed))

    return ReleaseBundle(
        directory=output_dir,
        version=version,
        checkpoint_sha256=provenance["checkpoint_sha256"],
        input_shape=resolved_shape,
        files=BUNDLE_FILES + assets,
        card_review_status=CARD_REVIEW_PENDING if authored_text is None else "reviewed",
    )


def load_bundle_model(bundle_dir: Path | str):
    """Load a released bundle's ``config.json`` + ``model.safetensors``.

    The inference-side counterpart of :func:`build_release_bundle`: takes a
    directory or an already-downloaded snapshot and returns a ready
    :class:`~bidoze.model.BiDozeNetModel` without touching ``.pisces``.
    """
    from bidoze.model import BiDozeNetConfig, BiDozeNetModel

    bundle_dir = Path(bundle_dir)
    config_path = bundle_dir / "config.json"
    weights_path = bundle_dir / "model.safetensors"
    for path in (config_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"{bundle_dir} is not a release bundle; missing {path.name}")

    config = BiDozeNetConfig.from_dict(_read_json(config_path))
    model = BiDozeNetModel(config)
    model.load_weights_safetensors(weights_path)
    return model


def verify_bundle(bundle_dir: Path | str) -> None:
    """Check every file in ``checksums.sha256`` matches, and required files exist."""
    bundle_dir = Path(bundle_dir)
    checksums = bundle_dir / "checksums.sha256"
    if not checksums.is_file():
        raise ValueError(f"{bundle_dir} has no checksums.sha256")
    expected: dict[str, str] = {}
    for line in checksums.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        expected[name] = digest
    missing = [
        name
        for name in BUNDLE_FILES
        if name != "checksums.sha256" and not (bundle_dir / name).is_file()
    ]
    if missing:
        raise ValueError(f"{bundle_dir} is missing bundle files: {missing}")
    for name, digest in expected.items():
        path = bundle_dir / name
        if not path.is_file():
            raise ValueError(f"{bundle_dir} is missing a checksummed file: {name}")
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"{name}: checksum mismatch ({actual} != {digest})")


def publish_to_hub(
    bundle_dir: Path | str,
    repo_id: str,
    *,
    revision: str | None = None,
    private: bool = False,
    dry_run: bool = False,
    token: str | None = None,
    client=None,
) -> str:
    """Upload a built, verified bundle to a Hugging Face model repository.

    The whole bundle goes up in one ``upload_folder`` call, so an approved
    publication is one atomic Hub commit rather than a run of per-file commits.
    Every regular file in the bundle -- the fixed files plus any :data:`ASSETS_DIR`
    images the card references -- is included, restricted to what
    :func:`verify_bundle` accepted. ``revision`` optionally targets a branch
    (e.g. ``main``); tags are created separately by the release process. Without
    the optional dependency and without a ``client`` this raises, so importing
    the publish path never forces ``huggingface_hub`` on the base install.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"no such bundle directory: {bundle_dir}")
    verify_bundle(bundle_dir)

    if dry_run:
        return repo_id

    if client is None:
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "uploading to the Hub needs the optional dependency: "
                "pip install 'arcasleep-bidoze[hub]'",
                name="huggingface_hub",
            ) from exc
        client = HfApi(token=token)

    client.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    client.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        revision=revision or "main",
        commit_message=f"Release {bundle_dir.name}",
        allow_patterns=_bundle_relpaths(bundle_dir),
    )
    return repo_id
