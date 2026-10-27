"""Build the feature cache that training reads.

One dataset directory in, one NPZ per subject out, named
``<dataset>``, :data:`~arcasleep.data.cache.SUBJECT_SEPARATOR`, ``<id>``,
:data:`~arcasleep.data.cache.CACHE_SUFFIX` and kept under
``<cache_root>/<cache_prefix>/``.

This is the only step that needs senpy and a NUFFT backend. Training itself
reads the cache and nothing else, which is what keeps a training host free of
signal-processing dependencies.

A dataset directory is either a tree of ``cleaned_*`` feature directories that
:meth:`DataSetObject.find_data_sets` understands, or one carrying its own
:data:`ADAPTER_FILENAME`.
"""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from pisces_lite.datasets import (
    DataSetObject,
    align_trim,
    load_data_sets_from_adapter,
    load_subject,
    mask_data,
)
from pisces_lite.datasets.constants import PSG_COL, PSG_DT, TIMESTAMP_COL
from pisces_lite.model_io import get_features_cache_name
from pisces_lite.proc import ProcessingConfig

from arcasleep.data.bundle import CacheBundle
from arcasleep.data.cache import CACHE_SUFFIX
from arcasleep.data.pseudonyms import cache_ids, load_pseudonym_key
from arcasleep.ggir import anglez_bins, epoch_bins
from arcasleep.processing import (
    FeatureExtractionError,
    clean_accelerometer_rows,
    frames_per_label,
    median_sampling_rate,
)
from arcasleep.data.constants import MIN_SCORED_MINUTES, OUTSIDE_PSG_LABEL


ADAPTER_FILENAME = "pisces_lite_adapter.py"

__all__ = ["MIN_SCORED_MINUTES"]


@dataclass(frozen=True)
class PreparedSubject:
    """One recording, cleaned and aligned, before its NUFFT runs."""

    dataset: str
    subject_id: str
    accel_values: np.ndarray
    labels: np.ndarray
    #: ``(bin_start_s, anglez_5s)`` from :func:`arcasleep.ggir.anglez_bins`,
    #: taken while the raw night is still in memory.
    anglez_bins: tuple[np.ndarray, np.ndarray] | None = None


def scored_minutes(labels: np.ndarray) -> float:
    """Minutes of actually-staged PSG in a recording."""
    return float(np.count_nonzero(np.asarray(labels) >= 0)) * (PSG_DT / 60.0)


def _match_dataset(
    found: dict, name: str | None, source: Path | str
) -> DataSetObject:
    """Pick one dataset out of what a loader discovered.

    ``name`` selects either a discovered key (the directory basename, or an
    adapter's own key) or the ``name`` in a dataset's ``data_set.json``. With no
    ``name``, a single discovered dataset is unambiguous.
    """
    if name is not None and name in found:
        return found[name]
    if name is not None:
        for data_set in found.values():
            if getattr(data_set.config, "name", None) == name:
                return data_set
    if len(found) == 1:
        return next(iter(found.values()))
    if not found:
        raise RuntimeError(
            f"no dataset under {source}; expected cleaned_* feature "
            f"directories or {ADAPTER_FILENAME}"
        )
    raise RuntimeError(
        f"{source} exposed {sorted(found)}, not {name!r}; pass --name to pick one"
    )


def load_dataset(dataset_dir: Path | str, name: str | None = None) -> DataSetObject:
    """Load one dataset directory and name it as its own metadata does.

    The name comes from ``data_set.json``'s ``name`` field, or from the key an
    adapter exposes -- not from the directory's basename. A dataset directory
    named for the symlink that points at it (``.../submission``) still reports
    the cohort it holds (``SleepAccel-Clinical``). ``name`` only selects which
    dataset to load when a directory exposes more than one.
    """
    path = Path(dataset_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {path}")

    adapter = path / ADAPTER_FILENAME
    if adapter.exists():
        found = load_data_sets_from_adapter(adapter, root=path)
        data_set = _match_dataset(found, name, adapter)
    else:
        found = DataSetObject.find_data_sets(path)
        data_set = _match_dataset(found, name, path)

    config_name = getattr(getattr(data_set, "config", None), "name", None)
    if config_name:
        data_set.name = config_name
    return data_set


def align_uncropped(
    accel: pd.DataFrame, psg: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Place the PSG on the epoch grid of the whole accelerometer stream.

    The grid has :func:`~pisces_lite.datasets.align_trim`'s phase (the PSG's
    first timestamp modulo the epoch length) and covers every whole epoch of
    ``accel``, which is trimmed to it. Epochs inside the span ``align_trim``
    keeps carry its labels; every other epoch is
    :data:`~arcasleep.data.constants.OUTSIDE_PSG_LABEL`. The two streams must
    overlap, as ``align_trim`` requires.
    """
    accel = accel.sort_values(by=TIMESTAMP_COL)
    _, psg_inside = align_trim(accel, psg)
    if psg_inside.empty:
        raise ValueError("accelerometer and PSG do not overlap")
    phase = float(psg[TIMESTAMP_COL].min()) % float(PSG_DT)
    first = accel[TIMESTAMP_COL].iloc[0]
    last = accel[TIMESTAMP_COL].iloc[-1]
    start = phase + PSG_DT * np.ceil((first - phase) / PSG_DT)
    end = phase + PSG_DT * np.floor((last - phase) / PSG_DT)
    accel = accel[(accel[TIMESTAMP_COL] >= start) & (accel[TIMESTAMP_COL] <= end)]
    n_epochs = int(round((end - start) / PSG_DT)) + 1
    inside_from = int(round((psg_inside[TIMESTAMP_COL].iloc[0] - start) / PSG_DT))
    labels = np.full(n_epochs, OUTSIDE_PSG_LABEL, dtype=int)
    labels[inside_from : inside_from + len(psg_inside)] = psg_inside[PSG_COL].to_numpy()
    grid = pd.DataFrame(
        {TIMESTAMP_COL: start + PSG_DT * np.arange(n_epochs, dtype=float), PSG_COL: labels}
    )
    return accel, grid


def prepare_subject(
    data_set: DataSetObject, subject_id: str, *, uncropped: bool = False
) -> PreparedSubject:
    """Load, clean, and align one recording without running its NUFFT.

    Cleaning happens *before* aligning. ``align_trim`` anchors the PSG grid at
    the recording's start time while the NUFFT grid is anchored at the first
    accelerometer sample; dropping leading rows afterwards would shift X's
    origin while y still began at the old start, pairing every epoch with an
    earlier wall-clock label.

    With ``uncropped`` the accelerometer stream is kept whole instead of being
    trimmed to the PSG; see :func:`align_uncropped`. Epochs with too little
    accelerometer data are :data:`~arcasleep.data.constants.GAP_LABEL` wherever
    they fall.
    """
    subject = load_subject(data_set, subject_id)
    cleaned = pd.DataFrame(
        clean_accelerometer_rows(subject.accel_df.to_numpy()),
        columns=subject.accel_df.columns,
    )
    if uncropped:
        accel_aligned, psg_aligned = align_uncropped(cleaned, subject.psg_df)
    else:
        accel_aligned, psg_aligned = align_trim(cleaned, subject.psg_df)
    psg_masked = mask_data(accel_aligned, psg_aligned)
    return PreparedSubject(
        dataset=data_set.name,
        subject_id=subject_id,
        accel_values=np.asarray(accel_aligned.to_numpy(), dtype=np.float64),
        labels=psg_masked[PSG_COL].to_numpy(),
    )


def bundle_subject(
    prepared: PreparedSubject,
    features: np.ndarray,
    processing_config: ProcessingConfig,
) -> CacheBundle:
    """Pair features with labels on the shared wall-clock range.

    Windowing leaves X slightly longer than the PSG count, and with a sub-epoch
    hop each label owns ``frames_per_label`` frames, so X is trimmed to a whole
    number of labelled epochs.
    """
    ratio = max(frames_per_label(processing_config), 1)
    n_labels = min(features.shape[0] // ratio, prepared.labels.shape[0])
    if n_labels < 1:
        raise FeatureExtractionError(
            f"{prepared.dataset}/{prepared.subject_id}: no overlapping epochs "
            f"({features.shape[0]} frames, {prepared.labels.shape[0]} labels)"
        )
    n_frames = n_labels * ratio

    anglez_5s = None
    if prepared.anglez_bins is not None:
        anglez_5s = epoch_bins(*prepared.anglez_bins, n_labels)[None]

    return CacheBundle(
        X=features[:n_frames][None],
        y=prepared.labels[:n_labels][None],
        datasets=np.array([prepared.dataset]),
        subject_ids=np.array([prepared.subject_id]),
        # Stashing the axes means plotting can read them off the cache instead
        # of re-deriving them from a processing config that may not be at hand.
        time=np.arange(n_frames, dtype=np.float64)
        * float(processing_config.window_step_seconds),
        freq=np.asarray(processing_config.get_frequencies(), dtype=np.float64),
        anglez_5s=anglez_5s,
    )


def _uses_cpu_backend(processing_config: ProcessingConfig) -> bool:
    backend = str(getattr(processing_config, "nufft_backend", "cpu"))
    return backend.replace("-", "_").lower() in {"cpu", "finufft"}


def transform_subjects(
    prepared: Sequence[PreparedSubject],
    processing_config: ProcessingConfig,
) -> List[np.ndarray]:
    """Run the NUFFT over a batch, packing it when the backend supports that."""
    if not prepared:
        return []

    apply_many = getattr(processing_config, "apply_many", None)
    if len(prepared) > 1 and not _uses_cpu_backend(processing_config) and callable(apply_many):
        transformed = apply_many(
            [item.accel_values for item in prepared],
            data_set_names=[item.dataset for item in prepared],
            normalize=False,
            padding=False,
        )
        if len(transformed) != len(prepared):
            raise RuntimeError(
                f"apply_many returned {len(transformed)} outputs "
                f"for {len(prepared)} recordings"
            )
        return list(transformed)

    return [
        processing_config.apply(item.accel_values, normalize=False, padding=False)
        for item in prepared
    ]


_LOAD_STATE: dict = {}


def _worker_count(requested: int | None) -> int:
    if requested:
        return max(1, int(requested))
    return max(1, min(12, (os.cpu_count() or 3) // 3))


def _deprioritize() -> None:
    """Yield the CPU to the process feeding the GPU when cores are contended."""
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    cpus = os.environ.get("BIDOZE_WORKER_CPUS")
    if cpus:
        os.sched_setaffinity(0, {int(c) for c in cpus.split(",")})


def _init_load_worker(
    dataset_dir: str,
    name: str | None,
    min_scored_minutes: float,
    target_fs: float | None,
    uncropped: bool = False,
) -> None:
    _deprioritize()
    _LOAD_STATE["data_set"] = load_dataset(dataset_dir, name)
    _LOAD_STATE["min_scored_minutes"] = min_scored_minutes
    _LOAD_STATE["target_fs"] = target_fs
    _LOAD_STATE["uncropped"] = uncropped


def _load_one(subject_id: str, cache_id: str) -> tuple[str, object]:
    """Worker-side load: ``("ok", PreparedSubject)`` or a reason it was dropped.

    ``subject_id`` reads the source files; the subject is reported and cached
    as ``cache_id``, its pseudonym when the dataset has a key.
    """
    data_set = _LOAD_STATE["data_set"]
    try:
        item = prepare_subject(data_set, subject_id, uncropped=_LOAD_STATE["uncropped"])
    except Exception as exc:  # noqa: BLE001 - reported with the subject attached
        return "failed", f"{cache_id}: {type(exc).__name__}: {exc}"
    finally:
        # DataSetObject keeps every DataFrame it has read; a worker sees many
        # nights, so drop them once the subject is prepared.
        for frames in getattr(data_set, "_feature_cache", {}).values():
            frames.pop(subject_id, None)
    item = replace(item, subject_id=cache_id, anglez_bins=anglez_bins(item.accel_values))
    # Checked before the NUFFT, which is the expensive part of prep.
    minutes = scored_minutes(item.labels)
    if minutes < _LOAD_STATE["min_scored_minutes"]:
        return "excluded", f"{cache_id}: {minutes:.1f} min scored"
    # senpy refuses a batch holding any recording sampled below the target
    # rate: its bins above that recording's Nyquist would be aliasing.
    target_fs = _LOAD_STATE["target_fs"]
    if target_fs and item.accel_values.shape[0] > 1:
        fs = median_sampling_rate(item.accel_values[:, 0])
        if fs < target_fs:
            return "failed", f"{cache_id}: sampled at {fs:.2f} Hz, below target {target_fs:g} Hz"
    return "ok", item


def _save_one(
    item: PreparedSubject,
    features: np.ndarray,
    processing_config: ProcessingConfig,
    destination: Path,
) -> tuple[Path, str]:
    bundle = bundle_subject(item, features, processing_config)
    bundle.save_npz(destination)
    return destination, f"  {item.subject_id}: X{bundle.X.shape[1:]} y{bundle.y.shape[1:]}"


def process_dataset(
    dataset_dir: Path | str,
    processing_config: ProcessingConfig,
    cache_dir: Path | str,
    *,
    name: str | None = None,
    batch_size: int = 8,
    overwrite: bool = False,
    min_scored_minutes: float = MIN_SCORED_MINUTES,
    load_workers: int | None = None,
    write_workers: int | None = None,
    uncropped: bool = False,
) -> List[Path]:
    """Build the cache for one dataset. Returns the NPZ paths written.

    With ``uncropped`` each recording keeps its whole wrist stream; epochs
    outside the PSG are labelled
    :data:`~arcasleep.data.constants.OUTSIDE_PSG_LABEL`, and the cache belongs
    in a directory of its own.

    Subjects that fail are reported and skipped rather than aborting the run;
    a dataset usually has a few unusable recordings and re-running the whole
    cohort to get past one is expensive.

    Recordings with under ``min_scored_minutes`` of staged PSG are skipped the
    same way. They are dropped here rather than at training time so the cache
    is the cohort: a short or wholly unscored night cannot be trained on, and
    scoring one yields metrics that are undefined rather than merely poor.

    A dataset directory with a pseudonym key is cached under its subjects'
    pseudonyms (see :mod:`arcasleep.data.pseudonyms`), never their source ids.
    """
    data_set = load_dataset(dataset_dir, name)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = list(data_set.ids)
    key = load_pseudonym_key(dataset_dir)
    to_cache_id = cache_ids(key, subject_ids)
    print(
        f"{data_set.name}: {len(subject_ids)} subjects -> {cache_dir}"
        + (" (pseudonymised)" if key is not None else "")
    )

    pending: List[str] = []
    written: List[Path] = []
    for subject_id in subject_ids:
        cache_id = to_cache_id[subject_id]
        destination = cache_dir / get_features_cache_name(data_set.name, cache_id)
        if destination.exists() and not overwrite:
            print(f"  {cache_id}: cached")
            written.append(destination)
            continue
        pending.append(subject_id)

    failures: List[str] = []
    excluded: List[str] = []
    workers = _worker_count(load_workers)
    writers = _worker_count(write_workers)

    # Three stages, each kept busy by the next: worker processes parse and
    # clean CSVs, this process owns the GPU and runs the NUFFT a batch of
    # recordings at a time, and a second process pool bundles and compresses
    # the NPZs. Both pools are processes because this one's GIL is the scarce
    # resource: it feeds the GPU. Loads are submitted a bounded distance ahead
    # so a fast loader cannot pile whole nights up in memory.
    context = multiprocessing.get_context("spawn")
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_load_worker,
            initargs=(
                str(dataset_dir),
                name,
                min_scored_minutes,
                getattr(processing_config, "fs", None),
                uncropped,
            ),
        ) as loaders,
        ProcessPoolExecutor(
            max_workers=writers, mp_context=context, initializer=_deprioritize
        ) as savers,
    ):
        ids = iter(pending)
        loading: set = set()
        saving: set = set()

        def top_up() -> None:
            while len(loading) < max(2 * workers, batch_size):
                subject_id = next(ids, None)
                if subject_id is None:
                    return
                loading.add(loaders.submit(_load_one, subject_id, to_cache_id[subject_id]))

        def collect_saves(block: bool) -> None:
            done = [f for f in saving if f.done()] if not block else list(saving)
            for future in done:
                saving.discard(future)
                try:
                    destination, shape_note = future.result()
                except FeatureExtractionError as exc:
                    failures.append(str(exc))
                    continue
                written.append(destination)
                print(shape_note)

        top_up()
        ready: List[PreparedSubject] = []
        while loading or ready:
            if loading:
                done, _ = wait(loading, return_when=FIRST_COMPLETED)
                for future in done:
                    loading.discard(future)
                    status, payload = future.result()
                    if status == "ok":
                        ready.append(payload)
                    elif status == "excluded":
                        excluded.append(payload)
                    else:
                        failures.append(payload)
                top_up()
            if ready and (len(ready) >= batch_size or not loading):
                batch, ready = ready[:batch_size], ready[batch_size:]
                for item, features in zip(batch, transform_subjects(batch, processing_config)):
                    # The raw night is no longer needed; don't ship it again.
                    item = replace(item, accel_values=np.empty((0, 0)))
                    destination = cache_dir / get_features_cache_name(
                        data_set.name, item.subject_id
                    )
                    saving.add(
                        savers.submit(
                            _save_one, item, features, processing_config, destination
                        )
                    )
                collect_saves(block=False)
                # Back-pressure: never hold more than a few batches of
                # finished features waiting on compression.
                while len(saving) > 2 * max(batch_size, writers):
                    wait(saving, return_when=FIRST_COMPLETED)
                    collect_saves(block=False)
        wait(saving)
        collect_saves(block=True)

    if excluded:
        print(
            f"\n{len(excluded)} subject(s) below {min_scored_minutes:.0f} min scored PSG:"
        )
        for entry in excluded:
            print(f"  {entry}")

    if failures:
        print(f"\n{len(failures)} subject(s) skipped:")
        for failure in failures:
            print(f"  {failure}")
    print(f"{data_set.name}: {len(written)} cached NPZ files ({CACHE_SUFFIX})")
    return written


def backfill_sib(
    dataset_dir: Path | str,
    cache_dir: Path | str,
    *,
    name: str | None = None,
    overwrite: bool = False,
) -> List[Path]:
    """Add ``anglez_5s`` to a dataset's existing cache NPZs, without a NUFFT.

    Only the raw signal is re-read; ``X`` and ``y`` are carried over untouched.
    Each subject is re-prepared exactly as :func:`process_dataset` did, and the
    cached labels must match the freshly prepared ones: if they do not, the raw
    data no longer lines up with the cache and the subject is refused rather
    than given a misaligned series. Subjects with no NPZ, or one that already
    has the array (unless ``overwrite``), are skipped. Returns the NPZs written.
    """
    data_set = load_dataset(dataset_dir, name)
    cache_dir = Path(cache_dir)
    to_cache_id = cache_ids(load_pseudonym_key(dataset_dir), data_set.ids)
    written: List[Path] = []
    failures: List[str] = []
    for subject_id in data_set.ids:
        cache_id = to_cache_id[subject_id]
        path = cache_dir / get_features_cache_name(data_set.name, cache_id)
        if not path.exists():
            continue
        bundle = CacheBundle.from_npz(path)
        if bundle.anglez_5s is not None and not overwrite:
            print(f"  {cache_id}: already has anglez_5s")
            continue
        try:
            item = prepare_subject(data_set, subject_id)
            n_labels = bundle.y.shape[1]
            if not np.array_equal(item.labels[:n_labels], bundle.y[0]):
                raise ValueError("cached labels differ from the source data's")
            bundle.anglez_5s = epoch_bins(*anglez_bins(item.accel_values), n_labels)[None]
        except Exception as exc:  # noqa: BLE001 - reported with the subject attached
            failures.append(f"{cache_id}: {type(exc).__name__}: {exc}")
            continue
        finally:
            for frames in getattr(data_set, "_feature_cache", {}).values():
                frames.pop(subject_id, None)
        staging = path.with_name(path.stem + ".tmp.npz")
        bundle.save_npz(staging)
        staging.replace(path)
        written.append(path)
        print(f"  {cache_id}: anglez_5s {bundle.anglez_5s.shape[1:]}")
    if failures:
        print(f"{len(failures)} subject(s) not updated:")
        for line in failures:
            print(f"  {line}")
    return written
