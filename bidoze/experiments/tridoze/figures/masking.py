"""What the gap rule does: one SleepAccel-Clinical test night of draw 0 scored as cached and inside 24 h.

Under the gap rule, epochs whose gap row is ≥ 1 are absent: zeroed, hidden from attention and output
fixed to gap. "Short recording" is the night as cached (context epoch 0); "inside 24 h" is the same
night after 1,700 absent epochs, built by ``gap_checks.build_inputs`` ("shifted"). The night is the
test-adult night where independent ×3 loses the most sleep called sleep (points) when shifted.

Run with JAX_PLATFORMS=cpu.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gap_checks as gc  # noqa: E402
from known_style import INK, MUTED, STAGE_COLORS, save  # noqa: E402

DRAW = 0
EPOCH_HOURS = 30 / 3600
OUT_DIR = gc.RESULTS / "tridoze-2026-10-16/gap_checks"
INDEPENDENT = gc.BASELINE / f"independent/draw{DRAW}/model_outputs"
TRIDOZE = gc.RESULTS / f"tridoze-2026-10-16/tridoze_j/draw{DRAW}/model_outputs"
MODELS = {
    "independent_x3": ("Independent ×3, no gap rule",
                       [INDEPENDENT / f"model_draw{DRAW}_seed_{s}.pisces" for s in range(3)]),
    "tridoze_j": ("TriDoze-J, gap rule",
                  [TRIDOZE / f"model_draw{DRAW}_seed_0_member{m}.pisces" for m in range(3)]),
}
CONDITIONS = {"cached": "short recording", "shifted": "inside 24 h"}
# Raw PSG labels in the cache: 0 wake, 1 N1, 2 N2, 3 N3, 4 N4, 5 REM, negative unscored.
PSG_STAGE = {0: "wake", 1: "light", 2: "light", 3: "deep", 4: "deep", 5: "REM"}
SHORT = dict(color="#b8bec5", lw=2.6, ls="-")
INSIDE = dict(color=INK, lw=1.2, ls=(0, (3, 2)))
STRIP_ORDER = ["wake", "light", "deep", "REM", "unscored"]
STRIP_COLORS = [STAGE_COLORS["wake"], STAGE_COLORS["light"], STAGE_COLORS["deep"], STAGE_COLORS["REM"], "#ffffff"]


def pick_night() -> pd.DataFrame:
    rows = pd.read_csv(OUT_DIR / f"independent_x3_draw{DRAW}.recordings.csv")
    rows = rows[rows.group == "SleepAccel-Clinical"]
    pct = rows.pivot(index="recording", columns="condition", values="sleep_called_sleep_epochs")
    pct = pct.div(rows[rows.condition == "cached"].set_index("recording").sleep_epochs, axis=0) * 100
    pct["loss_points"] = pct.cached - pct.shifted
    return pct.sort_values("loss_points", ascending=False)


def main() -> None:
    ranking = pick_night()
    night = ranking.index[0]
    print(ranking.head(5).round(2).to_string())

    draws = json.loads((gc.BASELINE / "setup/draws.json").read_text())[f"draw{DRAW}"]
    assert night in draws["testClinical"]
    path = gc.BASELINE / f"setup/cache/draw{DRAW}" / gc.CACHE_NAME / f"testClinical_subject_{night}_features.npz"
    recordings = gc.build_inputs([("SleepAccel-Clinical", night, path)], set())
    n = int(np.load(path, allow_pickle=True)["y"][0].shape[0])
    start = {"cached": 0, "shifted": gc.SHIFT}
    assert gc.SHIFT + n <= gc.X_LENGTH

    cache: dict = {}
    probas = {k: gc.predict(paths, recordings, cache)[0] for k, (_, paths) in MODELS.items()}
    x_c, y_c = recordings[0]["conditions"]["cached"]
    y = y_c[:n]
    absent = x_c[:n, -1, 0] >= 1.0
    sleep = np.isin(y, gc.SLEEP_LABELS)

    pooled = pd.read_csv(OUT_DIR / "gap_checks.csv")
    pooled = pooled[(pooled.draw == DRAW) & (pooled.cohort == "SleepAccel-Clinical")
                    & pooled.model.isin(MODELS) & pooled.condition.isin(CONDITIONS)]
    print(pooled[["model", "condition", "recordings", "sleep_called_sleep_pct"]].to_string(index=False))

    hours = np.arange(n) * EPOCH_HOURS
    stage_index = np.array([STRIP_ORDER.index(PSG_STAGE.get(int(v), "unscored")) for v in y])
    fig = plt.figure(figsize=(9, 5.6))
    outer = fig.add_gridspec(2, 1, hspace=0.42)
    summary = {}
    for row, (key, (label, _)) in enumerate(MODELS.items()):
        inner = outer[row].subgridspec(2, 1, height_ratios=[1, 0.09], hspace=0.06)
        ax = fig.add_subplot(inner[0])
        strip = fig.add_subplot(inner[1], sharex=ax)
        pct = {}
        for cond, style in (("cached", SHORT), ("shifted", INSIDE)):
            p = probas[key][cond][start[cond]:start[cond] + n]
            ax.plot(hours, 1 - p[:, 0] - p[:, gc.GAP], **style, zorder=3 if cond == "shifted" else 2)
            pct[cond] = 100 * np.isin(p.argmax(-1)[sleep], (1, 2, 3)).mean()
        summary[key] = pct
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("p(sleep)")
        ax.set_title(f"{label}: PSG sleep called sleep {pct['cached']:.1f}% short recording, "
                     f"{pct['shifted']:.1f}% inside 24 h", fontsize=9.5, loc="left")
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis="x", length=0)
        strip.imshow(stage_index[None], aspect="auto", cmap=ListedColormap(STRIP_COLORS), vmin=-0.5,
                     vmax=len(STRIP_ORDER) - 0.5, interpolation="nearest",
                     extent=(-EPOCH_HOURS / 2, hours[-1] + EPOCH_HOURS / 2, 0, 1))
        strip.set_yticks([0.5], ["PSG"])
        strip.tick_params(axis="y", length=0)
        for side in ("left", "bottom"):
            strip.spines[side].set_visible(False)
        strip.set_xlim(-EPOCH_HOURS / 2, hours[-1] + EPOCH_HOURS / 2)
        if row == 1:
            strip.set_xlabel("hours since the start of the night")
        else:
            plt.setp(strip.get_xticklabels(), visible=False)
    handles = [Line2D([], [], **SHORT, label="short recording (night at context epoch 0)"),
               Line2D([], [], **INSIDE, label="inside 24 h (1,700 absent epochs before the night)")]
    stages = [Patch(color=STAGE_COLORS[s], label=s) for s in ("wake", "light", "deep", "REM")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.12, 1.0), ncol=2, frameon=False, fontsize=8.5)
    fig.legend(handles=stages, loc="upper right", bbox_to_anchor=(0.91, 0.035),
               ncol=4, frameon=False, fontsize=8, handlelength=1.2)
    fig.text(0.125, -0.035, f"SleepAccel-Clinical test night {night}, draw {DRAW}; {int(sleep.sum())} PSG sleep epochs, "
             f"{int(absent.sum())} absent epochs in the night.", fontsize=8, color=MUTED)
    fig.subplots_adjust(top=0.9)
    save(fig, "tridoze-masking.png")

    for key, pct in summary.items():
        rows = pd.read_csv(OUT_DIR / f"{key}_draw{DRAW}.recordings.csv")
        rows = rows[rows.recording == night].set_index("condition")
        check = {c: 100 * rows.loc[c, "sleep_called_sleep_epochs"] / rows.loc[c, "sleep_epochs"] for c in CONDITIONS}
        print(key, {c: round(v, 2) for c, v in pct.items()}, "recordings.csv:", {c: round(v, 2) for c, v in check.items()})
        p = probas[key]
        d = np.abs(p["cached"][:n] - p["shifted"][gc.SHIFT:gc.SHIFT + n]).max()
        print(key, "max |Δ softmax| over the night:", f"{d:.2e}")


if __name__ == "__main__":
    main()
