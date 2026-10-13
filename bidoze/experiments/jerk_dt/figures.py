"""Whole-night jerk spectrograms per kept night: release, divided by Δt, and the difference.

Panels, sharing the time axis: the release's jerk log-PSD; the ``dt`` jerk log-PSD (its own
colour scale, since it is offset by about ``2·log(fs)``); ``dt − median_dt``, the change from
timestamp irregularity alone, on a symmetric scale. Padding frames are blank.

Writes ``<spectra dir>/figures/<cohort>_<night>.png``.

Usage: python figures.py <spectra dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP_S = 2.0
FMIN, FMAX = 0.1, 6.0


def main(spectra: Path) -> None:
    (spectra / "figures").mkdir(exist_ok=True)
    for path in sorted((spectra / "frames").glob("*.npz")):
        f = np.load(path)
        pad = (f["release"] == -1).all(axis=1)
        mask = lambda a: np.where(pad[:, None], np.nan, a).T  # noqa: E731  [bins, frames]
        hours = f["release"].shape[0] * STEP_S / 3600
        extent = (0, hours, FMIN, FMAX)
        diff = mask(f["dt"] - f["median_dt"])
        lim = float(np.nanpercentile(np.abs(diff), 99.5)) or 1.0
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        for ax, data, title, kw in [
            (axes[0], mask(f["release"]), "Release jerk ‖Δa‖: log-PSD", {"cmap": "Blues"}),
            (axes[1], mask(f["dt"]), "Jerk ‖Δa / Δt‖: log-PSD", {"cmap": "Blues"}),
            (axes[2], diff, "‖Δa / Δt‖ − ‖Δa‖ / median Δt (timestamp irregularity alone)",
             {"cmap": "coolwarm", "vmin": -lim, "vmax": lim}),
        ]:
            im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, interpolation="nearest", **kw)
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_ylabel("Hz")
            fig.colorbar(im, ax=ax, label="log-PSD difference" if ax is axes[2] else "log-PSD")
        axes[2].set_xlabel("hours from recording start")
        fig.suptitle(path.stem.replace("_", "/", 1), x=0.01, ha="left")
        fig.savefig(spectra / "figures" / f"{path.stem}.png", dpi=110)
        plt.close(fig)
        print(spectra / "figures" / f"{path.stem}.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
