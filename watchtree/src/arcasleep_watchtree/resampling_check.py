"""Does bringing a cohort to 25 Hz add spectral lines in the 0.1-0.6 Hz breathing band?

Two checks, written to ``--out``:

* Synthetic: a smooth noise process (flat spectrum below 10 Hz) is sampled at the recorded,
  jittered timestamps of one night, taken through :func:`~arcasleep_watchtree.preprocess.to_25hz`,
  and compared with the same process sampled exactly on a 25 Hz grid. The ratio of the two
  spectra, and the largest narrow peak in each, are reported.
* Real nights: the median-averaged Welch spectrum, summed over axes, of the longest gap-free
  stretch of a few nights per cohort, resampled by timestamp and by sample count, and the line
  prominence (each bin over the running median of the bins within 0.15 Hz) of each.

    python -m arcasleep_watchtree.resampling_check --out results/features-2026-10-27/resampling
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt, welch

from arcasleep_watchtree import constants as C
from arcasleep_watchtree.cohorts import COHORTS, subject_ids
from arcasleep_watchtree.preprocess import to_25hz

BAND = (0.1, 0.6)
NPERSEG = 25 * 100


def prominence(freq: np.ndarray, power: np.ndarray) -> float:
    """Largest ratio, within the breathing band, of a bin to the running median around it."""
    step = freq[1] - freq[0]
    width = 2 * int(round(0.15 / step)) + 1
    ratio = power / median_filter(power, size=width, mode="nearest")
    band = (freq >= BAND[0]) & (freq <= BAND[1])
    return float(ratio[band].max())


def axis_summed_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq, power = welch(x, fs=C.FS, nperseg=NPERSEG, average="median", axis=0)
    return freq, power.sum(axis=1)


def synthetic_check(timestamps: np.ndarray, seed: int = 0) -> dict:
    """Smooth noise at real timestamps vs the same noise on an ideal 25 Hz grid."""
    rng = np.random.default_rng(seed)
    t = timestamps - timestamps[0]
    duration = float(min(t[-1], 3 * 3600.0))
    t = t[t <= duration]
    dense_fs = 400.0
    n_dense = int(duration * dense_fs) + 2
    sos = butter(4, 10.0, fs=dense_fs, output="sos")
    dense = sosfiltfilt(sos, rng.normal(size=(n_dense, 3)), axis=0)
    dense_t = np.arange(n_dense) / dense_fs

    def sample(times: np.ndarray) -> np.ndarray:
        return np.column_stack([np.interp(times, dense_t, dense[:, a]) for a in range(3)])

    n_epochs = int(duration // C.EPOCH_S)
    acc, _ = to_25hz(t, sample(t), 0.0, n_epochs)
    ideal = sample(np.arange(n_epochs * C.SAMPLES_PER_EPOCH) / C.FS)
    freq, p_out = axis_summed_spectrum(acc)
    _, p_ideal = axis_summed_spectrum(ideal)
    band = (freq >= BAND[0]) & (freq <= BAND[1])
    ratio = p_out[band] / p_ideal[band]
    return {
        "duration_h": duration / 3600,
        "band_power_ratio_mean": float(ratio.mean()),
        "band_power_ratio_min": float(ratio.min()),
        "band_power_ratio_max": float(ratio.max()),
        "prominence_resampled": prominence(freq, p_out),
        "prominence_ideal": prominence(freq, p_ideal),
        "freq": freq.tolist(),
        "resampled": p_out.tolist(),
        "ideal": p_ideal.tolist(),
    }


def real_check(cohort_name: str, n_nights: int) -> dict:
    """Timestamp-aware resampling against treating the samples as uniform, on real nights.

    For each night the longest stretch without a gap is resampled two ways: by the recorded
    timestamps (:func:`to_25hz`), and as if the samples were exactly evenly spaced at the nominal
    rate (a polyphase filter on the sample sequence alone). A line the first adds and the second
    does not shows in the band as a power ratio away from 1 and a higher prominence.
    """
    from math import gcd

    from pisces_lite.datasets import load_subject
    from scipy.signal import resample_poly

    from arcasleep.data.prepare import load_dataset
    from arcasleep.processing import clean_accelerometer_rows
    from arcasleep_watchtree.cohorts import DATA_ROOT
    from arcasleep_watchtree.preprocess import nominal_rate

    cohort = COHORTS[cohort_name]
    data_set = load_dataset(DATA_ROOT / cohort.directory)
    ids, _ = subject_ids(cohort)
    rows_out = []
    for subject in ids[:: max(1, len(ids) // n_nights)][:n_nights]:
        try:
            rows = clean_accelerometer_rows(load_subject(data_set, subject).accel_df.to_numpy())
        except Exception:  # noqa: BLE001 - a night that does not load is skipped
            continue
        t, xyz = rows[:, 0], rows[:, 1:4]
        fs_in = nominal_rate(t)
        if fs_in == C.FS:
            continue
        cuts = np.flatnonzero(np.diff(t) > C.GAP_S) + 1
        edges = np.concatenate([[0], cuts, [t.size]])
        a, b = max(zip(edges[:-1], edges[1:], strict=True), key=lambda e: e[1] - e[0])
        t, xyz = t[a:b], xyz[a:b]
        n_epochs = int((t[-1] - t[0]) // C.EPOCH_S)
        if n_epochs < 120:
            continue
        by_time, _ = to_25hz(t, xyz, float(t[0]), n_epochs)
        g = gcd(C.FS, fs_in)
        by_count = resample_poly(xyz, C.FS // g, fs_in // g, axis=0, padtype="line")
        freq, p_time = axis_summed_spectrum(by_time)
        _, p_count = axis_summed_spectrum(by_count)
        band = (freq >= BAND[0]) & (freq <= BAND[1])
        ratio = p_time[band] / p_count[band]
        rows_out.append(
            {
                "hours": round(n_epochs * C.EPOCH_S / 3600, 2),
                "band_power_ratio_min": round(float(ratio.min()), 3),
                "band_power_ratio_max": round(float(ratio.max()), 3),
                "prominence_by_timestamp": round(prominence(freq, p_time), 3),
                "prominence_by_count": round(prominence(freq, p_count), 3),
            }
        )
    return {"nights": rows_out}


def main() -> None:
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nights", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from arcasleep.data.prepare import load_dataset
    from arcasleep_watchtree.cohorts import DATA_ROOT

    cohort = COHORTS["sleepaccel"]
    data_set = load_dataset(DATA_ROOT / cohort.directory)
    ids, _ = subject_ids(cohort)
    from pisces_lite.datasets import load_subject

    rows = load_subject(data_set, ids[1]).accel_df.to_numpy()
    jitter = np.diff(rows[:, 0])
    synthetic = synthetic_check(rows[:, 0])
    result = {
        "apple_timestamp_step_ms": {
            "median": float(np.median(jitter) * 1e3),
            "p1": float(np.percentile(jitter, 1) * 1e3),
            "p99": float(np.percentile(jitter, 99) * 1e3),
        },
        "synthetic": {
            k: v for k, v in synthetic.items() if k not in ("freq", "resampled", "ideal")
        },
        "real": {
            name: real_check(name, args.nights)
            for name in ("sleepaccel", "weaver_fitbit", "wearanize", "aauwss")
        },
    }
    (args.out / "resampling_check.json").write_text(json.dumps(result, indent=1) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freq = np.array(synthetic["freq"])
    band = (freq > 0.03) & (freq < 1.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(freq[band], np.array(synthetic["ideal"])[band], label="ideal 25 Hz sampling")
    ax.semilogy(
        freq[band],
        np.array(synthetic["resampled"])[band],
        label="jittered 50 Hz timestamps, resampled",
        alpha=0.8,
    )
    ax.axvspan(*BAND, color="grey", alpha=0.15, label="breathing band")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power, axes summed (g$^2$/Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "resampling_synthetic.png", dpi=150)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
