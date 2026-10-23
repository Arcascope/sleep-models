"""Convert the EESM19 cohort (OpenNeuro ds005185) into the arcasleep input layout.

    python convert.py RAW_ROOT OUT_DIR [--workers N]

RAW_ROOT is the downloaded dataset folder (BIDS, with ``sourcedata/``). It is
only read. OUT_DIR gets the layout `pisces_lite` loads:

    cleaned_accelerometer/<id>_cleaned_motion.out   time_s x_g y_g z_g
    cleaned_psg/<id>_cleaned_psg.out                time_s stage
    scorer2/<id>_scorer2.out                        time_s stage (side file)
    data_set.json, conversion_manifest.json

Time zero of every file is the start of the first scored epoch.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

EPOCH_S = 30.0
GENEACTIV_HZ = 100.0
# BIDS scoring codes: 1 Wake, 2 REM, 3 N1, 4 N2, 5 N3, 6 A, 7 Artefact, 8 Unscored (0: no label).
# SleepAccel codes: -1 unscored, 0 wake, 1 N1, 2 N2, 3 N3, 5 REM.
STAGE_MAP = {1: 0, 3: 1, 4: 2, 5: 3, 2: 5}
MATLAB_DATENUM_UNIX_EPOCH = 719529.0

# Shake alignment.
BIN_S = 0.25
PSG_WINDOW_S = 1800.0
SEARCH_S = 7200.0
MIN_PEAK_R = 0.3
MIN_PEAK_Z = 8.0
# A weaker peak (z >= MIN_PEAK_Z_NEAR_TYPICAL) is also accepted when it lies within NEAR_TYPICAL_S of the
# clock difference most nights show (the EDF starts about 5 s after the watch clock says it does).
TYPICAL_CLOCK_DIFF_S = 5.0
NEAR_TYPICAL_S = 30.0
MIN_PEAK_Z_NEAR_TYPICAL = 4.0
EXCLUDE_S = 60.0
MIN_COVERED_FRACTION = 0.5


def night_id(sub: str, ses: str) -> str:
    """``sub-001``, ``ses-002`` -> ``sub001_s2``."""
    return f"sub{int(sub.split('-')[1]):03d}_s{int(ses.split('-')[1])}"


def read_edf_window(path: Path, seconds: float) -> tuple[dict, dict]:
    """Header fields and the signal channels of the first ``seconds`` of an EDF file."""
    with open(path, "rb") as f:
        head = f.read(256)
        n_sig = int(head[252:256])
        n_rec = int(head[236:244])
        rec_s = float(head[244:252])
        fields = f.read(n_sig * 256)

        # Header fields are stored one after another, each as n_sig consecutive values of the given width.
        widths = {"label": 16, "transducer": 80, "unit": 8, "pmin": 8, "pmax": 8, "dmin": 8, "dmax": 8, "prefilter": 80, "samples": 8}
        starts = dict(zip(widths, np.cumsum([0] + [w * n_sig for w in widths.values()])))

        def column(name: str) -> list[str]:
            w, s = widths[name], int(starts[name])
            return [fields[s + i * w : s + (i + 1) * w].decode("latin1").strip() for i in range(n_sig)]

        labels = column("label")
        pmin, pmax, dmin, dmax = column("pmin"), column("pmax"), column("dmin"), column("dmax")
        samples = [int(x) for x in column("samples")]
        n_read = min(n_rec, int(seconds / rec_s))
        total = sum(samples)
        data = np.fromfile(f, dtype="<i2", count=total * n_read).reshape(n_read, total)
    edges = np.cumsum([0] + samples)
    channels = {}
    for i, label in enumerate(labels):
        if label.startswith("Status") or label.startswith("EDF"):
            continue
        gain = (float(pmax[i]) - float(pmin[i])) / (float(dmax[i]) - float(dmin[i]))
        raw = data[:, edges[i] : edges[i + 1]].reshape(-1).astype(np.float32)
        channels[label] = ((raw - float(dmin[i])) * gain + float(pmin[i]), samples[i] / rec_s)
    start = head[168:176].decode().split(".") + head[176:184].decode().split(".")
    day, month, yy, hh, mm, ss = (int(x) for x in start)
    header = {
        "start": pd.Timestamp(year=2000 + yy, month=month, day=day, hour=hh, minute=mm, second=ss),
        "n_records": n_rec,
    }
    return header, channels


def psg_activity(channels: dict) -> np.ndarray:
    """Mean over channels of the robust z-scored log power above 1 Hz, in bins of ``BIN_S`` seconds."""
    series = []
    for signal, rate in channels.values():
        high = sosfiltfilt(butter(3, 1.0, "high", fs=rate, output="sos"), signal)
        size = int(round(BIN_S * rate))
        n = high.size // size
        log_power = np.log(np.mean(high[: n * size].reshape(n, size) ** 2, axis=1) + 1e-3)
        mad = np.median(np.abs(log_power - np.median(log_power))) + 1e-6
        series.append((log_power - np.median(log_power)) / mad)
    n = min(s.size for s in series)
    return np.mean([s[:n] for s in series], axis=0)


def watch_activity(magnitude: np.ndarray) -> np.ndarray:
    """Log of the summed absolute change of the acceleration magnitude, in bins of ``BIN_S`` seconds."""
    change = np.abs(np.diff(magnitude, prepend=magnitude[0]))
    size = int(round(BIN_S * GENEACTIV_HZ))
    n = change.size // size
    return np.log(change[: n * size].reshape(n, size).sum(axis=1) + 0.05)


def sliding_pearson(template: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Pearson correlation of ``template`` with every same-length window of ``series``."""
    from scipy.signal import fftconvolve

    n = template.size
    centred = template - template.mean()
    norm = np.sqrt(np.sum(centred**2))
    numerator = fftconvolve(series, centred[::-1], mode="valid")
    cs, css = np.cumsum(np.insert(series, 0, 0.0)), np.cumsum(np.insert(series**2, 0, 0.0))
    window_sum = cs[n:] - cs[:-n]
    window_var = (css[n:] - css[:-n]) - window_sum**2 / n
    return numerator / (norm * np.sqrt(np.maximum(window_var, 1e-12)))


def align(psg_feature: np.ndarray, watch_feature: np.ndarray, predicted_s: float) -> dict:
    """Watch time of EDF time zero, from the shake at the start of the night.

    Returns the offset in watch seconds (the watch is ``offset_s`` into its recording when the EDF starts),
    the peak correlation, the peak's z-score against correlations further than ``EXCLUDE_S`` from it,
    and the best correlation outside that margin.
    """
    centre = int(round(predicted_s / BIN_S))
    reach = int(SEARCH_S / BIN_S)
    lo, hi = max(0, centre - reach), min(watch_feature.size - psg_feature.size, centre + reach)
    if hi <= lo:
        return {"offset_s": None}
    r = sliding_pearson(psg_feature, watch_feature[lo : hi + psg_feature.size])[: hi - lo + 1]
    peak = int(np.argmax(r))
    margin = int(EXCLUDE_S / BIN_S)
    outside = np.concatenate([r[: max(peak - margin, 0)], r[peak + margin + 1 :]])
    spread = outside.std() if outside.size > 1 else np.nan
    return {
        "offset_s": (lo + peak) * BIN_S,
        "peak_r": float(r[peak]),
        "peak_z": float((r[peak] - outside.mean()) / spread) if outside.size > 1 else None,
        "second_peak_r": float(outside.max()) if outside.size else None,
    }


def read_geneactiv(path: Path) -> tuple[pd.Timestamp, np.ndarray, float]:
    """Start time, ``(n, 3)`` acceleration in g and the largest timestamp deviation from a 100 Hz clock (s)."""
    with open(path, errors="replace") as f:
        for skip, line in enumerate(f):
            if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d:\d{3},", line):
                break
    table = pd.read_csv(path, skiprows=skip, header=None, usecols=[0, 1, 2, 3], dtype={0: str, 1: np.float32, 2: np.float32, 3: np.float32})
    stamps = pd.to_datetime(table[0].iloc[::100], format="%Y-%m-%d %H:%M:%S:%f")
    start = stamps.iloc[0]
    elapsed = (stamps - start).dt.total_seconds().to_numpy()
    deviation = float(np.max(np.abs(elapsed - np.arange(elapsed.size))))
    return start, table[[1, 2, 3]].to_numpy(), deviation


def read_scores(path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path, sep="\t")
    stages = np.array([STAGE_MAP.get(int(v), -1) for v in table[column]], dtype=float)
    return table["onset"].to_numpy(dtype=float), stages


def write_table(path: Path, columns: list[np.ndarray], formats: list[str]) -> None:
    np.savetxt(path, np.column_stack(columns), fmt=formats, delimiter=" ")


def convert_one(args: tuple) -> dict:
    raw_root, out, sub, ses = Path(args[0]), Path(args[1]), args[2], args[3]
    sid = night_id(sub, ses)
    info: dict = {"id": sid, "raw": f"{sub}/{ses}"}

    def drop(reason: str) -> dict:
        info["status"] = "dropped"
        info["reason"] = reason
        return info

    source = raw_root / "sourcedata" / sub / ses
    bids = raw_root / sub / ses / "eeg"
    scoring1 = bids / f"{sub}_{ses}_task-sleep_acq-scoring1_events.tsv"
    scoring2 = bids / f"{sub}_{ses}_task-sleep_acq-scoring2_events.tsv"
    edf = source / "scoring" / "edf_used_to_score1.edf"
    csvs = sorted((source / "actigraphy").glob("*.csv"))
    if not scoring1.exists():
        return drop("no scorer 1 file")
    if not csvs:
        return drop("no actigraphy csv")
    if not edf.exists():
        return drop("no EDF to align against")

    onsets, stages = read_scores(scoring1, "Scoring1")
    if onsets.size == 0:
        return drop("no scored epochs")
    header, channels = read_edf_window(edf, PSG_WINDOW_S)
    watch_start, acc, deviation = read_geneactiv(csvs[0])
    info["timestamp_deviation_s"] = deviation
    info["psg_start"] = str(header["start"])
    info["watch_start"] = str(watch_start)
    info["watch_duration_s"] = acc.shape[0] / GENEACTIV_HZ

    magnitude = np.linalg.norm(acc.astype(np.float64), axis=1)
    predicted = (header["start"] - watch_start).total_seconds()
    found = align(psg_activity(channels), watch_activity(magnitude), predicted)
    info["alignment"] = found
    if found["offset_s"] is None:
        return drop("watch recording does not cover the start of the PSG")
    info["clock_difference_s"] = found["offset_s"] - predicted
    z = found["peak_z"] or 0.0
    near_typical = abs(info["clock_difference_s"] - TYPICAL_CLOCK_DIFF_S) <= NEAR_TYPICAL_S
    if found["peak_r"] < MIN_PEAK_R or z < (MIN_PEAK_Z_NEAR_TYPICAL if near_typical else MIN_PEAK_Z):
        return drop("shake alignment not confident")
    info["alignment_rule"] = "near typical clock difference" if near_typical and z < MIN_PEAK_Z else "strong peak"

    # Epoch k starts at EDF time onsets[k]; output time zero is the first epoch.
    zero = onsets[0]
    times = onsets - zero
    span = times[-1] + EPOCH_S
    watch_time = np.arange(acc.shape[0]) / GENEACTIV_HZ - found["offset_s"] - zero
    inside = (watch_time >= 0) & (watch_time < span)
    if not inside.any():
        return drop("no actigraphy inside the scored window")
    t, acc = watch_time[inside], acc[inside].astype(np.float64)

    counts = np.histogram(t, bins=np.append(times, span))[0]
    covered = counts > 0
    info["epochs"] = int(times.size)
    info["artifact_epochs"] = int(np.sum(stages == -1))
    info["epochs_without_watch"] = int(np.sum(~covered))
    if covered.mean() < MIN_COVERED_FRACTION:
        return drop("actigraphy covers under half of the scored epochs")
    stages = np.where(covered, stages, -1.0)

    fmt_acc = ["%.3f", "%.4f", "%.4f", "%.4f"]
    write_table(out / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out", [t, acc[:, 0], acc[:, 1], acc[:, 2]], fmt_acc)
    write_table(out / "cleaned_psg" / f"{sid}_cleaned_psg.out", [times, stages], ["%.3f", "%d"])
    if scoring2.exists():
        onsets2, stages2 = read_scores(scoring2, "Scoring2")
        write_table(out / "scorer2" / f"{sid}_scorer2.out", [onsets2 - zero, stages2], ["%.3f", "%d"])
        info["scorer2"] = True
    info["status"] = "converted"
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw_root")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    raw_root, out = Path(a.raw_root).resolve(), Path(a.out).resolve()
    if raw_root in out.parents or out == raw_root:
        raise SystemExit("output must be outside the raw tree")
    for sub in ("cleaned_accelerometer", "cleaned_psg", "scorer2"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    nights = [(s.parent.name, s.name) for s in sorted((raw_root / "sourcedata").glob("sub-*/ses-*"))]
    jobs = [(str(raw_root), str(out), sub, ses) for sub, ses in nights]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(convert_one, jobs))

    (out / "data_set.json").write_text(json.dumps({
        "name": "eesm19",
        "csv": {"delimiter": " "},
        "id_pattern_by_feature": {
            "accelerometer": "<<ID>>_cleaned_motion.out",
            "psg": "<<ID>>_cleaned_psg.out",
        },
        "metadata": {
            "note": "EESM19 (OpenNeuro ds005185): GENEActiv wrist accelerometer (100 Hz, g) with manual scores of "
                    "partial PSG from scorer 1. Time zero is the first scored epoch; the wrist data is aligned to the "
                    "PSG with the shake at the start of each night. PSG stages use the 0-5 convention (0=Wake, 5=REM); "
                    "-1 marks artifact or unscored epochs and epochs without wrist data. scorer2/ holds the second scoring.",
        },
    }, indent=2) + "\n")
    (out / "conversion_manifest.json").write_text(json.dumps(results, indent=1) + "\n")
    done = [r for r in results if r["status"] == "converted"]
    print(f"converted {len(done)} of {len(results)} nights")
    reasons: dict[str, int] = {}
    for r in results:
        if r["status"] == "dropped":
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print(reasons)


if __name__ == "__main__":
    main()
