"""Inventory Weaver and build a 60-s / 15-s accelerometer-only HR experiment cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, welch

from nightbeat_adapter import predict, rolling_median

DEVICES = ("actigraph", "apple", "fitbit", "garmin", "axivity", "pixel")
FEATURES = ["anchor", "anchor_available", "spectral_hr", "snr", "second_harmonic",
            "subharmonic", "half_anchor", *["psd_" + str(i) for i in range(18)], "delta"]
FS = 100
WINDOW = 60
STEP = 15


def normalized(name):
    return re.sub(r"[\s#]", "", name).lower()


def header(path):
    with path.open(newline="") as f:
        names = next(csv.reader(f))
    result = {normalized(name): name for name in names}
    if "psgtime" not in result and all(c in result for c in ("time", "epoch", "bp", "stg")):
        for c in ("time", "epoch", "bp", "stg"):
            result["psg" + c] = result[c]
    return result


def inventory(root, output):
    rows = []
    for path in sorted((root / "raw").glob("*.csv")):
        h = header(path)
        child, night = path.stem.rsplit("_n", 1)
        for device in DEVICES:
            if all(device + c in h for c in ("time", "x", "y", "z")):
                rows.append(dict(participant=child, night=int(night), device=device,
                    filename=path.name, bytes=path.stat().st_size,
                    kubios_reference="kubiosnonemeanhr" in h,
                    psg="psgtime" in h and "psgstg" in h,
                    optical_reference=";".join(d for d in DEVICES if d + "heartrate" in h)))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "inventory.csv", index=False)
    return frame


def load_columns(path, device):
    h = header(path)
    keys = [device + c for c in ("time", "x", "y", "z")]
    for key in h:
        if key.startswith("kubios") or key in ("psgtime", "psgstg"):
            keys.append(key)
    # Optical reference is used only where Kubios is absent; never an input feature.
    if "kubiosnonemeanhr" not in h:
        for d in DEVICES:
            if d + "heartrate" in h:
                keys += [d + "time", d + "heartrate"]
    wanted = list(dict.fromkeys(h[k] for k in keys))
    d = pd.read_csv(path, usecols=wanted, low_memory=False)
    d = d.rename(columns={v: k for k, v in h.items()})
    return d, h


def resample(d, device):
    cols = [device + c for c in ("time", "x", "y", "z")]
    s = d[cols].apply(pd.to_numeric, errors="coerce").dropna()
    s = s.sort_values(cols[0], kind="stable")
    duplicates = int(s[cols[0]].duplicated().sum())
    if duplicates:
        s = s.groupby(cols[0], sort=True, as_index=False).mean()
    t = s.iloc[:, 0].to_numpy(float)
    axes = s.iloc[:, 1:].to_numpy(float)
    if len(t) < FS * WINDOW or not np.all(np.diff(t) > 0):
        raise ValueError("Insufficient increasing timestamps")
    dt = np.diff(t)
    origin = np.ceil(t[0] * FS) / FS
    grid = origin + np.arange(int(np.floor((t[-1] - origin) * FS)) + 1) / FS
    xyz = np.column_stack([np.interp(grid, t, axes[:, k]) for k in range(3)]).astype(np.float32)
    right = np.searchsorted(t, grid, side="left").clip(1, len(t) - 1)
    gap = t[right] - t[right - 1]
    valid = (gap <= 0.25) & (grid >= t[0]) & (grid <= t[-1])
    # Reject any window crossing a >250 ms missing-data interval.
    return xyz, valid, origin, dict(native_hz=float(1 / np.median(dt)),
        samples=len(t), duplicate_timestamps=duplicates,
        gap_count=int((dt > 0.25).sum()), duration_seconds=float(t[-1] - t[0]))


def reference(d, h):
    if "kubiosnonemeanhr" in h:
        qcol = next((k for k in d if k.startswith("kubiossignalquality")), None)
        cols = ["kubiostime", "kubiosnonemeanhr"] + ([qcol] if qcol else [])
        r = d[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=cols[:2])
        r = r.sort_values(cols[0]).drop_duplicates(cols[0])
        t, y = r.iloc[:, 0].to_numpy(), r.iloc[:, 1].to_numpy()
        q = r[qcol].to_numpy() / 100 if qcol else np.ones(len(r))
        return t, y, np.nan_to_num(q, nan=0).clip(0, 1), "kubios_none_mean_hr"
    for device in ("apple", "pixel", "garmin", "fitbit"):
        if device + "heartrate" in h:
            cols = [device + "time", device + "heartrate"]
            r = d[cols].apply(pd.to_numeric, errors="coerce").dropna().sort_values(cols[0])
            r = r.drop_duplicates(cols[0])
            return r.iloc[:, 0].to_numpy(), r.iloc[:, 1].to_numpy(), np.zeros(len(r)), device + "_optical_hr"
    return np.array([]), np.array([]), np.array([]), "none"


def extract(xyz, valid, origin, rt, ry, rq, d):
    starts = np.arange(np.ceil(origin / STEP) * STEP,
                       origin + len(xyz) / FS - WINDOW + 1e-7, STEP)
    records, spectra = [], []
    frequency = np.fft.rfftfreq(FS * WINDOW - 1, 1 / FS)
    hann = np.hanning(FS * WINDOW - 1)
    sos = butter(4, [0.5, 4], fs=FS, btype="bandpass", output="sos")
    if "psgtime" in d and "psgstg" in d:
        p = d[["psgtime", "psgstg"]].dropna().drop_duplicates("psgtime").sort_values("psgtime")
        pt, ps = p.psgtime.to_numpy(float), p.psgstg.astype(str).to_numpy()
    else:
        pt, ps = np.array([]), np.array([])
    for start in starts:
        lo = int(round((start - origin) * FS))
        hi = lo + FS * WINDOW
        if lo < 0 or hi > len(xyz) or not valid[lo:hi].all():
            continue
        seg = xyz[lo:hi].astype(float)
        jerk = np.linalg.norm(np.diff(seg, axis=0), axis=1) * FS
        power = np.abs(np.fft.rfft((jerk - jerk.mean()) * hann)) ** 2
        row = dict(endpoint=float(start + WINDOW), motion=float(np.linalg.norm(seg, axis=1).std()))
        for name, low, high in [("peak_narrow", 0.9, 1.3), ("peak_wide", 0.5, 2.0)]:
            band = (frequency >= low) & (frequency <= high)
            bp = power[band]
            k = int(np.argmax(bp))
            row[name] = float(frequency[band][k] * 60)
            row[name + "_clarity"] = float(bp[k] / (bp.mean() + 1e-20))
        a, b = np.searchsorted(rt, [start, start + WINDOW])
        ref = ry[a:b]
        row["reference"] = float(np.mean(ref)) if len(ref) else np.nan
        row["reference_samples"] = len(ref)
        row["reference_quality"] = float(np.mean(rq[a:b])) if len(ref) else 0
        # PSG is descriptive/scoring only and never a model feature.
        idx = np.searchsorted(pt, start + WINDOW / 2, side="right") - 1
        row["psg_stage"] = ps[idx] if idx >= 0 and start + WINDOW / 2 < pt[idx] + 30 else ""
        mag = sosfilt(sos, np.linalg.norm(seg, axis=1))
        f, p = welch(mag, fs=FS, window="hann", nperseg=512, noverlap=256, nfft=512)
        mask = (f >= 0.5) & (f <= 4)
        spec = p[mask]
        assert len(spec) == 18
        row["spectral_hr"] = float(f[mask][np.argmax(spec)] * 60)
        row["spectral_snr"] = float(spec.max() / (np.mean(spec) + 1e-20))
        records.append(row)
        spectra.append(spec)
    return pd.DataFrame(records), np.asarray(spectra), f[mask]


def features(frame, psd, frequency):
    anchor = frame.nb_anchor.to_numpy(float) if "nb_anchor" in frame else np.full(len(frame), np.nan)
    available = np.isfinite(anchor)
    anchor = np.where(available, anchor, frame.spectral_hr)
    at = lambda hz: np.array([np.interp(x, frequency, p, left=0, right=0) for x, p in zip(hz, psd)])
    base = at(anchor / 60)
    second = at(anchor / 30) / (base + 1e-10)
    half = at(anchor / 120) / (base + 1e-10)
    delta = np.zeros(len(frame))
    if len(frame) > 1:
        delta[1:] = np.clip(np.diff(anchor) / STEP, -2, 2)
        delta[1:][np.diff(frame.endpoint) > 45] = 0
    # Standardization is fit on training participants only by train.py.
    logp = np.log(psd + 1e-10)
    logp -= logp.mean(axis=1, keepdims=True)
    return np.column_stack([anchor, available, frame.spectral_hr, frame.spectral_snr,
                            second, half, anchor / 2, logp, delta]).astype(np.float32)


def cache_arrays(frame):
    arrays = {col: frame[col].to_numpy() for col in frame}
    return {key: value.astype(str) if value.dtype.kind == "O" else value
            for key, value in arrays.items()}


def process(job):
    path, device, output, vendor, use_nightbeat, max_seconds, replace = job
    output = Path(output)
    stem = path.stem + "_" + device
    dest = output / "cache" / (stem + ".npz")
    if dest.exists() and not replace:
        return dict(recording=stem, status="already_cached")
    begun = time.monotonic()
    print("Reading " + stem, flush=True)
    d, h = load_columns(path, device)
    if max_seconds:
        # Smoke-run limit affects each independent sensor timestamp, not CSV row index.
        for key in list(d.columns):
            if key.endswith("time"):
                d.loc[pd.to_numeric(d[key], errors="coerce") > max_seconds, key] = np.nan
    xyz, valid, origin, audit = resample(d, device)
    rt, ry, rq, source = reference(d, h)
    frame, psd, freq = extract(xyz, valid, origin, rt, ry, rq, d)
    if frame.empty:
        raise ValueError("No eligible windows for " + stem)
    if use_nightbeat:
        nb = predict(xyz, valid, origin, Path(vendor),
            lambda n, total: print(stem + ": Nightbeat " + str(n) + "/" + str(total), flush=True))
        frame = frame.merge(nb, on="endpoint", how="left", validate="one_to_one")
    else:
        for col in ("nb_anchor", "nb_motion", "nb_peak", "nightbeat"):
            frame[col] = np.nan
    frame["peak_median"] = rolling_median(frame.endpoint.to_numpy(), frame.peak_wide.to_numpy())
    # Neither screen uses the reference HR.
    screen = (frame.motion < frame.motion.median()) & (
        frame.peak_wide_clarity > frame.peak_wide_clarity.median())
    frame["peak_screened"] = frame.peak_wide.where(screen)
    frame["anchor_filled"] = frame.nb_anchor.where(np.isfinite(frame.nb_anchor), frame.spectral_hr)
    frame["anchor_median"] = rolling_median(frame.endpoint.to_numpy(), frame.anchor_filled.to_numpy())
    primary = np.isfinite(frame.reference) & frame.reference.between(30, 120)
    child, night = path.stem.rsplit("_n", 1)
    audit.update(participant=child, night=int(night), device=device, reference_source=source,
        source_path=str(path), source_size=path.stat().st_size, source_mtime_ns=path.stat().st_mtime_ns,
        header_sha256=hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest(),
        windows=len(frame), label_windows=int(primary.sum()), optical_reference=source != "kubios_none_mean_hr",
        nightbeat=bool(use_nightbeat), max_seconds=max_seconds,
        feature_names=FEATURES, seconds_elapsed=round(time.monotonic() - begun, 2))
    arrays = cache_arrays(frame)
    arrays.update(X=features(frame, psd, freq), psd=psd, frequency=freq, eligible=primary.to_numpy(),
                  metadata=np.array(json.dumps(audit)))
    temp = dest.with_suffix(".partial.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(dest)
    print(stem + ": " + str(len(frame)) + " windows, " + str(int(primary.sum())) +
          " primary labels, " + str(audit["seconds_elapsed"]) + " s", flush=True)
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vendor", type=Path)
    ap.add_argument("--devices", nargs="+", choices=DEVICES, default=["actigraph"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--participants", nargs="+")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--nightbeat", action="store_true")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--inventory-only", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cache").mkdir(exist_ok=True)
    inv = inventory(args.data, args.output)
    counts = inv.groupby("device").agg(recordings=("filename", "size"),
        kubios_headers=("kubios_reference", "sum"), psg_headers=("psg", "sum"))
    print(counts.to_string(), flush=True)
    if args.inventory_only:
        return
    chosen = inv[inv.device.isin(args.devices)]
    if args.participants:
        chosen = chosen[chosen.participant.isin(args.participants)]
    if args.limit:
        chosen = chosen.iloc[:args.limit]
    jobs = [(args.data / "raw" / r.filename, r.device, args.output, args.vendor,
             args.nightbeat, args.max_seconds, args.replace) for r in chosen.itertuples()]
    if args.nightbeat and not args.vendor:
        ap.error("--nightbeat requires --vendor")
    manifest = dict(protocol="Weaver external methodological replication, not exact NightbeatDB reproduction",
        command_vars={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        prepared_recordings=len(jobs), devices=args.devices, features=FEATURES,
        window_seconds=WINDOW, step_seconds=STEP, resampled_hz=FS,
        reference="mean Kubios None Mean HR within [endpoint-60,endpoint), quality averaged /100",
        reference_provenance="unconfirmed; no raw ECG/beat timestamps in revised CSV",
        eligibility="finite reference in [30,120] bpm; no hard quality threshold; complete ACC support",
        gap_policy="reject windows crossing >250ms missing ACC; sequence gaps <=45s",
        sources={"paper":"https://arxiv.org/pdf/2610.06823v1",
                 "nightbeat":"https://github.com/eth-siplab/Nightbeat"})
    if args.vendor:
        manifest["nightbeat_commit"] = subprocess.check_output(
            ["git", "-C", str(args.vendor), "rev-parse", "HEAD"], text=True).strip()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.workers == 1:
        result = list(map(process, jobs))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            result = list(pool.map(process, jobs))
    (args.output / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
