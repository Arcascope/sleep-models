"""Compare original SleepAccel peak finding, Weaver transfer, and SleepAccel CV."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from full import grouped_folds
from nightbeat_adapter import predict, rolling_median
from prepare import FEATURES, cache_arrays, extract, features, resample
from report import bootstrap
from train import MODEL, dataset, predict_dataset, sequence_ends

HERE = Path(__file__).resolve().parent
SOURCE = "apple_optical_hr_median"


def peaks50(raw, endpoints):
    """October 6 numpy FFT on a 50-Hz grid, at the requested 60-s windows."""
    raw = raw[np.argsort(raw[:, 0])]
    grid = np.arange(0., raw[-1, 0], .02)
    xyz = np.column_stack([np.interp(grid, raw[:, 0], raw[:, k]) for k in (1, 2, 3)])
    f = np.fft.rfftfreq(2999, .02)
    hann = np.hanning(2999)
    rows = []
    for end in endpoints:
        lo = int(round((end-60) * 50))
        seg = xyz[lo:lo+3000]
        row = dict(endpoint=end, peak50_narrow=np.nan, peak50_wide=np.nan, clarity50=np.nan, motion50=np.nan)
        if lo >= 0 and len(seg) == 3000:
            jerk = np.linalg.norm(np.diff(seg, axis=0), axis=1) * 50
            power = np.abs(np.fft.rfft((jerk-jerk.mean()) * hann)) ** 2
            row["motion50"] = np.linalg.norm(seg, axis=1).std()
            for method, low, high in [("peak50_narrow", .9, 1.3), ("peak50_wide", .5, 2.)]:
                band = (f >= low) & (f <= high)
                bp = power[band]
                k = np.argmax(bp)
                row[method] = float(f[band][k] * 60)
                if method == "peak50_narrow":
                    row["clarity50"] = float(bp[k] / bp.mean()) if bp.mean() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_subject(job):
    sid, data, root, vendor = job
    dest = root / "cache" / (sid + "_n1_apple.npz")
    if dest.exists():
        return dict(participant=sid, status="already_cached")
    begun = time.monotonic()
    motion_path = data / "cleaned_accelerometer" / (sid + "_cleaned_motion.out")
    hr_path = data / "cleaned_heartrate" / (sid + "_cleaned_hr.out")
    psg_path = data / "cleaned_psg" / (sid + "_cleaned_psg.out")
    raw = np.loadtxt(motion_path)
    hr = np.loadtxt(hr_path)
    hr = hr[np.isfinite(hr).all(axis=1)]
    hr = hr[np.argsort(hr[:, 0], kind="stable")]
    psg = np.loadtxt(psg_path)
    d = pd.DataFrame(raw, columns=["appletime", "applex", "appley", "applez"])
    d["psgtime"] = pd.Series(psg[:, 0])
    d["psgstg"] = pd.Series(psg[:, 1]).map({0:"W", 1:"N1", 2:"N2", 3:"N3", 4:"N3", 5:"REM"})
    xyz, valid, origin, meta = resample(d, "apple")
    frame, psd, frequency = extract(xyz, valid, origin, hr[:, 0], hr[:, 1], np.ones(len(hr)), d)
    frame["reference_mean"] = frame.reference
    medians = []
    for end in frame.endpoint:
        a, b = np.searchsorted(hr[:, 0], [end-60, end])
        medians.append(float(np.median(hr[a:b, 1])) if b > a else np.nan)
    frame["reference"] = medians
    nb = predict(xyz, valid, origin, vendor)
    frame = frame.merge(nb, on="endpoint", how="left", validate="one_to_one")
    frame = frame.merge(peaks50(raw, frame.endpoint.to_numpy()), on="endpoint", validate="one_to_one")
    frame["still_clear50"] = (frame.motion50 < frame.motion50.median()) & (frame.clarity50 > frame.clarity50.median())
    frame["peak_median"] = rolling_median(frame.endpoint.to_numpy(), frame.peak_wide.to_numpy())
    frame["peak_screened"] = frame.peak_wide.where((frame.motion < frame.motion.median()) & (frame.peak_wide_clarity > frame.peak_wide_clarity.median()))
    frame["anchor_filled"] = frame.nb_anchor.where(np.isfinite(frame.nb_anchor), frame.spectral_hr)
    frame["anchor_median"] = rolling_median(frame.endpoint.to_numpy(), frame.anchor_filled.to_numpy())
    eligible = np.isfinite(frame.reference) & frame.reference.between(30, 120)
    meta.update(participant=sid, night=1, device="apple", reference_source=SOURCE, nightbeat=True,
                max_seconds=None, reference_quality="not provided; unit training weights",
                windows=len(frame), label_windows=int(eligible.sum()), source_path=str(motion_path),
                motion_size=motion_path.stat().st_size, motion_mtime_ns=motion_path.stat().st_mtime_ns,
                hr_size=hr_path.stat().st_size, hr_mtime_ns=hr_path.stat().st_mtime_ns,
                seconds_elapsed=time.monotonic()-begun, feature_names=FEATURES)
    arrays = cache_arrays(frame)
    arrays.update(X=features(frame, psd, frequency), eligible=eligible.to_numpy(), metadata=np.array(json.dumps(meta)))
    np.savez_compressed(dest, **arrays)
    print(f"{sid}: {len(frame)} windows, prepared in {meta['seconds_elapsed']:.1f}s", flush=True)
    return meta


def load(cache):
    recordings = []
    for p in sorted(cache.glob("*.npz")):
        with np.load(p, allow_pickle=False) as a:
            recordings.append(dict(path=p, meta=json.loads(str(a["metadata"])), arrays={k:a[k] for k in a.files if k != "metadata"}))
    return recordings


def base_frame(recordings):
    frames = []
    for r in recordings:
        a = r["arrays"]
        cols = {k:v for k,v in a.items() if np.ndim(v) == 1 and len(v) == len(a["endpoint"])}
        f = pd.DataFrame(cols)
        f["participant"] = r["meta"]["participant"]
        f["recording"] = r["path"].stem
        ends = sequence_ends(a["endpoint"])
        f["model_context"] = False
        f.loc[ends, "model_context"] = True
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def transfer(recordings, weaver, root):
    path = root / "weaver_transfer.csv.gz"
    if path.exists():
        return pd.read_csv(path, dtype={"participant":str})
    ids = [r["meta"]["participant"] for r in recordings]
    data = dataset(recordings, ids, 1)
    frame = pd.DataFrame([dict(participant=r["meta"]["participant"], recording=r["path"].stem,
                               endpoint=float(r["arrays"]["endpoint"][end])) for r,end in data["info"]])
    template = MODEL.init(jax.random.PRNGKey(0), jnp.zeros((1,40,26)))["params"]
    predictions, constants = [], []
    for fold in sorted((weaver / "models").glob("fold*")):
        config = json.loads((fold / "best.json").read_text())
        assert config["feature_names"] == FEATURES
        params = serialization.from_bytes(template, (fold / "best.msgpack").read_bytes())
        with np.load(fold / "best.normalization.npz") as norm:
            assert norm["feature_names"].tolist() == FEATURES
            pred = predict_dataset(params, data, norm["mean"], norm["std"])
        frame["transfer_"+fold.name] = pred
        predictions.append(pred)
        constants.append(config["training_mean"])
    assert len(predictions) == 5
    frame["seqsmoother_weaver_transfer"] = np.mean(predictions, axis=0)
    frame["constant_weaver_training_mean"] = float(np.mean(constants))
    frame.to_csv(path, index=False)
    return frame


def summary(frame, root, stage):
    rng = np.random.default_rng(20261008)
    methods = ["peak50_narrow", "peak50_wide", "peak_narrow", "peak_wide", "peak_median", "nb_anchor", "anchor_median", "nightbeat",
               "seqsmoother_weaver_transfer", "constant_weaver_training_mean", "seqsmoother", "training_mean"]
    rows = []
    eligible = frame.eligible.astype(bool)
    shared = eligible & frame.model_context
    if "seqsmoother" in frame:
        shared &= np.isfinite(frame.seqsmoother)
    masks = [("all_baseline_eligible", eligible), ("matched_model_windows", shared),
             ("matched_still_clear", shared & frame.still_clear50),
             ("matched_still_clear_in_band", shared & frame.still_clear50 & frame.reference.between(54,78)),
             ("matched_in_band", shared & frame.reference.between(54,78)),
             ("nightbeat_retained_matched", shared & np.isfinite(frame.nightbeat))]
    for subset, mask in masks:
        for method in methods:
            if method not in frame:
                continue
            selected = mask & np.isfinite(frame[method])
            if not selected.any():
                continue
            p = frame.loc[selected].copy()
            p["error"] = (p[method]-p.reference).abs()
            macro = p.groupby(["participant","recording"]).error.mean().groupby("participant").mean()
            low, high = bootstrap(macro, rng)
            rows.append(dict(subset=subset, method=method, participants=len(macro), windows=len(p),
                             eligible_windows=int(mask.sum()), availability=len(p)/mask.sum(),
                             participant_macro_mae=float(macro.mean()), mae_ci_low=low, mae_ci_high=high,
                             pooled_mae=float(p.error.mean()), pooled_median_abs_error=float(p.error.median()),
                             within_5_bpm=float((p.error<=5).mean())))
    table = pd.DataFrame(rows)
    table.to_csv(root/"summary.csv", index=False)
    frame.to_csv(root/"predictions.csv.gz", index=False)
    text = "Status: " + stage + "\n\nReference: median Apple optical HR in the same half-open 60-s window.\nWeaver transfer: mean of five frozen Weaver fold predictions; no SleepAccel labels used for model fitting.\nSleepAccel CV: five participant-grouped folds, training on optical HR labels.\n\n"
    text += table[table.subset.eq("matched_model_windows")].to_string(index=False)+"\n"
    (root/"report.txt").write_text(text)
    print(text, flush=True)
    # Preserve the October 6 cohort and original metric as an independent audit.
    (root/"status.json").write_text(json.dumps(dict(status=stage, scored_participants=frame.loc[shared,"participant"].nunique(),
        model_training_devices=dict(weaver_transfer="Actigraph",sleepaccel_cv="Apple Watch")),default=int,indent=2)+"\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weaver", type=Path, required=True)
    ap.add_argument("--vendor", type=Path, required=True)
    ap.add_argument("--old-epochs", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    root=a.output.resolve()
    (root/"cache").mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(a.old_epochs,dtype={"subject_id":str})
    old=old[old.cohort.eq("SleepAccel")].copy()
    # Epoch 0's centered window ends at 45 s; epoch 1 ends at 75 s.
    old["endpoint"]=45.+30.*old.groupby("subject_id").cumcount()
    ids=sorted(old.subject_id.unique())
    protocol=dict(seed=20261007, reference_source=SOURCE, pilot_participants=ids, folds=grouped_folds(ids),
                  comparison="Same 28 SleepAccel subjects as October 6; all nights/devices grouped by person")
    split=root/"splits.json"
    if split.exists() and json.loads(split.read_text()) != protocol:
        raise ValueError("Frozen split changed")
    split.write_text(json.dumps(protocol,indent=2)+"\n")
    with ProcessPoolExecutor(a.workers) as pool:
        audits=list(pool.map(prepare_subject,[(sid,a.data,root,a.vendor) for sid in ids]))
    (root/"preparation.json").write_text(json.dumps(audits,indent=2)+"\n")
    records=load(root/"cache")
    frame=base_frame(records)
    matched=frame.merge(old[["subject_id","endpoint","hr_est","watch_hr","still_clear"]],
                        left_on=["participant","endpoint"],right_on=["subject_id","endpoint"],how="inner",validate="one_to_one")
    original=old.dropna(subset=["hr_est","watch_hr"])
    delta=(matched.peak50_narrow-matched.hr_est).abs()
    refdelta=(matched.reference-matched.watch_hr).abs()
    audit=dict(original_windows=len(original),matched_windows=len(matched),
               peak50_max_difference=float(delta.max()),reference_median_max_difference=float(refdelta.max()),
               discarded_original_windows=len(original)-int(matched.hr_est.notna().sum()),
               gap_policy="New features reject ACC gaps over .25 s; original Oct 6 interpolated across all gaps")
    if audit["peak50_max_difference"] > 1e-5 or audit["reference_median_max_difference"] > 1e-5:
        raise ValueError("Original October 6 reproduction mismatch: "+str(audit))
    (root/"oct6_reproduction_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    original.to_csv(root/"oct6_original_epochs.csv.gz",index=False)
    frame=frame.merge(transfer(records,a.weaver,root),on=["participant","recording","endpoint"],how="left",validate="one_to_one")
    summary(frame,root,"transfer_complete_sleepaccel_cv_pending")
    for fold in protocol["folds"]:
        subprocess.run([sys.executable,str(HERE/"train.py"),"--cache",str(root/"cache"),"--reference-source",SOURCE,
                        "--splits",str(split),"--output",str(root/"models"),"--folds",str(fold["fold"])],check=True)
    learned=pd.concat([pd.read_csv(p,dtype={"participant":str}) for p in (root/"models").glob("fold*/test_predictions.csv.gz")],ignore_index=True)
    frame=frame.merge(learned[["participant","recording","endpoint","seqsmoother","training_mean"]],on=["participant","recording","endpoint"],how="left",validate="one_to_one")
    summary(frame,root,"complete")


if __name__ == "__main__":
    main()
