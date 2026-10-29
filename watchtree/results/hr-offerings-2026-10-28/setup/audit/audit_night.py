"""Per-night BPM stream audit for the WatchTree cohorts (Experiment 4, step 1).

    python audit_night.py --cohort sleepaccel --out ../local [--limit N]

Writes participant-level tables to --out (nights, windows, epochs parquet). No aggregation here.
Spectra are senpy NUFFT PSDs computed per window; no spectra are averaged over time.
"""
from __future__ import annotations

import argparse
import zlib
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from senpy import compute_nufft_spectrogram

from arcasleep_watchtree import constants as C
from arcasleep_watchtree.baseline import low_baseline
from arcasleep_watchtree.cohorts import COHORTS, load_night, subject_ids
from arcasleep_watchtree.preprocess import epoch_signals

WINDOWS_MIN = (2, 5, 10)
BANDS = {"low": (0.01, 0.04), "mid": (0.04, 0.15), "hi": (0.10, 0.40), "mida": (0.04, 0.08), "midb": (0.08, 0.15)}
MIN_COVERAGE = 0.8
SAMPEN_MIN_RUN = 100
RUN_BINS = [1, 2, 3, 4, 5, 10, 30, 60, 10**9]
_STATE: dict = {}


def _init(cohort_name):
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    from arcasleep_watchtree.cohorts import DATA_ROOT

    c = COHORTS[cohort_name]
    _STATE["cohort"] = c
    _STATE["data_set"] = load_dataset(DATA_ROOT / c.directory)


def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    n = x.size
    sd = x.std()
    if n < m + 2 or sd == 0:
        return np.nan
    r = r_frac * sd

    def count(mm):
        tem = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        d = np.abs(tem[:, None, :] - tem[None, :, :]).max(axis=2)
        return (np.count_nonzero(d <= r) - tem.shape[0]) / 2.0

    b, a = count(m), count(m + 1)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def band_power(freqs, psd, lo, hi):
    sel = (freqs >= lo) & (freqs < hi)
    if sel.sum() == 0:
        return np.full(psd.shape[0], np.nan)
    return psd[:, sel].sum(axis=1) * (freqs[1] - freqs[0])


def spectra(t, x, win_s, nwin, seed):
    """Per-window NUFFT PSD of the series and of the same series shuffled within each window."""
    out = {}
    rng = np.random.default_rng(seed)
    wid = (t // win_s).astype(int)
    xs = x.copy()
    for w in np.unique(wid):
        idx = np.flatnonzero(wid == w)
        xs[idx] = x[rng.permutation(idx)]
    for key, y in (("obs", x), ("shuf", xs)):
        try:
            r = compute_nufft_spectrogram(
                t, y, float(win_s), 0.0, kind="psd", origin_s=0.0, empty_windows="keep", min_samples=8
            )
        except Exception:  # too few samples overall
            return None
        full = np.full((nwin, r.Sxx.shape[1]), np.nan)
        ok = r.valid & (r.window_index < nwin)
        full[r.window_index[ok]] = r.Sxx[ok]
        out[key] = full
        out["freqs"] = r.frequencies
    return out


def runs_of(values, t, brk):
    """Run lengths (samples) and durations (s) of identical consecutive values; ``brk`` marks a gap before i."""
    change = np.r_[True, (values[1:] != values[:-1]) | brk[1:]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], values.size]
    return ends - starts, t[ends - 1] - t[starts]


def audit_night(args):
    subject, reported = args
    c = _STATE["cohort"]
    try:
        night = load_night(c, subject, reported, _STATE["data_set"])
    except Exception as exc:  # noqa: BLE001
        return {"failed": type(exc).__name__}
    n_ep = night.labels.size
    span = n_ep * C.EPOCH_S
    t_all, b_all = night.hr_t.astype(float), night.hr_bpm.astype(float)
    order = np.argsort(t_all, kind="stable")
    nonmono = int(np.count_nonzero(np.diff(t_all) < 0))
    t_all, b_all = t_all[order], b_all[order]
    sel = (t_all >= 0) & (t_all < span)
    t, b = t_all[sel], b_all[sel]
    rec = {"cohort": c.name, "subject": night.subject, "n_epochs": n_ep, "span_s": span,
           "n_hr_total": int(t_all.size), "n_hr_span": int(t.size), "nonmonotonic": nonmono}
    wins, eps = [], None
    if t.size < 10:
        rec["no_hr"] = True
        return {"night": rec, "windows": pd.DataFrame(), "epochs": pd.DataFrame()}
    dt = np.diff(t)
    pos = dt[dt > 0]
    nominal = float(np.median(pos)) if pos.size else np.nan
    rec.update(nominal_dt=nominal, dup_ts=int(np.count_nonzero(dt == 0)),
               dt_p01=np.percentile(dt, 1), dt_p05=np.percentile(dt, 5), dt_p50=np.median(dt),
               dt_p95=np.percentile(dt, 95), dt_p99=np.percentile(dt, 99), dt_max=dt.max(),
               frac_dt_1s=float(np.mean(np.abs(dt - 1.0) <= 0.05)),
               frac_dt_le1p5=float(np.mean(dt <= 1.5)),
               frac_dt_gt2nom=float(np.mean(dt > 2 * nominal)),
               dt_cv_nongap=float(np.std(dt[dt <= 10]) / np.mean(dt[dt <= 10])) if np.any(dt <= 10) else np.nan)
    # histogram counts of dt for pooling
    for name, lo, hi in [("0", 0, 0), ("le0.9", 1e-9, 0.9), ("0.9-1.1", 0.9, 1.1), ("1.1-2", 1.1, 2.0),
                         ("2-4", 2.0, 4.0), ("4-6", 4.0, 6.0), ("6-10", 6.0, 10.0), ("10-30", 10.0, 30.0),
                         ("30+", 30.0, 1e18)]:
        if name == "0":
            rec["dtbin_0"] = int(np.count_nonzero(dt <= 0))
        else:
            rec[f"dtbin_{name}"] = int(np.count_nonzero((dt > lo) & (dt <= hi))) if name != "0.9-1.1" else int(np.count_nonzero((dt > 0.9) & (dt <= 1.1)))
    valid = np.isfinite(b) & (b >= C.HR_MIN_BPM) & (b <= C.HR_MAX_BPM)
    rec["frac_out_of_range"] = float(1 - valid.mean())
    rec["frac_integer"] = float(np.mean(b == np.round(b)))
    u = np.unique(np.round(b, 4))
    rec["min_step_bpm"] = float(np.diff(u).min()) if u.size > 1 else np.nan
    rec["n_distinct_values"] = int(u.size)
    # runs of identical consecutive values (raw stream, break at gaps)
    brk = np.r_[False, dt > 2 * nominal]
    rl, rd = runs_of(b, t, brk)
    rd = rd + nominal  # a run of k samples spans k * nominal
    per_sample_len = np.repeat(rl, rl)
    per_sample_dur = np.repeat(rd, rl)
    for lo, hi in zip(RUN_BINS[:-1], RUN_BINS[1:]):
        rec[f"runfrac_{lo}_{hi if hi < 10**8 else 'inf'}"] = float(np.mean((per_sample_len >= lo) & (per_sample_len < hi)))
    rec["runfrac_ge5s"] = float(np.mean(per_sample_dur >= 5))
    rec["runfrac_ge30s"] = float(np.mean(per_sample_dur >= 30))
    rec["runfrac_ge5samples"] = float(np.mean(per_sample_len >= 5))
    rec["runfrac_ge30samples"] = float(np.mean(per_sample_len >= 30))
    rec["run_len_median_samples"] = float(np.median(per_sample_len))
    rec["run_len_p95_samples"] = float(np.percentile(per_sample_len, 95))
    rec["run_len_max_samples"] = int(rl.max())
    # gaps (valid samples only; includes the lead-in and tail-out of the PSG span)
    tv, bv = t[valid], b[valid]
    edges = np.r_[0.0, tv, float(span)] if tv.size else np.array([0.0, float(span)])
    gaps = np.diff(edges)
    thr = max(10.0, 2 * nominal)
    big = gaps[gaps > thr]
    rec["n_gaps_gt10s"] = int(np.count_nonzero(gaps > 10))
    rec["n_gaps_gt60s"] = int(np.count_nonzero(gaps > 60))
    rec["n_gaps_gt300s"] = int(np.count_nonzero(gaps > 300))
    rec["gap_dur_median_s"] = float(np.median(big)) if big.size else 0.0
    rec["gap_dur_max_s"] = float(gaps.max())
    rec["gap_total_s"] = float(big.sum())
    cover = np.minimum(gaps, thr) if False else None
    rec["frac_time_in_gaps"] = float(big.sum() / span)
    hr_n_ep = np.bincount((tv // C.EPOCH_S).astype(int), minlength=n_ep)[:n_ep] if tv.size else np.zeros(n_ep)
    exp_n = C.EPOCH_S / nominal
    rec["frac_epochs_with_hr"] = float(np.mean(hr_n_ep > 0))
    rec["frac_epochs_hr_ge50pct"] = float(np.mean(hr_n_ep >= 0.5 * exp_n))
    rec["frac_epochs_hr_ge80pct"] = float(np.mean(hr_n_ep >= 0.8 * exp_n))
    # consecutive pairs
    cons_thr = 1.5 * nominal
    dtv = np.diff(tv)
    cons = (dtv <= cons_thr) & (dtv > 0)
    d = np.diff(bv)
    dc = d[cons]
    rec["sd_diff"] = float(np.sqrt(np.mean(dc**2))) if dc.size else np.nan
    rec["frac_diff_zero"] = float(np.mean(dc == 0)) if dc.size else np.nan
    # lag-1 autocorrelation of successive differences (consecutive triples), whole night
    tri = cons[1:] & cons[:-1]
    d1, d2 = d[:-1][tri], d[1:][tri]
    rec["acf1_diffs_night"] = float(np.corrcoef(d1, d2)[0, 1]) if d1.size > 30 and d1.std() > 0 and d2.std() > 0 else np.nan
    # windows
    int_cohort = rec["frac_integer"] > 0.9
    for wm in WINDOWS_MIN:
        W = wm * 60
        nwin = int(span // W)
        if nwin == 0:
            continue
        sp = spectra(tv, bv, W, nwin, seed=zlib.crc32(night.subject.encode()) + wm)
        P = {}
        if sp is not None:
            for k in ("obs", "shuf"):
                for bn, (lo, hi) in BANDS.items():
                    P[(k, bn)] = band_power(sp["freqs"], sp[k], lo, hi)
        wid = (tv // W).astype(int)
        starts = np.searchsorted(wid, np.arange(nwin))
        ends = np.searchsorted(wid, np.arange(nwin), side="right")
        for w in range(nwin):
            tw, bw = tv[starts[w]: ends[w]], bv[starts[w]: ends[w]]
            n = tw.size
            row = {"cohort": c.name, "subject": night.subject, "win_min": wm, "win": w, "n_valid": n,
                   "expected": W / nominal, "coverage": min(1.0, n * nominal / W) if n else 0.0}
            if n:
                e = np.r_[w * W, tw, (w + 1) * W]
                row["max_gap_s"] = float(np.diff(e).max())
            else:
                row["max_gap_s"] = float(W)
            row["med_dt"] = float(np.median(np.diff(tw))) if n > 2 else np.nan
            rowcons = (np.diff(tw) <= cons_thr)
            row["n_cons_pairs"] = int(rowcons.sum())
            row["frac_cons_pairs"] = float(rowcons.sum() / max(1, W / nominal - 1))
            # longest consecutive run
            if n:
                br = np.flatnonzero(~rowcons)
                seg = np.diff(np.r_[-1, br, n - 1])
                longest_idx = int(np.argmax(seg))
                lo_i = (np.r_[-1, br][longest_idx]) + 1
                row["longest_run"] = int(seg.max()) if br.size else n
                seg_start = lo_i
                seg_len = row["longest_run"]
            else:
                row["longest_run"] = 0
                seg_start = seg_len = 0
            good = row["coverage"] >= MIN_COVERAGE and row["max_gap_s"] <= max(15.0, 3 * nominal)
            row["valid_data"] = bool(good)
            if n >= 10:
                row["sd_bpm"] = float(bw.std())
                dw = np.diff(bw)[rowcons]
                row["rmssd"] = float(np.sqrt(np.mean(dw**2))) if dw.size else np.nan
                row["frac_diff_zero"] = float(np.mean(dw == 0)) if dw.size else np.nan
                triw = rowcons[1:] & rowcons[:-1]
                a1, a2 = np.diff(bw)[:-1][triw], np.diff(bw)[1:][triw]
                row["acf1_diffs"] = float(np.corrcoef(a1, a2)[0, 1]) if a1.size > 10 and a1.std() > 0 and a2.std() > 0 else np.nan
                xc = bw - bw.mean()
                row["acf1_series"] = float(np.sum(xc[:-1] * xc[1:]) / np.sum(xc**2)) if np.sum(xc**2) > 0 else np.nan
            if sp is not None and np.isfinite(P[("obs", "mid")][w]):
                for bn in BANDS:
                    row[f"P_{bn}"] = P[("obs", bn)][w]
                    row[f"ratio_{bn}"] = P[("obs", bn)][w] / P[("shuf", bn)][w] if P[("shuf", bn)][w] > 0 else np.nan
                if int_cohort and n >= 10 and bw.var() > 0:
                    row["q_floor_ratio_mid"] = P[("obs", "mid")][w] / (P[("shuf", "mid")][w] * (1 / 12) / bw.var())
                    row["q_floor_ratio_low"] = P[("obs", "low")][w] / (P[("shuf", "low")][w] * (1 / 12) / bw.var())
            if seg_len >= SAMPEN_MIN_RUN and good:
                row["sampen"] = sample_entropy(bw[seg_start: seg_start + seg_len])
            wins.append(row)
    # activity vs bpm jumps per epoch
    sig = epoch_signals(night.acc25, night.valid25)
    jerk = np.asarray(sig.jerk, float)
    base = low_baseline(jerk)
    rel = jerk / base
    ep_of = (tv // C.EPOCH_S).astype(int)
    pair_ok = np.r_[False, cons]  # pair (i-1, i) consecutive
    same_ep = np.r_[False, ep_of[1:] == ep_of[:-1]]
    pm = pair_ok & same_ep
    absd = np.r_[0.0, np.abs(d)]
    n_pairs = np.bincount(ep_of[pm], minlength=n_ep)[:n_ep]
    sum_abs = np.bincount(ep_of[pm], weights=absd[pm], minlength=n_ep)[:n_ep]
    j3 = np.bincount(ep_of[pm], weights=(absd[pm] >= 3), minlength=n_ep)[:n_ep]
    j5 = np.bincount(ep_of[pm], weights=(absd[pm] >= 5), minlength=n_ep)[:n_ep]
    j10 = np.bincount(ep_of[pm], weights=(absd[pm] >= 10), minlength=n_ep)[:n_ep]
    eps = pd.DataFrame({"cohort": c.name, "subject": night.subject, "epoch": np.arange(n_ep),
                        "jerk": jerk, "jerk_rel": rel, "hr_n": hr_n_ep, "n_pairs": n_pairs,
                        "sum_abs_diff": sum_abs, "jumps3": j3, "jumps5": j5, "jumps10": j10,
                        "psg_scored": night.labels >= 0})
    # per-night Spearman of epoch jerk vs mean |diff| (epochs with >=5 pairs)
    m = np.isfinite(rel) & (n_pairs >= 5)
    if m.sum() > 30:
        rec["spearman_jerk_absdiff"] = float(spearmanr(rel[m], sum_abs[m] / n_pairs[m])[0])
        rec["spearman_jerk_hrn"] = float(spearmanr(rel[np.isfinite(rel)], hr_n_ep[np.isfinite(rel)])[0])
    return {"night": rec, "windows": pd.DataFrame(wins), "epochs": eps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=sorted(COHORTS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ids, reported = subject_ids(COHORTS[a.cohort])
    ids = ids[: a.limit] if a.limit else ids
    a.out.mkdir(parents=True, exist_ok=True)
    nights, wins, eps, failed = [], [], [], {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_init, initargs=(a.cohort,)) as pool:
        for r in pool.map(audit_night, [(s, reported[s]) for s in ids]):
            if "failed" in r:
                failed[r["failed"]] = failed.get(r["failed"], 0) + 1
                continue
            nights.append(r["night"]); wins.append(r["windows"]); eps.append(r["epochs"])
    pd.DataFrame(nights).to_parquet(a.out / f"nights-{a.cohort}.parquet")
    pd.concat(wins, ignore_index=True).to_parquet(a.out / f"windows-{a.cohort}.parquet")
    pd.concat(eps, ignore_index=True).to_parquet(a.out / f"epochs-{a.cohort}.parquet")
    print(a.cohort, "nights", len(nights), "failed", failed, flush=True)


if __name__ == "__main__":
    main()
