"""Aggregate the per-night BPM audit tables into cohort-level CSVs (no participant rows) and one figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COHORTS = ["wearanize", "aauwss", "sleepaccel", "weaver_garmin", "weaver_fitbit"]
DEVICE = {"wearanize": "Empatica E4", "aauwss": "Empatica E4", "sleepaccel": "Apple Watch (cleaned HR)",
          "weaver_garmin": "Garmin", "weaver_fitbit": "Fitbit"}
NYQ_OK = 1 / (2 * 0.15)  # nominal dt (s) at or below which 0.15 Hz is below Nyquist
Q = [0.05, 0.25, 0.5, 0.75, 0.95]


def qs(x, name):
    x = pd.Series(x).dropna()
    if x.empty:
        return {f"{name}_{int(q*100):02d}": np.nan for q in Q} | {f"{name}_n": 0}
    return {f"{name}_{int(q*100):02d}": float(x.quantile(q)) for q in Q} | {f"{name}_n": int(x.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    run = ap.parse_args().run
    loc = run / "local-v2"
    nights = {c: pd.read_parquet(loc / f"nights-{c}.parquet") for c in COHORTS if (loc / f"nights-{c}.parquet").exists()}
    wins = {c: pd.read_parquet(loc / f"windows-{c}.parquet") for c in nights}
    eps = {c: pd.read_parquet(loc / f"epochs-{c}.parquet") for c in nights}

    samp, rep, gap, smooth, act, feas = [], [], [], [], [], []
    for c, n in nights.items():
        n = n[~n.get("no_hr", pd.Series(False, index=n.index)).fillna(False).astype(bool)] if "no_hr" in n else n
        dev = DEVICE[c]
        hours = n.span_s.sum() / 3600
        tot = {k: int(n[k].sum()) for k in n.columns if k.startswith("dtbin_")}
        tn = sum(tot.values())
        samp.append({"cohort": c, "device": dev, "nights": len(n), "psg_hours": round(hours, 1),
                     "samples_in_psg_span": int(n.n_hr_span.sum()),
                     "nominal_dt_s_median": n.nominal_dt.median(),
                     "frac_dt_1s_pooled": float(np.average(n.frac_dt_1s, weights=n.n_hr_span)),
                     "frac_dt_1s_night_min": n.frac_dt_1s.min(), "frac_dt_1s_night_max": n.frac_dt_1s.max(),
                     "dt_p01_median": n.dt_p01.median(), "dt_p50_median": n.dt_p50.median(),
                     "dt_p95_median": n.dt_p95.median(), "dt_p99_median": n.dt_p99.median(),
                     "dt_cv_nongap_median": n.dt_cv_nongap.median(),
                     "nights_nonmonotonic": int((n.nonmonotonic > 0).sum()),
                     "duplicate_timestamps": int(n.dup_ts.sum()),
                     "frac_dt_gt2nominal_pooled": float(np.average(n.frac_dt_gt2nom, weights=n.n_hr_span))}
                    | {f"pooled_frac_dt_{k[6:]}": v / tn for k, v in tot.items()})
        rep.append({"cohort": c, "device": dev,
                    "frac_samples_out_of_30_200": float(np.average(n.frac_out_of_range, weights=n.n_hr_span)),
                    "frac_integer_valued_median": n.frac_integer.median(), "min_step_bpm_median": n.min_step_bpm.median(),
                    "distinct_values_per_night_median": n.n_distinct_values.median(),
                    "frac_successive_diffs_zero_median": n.frac_diff_zero.median(),
                    "rms_successive_diff_bpm_median": n.sd_diff.median(),
                    "runfrac_ge5s_mean": n.runfrac_ge5s.mean(), "runfrac_ge30s_mean": n.runfrac_ge30s.mean(),
                    "runfrac_ge5samples_mean": n.runfrac_ge5samples.mean(), "runfrac_ge30samples_mean": n.runfrac_ge30samples.mean(),
                    "run_len_samples_median_of_nights": n.run_len_median_samples.median(),
                    "run_len_samples_p95_median_of_nights": n.run_len_p95_samples.median(),
                    "run_len_max_samples_over_nights": int(n.run_len_max_samples.max())}
                   | {k + "_mean": n[k].mean() for k in n.columns if k.startswith("runfrac_") and k.count("_") == 2})
        gap.append({"cohort": c, "device": dev, "nights": len(n),
                    "gaps_gt10s_per_night_median": n.n_gaps_gt10s.median(), "gaps_gt10s_per_night_mean": n.n_gaps_gt10s.mean(),
                    "gaps_gt60s_per_night_mean": n.n_gaps_gt60s.mean(), "gaps_gt300s_per_night_mean": n.n_gaps_gt300s.mean(),
                    "gap_total_gt10s_over_cohort": int(n.n_gaps_gt10s.sum()), "gap_total_gt60s": int(n.n_gaps_gt60s.sum()),
                    "gap_total_gt300s": int(n.n_gaps_gt300s.sum()),
                    "gap_dur_median_s_median_of_nights": n.gap_dur_median_s[n.n_gaps_gt10s > 0].median(),
                    "gap_dur_max_s_max": n.gap_dur_max_s.max(),
                    "frac_psg_time_in_gaps_mean": n.frac_time_in_gaps.mean(), "frac_psg_time_in_gaps_median": n.frac_time_in_gaps.median(),
                    "frac_psg_time_in_gaps_p90": n.frac_time_in_gaps.quantile(.9),
                    "frac_epochs_with_any_hr_mean": n.frac_epochs_with_hr.mean(),
                    "frac_epochs_hr_ge80pct_mean": n.frac_epochs_hr_ge80pct.mean(),
                    "nights_with_gap_time_gt10pct": int((n.frac_time_in_gaps > 0.1).sum()),
                    "nights_with_gap_time_gt50pct": int((n.frac_time_in_gaps > 0.5).sum())})
        # smoothing per window
        w = wins[c]
        nominal = n.nominal_dt.median()
        for wm, g in w.groupby("win_min"):
            g = g[g.valid_data]
            row = {"cohort": c, "device": dev, "win_min": wm, "valid_windows": len(g), "nights": g.subject.nunique(),
                   "frac_diff_zero_mean": g.frac_diff_zero.mean()}
            row |= qs(g.acf1_diffs, "acf1_diffs") | qs(g.acf1_series, "acf1_series") | qs(g.rmssd, "rms_diff_bpm")
            row |= qs(g.ratio_low, "ratio_low") | qs(g.ratio_mid, "ratio_mid")
            row |= qs(g.ratio_mida, "ratio_mida")
            if nominal <= 2:
                row |= qs(g.ratio_midb, "ratio_midb")
                row |= qs(g.ratio_hi, "ratio_hi")
                row["frac_windows_ratio_hi_lt_0.5"] = float((g.ratio_hi.dropna() < 0.5).mean())
                row["frac_windows_ratio_hi_lt_0.1"] = float((g.ratio_hi.dropna() < 0.1).mean())
            row["frac_windows_ratio_mid_gt_2"] = float((g.ratio_mid.dropna() > 2).mean())
            row["frac_windows_ratio_low_gt_2"] = float((g.ratio_low.dropna() > 2).mean())
            if "q_floor_ratio_mid" in g:
                row |= qs(g.q_floor_ratio_mid, "quant_floor_ratio_mid")
                row["frac_windows_mid_power_lt_2x_quant_floor"] = float((g.q_floor_ratio_mid.dropna() < 2).mean())
            smooth.append(row)
        # activity
        e = eps[c]
        e = e[e.jerk_rel.notna() & e.psg_scored]
        e["bin"] = pd.cut(e.jerk_rel, [0, 1.5, 3, 10, 30, np.inf], labels=["<1.5x floor", "1.5-3x", "3-10x", "10-30x", ">=30x"], right=False)
        for b, g in e.groupby("bin", observed=True):
            pairs = g.n_pairs.sum()
            act.append({"cohort": c, "device": dev, "jerk_over_floor": b, "epochs": len(g),
                        "mean_abs_successive_diff_bpm": g.sum_abs_diff.sum() / pairs if pairs else np.nan,
                        "frac_pairs_jump_ge3": g.jumps3.sum() / pairs if pairs else np.nan,
                        "frac_pairs_jump_ge5": g.jumps5.sum() / pairs if pairs else np.nan,
                        "frac_pairs_jump_ge10": g.jumps10.sum() / pairs if pairs else np.nan,
                        "frac_epochs_no_valid_hr": float((g.hr_n == 0).mean())})
        act.append({"cohort": c, "device": dev, "jerk_over_floor": "per-night Spearman, jerk vs mean |diff| (median, IQR lo, IQR hi)",
                    "epochs": len(n), "mean_abs_successive_diff_bpm": n.spearman_jerk_absdiff.median(),
                    "frac_pairs_jump_ge3": n.spearman_jerk_absdiff.quantile(.25), "frac_pairs_jump_ge5": n.spearman_jerk_absdiff.quantile(.75)})
        # feasibility
        for wm, g in w.groupby("win_min"):
            W = wm * 60
            tot_w = len(g)
            ok = g.valid_data
            fin = lambda col: g[col].notna() if col in g else pd.Series(False, index=g.index)
            nyq_full = (g.med_dt <= NYQ_OK)
            nyq_hz = 1 / (2 * g.med_dt)
            low_res = W >= 300  # at least 3 cycles of 0.01 Hz
            mid = ok & fin("P_mid") & nyq_full
            mid_partial = ok & fin("P_mid") & (nyq_hz >= 0.08)  # at least 0.04-0.08 Hz below Nyquist
            low = ok & fin("P_low") & low_res
            low_upper = ok & fin("P_low") & (W >= 120)  # 0.025-0.04 Hz only
            diffs = g.frac_cons_pairs >= 0.8
            samp_len = ok & (g.longest_run >= 100)
            se = fin("sampen") if "sampen" in g else pd.Series(False, index=g.index)
            feas.append({"cohort": c, "device": dev, "win_min": wm, "windows_all": tot_w, "nights": g.subject.nunique(),
                         "valid_data (>=80% coverage, no gap > max(15 s, 3 dt))": ok.mean(),
                         "any_valid_sample": (g.n_valid > 0).mean(),
                         "band_0.04-0.15_estimable": mid.mean(),
                         "band_0.04-0.15_partial (to Nyquist, >= 0.08 Hz)": mid_partial.mean(),
                         "band_0.01-0.04_estimable (window >= 300 s)": low.mean(),
                         "band_0.01-0.04_resolved_upper_part_only (0.025-0.04)": low_upper.mean() if W < 300 else 0.0,
                         "successive_diffs (>=80% consecutive pairs)": diffs.mean(),
                         "sampen_run_ge100_samples": samp_len.mean(),
                         "sampen_finite": se.mean(),
                         "valid_data_relaxed (>=50% coverage)": (g.coverage >= 0.5).mean()})

    out = {"sampling": samp, "repeats": rep, "gaps": gap, "smoothing": smooth, "activity": act, "feasibility": feas}
    for k, v in out.items():
        pd.DataFrame(v).to_csv(run / f"{k}_summary.csv", index=False)
        print("==", k); print(pd.DataFrame(v).T.to_string())

    # figure: per-window distributions, 5-min windows
    cols = {"wearanize": "#0072B2", "aauwss": "#56B4E9", "sleepaccel": "#D55E00", "weaver_garmin": "#009E73", "weaver_fitbit": "#CC79A7"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for c, w in wins.items():
        g = w[(w.win_min == 5) & w.valid_data]
        for a, col, lg in ((ax[0], "acf1_diffs", False), (ax[1], "ratio_mid", True), (ax[2], "ratio_hi", True)):
            if col == "ratio_hi" and nights[c].nominal_dt.median() > 2:
                continue
            x = np.sort(g[col].dropna().to_numpy())
            if x.size:
                a.plot(x, np.arange(1, x.size + 1) / x.size, color=cols[c], label=f"{c} (n={x.size})")
            if lg:
                a.set_xscale("log")
    ax[0].set_xlabel("lag-1 autocorrelation of successive differences"); ax[0].axvline(-0.5, color="gray", ls=":")
    ax[1].set_xlabel("power 0.04-0.15 Hz / same window shuffled"); ax[1].axvline(1, color="gray", ls=":")
    ax[2].set_xlabel("power 0.10-0.40 Hz / same window shuffled"); ax[2].axvline(1, color="gray", ls=":")
    ax[1].set_xlim(1e-5, 1e2); ax[2].set_xlim(1e-7, 1e1)
    for a in ax:
        a.set_ylabel("share of 5-min windows"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Per-window BPM statistics, 5-min windows with >= 80% coverage (cumulative distributions; white noise: -0.5 and 1)")
    fig.tight_layout(); fig.savefig(run / "bpm_window_statistics.png", dpi=130)


if __name__ == "__main__":
    main()
