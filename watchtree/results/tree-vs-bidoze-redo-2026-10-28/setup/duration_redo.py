"""Experiment 3a: sleep-duration measures of the tree stager and BiDoze v0.2.0 on cohorts other than the discovery set.

Models per night
  tree_{hr,nohr}_{off,native}  tree with and without the fitted stage offsets, mean over 3 seeds per night (hr = with heart rate;
                               not available for EESM19 and SleepAccel-Clinical)
  bidoze_argmax                original BiDoze call: wake when P(wake) is the largest of the four stage probabilities
  bidoze_sum05                 sleep when P(light)+P(deep)+P(REM) >= P(wake) (stages are not used)
Trees: SleepAccel nights are scored by the fold model that held them out; AAUWSS, BIDSleep, EESM19 and SleepAccel-Clinical by the
final model (trained on Weaver Garmin/Fitbit, SleepAccel and Wearanize+; none of these cohorts trains it).
BiDoze: SleepAccel out-of-fold; the others the adult v0.2.0 release (see bidoze_probs.py).
Wearanize+ (out-of-fold, whole PSG span) is run as a reference: it reproduces sleep_metrics_summary.csv of tst-complete-context.

Two result sets, kept apart
  complete  nights in which every labelled PSG epoch has complete tree context (>= 2 h behind, >= 45 min ahead). Each model is scored on
            every labelled epoch (BiDoze: epochs it scores; its gap epochs, like missing epochs, count as wake only when scored).
            Metrics as in tst-complete-context-2026-10-28/setup/sleep_metrics.py.
  shared    every night; only the labelled epochs that the trees (complete context) and BiDoze (not a gap) all score; reference and
            all models use this same epoch set. Nights need >= 120 shared epochs. TST is the shared-epoch sum. Onset and final wake
            are reported only for nights whose reference onset (final wake) over the whole labelled PSG falls inside the shared
            window, and WASO and efficiency only when both do; otherwise they are NaN (the tree cannot score that part of the night).
Per night metrics (minutes, model minus PSG): TST, WASO, sleep onset (from the start of the PSG span; 10 min run of sleep), final wake
(end of the last 10 min run of sleep), sleep period efficiency (%).
"""
import argparse, json, sys, time
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, "/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees/watchtree/results/cv-bidoze-scheme-2026-10-27/setup")
from common import *  # noqa: E402,F401,F403
from arcasleep_watchtree import cv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent)
ap.add_argument("--oof", type=Path, required=True)
ap.add_argument("--probs", type=Path, required=True)
ap.add_argument("--limit", type=int, help="nights per cohort (smoke test)")
ap.add_argument("--cohorts", default="sleepaccel,aauwss,eesm19,bidsleep,sleepaccel_clinical,wearanize")
ap.add_argument("--draws", type=int, default=5000)
ap.add_argument("--seed", type=int, default=20261028)
a = ap.parse_args()
OUT = a.out
OUT.mkdir(exist_ok=True)
STEP, RUN_MIN, MIN_SHARED = 0.5, 20, cv.MIN_NIGHT_EPOCHS
METRICS = ("tst", "waso", "onset", "final_wake", "eff")
NO_HR = {"eesm19", "sleepaccel_clinical"}
OOFC = {"wearanize": "Wearanizeplus", "sleepaccel": "SleepAccel"}
BZ = ("bidoze_argmax", "bidoze_sum05", "bidoze_sum05_8h")
TREES = ("tree_hr_off", "tree_hr_native", "tree_nohr_off", "tree_nohr_native")
t0 = time.time()


def log(*x):
    print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)


# ------------------------------------------------------------------------------------------------ tables
def person(cohort, s):
    return s.split("_n")[0] if cohort == "bidsleep" else cv.person_of(cohort, s)


def table(cohort):
    path = {"wearanize": None, "sleepaccel": None, "aauwss": None}.get(cohort, RUN / "features" / f"features-{cohort}.parquet")
    t = load(cohort, path)[1].reset_index(drop=True)
    if a.limit:  # smoke test: the first nights that have BiDoze scores (Wearanize+: first sorted nights)
        have = sorted(set(t.subject) & set(pd.read_parquet(a.probs, columns=["cohort", "subject"]).query("cohort == @cohort").subject)) \
            if cohort != "wearanize" else sorted(t.subject.unique())
        keep = have[: a.limit]
        t = t[t.subject.isin(keep)].reset_index(drop=True)
    t["person"] = [person(cohort, s) for s in t.subject]
    return t


def bidoze(cohort, tab):
    """subject -> dict(call, p) arrays aligned to the table's epoch grid (call -1 = not scored)."""
    out = {}
    if cohort == "wearanize":
        w = pd.read_parquet(WT / "results/cv-2026-10-27/bidoze_epochs.parquet")
        first = pd.read_parquet(UNC, columns=["subject", "epoch", "in_psg"]).query("in_psg").groupby("subject").epoch.min().rename("first")
        w = w.merge(first, left_on="subject", right_index=True)
        w["epoch"] += w["first"]
        for s, g in w.groupby("subject"):
            out[s] = g.set_index("epoch")[["call"]]
    else:
        p = pd.read_parquet(a.probs)
        p = p[p.cohort == cohort]
        for s, g in p.groupby("subject"):
            out[s] = g.set_index("epoch")[["call", "p_wake", "p_light", "p_deep", "p_rem"]]
    if cohort in OOFC:                     # out-of-fold summed 0.5 calls: 25 h (PSG span) and 8 h (whole recording, probability average)
        for m in ("25h", "8h"):
            o = pd.read_parquet(a.oof / f"{m}_{OOFC[cohort]}.parquet")
            for s, g in o.groupby("night"):
                if s not in out:
                    continue
                g = g.set_index("epoch")
                out[s] = out[s].join(pd.DataFrame({f"s{m}": g.sleep_call_summed05.astype(float), f"g{m}": (g.gap_flag | g.get("gap_head_call", False)).astype(float)}), how="left")
    return out


def tree_preds(cohort, tab):
    """config -> (n_seeds, n_rows) predicted stage for every row of tab."""
    cfgs = [c for c in ("hr", "nohr") if not (c == "hr" and cohort in NO_HR)]
    res = {f"tree_{c}_{o}": np.zeros((len(SEEDS), len(tab)), int) for c in cfgs for o in ("off", "native")}
    if cohort in ("sleepaccel", "wearanize"):
        ds = {"sleepaccel": "BiDozeSleepAccelFold{}", "wearanize": "BiDozeWearanizeFold{}"}[cohort]
        parts = []
        for k in range(5):
            ids = set(test_ids(f"fold{k}").get(ds.format(k), []))
            parts.append((f"fold{k}", np.flatnonzero(tab.subject.isin(ids).to_numpy())))
    else:
        parts = [("final", np.arange(len(tab)))]
    for phase, ix in parts:
        if len(ix) == 0:
            continue
        sub = tab.iloc[ix]
        for c in cfgs:
            for si, seed in enumerate(SEEDS):
                b, off = load_model(phase, c, seed)
                r = raw(b, sub, CFGS[c])
                res[f"tree_{c}_native"][si, ix] = r.argmax(1)
                res[f"tree_{c}_off"][si, ix] = (r + off).argmax(1)
    covered = np.zeros(len(tab), bool)
    for _, ix in parts:
        covered[ix] = True
    return res, covered


# ------------------------------------------------------------------------------------------------ night metrics
def runs(sleep):
    out, s0 = [], None
    for i, s in enumerate(np.append(sleep, False)):
        if s and s0 is None:
            s0 = i
        if not s and s0 is not None:
            if i - s0 >= RUN_MIN:
                out.append((s0, i))
            s0 = None
    return out


def night(sleep, mask):
    """Minutes over the epochs in mask; None fields when no 10 min sleep run."""
    sl = sleep & mask
    r = runs(sl)
    d = dict(tst=sl.sum() * STEP, onset=np.nan, final_wake=np.nan, waso=np.nan, eff=np.nan, on_ep=None, off_ep=None)
    if r:
        on, off = r[0][0], r[-1][1]
        seg = slice(on, off)
        d.update(onset=on * STEP, final_wake=off * STEP, waso=(~sleep & mask)[seg].sum() * STEP,
                 eff=100 * sl[seg].sum() / max(off - on, 1), on_ep=on, off_ep=off)
    return d


def row(cohort, setname, subject, per, model, ref, pred, mask, ref_full, valid=None, extra=None):
    A, B = night(ref, mask), night(pred, mask)
    keys = {"tst": "tst", "onset": "onset", "final_wake": "final_wake", "waso": "waso", "eff": "eff"}
    r = dict(cohort=cohort, set=setname, subject=subject, person=per, model=model, n_epochs=int(mask.sum()),
             **{f"ref_{k}": A[k] for k in keys}, **{f"err_{k}": B[k] - A[k] for k in keys})
    if valid is not None:  # shared set: only the parts of the night the shared epochs contain
        ok_on, ok_off = valid
        for k, good in (("onset", ok_on), ("final_wake", ok_off), ("waso", ok_on and ok_off), ("eff", ok_on and ok_off)):
            if not good:
                r[f"ref_{k}"] = r[f"err_{k}"] = np.nan
    r.update(extra or {})
    return r


def process(cohort):
    tab = table(cohort)
    bz = bidoze(cohort, tab)
    preds, covered = tree_preds(cohort, tab)
    log(cohort, "nights", tab.subject.nunique(), "rows", len(tab), "tree-scored rows", int(covered.sum()))
    rows, counts, crop = [], dict(cohort=cohort, nights_table=tab.subject.nunique(), nights_tree_model=0, nights_with_bidoze=0,
                                  complete=0, shared=0, shared_short=0), []
    for subject, g in tab.groupby("subject"):
        ix = g.index.to_numpy()
        if not covered[ix].all():
            continue
        counts["nights_tree_model"] += 1
        y, ctx, ep = g.y4.to_numpy(), g.ctx.to_numpy(), g.epoch.to_numpy()
        if cohort == "wearanize":
            span = np.flatnonzero(g.in_psg.to_numpy())
            lo, hi = span[0], span[-1] + 1
        else:
            lo, hi = 0, len(g)
        sl = slice(lo, hi)
        y, ctx, ep, ix = y[sl], ctx[sl], ep[sl], ix[sl]
        lab = y >= 0
        n_lab = int(lab.sum())
        ref = y > 0
        # tree-excluded PSG minutes: labelled epochs without complete context, split into first / last part of the span
        first = np.arange(len(y)) < cv.BACK_EPOCHS
        last = np.arange(len(y)) >= len(y) - cv.FWD_EPOCHS
        crop.append(dict(cohort=cohort, subject=subject, person=g.person.iloc[0], psg_span_min=len(y) * STEP, labelled_min=n_lab * STEP,
                         excl_first_min=(lab & ~ctx & first).sum() * STEP, excl_last_min=(lab & ~ctx & last).sum() * STEP,
                         excl_other_min=(lab & ~ctx & ~first & ~last).sum() * STEP, scored_min=(lab & ctx).sum() * STEP,
                         ref_sleep_min=(ref & lab).sum() * STEP, ref_sleep_excl_min=(ref & lab & ~ctx).sum() * STEP,
                         ref_onset_excluded=bool(np.argmax(ref & lab) < cv.BACK_EPOCHS) if (ref & lab).any() else False))
        b = bz.get(subject)
        if b is None:
            continue
        counts["nights_with_bidoze"] += 1
        bb = b.reindex(ep)
        avail = bb.call.notna().to_numpy()
        call = bb.call.fillna(-1).to_numpy().astype(int)
        gap = call == 4
        arg = (call > 0) & (call < 4)
        bsleep = {"bidoze_argmax": arg}
        bavail = {"bidoze_argmax": avail}
        if "s25h" in bb:                    # CV cohorts: out-of-fold summed 0.5 (25 h and 8 h)
            for name, c in (("bidoze_sum05", "25h"), ("bidoze_sum05_8h", "8h")):
                av = bb[f"s{c}"].notna().to_numpy(); gp = bb[f"g{c}"].fillna(1).to_numpy() > 0
                bsleep[name] = (bb[f"s{c}"].fillna(0).to_numpy() > 0) & ~gp; bavail[name] = av; gap = gap | (av & gp)
        elif "p_wake" in bb:
            bsleep["bidoze_sum05"] = ((bb.p_light + bb.p_deep + bb.p_rem) >= bb.p_wake).to_numpy() & ~gap; bavail["bidoze_sum05"] = avail
        avail_all = np.logical_and.reduce(list(bavail.values()))
        tseeds = {m: p[:, ix] > 0 for m, p in preds.items()}
        # ---- complete-night set
        if ctx[lab].all() and n_lab >= MIN_SHARED:
            counts["complete"] += 1
            for m, ps in tseeds.items():
                errs = [row(cohort, "complete", subject, g.person.iloc[0], m, ref, p, lab, None) for p in ps]
                rows.append(pd.DataFrame(errs).drop(columns=["subject", "person", "cohort", "set", "model"]).mean().to_frame().T.assign(
                    cohort=cohort, set="complete", subject=subject, person=g.person.iloc[0], model=m))
            for m, s in bsleep.items():
                rows.append(pd.DataFrame([row(cohort, "complete", subject, g.person.iloc[0], m, ref, s, lab & bavail[m], None)]))
        # ---- shared-epoch set
        mask = lab & ctx & avail_all & ~gap
        if mask.sum() < MIN_SHARED:
            counts["shared_short"] += 1
            continue
        counts["shared"] += 1
        full = night(ref, lab)
        idx = np.flatnonzero(mask)
        valid = (full["on_ep"] is not None and full["on_ep"] >= idx[0], full["off_ep"] is not None and full["off_ep"] <= idx[-1] + 1)
        for m, ps in tseeds.items():
            errs = [row(cohort, "shared", subject, g.person.iloc[0], m, ref, p, mask, full, valid) for p in ps]
            rows.append(pd.DataFrame(errs).drop(columns=["subject", "person", "cohort", "set", "model"]).mean().to_frame().T.assign(
                cohort=cohort, set="shared", subject=subject, person=g.person.iloc[0], model=m))
        for m, s in bsleep.items():
            rows.append(pd.DataFrame([row(cohort, "shared", subject, g.person.iloc[0], m, ref, s, mask, full, valid)]))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), counts, pd.DataFrame(crop)


# ------------------------------------------------------------------------------------------------ statistics
def boot_ci(d, people, rng, draws):
    """Participant bootstrap of the mean difference: resample participants, pool their nights."""
    up, inv = np.unique(people, return_inverse=True)
    sums, cnt = np.bincount(inv, d, len(up)), np.bincount(inv, minlength=len(up))
    idx = rng.integers(0, len(up), (draws, len(up)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return np.percentile(m, [2.5, 97.5])


def wilcoxon(d):
    d = d[~np.isnan(d)]
    return stats.wilcoxon(d).pvalue if len(d) >= 6 and (d != 0).any() else np.nan


def summarise(N):
    rows, paired = [], []
    rng = np.random.default_rng(a.seed)
    for (cohort, setname), g in N.groupby(["cohort", "set"]):
        for model, h in g.groupby("model"):
            r = dict(cohort=cohort, set=setname, model=model)
            for m in METRICS:
                e = h[f"err_{m}"].dropna()
                pp = h.loc[e.index, "person"].nunique()
                sd = e.std() if len(e) > 1 else np.nan
                r.update({f"{m}_n": len(e), f"{m}_participants": pp, f"{m}_ref_mean": h.loc[e.index, f"ref_{m}"].mean(),
                          f"{m}_bias": e.mean(), f"{m}_mae": e.abs().mean(), f"{m}_sd": sd,
                          f"{m}_loa_lo": e.mean() - 1.96 * sd, f"{m}_loa_hi": e.mean() + 1.96 * sd})
            rows.append(r)
        for tm in sorted(set(g.model) & set(TREES)):
            for bm in sorted(set(g.model) & set(BZ)):
                t = g[g.model == tm].set_index("subject")
                b = g[g.model == bm].set_index("subject")
                both = t.index.intersection(b.index)
                for m in METRICS:
                    et, eb = t.loc[both, f"err_{m}"], b.loc[both, f"err_{m}"]
                    ok = et.notna() & eb.notna()
                    et, eb, ppl = et[ok].to_numpy(), eb[ok].to_numpy(), t.loc[both[ok.to_numpy()], "person"].to_numpy()
                    p = dict(cohort=cohort, set=setname, tree=tm, bidoze=bm, metric=m, nights=len(et), participants=len(set(ppl)))
                    if len(et):
                        for name, d in (("bias", et - eb), ("abs_err", np.abs(et) - np.abs(eb))):
                            lo, hi = boot_ci(d, ppl, rng, a.draws)
                            p.update({f"{name}_diff_tree_minus_bidoze": d.mean(), f"{name}_ci_lo": lo, f"{name}_ci_hi": hi,
                                      f"{name}_wilcoxon_p": wilcoxon(d)})
                    paired.append(p)
    return pd.DataFrame(rows), pd.DataFrame(paired)


if __name__ == "__main__":
    parts, counts, crop = [], [], []
    for cohort in a.cohorts.split(","):
        n, c, x = process(cohort)
        parts.append(n); counts.append(c); crop.append(x)
    N = pd.concat(parts, ignore_index=True)
    N.to_csv(OUT / "per_night.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "night_counts.csv", index=False)
    crop = pd.concat(crop, ignore_index=True)
    crop.to_csv(OUT / "crop_exclusion_per_night.csv", index=False)
    cs = crop.groupby("cohort").agg(nights=("subject", "size"), participants=("person", "nunique"), psg_min=("labelled_min", "mean"),
                                    excl_first_min=("excl_first_min", "mean"), excl_last_min=("excl_last_min", "mean"),
                                    excl_total_min=("excl_first_min", lambda s: 0.0), scored_min=("scored_min", "mean"),
                                    ref_sleep_min=("ref_sleep_min", "mean"), ref_sleep_excl_min=("ref_sleep_excl_min", "mean"),
                                    nights_onset_excluded=("ref_onset_excluded", "sum"))
    cs["excl_total_min"] = crop.assign(t=crop.excl_first_min + crop.excl_last_min + crop.excl_other_min).groupby("cohort").t.mean()
    cs["excl_share_pct"] = 100 * cs.excl_total_min / cs.psg_min
    cs.round(1).to_csv(OUT / "crop_exclusion_summary.csv")
    S_, P_ = summarise(N)
    S_.to_csv(OUT / "summary.csv", index=False)
    P_.to_csv(OUT / "paired_vs_bidoze.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(pd.DataFrame(counts).to_string()); print(cs.round(1).to_string())
    print(S_[["cohort", "set", "model", "tst_n", "tst_participants", "tst_bias", "tst_mae", "onset_n", "onset_bias", "onset_mae",
              "final_wake_bias", "final_wake_mae", "waso_n", "waso_bias", "waso_mae", "eff_bias", "eff_mae"]].round(1).to_string())
