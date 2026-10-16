#!/usr/bin/env python3
"""Contrasts, gate and checks for the distant-context transplant (research log 2026-10-16 §3).

Reads ``views/<model>/seed<k>.npz`` and the manifest; writes into ``analysis/``:

* ``per_person.csv``: per model (each seed and ``ens``, the three-seed mean of the five
  probabilities), dose, PSG class (sleep or wake epochs of the targets), stratum (all, in or
  outside GGIR's window) and person: mean log odds of sleep given not gap under the
  unmodified input and each carrier, the contrasts, and the share called sleep by five-way
  argmax.
* ``summary.csv``: the same, averaged over people, with the count of people below zero.
* ``arm_vs_control.csv``: per seed and ``ens``, the arm − control difference of the
  wake − sleep contrast, per person, for foundry − random and foundry-circular − circular.
* ``checks.csv``: identity-splice and repeat-inference differences, agreement with the saved
  prediction CSVs, and clipping counts.
* ``gate.json``: the pre-registered gate at the largest dose.

    python analyze.py --run <run dir> --labels <release seed0 CSVs> --cache <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import SLEEP_STAGES, TARGET, conditional_sleep, load_recording, log_odds

NAMES = [f"sen{i:03d}" for i in range(3, 16)]
MODELS = ["release", "random", "foundry", "circular", "foundry-circular"]
PAIRS = [("foundry", "random"), ("foundry-circular", "circular")]
EPS = 1e-6


def load(run: Path):
    out = {}
    for m in MODELS:
        seeds = sorted((run / "views" / m).glob("seed*.npz"))
        if len(seeds) < 3:
            continue
        zs = [np.load(p) for p in seeds]
        out[m] = {f"s{k}": {key: z[key] for key in ("targets", "original", "original_batched", "view_id")}
                  for k, z in enumerate(zs)}
        out[m]["ens"] = {key: np.mean([z[key] for z in zs], axis=0) for key in ("targets", "original", "original_batched")}
        out[m]["ens"]["view_id"] = zs[0]["view_id"]
    return out


def saved_csv(run: Path, model: str, k: int, name: str) -> Path:
    """The saved SENSE prediction CSV of one model seed in ``free-living-1``."""
    base = run.parent / "free-living-1"
    if model == "release":
        return base / "release" / f"seed{k}" / f"SENSE_{name}.csv"
    return base / model / "final" / "predictions" / f"final_seed_{k}" / f"SENSE_{name}.csv"


def csv_difference(run: Path, model: str, k: int, original: np.ndarray, recs) -> float:
    """Largest absolute difference between the unmodified input's probabilities, scored
    alone, and the saved CSV's (four rescaled stage columns and gap_head, 4 decimals)."""
    worst = 0.0
    for i, name in enumerate(NAMES):
        f = pd.read_csv(saved_csv(run, model, k, name)).iloc[:recs[name].n]
        p = original[i, :len(f)]
        stages = p[:, :4] / p[:, :4].sum(1, keepdims=True)
        worst = max(worst, np.abs(stages - f[["wake", "light", "deep", "rem"]].to_numpy()).max(),
                    np.abs(p[:, 4] - f["gap_head"].to_numpy()).max())
    return float(worst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)

    recs = {n: load_recording(args.cache, args.labels, n) for n in NAMES}
    views = pd.read_csv(args.run / "manifest" / "views.csv", index_col="view_id")
    data = load(args.run)

    # Per-target masks: PSG class and GGIR stratum of each target epoch.
    tgt = views[views.view == "identity"][["target", "recording", "start"]].set_index("target")
    psg = {t: recs[r.recording].psg[r.start:r.start + TARGET] for t, r in tgt.iterrows()}
    spt = {t: recs[r.recording].is_spt[r.start:r.start + TARGET] for t, r in tgt.iterrows()}

    checks, rows = [], []
    for m, by_seed in data.items():
        for s, d in by_seed.items():
            assert np.array_equal(d["view_id"], views.index.to_numpy())
            lo_t = log_odds(conditional_sleep(d["targets"]), EPS)
            call = d["targets"].argmax(-1)
            orig = d["original_batched"]
            lo_o = log_odds(conditional_sleep(orig), EPS)
            # Checks.
            ident = views.view.to_numpy() == "identity"
            ri = views.recording.map(NAMES.index).to_numpy()
            st = views.start.to_numpy()
            diff = [np.abs(lo_t[i] - lo_o[ri[i], st[i]:st[i] + TARGET]).max() for i in np.flatnonzero(ident)]
            rep = np.abs(log_odds(conditional_sleep(d["original"]), EPS) - lo_o)
            p_all = conditional_sleep(d["targets"])
            checks.append({"model": m, "seed": s, "identity_max_abs_log_odds": float(np.max(diff)),
                           "identity_mean_abs_log_odds": float(np.mean(diff)),
                           "repeat_max_abs_log_odds": float(max(rep[i, :recs[n].n].max() for i, n in enumerate(NAMES))),
                           "saved_csv_max_abs_prob": csv_difference(args.run, m, int(s[1:]), d["original"], recs)
                           if s != "ens" else np.nan,
                           "clipped_share": float(((p_all < EPS) | (p_all > 1 - EPS)).mean())})
            # Per view: mean log odds and share called sleep, by PSG class and stratum.
            for vid, v in views.iterrows():
                lab, inside = psg[v.target], spt[v.target]
                o_lo = lo_o[NAMES.index(v.recording), v.start:v.start + TARGET]
                o_call = orig[NAMES.index(v.recording), v.start:v.start + TARGET].argmax(-1)
                for cls, mask in (("sleep", np.isin(lab, SLEEP_STAGES)), ("wake", lab == "wake")):
                    for stratum, smask in (("all", mask), ("in_spt", mask & inside), ("out_spt", mask & ~inside)):
                        if not smask.any():
                            continue
                        rows.append({"model": m, "seed": s, "target": v.target, "recording": v.recording,
                                     "dose": v.dose, "view": v.view, "draw": v.draw, "cls": cls, "stratum": stratum,
                                     "lo": lo_t[vid][smask].mean(), "lo_orig": o_lo[smask].mean(),
                                     "called_sleep": np.isin(call[vid][smask], [1, 2, 3]).mean(),
                                     "called_sleep_orig": np.isin(o_call[smask], [1, 2, 3]).mean()})
        print(f"{m}: views summarized", flush=True)
    checks = pd.DataFrame(checks)
    checks.to_csv(out / "checks.csv", index=False, float_format="%.5f")
    df = pd.DataFrame(rows)

    # Per target: mean over draws of each carrier; contrasts; then mean over targets per person.
    key = ["model", "seed", "dose", "cls", "stratum", "recording", "target"]
    car = df[df.view.isin(["wake", "sleep"])].pivot_table(index=key + ["draw"], columns="view",
                                                          values=["lo", "called_sleep"]).reset_index()
    car.columns = ["_".join(c).strip("_") for c in car.columns]
    orig = df[df.view == "identity"].groupby(["model", "seed", "cls", "stratum", "recording", "target"])[
        ["lo_orig", "called_sleep_orig", "lo"]].first().rename(columns={"lo": "lo_identity"}).reset_index()
    car["contrast"] = car.lo_wake - car.lo_sleep
    per_target = car.groupby(key)[["lo_wake", "lo_sleep", "contrast", "called_sleep_wake", "called_sleep_sleep"]].mean().reset_index()
    per_target = per_target.merge(orig, on=["model", "seed", "cls", "stratum", "recording", "target"])
    per_target["wake_vs_orig"] = per_target.lo_wake - per_target.lo_orig
    per_target["sleep_vs_orig"] = per_target.lo_sleep - per_target.lo_orig
    cols = ["lo_orig", "lo_wake", "lo_sleep", "contrast", "wake_vs_orig", "sleep_vs_orig",
            "called_sleep_orig", "called_sleep_wake", "called_sleep_sleep"]
    per_person = per_target.groupby(key[:-1])[cols].mean().reset_index()
    per_person["targets"] = per_target.groupby(key[:-1]).size().to_numpy()
    per_person.to_csv(out / "per_person.csv", index=False, float_format="%.4f")
    summ = per_person.groupby(key[:-2])[cols].mean()
    summ["people"] = per_person.groupby(key[:-2]).size()
    for c in ("contrast", "wake_vs_orig", "sleep_vs_orig"):
        summ[f"{c}_neg"] = per_person.groupby(key[:-2])[c].apply(lambda x: int((x < 0).sum()))
    summ = summ.reset_index()
    summ.to_csv(out / "summary.csv", index=False, float_format="%.4f")

    # Arm − control, per person.
    pp = per_person.set_index(["model", "seed", "dose", "cls", "stratum", "recording"])
    avc = []
    for arm, ctl in PAIRS:
        if arm not in data or ctl not in data:
            continue
        a, c = pp.loc[arm], pp.loc[ctl]
        dd = (a[["contrast", "wake_vs_orig", "sleep_vs_orig"]] - c[["contrast", "wake_vs_orig", "sleep_vs_orig"]]).reset_index()
        dd.insert(0, "pair", f"{arm} - {ctl}")
        avc.append(dd)
    avc = pd.concat(avc)
    avc.to_csv(out / "arm_vs_control.csv", index=False, float_format="%.4f")
    avc_s = avc.groupby(["pair", "seed", "dose", "cls", "stratum"]).agg(
        mean=("contrast", "mean"), neg=("contrast", lambda x: int((x < 0).sum())), n=("contrast", "size"),
        p=("contrast", lambda x: wilcoxon(x).pvalue if len(x) > 5 else np.nan),
        wake_vs_orig=("wake_vs_orig", "mean"), sleep_vs_orig=("sleep_vs_orig", "mean"),
        sleep_vs_orig_neg=("sleep_vs_orig", lambda x: int((x < 0).sum()))).reset_index()
    avc_s.to_csv(out / "arm_vs_control_summary.csv", index=False, float_format="%.4f")

    # Gate at the largest dose.
    sel = lambda frame, **kw: frame[np.logical_and.reduce([frame[k] == v for k, v in kw.items()])]
    fr = sel(avc_s, pair="foundry - random", dose="all", cls="sleep", stratum="all")
    ens = fr[fr.seed == "ens"].iloc[0]
    seeds_neg = int((fr[fr.seed != "ens"]["mean"] < 0).sum())
    f_wake = sel(summ, model="foundry", seed="ens", dose="all", cls="sleep", stratum="all").iloc[0]
    joins = pd.read_csv(args.run / "manifest" / "joins_pooled.csv")
    jw, js = (joins[(joins.dose == "all") & (joins.carrier == c)].iloc[0] for c in ("wake", "sleep"))
    ident_ok = bool((checks.identity_max_abs_log_odds <= checks.repeat_max_abs_log_odds).all())
    gate = {
        "foundry_minus_random_mean": float(ens["mean"]), "criterion_mean": bool(ens["mean"] <= -0.25),
        "people_negative": int(ens.neg), "people": int(ens.n), "criterion_people": bool(ens.neg >= 9),
        "wilcoxon_p": float(ens.p),
        "seeds_negative": seeds_neg, "seed_means": fr[fr.seed != "ens"]["mean"].round(4).tolist(),
        "criterion_seeds": bool(seeds_neg >= 2),
        "foundry_wake_vs_orig": float(f_wake.wake_vs_orig), "criterion_wake_suppression": bool(f_wake.wake_vs_orig < 0),
        "identity_within_repeat": ident_ok,
        "join_iqr_wake": [float(jw.z_q1), float(jw.z_q3)], "join_iqr_sleep": [float(js.z_q1), float(js.z_q3)],
        "criterion_joins_overlap": bool(jw.z_q1 <= js.z_q3 and js.z_q1 <= jw.z_q3),
    }
    gate["pass"] = all(v for k, v in gate.items() if k.startswith("criterion_")) and ident_ok
    (out / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(checks.round(5).to_string(index=False))
        print(sel(summ, seed="ens", stratum="all").round(3).to_string(index=False))
        print(sel(avc_s, stratum="all").round(4).to_string(index=False))
        print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
