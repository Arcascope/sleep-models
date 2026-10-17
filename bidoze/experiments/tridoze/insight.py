"""What the members of each 3-member ensemble learn, where they disagree, and whether confidence says who is right.

Test subjects are the draws' held-out adults (SleepAccel, SleepAccel-Clinical). A member's call on
an epoch is the argmax of its 5-way output (wake, light, deep, REM, gap), with no stage divisor:
this describes the members' own outputs, not the release decoding. Only PSG-scored epochs count.

Movement level is the epoch's mean absolute change in arm angle (angle-z, 6 samples per 30 s
epoch, degrees per 5 s) from the feature cache; the levels are the tertiles of the pooled test
epochs of each cohort, and their cut points are written to ``movement_levels.csv``.

Writes to ``<run>/analysis/insight/``:
- confusion.csv     ensemble, draw, cohort, member (or "mean"), psg, call, epochs;
- recall.csv        per member and stage: recall (share of the stage's epochs called it), call share;
- disagreement.csv  share of epochs where the 3 members' calls are not all equal, by PSG stage, by
                    transition (within 1 epoch of a PSG stage change, or not) and by movement level,
                    with the mean output's accuracy on those epochs;
- diversity.csv     per PSG stage: mean member log loss − log loss of the mean output (nats per epoch),
                    the part of the ensemble's log loss that averaging removes;
- aligned.csv       operating-point check: each member's 4 stage log-probabilities are shifted by
                    per-stage offsets until its call shares equal the mean output's on the cohort
                    (no labels used); then the share of epochs where members differ, each member's
                    κ 4-stage (argmax, pooled epochs), and the κ of the mean output, before and after;
- confidence.csv    per member: AUROC of the member's top probability for whether its call is right;
                    on disagreement epochs, accuracy of the mean output, of the most confident member,
                    of the majority vote, and the share where at least one member is right.

Usage: python insight.py [--run RUN] [--draws 0 1 2 3 4] [--ensemble label ...]
"""
from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1] / "results/tridoze-2026-10-16"
BASELINE = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results/bidoze-joint-ensemble-2026-10-15")
CACHE_NAME = "12Hz_log_PSD_aligned"
STAGES = ["wake", "light", "deep", "rem"]
CALLS = [*STAGES, "gap"]
COHORTS = {"testSleepAccel": "SleepAccel", "testClinical": "SleepAccel-Clinical"}


def ensembles(run: Path, baseline: Path) -> dict[str, callable]:
    """label -> function(draw) -> the 3 members' prediction directories."""
    def pred(arm_dir: Path, d: int, name: str) -> Path:
        return arm_dir / f"draw{d}/predictions/draw{d}_{name}"

    def t1(arm: str):
        def dirs(d):
            arm_dir = run / ("tridoze_j_split" if arm == "tridoze_j" and d == 0 else arm)
            return [pred(arm_dir, d, f"seed_0_member{m}") for m in range(3)]
        return dirs

    def boosted(d):
        # A draw whose boosted members stalled is rerun as bidoze_g_boost<k>r, with the patience
        # counted from the first eval kappa above 0.3; the rerun replaces both boosted members.
        rerun = all(pred(run / f"bidoze_g_boost{k}r", d, f"seed_{k}").is_dir() for k in (1, 2))
        arms = ["bidoze_g", *(f"bidoze_g_boost{k}{'r' if rerun else ''}" for k in (1, 2))]
        return [pred(run / a, d, f"seed_{s}") for s, a in enumerate(arms)]

    return {
        "independent ×3": lambda d: [pred(baseline / "independent", d, f"seed_{s}") for s in range(3)],
        "BiDoze-G ×3": lambda d: [pred(run / a, d, f"seed_{s}") for s, a in
                                  enumerate(["bidoze_g", "bidoze_g_s1", "bidoze_g_s2"])],
        "T1 β 0.5": t1("tridoze_j"),
        "T1 β 0.25": t1("tridoze_j_b025"),
        "Bagged G ×3": lambda d: [pred(run / f"bidoze_g_bag{s}", d, f"seed_{s}") for s in range(3)],
        "Boosted G ×3": boosted,
    }


def load_proba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_csv(path, dtype={"psg": str})
    gap = d["gap_head"].to_numpy(float)
    proba = np.column_stack([d[STAGES].to_numpy(float) * (1 - gap)[:, None], gap])
    y = d["psg"].map({s: i for i, s in enumerate(STAGES)}).fillna(-1).astype(int).to_numpy()
    return y, proba


def angle_change(cache: Path, dataset: str, sid: str, n: int) -> np.ndarray:
    z = np.load(cache / f"{dataset}_subject_{sid}_features.npz")
    a = z["anglez_5s"][0][:n]
    out = np.full(n, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # epochs with no angle samples stay NaN
        out[: a.shape[0]] = np.nanmean(np.abs(np.diff(a, axis=1)), axis=1)
    return out


def transition_mask(y: np.ndarray) -> np.ndarray:
    change = np.zeros(y.shape, dtype=bool)
    scored = (y >= 0) & (y < 4)
    step = scored[1:] & scored[:-1] & (y[1:] != y[:-1])
    change[1:] |= step
    change[:-1] |= step
    return change


def align_offsets(logp: np.ndarray, target: np.ndarray, steps: int = 300) -> np.ndarray:
    """Per-stage offsets b so that argmax(logp + b) calls each stage on the ``target`` share of epochs."""
    b = np.zeros(logp.shape[1])
    for _ in range(steps):
        share = np.bincount((logp + b).argmax(axis=1), minlength=logp.shape[1]) / len(logp)
        b += 0.5 * (np.log(np.clip(target, 1e-4, None)) - np.log(np.clip(share, 1e-4, None)))
        b -= b.mean()
    return b


def collect(run: Path, baseline: Path, label: str, dirs_for, d: int, draws_json: dict):
    """Per cohort: (y, member probabilities (3, N, 5), transition mask, angle change), PSG-scored epochs only."""
    dirs = dirs_for(d)
    if not all(p.is_dir() for p in dirs):
        return None
    cache = baseline / f"setup/cache/draw{d}" / CACHE_NAME
    out = {}
    for dataset, cohort in COHORTS.items():
        ys, ps, ts, ms = [], [], [], []
        for sid in draws_json[f"draw{d}"][dataset]:
            loaded = [load_proba(p / f"{dataset}_{sid}.csv") for p in dirs]
            y = loaded[0][0]
            scored = (y >= 0) & (y < 4)
            ys.append(y[scored])
            ps.append(np.stack([q for _, q in loaded])[:, scored])
            ts.append(transition_mask(y)[scored])
            ms.append(angle_change(cache, dataset, sid, len(y))[scored])
        out[cohort] = (np.concatenate(ys), np.concatenate(ps, axis=1), np.concatenate(ts), np.concatenate(ms))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--baseline-run", type=Path, default=BASELINE)
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ensemble", nargs="+", default=None)
    args = ap.parse_args()
    out = args.run / "analysis/insight"
    out.mkdir(parents=True, exist_ok=True)
    draws_json = json.loads((args.baseline_run / "setup/draws.json").read_text())
    defs = ensembles(args.run, args.baseline_run)
    labels = args.ensemble or list(defs)

    data = {}
    for label in labels:
        for d in args.draws:
            got = collect(args.run, args.baseline_run, label, defs[label], d, draws_json)
            if got is None:
                print(f"skip {label} draw {d}: predictions missing")
                continue
            data[(label, d)] = got
    cuts = {}
    for cohort in COHORTS.values():
        per_draw = {d: v[cohort][3] for (label, d), v in data.items()}
        pooled = np.concatenate(list(per_draw.values()))
        cuts[cohort] = np.nanquantile(pooled, [1 / 3, 2 / 3])
    pd.DataFrame([{"cohort": c, "tertile_1_upper_deg_per_5s": v[0], "tertile_2_upper_deg_per_5s": v[1]}
                  for c, v in cuts.items()]).to_csv(out / "movement_levels.csv", index=False)

    conf_rows, recall_rows, dis_rows, div_rows, cfd_rows, aligned_rows = [], [], [], [], [], []
    for (label, d), per_cohort in data.items():
        for cohort, (y, P, trans, move) in per_cohort.items():
            mean = P.mean(axis=0)
            calls = P.argmax(axis=2)
            mean_call = mean.argmax(axis=1)
            key = {"ensemble": label, "draw": d, "cohort": cohort}
            for name, c in [*((str(m), calls[m]) for m in range(3)), ("mean", mean_call)]:
                cm = np.zeros((4, 5), dtype=int)
                np.add.at(cm, (y, c), 1)
                for i in range(4):
                    for j in range(5):
                        conf_rows.append({**key, "member": name, "psg": STAGES[i], "call": CALLS[j], "epochs": int(cm[i, j])})
                for i, s in enumerate(STAGES):
                    recall_rows.append({**key, "member": name, "stage": s,
                                        "recall": float(np.mean(c[y == i] == i)) if np.any(y == i) else np.nan,
                                        "call_share": float(np.mean(c == i))})
            from sklearn.metrics import cohen_kappa_score

            logp4 = np.log(np.clip(P[:, :, :4], 1e-9, None))
            target = np.bincount(mean[:, :4].argmax(axis=1), minlength=4) / len(y)
            shifted = np.stack([logp4[m] + align_offsets(logp4[m], target) for m in range(3)])
            a_calls = shifted.argmax(axis=2)
            a_mean = np.exp(shifted - np.log(np.exp(shifted).sum(axis=2, keepdims=True))).mean(axis=0).argmax(axis=1)
            for when, c3, cm in [("as trained", P[:, :, :4].argmax(axis=2), mean[:, :4].argmax(axis=1)),
                                 ("aligned", a_calls, a_mean)]:
                aligned_rows.append({**key, "members": when,
                                     "disagree_share": float(np.mean(~((c3[0] == c3[1]) & (c3[1] == c3[2])))),
                                     **{f"kappa_member{m}": float(cohen_kappa_score(y, c3[m])) for m in range(3)},
                                     "kappa_mean_output": float(cohen_kappa_score(y, cm))})
            disagree = ~((calls[0] == calls[1]) & (calls[1] == calls[2]))
            pair = np.mean([np.mean(calls[a] != calls[b]) for a, b in combinations(range(3), 2)])
            level = np.digitize(move, cuts[cohort])
            groups = {"all": np.ones_like(y, bool)}
            groups |= {f"psg {s}": y == i for i, s in enumerate(STAGES)}
            groups |= {"near transition": trans, "away from transition": ~trans}
            groups |= {f"movement {n}": (level == i) & np.isfinite(move) for i, n in enumerate(["low", "mid", "high"])}
            for g, mask in groups.items():
                if not mask.any():
                    continue
                dis_rows.append({**key, "group": g, "epochs": int(mask.sum()), "epoch_share": float(mask.mean()),
                                 "disagree_share": float(disagree[mask].mean()),
                                 "pairwise_disagree": float(pair) if g == "all" else np.nan,
                                 "mean_accuracy": float(np.mean(mean_call[mask] == y[mask])),
                                 "mean_accuracy_when_disagree": float(np.mean(mean_call[mask & disagree] == y[mask & disagree]))
                                 if (mask & disagree).any() else np.nan})
            eps = 1e-9
            member_nll = -np.log(np.clip(P[:, np.arange(len(y)), y], eps, None))
            mean_nll = -np.log(np.clip(mean[np.arange(len(y)), y], eps, None))
            for i, s in enumerate(["all", *STAGES]):
                mask = np.ones_like(y, bool) if s == "all" else y == i - 1
                div_rows.append({**key, "stage": s, "member_nll": float(member_nll[:, mask].mean()),
                                 "mean_nll": float(mean_nll[mask].mean()),
                                 "diversity_nats": float(member_nll[:, mask].mean() - mean_nll[mask].mean())})
            top = P.max(axis=2)
            for m in range(3):
                right = calls[m] == y
                cfd_rows.append({**key, "member": m, "accuracy": float(right.mean()),
                                 "auroc_top_prob_vs_right": float(roc_auc_score(right, top[m])) if 0 < right.mean() < 1 else np.nan})
            if disagree.any():
                Pd, yd = P[:, disagree], y[disagree]
                confident = calls[top[:, disagree].argmax(axis=0), np.flatnonzero(disagree)]
                c3 = calls[:, disagree]
                majority = np.where(c3[0] == c3[1], c3[0], np.where(c3[2] == c3[0], c3[0],
                                    np.where(c3[1] == c3[2], c3[1], mean_call[disagree])))
                cfd_rows.append({**key, "member": "disagreement epochs", "epochs": int(disagree.sum()),
                                 "mean_output_accuracy": float(np.mean(Pd.mean(axis=0).argmax(axis=1) == yd)),
                                 "most_confident_accuracy": float(np.mean(confident == yd)),
                                 "majority_accuracy": float(np.mean(majority == yd)),
                                 "any_member_right": float(np.mean((c3 == yd).any(axis=0)))})
    for name, rows in [("confusion", conf_rows), ("recall", recall_rows), ("disagreement", dis_rows),
                       ("diversity", div_rows), ("confidence", cfd_rows), ("aligned", aligned_rows)]:
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)

    rec = pd.DataFrame(recall_rows)
    members = rec[rec.member != "mean"]
    spread = (members.groupby(["ensemble", "draw", "cohort", "stage"]).recall.agg(lambda r: r.max() - r.min())
              .groupby(["ensemble", "cohort", "stage"]).mean().unstack("stage")[STAGES])
    print("Spread of member recall (max − min over the 3 members), mean over draws:\n", spread.round(3))
    dis = pd.DataFrame(dis_rows)
    print("\nShare of epochs where members disagree:\n",
          dis.groupby(["ensemble", "cohort", "group"]).disagree_share.mean().unstack("group").round(3).to_string())
    div = pd.DataFrame(div_rows)
    print("\nLog loss removed by averaging (nats/epoch):\n",
          div.groupby(["ensemble", "cohort", "stage"]).diversity_nats.mean().unstack("stage").round(3).to_string())
    cfd = pd.DataFrame(cfd_rows)
    dd = cfd[cfd.member == "disagreement epochs"]
    print("\nOn disagreement epochs (accuracy, mean over draws):\n",
          dd.groupby(["ensemble", "cohort"])[["mean_output_accuracy", "most_confident_accuracy", "majority_accuracy",
                                              "any_member_right"]].mean().round(3).to_string())
    al = pd.DataFrame(aligned_rows)
    al["kappa_members_mean"] = al[[f"kappa_member{m}" for m in range(3)]].mean(axis=1)
    al["ensemble_gain"] = al.kappa_mean_output - al.kappa_members_mean
    print("\nOperating-point check (members shifted to the mean output's call shares):\n",
          al.groupby(["ensemble", "cohort", "members"])[["disagree_share", "kappa_members_mean", "kappa_mean_output",
                                                         "ensemble_gain"]].mean().round(3).to_string())
    au = cfd[cfd.member.isin([0, 1, 2])]
    print("\nAUROC of a member's top probability for its own correctness:\n",
          au.groupby(["ensemble", "cohort"]).auroc_top_prob_vs_right.mean().round(3).to_string())


if __name__ == "__main__":
    main()
