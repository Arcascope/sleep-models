"""Fit a gated fusion on each fit's eval share of frozen strands; score it on that draw's test recordings.

Families (``--family``):
- ``coop``:  cooperating-arms fits (``<coop>/<design>/draw<d>_seed0``), designs of October 19;
- ``g3``:    the three independent BiDoze-G seeds of draw d (October 16 §8), ``--design bidoze_g3``;
- ``split``: a split-batch ensemble of October 17 (``<ensemble-size>/<design>/draw0``).

Per fit this writes ``<out>/<family>/<design>/draw<d>.json`` (gate steps, parameter count, weights by
PSG stage) and ``.csv`` (per-recording metrics for the mean of probabilities, three output-level
combiners and the gate). Eval share = the recordings the fit early-stopped on; test = the draw's
held-out participants, never used to fit or select anything here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "bidoze/experiments/cooperating_arms"))
sys.path.insert(0, str(ROOT / "bidoze/experiments/tridoze"))

import evaluate as scoring  # noqa: E402
import gate as G  # noqa: E402
import strands as ST  # noqa: E402
from arcasleep.data.cache import discover_cached_subjects, participant_id  # noqa: E402

MAIN = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results")
TRI = Path("/home/eric/Engineering/Work/autopisces/sleep-models-tridoze/bidoze/results")
COOP = MAIN / "cooperating-arms-2026-10-19"
ENS = TRI / "ensemble-size-2026-10-17"
G3 = TRI / "tridoze-2026-10-16"
COHORTS = scoring.COHORTS
KEEP = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity"]
EPS = 1e-9
ADULT = ("trainSleepAccel", "trainClinical")
GRID = np.arange(-2.0, 2.01, 0.2)
METHODS = ["mean", "logit mean", "median", "vote", "gate"]


def fast_kappa(y, calls):
    ok = (y >= 0) & (y < 4) & (calls < 4)
    c = np.zeros((4, 4))
    np.add.at(c, (y[ok], calls[ok]), 1)
    n = c.sum()
    po = np.trace(c) / n
    pe = (c.sum(0) * c.sum(1)).sum() / n**2
    return (po - pe) / (1 - pe)


def refit_divisors(adult_p, adult_y, divisor, pi_adult):
    """Two refits of the stage divisor on one method's own outputs over the adult eval recordings.

    ``prior``: light/deep/REM called shares among called-sleep epochs match the adult reference shares
    (the target the draw's divisor is built for). ``kappa``: pooled 4-stage κ on those recordings is highest.
    Both search deep and REM offsets (log scale, light fixed) around the draw's divisor on a grid."""
    p = np.concatenate(adult_p)
    y = np.concatenate(adult_y)
    sleep = (p.argmax(1) >= 1) & (p.argmax(1) <= 3)
    stages = p[sleep, 1:4]
    ys = y[sleep]
    target = pi_adult[1:4] / pi_adult[1:4].sum()
    base = np.log(divisor[1:4])
    best = {"prior": (np.inf, None), "kappa": (-np.inf, None)}
    for a in GRID:
        for b in GRID:
            d = base + np.array([0.0, a, b])
            call = (stages / np.exp(d)).argmax(1) + 1
            share = np.bincount(call, minlength=4)[1:4] / max(len(call), 1)
            err = float(np.abs(share - target).sum())
            if err < best["prior"][0]:
                best["prior"] = (err, d)
            full = np.full(len(p), 4)
            full[sleep] = call
            full[~sleep] = p.argmax(1)[~sleep]
            k = fast_kappa(y, full) if (y >= 0).any() else -1
            if k > best["kappa"][0]:
                best["kappa"] = (k, d)
    return {name: np.concatenate([[1.0], np.exp(v[1])]) for name, v in best.items()}


def normalize(p):
    return p / np.clip(p.sum(axis=-1, keepdims=True), EPS, None)


def output_level(P, how):
    """``P`` (S, n, 5) -> (n, 5), as ``tridoze/combiners.py`` combines."""
    if how == "mean":
        return P.mean(axis=0)
    if how == "logit mean":
        z = np.log(np.clip(P, EPS, None)).mean(axis=0)
        return normalize(np.exp(z - z.max(axis=-1, keepdims=True)))
    if how == "median":
        return normalize(np.median(P, axis=0))
    if how == "vote":
        return np.eye(5)[P.argmax(axis=-1)].mean(axis=0) + 1e-3 * P.mean(axis=0)
    raise ValueError(how)


def make_source(family, design, d):
    split = json.loads((COOP / "final7" / f"draw{d}_seed0" / "split.json").read_text())
    divisor = np.load(COOP / "final7" / f"draw{d}_seed0" / "stage_divisor.npy")
    if family == "coop":
        strands = ST.CoopStrands(COOP / design / f"draw{d}_seed0")
        check = None
    elif family == "g3":
        arms = ["bidoze_g", "bidoze_g_s1", "bidoze_g_s2"]
        paths = [G3 / a / f"draw{d}/model_outputs/model_draw{d}_seed_{i}.pisces" for i, a in enumerate(arms)]
        assert all(p.exists() for p in paths), [str(p) for p in paths if not p.exists()]
        strands = ST.PiscesStrands(paths)
        check = [G3 / a / f"draw{d}/predictions/draw{d}_seed_{i}" for i, a in enumerate(arms)]
    elif family == "split":
        k = int(design.split("split")[1].split("_")[0])
        paths = [ENS / design / f"draw{d}/model_outputs/model_draw{d}_seed_{s}_member{m}.pisces"
                 for s in [1 if design.endswith("_s1") else 0] for m in range(k)]
        assert all(p.exists() for p in paths), [str(p) for p in paths if not p.exists()]
        strands = ST.PiscesStrands(paths)
        s = 1 if design.endswith("_s1") else 0
        check = [ENS / design / f"draw{d}/predictions/draw{d}_seed_{s}_member{m}" for m in range(k)]
    else:
        raise ValueError(family)
    return split, divisor, strands, check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=["coop", "g3", "split"])
    ap.add_argument("--design", required=True)
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    logger = scoring.MetricsLogger(scoring.load_metrics_config(ROOT / "bidoze/config/metrics.json")[0])
    for d in a.draws:
        out = a.out / a.family / a.design / f"draw{d}"
        if out.with_suffix(".json").exists():
            print("done already", out, flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        split, divisor, strands, check = make_source(a.family, a.design, d)
        cache = ENS / f"setup/cache/draw{d}" / ST.CACHE
        found = discover_cached_subjects(["weaverDraw", "trainSleepAccel", "trainClinical", "testSleepAccel", "testClinical"], cache)
        paths = {(s.ref.dataset_name, s.ref.subject_id): s.path for ds in found.values() for s in ds}
        ev_refs = [(r["dataset"], r["subject_id"]) for r in split["validation"]]
        te_refs = [(r["dataset"], r["subject_id"]) for r in split["test"]]
        ev = [ST.run_recording(strands, paths[r]) for r in ev_refs]
        te = [ST.run_recording(strands, paths[r]) for r in te_refs]
        if check is not None:
            ds, sid = te_refs[0]
            for m, cdir in enumerate(check):
                cy, cp = scoring.load_subject(cdir / f"{ds}_{sid}.csv")
                np.testing.assert_allclose(te[0][1][m], cp[: te[0][1].shape[1]], atol=2e-3)
        eval_data = G.Data(ev, ST.PAD)
        test_data = G.Data(te, ST.PAD, eval_data.mean, eval_data.std)
        groups = [participant_id(sid) for _, sid in ev_refs]
        lr, steps, cv_loss = G.select_steps(eval_data, groups)
        params, _ = G.train(eval_data, steps, lr)
        weights = G.gate_weights(params, test_data)  # (B, T, S)
        n_params = G.parameter_count(params)
        weights_ev = G.gate_weights(params, eval_data)
        bids = scoring.night_counts([x.path for x in discover_cached_subjects(["bidsleep"], scoring.LABELS / scoring.CACHE_NAME)["bidsleep"]])
        pi_adult = np.asarray(bids, float).sum(0)
        pi_adult = pi_adult / pi_adult.sum()
        pi_adult = np.concatenate([[0.0], pi_adult[1:4]]) if len(pi_adult) >= 4 else pi_adult

        def method_p(P, w, how):
            if how == "gate":
                return np.einsum("ns,nsc->nc", w, np.moveaxis(P, 0, 1))
            return output_level(P, how)

        adult_idx = [i for i, (ds, _) in enumerate(ev_refs) if ds in ADULT]
        divisors = {}
        for how in METHODS:
            ap = [method_p(ev[i][1], weights_ev[i, : len(ev[i][0])], how) for i in adult_idx]
            ay = [np.asarray(ev[i][0]) for i in adult_idx]
            divisors[how] = {"orig": divisor, **refit_divisors(ap, ay, divisor, pi_adult)}
        rows, by_stage = [], {}
        for i, ((ds, sid), (y, P, _)) in enumerate(zip(te_refs, te)):
            n = len(y)
            for how in METHODS:
                p = method_p(P, weights[i, :n], how)
                for dec, div in divisors[how].items():
                    calls = scoring.hybrid_calls(p, p, div)
                    m = scoring.subject_metrics(logger, y, p, calls)
                    rec = {}
                    for st in range(4):
                        on = y == st
                        rec[f"hits{st}"], rec[f"total{st}"] = int((calls[on] == st).sum()), int(on.sum())
                    rows.append(dict(design=a.design, draw=d, method=how, decoding=dec, dataset=ds, subject_id=sid,
                                     **{k: float(m[k]) for k in KEEP}, **rec))
            for st in range(4):
                on = y == st
                if on.any():
                    by_stage.setdefault(ds, {}).setdefault(st, []).append(weights[i, :n][on])
        pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
        stage_weights = {ds: {str(st): np.concatenate(v).mean(axis=0).tolist() for st, v in sorted(s.items())}
                         for ds, s in by_stage.items()}
        stage_counts = {ds: {str(st): int(sum(len(x) for x in v)) for st, v in sorted(s.items())} for ds, s in by_stage.items()}
        info = dict(family=a.family, design=a.design, draw=d, strands=strands.n, strand_parameters=strands.parameters, gate_parameters=n_params,
                    chosen_steps=steps, chosen_lr=lr, step_grid=list(G.STEP_GRID), lr_grid=list(G.LRS), cv_loss=cv_loss.tolist(),
                    eval_recordings=len(ev_refs), test_recordings=len(te_refs),
                    weights_by_stage=stage_weights, epochs_by_stage=stage_counts,
                    seconds=time.time() - t0,
                    divisors={h: {k: np.asarray(v).tolist() for k, v in dd.items()} for h, dd in divisors.items()})
        out.with_suffix(".json").write_text(json.dumps(info, indent=1))
        print(a.family, a.design, d, "lr", lr, "steps", steps, "params", n_params, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
