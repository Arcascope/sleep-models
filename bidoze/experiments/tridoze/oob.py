"""Out-of-bag κ of the bagged BiDoze-G members, beside their in-bag and eval-share κ.

Each bagged member s (``bidoze_g_bag<s>``) trains on a bootstrap resample of its draw's fit
recordings (``bootstrap_seed`` 100 + s); its out-of-bag recordings are the fit recordings the
resample left out. κ is Cohen's κ over the 4 stages of the argmax of the member's output (no stage
divisor), pooled over the PSG-scored epochs of a group of recordings, for adults (SleepAccel and
SleepAccel-Clinical training nights) and children (weaverDraw) apart. The bagged ensemble's
out-of-bag κ scores each fit recording with the mean of the members for which it is out of bag.
The unbagged BiDoze-G ×3 members are listed for comparison (they trained on every fit recording,
so they have no out-of-bag share).

Writes ``<run>/analysis/oob.csv`` and prints it. ``oob_check.csv`` gives the κ of each BiDoze-G ×3
member (trained on every fit recording) on each bagged member's in-bag and out-of-bag recordings, so
the bagged member's in-bag − out-of-bag difference can be set against how hard the two sets are.

Usage: python oob.py [--run RUN] [--draws 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pisces_lite.cv import SubjectRef
from sklearn.metrics import cohen_kappa_score

from arcasleep_bidoze.training import bootstrap_recordings

RUN = Path(__file__).resolve().parents[2] / "results/tridoze-2026-10-16"


def kappa(store_list, keys) -> tuple[float, int]:
    ys, cs = [], []
    for k in keys:
        y = store_list[0][f"__labels__/{k}"].astype(int)
        p = np.mean([s[k] for s in store_list], axis=0)
        scored = (y >= 0) & (y < 4)
        ys.append(y[scored])
        cs.append(p[scored, :4].argmax(axis=1))
    if not ys:
        return np.nan, 0
    y, c = np.concatenate(ys), np.concatenate(cs)
    return float(cohen_kappa_score(y, c)), len(y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    rows, checks = [], []
    for d in args.draws:
        bags = {}
        for s in range(3):
            path = args.run / f"bidoze_g_bag{s}/draw{d}/train_predictions/draw{d}_seed_{s}.npz"
            if path.exists():
                bags[s] = np.load(path)
        if len(bags) < 3:
            print(f"skip draw {d}: bagged members missing")
            continue
        fit = [str(k) for k in bags[0]["__fit__"]]
        evals = [str(k) for k in bags[0]["__eval__"]]
        refs = [SubjectRef(*k.split("/", 1)) for k in fit]
        in_bag = {s: {f"{r.dataset_name}/{r.subject_id}" for r in bootstrap_recordings(refs, 100 + s)} for s in bags}
        for group, keep in [("adult", lambda k: not k.startswith("weaverDraw/")), ("child", lambda k: k.startswith("weaverDraw/"))]:
            for s, store in bags.items():
                for share, keys in [("in bag", [k for k in fit if k in in_bag[s]]),
                                    ("out of bag", [k for k in fit if k not in in_bag[s]]),
                                    ("eval share", evals)]:
                    keys = [k for k in keys if keep(k)]
                    kap, n = kappa([store], keys)
                    rows.append({"draw": d, "group": group, "model": f"bagged member {s}", "share": share,
                                 "recordings": len(keys), "epochs": n, "kappa_4stage": kap})
            # The ensemble's out-of-bag κ: each recording by the members it is out of bag for.
            ys_all = []
            for k in fit:
                if not keep(k):
                    continue
                outs = [bags[s] for s in bags if k not in in_bag[s]]
                if outs:
                    ys_all.append((k, outs))
            pooled_y, pooled_c = [], []
            for k, outs in ys_all:
                y = outs[0][f"__labels__/{k}"].astype(int)
                p = np.mean([o[k] for o in outs], axis=0)
                scored = (y >= 0) & (y < 4)
                pooled_y.append(y[scored])
                pooled_c.append(p[scored, :4].argmax(axis=1))
            if pooled_y:
                rows.append({"draw": d, "group": group, "model": "bagged ensemble", "share": "out of bag",
                             "recordings": len(ys_all), "epochs": int(sum(len(y) for y in pooled_y)),
                             "kappa_4stage": float(cohen_kappa_score(np.concatenate(pooled_y), np.concatenate(pooled_c)))})
            kap, n = kappa(list(bags.values()), [k for k in evals if keep(k)])
            rows.append({"draw": d, "group": group, "model": "bagged ensemble", "share": "eval share",
                         "recordings": sum(keep(k) for k in evals), "epochs": n, "kappa_4stage": kap})
            for s, arm in enumerate(["bidoze_g", "bidoze_g_s1", "bidoze_g_s2"]):
                path = args.run / f"{arm}/draw{d}/train_predictions/draw{d}_seed_{s}.npz"
                if not path.exists():
                    continue
                store = np.load(path)
                for b in bags:
                    for share, keys in [("in bag", [k for k in fit if k in in_bag[b]]),
                                        ("out of bag", [k for k in fit if k not in in_bag[b]])]:
                        kap, n = kappa([store], [k for k in keys if keep(k)])
                        checks.append({"draw": d, "group": group, "bagged_member": b, "share": share,
                                       "model": f"BiDoze-G member {s}", "kappa_4stage": kap})
                for share, keys in [("fit (trained on)", fit), ("eval share", evals)]:
                    keys = [k for k in keys if keep(k)]
                    kap, n = kappa([store], keys)
                    rows.append({"draw": d, "group": group, "model": f"BiDoze-G member {s}", "share": share,
                                 "recordings": len(keys), "epochs": n, "kappa_4stage": kap})
    table = pd.DataFrame(rows)
    out = args.run / "analysis/oob.csv"
    table.to_csv(out, index=False)
    pd.DataFrame(checks).to_csv(args.run / "analysis/oob_check.csv", index=False)
    if not table.empty:
        print(table.groupby(["group", "model", "share"])[["recordings", "kappa_4stage"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
