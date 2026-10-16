#!/usr/bin/env python3
"""Score every transplant view of the frozen manifest with one model's seeds.

Views per target (``views.csv``): the identity splice (every distant slot refilled with the
recording's own rows through the same assembly path, at the largest dose), and for each
dose and donor draw the PSG-wake and PSG-sleep carriers. Each unmodified recording is also
scored twice: alone, as the evaluation scored it, and inside a full batch, to measure
repeat-inference tolerance.

Writes ``<out>/<model>/seed<k>.npz``: ``targets`` (views, 60, 5) float32 probabilities of
the target epochs (wake, light, deep, REM, gap), ``original`` and ``original_batched``
(13, 3000, 5). The models' own seeds only; the ensemble is formed in ``analyze.py``.

    python infer.py --cache <dir> --predictions <dir> --manifest <dir> --out <dir> \\
        --model foundry=<phase dir> [--config <phase dir with config/>] [--limit N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arcasleep.cli import _load_phase_config
from arcasleep.training.driver import softmax
from common import DOSES, DRAWS, TARGET, X_LENGTH, build_view, load_recording, remote_order

NAMES = [f"sen{i:03d}" for i in range(3, 16)]


def view_table(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in targets.itertuples():
        rows.append((t.target, t.recording, t.start, "all", "identity", -1))
        for dose in DOSES:
            for k in range(DRAWS):
                for carrier in ("wake", "sleep"):
                    rows.append((t.target, t.recording, t.start, dose, carrier, k))
    return pd.DataFrame(rows, columns=["target", "recording", "start", "dose", "view", "draw"])


class Builder:
    def __init__(self, recs, assignments):
        self.recs, self.a = recs, assignments

    def __call__(self, v) -> np.ndarray:
        rec = self.recs[v.recording]
        a = lambda key: self.a[f"{v.target}/{key}"]
        lens, pos = a("slot_len"), a("slot_pos")
        if v.view == "identity":
            rows = rec.x[pos]
        else:
            donor = self.recs[str(a(f"donor{v.draw}"))]
            blocks = a(f"{v.view}{v.draw}")
            rows = np.concatenate([donor.x[b:b + n] for b, n in zip(blocks, lens)])
        d = DOSES[v.dose]
        if d is not None:
            keep = np.isin(pos, remote_order(rec, v.start)[:d])
            pos, rows = pos[keep], rows[keep]
        x = build_view(rec, pos, rows)
        lo, hi = v.start - 180, v.start + TARGET + 180
        assert np.array_equal(x[lo:hi], rec.x[lo:hi]) and np.array_equal(x[rec.n:], rec.x[rec.n:])
        assert np.array_equal(x[:, -1], rec.x[:, -1])
        return x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True, help="name=<phase dir holding model_outputs/>")
    parser.add_argument("--config", type=Path, help="phase dir holding config/ (default: the model's)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, help="first N views only (timing pilot)")
    args = parser.parse_args()

    name, phase = args.model.split("=", 1)
    phase = Path(phase)
    config = _load_phase_config(args.config or phase, "bidoze")
    recs = {n: load_recording(args.cache, args.predictions, n) for n in NAMES}
    targets = pd.read_csv(args.manifest / "targets.csv")
    views = view_table(targets)
    views.to_csv(args.manifest / "views.csv", index_label="view_id")
    if args.limit:
        views = views.iloc[: args.limit]
    build = Builder(recs, np.load(args.manifest / "assignments.npz"))
    input_shape = recs[NAMES[0]].x.shape
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    originals = np.stack([recs[n].x for n in NAMES])

    for k in args.seeds:
        t0 = time.time()
        model = config.family.load_checkpoint(phase / f"model_outputs/model_final_seed_{k}.pisces", input_shape=input_shape)
        alone = np.stack([softmax(np.asarray(model.infer(x[None])))[0] for x in originals])
        batched = []
        for i in range(0, len(originals), args.batch):
            xb = originals[i:i + args.batch]
            fill = np.concatenate([xb, np.repeat(xb[-1:], args.batch - len(xb), 0)])
            batched.append(softmax(np.asarray(model.infer(fill)))[: len(xb)])
        batched = np.concatenate(batched)
        probs = np.empty((len(views), TARGET, 5), dtype=np.float32)
        rows = list(views.itertuples())
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            xb = np.stack([build(v) for v in chunk])
            if len(chunk) < args.batch:
                xb = np.concatenate([xb, np.repeat(xb[-1:], args.batch - len(chunk), 0)])
            p = softmax(np.asarray(model.infer(xb)))
            for j, v in enumerate(chunk):
                probs[i + j] = p[j, v.start:v.start + TARGET]
            if i % (args.batch * 50) == 0:
                print(f"{name} seed {k}: {i + len(chunk)}/{len(rows)} views, {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(out_dir / f"seed{k}.npz", targets=probs, view_id=views.index.to_numpy(),
                            original=alone.astype(np.float32), original_batched=batched.astype(np.float32),
                            recordings=np.array(NAMES))
        print(f"{name} seed {k}: {len(rows)} views in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
