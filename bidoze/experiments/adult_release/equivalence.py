"""Do the flat adult bundles score as the composite bundles they replace?

Usage (``PYTHONPATH`` = the ``core/src`` and ``bidoze/src`` of the code that loads the bundle):
    python equivalence.py dump OUT_DIR BUNDLE
    python equivalence.py compare OLD_DIR NEW_DIR

``dump`` writes per-night ``.npz`` files: on the 32 SleepAccel-Clinical nights (cached features
through the bundle's network, 5-way mean probabilities and the divisor-decoded calls), and on the
33 EESM19 nights (raw accelerometer through ``predict`` with the GGIR filter off: probabilities
and raw stage labels). ``BUNDLE`` is a composite bundle for the code before the flat layout
(``AdultPredictor``) and a flat bundle for the code after (``Predictor``). ``compare`` reports the
largest probability difference and the number of epochs whose stage call differs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GRID = Path("/home/eric/Engineering/Work/autopisces/sleep-models-model-grid/bidoze/experiments")
sys.path[:0] = [str(GRID / "model_grid")]


def load(bundle: Path):
    try:
        from arcasleep_bidoze.adult import AdultPredictor
    except ImportError:
        from arcasleep.predict import Predictor
        return Predictor(bundle)
    return AdultPredictor(bundle)


def dump(out: Path, bundle: Path) -> None:
    import json

    import build as g
    from arcasleep_bidoze.adult import hybrid_calls
    from score_releases import EPOCH_S, read_accel, read_pairs

    from arcasleep.data.batching import pool_frames_to_epochs, split_gap_row
    from arcasleep.data.bundle import CacheBundle

    predictor = load(bundle)
    core = getattr(predictor, "ensemble", predictor)
    out.mkdir(parents=True, exist_ok=True)
    for sid, path in g.cache_nights(g.CACHE_PREFIX["SleepAccel-Clinical"]).items():
        b = CacheBundle.from_npz(path)
        n = len(b.y_for_n_classes(4)[0])
        is_gap = split_gap_row(pool_frames_to_epochs(b.X[0][None], np.zeros((1, n)))[0])[1] >= 1.0
        logits = g.model_logits(core.model, core.context_epochs, core.model.config.frames_per_epoch, b.X[0], n,
                                core.epoch_multiple)
        mean = np.exp(logits)
        stage = mean[:, :4] / mean[:, :4].sum(1, keepdims=True)
        calls = np.where(is_gap, 4, hybrid_calls(stage, stage, predictor.divisor))
        np.savez(out / f"clinical_{sid}.npz", probabilities=mean, calls=calls)
    manifest = json.loads((g.EESM_RAW / "conversion_manifest.json").read_text())
    for sid in sorted(m["id"] for m in manifest if m["status"] == "converted"):
        psg = read_pairs(g.EESM_RAW / "cleaned_psg" / f"{sid}_cleaned_psg.out")
        accel = read_accel(g.EESM_RAW / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
        result = predictor.predict(accel, origin=float(psg[0, 0]), ggir_sleep_filter=False)
        np.savez(out / f"eesm19_{sid}.npz", probabilities=result.probabilities,
                 calls=np.asarray(result.raw_labels, dtype=str), is_gap=result.is_gap)
    print("dumped", len(list(out.glob("*.npz"))), "nights to", out)


def compare(old: Path, new: Path) -> None:
    for prefix in ("clinical", "eesm19"):
        files = sorted(old.glob(f"{prefix}_*.npz"))
        worst, differing, epochs = 0.0, 0, 0
        for f in files:
            a, b = np.load(f), np.load(new / f.name)
            assert a["probabilities"].shape == b["probabilities"].shape, f.name
            worst = max(worst, float(np.abs(a["probabilities"] - b["probabilities"]).max()))
            differing += int((a["calls"] != b["calls"]).sum())
            epochs += a["calls"].size
        print(f"{prefix}: {len(files)} nights, {epochs} epochs; max |probability difference| {worst:.3e}; "
              f"epochs with a different stage call {differing}")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "dump":
        dump(Path(args[0]), Path(args[1]))
    else:
        compare(Path(args[0]), Path(args[1]))
