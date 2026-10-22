"""Save the three BiDoze-G seeds' probabilities of one draw (October 16 §8) in the format of run.py.

Writes ``<run>/g3/draw<d>/predictions/<role>_<dataset>_<subject>.npz`` for the draw's test recordings
and adult eval-share recordings, and the draw's ``stage_divisor.npy``.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import data as D
from arcasleep_bidoze.model import BiDozeNetModel

ARMS = ["bidoze_g", "bidoze_g_s1", "bidoze_g_s2"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--draw", type=int, required=True)
    a = ap.parse_args()
    out = a.run / "g3" / f"draw{a.draw}"
    if (out / "complete.json").exists():
        print("already complete", out)
        return
    import jax
    import jax.numpy as jnp
    paths = [D.G3_RUN / arm / f"draw{a.draw}/model_outputs/model_draw{a.draw}_seed_{i}.pisces" for i, arm in enumerate(ARMS)]
    assert all(p.exists() for p in paths), [str(p) for p in paths if not p.exists()]
    models = [BiDozeNetModel.loadz(p) for p in paths]
    parameters = int(sum(x.size for m in models for x in jax.tree.leaves(m.params)))
    runs = [jax.jit(lambda x, m=m: jax.nn.softmax(m.module.apply({"params": m.params}, x, jnp.ones(x.shape[:2]),
                                                                 train=False), axis=-1)) for m in models]
    found, fit, val, tests = D.draw_data(D.SOURCE_RUN, a.draw)
    pred = out / "predictions"
    pred.mkdir(parents=True, exist_ok=True)
    np.save(out / "stage_divisor.npy", D.stage_divisor(found, D.LABELS_ROOT))
    files = {(s.ref.dataset_name, s.ref.subject_id): s.path for ds in found.values() for s in ds}
    for role, refs in [("test", tests), ("eval", [r for r in val if r.dataset_name in D.ADULT_EVAL])]:
        for ref in refs:
            x, labels = D.recording_batch(files[(ref.dataset_name, ref.subject_id)])
            n = len(labels)
            probs = np.stack([np.asarray(r(x))[0, :n] for r in runs])
            D.save_recording(pred / f"{role}_{ref.dataset_name}_{ref.subject_id}.npz", labels, probs)
    (out / "complete.json").write_text(json.dumps(dict(parameters=parameters, models=[str(p) for p in paths])))
    print("COMPLETE draw", a.draw, parameters)


if __name__ == "__main__":
    main()
