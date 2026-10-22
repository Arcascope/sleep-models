"""Train one strand-fusion arm on one draw and save every recording's strand and output probabilities.

The draws, fit/eval split, recipe and stage divisor are those of ``cooperating_arms/run.py``
(batch 2, LR .001, step patience 700/1750, smoothing 350, participant eval fraction .3).
Predictions are saved for the draw's test recordings and for its adult eval-share recordings
(the recordings a decoding refit uses), as ``predictions/<role>_<dataset>_<subject>.npz``.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from arcasleep.data.batching import assemble_batch
from arcasleep.data.cache import load_cached_features
from arcasleep.training.balanced import Patience

import data as D
from fusion_model import ARMS, StrandFusionModel, arm_spec, model_config


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def predict_recordings(model, found, refs, role, out):
    paths = {(s.ref.dataset_name, s.ref.subject_id): s.path for ds in found.values() for s in ds}
    run = jax.jit(lambda p, x: model.module.apply({"params": p}, x, jnp.ones(x.shape[:2]), return_outputs=True))
    for ref in refs:
        x, labels = D.recording_batch(paths[(ref.dataset_name, ref.subject_id)])
        n = len(labels)
        outs = run(model.params, x)
        fusion = np.exp(np.asarray(outs["log_head"]))[0, :n] if model.spec.fusion else None
        D.save_recording(out / f"{role}_{ref.dataset_name}_{ref.subject_id}.npz", labels,
                         np.asarray(outs["strand_probabilities"])[:, 0, :n], fusion)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-run", type=Path, default=D.SOURCE_RUN)
    ap.add_argument("--labels-root", type=Path, default=D.LABELS_ROOT)
    ap.add_argument("--max-seconds", type=int, default=14400)
    ap.add_argument("--epochs", type=int, default=3000)
    a = ap.parse_args()
    out = a.run / a.arm / f"draw{a.draw}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists() and not json.loads((out / "complete.json").read_text()).get("time_capped"):
        print("already complete", out)
        return
    spec = arm_spec(a.arm)
    cfg = model_config(spec)
    model = StrandFusionModel(cfg, spec)
    model._ensure_built((8, 61, 2), seed=a.seed)
    counts = {k: sum(int(x.size) for x in jax.tree.leaves(v)) for k, v in model.params.items()}
    model.input_shape = (3000, 61, 2)
    config = dict(arm=a.arm, label=ARMS[a.arm][0], draw=a.draw, seed=a.seed, config=cfg.to_dict(), spec=asdict(spec),
                  parameters=sum(counts.values()), parameter_groups=counts,
                  code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  started=timestamp(), source_run=str(a.source_run),
                  batch_size=2, max_epochs=a.epochs, max_train_seconds=a.max_seconds,
                  patience=dict(lr_drop=700, early_stop=1750, unit="steps", smoothing_steps=350),
                  initial_lr=0.001, weight_decay=0, balance_beta=0.999)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ["arm", "parameters", "started", "code_commit"]}), flush=True)
    print("devices:", jax.devices(), flush=True)
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("a local GPU is required for training")
    found, fit, val, tests = D.draw_data(a.source_run, a.draw)
    split = D.split_record(fit, val, tests)
    for name in ["fit", "validation", "test"]:
        for cohort in (["testSleepAccel", "testClinical"] if name == "test" else ["trainSleepAccel", "trainClinical"]):
            assert split["counts"][name].get(cohort, 0) > 0, (name, cohort)
    (out / "split.json").write_text(json.dumps(split, indent=2))
    features = load_cached_features([s for ds in D.TRAIN for s in found[ds]])
    X, y = assemble_batch(features, fit, 3000, 3000, 4)
    V, v = assemble_batch(features, val, 3000, 3000, 4)
    del features
    np.save(out / "stage_divisor.npy", D.stage_divisor(found, a.labels_root))

    def callback(epoch, params, history):
        (out / "history.json").write_text(json.dumps(history, indent=2))

    start = time.monotonic()
    history = model.fit_balanced(X, y, V, v, batch_size=2, initial_lr=0.001, weight_decay=0, balance_beta=0.999,
                                 patience=Patience(700, 1750, "steps", 350), max_epochs=a.epochs, fit_seed=a.seed,
                                 eval_batch_size=1, max_train_seconds=a.max_seconds,
                                 fit_state_path=out / "fit_state.pkl", epoch_callback=callback)
    elapsed = time.monotonic() - start
    (out / "history.json").write_text(json.dumps(history, indent=2))
    model.save_weights_to_directory(out / "weights")
    prior = json.loads((out / "training_finished.json").read_text())["seconds"] if (out / "training_finished.json").exists() else 0
    (out / "training_finished.json").write_text(json.dumps(
        dict(at=timestamp(), seconds=elapsed + prior, time_capped=history.get("time_capped", False)), indent=2))
    del X, y, V, v
    jax.clear_caches()
    pred = out / "predictions"
    pred.mkdir(exist_ok=True)
    predict_recordings(model, found, tests, "test", pred)
    predict_recordings(model, found, [r for r in val if r.dataset_name in D.ADULT_EVAL], "eval", pred)
    done = dict(finished=timestamp(), train_seconds=elapsed + prior, time_capped=history.get("time_capped", False),
                test_recordings=len(tests), parameters=config["parameters"])
    (out / "complete.json").write_text(json.dumps(done, indent=2))
    print("COMPLETE", json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
