"""SAMME-style per-epoch loss weights for the next boosted member, from the members before it.

The ensemble of the members so far is the mean of their probabilities (train_predictions.py) on
the draw's fit recordings. An epoch is wrong when the ensemble's most likely of the 4 stages is
not its PSG stage. With ``err`` the share of PSG-scored fit epochs the ensemble gets wrong and
K = 4 stages, SAMME's member weight is ``α = ln((1 − err)/err) + ln(K − 1)``; each wrong epoch
weighs ``exp(α)`` and each right one 1, and the weights are scaled to a mean of 1 over the scored
fit epochs. Unscored and gap epochs carry no loss and keep weight 1.

Writes ``<out>.npz`` (keyed ``<dataset>/<subject_id>``, for training's ``epoch_weights``) and
``<out>.json``: err, α, and how concentrated the weights are:
- ``ess_fraction``: effective sample size (Σw)²/(N·Σw²) over N scored epochs, 1 for equal weights;
- ``top5_epoch_share``, ``top10_epoch_share``: share of the total weight on the 5% and 10% heaviest
  epochs;
- ``transition_*``: epochs within 1 epoch of a PSG stage change, as a share of epochs and of weight
  (where PSG scorers disagree most, so weight that moves there may follow label noise);
- per stage: share of epochs and share of weight;
- ``top10_recording_share``: share of the total weight on the 10% heaviest recordings.

Usage: python boost_weights.py <out stem> <train_predictions npz> [<npz> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STAGES = ["wake", "light", "deep", "rem"]


def ensemble_on_fit(paths: list[Path]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    stores = [np.load(p) for p in paths]
    fit = list(stores[0]["__fit__"])
    proba = {k: np.mean([s[k] for s in stores], axis=0) for k in fit}
    labels = {k: stores[0][f"__labels__/{k}"].astype(int) for k in fit}
    return proba, labels


def transition_mask(y: np.ndarray) -> np.ndarray:
    change = np.zeros(y.shape, dtype=bool)
    scored = (y >= 0) & (y < 4)
    step = scored[1:] & scored[:-1] & (y[1:] != y[:-1])
    change[1:] |= step
    change[:-1] |= step
    return change


def main(out: Path, paths: list[Path]) -> None:
    proba, labels = ensemble_on_fit(paths)
    wrong, scored = {}, {}
    for k, p in proba.items():
        y = labels[k]
        scored[k] = (y >= 0) & (y < 4)
        wrong[k] = scored[k] & (np.argmax(p[:, :4], axis=1) != y)
    n_scored = sum(int(s.sum()) for s in scored.values())
    err = sum(int(w.sum()) for w in wrong.values()) / n_scored
    alpha = float(np.log((1 - err) / err) + np.log(len(STAGES) - 1))
    raw = {k: np.where(wrong[k], np.exp(alpha), 1.0) for k in proba}
    scale = n_scored / sum(float(raw[k][scored[k]].sum()) for k in proba)
    weights = {k: np.where(scored[k], raw[k] * scale, 1.0).astype(np.float32) for k in proba}

    w_all = np.concatenate([weights[k][scored[k]] for k in proba])
    y_all = np.concatenate([labels[k][scored[k]] for k in proba])
    t_all = np.concatenate([transition_mask(labels[k])[scored[k]] for k in proba])
    ordered = np.sort(w_all)[::-1]
    top = ordered[: max(1, len(w_all) // 10)]
    per_recording = np.sort([float(weights[k][scored[k]].sum()) for k in proba])[::-1]
    stats = {
        "members": [p.name for p in paths], "fit_recordings": len(proba), "scored_epochs": n_scored,
        "err": err, "alpha": alpha, "wrong_weight": float(np.exp(alpha) * scale), "right_weight": float(scale),
        "ess_fraction": float(w_all.sum() ** 2 / (len(w_all) * (w_all**2).sum())),
        "top5_epoch_share": float(ordered[: max(1, len(w_all) // 20)].sum() / w_all.sum()),
        "top10_epoch_share": float(top.sum() / w_all.sum()),
        "top10_recording_share": float(per_recording[: max(1, len(per_recording) // 10)].sum() / per_recording.sum()),
        "transition_epoch_share": float(t_all.mean()),
        "transition_weight_share": float(w_all[t_all].sum() / w_all.sum()),
        "transition_err": float(np.mean(np.concatenate([wrong[k][scored[k]] for k in proba])[t_all])),
        "stage_epoch_share": {s: float(np.mean(y_all == i)) for i, s in enumerate(STAGES)},
        "stage_weight_share": {s: float(w_all[y_all == i].sum() / w_all.sum()) for i, s in enumerate(STAGES)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **weights)
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
