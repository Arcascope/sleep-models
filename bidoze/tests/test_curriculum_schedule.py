"""The curriculum recipe's learning-rate schedule counts the optimizer steps an epoch takes."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from arcasleep_bidoze.model import BiDozeNetConfig  # noqa: E402


@pytest.mark.parametrize("sliding_copies", [1, 2])
def test_the_schedule_epoch_is_the_steps_one_epoch_takes(sliding_copies: int) -> None:
    config = BiDozeNetConfig.from_dict(
        {
            "num_classes": 4,
            "score_gap": True,
            "d_model": 16,
            "n_heads": 2,
            "n_transformer_layers": 1,
            "d_ff": 32,
            "conv_kernels": [3],
            "dropout": 0.0,
            "temporal_slide": False,
        }
    )
    model = config.make_model()
    rng = np.random.default_rng(0)
    rows, length, batch_size = 6, 16, 4
    x = rng.normal(size=(rows, length, 6, 2)).astype(np.float32)
    y = rng.integers(0, 4, size=(rows, length)).astype(np.int32)
    seen: list[dict] = []

    model.train_with_callbacks(
        train_x=x,
        train_y=y,
        val_x=None,
        val_y=None,
        initial_lr=1e-3,
        weight_decay=0.0,
        epochs=6,
        batch_size=batch_size,
        callbacks=[],
        fit_seed=0,
        sliding_copies=sliding_copies,
        save_history_plot=False,
        train_verbose=0,
        epoch_observer=lambda _model, info, _history: seen.append(info),
    )

    final = [info for info in seen if info["final_stage"]]
    steps = [info["stage_optimizer_step"] for info in final]
    taken = np.diff([0, *steps])
    assert set(taken) == {final[0]["steps_per_epoch"]}
    assert final[0]["steps_per_epoch"] == -(-rows * sliding_copies // batch_size)
