import numpy as np
import pytest

pytest.importorskip("jax")

from arcasleep_bidoze.model import BiDozeNetConfig  # noqa: E402


def test_release_compatible_mlp_emits_stage_and_gap_logits() -> None:
    config = BiDozeNetConfig.from_dict(
        {
            "num_classes": 4,
            "score_gap": True,
            "d_model": 32,
            "n_heads": 4,
            "n_transformer_layers": 1,
            "d_ff": 64,
            "conv_kernels": [3, 7],
            "dropout": 0.0,
        }
    )
    model = config.make_model()
    features = np.zeros((1, 16, 60, 2), dtype=np.float32)
    assert model.infer(features).shape == (1, 16, 5)
