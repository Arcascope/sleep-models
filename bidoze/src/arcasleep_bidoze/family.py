"""BiDoze as a model family of the shared pipeline."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arcasleep.models import MODEL_TYPE_KEY, ModelFamily


class BiDozeFamily(ModelFamily):
    name = "bidoze"
    checkpoint_class = "BiDozeNetModel"

    def model_config(self, data: Mapping[str, Any]):
        from arcasleep_bidoze.model import BiDozeNetConfig

        return BiDozeNetConfig.from_dict(dict(data))

    def training_config(self, data: Mapping[str, Any]):
        from arcasleep_bidoze.training import TrainingConfig

        return TrainingConfig.from_dict(dict(data))

    def scoring_lengths(self, config, epoch_counts: Sequence[int]) -> tuple[int, int]:
        # The model is fitted at a fixed context and always scored at it.
        return config.training.x_length, config.training.y_length

    def describe_training(self, config) -> str:
        training = config.training
        return (
            f"x_length={training.x_length} y_length={training.y_length} "
            f"sliding_copies={training.sliding_copies} "
            f"curriculum={training.curriculum_loss_mode}\n"
            f"deblip: {training.deblip_schedule} "
            f"{training.deblip_adjustment_epochs} epochs @ {training.deblip_logit_weight}"
        )

    def train_fold(self, config, *, fold, fit_seed, features, datasets):
        from arcasleep_bidoze.training import train_fold

        return train_fold(
            config, fold=fold, fit_seed=fit_seed, features=features, datasets=datasets
        )

    def load_checkpoint(self, path: Path, input_shape: tuple[int, ...] | None = None):
        from arcasleep_bidoze.model import BiDozeNetModel

        return BiDozeNetModel.loadz(path, input_shape=input_shape)

    def load_release(self, config: Mapping[str, Any], weights: Path):
        from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel

        data = {k: v for k, v in config.items() if k != MODEL_TYPE_KEY}
        model = BiDozeNetModel(BiDozeNetConfig.from_dict(data))
        model.load_weights_safetensors(weights)
        return model

    def describe_architecture(
        self, config: Mapping[str, Any], parameter_count: int | None
    ) -> list[str]:
        d_model = config.get("d_model", 256)
        n_heads = config.get("n_heads", 8)
        n_layers = config.get("n_transformer_layers", 4)
        kernels = ", ".join(str(k) for k in config.get("conv_kernels", [3, 7, 31, 63]))
        direction = "bi-causal" if config.get("use_bicausal", True) else "causal"
        size = f" ({parameter_count / 1e6:.2f} M parameters)" if parameter_count else ""
        return [
            f"A convolution bank (kernels `{kernels}` epochs) captures local movement "
            f"patterns; a {direction} transformer (`{n_layers}` layers per direction, "
            f"width `{d_model}`, `{n_heads}` heads) carries context across the whole "
            f"recording; a learned gate fuses the two before a per-epoch readout{size}.",
        ]


FAMILY = BiDozeFamily()
