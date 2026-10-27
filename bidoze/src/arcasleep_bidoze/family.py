"""BiDoze as a model family of the shared pipeline."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arcasleep.models import MEMBER_PREFIX, MODEL_TYPE_KEY, ModelFamily


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
        if training.recipe == "balanced":
            windows = (
                f"\nwindowed: {training.windows_per_recording} x {training.window_epochs}-epoch "
                f"windows per recording, mix={training.window_mix}, "
                f"weak label rules={training.weak_label_rules}"
                if training.windowed
                else ""
            )
            return (
                f"recipe=balanced x_length={training.x_length} y_length={training.y_length} "
                f"batch_size={training.batch_size} lr={training.lr} "
                f"balance_beta={training.balance_beta}\n"
                f"max epochs={training.epochs} lr_drop_patience={training.lr_drop_patience} "
                f"early_stop_patience={training.early_stop_patience} "
                f"({training.patience_unit}) eval_smoothing_steps={training.eval_smoothing_steps} "
                f"eval_fraction={training.eval_fraction}{windows}"
            )
        return (
            f"x_length={training.x_length} y_length={training.y_length} "
            f"sliding_copies={training.sliding_copies} "
            f"curriculum={training.curriculum_loss_mode}\n"
            f"deblip: {training.deblip_schedule} "
            f"{training.deblip_adjustment_epochs} epochs @ {training.deblip_logit_weight}"
        )

    def fit_eval_split(self, config, fold, split_seed):
        from arcasleep_bidoze.training import fit_eval_split

        return fit_eval_split(config, fold, split_seed)

    def train_fold(self, config, *, fold, fit_seed, split_seed, features, datasets):
        from arcasleep_bidoze.training import train_fold

        return train_fold(
            config,
            fold=fold,
            fit_seed=fit_seed,
            split_seed=split_seed,
            features=features,
            datasets=datasets,
        )

    def load_checkpoint(self, path: Path, input_shape: tuple[int, ...] | None = None):
        from arcasleep_bidoze.model import BiDozeNetModel

        return BiDozeNetModel.loadz(path, input_shape=input_shape)

    def load_release(self, config: Mapping[str, Any], weights: Path):
        """One model, or the ``members`` models of an ensemble bundle as one batched call."""
        from arcasleep_bidoze.adult import StackedModel
        from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel

        data = {k: v for k, v in config.items() if k != MODEL_TYPE_KEY}
        members = config.get("members")
        if members is None:
            model = BiDozeNetModel(BiDozeNetConfig.from_dict(data))
            model.load_weights_safetensors(weights)
            return model
        models = []
        for k in range(int(members)):
            model = BiDozeNetModel(BiDozeNetConfig.from_dict(data))
            model.load_weights_safetensors(weights, prefix=f"{MEMBER_PREFIX}{k}/")
            models.append(model)
        return StackedModel(models)

    def prediction_window(self, model) -> tuple[int | None, int]:
        return int(model.input_shape[0]), 1

    def describe_length(self, input_shape: Sequence[int]) -> str:
        hours = input_shape[0] * 30 / 3600  # one row per 30-s epoch
        return (
            f"Up to {hours:g} hours are scored in one pass; longer recordings are "
            "scored in consecutive passes."
        )

    def describe_architecture(
        self, config: Mapping[str, Any], parameter_count: int | None
    ) -> list[str]:
        if config.get("strands"):
            return self._describe_strands(config, parameter_count)
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

    @staticmethod
    def _describe_strands(config: Mapping[str, Any], parameter_count: int | None) -> list[str]:
        kernels = ", ".join(str(k) for k in config.get("conv_kernels", [3, 7, 31, 63]))
        size = f" ({parameter_count / 1e6:.2f} M parameters)" if parameter_count else ""
        return [
            f"`{config['strands']}` narrow bi-causal networks (strands) each score every epoch: a "
            f"convolution bank (kernels `{kernels}` epochs, width `{config.get('strand_width', 112)}`), "
            f"a bi-causal transformer (`{config.get('n_transformer_layers', 4)}` layers per direction) and "
            f"a learned gate. A fusion transformer (`{config.get('fusion_layers', 2)}` layers, width "
            f"`{config.get('fusion_width', 128)}`, `{config.get('fusion_heads', 4)}` heads) reads the "
            f"strands' features across the whole recording and gives the per-epoch output{size}.",
        ]


FAMILY = BiDozeFamily()
