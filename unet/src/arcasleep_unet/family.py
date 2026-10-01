"""UNet as a model family of the shared pipeline."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arcasleep.models import MODEL_TYPE_KEY, ModelFamily


class UNetFamily(ModelFamily):
    name = "unet"
    checkpoint_class = "UNetModel"

    def model_config(self, data: Mapping[str, Any]):
        from arcasleep_unet.model import UNetConfig

        return UNetConfig.from_dict(dict(data))

    def training_config(self, data: Mapping[str, Any]):
        from arcasleep_unet.training import TrainingConfig

        return TrainingConfig.from_dict(dict(data))

    def scoring_lengths(self, config, epoch_counts: Sequence[int]) -> tuple[int, int]:
        # Whole recordings: the longest, rounded up to a length the network
        # accepts, so every recording is scored in one compiled pass.
        multiple = config.model.padded_epoch_multiple
        length = -(-max(epoch_counts, default=1) // multiple) * multiple
        return length, length

    def describe_training(self, config) -> str:
        training = config.training
        return (
            f"segment_epochs={training.segment_epochs} batch_size={training.batch_size} "
            f"lr={training.lr} balance_beta={training.balance_beta}\n"
            f"max_epochs={training.max_epochs} lr_drop_patience={training.lr_drop_patience} "
            f"early_stop_patience={training.early_stop_patience} "
            f"eval_fraction={training.eval_fraction}"
        )

    def train_fold(self, config, *, fold, fit_seed, split_seed, features, datasets):
        from arcasleep_unet.training import train_fold

        return train_fold(
            config,
            fold=fold,
            fit_seed=fit_seed,
            split_seed=split_seed,
            features=features,
            datasets=datasets,
        )

    def load_checkpoint(self, path: Path, input_shape: tuple[int, ...] | None = None):
        from arcasleep_unet.model import UNetModel

        return UNetModel.loadz(path, input_shape=input_shape)

    def load_release(self, config: Mapping[str, Any], weights: Path):
        from arcasleep_unet.model import UNetConfig, UNetModel

        data = {k: v for k, v in config.items() if k != MODEL_TYPE_KEY}
        model = UNetModel(UNetConfig.from_dict(data))
        model.load_weights_safetensors(weights)
        return model

    def prediction_window(self, model) -> tuple[int | None, int]:
        return None, model.config.padded_epoch_multiple

    def describe_length(self, input_shape: Sequence[int]) -> str:
        return "Recordings of any length are scored whole, in one pass."

    def describe_architecture(
        self, config: Mapping[str, Any], parameter_count: int | None
    ) -> list[str]:
        m_blocks = config.get("m_blocks", 10)
        c_u = config.get("c_u", 16)
        kernel = (config.get("kernel_temporal", 16), config.get("kernel_freq", 3))
        size = f" ({parameter_count / 1e6:.2f} M parameters)" if parameter_count else ""
        return [
            f"A temporal U-Net over the spectrogram frames (`{m_blocks}` encoder and "
            f"decoder blocks, `{c_u}` base filters, `{kernel[0]}`×`{kernel[1]}` "
            "time×frequency kernels) followed by a per-epoch readout, the architecture "
            "described in Olsen et al., IEEE TBME 2022, "
            f"[doi:10.1109/TBME.2022.3187945](https://doi.org/10.1109/TBME.2022.3187945){size}.",
        ]


FAMILY = UNetFamily()
