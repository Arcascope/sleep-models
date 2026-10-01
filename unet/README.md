# UNet

UNet is a temporal U-Net sleep-stage model: the accelerometry architecture
described in

> M. Olsen, J. M. Zeitzer, R. N. Richardson, P. Davidenko, P. J. Jennum,
> H. B. D. Sørensen and E. Mignot, "A flexible deep learning architecture for
> temporal sleep stage classification using accelerometry and
> photoplethysmography," *IEEE Transactions on Biomedical Engineering*, 2022.
> [doi:10.1109/TBME.2022.3187945](https://doi.org/10.1109/TBME.2022.3187945)

It ships in the `arcasleep` distribution as the `arcasleep_unet` import
package and the `arcasleep unet` command group; installing and the shared
commands are described in the [repository README](../README.md).

## Model

The network reads the raw spectrogram frames of the shared feature cache
(`config/inputs.json`: 10 s windows every 2 s, 60 bins from 0.1 to 6 Hz, `mag`
and `jerk` channels), 15 frames per 30 s epoch, and predicts wake, light, deep
and REM per epoch, plus a gap class for epochs with no accelerometer data
(`score_gap`). `config/model.json` holds the paper's Table II settings: ten
encoder and decoder blocks (`m_blocks`), 16 base filters (`c_u`) growing by
the cube root of 2 per block, and 16×3 time×frequency kernels.

Recordings are scored whole, padded to the longest in the cohort rounded up
to a multiple of 1024 epochs, so the frame count divides by `2**m_blocks`.

## Training

`config/training.json` follows the paper's procedure. Each epoch samples
1024-epoch segments from the training nights, each segment drawn to contain a
uniformly chosen sleep stage, and fits them with the effective-number
class-balanced cross-entropy (`balance_beta`). Adam starts at `lr`; 30% of the
training subjects (`eval_fraction`) are held out, the learning rate drops by
√10 after `lr_drop_patience` epochs without an improvement in their Cohen's
kappa, training stops after `early_stop_patience`, and the best epoch is kept.

`scripts/harness.sh` runs the whole recipe: one model per fold of the
committed split plan (`config/splits/weaver_5fold.json`), each scored on the
fold it held out, then a final model on every fold, scored on the external
test sets. Phases can be named to run a subset, in order:

```bash
FEATURE_CACHE_DIR=<cache root> UNET_RUN_ID=<run> scripts/harness.sh [--dry-run] [final fold0 ...]
```

Results land in `results/<run>/training/<phase>/`. One phase by hand:

```bash
arcasleep unet train \
    -pc config/inputs.json -mc config/model.json -cvc config/cv.json \
    -tc config/training.json -mtc config/metrics.json -extn <run>/training
arcasleep unet evaluate -t results/<run>/training -i '<cache>/*_features.npz' -o results/<run>/evaluation
```

## Release and inference

`arcasleep unet publish` builds the same release bundle as BiDoze
(`model.safetensors` with the BatchNorm statistics beside the parameters,
`config.json`, `preprocessing.json`, the card), and `arcasleep unet predict`
scores raw accelerometer files with it:

```bash
arcasleep unet publish -t results/<run>/training/final -o release/v0.1.0 --version v0.1.0 --preview
arcasleep unet predict release/v0.1.0 recordings/*.csv -o predictions/
```
