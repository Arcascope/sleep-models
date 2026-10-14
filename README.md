# Arcasleep

Arcascope's open source, open weights sleep models.

---

One Python distribution, `arcasleep`, carries the shared data handling, training,
evaluation and release tooling (`core/`, the `arcasleep` import package) and
every model family (`bidoze/` and `unet/`, the `arcasleep_bidoze` and
`arcasleep_unet` import packages). Model
weights are released on the Hugging Face Hub, one repository per model variant.

## Install

The base install is dependency-free: it provides the manifest tools and a
working CLI, and every other command imports its dependencies on demand.

```bash
pip install arcasleep                           # CLI + manifest tools, no heavy deps
pip install "arcasleep[bidoze,inference,hub]"   # run a released BiDoze model
pip install "arcasleep[unet,train]"             # training stack (adds optax, matplotlib)
pip install "arcasleep[prep]"                   # build the feature cache
pip install "arcasleep[aws]"                    # S3 manifests and result upload
pip install "arcasleep[publish]"                # build a release bundle
pip install "arcasleep[all]"                    # installs all extras
```

`pisces-lite` and `arcascope-senpy` are public on PyPI; pisces-lite supplies
the dataset/IO layer and the metric registry, and its `[proc]` extra (pulled by
`[prep]`) the prebuilt senpy NUFFT wheel.

## Commands

Every model family reads the same feature cache, so the cache and results
commands are shared; training, scoring and release sit under each family:

```bash
arcasleep process          # build the feature cache (needs [prep])
arcasleep splits-plan / splits-apply
arcasleep summarize        # pool finished phases into a metrics_summary.csv
arcasleep compare          # draw a metric grid across summaries
arcasleep hypnogram        # hypnogram PNGs from prediction CSVs
arcasleep card-preview     # render a built bundle's model card to HTML
arcasleep manifest-local / manifest-s3 / upload-results

arcasleep bidoze train     # run the recipe from a feature cache
arcasleep bidoze evaluate  # score a checkpoint on a cohort
arcasleep bidoze publish   # build a safetensors release bundle, optionally upload it
arcasleep bidoze predict   # per-epoch stages for raw accelerometer files

arcasleep unet train | evaluate | publish | predict
```

### Tab completion

Install the `completion` extra, then register the shell hook once:

```bash
pip install "arcasleep[completion]"
eval "$(register-python-argcomplete arcasleep)"          # bash, zsh
register-python-argcomplete --shell fish arcasleep | source   # fish
```

Add the `eval` line to `~/.bashrc` or `~/.zshrc` to keep it across sessions. In
zsh, run `autoload -U bashcompinit && bashcompinit` before it if `compdef` or
`complete` is not found. After that, `arcasleep hyp<Tab>` completes to
`arcasleep hypnogram`, `arcasleep bidoze <Tab>` lists `train`, `evaluate`,
`publish` and `predict`, and `--<Tab>` lists the options of the command typed so
far. Without the extra, the CLI works the same but does not complete.

## Models

The current pediatric model variants are trained on the clinical pediatric pool described in [Weaver et al. 2026](https://pubmed.ncbi.nlm.nih.gov/40676371/). This set has subjects with no recorded diagnosis (n=37), as well as mild (n=94), moderate (n=33), and severe (n=38) obstructive sleep apnea (OSA). We are grateful to Glenn Weaver and his group for sharing their data with us.

### BiDoze

Published on Hugging Face as [`arcascope/arcasleep-bidoze-pediatric`](https://huggingface.co/arcascope/arcasleep-bidoze-pediatric); the model lives in [`bidoze/`](bidoze/).

Wake/Light/Deep/REM predictions, plus a 5th head for identifying gaps in accelerometer. 

1. **Architecture:** `[CNN encoder] -> [ALiBi Transformers] -> [5-head MLP]`.
2. Developed via a semi-autonomous research loop using LLMs.
    1. Agents were given the goal to optimize model architecture and training regime with fixed train/test splits. 
    2. "Best" as measured by the geometric mean of TST and WASO MAPE. 

## UNet

Published on Hugging Face as [`arcascope/arcasleep-unet-pediatric`](https://huggingface.co/arcascope/arcasleep-unet-pediatric); the model lives in [`unet/`](unet/).

Wake/Light/Deep/REM predictions, plus a 5th head for identifying gaps in accelerometer.

1. **Architecture:** the temporal U-Net described in Olsen et al., "A flexible deep learning architecture for temporal sleep stage classification using accelerometry and photoplethysmography," IEEE TBME 2022 ([doi:10.1109/TBME.2022.3187945](https://doi.org/10.1109/TBME.2022.3187945)), over raw 2-s spectrogram frames.
2. Trained with the paper's procedure: class-balanced segment sampling and loss, early stopping on held-out subjects.

## Releasing the package

The distribution is published to PyPI from CI with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC): no API
token is stored. `.github/workflows/release.yml` builds sdist + wheel from the
repository root, tests the built wheel, and publishes when a GitHub Release is
published for an `arcasleep-vX.Y.Z` tag.

The tag is the source of truth for the version: `setuptools-scm` reads the
nearest `arcasleep-vX.Y.Z` tag (configured in `pyproject.toml`), so the
distribution version and the release tag cannot drift. To release, push an
`arcasleep-vX.Y.Z` tag and publish a GitHub Release for it -- there is no
version literal to edit. The workflow fails if the tag and the built version
disagree. A manual `workflow_dispatch` runs the build and wheel tests without
publishing; commits past a tag build as the next patch dev version
(`X.Y.(Z+1).devN`), which is what `.github/workflows/dry-run-testpypi.yml`
rehearses against TestPyPI.
