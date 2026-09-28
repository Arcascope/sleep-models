# Autofish

`autofish` is the production training, evaluation, and release package for
Autofish sleep-stage models.

The distribution is named `arcascope-autofish` on PyPI; it installs the
`autofish` import package and the `autofish` command.

Model releases live on the Hugging Face Hub, one repository per model variant:
the first is the pediatric clinical cohort,
[`arcascope/autofish-pediatric`](https://huggingface.co/arcascope/autofish-pediatric).
Versions (starting `v1.0.0`) are tags within that repository. `arcascope/autofish`
is the family landing page, grouping the variants.

## Install

The base install is dependency-free: it provides the manifest tools and a
working CLI, and every other command imports its dependencies on demand.

```bash
pip install arcascope-autofish               # CLI + manifest tools, no heavy deps
pip install "arcascope-autofish[inference]"  # JAX/Flax + pisces-lite + safetensors
pip install "arcascope-autofish[train]"      # training stack (adds optax, matplotlib)
pip install "arcascope-autofish[aws]"        # S3 manifests and result upload
pip install "arcascope-autofish[publish]"    # build a release bundle
pip install "arcascope-autofish[hub]"        # build + upload to the Hugging Face Hub
```

`pisces-lite` (4.0.0) and `arcascope-senpy` are public on PyPI; pisces-lite
supplies the dataset/IO layer and the metric registry. Feature preparation
additionally needs its `[proc]` extra, which pulls the prebuilt senpy wheel:

```bash
pip install "arcascope-autofish[prep]"
```

## Commands

```bash
autofish train        # run the recipe from a feature cache
autofish evaluate     # score a checkpoint on a cohort
autofish summarize    # pool finished phases into a metrics_summary.csv
autofish compare      # draw a metric grid across summaries
autofish publish      # build a safetensors release bundle, optionally upload it
autofish card-preview # render a built bundle's model card to HTML
autofish process      # build the feature cache (needs [prep])
autofish manifest-local / manifest-s3 / upload-results
```

## Releasing a model

A training phase directory (`<run>/training/final`) holds the config snapshots,
the `.pisces` checkpoint and its own `cv_results_<n>_stages.csv`. `publish`
turns it into the public bundle:

```bash
autofish publish \
    -t runs/<run>/training/final \
    -o release/v1.0.0 \
    --version v1.0.0 \
    --card card.md \                      # your authored model card
    --preview release/v1.0.0 \            # optional local HTML preview
    --repo arcascope/autofish-pediatric
```

`--metrics-summary` is optional. When omitted, the phase's own results CSV and
any sibling `fold<i>` directories are pooled automatically: the phase's datasets
become cohorts and the folds a single `cv` cohort, and a `assets/metrics_grid.png`
comparison figure is drawn and embedded in the card. A downloaded phase
directory is therefore enough to build a fully scored card.

The bundle contains `model.safetensors`, `config.json`, `preprocessing.json`,
`metrics.json`, `provenance.json`, `checksums.sha256`, a model card
(`README.md`) and any `assets/` figures. The card is titled for `--repo` (default
`arcascope/autofish-pediatric`), so a variant built for its own repository is
labelled as that variant. Checksums are verified before any upload. Without
`--card` the card carries a `PENDING` review marker and an upload is refused
unless `--allow-unreviewed` is passed.

### The model card

`--card` is your Markdown: intended use and limitations, how to apply the model,
your own metrics and comparisons, and figures. It is placed after the title; the
generated model, input-pipeline, labels, evaluation and provenance sections
always follow, so the card describes what the network does rather than listing
raw config keys, and cannot drift from the release. Relative images
(`![...](figure.png)`) are copied into a checksummed `assets/` directory, their
links rewritten, and shipped in the same Hub commit; remote URLs are left as-is.

Preview it before publishing. `autofish card-preview <bundle>` (or
`publish --preview path.html`) renders the card to a standalone HTML page with
images inlined, so you can open it in a browser and see what the Hub will show.
With `publish --preview` nothing is uploaded: the command builds, verifies and
renders the card, then stops, even when `--repo` is also given. `--preview`
takes an optional path; omit it to write `card.html` into the bundle, or point
it at a directory to write `card.html` there. This needs the `preview` extra:

```bash
pip install "arcascope-autofish[preview]"
autofish card-preview release/v1.0.0 -o card.html
```

Adding `--repo` uploads the verified bundle as one atomic Hub commit. Loading a
released bundle does not need `.pisces`:

```python
from autofish.artifacts.publish import load_bundle_model
model = load_bundle_model("release/v1.0.0")
```

## Releasing the package

The distribution is published to PyPI from CI with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC): no API
token is stored. `.github/workflows/release.yml` builds sdist + wheel from this
directory, tests the built wheel, and publishes when a GitHub Release is
published. The tag is namespaced for the monorepo: `autofish-v1.0.0`.

The tag is the source of truth for *when* to publish and `autofish.__version__`
for *what*; the workflow fails if they disagree. Before releasing, bump the
version and push an `autofish-vX.Y.Z` tag. A manual `workflow_dispatch` runs the
build and wheel tests without publishing. `.github/workflows/dry-run-testpypi.yml`
rehearses the same path against TestPyPI.
