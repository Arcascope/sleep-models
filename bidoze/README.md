# BiDoze

BiDoze is a convolution bank and bi-causal transformer sleep-stage model. It
ships in the `arcasleep` distribution as the `arcasleep_bidoze` import package
and the `arcasleep bidoze` command group; installing, the shared commands and
releasing the package are described in the [repository README](../README.md).

Model releases are published on the Hugging Face Hub, one repository per
model variant. The first variant is trained on the pediatric clinical cohort:
[`arcascope/arcasleep-bidoze-pediatric`](https://huggingface.co/arcascope/arcasleep-bidoze-pediatric).
Versions (starting at `v0.1.0`) are tags in that repository.
`arcascope/arcasleep-bidoze` is the family landing page; it lists the variants.

This directory holds the model's run configs (`config/`), release info and
cards (`cards/`), and the training image and harness (`docker/`, `scripts/`).
Training on AWS GPUs is described in [`deploy/`](../deploy/README.md).
Commands below run from here.

Retraining the pediatric and adult releases with the gap rule (`scripts/harness.sh` with
`BIDOZE_RELEASE`) is described in [ADULT.md](ADULT.md#retraining-a-release).

Scoring adult nights (the release's sleep/wake calls with light/deep/REM from a model
trained on adult nights) is described in [ADULT.md](ADULT.md).

## Releasing a model

A training phase directory (`<run>/training/final`) holds the config snapshots,
the `.pisces` checkpoint and its own `cv_results_<n>_stages.csv`. `publish`
builds the public bundle from it:

```bash
arcasleep bidoze publish \
    -t runs/<run>/training/final \
    -o release/v0.1.0 \
    --version v0.1.0 \
    --info cards/bidoze-pediatric.toml \
    --preview \
    --overwrite
```

To upload, add `--repo arcascope/arcasleep-bidoze-pediatric` and drop
`--preview`. `--overwrite` replaces an existing output directory.

`--metrics-summary` is optional. Without it, the phase's own results CSV and
any sibling `fold<i>` directories are pooled: each of the phase's datasets
becomes a cohort, the folds together become one `cv` cohort, and an
`assets/metrics_grid.png` comparison figure is drawn and embedded in the card.
A downloaded phase directory is therefore enough to build a fully scored card.

The bundle contains `model.safetensors`, `config.json`, `preprocessing.json`,
`metrics.json`, `provenance.json`, `checksums.sha256`, a model card
(`README.md`) and any `assets/` figures. Checksums are verified before any
upload.

Without an authored card (from `--info` or `--card`), the card carries a
`PENDING` review marker, and upload is refused unless `--allow-unreviewed` is
passed.

### The release info file

Each model variant has a small TOML file, passed with `--info`, that describes
the release, so none of it is hard-coded
(`cards/bidoze-pediatric.toml` is the example). Every key is optional:

| Key | Purpose |
|:---|:---|
| `repo` | Hub repository the card is titled for; `--repo` overrides it |
| `card` | the authored Markdown, relative to the TOML |
| `audience` | the "For" row of the at-a-glance table |
| `protocol` | how the reported numbers were produced |
| `require_cv` | take the glance score only from the pooled `cv` cohort |
| `[cohorts]` | cohort name to one-line description, for the performance section |

`--info` does not upload anything; upload still needs `--repo`.

### The model card

The authored Markdown named by `--info` (or passed directly as `--card`, which
takes precedence) holds intended use and limitations, your own metrics and
comparisons, and figures. The card also has generated sections:

- an at-a-glance table
- performance
- quickstart
- input requirements
- outputs
- model
- provenance

Their order is set in `_model_card` in
`core/src/arcasleep/artifacts/publish.py`. They are read from the bundle, so
they always match the release. The quickstart is plain Markdown in
`core/src/arcasleep/artifacts/templates/quickstart.md`. Relative images
(`![...](figure.png)`) are copied into a checksummed `assets/` directory, their
links rewritten, and uploaded in the same Hub commit; remote URLs are left
unchanged.

Preview the card before publishing. `arcasleep card-preview <bundle>` (or
`publish --preview`) renders the card to a standalone HTML page with images
inlined; open it in a browser to see what the Hub will show.

**`publish --preview` never uploads,** even when `--repo` is given: the command
builds, verifies and renders the card, then stops.

`--preview` takes an optional directory: without one, `card.html` is written
into the bundle; with one, `card.html` is written there. This needs the
`preview` extra:

```bash
pip install "arcasleep[preview]"
arcasleep card-preview release/v0.1.0 -o card.html
```

Adding `--repo` uploads the verified bundle as one atomic Hub commit. Loading a
released bundle does not need `.pisces`:

```python
from arcasleep.artifacts.publish import load_bundle_model
model = load_bundle_model("release/v0.1.0")
```
