# Cooperating transformer arms

One supervised class-balanced cross-entropy scores the arithmetic mean of arm
probabilities. Every arm receives the same examples; there are no individual
supervised losses, batch routing, bagging, or arm removal during training.

- `final7`: seven separate CNN/transformer paths; communication through the loss.
- `exchange7`: the same, with a rank-16 additive exchange after transformer layer 2.
  Forward and reverse streams exchange only with the corresponding direction.
  Each arm writes to and reads from the other arms' mean through learned adapters;
  the residual scale starts and stays at 0.1.
- `shared7`: one width-256 CNN, seven separate projections and width-88 transformers.
- `control1`: one width-256 path. The current controls include the same additional
  feature dropout used by all new arms; historical G is a separate reference.

Four transformer layers, FFN width twice arm width, 0.2 existing dropout plus
0.2 independent channel dropout at each arm input (constant over time per row).
All use the fixed gap rule. The extra feature dropout is held fixed across new
architectures; an old split-batch comparison also changes objective and dropout.

`run.py` takes `--run`, `--source-run` (existing `setup/cache/drawN`),
`--labels-root` (BIDsleep reference cache), `--arm`, `--draw`, and `--seed`.
It inherits the balanced recipe, batch 2, LR .001, step patience 700/1750,
smoothing 350, participant eval fraction .3 with split seed 0. Both adult cohorts
must occur in fit, validation and test. The fit checks participant separation and
saves recording lists and source-cache SHA256 hashes. All new fits use identical
sampler/initialization seeds for paired architecture comparisons.

Weights live in an ordinary tensor directory with a companion `config.json` that
records the full arm specification. These research weights are not plain BiDoze
`.pisces` bundles. Use the experiment model class with the recorded specification
to load them. A capped fit is provisional and may be resumed with the same command;
remove neither its state nor its predictions to resume it. Completed fits refuse
accidental overwrite. Do not compare capped results as converged fits.

Per-epoch history and selected validation snapshots describe training progression;
validation probes use one recording from each adult cohort. Test predictions are
saved once training has selected the checkpoint. Report all test subjects by cohort;
selected hypnograms are illustrations, not a separate evaluation sample.

Initial widths for 3/5/7 arms target one full model's approximately 2.24M parameters.
The first phase runs the three seven-arm architectures and a current one-path control,
then expands only a promising architecture to 3/5 arms and another seed/split.
