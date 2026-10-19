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


The scripts run from the repository root with `PYTHONPATH=core/src:bidoze/src`.
Use the existing arcasleep environment's Python. `run_local.py` accepts the same
three cache/output paths plus `--baseline-run` and `--arms`; it serializes fits
under the shared GPU lock and runs the CPU report after each fit. `report.py`
can also be called separately with `--run`, `--source-run`, `--baseline-run`,
`--arm`, `--draw`, and `--seed`. It verifies the complete test population and
finite normalized probabilities before marking a report ready.

`CooperatingModel.load_research(fit_directory)` reloads a selected checkpoint
from its companion configuration and weight directory. Validation probes are
milestone snapshots, not necessarily the checkpoint selected at the end.
`progress.py` renders those snapshots without further inference. A blank stage
panel is explicitly labelled when the validation night has no PSG epochs of it.

`publish_completed.py` is for this session's explicitly authorized direct-to-main
research-log workflow. It waits for specified completed reports, fast-forwards
the clean tracked log tree, appends one timestamped section, verifies image
references, stages only that log and the fit's images, and pushes without force.
It stops on unrelated tracked edits or conflicts. Its publication receipt is
saved beside the report. Never run this publisher without authorization to post.


`advance_counts.py` waits for publication of the initial control, then schedules
one architecture at 3 and 5 arms if its mean four-stage κ across the two cohorts
is within .02 of the one-path control and neither cohort is more than .03 below
it. Among eligible architectures within .01 of the best cohort-equal mean, it
prefers shared CNN, then intermediate exchange, then final-loss-only. This is an
exploratory scheduling heuristic for development data, declared after A7 and
before the remaining initial results; it is not a statistical acceptance test.
It records `setup/count_followup.json` and publishes its decision and each fit.
No continuation is chosen from a session-capped initial comparison. Higher counts
and capacity increases await assessment of these comparisons and replication.

## October 19 revision: all draws before selection, then about 5M parameters

The draw-0 near-control gate in `advance_counts.py` was an exploratory scheduling
choice. It is superseded by `revision_matrix.py`: retain control1, final3, final5,
final7, exchange7 and shared7 on every existing draw 0–4 (30 fits including
already completed fits). Then run control1_5m, final5_5m, exchange5_5m and
shared5_5m on all five draws (20 fits). This is a compact capacity comparison at
five arms, not an exhaustive cross of branch count and capacity. Widths are
416/152/152/192 and instantiated counts are 5,065,428/4,940,400/4,967,160/4,956,724.
The shared CNN remains width 256; transformer allocation grows. Every multiarm
configuration retains one balanced loss on the probability mean.

`revision_matrix.py` takes the same five paths as `advance_counts.py`, preflights
participant separation and both adult cohorts in every split, verifies historical
G ×3 prediction coverage, and writes `setup/revision_matrix.json`. `--prepare-only`
performs preflight without fitting. The worker waits for the original count
service to finish, then skips published fits and runs the remaining matrix under
the shared GPU lock. Each fit publishes its report and refreshes the overview.
A session-capped fit stops the matrix for review rather than entering comparisons.
Use one initialization seed per draw here; these runs do not measure pure seed
variance. The adult participants overlap across draws: compare paired changes
to the same-draw, same-budget control separately per cohort; report every draw,
mean and range, without treating five overlapping draws as five independent
population samples. The fixed draw manifests cover 27/28 SleepAccel and 32/32
Clinical participants in test sets.

`comparison.py` writes a top-of-section 2×4 overview: four-stage κ, sleep/wake κ,
wake specificity at 95% sleep sensitivity, and signed TST bias. All displayed
experiments use their common completed draws; coverage is explicit in the figure
and adjacent JSON, with source rows in CSV. Rate/κ axes span 0–1 and TST bias
uses symmetric limits. Per-fit comparison embeds are removed; learning,
contribution and selected hypnogram figures remain.
