# October 11: surviving wake evidence

Confidential research. Push this branch only to `confidential-work` (`Arcascope/cui-sleep-models`).
The plan is `sleep-models-research-logs/2026/October/11/log.md`.

`prepare.py` freezes conservative vessel groups and an inner vessel holdout, reuses the release's
participant-grouped land split and processing, and writes model inputs plus label-free raw-axis
moments. `train.py` compares warm head-only/full-backbone adaptation with a land-only control.
The first seed is screening; no single-seed gain is a confirmed result. Every best checkpoint
is saved, including the initial model at epoch zero. Head-only training asserts that all other
parameter roots remain exactly unchanged. Initial model predictions accompany every held-out
prediction. The stage divisor is frozen at the adult release's value.

`probes.py` fits fixed logistic readouts of baseline wake scores, with/without raw-axis moments,
holding out outer vessels and an inner vessel group. Its ROC comparison is descriptive;
thresholds for actual test calls use only inner evaluation sleep epochs. `common_motion.py`
estimates scalar common motion without PSG on one time block and reports residual variance on
another, with displaced and unrelated references. Noise reduction is not a wake result.
`inventory_physiology.py` inventories actual measurement candidates and excludes vendor sleep
labels. Coverage and alignment must be validated before a physiology experiment.

Outputs: `bidoze/results/boat-evidence-2026-10-11/`. This run is registered in the log repo's
`results-s3-sync/confidential-runs.txt`. S3 destination:
`s3://arcasleep-conf-sleep-models-data/research_log_artifacts/bidoze/results/boat-evidence-2026-10-11/`.
AWS tasks use a dedicated role restricted to this prefix. Do not use the ordinary task role's
public-repository artifact destination for this study. No raw cohort files are required by the
training image: only prepared inputs and initial bundles.

Screen in fold E (SURF-to-Somerset transfer), then A–D. Start with one seed and fixed learning
rate 1e-4; this phase tests mechanisms, not hyperparameter optimization. Max 100 warm epochs,
no learning-rate sweep, early stopping after 700 non-improving steps using 175-step smoothed
inner eval κ. Keep the best unsmoothed inner eval checkpoint. The same code/recipe applies to
all arms; the no-adaptation baseline is saved alongside them. Fold E's initial checkpoint has
already seen all land training participants, including the inner land eval. Boat outer tests
and SleepAccel-Clinical are independent; the land eval is a stopping monitor, not an unbiased
performance estimate.
