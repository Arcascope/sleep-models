# Weaver external replication of SeqSmoother

Compare accelerometer-only HR estimation from simple jerk peak finding, ordinary
temporal smoothing, the official Nightbeat numerical pipeline, and a reconstructed
SeqSmoother model. The target is the incremental value of the learned corrector
at matched timestamps and coverage.

Sources:
- Paper: https://arxiv.org/pdf/2610.06823v1
- Official Nightbeat: https://github.com/eth-siplab/Nightbeat
- Pinned Nightbeat commit: 119dd8e818509ae49e7429879f9e4140bc2fb671

This is a **provisional external methodological replication**. It is not an exact
reproduction of the paper's NightbeatDB results. The source of Weaver's Kubios
columns needs confirmation; raw ECG and beat timestamps are absent from the revised
CSVs. No source data or existing sleep-model experiment is modified.

## Data and frozen protocol

The 2026-10-07 header inventory found 246 files / 244 children:
239 Actigraph, 158 Apple, 135 Fitbit, 149 Garmin, 5 Axivity, and 1 Pixel recordings.
Headers indicate Kubios in 237 Actigraph recordings. None of the five Axivity files
has PSG or Kubios; they have optical wrist-HR columns instead. Header availability
does not establish usable overlapping data.

The primary pilot uses 38 Actigraph children selected by a seeded permutation of
header-eligible participants, excluding development-only child 00lhv171. The
frozen splits file records all remaining untouched participants. The first 13
selected children are each held out once. Per fold, the other 37 are divided into
29 training and 8 validation children. All nights and devices of one child stay
together. This prevents leakage from duplicate PSG across wrist devices.

- 60-second windows, 15-second step, absolute timestamps; endpoints identify windows.
- Resample to 100 Hz with linear interpolation. Reject windows crossing gaps over
  250 ms. Record native sampling rate and duplicate timestamps in each cache.
  Upsampling a slower sensor adds no frequency information.
- Primary reference: arithmetic mean of Kubios None Mean HR samples in the same
  half-open window. Average the percentage quality column and divide by 100.
  Quality weights only training loss; there is no hard quality threshold.
- Retain finite primary reference HR in 30–120 bpm, matching the paper's range.
  This is a limited physiological range for pediatric data. Report excluded
  label-window counts; do not generalize to higher rates.
- Forty-token sequences cannot cross timestamp gaps greater than 45 seconds.
  First 39 tokens and endpoints without full context are not model-eligible.
- Participant-macro error averages recording errors within each child, then children.
  Report pooled metrics only as descriptive summaries.
- Final inference consumes accelerometry only. Reference HR, its quality, PSG,
  demographics, and optical HR are not model inputs.

## Methods

1. Original-style rectified jerk FFT peak in 54–78 bpm, with 1-bpm bin spacing.
2. Same FFT with a 30–120 bpm search range to cover more pediatric HR.
3. A centered five-minute median of the wide-band peaks.
4. A stillness/peak-clarity screen using recording medians (accelerometry only).
5. Official Nightbeat intermediate curve-tracing anchor.
6. Official Nightbeat final gates: zero motion-artifact fraction, curve/time-domain
   estimates within 10 bpm, and curve estimate within 10 bpm of a five-minute
   candidate median. This uses the published repository notebook's final gates.
7. Intermediate anchor with spectral fallback, and its five-minute median.
8. A constant HR estimated from training labels only, with no test-label access.
9. A Flax SeqSmoother reconstruction with 26 inputs, 40-token context, width 64,
   two pre-norm encoder blocks, four heads, MLP width 256, dropout 0.15, and a
   bounded HR output. Input/output normalization and a 32-wide GELU head produce
   the reported 106625 parameter count. **The paper omits head details; matching
   the count does not prove matching the original architecture.**

Nightbeat's numerical function bodies are loaded unchanged from the pinned source
using Python AST. This avoids unrelated import-time Torch/Polars/dataset-loader
dependencies. The upstream loader is never used, and reference arrays passed to
the numerical predictor are empty. Upstream functions may abstain. Failed curve
tracking and rejected chunks reduce availability; they are not silently filled in
the final Nightbeat result. Fallback is a separate explicitly named baseline.

## Model recipe and declared differences

AdamW: learning rate 0.0003, weight decay 0.001, five warm-up epochs, cosine decay
to 0.000001; batch size 128, gradient norm clipped to 1, maximum 120 epochs,
minimum 15 epochs, patience 30. The loss combines quality-weighted Huber loss
(delta 1.5), a squared tail penalty above 3 bpm (weight 0.08), and a batch
correlation penalty (weight 0.15). Every sequence position is supervised;
final-token predictions form the evaluation grid.

Feature standardization is fit only on training children. Training augmentation
includes the paper's anchor/spectral/subharmonic jitter and PSD noise/masking.
The paper does not give every preprocessing detail: the order of augmentation,
PSD normalization, anchor quality representation, output head, and sequence
stride are implementation choices recorded here. The training stride is 40,
and validation/test stride is 1. Validation selection uses participant-macro
final-token MAE. This is a declared adaptation, not an exact-source claim.

Save per-fold config, training history, checkpoint, feature order, normalization,
and participant-disjoint per-window predictions. No checkpoint is a validated
deployable HR estimator.

## Running

The tested CPU environment is recorded in requirements.txt. The frozen child
selection and folds are preserved in pilot_splits.json.

For a fresh output directory, copy pilot_splits.json to OUTPUT/splits.json and
clone the official Nightbeat repository into OUTPUT/vendor/Nightbeat. Check out
commit 119dd8e818509ae49e7429879f9e4140bc2fb671 before running. Existing runs retain
their own frozen splits and cached artifacts.

Use the project's Python environment. Set the following paths to your local
inputs and desired output directories; these relative paths are examples. The
commands below assume the repository root is the current directory. Script paths
can also be supplied in full when running from another directory.

```sh
WEAVER_DATA=data/weaver-2026-10-revised
SLEEPACCEL_DATA=data/SleepAccel
PILOT_RUN=bidoze/results/seqsmoother-weaver-pilot
WEAVER_RUN=bidoze/results/seqsmoother-weaver-full
SLEEPACCEL_RUN=bidoze/results/seqsmoother-sleepaccel
OLD_HR_EPOCHS=analysis/hr-epochs.csv.gz
```

`OLD_HR_EPOCHS` is the saved earlier peak-finding CSV, not a file included in this
repository. Paths supplied to CLI options are relative to your current directory;
all data locations are explicit inputs.

A reference command is:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  python bidoze/experiments/seqsmoother_weaver/run.py \
  --data "$WEAVER_DATA" \
  --output "$PILOT_RUN"
```

The orchestration script resumes completed caches/folds, checks the pinned
Nightbeat checkout, and updates a report after every fold. The first two-epoch
end-to-end validation runs live under smoke-training and are excluded from
formal reports. A full run can take substantial CPU time.

Artifacts include inventory.csv, splits.json, manifest.json, compressed feature
caches, checkpoints and normalization, predictions.csv.gz, summary.csv,
participants.csv, paired.csv, coverage.csv, error_coverage.png, and an explicit
status.json. Status remains incomplete until every requested fold finishes.
Participant-bootstrap intervals use 5000 child resamples. The harmonic AUROC
is a pooled descriptive diagnostic, never independent-window inference.

## Reporting device differences

devices.py evaluates each wrist device separately on the same frozen held-out
children. The existing SeqSmoother checkpoints and their Actigraph normalization
are reused; no device-specific retraining occurs. Neural results on Apple,
Fitbit, and Garmin are therefore **device-transfer evaluations of an
Actigraph-trained model**, not device-specific learned estimators.

Within each device and reference source, first average absolute error within a
recording, then average recording MAEs within each child, then average children.
Do not average devices together. Report epoch availability and participant/night
counts alongside error. Raw device summaries can contain different children;
paired_device_vs_actigraph.csv instead matches child, night, timestamp,
reference source, and method availability. Matching references must agree.

Header counts can exceed scored counts: missing or fragmented accelerometry,
out-of-range/missing reference HR, or insufficient 40-token context can leave a
recording unscorable. These exclusions appear in recording_audit.json,
preparation.json, and availability_by_device.csv.

Run the device evaluation with:

```sh
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python bidoze/experiments/seqsmoother_weaver/devices.py \
  --data "$WEAVER_DATA" \
  --run "$PILOT_RUN" --workers 4
```

--baselines-only makes a quick preliminary peak-finding report; its status is
baselines_complete_model_transfer_pending. The full run prepares the
Nightbeat-based features and scores the frozen checkpoints. --report-only
reuses those device caches. Artifacts live under RUN/devices, including per-
recording, per-child, per-device, and paired CSVs plus device_comparison.png.
The original Actigraph pilot artifacts and models are retained.

Axivity secondary summaries use four different children with Apple optical HR
as the reference. They contain all baseline-eligible windows, without requiring
40-token model context. Keep these separate from the Kubios pilot; no neural
transfer or Nightbeat comparison is reported for that cohort.

### Full-cohort GPU study

`full.py` includes all header-eligible Actigraph/Kubios/PSG children, audits signal
and reference eligibility, and freezes five participant-grouped folds before
training. Each child is held out once; 20% of the remaining children validate.
The run reuses completed feature caches, trains on CUDA, evaluates other wrist
devices with the corresponding held-out Actigraph checkpoint, and updates the
research-log section after each fold and after device evaluation when `--log` is supplied.
Previously explored children are included in this CV; no separate group of previously unexamined children is reserved for a final test.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python bidoze/experiments/seqsmoother_weaver/full.py \
  --data "$WEAVER_DATA" \
  --output "$WEAVER_RUN" \
  --pilot "$PILOT_RUN" \
  --workers 4
```

The full run uses CUDA by default; pass `--platform cpu` on a machine without a
CUDA GPU. To update an existing research log, add `--log path/to/log.md`. Without
`--log`, the experiment writes only to its output directory. Completed caches
can be reused even when the pilot and full-run outputs are on different filesystems.

### SleepAccel comparison (October 8)

`sleepaccel.py` uses the same 28 SleepAccel participants as the October 6 HR
analysis. It verifies the original 50-Hz jerk peak and median Apple optical HR
against saved October 6 windows, then compares predictions on a 60-s / 15-s grid.
New features reject accelerometer gaps over 250 ms and models require 40-window
context. A separate 50-Hz peak calculation preserves the original preprocessing;
the 100-Hz peak calculation measures the resampling difference.

The frozen Weaver transfer estimator averages the five completed Weaver fold
models; no SleepAccel labels fit those models. SleepAccel CV uses the same
SeqSmoother reproduction with five participant-grouped folds: 18 training,
4–5 validation, and 5–6 test participants. Reference HR is the within-window
median Apple optical HR; training uses unit label weights because HR quality is
not provided. `train.py --reference-source apple_optical_hr_median` explicitly
selects these caches; the default remains Kubios. The experiment estimates
agreement with optical HR, not ECG accuracy.

`summary.csv` reports participant-macro mean absolute error and pooled median
absolute error separately. `paired.csv` compares methods on identical windows.
Still/clear subsets use per-recording motion and narrow-band clarity medians;
the 54–78 bpm reference filter describes a best-case subset, not a deployable
selection rule. Nightbeat error must be compared with other methods on its own
retained windows. `quantization.py` holds the October 6 labels and masks fixed
while rounding raw coordinates before resampling. `reference_audit.py` and
`conditioned_errors.py` preserve the dated Weaver diagnostics.

```sh
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python bidoze/experiments/seqsmoother_weaver/sleepaccel.py \
  --data "$SLEEPACCEL_DATA" \
  --output "$SLEEPACCEL_RUN" \
  --weaver "$WEAVER_RUN" \
  --vendor "$PILOT_RUN/vendor/Nightbeat" \
  --old-epochs "$OLD_HR_EPOCHS" --workers 4

python bidoze/experiments/seqsmoother_weaver/sleepaccel_report.py \
  --run "$SLEEPACCEL_RUN"
```

### Reference and conditioning diagnostics

These diagnostics accept the input files explicitly and can be run from any
working directory. Use the pilot splits to reproduce the 13-child reference
audit, or pass other frozen splits to select a different held-out population.
Output names do not assume a participant count or run date.

```sh
python bidoze/experiments/seqsmoother_weaver/conditioned_errors.py \
  --weaver-run "$WEAVER_RUN" \
  --old-epochs "$OLD_HR_EPOCHS" \
  --output "$SLEEPACCEL_RUN"

python bidoze/experiments/seqsmoother_weaver/reference_audit.py \
  --data "$WEAVER_DATA" \
  --predictions "$WEAVER_RUN/devices/predictions_by_device.csv.gz" \
  --splits "$PILOT_RUN/splits.json" \
  --output "$SLEEPACCEL_RUN"

python bidoze/experiments/seqsmoother_weaver/quantization.py \
  --data "$SLEEPACCEL_DATA" \
  --old-epochs "$OLD_HR_EPOCHS" \
  --output "$SLEEPACCEL_RUN" --workers 2
```

The reference audit writes `weaver_hr_column_headers.json`,
`weaver_reference_recordings.csv`, `weaver_reference_summary.csv` and
`weaver_reference_samples.json`. The conditioning diagnostic writes
`conditioned_error_comparison.csv`.
