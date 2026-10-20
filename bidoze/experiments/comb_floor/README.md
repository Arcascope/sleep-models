# Broadband background, harmonic combs and stage information

CPU investigation of four questions: background versus comb prominence, moving comb spacing, sustained small motion versus bursts, and stage differences after matching movement and comb strength.

Run from the original sleep-models checkout containing the saved feature caches and cooperating-arm predictions. Set `SLEEP_ANALYSIS_DATA` to the cleaned dataset root, then execute this directory's `run.sh`. It requires numpy, scipy, pandas, matplotlib, scikit-learn, pisces-lite and senpy from the existing arcasleep environment. No BiDoze training is run.

Outputs are under `bidoze/results/comb-floor-2026-10-20/`. `extract.py` reads all 60 adult recordings with the frozen draw-0 fit/validation/test assignment (21/9/30 participants). It retains absolute background level and adds background-relative harmonic and motion descriptors. `models.py` fits cumulative descriptor ablations and within-participant matched contrasts. `raw.py` makes raw-timestamp 30/60-second aws021 spectra and physical-amplitude checks. `figures.py` renders population, night, model and harmonic diagnostics. `stage_highres.py` examines stage-pure, movement/comb-matched 60-second raw windows, with separate cardiac-candidate and low-frequency respiratory-candidate feature sets. `representation.py` draws directional examples and checks rotation-invariant total-axis PSD. `audit.py` independently replays all saved probabilities and scores, validates labels and matched-window purity, and hashes the inputs.

Important definitions:

- Raw cache labels use the walch-like 0–5 mapping. Remap them with pisces-lite before applying scored-epoch masks. Clinical source CSVs use REM = 4; the dataset preset maps that to 5 first.
- Cached frames are 10 seconds wide at a 2-second step. They overlap. Context features use a centered 11-epoch window and are suitable for offline analysis.
- Broadband background is a descriptive lower-spectrum proxy, not an identified sensor-noise process. Frame-level and pooled-epoch flatness are different statistics.
- Harmonic spacing is searched over 0.6–2 Hz, with three harmonics, local median background subtraction and a smooth Viterbi track. A scrambled-frequency comparison gates visibility. It is a quality heuristic, not a calibrated physiological identification test. Half/double-frequency ambiguity remains.
- Raw 60-second examples use the official PSG-phase and first-retained-sample origin. The 60-second overview windows overlap at a 30-second step. Matched stage windows are separated by at least 60 seconds within each recording and have the same macro-stage over the full support.
- Native PSD physical scaling is checked using injected 1-mg sinusoids. The installed scalar jerk is the norm of successive acceleration differences, in g/sample. It is sample-rate dependent.
- `PSD_x + PSD_y + PSD_z` is rotation invariant and retains directional variation that `PSD_|a|` can suppress. Low-frequency features are respiratory candidates; no breathing reference establishes their source here.
- The matched high-resolution stage sample is a selected quiet-window population, not full-night staging. Participants were held out from fitting, but these are repeatedly examined development cohorts and user-selected examples. There is no untouched confirmation or release claim.

Read `audit.json` and the dated research log for results, denominators and uncertainty. Raw acceleration is never copied into the output run. Derived descriptors, selected spectra, coefficients, per-window predictions, input hashes and figures are retained.
