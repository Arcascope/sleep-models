# EESM19 conversion

Converts EESM19 (OpenNeuro ds005185: 20 participants, 4 nights each, ear-EEG
and partial PSG scored manually by two raters) into the arcasleep input layout,
using the wrist GENEActiv accelerometer, ready for `arcasleep process`.

    python convert.py RAW_ROOT OUT_DIR
    python checks.py OUT_DIR

The raw folder is only read. `OUT_DIR` holds `cleaned_accelerometer/`,
`cleaned_psg/`, `data_set.json` and `conversion_manifest.json` (status, drop
reason and alignment values per night), plus the side file `scorer2/`. The
dataset name is `eesm19`; subject ids are `sub001_s1` (participant, session).

Feature cache (run from `bidoze/`):

    arcasleep process -pc config/inputs.json -dd <out dir> --name eesm19 \
        -o <feature cache root> --batch-size 8

## Decisions

- **Device.** Only the GENEActiv wrist accelerometer is used, from the `.csv` export (100 Hz, g). Ear-EEG and PSG signals are used only for alignment.
- **Time zero.** Time zero of every file is the start of the first epoch scored by scorer 1; the unit is seconds. Accelerometer samples before it and after the last scored epoch are dropped.
- **Labels.** `y` is scorer 1 (the BIDS `acq-scoring1` events). Codes: 0 wake, 1 N1, 2 N2, 3 N3, REM becomes 5 (SleepAccel convention). Artifact, "A" and unscored epochs are −1. Epochs with no accelerometer sample are also −1.
- **Scorer 2.** Written to `scorer2/` with the same codes, on the same clock as scorer 1.
- **Alignment.** The watch and EDF clocks differ, so each night is aligned with the movement shake at the start of the recording. The EDF's first 30 minutes (high-passed power, robust z-scored, mean over channels) are correlated with the watch's movement (log summed absolute change of acceleration magnitude), within ±2 h of the offset the two clock times predict.
- **Alignment rule.** A night is converted when the best correlation r ≥ 0.3 and one of these holds:
  - peak z-score ≥ 8 against correlations elsewhere in the search window;
  - peak z-score ≥ 4 and the clock difference is within 30 s of 5 s (the median over converted nights is 4.8 s).
- **Dropped nights.** A night is dropped when the shake alignment does not meet the rule, when the night has no scorer 1 file, actigraphy csv or EDF, or when the watch covers under half of the scored epochs. Reasons are in `conversion_manifest.json`.
- **Clock differences.** 31 of 33 converted nights differ by 0 to 24 s. The other two differ by 66 s (z = 8.9) and 540 s (z = 13.1).
- **Sessions.** Each participant has up to four nights (`s1` to `s4`); each is its own recording.

## Checks

`checks.py` reports, for the converted nights:

- The shake peak r and z ranges and the clock differences from the manifest.
- The AUROC of wrist movement for scorer 1 wake vs sleep at epoch lags −10 to +10, and each night's peak lag. Movement lines up with scored wake at lag 0.
- Cohen's kappa of scorer 1 against scorer 2 (4-stage: wake, light = N1 + N2, deep = N3, REM; and sleep/wake).
- Stage proportions.

## License

The dataset's `dataset_description.json` gives the license as CC0.

— Linden
