# Out-of-fold per-epoch predictions (v0.2.0 adult folds)

One parquet per model x cohort: `25h_Wearanizeplus.parquet`, `25h_SleepAccel.parquet`, `8h_Wearanizeplus.parquet`, `8h_SleepAccel.parquet`.
Each night is scored only by the fold model that held its participant out (3 seeds averaged; 15 members over the 5 folds).

- 25 h models: input is the PSG span. Probabilities are the 3-seed mean (wake, light, deep, REM; the gap-head column is dropped).
- 8 h models: whole recording in hourly 8 h windows; softmax per window, taper-weighted mean (probability average), then 3-seed mean.

## Columns
- `cohort`, `night` (night key), `participant`, `fold` (test fold of that participant)
- `epoch`: 30 s epoch index in the cohort's full feature-cache recording (same index for both models). 25 h files cover the PSG span only.
- `psg_label`: 0 wake, 1 light, 2 deep, 3 REM, -1 unlabelled
- `p_wake`, `p_light`, `p_deep`, `p_rem`; 8 h files also `p_gap`
- `gap_flag`: input-gap epoch. 8 h files also `gap_head_call` (argmax of the gap-head rule is the gap class) and `in_psg_span`
- `scored`: epoch used in the metrics (`psg_label` >= 0, not a gap; 8 h also inside the PSG span)
- `sleep_call_summed05`: True when p_light + p_deep + p_rem >= p_wake (summed 0.5)
- `stage_call_offsets`: 0 wake, 1-3 light/deep/REM; wake where `sleep_call_summed05` is False, otherwise argmax of log p plus the fold's stage offsets
  (offsets from blend-exp2a-2026-10-28 `stage_offsets.csv`, family 25h or 8h, fitted on validation participants; light fixed at 0)
- 25 h files also `stage_call_divisor`: stage within sleep by the fold's stage divisor (as section 7)
- `complete_context_26`: True for the 26 Wearanize+ nights with complete context (all columns False for SleepAccel)

## Complete-context nights
The 26 nights are the `primary26` set of tst-investigation-2026-10-28 (`per_night.csv`, scope `primary26`, model `8h_context_original`);
flagged here by `complete_context_26 == True`.

## Notes
- The fold-phase 25 h predictions end one epoch earlier than the PSG-span input for Wearanize+ nights (one epoch fewer per night); labels match epoch for epoch.
- SleepAccel feature caches equal the PSG span, so `epoch` starts at 0 for both models.
