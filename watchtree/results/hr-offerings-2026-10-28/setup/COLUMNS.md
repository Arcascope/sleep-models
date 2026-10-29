# Columns of the HR offerings run

Arms (rows of every table): `nohr` (62 columns, accelerometer-only baseline), `nohr_ahr` (82 = 62 + 20;
still no HR sensor), `hr` (87 columns, the HR baseline), `hr_bpm` (105 = 87 + 18),
`hr_bpm_ahr` (125 = 105 + 20).

## BPM columns (from the device HR stream)

Each window of W min ends at the end of the scored epoch (half-open [end - W, end)); no sample at or after the end of the
epoch is used. No clock-time or time-since-start input exists; the level baseline is a fixed 2 h lookback.
Window usable: >= 80% coverage (n x dt / W) and no gap > max(15 s, 3 dt), gaps counted to both window edges, dt = the night's median
sampling interval; unusable windows give NaN in every column except `bpm_valid_W` (0).

| column | arms | definition |
|---|---|---|
| `bpm_midlogp_2m` | none (2-min columns are not in any arm) | ln(power 0.04 Hz to MID_HI + 1e-3), NUFFT PSD of the window |
| `bpm_rmssd_2m` | none (2-min columns are not in any arm) | RMS of successive differences (consecutive samples) |
| `bpm_npairs_2m` | none (2-min columns are not in any arm) | number of successive-difference pairs used |
| `bpm_sampen_2m` | none (2-min columns are not in any arm) | sample entropy (m = 2, r = 0.2 SD) of the longest run of consecutive samples |
| `bpm_slope_2m` | none (2-min columns are not in any arm) | least-squares slope, bpm per min |
| `bpm_level_2m` | none (2-min columns are not in any arm) | window mean minus the lowest usable 10-min mean over the preceding 2 h (fixed 240-epoch lookback) |
| `bpm_valid_2m` | none (2-min columns are not in any arm) | 1 if the window has >= 80% coverage and no gap > max(15 s, 3 dt), else 0 |
| `bpm_lowlogp_5m` | hr_bpm, hr_bpm_ahr | ln(power 0.01-0.04 Hz + 1e-3), NUFFT PSD of the window |
| `bpm_midlogp_5m` | hr_bpm, hr_bpm_ahr | ln(power 0.04 Hz to MID_HI + 1e-3), NUFFT PSD of the window |
| `bpm_lowratio_5m` | hr_bpm, hr_bpm_ahr | low / (low + mid + 1e-3) |
| `bpm_rmssd_5m` | hr_bpm, hr_bpm_ahr | RMS of successive differences (consecutive samples) |
| `bpm_npairs_5m` | hr_bpm, hr_bpm_ahr | number of successive-difference pairs used |
| `bpm_sampen_5m` | hr_bpm, hr_bpm_ahr | sample entropy (m = 2, r = 0.2 SD) of the longest run of consecutive samples |
| `bpm_slope_5m` | hr_bpm, hr_bpm_ahr | least-squares slope, bpm per min |
| `bpm_level_5m` | hr_bpm, hr_bpm_ahr | window mean minus the lowest usable 10-min mean over the preceding 2 h (fixed 240-epoch lookback) |
| `bpm_valid_5m` | hr_bpm, hr_bpm_ahr | 1 if the window has >= 80% coverage and no gap > max(15 s, 3 dt), else 0 |
| `bpm_lowlogp_10m` | hr_bpm, hr_bpm_ahr | ln(power 0.01-0.04 Hz + 1e-3), NUFFT PSD of the window |
| `bpm_midlogp_10m` | hr_bpm, hr_bpm_ahr | ln(power 0.04 Hz to MID_HI + 1e-3), NUFFT PSD of the window |
| `bpm_lowratio_10m` | hr_bpm, hr_bpm_ahr | low / (low + mid + 1e-3) |
| `bpm_rmssd_10m` | hr_bpm, hr_bpm_ahr | RMS of successive differences (consecutive samples) |
| `bpm_npairs_10m` | hr_bpm, hr_bpm_ahr | number of successive-difference pairs used |
| `bpm_sampen_10m` | hr_bpm, hr_bpm_ahr | sample entropy (m = 2, r = 0.2 SD) of the longest run of consecutive samples |
| `bpm_slope_10m` | hr_bpm, hr_bpm_ahr | least-squares slope, bpm per min |
| `bpm_level_10m` | hr_bpm, hr_bpm_ahr | window mean minus the lowest usable 10-min mean over the preceding 2 h (fixed 240-epoch lookback) |
| `bpm_valid_10m` | hr_bpm, hr_bpm_ahr | 1 if the window has >= 80% coverage and no gap > max(15 s, 3 dt), else 0 |

`bpm_lowlogp_2m` and `bpm_lowratio_2m` are not defined (0.01 Hz needs 3 cycles = 300 s).

### Rules by cohort (NaN, not zero)

| stream | low band 0.01-0.04 Hz | mid band | successive differences | sample entropy |
|---|---|---|---|---|
| Empatica E4 (Wearanize+, AAUWSS), 1 s, low-passed above 0.04 Hz (measured, log section 14) | yes | NaN | NaN | NaN |
| SleepAccel (Apple Watch HR, median 5 s, Nyquist 0.10 Hz) | yes | **0.04-0.10 Hz** (MID_HI = 0.10, the cohort's Nyquist) | consecutive samples at most max(3 s, 1.5 dt) = 7.5 s apart | longest run of consecutive samples, >= max(30, ceil(100 s / dt)) = 30 samples |
| Weaver Garmin, Fitbit (1 s) | yes | 0.04-0.15 Hz | consecutive samples <= 3 s apart | longest run >= 100 samples |
| accelerometer-derived series (`ahr_`, 5 s steps, Nyquist 0.10 Hz) | yes | 0.04-0.10 Hz | as SleepAccel | as SleepAccel |

`bpm_midlogp_*` is therefore a power over a different band on SleepAccel (0.04-0.10 Hz) than on Weaver (0.04-0.15 Hz); `bpm_rmssd_*` is
over successive samples 5 s apart on SleepAccel, 1 s apart on Weaver. A sample-entropy run of 30 samples needs 150 s of consecutive 5 s samples.

## Accelerometer-derived HR columns (`ahr_`)

Wrist accelerometer on the 25 Hz stream, causal forward 4th-order Butterworth band-pass 0.67-3.0 Hz (40-180 bpm) per axis;
senpy NUFFT PSD (Hann taper, window mean removed) per axis over the window [e - 30 s, e), axes summed; invalid (gap) samples dropped,
windows with < 80% valid samples give NaN. One estimate every 5 s (window ends 30 + 5 j s); the epoch's `ahr_bpm`, `ahr_q` are the
window ending at the epoch end. Peak frequency refined by a parabola through the three bins around the peak.
The `ahr_` variability columns apply the `bpm_` definitions to the 5 s estimate series (estimate stamped 1 ms before its window end).

| column | arms | definition |
|---|---|---|
| `ahr_bpm` | nohr_ahr, hr_bpm_ahr | 60 x peak frequency (Hz) of the summed-axis NUFFT PSD of the band-passed accelerometer in 0.67-3.0 Hz, window of 30 s ending at the epoch end |
| `ahr_q` | nohr_ahr, hr_bpm_ahr | peak-bin power / total power in 0.67-3.0 Hz, same window |
| `ahr_lowlogp_5m` | nohr_ahr, hr_bpm_ahr | ln(power 0.01-0.04 Hz + 1e-3), NUFFT PSD of the window; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_midlogp_5m` | nohr_ahr, hr_bpm_ahr | ln(power 0.04 Hz to MID_HI + 1e-3), NUFFT PSD of the window; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_lowratio_5m` | nohr_ahr, hr_bpm_ahr | low / (low + mid + 1e-3); computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_rmssd_5m` | nohr_ahr, hr_bpm_ahr | RMS of successive differences (consecutive samples); computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_npairs_5m` | nohr_ahr, hr_bpm_ahr | number of successive-difference pairs used; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_sampen_5m` | nohr_ahr, hr_bpm_ahr | sample entropy (m = 2, r = 0.2 SD) of the longest run of consecutive samples; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_slope_5m` | nohr_ahr, hr_bpm_ahr | least-squares slope, bpm per min; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_level_5m` | nohr_ahr, hr_bpm_ahr | window mean minus the lowest usable 10-min mean over the preceding 2 h (fixed 240-epoch lookback); computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_valid_5m` | nohr_ahr, hr_bpm_ahr | 1 if the window has >= 80% coverage and no gap > max(15 s, 3 dt), else 0; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_lowlogp_10m` | nohr_ahr, hr_bpm_ahr | ln(power 0.01-0.04 Hz + 1e-3), NUFFT PSD of the window; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_midlogp_10m` | nohr_ahr, hr_bpm_ahr | ln(power 0.04 Hz to MID_HI + 1e-3), NUFFT PSD of the window; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_lowratio_10m` | nohr_ahr, hr_bpm_ahr | low / (low + mid + 1e-3); computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_rmssd_10m` | nohr_ahr, hr_bpm_ahr | RMS of successive differences (consecutive samples); computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_npairs_10m` | nohr_ahr, hr_bpm_ahr | number of successive-difference pairs used; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_sampen_10m` | nohr_ahr, hr_bpm_ahr | sample entropy (m = 2, r = 0.2 SD) of the longest run of consecutive samples; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_slope_10m` | nohr_ahr, hr_bpm_ahr | least-squares slope, bpm per min; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_level_10m` | nohr_ahr, hr_bpm_ahr | window mean minus the lowest usable 10-min mean over the preceding 2 h (fixed 240-epoch lookback); computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |
| `ahr_valid_10m` | nohr_ahr, hr_bpm_ahr | 1 if the window has >= 80% coverage and no gap > max(15 s, 3 dt), else 0; computed on the accelerometer-derived series (one estimate per 5 s, every estimable window) |

Feature selection: none. Every column of an arm goes to LightGBM; early stopping uses the BiDoze eval subjects of the training
side, and stage offsets are fitted on the training subjects only (3 inner person folds).
