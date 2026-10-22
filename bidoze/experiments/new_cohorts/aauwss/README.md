# AAUWSS conversion

Aalborg University Wearable Sleep Study: 13 healthy adults (ages 22–71), one
laboratory night each, Empatica E4 wrist accelerometer, full PSG scored by one
rater in 30 s epochs.

- `convert.py <raw dir> <out dir>` writes `cleaned_accelerometer/`,
  `cleaned_psg/` and `data_set.json` in the SleepAccel layout, subject ids
  `s01` to `s13`.
- `check_alignment.py <raw dir> <converted dir>` re-runs the alignment checks.

Feature cache (run from `bidoze/`):

    arcasleep process -pc config/inputs.json -dd <out dir> --name aauwss \
        -o <feature cache root> --batch-size 16

`--name aauwss` is needed because the output directory's name differs from the
dataset name.

## Decisions

- **Alignment.** E4 `ACC.csv` starts at a UTC unix time; the annotation
  workbooks carry Europe/Copenhagen wall-clock time. The converter converts the
  E4 start to Copenhagen local time (CET, +3600 s, on the 24 March nights;
  CEST, +7200 s, from 3 April) and applies no other offset. The authors'
  `aligned_sleep_data_set` PPG pickles use the same mapping: the first epoch of
  each pickle equals the E4 `BVP.csv` samples at that local time, for all 13
  subjects, to the sample.
- **Movement check.** Per-epoch accelerometer variability correlates with
  scored wake at r = 0.27 to 0.59 at lag 0 for every subject. The correlation
  peaks one epoch (30 s) away from lag 0 for s03, s06, s08 and s10; the other
  nine peak at lag 0. No per-subject shift is applied.
- **Time zero.** Output times are seconds from the first scored epoch. The E4
  starts 577 s to 6132 s before the first scored epoch; E4 samples before it are
  dropped, as are samples after the last scored epoch.
- **Units and codes.** Accelerometer values are E4 counts / 64 (g), written at
  the E4's 32 Hz. Stages: 0 Wake, 1 N1, 2 N2, 3 N3, 5 REM (no N4 in this
  annotation). Epochs the workbook marks `Missing`, and epochs absent from it
  (gaps between the EDF segments of s11 and s13), are written as −1, the PSG
  mask value.
- **Scored span, not EDF span.** The PSG is taken from the annotation
  workbooks; the EDF files are not read. Scoring starts after the EDF start.
- **Short E4 recording.** s12's accelerometer ends after about 64% of the scored
  span; `arcasleep process` crops that night to the accelerometer's end
  (573 epochs).
- **Participants 2 and 6** are 67 and 71 years old; the other 11 are 22 to 28.

— Juniper-2
