# Wearanize+ conversion

Converts the Wearanize+ open-access cohort (Empatica E4 wrist accelerometer
with manual PSG scores) into the arcasleep input layout, ready for
`arcasleep process`.

    python convert.py RAW_ROOT SCRATCH_DIR OUT_DIR
    python checks.py OUT_DIR

The raw folder is only read; Empatica `ACC.csv` files are extracted into
`SCRATCH_DIR`. `OUT_DIR` holds `cleaned_accelerometer/`, `cleaned_psg/`,
`data_set.json` and `conversion_manifest.json` (status and drop reason per
night), plus the side files `scorer2/` and `usleep/`. The dataset name is
`wearanize`; subject ids are `sub001_s1` (participant, session).

## Decisions

- **Device.** Only the wrist E4 accelerometer is used (32 Hz). ZMax, PSG signals and ActivPAL are ignored.
- **Units.** E4 ACC is stored in 1/64 g; the files are divided by 64 and written in g.
- **Time.** Time zero of every file is the ZMax lights-out moment; the unit is seconds.
- **Window.** Only the lights-out to lights-on window is kept (`Zmax_dur_sec`). Scorer 1 and U-Sleep also scored the time outside it; those epochs are left out.
- **Synchronization.** A device sample or epoch at device time `d` is at window time `d - start_sec`, where `start_sec` is the device's start value in the sync sheet. A negative start means the device missed lights-out, so the first window seconds have no data for it. A positive `PSG_offset` means the PSG ended before lights-on, so scores end early. In both cases the missing span is simply absent, not filled.
- **Sample times.** The README says counting starts at 1, with the first data point at 1 + 1/fs s; E4 sample `i` (from 0) is placed at `1 + (i + 1)/32` s.
- **Epoch grid.** Scorer 1 epoch `k` starts at PSG time 30 k, so in window time it starts at `30 k - PSG_start_sec`. These times are not multiples of 30 and are written as they are; arcasleep's alignment handles any phase.
- **Labels.** `y` is scorer 1. Codes: 0 wake, 1 N1, 2 N2, 3 N3, REM 4 becomes 5 (SleepAccel convention). Artifact epochs (−1) stay −1, like SleepAccel's unscored epochs. Epochs with no E4 sample are also −1.
- **Scorer 2.** Written to `scorer2/` with the same stage codes. Scorer 2 scored the window itself, so its epoch `j` is placed at window time 30 j. This is inferred from epoch counts and from agreement with scorer 1; it is accurate to within one epoch.
- **U-Sleep.** Written to `usleep/` on scorer 1's epoch grid (it scored the whole recording, like scorer 1).
- **Dropped nights.** A night is dropped when the sync sheet gives `0` (unavailable) or `-999` (not synchronizable) for the PSG or the E4, or when it has no scorer 1 file. The reasons are in `conversion_manifest.json`.
- **Sessions.** Every participant has one night (`s1`).

## Checks

`checks.py` reports, for the converted nights:

- The AUROC of E4 movement for scorer 1 wake vs sleep at epoch lags −10 to +10, and each night's peak lag. Movement lines up with scored wake at lags −1 to 0.
- Cohen's kappa of scorer 1 against scorer 2 and against U-Sleep (4-stage: wake, light = N1 + N2, deep = N3, REM; and sleep/wake), as the mean over nights, on epochs both score.
- Stage proportions.

## Demographics and license

The demographic sheet has age, birth year, gender and handedness only. There
are no questionnaire or clinical data in the open-access release. The cohort
is described as healthy adults, aged 18 to 39.

The dataset's repository is MIT licensed; its page states scientific research use only
and forbids attempts to identify participants. Check the terms before redistributing.

— Hazel
