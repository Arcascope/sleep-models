# Revised Weaver: which devices to train the adult stage source on

**Question.** Revised Weaver records each child with 2–3 devices on the same night, 634
recordings from 237 children. The adult stage source of the [adult formulation](../../ADULT.md)
can train on one device or on all of them. This experiment asks:

- whether training on every device scores adult stages better than one device;
- whether any single device does better than the others.

**Arms.** Each arm trains on its revised Weaver recordings plus 4 of the 5 SleepAccel folds.

| Arm | Revised Weaver recordings | Children |
|:---|--:|--:|
| `all` | 634 | 237 |
| `actigraph` | 238 | 236 |
| `actigraph-119` | 120 | 119 |
| `apple-119` | 120 | 119 |
| `fitbit-119` | 119 | 119 |
| `garmin-119` | 121 | 119 |
| `matched` | 202 | 201 |
| `matched-actigraph` | 201 | 200 |
| `earlier` | 202 Weaver nights | 202 |
| `weaver-2026-10-06` (reference, already trained) | 202 Weaver nights | 202 |

- The `-119` arms draw 119 children per device (seed 0), the number of children with a Fitbit
  recording. These four arms differ in device and in which children are included, not in
  training-set size.
- `actigraph` against `actigraph-119` shows the effect of training-set size for one device.
- The reference arm is the research log's October 6 §2 mix runs, seeds 0–2.
- `matched` is the reference's Weaver nights as they appear in revised Weaver: 111 Fitbit, 70
  Apple and 21 ActiGraph recordings, found by identical PSG labels and spectra. Their labels and
  spectra are identical to the earlier cache, so `matched` should reproduce the reference within
  seed variation.
- `matched-actigraph` keeps those children and nights but uses each one's ActiGraph recording.
- `earlier` trains on the earlier `weaver` cache itself, as the reference did.

**Recipe.**
- `config/training-balanced.json` with `run_seed 0, seed_count 3`.
- Folds: `config/splits/sleepaccel_5fold.json`.
- The early-stopping holdout keeps every recording of a participant on one side.
- Phase `<arm>/fold<k>` draws the holdout with split seed 0; phase `<arm>/split<s>/fold<k>`
  draws it with split seed s. Repeating an arm over split seeds measures how much the holdout
  draw moves the scores.

**Scoring** (`score.py`): mean over subjects; p: paired Wilcoxon over subjects.
- SleepAccel subjects are scored by the fold model that held them out (28 subjects).
- SleepAccel-Clinical subjects (32) are scored by the mean output of the 5 fold models.
- Each arm is scored two ways:
  - `alone`: the stage source's own argmax;
  - `adult`: the pediatric release's wake calls with the stage source's light/deep/REM, divided
    by the adult stage divisor.
- Seeds are scored singly and as the 3-seed mean output (`ensemble`).
- The divisor is computed per fold model from that model's own training nights. October 6 §2
  instead used Weaver's weights for every fold model.

**Running.** Run from this checkout. Cache roots are the directories that contain
`12Hz_log_PSD_aligned/`.

```bash
RUN=<sleep-models>/bidoze/results/bidoze-adult-devices-0
python prepare.py $RUN <root with weaver_revised> <root with SleepAccel, SleepAccel-Clinical, bidsleep, weaver>
./run.sh $RUN            # 6 arms x 5 folds, 3 seeds per call; finished folds are skipped
python score.py $RUN <root with SleepAccel, SleepAccel-Clinical, bidsleep, weaver>
```

- `score.py` reads the pediatric release's predictions and the October 6 §2 runs from the results
  directory next to `$RUN`.
- On an RTX 5090, a 3-seed fold of `all` takes about 30 minutes; the whole queue is projected
  at about 10 hours.
