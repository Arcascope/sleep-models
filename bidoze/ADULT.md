# Adult scoring with BiDoze

The pediatric release (`arcasleep-bidoze-pediatric` v0.1.0) is trained on Weaver, a pediatric
clinical cohort. On adult PSG nights its sleep/wake calls are as good as any retrained model,
but it calls far too much deep sleep: 29.6% of scored epochs on SleepAccel (PSG: 14.7%) and
33.2% on SleepAccel-Clinical (PSG: 8.7%).

The adult formulation keeps the release's sleep/wake calls and replaces only its light/deep/REM
calls. The code is `arcasleep_bidoze.adult`.

## The formulation

An adult night is scored by two models.

1. **Wake source: the pediatric release, unchanged.**
   - Its argmax wake and gap calls are the final wake and gap calls.
   - Sleep/wake metrics (κ, sensitivity, specificity, TST, WASO) therefore equal the release's.
2. **Stage source: BiDoze trained on Weaver plus adult PSG nights.**
   - Training uses the balanced recipe, `config/training-balanced.json` (patience in optimizer
     steps, smoothed eval κ), with the loss on every class.
   - Several seeds are trained, and their 5-way outputs are averaged.
3. **Decoding.** Only epochs the wake source calls sleep are staged.
   - On those epochs the stage is the argmax of the stage source's light, deep and REM
     probabilities, each divided by the adult stage divisor:

     divisor_c = w_c × π_train,c / π_adult,c

     - w_c is the stage's effective loss weight in the class-balanced training batches.
     - π_train,c is the stage's share of the stage source's training epochs.
     - π_adult,c is its share of adult PSG epochs. BIDsleep is the reference, because it is
       not a test cohort.
   - The divisor removes the up-weighting of rare stages by the balanced loss, and the
     training cohort's stage mix.

`adult_stage_divisor` computes the divisor from the training and adult nights' stage counts.
`hybrid_calls` applies the decoding.

## Scoring recordings (beta)

An *adult bundle* is a directory with the pediatric release (`wake/`), one release bundle per
stage-source seed (`stage/seed<k>/`) and `composite.json` (the divisor and provenance). Score
recordings with it as with any release:

```bash
arcasleep bidoze predict <adult bundle> night.csv -o out/
```

- The `stage` column is the formulation's call: the release decides wake or sleep, and the
  stage source stages the release's sleep epochs.
- The `wake` probability is the release's. The `light`, `deep` and `rem` probabilities share
  1 − P(wake) in the stage source's proportions after the divisor.
- The calls are not the argmax of these probabilities: an epoch is wake when wake is the
  release's most probable stage.
- The GGIR sleep filter and the summaries work as for the release.

`experiments/adult_devices/bundle.py` builds the bundle from an arm's `final` phase: the stage
source trained on the arm's Weaver recordings and all of SleepAccel.

## Evidence

All results come from the research log, October 6–7, 2026. They are means over subjects.
SleepAccel subjects are scored by stage-source models that did not train on them (5 folds).
p: paired Wilcoxon over subjects, against the pediatric release.

| Cohort | Scoring | κ 4-stage | κ sleep/wake | Deep % |
|:---|:---|--:|--:|--:|
| SleepAccel | pediatric release | 0.397 | 0.578 | 29.6 |
| SleepAccel | adult formulation, 5-seed stage source | 0.462 (p < 0.001) | 0.578 | 20.0 |
| SleepAccel-Clinical | pediatric release | 0.261 | 0.548 | 33.2 |
| SleepAccel-Clinical | adult formulation, 5-seed stage source | 0.286 (p = 0.10) | 0.548 | 21.4 |

- The stage source in this table was trained on the 202 Weaver nights plus 4 of the 5 SleepAccel
  folds. Its divisor used Weaver's weights and stage mix.
- SleepAccel-Clinical is never in training.

Alternatives that were tested and scored lower:

- **The stage source alone**, without the release's wake calls:
  - SleepAccel sleep/wake κ 0.581 for the 5-seed ensemble vs 0.578 for the release,
    p = 0.47;
  - SleepAccel 4-stage κ 0.454, against 0.462 for the formulation.
- **A stage source trained with the loss on sleep epochs only:** SleepAccel 4-stage κ 0.420 vs
  0.462 (p = 0.016). This version also stalls more often.
- **Readouts on the release's frozen features** (relative-to-night inputs, within-night contrast,
  cohort-specific heads): none beats 3 per-class biases on the release's own output (SleepAccel
  0.418).
- **BIDsleep nights in the stage source's training data:** SleepAccel sleep/wake κ 0.530 vs
  0.563 for Weaver alone.
- **60-s spectrogram windows:** SleepAccel sleep/wake κ 0.521 vs 0.572.

## Training-data rules

- **The early-stopping holdout keeps participants whole.** A participant is the subject id up
  to its first `_`, so `00lhv171_apple_n1` and `00lhv171_fitbit_n1` are one child, and
  `Bidslab00_n1` and `Bidslab00_n2` are one adult.
- **Cross-validation folds also keep participants whole**
  (`arcasleep splits-plan --group-separator _`).
- **Scoring the stage source on an adult cohort it trained on requires folds.** SleepAccel uses
  `config/splits/sleepaccel_5fold.json`.

## Open questions

- **Which Weaver recordings to train the stage source on.** The revised Weaver data has 634
  recordings from 237 children, each recorded by 2–3 devices (ActiGraph, Apple Watch, Fitbit,
  Garmin). `experiments/adult_devices/` compares the devices, each alone and all together.
- **The divisor's adult reference.** It is BIDsleep, an adult cohort with 18.9% deep sleep.
  SleepAccel-Clinical PSG has 8.7%, so one adult prior does not fit every adult cohort.
- **Validation.** Every number above comes from one split, and from SleepAccel (28 subjects) and
  SleepAccel-Clinical (32 subjects). No adult cohort outside these has been scored.
