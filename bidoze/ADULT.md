# Adult scoring with BiDoze

The pediatric release (`arcasleep-bidoze-pediatric` v0.1.0) is trained on Weaver, a pediatric
clinical cohort. On adult PSG nights its sleep/wake calls hold up, but it calls far too much
deep sleep: 29.6% of scored epochs on SleepAccel (PSG: 14.7%) and 33.2% on SleepAccel-Clinical
(PSG: 8.7%).

The adult model, `arcasleep-bidoze-adult`, is BiDoze trained on Weaver plus adult PSG nights,
with an adult stage divisor applied when it labels sleep epochs. Its card is
`cards/bidoze-adult.md`; the code is `arcasleep_bidoze.adult`.

## The formulation

1. **One architecture, several seeds.** BiDoze is trained with the balanced recipe,
   `config/training-balanced.json` (patience in optimizer steps, smoothed eval κ), with the loss
   on every class. Three seeds are trained; their 5-way outputs are averaged.
2. **Sleep or wake:** an epoch is wake when wake is the most probable of wake, light, deep and
   REM in the averaged output.
3. **Light, deep or REM**, on sleep epochs only: the argmax of the light, deep and REM
   probabilities, each divided by the adult stage divisor:

   divisor_c = w_c × π_train,c / π_adult,c

   - w_c is the stage's effective loss weight in the class-balanced training batches.
   - π_train,c is the stage's share of the training epochs.
   - π_adult,c is its share of the epochs of BIDSleep, an adult cohort that is not a
     test cohort.
   - The divisor removes the up-weighting of rare stages by the balanced loss, and the training
     cohort's stage mix.

`adult_stage_divisor` computes the divisor from the training and adult nights' stage counts.
`hybrid_calls(mean, mean, divisor)` applies steps 2 and 3. Given a second model's output as its
first argument, it takes the sleep/wake decision from that model instead, as the research
comparisons do.

**Wake is not divided.** Dividing all four probabilities, which is the same as subtracting
log(divisor) from the logits, moves sleep/wake to 97–99% sleep sensitivity and 50–62% wake
specificity on SleepAccel. The divisor's wake entry depends on the adult reference's share of
wake, which differs between cohorts.

## Scoring recordings

An *adult bundle* is a directory with one release bundle per seed (`seed<k>/`),
`composite.json` (the divisor and provenance) and the card as `README.md`. Score recordings with
it as with any release:

```bash
arcasleep bidoze predict <adult bundle> night.csv -o out/
```

- The seeds run as one batched call (`StackedModel`, `jax.vmap` over the stacked weights).
- The `stage` column comes from steps 2 and 3, so it is not always the largest probability.
- The `wake` probability is the seeds' mean. The `light`, `deep` and `rem` probabilities share
  1 − P(wake) in proportion to their divided probabilities.
- The GGIR sleep filter and the summaries work as for the release.

`experiments/adult_devices/bundle.py` builds the bundle from an arm's `final` phase, which trains
on the arm's Weaver recordings and all of SleepAccel. `experiments/adult_devices/card_metrics.py`
computes the card's metrics.

## Evidence

From the research log, October 6–8, 2026. Means over subjects; p: paired Wilcoxon over subjects,
against the pediatric release. SleepAccel is cross-validated: each subject is scored by the
3-seed fold models trained without its fold. SleepAccel-Clinical is never in training and is
scored by the released model.

| Cohort | Model | κ 4-stage | κ sleep/wake | Wake spec. at 95% sens. | Deep % (PSG) |
|:---|:---|--:|--:|--:|--:|
| SleepAccel | pediatric release | 0.397 | 0.578 | 69.1% | 29.6 (14.7) |
| SleepAccel | adult, v0.1.0 recipe | 0.437 (p = 0.060) | 0.562 (p = 0.13) | 68.2% | 21.5 |
| SleepAccel-Clinical | pediatric release | 0.261 | 0.548 | 54.0% | 33.2 (8.7) |
| SleepAccel-Clinical | adult v0.1.0 | 0.302 (p = 0.011) | 0.555 (p = 0.90) | 53.9% | 21.8 |

- v0.1.0 trains on 202 revised-Weaver recordings, the same nights as the earlier Weaver cache
  (111 Fitbit, 70 Apple Watch, 21 ActiGraph), plus SleepAccel.
- **The same model, scored with the pediatric release's sleep/wake calls**, gives the same
  4-stage κ: 0.441 on SleepAccel (p = 0.05 against 0.437) and 0.295 on SleepAccel-Clinical
  (p = 0.19). The pediatric release is not needed.
- One early-stopping holdout draw moves SleepAccel 4-stage κ by about 0.02. The earlier Weaver
  cache gave 0.460 with a different draw.

Alternatives that were tested and scored lower or the same:

- **One seed instead of three:** SleepAccel 4-stage κ 0.402–0.431 for the three v0.1.0-recipe
  seeds, against 0.437 for their mean.
- **Training only on ActiGraph recordings of the same nights:** SleepAccel 4-stage κ 0.414 vs
  0.441 with the pediatric release's sleep/wake calls (p = 0.048).
- **Fine-tuning all of the pediatric release's weights on 4/5 SleepAccel:** early stopping on
  pediatric or adult nights stops it within 16 epochs. Its sleep/wake shifts toward wake (94.5%
  sleep sensitivity) at the same 95%-sensitivity wake specificity; 4-stage κ 0.422.
- **A stage source trained with the loss on sleep epochs only:** SleepAccel 4-stage κ 0.420 vs
  0.462 (p = 0.016). This version also stalls more often.
- **Readouts on the pediatric release's frozen features** (relative-to-night inputs,
  within-night contrast, cohort-specific heads): none beats 3 per-class biases on the release's
  own output (SleepAccel 0.418).
- **BIDSleep nights in the training data:** SleepAccel sleep/wake κ 0.530 vs 0.563 for Weaver
  alone.
- **60-s spectrogram windows:** SleepAccel sleep/wake κ 0.521 vs 0.572.

## Training-data rules

- **The early-stopping holdout keeps participants whole.** A participant is the subject id up
  to its first `_`, so `child01_apple_n1` and `child01_fitbit_n1` are one child, and
  `adult01_n1` and `adult01_n2` are one adult.
- **Cross-validation folds also keep participants whole**
  (`arcasleep splits-plan --group-separator _`).
- **Scoring on an adult cohort the model trained on requires folds.** SleepAccel uses
  `config/splits/sleepaccel_5fold.json`.

## Open questions

- **The divisor's adult reference.** It is BIDSleep, an adult cohort with 18.9% deep sleep.
  SleepAccel-Clinical PSG has 8.7%, so one adult prior does not fit every adult cohort.
- **One network instead of three.** A student trained on the three seeds' mean output could
  keep most of their gain over one seed.
- **Validation.** Every number above comes from SleepAccel (28 subjects) and SleepAccel-Clinical
  (32 subjects). No other adult cohort has been scored.
