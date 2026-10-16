# Free-living: where a night sits in the context, and Foundry

**Question.** The adult release scores 24 h free-living recordings worse than single nights.
Every training recording is 243–1,084 epochs and the balanced recipe places each one at
epoch 0 of the 3,000-epoch context, so the release never saw data past about epoch 1,100. On
Foundry windows (09:00–09:00), where the night sits near epoch 1,500, seed 0 calls 45.5% of
Apple-scored sleep gap; moving SleepAccel-Clinical nights 1,700 epochs later costs every seed
sleep detection (research log 2026-10-15 §2). This experiment asks:

- whether placing each training recording at a random epoch of the context removes the
  position effect;
- whether adding Foundry's 24 h windows, labelled with Apple Watch sleep stages, reduces
  sleep called outside the main sleep block.

**Arms.** Every arm trains on the release's nights: the 202 matched revised Weaver nights
(`matched.json`, the adult release's `matched` arm) plus all 28 SleepAccel nights.

| Arm | Foundry windows | `segment_placement` |
|:---|:---|:---|
| `random` | none | `random` |
| `foundry` | the people not held out | `random` |
| `foundry-start` | the people not held out | `start` |
| `circular` | none | `circular` |
| `foundry-circular` | the people not held out | `circular` |
| `foundry-wakecut` | the people not held out | `wake-cut` |
| `foundry-wakecut-pair` | the people not held out | `wake-cut`, `shift_consistency_weight` 10 |
| `wakecut-pair` | none | `wake-cut`, `shift_consistency_weight` 10 |

- `segment_placement: wake-cut` places a recording that fills at most half the context as
  `random` does. A longer one, a Foundry window, is rotated by an offset drawn from those
  whose cut, at the context's edges, has no sleep within 60 epochs on either side; the
  night can then sit anywhere in the context, and no Foundry row starts or ends within 60
  epochs of sleep. Without Foundry, `wake-cut` places every night as `random` does.
- `shift_consistency_weight` places each drawn row twice, with independent offsets, and
  adds to the loss the symmetric KL divergence between the two views' outputs, aligned
  epoch by epoch, outside 60 epochs of either view's context edges. The weight is relative
  to the class-balanced cross-entropy's weight on a common class (`1 - balance_beta`).
  Each step runs the model on both views, so a phase takes about twice as long.

- `segment_placement: circular` rotates each drawn row by a uniformly drawn number of epochs,
  at every draw. A recording longer than its padding is split: its end, then the padding,
  then its start. Unlike `random`, this moves a 24 h window too: a 2,880-epoch window has
  only 120 epochs of padding, so `random` can shift it by at most 120 epochs.
- `segment_placement: random` moves each drawn recording to a uniformly drawn epoch that keeps
  it whole, at every draw; the padding that followed it fills the epochs before it. The
  early-stopping rows are placed once, from the fit seed.
- Foundry windows are the FoundryApple cache: 40 windows with ≥ 80% of the Apple-scored night
  covered by accelerometer data (Doug 13, Eric 14, Olivia 13; Whoop windows share the
  person's Apple Watch labels). Each person is one participant, so the early-stopping holdout
  keeps a person's windows on one side.
- The release itself is the baseline (adult bundle `bidoze-adult-v0.1.0`, stage source = the
  `matched/final` seeds of `bidoze-adult-devices-0`).

**Phases.** `<arm>/<person>` (eric, olivia, doug) trains without that person's windows;
`<arm>/final` trains on all of them. Arms without Foundry have only `<arm>/final`. 3 seeds per phase,
recipe `config/training-balanced.json`.

**Scoring.**
- Each person's windows are scored by the phase that held that person out (`<arm>/final`
  for the arms without Foundry).
- SleepAccel-Clinical (32) and SENSE (13, 23.4–25.0 h PSG recordings of shift workers) are
  scored by every phase; no arm trains on them.
- Foundry labels are Apple Watch staging, not PSG. A gain on Foundry measures agreement with
  Apple; SENSE and SleepAccel-Clinical measure agreement with PSG.

**Running** (AWS, from a training image of this commit):

```bash
deploy/scripts/run_task.sh --run-id free-living-0 --env BIDOZE_EXPERIMENT=experiments/free_living \
  --max-hours 3 bidoze <image> random/final foundry/eric foundry/olivia
```

`FREE_LIVING_TRAINING=config/training-smoke.json` runs a phase as a 2-epoch wiring check.
Locally: `python prepare.py <run dir> <cache root>`, then train each
`setup/cv/<arm>/cv_<phase>.json` with `setup/training-<arm>.json`.
