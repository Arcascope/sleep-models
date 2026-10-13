# Jerk divided by the time step

The adult release's jerk channel comes from `senpy.compute_jerk(use_diff=True)`, through
pisces-lite. That jerk is `‖a[i] − a[i−1]‖`; it is not divided by `t[i] − t[i−1]`. These
scripts measure how the release's input frames and calls change when it is.

The weights are unchanged; only the jerk is changed. Three jerk modes (`modes.py`):

- `release`: `‖Δa‖`, as released.
- `dt`: `‖Δa‖ / Δt`, per sample. This needs senpy's `compute_jerk(..., divide_by_dt=True)`
  (senpy branch `jerk-divide-dt`, not released).
- `median_dt`: `‖Δa‖ / median Δt`, one constant per recording.

In log-PSD, `median_dt` differs from `release` by exactly −2·ln(median Δt). Comparing `dt` with
`median_dt` therefore isolates the effect of irregular timestamps. A model retrained on `dt`
jerk would absorb the constant part but not the irregular part.

`fused_features/fused.py` takes the same modes (`plan(..., jerk=...)`); `layer_check.py`
compares the layer with the release pipeline.

## Results (2026-10-13, adult release v0.1.0)

Run: `bidoze/results/jerk-dt-2026-10-13/` (`setup/run.sh`). Nights: SleepAccel (28; two nights
at 10 Hz are refused by the release) and SleepAccel-Clinical (35). GGIR sleep filter on.

**The release's frames are not bit-reproducible.** Its JAX NUFFT gives log-PSD differences up to
0.0008 between two runs on the same night. Every comparison below is far above that.

**Layer.** `fused.frames` matches the release pipeline in all three modes: the same padding
frames and a 99.9th-percentile absolute log-PSD difference ≤ 0.002.

### Spectrograms (jerk channel; the |a| channel does not change)

`dt − release`: every cell shifts by about 2·ln(fs), which is 7.82 at 50 Hz and 8.40 at 66 Hz.
The jerk channel's scale in the release therefore depends on the sampling rate.

`dt − median_dt`, the effect of timestamp irregularity alone (absolute log-PSD difference over
frame × bin cells; median night, with the range over nights in brackets):

| Cohort | Median cell | 95th percentile | Cells changed > 1 |
|:---|--:|--:|--:|
| SleepAccel | 0.49 (0.01–0.92) | 2.2 (0.15–3.8) | 20% (0.8–46%) |
| SleepAccel-Clinical | 0.01 (0.00–0.55) | 0.11 (0.01–1.06) | 0.2% (0.0–5.1%) |

- Most SleepAccel-Clinical nights are sampled regularly. The jerk changes little beyond the
  constant.
- 11 of 28 SleepAccel nights have 25–38% of time steps more than 10% from the median. On these
  nights, the irregularity alone changes the median cell by 0.5–0.9.
- A sampling rate that changes within a night shifts the release's jerk by a constant from that
  point on. On SleepAccel-Clinical aws033, the jerk is offset by −0.8 after hour 8.4.

### Calls of the release

Share of epochs whose call (`stage`) is the same as the release's; a second release run agrees
on ≥ 99.9% of epochs on every night.

| Cohort | `dt` | `median_dt` | `dt` vs `median_dt` |
|:---|--:|--:|--:|
| SleepAccel | 81.9% (57.9–91.2%) | 83.7% | 95.5% |
| SleepAccel-Clinical | 83.7% (66.8–100%) | 83.7% | 98.1% |

About one epoch in six changes its call; nearly all of the change comes from the constant
offset, since `dt` and `median_dt` give the same call on 95.5% and 98.1% of epochs. The release
was trained on `‖Δa‖`, so jerk on another scale is outside its training inputs; this
measures the release's sensitivity to the input, not how a model trained on `dt` jerk would
score.

SleepAccel-Clinical against PSG (35 participants; mean difference from `release`, 95% CI,
Wilcoxon p):

| Metric | `release` | `dt` − `release` | p |
|:---|--:|--:|--:|
| Wake specificity | 55.4% | −5.4 (−8.2 to −2.6) | < 0.001 |
| Sleep sensitivity | 80.7% | +0.8 (−0.5 to 2.1) | 0.22 |
| κ sleep/wake | 0.453 | −0.027 (−0.047 to −0.007) | 0.006 |
| κ 4-stage | 0.252 | −0.038 (−0.058 to −0.019) | < 0.001 |
| TST error (min) | −29.6 | +5.9 (1.5 to 10.4) | 0.013 |

`median_dt` gives the same differences within 0.6 points. SleepAccel is the release's training
data, so it is not scored against PSG here.

- Feeding `dt` jerk to the released weights makes the release worse.
- The fix needs a retrain, with senpy's `divide_by_dt` in the training inputs.
