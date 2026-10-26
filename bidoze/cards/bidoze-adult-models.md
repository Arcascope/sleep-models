## The two adult releases

There are two adult releases of BiDoze v0.2.0:

- [`arcasleep-bidoze-adult`](https://huggingface.co/arcascope/arcasleep-bidoze-adult):
  three networks; their outputs are averaged.
- [`arcasleep-bidoze-adult-small`](https://huggingface.co/arcascope/arcasleep-bidoze-adult-small):
  one network.

![Adult model architecture](assets/adult-architecture.png)

**How they are related.**

- All four networks have the same architecture (panel a) and were trained on
  the same recordings. Only the random seed differs.
- The `adult-small` network is seed 0 of `adult`; the weights are identical.
- Both releases use the same stage divisor and label epochs the same way. In
  `adult`, the network output used for labeling is the average of the three
  networks' outputs.

| | `adult` | `adult-small` |
|:---|---:|---:|
| Networks | 3 | 1 |
| Weights | 6,590,376 | 2,196,792 |
| `model.safetensors` size | 26.5 MB | 8.8 MB |
| Compute per recording | 3× | 1× |

**How their scores compare.** Each value is the mean over nights,
`adult` vs `adult-small`. Differences marked * are significant at p ≤ 0.05
(paired Wilcoxon over nights); unmarked differences are not.

| Cohort (nights) | Sleep/wake κ | 4-stage κ | Sleep sensitivity | Wake specificity | TST error (min) |
|:---|---:|---:|---:|---:|---:|
| SleepAccel, cross-validation (28) | 0.597 vs 0.583 | 0.435 vs 0.429 | 95.8% vs 95.5% | 74.8% vs 74.4% | −5.6 vs −6.2 |
| Wearanize+, cross-validation (92) | 0.609 vs 0.588* | 0.420 vs 0.410* | 94.5% vs 94.2% | 82.8% vs 80.9%* | −12.1 vs −12.7 |
| SleepAccel-Clinical (32) | 0.583 vs 0.574 | 0.321 vs 0.309 | 94.5% vs 95.7%* | 67.0% vs 62.5%* | +17.1 vs +24.0* |
| AAUWSS (13) | 0.533 vs 0.529 | 0.343 vs 0.357 | 95.3% vs 96.7%* | 62.4% vs 55.7%* | +6.2 vs +15.9* |
| EESM19 (33) | 0.826 vs 0.824 | 0.507 vs 0.498 | 97.8% vs 97.9% | 87.6% vs 86.8% | −0.6 vs +0.7 |

TST error is model − PSG. Neither release was trained on SleepAccel-Clinical,
AAUWSS or EESM19.

- **AAUWSS:** 13 healthy adults aged 22–71, one lab night each, Empatica E4.
- **EESM19** ([OpenNeuro ds005185](https://openneuro.org/datasets/ds005185)):
  33 nights from 16 adults, GENEActiv wrist accelerometer at 100 Hz.

- Wake specificity is higher for `adult`: by 2.0 points on Wearanize+,
  4.5 points on SleepAccel-Clinical and 6.7 points on AAUWSS (all p < 0.001).
- Total sleep time is overestimated by 17.1 minutes with `adult` vs 24.0
  minutes with `adult-small` on SleepAccel-Clinical (p < 0.001), and by 6.2
  vs 15.9 minutes on AAUWSS (p = 0.001).
- No metric differs significantly on SleepAccel (sleep/wake κ difference
  +0.015, p = 0.093) or EESM19.
- Use `adult` unless model size or compute is a constraint.
