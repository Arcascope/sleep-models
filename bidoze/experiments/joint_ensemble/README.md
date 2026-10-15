# Joint ensemble: training the adult model's seeds as one network

**Question.** The adult release averages the outputs of 3 separately trained BiDoze seeds. If
the 3 networks are trained together, with the class-balanced loss applied to their mean output,
does the mean stage adults better than 3 separate seeds? With 2 networks?

**Arms**, each trained on the same draw of nights (`prepare.py`):

| Arm | Fits | Loss on | Early-stopping κ of |
|:---|--:|:---|:---|
| `independent` | 3 (seeds 0–2) | each network's output | each network |
| `joint3` | 1 | the mean of 3 networks' softmax | the mean |
| `joint2` | 1 | the mean of 2 networks' softmax | the mean |

- Member m of a joint fit starts from the weights independent seed m starts from, and draws its
  own dropout. The joint and independent fits differ in the loss and in early stopping.
- Every fit uses the release recipe (`config/training-balanced.json`) and early-stopping holdout
  draw 0 (30% of the training participants).

**Draws.** Draw d (generator seed d) trains on 101 of the 201 children of the release's revised
Weaver nights and on 30 of the 60 adults of SleepAccel (28) and SleepAccel-Clinical (32), drawn
from the pooled 60 without replacement. It is scored on the other 30 adults.

**Scoring** (`score.py`): every model is decoded as the release is: its output decides wake or
sleep, and on its sleep epochs light/deep/REM are divided by the adult stage divisor of the
draw's training nights. Each joint fit's members are also scored by themselves
(`member_predictions.py`).

**Running.** The cache root holds `12Hz_log_PSD_aligned/` built with `config/inputs.json` for
SleepAccel, SleepAccel-Clinical and revised Weaver (`weaver_revised`).

```bash
RUN=<sleep-models>/bidoze/results/<run>
python prepare.py $RUN <cache root> <adult-devices run>/setup/arms.json
./run.sh $RUN                 # every draw: joint3, independent, joint2
python score.py $RUN <cache root with bidsleep labels>
```

- Peak GPU memory of one fit: 5.4 GiB for one network, 10.7 GiB for 2, 16.0 GiB for 3.
