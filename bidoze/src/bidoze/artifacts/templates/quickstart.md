## Quickstart

```bash
pip install 'arcasleep-bidoze[inference,hub]'
hf download $repo_id --local-dir $local_dir

# One CSV per recording: epoch, start_s, $columns, stage, stage_raw, is_nap, is_sib, is_spt, is_gap
# plus <name>_summary.csv: subject_id, tst_minutes, missing_data_minutes, nap_minutes
# --hypnogram also saves <name>.png (needs the [report] extra)
bidoze predict $local_dir recordings/*.csv -o predictions/ --hypnogram

# Or draw hypnograms later from the CSVs
bidoze hypnogram predictions/*.csv
```

Or from Python:

```python
from bidoze.predict import Predictor, read_accelerometer

predictor = Predictor("$local_dir")  # load once, reuse
accel = read_accelerometer("night.csv")  # (N, 4): t [s], x, y, z [g]
result = predictor.predict(accel)

result.probabilities  # (epochs, $n_stages): $columns
result.labels  # corrected stage per 30-s epoch, "gap" where no data
result.raw_labels  # raw BiDoze stage, before filtering
result.summary  # subject_id, tst_minutes, missing_data_minutes, nap_minutes
result.epoch_start_s  # epoch start times, on the input clock
result.to_csv("night_stages.csv")
result.plot("night_stages.png")  # hypnogram; needs the [report] extra
```

By default the corrected `stage` applies a GGIR-style sleep filter: sleep the
model called where the wrist was not still for 5 minutes (`is_sib` is 0) is
demoted to `wake`. Sleep outside the estimated main sleep window (`is_spt`) is
kept and flagged in `is_nap`. `stage_raw` always holds the model's own argmax.
Pass `ggir_sleep_filter=False` (or `--no-ggir-sleep-filter`) for the unfiltered
output. The hypnogram shows the raw and filtered stages, the sleep the filter
removed, and the SIB and SPT tracks.

Pass `origin=` (seconds on the input clock, or `"unix"`) to put epoch
boundaries somewhere other than the first sample, for example at the start of a
PSG recording you want to compare with.
