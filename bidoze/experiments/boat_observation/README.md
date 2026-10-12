# Frozen observation controls

This confidential experiment compares the same nights with frozen release weights.
`sampling.py` builds native, every-third-sample, low-pass native and low-pass decimated
inputs for SURF; Somerset uses native and low-pass native. A 6/8 Hz pass/stop Kaiser FIR
filters axes before magnitude and difference-norm jerk, without interpolation across gaps.
All variants share original PSG, still-wake definitions, GGIR masks and common coverage.

The three release seeds are averaged. Native inputs and predictions are checked against
frozen preparation, allowing NUFFT low-power numerical error and float32 rounding of gap
fractions; coverage masks must match exactly. Results are descriptive frozen-model
perturbations, not a causal estimate of an isolated device floor or a retraining result.

Artifacts retain per-axis linear PSD, axis means and covariance, model log-PSD inputs,
predictions, per-night metrics and literal protocol/bundle hashes. `axis_diagnostics.py`
adds 32 native Clinical spectral reference nights on their source PSG clock and summarizes
still-wake/sleep levels. Lower percentiles are observed levels, not sensor-only floor truth.

Supply raw data, prepared bundle/inventory directory and output directory explicitly:

```bash
python sampling.py DATA_ROOT PREPARED_DATA RUN_DIR --calibration
python axis_diagnostics.py DATA_ROOT PREPARED_DATA RUN_DIR --prepare-land
```

Register the output directory in results-s3-sync confidential-runs.txt before syncing.
Publish this code only to confidential-work and artifacts only to the confidential bucket.
