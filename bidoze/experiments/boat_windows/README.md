# Window containment and learned pooling

Six seed-0 scratch screens cross three frame layouts with mean or restored
64-feature convolution pooling. All spectra remain 10 seconds long. Original
2-second-hop frames supply starts 0,2,...,28 (native), 0,2,...,20 (contained), or
0,10,20 seconds (nonoverlap). Selecting the latter from the original grid gives
exactly the same windows as a PSG-aligned 10-second hop, without recomputation.
The original tail-frame truncation, labels and conservative vessel splits persist.

CNN arms keep 15 slots, 64 learned features, the same kernel shape and initial
parameters. Excluded slots hold the padding sentinel. The physical missing
fraction divides by 15, 11 or 3 selected windows, not by unused slots. Mean arms
share their original two-channel architecture and initialization. Mean versus CNN
has different capacity and initialization; a gain alone cannot establish that
ordered timing caused it. Follow a promising CNN effect with a within-epoch
order control and independent vessel/seed replication before a gain claim.

Common unavailable epochs across all layouts and the frozen reference are masked
in every comparison. Record actual coverage separately in the input recipe.
Scoring uses release-compatible physical-gap and stage decoding, per-night ROC
specificity at 95% sensitivity, then recording-within-person averages. The ROC
threshold is descriptive; it is not independent threshold calibration. Scoring
also describes sleep/wake-boundary-adjacent versus stable epochs. PSG defines
those diagnostic strata only; it never enters features. The frozen reference is
release seed 0, not scratch epoch 0 or the three-seed released ensemble.

Use a project environment with this checkout on PYTHONPATH. Required CLI paths
are explicit; no working-directory data assumptions. A finite queue can be run:

```sh
PYTHONPATH=core/src:bidoze/src:unet/src:bidoze/experiments python \
  bidoze/experiments/boat_windows/queue.py --python PYTHON_EXECUTABLE \
  --root DATA_ROOT --source-data PREPARED_REFERENCE \
  --cache ORIGINAL_FRAME_CACHE --run RUN_DIRECTORY --fold D --max-epochs 100
```

These are bounded discovery screens with custom cached-input checkpoints.
The matched cap does not prove convergence. If cap selection is material, extend
matched arms rather than selectively extending a favored representation.
