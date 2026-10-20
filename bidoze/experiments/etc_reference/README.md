# ETC references and smooth logistic heads

These CPU analyses add an ETC reference to the saved October 15–19 BiDoze experiments and fit temporal linear heads. Run from the original sleep-models checkout containing `bidoze/results/`, with the adjacent `sleep-models-tridoze` checkout and its saved draw caches available.

```sh
set -e
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python ../sleep-models-etc-reference/bidoze/experiments/etc_reference/run.py
python ../sleep-models-etc-reference/bidoze/experiments/etc_reference/refine.py
python ../sleep-models-etc-reference/bidoze/experiments/etc_reference/finish.py
```

Outputs are written to `bidoze/results/etc-reference-2026-10-20/`. `run.py` computes the historical and linear-PSD ETC references and the first TV sweep; `refine.py` continues spectral optimization and fits reflection-tied activity heads; `finish.py` produces final plots and audits the label mapping, feature identity and numerical scoring contracts. Read `audit.json` before using the results.

`etc_existing.py` is copied unchanged from public sleep-models commit `7e18ba76dbeb9ff3d6764e694e5bb821fae5bd61`, `core/src/arcasleep/etc.py`. Its default Gaussian has 3 epochs and SD 0.5 epochs. Both the original score (sum of cached log-PSD) and a linear-power score use that implementation unchanged. Thresholds are selected on validation participants. Raw PSG labels are remapped with pisces-lite before any masking or fitting.

The free 31-epoch heads use class-balanced cross entropy, small L2 regularization and smoothed absolute first differences across time; spectral heads also penalize adjacent frequencies. The reflection-tied 15-epoch models explicitly assume time symmetry and add a squared-difference penalty. Gaussian curves are fitted afterward for shape diagnostics; no logistic kernel is parameterized as a Gaussian.

The six three-class plots show individual kernels in a sum-to-zero gauge and each requested A − (B + C) contrast. Those contrasts equal 2A in this gauge. True one-versus-rest softmax log odds have a state-dependent derivative. Feature standardization is undone before plotting; linear-PSD derivatives use the chain rule at a stated reference spectrum. All scoring and shape conclusions remain development results.
