# Tree restart on `instinct-tree-features`

Edit directly in `sleep-models/` (`git switch instinct-tree-features`), with no worktree. Use `~/venvs/arcasleep`.

## Files
- `arcasleep_tree/loaders.py`: try-1 cohort loaders; `build(data_dir, out)` → features parquet.
- `arcasleep_tree/features.py`: per-epoch features, `INPUTS`, integer units per feature.
- `arcasleep_tree/tree.py`: `DEFINITION`, `fit(tree, table, test)`, `predict`, `load`/`save` (model.json or .safetensors), `to_monkey_c(model, app_dir)`.
- `arcasleep_tree/__init__.py`: re-exports.

Everything else from the branch is deleted (cv, report, port*, fixed, costs, breathing, configs).

## Model
- The model is a plain dict: today's `model.json` (features, scale, edges, leaf_scale, rounds) plus `trees` (feature, threshold, leaf) and `inputs`.
- `predict` runs the integer trees, so it gives the same output as the watch.
- `to_monkey_c` writes the `stagerlive` files that `make-stager-resources.py` writes now, without the golden chunks.

## CLI
`arcasleep tree process | train | predict | publish | monkey-c`

## Decisions
1. Keep the 25 Hz resampling as it is.
2. Rounds work as on the branch:
   - `fit` trains for up to 130 rounds, with early stopping (patience 20) on 15% of the training persons.
   - The final model trains for the median of the 5 person-fold best iterations, with no early stopping (`rounds=` argument).
3. Tests don't count toward the 4 files and are written last.

## Check
- Tests: save/load round trip; `predict` agrees with LightGBM.
- SleepAccel parquet has the same columns as try-1.
- `monkey-c` on the current Garmin model.json + trees.bin reproduces `generated.xml`, `edges.json` and `StagerSpec.mc`.
