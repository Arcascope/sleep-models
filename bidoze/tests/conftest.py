import jax

# GPU matmuls default to reduced precision (errors near 1e-4); comparisons at 1e-5 between a batched and a
# separate computation need full precision.
jax.config.update("jax_default_matmul_precision", "highest")
