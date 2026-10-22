import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_default_matmul_precision", "highest")

from fusion_model import (ARMS, G3_PARAMETERS, StrandFusionModel, StrandFusionModule, arm_spec,
                          count_parameters, model_config)

F0_PARAMETERS = 2_194_254


def build(name):
    spec = arm_spec(name)
    module = StrandFusionModule(model_config(spec), spec)
    x = jax.random.normal(jax.random.PRNGKey(5), (1, 9, 61, 2)).at[:, :, -1, :].set(0)
    x = x.at[:, 4, -1, :].set(1)
    mask = jnp.ones((1, 9))
    params = module.init(jax.random.PRNGKey(4), x, mask)["params"]
    return module, params, x, mask


@pytest.mark.parametrize("name", list(ARMS))
def test_arm_builds_and_every_head_is_gap_exactly_where_the_row_says(name):
    module, params, x, mask = build(name)
    out = module.apply({"params": params}, x, mask, return_outputs=True)
    heads = {"output": jnp.exp(out["log_head"]), "mean": jnp.exp(out["log_mean"]),
             **{f"strand {i}": p for i, p in enumerate(out["strand_probabilities"])}}
    for label, p in heads.items():
        assert p.shape == (1, 9, 5), label
        np.testing.assert_allclose(p.sum(-1), 1, atol=1e-5)
        np.testing.assert_array_equal(p[0, 4, 4], 1)
        np.testing.assert_array_equal(np.delete(np.asarray(p[0, :, 4]), 4), 0)


@pytest.mark.parametrize("name", list(ARMS))
def test_output_is_the_mean_or_the_fusion_head(name):
    module, params, x, mask = build(name)
    out = module.apply({"params": params}, x, mask, return_outputs=True)
    np.testing.assert_allclose(jnp.exp(out["log_mean"]), out["strand_probabilities"].mean(0), atol=1e-5)
    if arm_spec(name).fusion:
        assert not np.allclose(jnp.exp(out["log_head"]), jnp.exp(out["log_mean"]), atol=1e-4)
    else:
        np.testing.assert_array_equal(out["log_head"], out["log_mean"])


@pytest.mark.parametrize("name", list(ARMS))
def test_weight_counts_are_within_two_percent_of_target(name):
    n = count_parameters(arm_spec(name))
    target = G3_PARAMETERS if name == "final3_large" else F0_PARAMETERS
    assert abs(n / target - 1) <= 0.02


def test_full_attention_strand_has_the_weights_of_a_bicausal_one():
    assert count_parameters(arm_spec("final3_mixed")) == count_parameters(arm_spec("final3"))
    module, params, x, mask = build("final3_mixed")
    a = module.apply({"params": params}, x, mask, return_outputs=True)["strand_probabilities"]
    spec = arm_spec("final3")
    b = StrandFusionModule(model_config(spec), spec).apply({"params": params}, x, mask, return_outputs=True)[
        "strand_probabilities"]
    assert not np.allclose(a[0], b[0], atol=1e-5)  # strand 0 attends differently
    np.testing.assert_allclose(a[1:], b[1:], atol=1e-5)  # the bicausal strands are unchanged


def test_fusion_loss_trains_every_strand_and_the_fusion_head():
    from arcasleep.training.balanced import balanced_cross_entropy  # noqa: F401
    spec = arm_spec("final3_fuse")
    model = StrandFusionModel(model_config(spec), spec)
    module, params, x, mask = build("final3_fuse")
    y = jnp.zeros((1, 9), dtype=jnp.int32)
    grads = jax.grad(lambda p: model._balanced_loss(p, x, mask, y, jax.random.PRNGKey(1), 5, 0.999))(params)
    for i in range(3):
        assert float(jnp.abs(grads[f"arm_{i}_head_logits"]["kernel"]).sum()) > 0
    assert float(jnp.abs(grads["fusion"]["head_logits"]["kernel"]).sum()) > 0
