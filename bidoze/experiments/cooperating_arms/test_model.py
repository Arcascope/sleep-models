from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arcasleep.training.balanced import balanced_cross_entropy
from arcasleep_bidoze.model import BiDozeNetConfig
from model import ArmSpec, CooperatingModule


def fixture(shared=False, exchange=()):
    cfg = BiDozeNetConfig(num_classes=4, score_gap=True, d_model=8, n_heads=4, d_ff=16, n_transformer_layers=2,
                         conv_kernels=(3, 7), dropout=0.0, gap_from_row=True)
    spec = ArmSpec(members=3, width=8, shared_cnn=shared, cnn_width=8,
                   exchange_after=exchange, exchange_rank=3, feature_dropout=0)
    module = CooperatingModule(cfg, spec)
    x = jax.random.normal(jax.random.PRNGKey(5), (1, 9, 5, 2)).at[:, :, -1, :].set(0)
    mask = jnp.ones((1, 9))
    params = module.init(jax.random.PRNGKey(4), x, mask)["params"]
    return module, params, x, mask


@pytest.mark.parametrize("shared,exchange", [(False, ()), (False, (1,)), (True, ())])
def test_mean_and_fixed_gap(shared, exchange):
    m, p, x, mask = fixture(shared, exchange)
    x = x.at[:, -1, -1, :].set(1)
    out = m.apply({"params": p}, x, mask, return_outputs=True)
    np.testing.assert_allclose(jax.nn.softmax(out["log_mean"]),
                               out["arm_probabilities"].mean(0), atol=1e-6)
    np.testing.assert_array_equal(out["arm_probabilities"][:, :, -1, 4], 1)
    np.testing.assert_array_equal(out["arm_probabilities"][:, :, :-1, 4], 0)
    again = m.apply({"params": p}, x, mask)
    np.testing.assert_array_equal(again, out["log_mean"])


def test_objective_couples_arm_gradients():
    m, p, x, mask = fixture()
    y = jnp.zeros((1, 9), dtype=jnp.int32)
    loss = lambda p: balanced_cross_entropy(m.apply({"params": p}, x, mask), y, 5, .999)
    g = jax.grad(loss)(p)["arm_0_head_logits"]["bias"]
    changed = {**p, "arm_1_head_logits": {**p["arm_1_head_logits"],
               "bias": p["arm_1_head_logits"]["bias"].at[0].add(3)}}
    g2 = jax.grad(loss)(changed)["arm_0_head_logits"]["bias"]
    assert np.linalg.norm(g - g2) > 1e-7


def test_exchange_transmits_between_arms_only_when_enabled():
    m, p, x, mask = fixture(exchange=(1,))
    def arm0(p, enabled):
        return m.apply({"params": p}, x, mask, return_outputs=True,
                       exchange_enabled=enabled)["arm_probabilities"][0, ..., 0].sum()
    on = jax.grad(lambda p: arm0(p, True))(p)
    off = jax.grad(lambda p: arm0(p, False))(p)
    assert np.linalg.norm(on["arm_1_fwd_0"]["ff_dense_0"]["kernel"]) > 1e-8
    np.testing.assert_array_equal(off["arm_1_fwd_0"]["ff_dense_0"]["kernel"], 0)


def test_shared_cnn_gradient_is_sum_of_arm_paths():
    m, p, x, mask = fixture(shared=True)
    def probs(p): return m.apply({"params": p}, x, mask, return_outputs=True)["arm_probabilities"]
    q = probs(p)
    # Cotangent of CE(mean) held fixed; VJP linearity verifies all arm contributions.
    y = jnp.zeros((1, 9), dtype=jnp.int32)
    outer = lambda q: balanced_cross_entropy(jnp.log(q.mean(0).clip(1e-30)), y, 5, .999)
    dq = jax.grad(outer)(q)
    _, pullback = jax.vjp(probs, p)
    joint = pullback(dq)[0]["shared_cnn"]
    pieces = [pullback(jnp.zeros_like(dq).at[i].set(dq[i]))[0]["shared_cnn"] for i in range(3)]
    for a, *bs in zip(jax.tree.leaves(joint), *[jax.tree.leaves(g) for g in pieces]):
        np.testing.assert_allclose(a, sum(bs), atol=1e-6, rtol=1e-4)
    assert all(sum(float(jnp.linalg.norm(v)) for v in jax.tree.leaves(g)) > 0 for g in pieces)


def test_exchange_adds_connections_without_changing_trunk_initialization():
    m, p, x, mask = fixture()
    b = CooperatingModule(m.cfg, replace(m.spec, exchange_after=(1,)))
    q = b.init(jax.random.PRNGKey(4), x, mask)["params"]
    for root in p:
        for left, right in zip(jax.tree.leaves(p[root]), jax.tree.leaves(q[root])):
            np.testing.assert_array_equal(left, right)


def test_research_checkpoint_roundtrip(tmp_path):
    import json
    from dataclasses import asdict
    from model import CooperatingModel
    module, params, x, mask = fixture(shared=True)
    model = CooperatingModel(module.cfg, module.spec)
    model.params = params
    model.input_shape = tuple(x.shape[1:])
    model.save_weights_to_directory(tmp_path / "weights")
    (tmp_path / "config.json").write_text(json.dumps(dict(config=module.cfg.to_dict(), spec=asdict(module.spec))))
    loaded = CooperatingModel.load_research(tmp_path)
    np.testing.assert_allclose(loaded.infer(np.asarray(x)), model.infer(np.asarray(x)), atol=1e-6)


@pytest.mark.parametrize('shared,exchange', [(False,()),(False,(1,)),(True,())])
def test_prepended_absence_does_not_change_present_calls(shared, exchange):
    m,p,x,mask=fixture(shared,exchange)
    expected=m.apply({'params':p},x,mask)
    absent=jnp.zeros((1,4,5,2)).at[:,:,-1,:].set(1)
    extended=jnp.concatenate([absent,x],axis=1)
    actual=m.apply({'params':p},extended,jnp.ones(extended.shape[:2]))[:,4:]
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
