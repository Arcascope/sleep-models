"""The TriDoze candidates build, train a step and keep their outputs on present epochs when
absent epochs are added before the recording."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arcasleep_bidoze.joint import JointEnsemble
from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel

T, F, C = 96, 7, 2


def small_config(**changes) -> BiDozeNetConfig:
    base = dict(num_classes=4, score_gap=True, gap_from_row=True, d_model=32, n_heads=4,
                n_transformer_layers=1, d_ff=32, conv_kernels=[3, 7], dropout=0.0)
    return BiDozeNetConfig.from_dict({**base, **changes})


def recording(rng: np.random.Generator, n: int, total: int, lead: int = 0) -> np.ndarray:
    """``(total, F + 1, C)``: ``n`` present epochs after ``lead`` absent ones, absent after."""
    x = np.tile(rng.normal(size=(1, F + 1, C)).astype(np.float32), (total, 1, 1))
    x[:, -1, :] = 1.0
    x[lead:lead + n] = rng.normal(size=(n, F + 1, C)).astype(np.float32)
    x[lead:lead + n, -1, :] = 0.0
    return x


def models():
    return [
        BiDozeNetModel(small_config()),
        BiDozeNetModel(small_config(architecture="roles")),
        JointEnsemble(small_config(), 3, member_loss_weight=0.5),
    ]


@pytest.mark.parametrize("model", models(), ids=["bidoze_g", "roles", "joint"])
def test_absent_epochs_before_the_recording_change_nothing(model):
    rng = np.random.default_rng(0)
    night = recording(rng, 40, T)
    shifted = np.tile(night[-1:], (T, 1, 1))
    shifted[30:70] = night[:40]
    model._ensure_built((T, F + 1, C), seed=0)
    a = np.asarray(jax.nn.softmax(model.infer(night[None])[0], axis=-1))
    b = np.asarray(jax.nn.softmax(model.infer(shifted[None])[0], axis=-1))
    np.testing.assert_allclose(a[:40], b[30:70], atol=1e-5)
    assert np.all(a[:40, 4] < 1e-6) and np.all(a[40:, 4] > 1 - 1e-6)


@pytest.mark.parametrize("model", models(), ids=["bidoze_g", "roles", "joint"])
def test_a_balanced_fit_runs(model):
    rng = np.random.default_rng(1)
    x = np.stack([recording(rng, 60, T), recording(rng, 50, T)])
    y = np.full((2, T), -2, dtype=np.int32)
    y[0, :60] = rng.integers(0, 4, 60)
    y[1, :50] = rng.integers(0, 4, 50)
    from arcasleep.training.balanced import Patience

    history = model.fit_balanced(
        x, y, x, y, batch_size=2, initial_lr=1e-3, weight_decay=0.0, balance_beta=0.999,
        patience=Patience(lr_drop=10, early_stop=20, unit="steps"), max_epochs=2,
        fit_seed=0, max_train_seconds=0.0,
    )
    assert np.isfinite(history["best_eval_kappa"]) or history["best_epoch"] >= 0


def test_role_probabilities_sum_to_one():
    from arcasleep_bidoze.model.definition import role_log_probabilities

    z = jnp.asarray(np.random.default_rng(2).normal(size=(3, 10)))
    total = jnp.exp(role_log_probabilities(z[0], z[1], z[2])).sum(-1)
    np.testing.assert_allclose(np.asarray(total), 1.0, atol=1e-6)


@pytest.mark.parametrize("make", [lambda: BiDozeNetModel(small_config()),
                                  lambda: JointEnsemble(small_config(), 3, member_loss_weight=0.5)],
                         ids=["bidoze_g", "joint"])
def test_a_capped_fit_resumed_from_its_state_matches_an_uninterrupted_fit(make, tmp_path):
    from arcasleep.training.balanced import Patience

    rng = np.random.default_rng(2)
    x = np.stack([recording(rng, 60, T), recording(rng, 50, T)])
    y = np.full((2, T), -2, dtype=np.int32)
    y[0, :60] = rng.integers(0, 4, 60)
    y[1, :50] = rng.integers(0, 4, 50)
    kw = dict(batch_size=2, initial_lr=1e-3, weight_decay=0.0, balance_beta=0.999,
              patience=Patience(lr_drop=2, early_stop=50, unit="steps"), max_epochs=5, fit_seed=0)
    whole = make()
    h_whole = whole.fit_balanced(x, y, x, y, **kw)
    state = tmp_path / "state.pkl"
    first = make()
    first.fit_balanced(x, y, x, y, max_train_seconds=0.0, fit_state_path=state, **kw)
    assert state.exists()
    second = make()
    h_split = second.fit_balanced(x, y, x, y, fit_state_path=state, **kw)
    assert not state.exists() and (tmp_path / "state.used.pkl").exists()
    np.testing.assert_allclose(h_split["eval_kappa"], h_whole["eval_kappa"], atol=1e-6)
    assert h_split["lr"] == h_whole["lr"]
    for a, b in zip(jax.tree_util.tree_leaves(whole.params), jax.tree_util.tree_leaves(second.params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def split_batch_inputs(members: int, per_member: int):
    rng = np.random.default_rng(5)
    x = np.stack([recording(rng, 60, T) for _ in range(members * per_member)])
    y = rng.integers(0, 4, (members * per_member, T)).astype(np.int32)
    mask = np.ones((members * per_member, T), dtype=bool)
    return jnp.asarray(x), jnp.asarray(mask), jnp.asarray(y)


def test_a_split_batch_member_gets_gradient_from_its_own_group_only():
    from arcasleep_bidoze.joint import SplitBatchEnsemble

    members, b = 3, 2
    model = SplitBatchEnsemble(small_config(), members)
    model._ensure_built((T, F + 1, C), seed=0)
    x, mask, y = split_batch_inputs(members, b)
    key = jax.random.PRNGKey(0)

    def grads(x_in):
        return jax.grad(lambda p: model._balanced_loss(p, x_in, mask, y, key, 5, 0.999))(model.params)

    g0 = grads(x)
    x_changed = x.at[b:2 * b].add(1.0)  # group 1 only
    g1 = grads(x_changed)
    for m, same in [(0, True), (1, False), (2, True)]:
        a = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda leaf: leaf[m], g0))
        c = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda leaf: leaf[m], g1))
        equal = all(np.allclose(u, v, atol=1e-7) for u, v in zip(a, c))
        assert equal == same


def test_a_split_batch_ensemble_outputs_the_mean_and_saves_each_member(tmp_path):
    from arcasleep_bidoze.joint import SplitBatchEnsemble

    model = SplitBatchEnsemble(small_config(), 3)
    model._ensure_built((T, F + 1, C), seed=0)
    x, mask, _ = split_batch_inputs(3, 1)
    out = model._apply(model.params, x, mask, train=False)
    assert out.shape == (3, T, 5)
    model.savez(tmp_path / "model.pisces")
    assert len(list(tmp_path.glob("model_member*.pisces"))) == 3
    with pytest.raises(NotImplementedError):
        model._balanced_loss(model.params, x, mask, mask.astype(jnp.int32), jax.random.PRNGKey(0), 5, 0.99,
                             weights=jnp.ones(3))


def test_a_split_batch_fit_runs():
    from arcasleep.training.balanced import Patience
    from arcasleep_bidoze.joint import SplitBatchEnsemble

    model = SplitBatchEnsemble(small_config(), 3)
    x, _, y = split_batch_inputs(3, 2)
    x, y = np.asarray(x), np.asarray(y)
    model.fit_balanced(
        x, y, x, y, batch_size=6, initial_lr=1e-3, weight_decay=0.0, balance_beta=0.999,
        patience=Patience(lr_drop=10, early_stop=20, unit="steps"), max_epochs=1, fit_seed=0,
    )


def test_training_config_accepts_member_batch_split():
    from arcasleep_bidoze.training import TrainingConfig
    from test_balanced_recipe import BALANCED_TRAINING

    cfg = TrainingConfig.from_dict({**BALANCED_TRAINING, "joint_members": 7, "member_batch_split": True})
    assert cfg.member_batch_split
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({**BALANCED_TRAINING, "member_batch_split": True})


def test_member_recording_rows_and_subsamples():
    from arcasleep.data.cache import SubjectRef
    from arcasleep_bidoze.training import member_recording_rows, subsample_recordings

    refs = [SubjectRef("CohortA", f"a{i}") for i in range(20)]
    boot = member_recording_rows(refs, "bootstrap", 5, 4)
    half = member_recording_rows(refs, "half", 5, 4)
    assert [len(r) for r in boot] == [20] * 4
    assert [len(set(r)) for r in half] == [10] * 4 and [len(r) for r in half] == [10] * 4
    assert len({tuple(r) for r in half}) > 1
    assert half == member_recording_rows(refs, "half", 5, 4)
    sub = subsample_recordings(refs, 0.5, 3)
    assert len(sub) == 10 and len(set(sub)) == 10 and set(sub) <= set(refs)
    assert sub == subsample_recordings(refs, 0.5, 3)


def test_members_draw_rows_only_from_their_own_recordings(monkeypatch):
    from arcasleep.training.balanced import Patience
    from arcasleep_bidoze.joint import SplitBatchEnsemble
    from arcasleep_bidoze.model import definition

    calls = []
    real = definition.sample_balanced_segments

    def spy(nights, rng, **kw):
        out = real(nights, rng, **kw)
        calls.append((len(nights), out[2][:, 0].copy()))
        return out

    monkeypatch.setattr(definition, "sample_balanced_segments", spy)
    model = SplitBatchEnsemble(small_config(), 2)
    x, _, y = split_batch_inputs(2, 3)
    x, y = np.asarray(x), np.asarray(y)
    model.fit_balanced(
        x, y, x, y, batch_size=4, initial_lr=1e-3, weight_decay=0.0, balance_beta=0.999,
        patience=Patience(lr_drop=10, early_stop=20, unit="steps"), max_epochs=1, fit_seed=0,
        member_rows=[[0, 1], [4, 5]],
    )
    assert [n for n, _ in calls] == [2, 2]
    assert all(0 <= r < 2 for r in calls[0][1]) and all(0 <= r < 2 for r in calls[1][1])


def test_training_config_accepts_member_recordings_and_subsampling():
    from arcasleep_bidoze.training import TrainingConfig
    from test_balanced_recipe import BALANCED_TRAINING

    split = {**BALANCED_TRAINING, "joint_members": 7, "member_batch_split": True}
    cfg = TrainingConfig.from_dict({**split, "member_recordings": "half", "member_seed": 3})
    assert cfg.member_recordings == "half"
    cfg = TrainingConfig.from_dict({**BALANCED_TRAINING, "subsample_fraction": 0.5, "subsample_seed": 200})
    assert cfg.subsample_fraction == 0.5
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({**BALANCED_TRAINING, "member_recordings": "half"})
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({**BALANCED_TRAINING, "subsample_fraction": 0.5})
