"""Opt-in weighted graph attraction: math and wiring, not improvement claims."""
import json

import numpy as np
import pytest
import torch

from open_deep_tda import DeepTDA, TDAConfig
from open_deep_tda import fuzzy


def test_attraction_analytic_values_and_gradients():
    pos = torch.tensor([0., .2, .8, 1.7], dtype=torch.double, requires_grad=True)
    neg = torch.tensor([0., .3, 1.2], dtype=torch.double, requires_grad=True)
    weights = torch.tensor([.4, .1, .5, 0.], dtype=torch.double, requires_grad=True)
    scale = .7
    positive, negative = fuzzy.fuzzy_losses(pos, neg, weights, scale, positive_mode="attraction")
    attraction = torch.log1p((pos / scale).square())
    mass = weights.sum()
    expected = (weights * attraction).sum() / mass
    nt = (neg / scale).square() + 1e-8
    torch.testing.assert_close(positive, expected)
    torch.testing.assert_close(negative, torch.log1p(1 / nt).mean())
    gp, gn, gw = torch.autograd.grad(positive + negative, (pos, neg, weights))
    torch.testing.assert_close(gp, weights / mass * 2 * pos / (scale**2 + pos.square()))
    torch.testing.assert_close(gn, -2 * neg / scale**2 / (nt * (nt + 1)) / len(neg))
    torch.testing.assert_close(gw, (attraction - expected) / mass)
    assert gp[-1] == 0  # A zero-weight graph edge is not repelled in this mode.
    assert gp[1] > 0 and gn[1] < 0


def test_attraction_gradcheck():
    pos = torch.tensor([.2, .8, 1.7], dtype=torch.double, requires_grad=True)
    neg = torch.tensor([.3, 1.2], dtype=torch.double, requires_grad=True)
    weights = torch.tensor([.1, .5, .9], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda p, n, w: fuzzy.fuzzy_losses(p, n, w, .7, positive_mode="attraction"),
        (pos, neg, weights))


@pytest.mark.parametrize("mass_scale", [1., .1, 1e-200])
def test_attraction_normalizes_by_weight_mass_not_edge_count(mass_scale):
    pos = torch.tensor([1., 2.], dtype=torch.double, requires_grad=True)
    weights = torch.tensor([.2, .6], dtype=torch.double) * mass_scale
    positive, _ = fuzzy.fuzzy_losses(pos, pos[:0], weights, positive_mode="attraction")
    torch.testing.assert_close(positive, .25 * torch.log1p(pos[0]**2)
                               + .75 * torch.log1p(pos[1]**2))
    torch.testing.assert_close(torch.autograd.grad(positive, pos)[0],
                               torch.tensor([.25, .6], dtype=torch.double))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("size", [0, 3])
def test_zero_mass_and_empty_terms_have_connected_zero_gradients(dtype, size):
    pos = torch.tensor([0., 1., torch.finfo(dtype).max][:size], dtype=dtype,
                       requires_grad=True)
    weights = torch.zeros(size, dtype=dtype, requires_grad=True)
    neg = torch.empty(0, dtype=dtype, requires_grad=True)
    positive, negative = fuzzy.fuzzy_losses(pos, neg, weights, positive_mode="attraction")
    assert positive.item() == negative.item() == 0
    assert positive.requires_grad and negative.requires_grad
    (positive + negative).backward()
    for tensor in (pos, neg, weights):
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1e-200, 1., 1e200])
def test_attraction_extreme_distances_remain_finite(dtype, scale):
    pos = torch.tensor([0., 1., torch.finfo(dtype).max], dtype=dtype, requires_grad=True)
    neg = pos.detach().clone().requires_grad_()
    weights = torch.tensor([1., .3, .7], dtype=dtype)
    positive, negative = fuzzy.fuzzy_losses(pos, neg, weights, scale, positive_mode="attraction")
    assert torch.isfinite(positive + negative)
    for grad in torch.autograd.grad(positive + negative, (pos, neg)):
        assert torch.isfinite(grad).all()


def test_default_cross_entropy_and_negative_gradients_unchanged():
    pos = torch.tensor([.2, .8, 1.7], dtype=torch.double, requires_grad=True)
    neg = torch.tensor([.3, 1.2], dtype=torch.double, requires_grad=True)
    weights = torch.tensor([.1, .5, .9], dtype=torch.double, requires_grad=True)
    default = fuzzy.fuzzy_losses(pos, neg, weights, .7)
    explicit = fuzzy.fuzzy_losses(pos, neg, weights, .7, positive_mode="cross_entropy")
    attraction = fuzzy.fuzzy_losses(pos, neg, weights, .7, positive_mode="attraction")
    for a, b in zip(default, explicit):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(default[1], attraction[1], rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(default[1], neg)[0],
                               torch.autograd.grad(attraction[1], neg)[0], rtol=0, atol=0)
    t = (pos / .7).square()
    expected = (weights * t.log1p() + (1 - weights) * (1 / (t + 1e-8)).log1p()).mean()
    torch.testing.assert_close(default[0], expected)
    for a, b in zip(torch.autograd.grad(default[0], (pos, weights)),
                    torch.autograd.grad(expected, (pos, weights))):
        torch.testing.assert_close(a, b)
    assert not torch.isclose(default[0], attraction[0])


@pytest.mark.parametrize("mode", ["bad", "fuzzy_graph", "", None, False, 0])
def test_invalid_positive_mode(mode):
    empty = torch.empty(0)
    with pytest.raises(ValueError, match="positive_mode"):
        fuzzy.fuzzy_losses(empty, empty, empty, positive_mode=mode)


def test_positive_mode_is_keyword_only():
    t = torch.ones(1)
    with pytest.raises(TypeError):
        fuzzy.fuzzy_losses(t, t, t, 1., "attraction")


@pytest.mark.parametrize("objective", ["stress", "fuzzy", "fuzzy_graph"])
def test_config_accepts_objectives_and_roundtrips(objective):
    cfg = TDAConfig(geometry_objective=objective).validate()
    restored = TDAConfig(**json.loads(json.dumps(cfg.to_dict()))).validate()
    assert restored.geometry_objective == objective
    assert DeepTDA(cfg).config.geometry_objective == objective
    assert TDAConfig().geometry_objective == "stress"


@pytest.mark.parametrize("options", [
    {"geometry_objective": "attraction"}, {"geometry_objective": "bad"},
    {"fuzzy_scale": 0}, {"fuzzy_scale": float("nan")}, {"fuzzy_repulsion": -1},
])
def test_config_rejects_invalid_options(options):
    values = dict(geometry_objective="fuzzy_graph")
    values.update(options)
    with pytest.raises(ValueError):
        TDAConfig(**values).validate()


def small(**options):
    values = dict(geometry_objective="fuzzy_graph", steps=3, warmup_steps=0,
                  n_neighbors=4, batch_size=16, h0_size=8, h1_size=8,
                  evaluation_size=8, subset_bank_size=1, topology_interval=1,
                  log_interval=1, validation_interval=20, fuzzy_repulsion=.37)
    values.update(options)
    return DeepTDA(**values)


@pytest.mark.parametrize("optimizer_mode", ["parametric", "coordinates"])
def test_fuzzy_graph_fit_save_load(tmp_path, optimizer_mode):
    X = np.random.default_rng(7).normal(size=(24, 4)).astype(np.float32)
    model = small(optimizer_mode=optimizer_mode).fit(X)
    assert model.embedding_.shape == (24, 2)
    assert np.isfinite(model.embedding_).all()
    assert len(model.history_) == model.config.steps
    assert all(np.isfinite(row["loss"]) for row in model.history_)
    assert any(row["h1_evaluated"] for row in model.history_)
    diagnostics = model.report_["geometry_objective"]
    assert diagnostics["objective"] == "fuzzy_graph"
    assert diagnostics["positive_mode"] == "attraction"
    assert diagnostics["negative_repulsion"] == .37
    assert "not an exact" in diagnostics["note"]
    json.dumps(model.report_, allow_nan=False)
    restored = DeepTDA.load(model.save(tmp_path / "fuzzy_graph.pt"))
    assert restored.config.to_dict() == model.config.to_dict()
    assert restored.geometry_diagnostics_ == diagnostics
    np.testing.assert_array_equal(restored.embedding_, model.embedding_)
    if optimizer_mode == "parametric":
        np.testing.assert_array_equal(restored.transform(X), model.transform(X))
    else:
        with pytest.raises(NotImplementedError):
            restored.transform(X)


@pytest.mark.parametrize("scale", [None, .7])
def test_estimator_reuses_fuzzy_graph_weights_scale_and_repulsion(monkeypatch, scale):
    X = np.random.default_rng(12).normal(size=(24, 4)).astype(np.float32)
    graphs, calls = [], []
    build_weights, loss = fuzzy.build_fuzzy_weights, fuzzy.fuzzy_losses

    def record_graph(reference, neighbors, edges):
        weights, diagnostics = build_weights(reference, neighbors, edges)
        graphs.append((reference.copy(), neighbors.copy(), edges.copy(), weights.copy()))
        return weights, diagnostics

    def record_loss(pos, neg, weights, scale=1., *, positive_mode="cross_entropy"):
        result = loss(pos, neg, weights, scale, positive_mode=positive_mode)
        calls.append((pos.detach().clone(), neg.detach().clone(), weights.detach().clone(),
                      scale, positive_mode))
        return result

    monkeypatch.setattr(fuzzy, "build_fuzzy_weights", record_graph)
    monkeypatch.setattr(fuzzy, "fuzzy_losses", record_loss)
    models = [small(geometry_objective=objective, steps=1, lambda_h0=0, lambda_h1=0,
                    lambda_sep=9., lambda_near=1.3, fuzzy_scale=scale).fit(X)
              for objective in ("fuzzy", "fuzzy_graph")]
    assert len(graphs) == len(calls) == 2
    for a, b in zip(graphs[0], graphs[1]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(calls[0][:3], calls[1][:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert calls[0][3] == calls[1][3] == (models[0].local_scale_ if scale is None else scale)
    assert [call[4] for call in calls] == ["cross_entropy", "attraction"]
    assert calls[0][1].numel() > 0  # Exercise sampled nonedge repulsion.
    for model, call in zip(models, calls):
        p, n, w, kernel_scale, positive_mode = call
        near, separation = loss(p, n, w, kernel_scale, positive_mode=positive_mode)
        row = model.history_[0]
        assert row["near"] == pytest.approx(near.item())
        assert row["separation"] == pytest.approx(separation.item())
        assert row["loss"] == pytest.approx(1.3 * near.item() + .37 * separation.item())
        assert model.geometry_diagnostics_["positive_mode"] == positive_mode
    assert models[0].history_[0]["separation"] == models[1].history_[0]["separation"]


@pytest.mark.parametrize("constant", [False, True])
def test_untrained_diagnostics(constant):
    X = np.zeros((12, 3)) if constant else np.random.default_rng(4).normal(size=(12, 3))
    model = small(steps=3 if constant else 0).fit(X)
    assert model.geometry_diagnostics_ == {
        "objective": "fuzzy_graph", "positive_mode": "attraction", "trained": False}
