"""Live-gradient regression tests, with no simulated native PH backend.

Gradcheck clouds have distinct independent edge lengths; Rips structural ties
between a triangle and its longest edge are intentionally not perturbed.
"""

import copy

import numpy as np
import pytest
import torch

from open_deep_tda import losses, topology


@pytest.fixture
def native():
    pytest.importorskip("open_deep_tda._core", reason="Native C++ extension not built")


def cloud():
    return torch.tensor([[0., 0.], [1.17, .04], [1.03, 1.26], [-.11, .93]], dtype=torch.double)


def perturbed():
    return cloud() + torch.tensor([[.03, -.02], [-.02, .07], [.08, .03], [-.04, -.01]], dtype=torch.double)


def result_for(X):
    return topology.persistence(losses.pairwise_distances(X).detach().numpy())


def test_pairwise_matches_numpy_and_gradcheck():
    X = perturbed().requires_grad_()
    D = losses.pairwise_distances(X)
    np.testing.assert_allclose(D.detach().numpy(), topology.distance_matrix(X.detach().numpy()), rtol=1e-14)
    assert torch.autograd.gradcheck(losses.pairwise_distances, (X,), eps=1e-6, atol=1e-5)


def test_pairwise_zero_safe_and_large_cloud_diagonal():
    X = torch.zeros((30, 3), dtype=torch.double, requires_grad=True)
    D = losses.pairwise_distances(X)
    D.sum().backward()
    assert torch.equal(X.grad, torch.zeros_like(X))
    Y = torch.randn((30, 3), dtype=torch.double) + 1e6
    E = losses.pairwise_distances(Y)
    assert torch.equal(E.diag(), torch.zeros(30, dtype=torch.double))
    assert torch.equal(E, E.T)
    assert losses.pairwise_distances(torch.empty((0, 3))).shape == (0, 0)


@pytest.mark.parametrize("X", [torch.ones(2), torch.ones((2, 2), dtype=torch.long),
    torch.tensor([[float('nan')]]), torch.tensor([[float('inf')]])])
def test_bad_pairwise(X):
    with pytest.raises(ValueError):
        losses.pairwise_distances(X)


def test_near_huber_and_separation_reference_margin():
    source = torch.tensor([.4, 3., 1.], dtype=torch.double, requires_grad=True)
    target = torch.tensor([.1, 0., 3.], dtype=torch.double, requires_grad=True)
    near = losses.near_loss(source, target, delta=2)
    # Huber errors .3, 3, 2: .045, 4, 2 (not smooth-L1).
    assert near.item() == pytest.approx((.045 + 4 + 2) / 3)
    sep = losses.separation_loss(source, target, margin=1)
    assert sep.item() == pytest.approx((.3 ** 2 + 1) / 3)
    (near + sep).backward()
    assert source.grad is None
    assert target.grad is not None and torch.isfinite(target.grad).all()
    # Finite differences must stay inside the nonnegative-length domain.
    interior = (target.detach() + .13).requires_grad_()
    assert torch.autograd.gradcheck(lambda t: losses.near_loss(source, t, delta=1.7), (interior,))
    assert torch.autograd.gradcheck(lambda t: losses.separation_loss(source, t, margin=1.2), (interior,))


@pytest.mark.parametrize("function", [losses.near_loss, losses.separation_loss])
def test_empty_geometric_backward_and_validation(function):
    source = torch.empty(0, dtype=torch.double)
    target = torch.empty(0, dtype=torch.double, requires_grad=True)
    loss = function(source, target)
    assert loss.item() == 0 and loss.requires_grad
    loss.backward()
    assert target.grad is not None
    with pytest.raises(ValueError):
        function(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError):
        function(torch.tensor([-1.]), torch.ones(1))
    for value in [0, -1, np.inf, np.nan, True]:
        key = "delta" if function is losses.near_loss else "margin"
        with pytest.raises(ValueError):
            function(torch.ones(1), torch.ones(1), **{key: value})


@pytest.mark.parametrize("n", [0, 1])
def test_h0_empty_without_backend(n):
    D = torch.zeros((n, n), dtype=torch.double, requires_grad=True)
    loss = losses.h0_loss(D.detach(), D)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(D.grad, torch.zeros_like(D))


def test_loss_matrix_and_configuration_validation():
    D = torch.zeros((2, 2), dtype=torch.double)
    for bad in [torch.zeros((2, 3)), torch.tensor([[0., 1.], [2., 0.]]),
                torch.eye(2), torch.full((2, 2), float("nan"))]:
        for function in [losses.h0_loss, losses.h1_loss]:
            with pytest.raises(ValueError):
                function(D, bad)
    for tau in [0, -1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            losses.h1_loss(D, D, tau=tau)
    for budgets in [{"max_radius": 1}, {"unknown": 3}, {"max_simplices": np.inf}]:
        with pytest.raises(ValueError):
            losses.h1_loss(D, D, budgets=budgets)
    with pytest.raises(ValueError):
        losses.h0_loss(D, D, source_edges=[[0, 2]])
    with pytest.raises(ValueError):
        losses.h0_loss(torch.zeros((4, 4)), torch.zeros((4, 4)), source_edges=[[0, 1], [1, 2], [2, 0]])


def test_h0_symmetric_identity_aware_analytic(native):
    # Source MST 01,12; target MST 02,12. The 02 false short connection
    # is invisible to a one-sided source tree loss.
    S = torch.tensor([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]], dtype=torch.double, requires_grad=True)
    T = torch.tensor([[0., 3., 1.], [3., 0., 2.], [1., 2., 0.]], dtype=torch.double, requires_grad=True)
    value = losses.h0_loss(S, T)
    assert value.item() == pytest.approx((4 + 0 + 4 + 0) / 4)
    edges = topology.mst(S.detach().numpy())
    assert losses.h0_loss(S, T, source_edges=edges).item() == value.item()
    value.backward()
    assert S.grad is None
    assert T.grad.abs().sum() > 0
    assert losses.h0_loss(S, S).item() == 0


def test_h0_coordinate_gradcheck(native):
    S = losses.pairwise_distances(cloud())
    X = perturbed().requires_grad_()
    assert torch.autograd.gradcheck(lambda x: losses.h0_loss(S, losses.pairwise_distances(x)), (X,))


@pytest.mark.parametrize("scale,expected_matched", [(1.0, 1), (4.0, 0)])
def test_h1_matched_and_both_diagonal_gradcheck(native, scale, expected_matched):
    source = cloud().requires_grad_()
    D_source = losses.pairwise_distances(source)
    cached = result_for(source)
    target = (perturbed() * scale).requires_grad_()
    reference = topology.diagram(cached)
    target_result = result_for(target)
    match = topology.diagram_matching(reference, topology.diagram(target_result))
    assert len(match["matched"]) == expected_matched
    def function(x):
        result = losses.h1_loss(D_source, losses.pairwise_distances(x), source_result=cached, tau=.07)
        return result["pd1"], result["crit1"]
    assert torch.autograd.gradcheck(function, (target,), eps=1e-6, atol=1e-5, rtol=1e-4)
    pd1, crit1 = function(target)
    assert pd1.item() == pytest.approx(match["cost"] / max(1, len(reference)))
    assert crit1.item() > 0
    (pd1 + crit1).backward()
    assert source.grad is None  # fixed source, including when supplied live
    assert target.grad.abs().sum() > 0


def test_critical_loss_reference_edges_and_weights(native):
    source = torch.cat((cloud(), .75 * cloud() + torch.tensor([7., .3])), dim=0)
    target = torch.cat((perturbed(), .81 * perturbed() + torch.tensor([7., .3])), dim=0)
    S = losses.pairwise_distances(source)
    T = losses.pairwise_distances(target)
    cached = result_for(source)
    result = losses.h1_loss(S, T, source_result=cached, tau=.17)
    positive = [p for p in cached["pairs"] if p["dimension"] == 1 and p["death"] > p["birth"]]
    assert len(positive) >= 2
    assert len({p["death"] - p["birth"] for p in positive}) >= 2
    numerator, weight_sum = 0., 0.
    for p in positive:
        lifetime = p["death"] - p["birth"]
        w = lifetime / (lifetime + .17)
        weight_sum += w
        for key in ["birth_edge", "death_edge"]:
            i, j = p[key]
            numerator += w * float((T[i, j] - S[i, j]) ** 2)
    assert result["crit1"].item() == pytest.approx(numerator / (2 * weight_sum + 1e-12))
    identical = losses.h1_loss(S, S, source_result=cached)
    assert identical["pd1"].item() == 0 and identical["crit1"].item() == 0


@pytest.mark.parametrize("n", [0, 1, 3])
def test_h1_empty_diagrams_backward(native, n):
    X = torch.tensor([[0., 0.], [1.3, .1], [.2, .7]], dtype=torch.double)[:n]
    target = (X * 1.1).clone().requires_grad_()
    result = losses.h1_loss(losses.pairwise_distances(X), losses.pairwise_distances(target))
    assert result["source_bars"] == result["target_bars"] == 0
    for key in ["pd1", "crit1"]:
        assert result[key].requires_grad and result[key].item() == 0
    (result["pd1"] + result["crit1"]).backward()
    assert target.grad is not None and torch.equal(target.grad, torch.zeros_like(target))


def test_missing_source_loop_has_constant_pd_but_live_critical_gradient(native):
    source = cloud()
    target = torch.tensor([[0., 0.], [.5, .02], [1.4, -.01], [2.8, .03]], dtype=torch.double, requires_grad=True)
    S = losses.pairwise_distances(source)
    def function(x):
        out = losses.h1_loss(S, losses.pairwise_distances(x))
        return out["pd1"], out["crit1"]
    result = losses.h1_loss(S, losses.pairwise_distances(target))
    assert result["source_bars"] == 1 and result["target_bars"] == 0
    assert torch.autograd.gradcheck(function, (target,))
    pd_gradient = torch.autograd.grad(result["pd1"], target, retain_graph=True)[0]
    critical_gradient = torch.autograd.grad(result["crit1"], target)[0]
    assert torch.equal(pd_gradient, torch.zeros_like(target))
    assert critical_gradient.abs().sum() > 0


def test_target_only_multiple_bars_not_normalized_by_target_count(native):
    target = torch.cat((cloud(), perturbed() + torch.tensor([7., .3])), dim=0).requires_grad_()
    source = torch.stack((torch.arange(8, dtype=torch.double), torch.zeros(8, dtype=torch.double)), dim=1)
    S = losses.pairwise_distances(source)
    result = losses.h1_loss(S, losses.pairwise_distances(target))
    bars = topology.diagram(result_for(target))
    assert result["source_bars"] == 0 and result["target_bars"] >= 2
    assert result["pd1"].item() == pytest.approx(np.sum((bars[:, 1] - bars[:, 0]) ** 2 / 2))
    assert result["crit1"].item() == 0
    assert torch.autograd.gradcheck(lambda x: losses.h1_loss(S, losses.pairwise_distances(x))["pd1"], (target,))


def test_multiple_source_bars_fixed_denominator(native):
    source = torch.cat((cloud(), perturbed() + torch.tensor([7., .3])), dim=0)
    target = (source * 1.08).requires_grad_()
    P, Q = topology.diagram(result_for(source)), topology.diagram(result_for(target))
    assert len(P) >= 2
    out = losses.h1_loss(losses.pairwise_distances(source), losses.pairwise_distances(target))
    assert out["pd1"].item() == pytest.approx(topology.diagram_matching(P, Q)["cost"] / len(P))


def test_target_native_recomputed_and_cached_source_used(native, monkeypatch):
    original = topology.h1_persistence
    calls = []
    def counted(D, **budgets):
        calls.append(np.array(D, copy=True))
        return original(D, **budgets)
    S = losses.pairwise_distances(cloud())
    cached = original(S.numpy())
    monkeypatch.setattr(topology, "h1_persistence", counted)
    first = losses.h1_loss(S, losses.pairwise_distances(perturbed()), source_result=cached)
    second = losses.h1_loss(S, losses.pairwise_distances(4 * perturbed()), source_result=cached)
    assert len(calls) == 2 and not np.array_equal(calls[0], calls[1])
    assert first["pd1"].item() != second["pd1"].item()


def test_h1_censor_cache_and_budget_rejection(native):
    S = losses.pairwise_distances(cloud())
    cached = result_for(cloud())
    for changes in [{"truncated": True}, {"n_vertices": 99}]:
        bad = dict(cached, **changes)
        with pytest.raises(ValueError):
            losses.h1_loss(S, S, source_result=bad)
    bad = copy.deepcopy(cached)
    p = next(p for p in bad["pairs"] if p["dimension"] == 1 and p["death"] > p["birth"])
    p["birth_edge"] = [-1, -1]
    with pytest.raises(ValueError, match="vertex"):
        losses.h1_loss(S, S, source_result=bad)
    with pytest.raises(ValueError, match="Cached source"):
        losses.h1_loss(2 * S, S, source_result=cached)
    with pytest.raises(RuntimeError):
        losses.h1_loss(S, S, budgets={"max_simplices": 1})
    with pytest.raises(RuntimeError, match="matching"):
        losses.h1_loss(S, S, max_matching_size=1)


def test_float64_native_cache_with_float32_training_distances(native):
    # Source caches commonly precede conversion to the model's float32 dtype.
    # Exact equality against native double endpoints incorrectly rejects this.
    D = topology.distance_matrix(cloud().numpy())
    cache = topology.persistence(D)
    S = torch.tensor(D, dtype=torch.float32)
    target = perturbed().float().requires_grad_()
    result = losses.h1_loss(S, losses.pairwise_distances(target), source_result=cache)
    assert result["source_bars"] == result["target_bars"] == 1
    assert result["pd1"].dtype == torch.float32
    (result["pd1"] + result["crit1"]).backward()
    assert target.grad.abs().sum() > 0 and torch.isfinite(target.grad).all()


def test_duplicate_cloud_backward_native(native):
    source = torch.zeros((4, 2), dtype=torch.double)
    target = source.clone().requires_grad_()
    S, T = losses.pairwise_distances(source), losses.pairwise_distances(target)
    h1 = losses.h1_loss(S, T)
    combined = losses.h0_loss(S, T) + h1["pd1"] + h1["crit1"]
    assert combined.item() == 0
    combined.backward()
    assert torch.equal(target.grad, torch.zeros_like(target))
