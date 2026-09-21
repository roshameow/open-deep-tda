"""Fuzzy graph math, degeneracies, validation and live gradients."""
import json

import numpy as np
import pytest
import torch

from open_deep_tda.fuzzy import build_fuzzy_weights, fuzzy_losses
from open_deep_tda.neighbors import build_neighbor_graph


def graph(X, k):
    result = build_neighbor_graph(X, k)
    return result['neighbors'], result['edges']


def test_weights_mass_union_and_scale():
    X = np.random.default_rng(42).normal(size=(20, 4))
    neighbors, edges = graph(X, 7)
    weights, diagnostics = build_fuzzy_weights(X, neighbors, edges)
    assert weights.dtype == np.float32 and weights.shape == (len(edges),)
    assert np.all((weights >= 0) & (weights <= 1))
    np.testing.assert_allclose(diagnostics['directed_mass_range'], np.log2(7), atol=1e-12)
    json.dumps(diagnostics, allow_nan=False)
    for scale in (1e-100, 100):
        scaled, _ = build_fuzzy_weights(X * scale, neighbors[:, ::-1], edges[:, ::-1])
        np.testing.assert_allclose(scaled, weights, atol=1e-7)
    # Independent scalar root/union oracle.
    from scipy.optimize import brentq
    directed = np.zeros((len(X), len(X)))
    for i, row in enumerate(neighbors):
        d = np.linalg.norm(X[row] - X[i], axis=1)
        gap = d - d.min()
        sigma = brentq(lambda s: np.exp(-gap / s).sum() - np.log2(7), 1e-12, 100)
        directed[i, row] = np.exp(-gap / sigma)
    expected = directed + directed.T - directed * directed.T
    np.testing.assert_allclose(weights, expected[edges[:, 0], edges[:, 1]], atol=1e-7)


@pytest.mark.parametrize('X,k', [(np.empty((0, 2)), 0), (np.zeros((1, 2)), 1),
                                 (np.zeros((5, 2)), 0)])
def test_empty_graph(X, k):
    neighbors, edges = graph(X, k)
    weights, diagnostics = build_fuzzy_weights(X, neighbors, edges)
    assert weights.shape == (0,)
    json.dumps(diagnostics, allow_nan=False)


def test_duplicates_ties_tiny_k_and_nonedge():
    X = np.array([[0.], [0.], [1.], [1.]])
    neighbors, edges = graph(X, 3)
    weights, diagnostics = build_fuzzy_weights(X, neighbors, edges)
    np.testing.assert_array_equal(weights, 1)
    assert diagnostics['unattainable_target_rows'] == 4
    assert diagnostics['sigma_zero_rows'] == 4
    assert diagnostics['rho_range'] == [1., 1.]
    all_same = np.zeros((4, 2))
    _, diag = build_fuzzy_weights(all_same, neighbors, edges)
    assert diag['rho_range'] == [0., 0.]
    neighbors = np.array([[1], [0], [3], [2]])
    edges = np.array([[0, 1], [1, 0], [0, 1], [0, 2]])
    weights, _ = build_fuzzy_weights(X, neighbors, edges)
    np.testing.assert_array_equal(weights, [1, 1, 1, 0])
    # k=2 target=1, unique closest gives the explicit zero-bandwidth limit.
    X = np.array([[0.], [1.], [3.]])
    weights, diag = build_fuzzy_weights(X, *graph(X, 2))
    assert diag['target_directed_mass'] == 1
    np.testing.assert_array_equal(weights, [1, 0, 1])


@pytest.mark.parametrize('which,value', [
    ('reference', [[np.nan], [1.], [2.]]),
    ('reference', [[np.inf], [1.], [2.]]),
    ('reference', [[1j], [1.], [2.]]),
    ('reference', [['1'], ['2'], ['3']]),
    ('reference', np.empty((3, 0))),
    ('neighbors', [[1, 1], [0, 2], [0, 1]]),
    ('neighbors', [[0, 2], [0, 2], [0, 1]]),
    ('neighbors', [[1, 3], [0, 2], [0, 1]]),
    ('neighbors', [[1., 2.], [0., 2.], [0., 1.]]),
    ('neighbors', [[1], [0]]),
    ('edges', [[0, 0]]), ('edges', [[0, -1]]), ('edges', [[0, 3]]),
    ('edges', [[0., 1.]]), ('edges', [0, 1]),
])
def test_invalid_graph_inputs(which, value):
    args = dict(reference=np.arange(3.)[:, None], neighbors=np.array([[1, 2], [0, 2], [0, 1]]),
                edges=np.array([[0, 1]]))
    args[which] = value
    with pytest.raises(ValueError):
        build_fuzzy_weights(**args)


def test_distance_overflow_rejected():
    with pytest.raises(ValueError, match='overflow'):
        build_fuzzy_weights(np.array([[1e308], [-1e308]]), np.array([[1], [0]]), np.array([[0, 1]]))


def test_fuzzy_formula_and_gradcheck():
    pos = torch.tensor([.2, .8, 1.7], dtype=torch.double, requires_grad=True)
    neg = torch.tensor([.3, 1.2], dtype=torch.double, requires_grad=True)
    weights = torch.tensor([.1, .5, .9], dtype=torch.double, requires_grad=True)
    scale = .7
    p, n = fuzzy_losses(pos, neg, weights, scale)
    t, nt = (pos / scale).square(), (neg / scale).square()
    expected = (weights * t.log1p() + (1 - weights) * (1 / (t + 1e-8)).log1p()).mean()
    torch.testing.assert_close(p, expected)
    torch.testing.assert_close(n, (1 / (nt + 1e-8)).log1p().mean())
    assert torch.autograd.gradcheck(lambda a, b, w: fuzzy_losses(a, b, w, scale), (pos, neg, weights))


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_zero_huge_distances_and_empty_connected(dtype):
    pos = torch.tensor([0., 1., torch.finfo(dtype).max], dtype=dtype, requires_grad=True)
    neg = pos.detach().clone().requires_grad_()
    weights = torch.tensor([1., .3, 0.], dtype=dtype)
    for scale in (1e-200, 1., 1e200):
        p, n = fuzzy_losses(pos, neg, weights, scale)
        assert torch.isfinite(p + n)
        grads = torch.autograd.grad(p + n, (pos, neg))
        assert all(torch.isfinite(g).all() for g in grads)
        zp, zn = fuzzy_losses(pos[:1], neg[:1], weights[:1], scale)
        assert zp.item() == 0
        assert zn.item() == pytest.approx(np.log1p(1e8), rel=1e-6)
    empty = torch.empty(0, dtype=dtype, requires_grad=True)
    p, n = fuzzy_losses(empty, empty, empty)
    assert p.item() == n.item() == 0
    (p + n).backward()
    assert empty.grad is not None
    p, n = fuzzy_losses(pos, empty, weights)
    assert n.item() == 0 and n.requires_grad
    p, n = fuzzy_losses(empty, neg, empty)
    assert p.item() == 0 and p.requires_grad


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf'), True, '1'])
def test_invalid_scale(scale):
    t = torch.ones(2)
    with pytest.raises(ValueError):
        fuzzy_losses(t, t, t, scale)


@pytest.mark.parametrize('index,value', [(0, [-1.]), (0, [float('inf')]),
    (1, [float('nan')]), (2, [1.1]), (2, [-.1])])
def test_invalid_loss_values(index, value):
    args = [torch.ones(1) for _ in range(3)]
    args[index] = torch.tensor(value)
    with pytest.raises(ValueError):
        fuzzy_losses(*args)


def test_attraction_and_repulsion_gradient_directions():
    pos = torch.tensor([.5, .5], dtype=torch.double, requires_grad=True)
    neg = torch.tensor([.5], dtype=torch.double, requires_grad=True)
    p, n = fuzzy_losses(pos, neg, torch.tensor([1., 0.], dtype=torch.double))
    gp, gn = torch.autograd.grad(p + n, (pos, neg))
    assert gp[0] > 0  # gradient descent shortens a fully positive edge
    assert gp[1] < 0 and gn[0] < 0  # and separates zero-weight/nonedges


def test_loss_shapes_and_dtypes():
    t = torch.ones(2)
    for args in ((t, t, t[:1]), (t, t.double(), t), (t[:, None], t, t),
                 (t, t.int(), t), (t, [1., 2.], t)):
        with pytest.raises(ValueError):
            fuzzy_losses(*args)


def test_pilot_loader_opens_only_train_feature_and_subject_members(tmp_path, monkeypatch):
    import importlib.util
    import io
    from pathlib import Path
    import zipfile
    spec = importlib.util.spec_from_file_location('fuzzy_pilot',
        Path(__file__).resolve().parents[1] / 'benchmarks/tune_fuzzy_geometry.py')
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    inner_bytes = io.BytesIO()
    with zipfile.ZipFile(inner_bytes, 'w') as inner:
        inner.writestr('UCI HAR Dataset/train/X_train.txt', 'features')
        inner.writestr('UCI HAR Dataset/train/subject_train.txt', 'subjects')
        # Deliberately do not provide TEST or y_* members: any access fails.
    archive = tmp_path / 'har.zip'
    with zipfile.ZipFile(archive, 'w') as outer:
        outer.writestr('UCI HAR Dataset.zip', inner_bytes.getvalue())
    monkeypatch.setattr(pilot, 'digest', lambda _: pilot.HAR_SHA256)
    parsed = []
    def loadtxt(stream, dtype):
        value = stream.read().decode()
        parsed.append(value)
        if value == 'features':
            return np.zeros((7352, 561), dtype=dtype)
        assert value == 'subjects'
        return np.resize(np.array([1, 2, 3, 5, 6, 7, 8, 11, 14, 15, 16,
                                   17, 19, 21, 22, 23, 25, 26, 27, 28, 29]), 7352)
    monkeypatch.setattr(pilot.np, 'loadtxt', loadtxt)
    train, validation, members = pilot.load_train(archive)
    assert parsed == ['features', 'subjects']
    assert len(train) + len(validation) == 7352
    assert len(validation) > 0
    assert members == ['UCI HAR Dataset/train/X_train.txt', 'UCI HAR Dataset/train/subject_train.txt']
    assert pilot.PLAN['validation_subject_ids'] == [3, 8, 17, 25]
    assert len(pilot.PLAN['candidates']) == 5
    assert 300 <= pilot.PLAN['steps'] <= 600
    assert len(pilot.PLAN['seeds']) in (1, 2)


def test_live_coordinate_gradient_duplicate_pair():
    z = torch.tensor([[0., 0.], [0., 0.], [.5, .2]], requires_grad=True)
    lengths = torch.linalg.vector_norm(z[[0, 0]] - z[[1, 2]], dim=1)
    p, n = fuzzy_losses(lengths, lengths, torch.tensor([1., .4]))
    (p + n).backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0
