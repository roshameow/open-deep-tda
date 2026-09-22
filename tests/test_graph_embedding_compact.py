"""Synthetic compact-domain contracts, not a topology-preservation certificate.

No datasets, private research imports, optional JIT, or training subprocesses.
Projection is checked against an independent all-anchor barycentric QP.
"""
import importlib
import inspect
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from open_deep_tda import graph_embedding as api
from open_deep_tda._graph_mapping import CompactMap
from open_deep_tda.graph_embedding import GraphEmbedding


@pytest.fixture
def square():
    return np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.], [0., 0.]])


def _solver_module():
    return importlib.import_module(CompactMap.__module__)


def _fixture():
    rng = np.random.default_rng(2018)
    return rng.normal(size=(24, 3)), rng.normal(size=(24, 2))


def _qp_projection(anchors, point):
    # Optimize convex weights over ALL original anchors, not implementation facets.
    def fg(w):
        residual = w @ anchors - point
        scale = 1 + point @ point
        return .5 * (residual @ residual) / scale, (anchors @ residual) / scale
    result = minimize(fg, np.full(len(anchors), 1 / len(anchors)), jac=True,
                      method='SLSQP', bounds=[(0., 1.)] * len(anchors),
                      constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1,
                                   'jac': lambda w: np.ones(len(w))},
                      options={'ftol': 1e-13, 'maxiter': 500})
    assert result.success, result.message
    return result.x @ anchors


def test_full_normalizer_gradient_and_infinity_escape_surrogate(square):
    mapper = CompactMap(square)
    ids, p = np.arange(4), np.full(4, .25)
    def oracle(y):
        w = 1 / (1 + np.sum((square - y) ** 2, axis=1))
        return np.dot(p, np.log(p / (w / w.sum())[ids]))
    y = np.array([.17, -.31])
    value, gradient = mapper.cauchy(y, ids, p)
    np.testing.assert_allclose(value, oracle(y), atol=1e-14, rtol=0)
    eps = 1e-5
    numeric = [(oracle(y + eps * e) - oracle(y - eps * e)) / (2 * eps)
               for e in np.eye(2)]
    np.testing.assert_allclose(gradient, numeric, atol=1e-10, rtol=0)
    far = [mapper.cauchy(np.array([r, 0.]), ids, p)[0] for r in (0., 1e3, 1e6)]
    assert far[0] > far[1] > far[2]
    assert abs(far[-1] - np.log(5 / 4)) < 1e-10
    bounded, record = mapper.solve(np.array([1e6, 0.]), ids, p, np.zeros(2))
    assert mapper.contains(bounded) and np.isfinite(bounded).all()
    assert record['phase'] == 'constrained_phase_II'
    assert record['final_objective'] <= record['bary_objective'] + 1e-13


@pytest.mark.parametrize('seed', [2018, 72])
def test_projection_original_polygon_and_independent_qp(seed):
    anchors = np.random.default_rng(seed).normal(size=(17, 2))
    mapper = CompactMap(anchors)
    hull = ConvexHull(anchors)
    for point in np.array([[0., 0.], [4., -5.], [-3., 2.], [8., 8.]]):
        actual = mapper.project(point)
        np.testing.assert_allclose(actual, _qp_projection(anchors, point), atol=2e-7, rtol=0)
        assert np.max(hull.equations[:, :2] @ actual + hull.equations[:, 2]) < 1e-12
        np.testing.assert_allclose(mapper.project(actual), actual, atol=1e-14, rtol=0)


def test_all_old_inside_points_retained_bitwise_without_solver(square, monkeypatch):
    mapper = CompactMap(square)
    def forbidden(*args, **kwargs):
        raise AssertionError('inside historical output must not be optimized or clipped')
    monkeypatch.setattr(_solver_module(), 'minimize', forbidden)
    ids, p = np.array([0, 1, 2]), np.array([.2, .3, .5])
    bary = np.einsum('i,ij->j', p, square[ids])
    points = np.vstack((square, [[-0., 0.], [.123456789, -.987654321]],
                        np.random.default_rng(2018).uniform(-1, 1, (20, 2))))
    for old in points:
        actual, record = mapper.solve(old, ids, p, bary)
        assert actual.tobytes() == old.tobytes()
        assert record['phase'] == 'retained_unbounded_inside'
        assert record['objective_calls'] == 1


def test_outside_starts_slsqp_at_bary_not_clipped_old(square, monkeypatch):
    mapper = CompactMap(square)
    ids, p = np.array([0, 1, 2]), np.array([.1, .2, .7])
    bary = np.einsum('i,ij->j', p, square[ids])
    old = np.array([4., -2.])
    seen = []
    real = _solver_module().minimize
    def checked(fg, x0, **kwargs):
        assert kwargs['method'] == 'SLSQP' and kwargs['jac'] is True
        assert kwargs['options'] == {'maxiter': 100, 'ftol': 1e-12}
        assert 'constraints' in kwargs
        np.testing.assert_array_equal(x0, bary)
        assert not np.array_equal(x0, mapper.project(old))
        seen.append(True)
        return real(fg, x0, **kwargs)
    monkeypatch.setattr(_solver_module(), 'minimize', checked)
    actual, record = mapper.solve(old, ids, p, bary)
    assert seen == [True] and mapper.contains(actual)
    assert record['final_objective'] <= mapper.cauchy(bary, ids, p)[0] + 1e-13
    assert record['objective_calls'] <= 2003
    assert np.isfinite(record['projected_gradient_l2'])


@pytest.mark.parametrize('failure', ['exception', 'unsuccessful', 'call_budget'])
def test_failure_retains_best_feasible_visited(square, monkeypatch, failure):
    mapper = CompactMap(square)
    ids, p = np.array([0, 1]), np.array([.3, .7])
    bary = np.einsum('i,ij->j', p, square[ids])
    # The barycenter is not assumed optimal: choose the best feasible point
    # from a deterministic grid and compare its exact objective independently.
    grid = np.array([(x, y) for x in np.linspace(-1, 1, 21)
                     for y in np.linspace(-1, 1, 21)])
    values = np.array([mapper.cauchy(y, ids, p)[0] for y in grid])
    best = grid[np.argmin(values)]
    assert values.min() < mapper.cauchy(bary, ids, p)[0]
    module = _solver_module()
    if failure == 'call_budget':
        monkeypatch.setattr(module, 'MAXCALLS', 3)
    def fake(fg, x0, **kwargs):
        fg(best)
        fg(bary)
        if failure == 'exception':
            raise RuntimeError('synthetic failure after feasible improvement')
        if failure == 'call_budget':
            for _ in range(4):
                fg(bary)
        return SimpleNamespace(x=np.array([100., 100.]), success=False,
                               status=9, message='synthetic failure', nit=2)
    monkeypatch.setattr(module, 'minimize', fake)
    result, record = mapper.solve(np.array([10., 10.]), ids, p, bary)
    np.testing.assert_array_equal(result, best)
    assert record['solver_success'] is False
    assert record['selected'] == 'visited'
    assert mapper.contains(result) and np.isfinite(result).all()
    if failure == 'call_budget':
        assert record['objective_calls'] <= 6


@pytest.mark.parametrize('anchors,expected_dimension', [
    (np.array([[-2., 1.], [0., 1.], [3., 1.]]), 1),
    (np.array([[2., -3.], [2., -3.]]), 0),
    (np.array([[2., -3.]]), 0),
])
def test_line_and_singleton_domains(anchors, expected_dimension):
    mapper = CompactMap(anchors)
    ids, p = np.arange(len(anchors)), np.full(len(anchors), 1 / len(anchors))
    bary = np.einsum('i,ij->j', p, anchors)
    old = np.array([10., 20.])
    result, record = mapper.solve(old, ids, p, bary)
    assert record['dimension'] == expected_dimension
    assert mapper.contains(result) and np.isfinite(result).all()
    np.testing.assert_allclose(mapper.project(old), _qp_projection(anchors, old), atol=1e-7)
    if expected_dimension == 0:
        np.testing.assert_array_equal(result, anchors[0])


@pytest.mark.parametrize('bad', [np.empty((0, 2)), np.ones((4, 3)),
    np.full((3, 2), np.nan), np.full((3, 2), np.inf),
    np.ones((3, 2), dtype=bool), np.ma.array(np.ones((3, 2)), mask=False)])
def test_invalid_compact_anchors(bad):
    with pytest.raises(ValueError):
        CompactMap(bad)


def test_vertex_budget_is_exact_and_inputs_detached(square):
    before = square.copy()
    mapper = CompactMap(square, max_vertices=len(square))
    with pytest.raises(ValueError):
        CompactMap(square, max_vertices=len(square) - 1)
    for bad in [True, 0, 1.5]:
        with pytest.raises(ValueError):
            CompactMap(square, max_vertices=bad)
    square[:] = 100
    np.testing.assert_array_equal(mapper.Y, before)
    for bad in [np.array([np.nan, 0]), np.array([np.inf, 0]), np.zeros(3)]:
        with pytest.raises(ValueError):
            mapper.project(bad)


@pytest.mark.parametrize('bad', [None, '', 'bounded', 'TRAIN_HULL', True, 1, [], {}])
def test_invalid_mapping_domain_constructor_and_factory(bad):
    X, Z = _fixture()
    with pytest.raises((ValueError, TypeError), match='mapping_domain'):
        GraphEmbedding(mapping_domain=bad)
    with pytest.raises((ValueError, TypeError), match='mapping_domain'):
        GraphEmbedding.from_layout(X, Z, mapping_domain=bad)


def test_default_domain_and_factory_rng_order_normalized_invariance(monkeypatch):
    assert inspect.signature(GraphEmbedding).parameters['mapping_domain'].default == 'train_hull'
    X, Z = _fixture()
    Q = np.vstack((X[:2], [[.2, -.1, .8]], X[:1]))
    originals = [v.copy() for v in (X, Z, Q)]
    ns, rs = np.random.get_state(), random.getstate()
    def forbidden(*args, **kwargs):
        raise AssertionError('from_layout/transform must not train or run a TDA worker')
    monkeypatch.setattr(api.subprocess, 'run', forbidden)
    model = GraphEmbedding.from_layout(X, Z)
    result = model.transform(Q)
    assert all(CompactMap(Z).contains(y) for y in result)
    np.testing.assert_array_equal(model.transform(Q[::-1]), result[::-1])
    np.testing.assert_array_equal(np.vstack([model.transform(q[None]) for q in Q]), result)
    np.testing.assert_array_equal(result[0], result[-1])
    moved = GraphEmbedding.from_layout(X, 4 * Z + np.array([8., -4.]))
    np.testing.assert_allclose((moved.transform(Q) - [8., -4.]) / 4, result,
                               atol=2e-6, rtol=0)
    for value, before in zip((X, Z, Q), originals):
        np.testing.assert_array_equal(value, before)
    now = np.random.get_state()
    assert ns[0] == now[0] and ns[2:] == now[2:]
    np.testing.assert_array_equal(ns[1], now[1])
    assert random.getstate() == rs
    assert model.diagnostics_['layout_origin'] == 'external'
    assert not model.diagnostics_['graph_available']


def test_factory_retains_all_historical_inside_queries_bitwise():
    X, Z = _fixture()
    Q = np.random.default_rng(19).normal(size=(30, 3))
    old = GraphEmbedding.from_layout(X, Z, mapping_domain='unbounded')
    bounded = GraphEmbedding.from_layout(X, Z, mapping_domain='train_hull')
    a, b = old.transform(Q), bounded.transform(Q)
    hull = CompactMap((Z - old.center_) / old.distance_unit_)
    inside = np.array([hull.contains((y - old.center_) / old.distance_unit_) for y in a])
    assert inside.any() and (~inside).any(), 'fixture must exercise both branches'
    assert a[inside].tobytes() == b[inside].tobytes()
    assert all(hull.contains((y - old.center_) / old.distance_unit_) for y in b)


@pytest.mark.parametrize('domain', ['train_hull', 'unbounded'])
def test_version3_persisted_policy_factory_serialized_exact_parity(tmp_path, domain):
    X, Z = _fixture()
    model = GraphEmbedding.from_layout(X, Z, mapping_domain=domain)
    path = tmp_path / 'predictor.npz'
    with pytest.warns(UserWarning, match='training features'):
        model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive['metadata'].tobytes().decode('utf8'))
    assert meta['version'] == 3
    assert meta['config']['mapping_domain'] == domain
    loaded = GraphEmbedding.load(path)
    Q = np.random.default_rng(19).normal(size=(12, 3))
    np.testing.assert_array_equal(loaded.transform(Q), model.transform(Q))
    np.testing.assert_array_equal(loaded.embedding_, model.embedding_)
