"""Analysis regressions and reproducible invariants, independent of native PH.

Run: PYTHONPATH=python python -m pytest tests/test_analysis.py
Native integration cases run when open_deep_tda._core is built/installed;
otherwise they are explicitly skipped, never substituted with approximate PH.
Verified pitfalls: tied duplicate points must exclude self from kNN; cover
components must restrict a FIXED graph, rather than rebuild kNN per interval;
raw topology costs must not disappear behind target scale normalization.
"""

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from open_deep_tda import datasets, evaluation, mapper, visualization


@pytest.mark.parametrize('name,dimension', [('circle', 2), ('figure_eight', 2),
    ('two_circles', 2), ('swiss_roll', 3), ('sphere', 3), ('torus', 3), ('blobs', 2)])
def test_datasets_reproducible_isometric(name, dimension):
    X, labels = datasets.make_dataset(name, 31, dimension, noise=0, seed=4)
    Y, other = datasets.make_dataset(name, 31, 9, noise=0, seed=4)
    assert X.shape == (31, dimension) and Y.shape == (31, 9)
    assert labels.dtype.kind in 'iu'
    np.testing.assert_array_equal(labels, other)
    np.testing.assert_allclose(pdist(X), pdist(Y), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(Y, datasets.make_dataset(name, 31, 9, 0, 4)[0])
    with pytest.raises(ValueError, match='cannot flatten'):
        datasets.make_dataset(name, 10, dimension - 1)


@pytest.mark.parametrize('kwargs', [{'name': 'unknown'}, {'n_samples': -1},
    {'n_samples': 2.5}, {'n_features': 0}, {'noise': -1}, {'noise': np.nan}])
def test_dataset_invalid(kwargs):
    with pytest.raises(ValueError):
        datasets.make_dataset(**kwargs)


def test_empty_dataset():
    X, labels = datasets.make_dataset(n_samples=0)
    assert X.shape == (0, 8) and labels.shape == (0,)


@pytest.fixture
def ph_spy(monkeypatch):
    """Plumbing spy only: NOT a replacement PH implementation or math oracle."""
    package = importlib.import_module('open_deep_tda')
    calls = []
    dimensions = []

    def persistence(D, **limits):
        calls.append((D.copy(), limits))
        return {'D': D, 'pairs': [], 'n_vertices': len(D), 'n_simplices': len(D),
                'reduction_operations': 0, 'peak_reduction_entries': 0, 'truncated': False}

    def diagram(result, dimension=1, **kwargs):
        dimensions.append(dimension)
        positive = result['D'][result['D'] > 0]
        if not len(positive):
            return np.empty((0, 2))
        if dimension == 0:
            return np.array([[0., positive.min()]])
        return np.array([[positive.min(), positive.max()]])

    # Use the real independent matching implementation, not a fake cost.
    from open_deep_tda.topology import diagram_matching, bottleneck_distance
    spy = SimpleNamespace(persistence=persistence, diagram=diagram,
                          diagram_matching=diagram_matching, bottleneck_distance=bottleneck_distance,
                          calls=calls, dimensions=dimensions)
    monkeypatch.setattr(package, 'topology', spy, raising=False)
    return spy


def test_evaluation_same_ids_bounded_raw_and_aligned(ph_spy):
    X = np.random.default_rng(12).normal(size=(10000, 4))
    report = evaluation.evaluate_embedding(X, 3 * X, topology_size=900, seed=7,
                                          budgets={'max_simplices': 700000, 'max_matching_size': 10})
    sampling = report['sampling']
    assert sampling['geometry_size'] == evaluation.GEOMETRY_SAMPLE_CAP
    assert sampling['topology_size'] == evaluation.TOPOLOGY_SAMPLE_CAP
    ids = sampling['topology_ids']
    np.testing.assert_allclose(ph_spy.calls[0][0], squareform(pdist(X[ids])))
    np.testing.assert_allclose(ph_spy.calls[1][0], 3 * ph_spy.calls[0][0])
    assert ph_spy.calls[0][1] == {'max_simplices': 700000}
    assert ph_spy.calls[1][1] == ph_spy.calls[0][1]
    assert ph_spy.dimensions == [0, 0, 1, 1]
    assert report['geometry']['target_scale'] == pytest.approx(1 / 3)
    assert report['geometry']['raw']['distance_rmse'] > 0
    assert report['geometry']['scale_aligned']['distance_rmse'] < 1e-12
    for dim in ('h0', 'h1'):
        assert report['topology'][dim]['raw_transport_squared'] > 0
        assert report['topology'][dim]['scale_aligned_transport_squared'] < 1e-24
    json.dumps(report, allow_nan=False)
    second = evaluation.evaluate_embedding(X, 3 * X, topology_size=900, seed=7)
    assert second['sampling'] == sampling


@pytest.mark.parametrize('n', [0, 1, 2, 3, 12])
def test_evaluation_constant_tiny_finite(ph_spy, n):
    report = evaluation.evaluate_embedding(np.ones((n, 3)), np.ones((n, 2)), k=30)
    json.dumps(report, allow_nan=False)
    assert report['geometry']['knn_overlap'] == 1
    assert report['geometry']['trustworthiness'] == 1
    assert report['geometry']['continuity'] == 1
    assert report['geometry']['raw']['distance_rmse'] == 0
    assert report['geometry']['target_scale'] == 1


def test_evaluation_collapsed_source_is_explicit(ph_spy):
    report = evaluation.evaluate_embedding(np.zeros((8, 2)), np.arange(16).reshape(8, 2))
    assert report['geometry']['source_constant']
    assert report['geometry']['raw']['normalized_stress'] is None
    assert report['geometry']['raw']['distance_rmse'] > 0
    assert report['geometry']['target_scale'] == 0
    json.dumps(report, allow_nan=False)


def test_evaluation_collapsed_target_not_masked(ph_spy):
    X = np.arange(16).reshape(8, 2)
    report = evaluation.evaluate_embedding(X, np.zeros((8, 2)))
    assert report['geometry']['target_constant']
    assert report['geometry']['raw']['normalized_stress'] == pytest.approx(1)
    assert report['geometry']['scale_aligned']['normalized_stress'] == pytest.approx(1)


def test_rank_metrics_match_independent_definition():
    rng = np.random.default_rng(5)
    A, B = squareform(pdist(rng.normal(size=(20, 4)))), squareform(pdist(rng.normal(size=(20, 2))))
    result = evaluation._geometry(A, B, 3)
    total = 0
    for i in range(20):
        source = sorted((j for j in range(20) if j != i), key=lambda j: A[i, j])
        target = sorted((j for j in range(20) if j != i), key=lambda j: B[i, j])
        for j in set(target[:3]) - set(source[:3]):
            total += source.index(j) + 1 - 3
    expected = 1 - 2 * total / (20 * 3 * (40 - 9 - 1))
    assert result['trustworthiness'] == pytest.approx(expected)
    assert 0 <= result['continuity'] <= 1
    tied, _ = evaluation._neighbors(np.zeros((8, 8)))
    assert all(i not in tied[i] for i in range(8))


def test_evaluation_resource_failures_propagate(ph_spy, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('budget exhausted')
    monkeypatch.setattr(ph_spy, 'persistence', fail)
    with pytest.raises(RuntimeError, match='budget'):
        evaluation.evaluate_embedding(np.ones((4, 2)), np.ones((4, 2)))


def test_evaluation_matching_failures_propagate(ph_spy):
    X = np.arange(8).reshape(4, 2)
    with pytest.raises(RuntimeError, match='matching'):
        evaluation.evaluate_embedding(X, X, budgets={'max_matching_size': 1})


def test_evaluation_censored_is_not_success(ph_spy, monkeypatch):
    monkeypatch.setattr(ph_spy, 'persistence', lambda *a, **kw: {'pairs': [], 'truncated': True})
    with pytest.raises(ValueError, match='uncensored'):
        evaluation.evaluate_embedding(np.ones((4, 2)), np.ones((4, 2)))


@pytest.mark.parametrize('X,Y', [(np.ones((3, 2)), np.ones((4, 2))),
    (np.array([[np.nan]]), np.zeros((1, 1))), (np.ones(3), np.ones((3, 1)))])
def test_evaluation_invalid(X, Y):
    with pytest.raises(ValueError):
        evaluation.evaluate_embedding(X, Y)


def test_mapper_fixed_graph_and_cover_components():
    # Two disconnected close pairs; rebuilding per-cover kNN would join them.
    X = np.array([[0.], [0.1], [10.], [10.1]])
    result = mapper.mapper_graph(X, lens=np.zeros(4), n_neighbors=1)
    assert [node['members'] for node in result['nodes']] == [[0, 1], [2, 3]]
    assert result['edges'] == []
    assert result['cover']['n_intervals_effective'] == 1
    assert 'not an exact Reeb' in result['kind']
    json.dumps(result, allow_nan=False)
    # Each preimage excludes every point's fixed nearest neighbor. Rebuilding
    # kNN inside each preimage would incorrectly connect [0,2] and [1,3].
    split = mapper.mapper_graph(X, lens=[0, 1, 0, 1], n_intervals=2,
                                overlap=0.2, n_neighbors=1)
    assert sorted(node['members'] for node in split['nodes']) == [[0], [1], [2], [3]]
    assert split['edges'] == []


def test_mapper_overlaps_are_exact_intersections():
    X = np.arange(9.)[:, None]
    result = mapper.mapper_graph(X, n_intervals=3, overlap=.5, n_neighbors=2)
    nodes = result['nodes']
    assert len(nodes) == 3
    assert set().union(*(set(node['members']) for node in nodes)) == set(range(9))
    expected = {}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            intersection = sorted(set(a['members']) & set(b['members']))
            if intersection:
                expected[(a['id'], b['id'])] = intersection
    actual = {(edge['source'], edge['target']): edge['members'] for edge in result['edges']}
    assert actual == expected and actual


def test_mapper_empty_singleton_callable_and_invalid():
    assert mapper.mapper_graph(np.zeros((0, 2)))['nodes'] == []
    result = mapper.mapper_graph(np.zeros((1, 2)), lens=lambda x: x[:, 0])
    assert result['nodes'][0]['members'] == [0]
    for kwargs in ({'overlap': 1}, {'n_intervals': 0}, {'n_neighbors': -1}, {'lens': [1, 2]}):
        with pytest.raises(ValueError):
            mapper.mapper_graph(np.zeros((1, 2)), **kwargs)


def test_report_offline_escaped_reuses_diagrams(tmp_path, monkeypatch):
    metrics = {'topology': {key: {'source_diagram': [[0, 1]], 'target_diagram': [[0, 2]]}
                            for key in ('h0', 'h1')}, 'note': '</pre><script>alert(1)</script>'}
    def fail(*args, **kwargs):
        raise AssertionError('supplied diagrams should avoid PH recomputation')
    monkeypatch.setattr(visualization, 'evaluate_embedding', fail)
    path = visualization.save_report(np.eye(3), np.eye(3), tmp_path / 'report.html',
        labels=['<script>', 'b', 'c'], metrics=metrics, mapper={'note': '<img onerror="bad">'})
    text = path.read_text()
    assert text.count('<svg ') == 4
    assert '<script' not in text and '<img' not in text
    assert '&lt;script&gt;' in text and '&lt;img' in text
    assert 'q=3' in text and 'not an exact Reeb' in text
    assert 'src=' not in text and 'cdn' not in text.lower()
    assert 'H0 persistence' in text and 'H1 persistence' in text


def test_report_computes_bounded_metrics(tmp_path, ph_spy):
    X = np.random.default_rng(1).normal(size=(1100, 3))
    path = visualization.save_report(X, X[:, :2], tmp_path / 'nested' / 'report.html', metrics={'custom': 2})
    text = path.read_text()
    assert '1000 of 1100' in text and 'Additional supplied metrics' in text
    assert len(ph_spy.calls) == 2 and len(ph_spy.calls[0][0]) == 64


def _native_topology():
    pytest.importorskip('open_deep_tda._core')
    return importlib.import_module('open_deep_tda.topology')


def test_native_square_ph_dimension_and_scale():
    _native_topology()
    X = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    result = evaluation.evaluate_embedding(X, 2 * X, topology_size=4)
    np.testing.assert_allclose(result['topology']['h1']['source_diagram'], [[1, np.sqrt(2)]])
    np.testing.assert_allclose(result['topology']['h1']['target_diagram'], [[2, 2 * np.sqrt(2)]])
    assert result['topology']['h0']['source_bars'] == 3
    for key in ('h0', 'h1'):
        assert result['topology'][key]['raw_transport_squared'] > 0
        assert result['topology'][key]['scale_aligned_transport_squared'] == pytest.approx(0)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('n', [0, 1, 2, 5])
def test_native_degenerate_json_and_report(n, tmp_path):
    _native_topology()
    X = np.zeros((n, 3))
    result = evaluation.evaluate_embedding(X, X[:, :2])
    json.dumps(result, allow_nan=False)
    assert result['topology']['h1']['source_bars'] == 0
    assert visualization.save_report(X, X[:, :2], tmp_path / 'report.html', metrics=result).is_file()


def test_native_budget_failure():
    _native_topology()
    X = np.arange(8.).reshape(4, 2)
    with pytest.raises(RuntimeError):
        evaluation.evaluate_embedding(X, X, budgets={'max_simplices': 1})
