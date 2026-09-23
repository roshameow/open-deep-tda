"""Synthetic-only contract checks; no quality search or external datasets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from open_deep_tda import precomputed_graph as ga
from open_deep_tda.graph_embedding import GraphEmbedding

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / 'python/open_deep_tda/_graph_worker.py'


def data(n=32):
    x = np.random.default_rng(12).normal(size=(n, 4))
    return x, cdist(x, x)


def reference(tmp_path, features, d):
    ids, distances = ga._source15(d)
    np.save(tmp_path / 'features.npy', features)
    np.savez(tmp_path / 'source.npz', ids=ids, distances=distances)
    cfg = dict(max_vertices=2000, max_feature_entries=4096**2, max_edges=122880,
               max_sgd_events=500_000_000, epochs=2, negative_rate=5, seed=0)
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    env = os.environ.copy()
    env.update({key: '1' for key in ga.THREAD_KEYS})
    run = subprocess.run([sys.executable, str(PRODUCTION), str(tmp_path)],
                         env=env, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr.decode()
    return (np.load(tmp_path / 'embedding.npy'),
            np.load(tmp_path / 'spectral_initial.npy'),
            json.loads((tmp_path / 'worker_status.json').read_text()))


@pytest.fixture(scope='module')
def fitted():
    _, d = data()
    return ga.PrecomputedGraphEmbedding(epochs=2).fit(d)


def capture_worker(monkeypatch):
    """Observe LIVE worker files before ephemeral scratch is removed."""
    original = subprocess.run
    captured = {}
    def run(args, **kwargs):
        assert Path(args[1]) == PRODUCTION.resolve()
        assert all(kwargs['env'][k] == '1' for k in ga.THREAD_KEYS)
        proc = original(args, **kwargs)
        out = Path(args[2])
        if proc.returncode == 0:
            captured['initial'] = np.load(out / 'spectral_initial.npy')
            with np.load(out / 'graph.npz') as archive:
                captured['graph'] = {k: archive[k].copy() for k in archive.files}
        return proc
    monkeypatch.setattr(subprocess, 'run', run)
    return captured


def test_connected_exact_raw_feature_kernel_parity(tmp_path, monkeypatch):
    x, d = data()
    z, initial, status = reference(tmp_path, x, d)
    with np.load(tmp_path / 'graph.npz') as archive:
        expected_graph = {k: archive[k].copy() for k in archive.files}
    captured = capture_worker(monkeypatch)
    fitted = ga.PrecomputedGraphEmbedding(epochs=2).fit(d)
    assert status['components'] == fitted.diagnostics_['components'] == 1
    assert z.tobytes() == fitted.embedding_.tobytes()
    assert initial.tobytes() == captured['initial'].tobytes()
    for key in expected_graph:
        assert np.array_equal(expected_graph[key], captured['graph'][key])


def test_disconnected_distance_profile_initialization(tmp_path, monkeypatch):
    d = np.full((32, 32), 100.)
    d[:16, :16] = d[16:, 16:] = 1.
    np.fill_diagonal(d, 0)
    z, initial, status = reference(tmp_path, d, d)
    captured = capture_worker(monkeypatch)
    model = ga.PrecomputedGraphEmbedding(epochs=2).fit(d)
    assert status['components'] == 2
    assert status['center_rule'] == 'sourcecentroidPCA'
    assert z.tobytes() == model.embedding_.tobytes()
    assert initial.tobytes() == captured['initial'].tobytes()
    assert model.diagnostics_['component_features'].startswith('D-row distance profiles')


def test_direct_distances_not_profile_neighbors(fitted):
    _, d = data()
    direct, sd = ga._source15(d)
    profiles, _ = ga._source15(cdist(d, d))
    assert not np.array_equal(direct, profiles)
    assert np.array_equal(fitted.source_ids_, direct)
    assert np.array_equal(sd, np.take_along_axis(d, direct, axis=1))


def test_zero_duplicates_ties_nonmetric():
    d = np.zeros((17, 17))
    ids, sd = ga._source15(d)
    for i in range(17):
        assert np.array_equal(ids[i], np.delete(np.arange(17), i)[:15])
    assert not sd.any()
    model = ga.PrecomputedGraphEmbedding(epochs=2)
    assert np.isfinite(model.fit_transform(d)).all()
    # Deliberate triangle inequality violation is accepted, not repaired.
    d = np.ones((17, 17)); np.fill_diagonal(d, 0)
    d[0, 1] = d[1, 0] = 10
    assert np.isfinite(model.fit_transform(d)).all()


def test_query_batch_independence_and_empty(fitted):
    _, d = data()
    q = np.vstack([d[0], d[0], d[3], np.zeros(len(d))])
    actual = fitted.transform(q)
    separate = np.vstack([fitted.transform(row[None, :]) for row in q])
    assert np.array_equal(actual, separate)
    assert np.array_equal(actual[::-1], fitted.transform(q[::-1]))
    assert fitted.transform(q[:0]).shape == (0, 2)


@pytest.mark.parametrize('kind', ['bool', 'complex', 'object', 'str', 'masked', 'list',
                                  'ndim', 'short', 'rect', 'nan', 'inf', 'negative',
                                  'asymmetric', 'diagonal'])
def test_invalid_fit_preserves_state(kind, fitted):
    _, d = data()
    if kind in ('bool', 'complex', 'object', 'str'):
        d = d.astype({'bool': bool, 'complex': complex, 'object': object, 'str': str}[kind])
    elif kind == 'masked': d = np.ma.array(d, mask=False)
    elif kind == 'list': d = d.tolist()
    elif kind == 'ndim': d = d[0]
    elif kind == 'short': d = d[:15, :15]
    elif kind == 'rect': d = d[:, :-1]
    elif kind == 'nan': d[0, 1] = np.nan
    elif kind == 'inf': d[0, 1] = np.inf
    elif kind == 'negative': d[0, 1] = -1
    elif kind == 'asymmetric': d[0, 1] += .1
    elif kind == 'diagonal': d[0, 0] = 1
    old = fitted.embedding_
    with pytest.raises(ValueError): fitted.fit(d)
    assert np.array_equal(old, fitted.embedding_)


@pytest.mark.parametrize('budget', [dict(max_matrix_bytes=8191), dict(max_feature_entries=1023),
                                    dict(max_reference=16), dict(max_edges=959),
                                    dict(max_sgd_events=13439)])
def test_budgets_before_copy(budget, monkeypatch):
    _, d = data()
    model = ga.PrecomputedGraphEmbedding(epochs=2, **budget)
    def no_copy(*args, **kwargs):
        raise AssertionError('copy before budget validation')
    monkeypatch.setattr(ga.np, 'array', no_copy)
    with pytest.raises(ValueError): model.fit(d)


def test_exact_budget_boundary():
    _, d = data(16)
    m = ga.PrecomputedGraphEmbedding(epochs=2, max_reference=16,
        max_matrix_bytes=d.nbytes, max_feature_entries=d.size, max_edges=480,
        max_sgd_events=2*480*7)
    assert m.fit_transform(d).shape == (16, 2)


@pytest.mark.parametrize('bad', [dict(max_reference=2001), dict(max_reference=15),
    dict(max_reference=True), dict(epochs=0), dict(negative_rate=-1), dict(seed=2**32),
    dict(transform_timeout=0), dict(transform_timeout=True), dict(fit_timeout=np.inf), dict(fit_timeout=True), dict(max_matrix_bytes=1.5)])
def test_invalid_config(bad):
    with pytest.raises(ValueError): ga.PrecomputedGraphEmbedding(**bad)


def test_defensive_properties_and_no_distances_retained(fitted):
    old = fitted.embedding_
    fitted.embedding_[:] = 0
    fitted.source_ids_[:] = -1
    fitted.config['epochs'] = 0
    fitted.diagnostics_['component_sizes'][0] = -1
    assert np.array_equal(old, fitted.embedding_)
    assert fitted.config['epochs'] == 2
    assert fitted.diagnostics_['component_sizes'][0] > 0
    assert set(fitted._state) == {'layout', 'ids', 'diagnostics'}


def test_input_views_copied():
    _, d = data(16)
    d.flags.writeable = False
    m = ga.PrecomputedGraphEmbedding(epochs=2).fit(d.T)
    old = m.embedding_
    d.flags.writeable = True
    d[:] = 0
    assert np.array_equal(old, m.embedding_)


def test_owned_state_validation_before_query_allocation(fitted):
    original = fitted._state
    try:
        for field, bad in [('layout', original['layout'].view()),
                           ('ids', original['ids'].astype(np.int32)),
                           ('ids', original['ids'][:1])]:
            fitted._state = dict(original, **{field: bad})
            with pytest.raises(ValueError, match='state'):
                fitted.transform(object())
    finally:
        fitted._state = original


@pytest.mark.parametrize('kind', ['negative', 'nan', 'complex', 'wrong_width', 'masked'])
def test_invalid_queries(kind, fitted):
    _, d = data()
    q = d[:2].copy()
    if kind == 'negative': q[0, 0] = -1
    if kind == 'nan': q[0, 0] = np.nan
    if kind == 'complex': q = q.astype(complex)
    if kind == 'wrong_width': q = q[:, :-1]
    if kind == 'masked': q = np.ma.array(q)
    with pytest.raises(ValueError): fitted.transform(q)


def test_timeout_cleanup_transaction(tmp_path, monkeypatch, fitted):
    original_cfg = fitted._config.copy()
    old = fitted.embedding_
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    try:
        fitted._config['fit_timeout'] = .000001
        with pytest.raises(subprocess.TimeoutExpired): fitted.fit(data()[1])
        assert not list(tmp_path.iterdir())
        assert np.array_equal(old, fitted.embedding_)
    finally:
        fitted._config = original_cfg


def test_worker_failure_cleanup_transaction(tmp_path, monkeypatch, fitted):
    old = fitted.embedding_
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    def failed(*args, **kwargs): return subprocess.CompletedProcess(args[0], 9)
    monkeypatch.setattr(subprocess, 'run', failed)
    with pytest.raises(RuntimeError, match='worker failed'): fitted.fit(data()[1])
    assert not list(tmp_path.iterdir())
    assert np.array_equal(old, fitted.embedding_)


def test_no_parent_numba_torch_and_thread_isolation():
    code = '''
import importlib.abc, sys, os
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch', 'numba'):
   raise RuntimeError('forbidden parent import: ' + fullname)
sys.meta_path.insert(0, Block())
from open_deep_tda.precomputed_graph import PrecomputedGraphEmbedding, THREAD_KEYS
import numpy as np
before = {k: os.environ.get(k) for k in THREAD_KEYS}
d = np.ones((16, 16)); np.fill_diagonal(d, 0)
m = PrecomputedGraphEmbedding(epochs=2).fit(d)
assert m.diagnostics_['torch_absent'] and m.diagnostics_['threads'] == 1
assert m.transform(d[:1]).shape == (1, 2)
assert all(os.environ.get(k) == v for k, v in before.items())
assert 'numba' not in sys.modules and 'torch' not in sys.modules
'''
    run = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr.decode()


def test_postworker_failure_is_transactional(tmp_path, monkeypatch, fitted):
    old = fitted.embedding_
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    def bad_normalization(*args):
        raise ValueError('forced normalization failure')
    monkeypatch.setattr(ga._graph, '_normalization', bad_normalization)
    with pytest.raises(ValueError, match='forced normalization'):
        fitted.fit(data()[1])
    assert not list(tmp_path.iterdir())
    assert np.array_equal(old, fitted.embedding_)


def test_query_budget_before_copy(fitted, monkeypatch):
    original_cfg = fitted._config.copy()
    q = np.ones((1, 32))
    try:
        fitted._config['max_matrix_bytes'] = q.nbytes - 1
        def no_copy(*args, **kwargs):
            raise AssertionError('query copied before budget check')
        monkeypatch.setattr(ga.np, 'array', no_copy)
        with pytest.raises(ValueError, match='max_matrix_bytes'):
            fitted.transform(q)
    finally:
        fitted._config = original_cfg


def test_not_fitted_and_dtype_conversion():
    model = ga.PrecomputedGraphEmbedding()
    with pytest.raises(ValueError, match='fit required'):
        model.transform(np.empty((0, 16)))
    for dtype in (np.float32, np.int64, np.uint8):
        d = np.ones((16, 16), dtype=dtype); np.fill_diagonal(d, 0)
        canonical = ga._matrix(d, 16, model.config, square=True)
        assert canonical.dtype == np.float64 and canonical.flags.owndata
        assert not np.shares_memory(d, canonical)


def test_live_query_mapper_parity_and_shared_functions(fitted, monkeypatch):
    x, d = data()
    q = np.vstack([x[:2], np.random.default_rng(1).normal(size=(3, 4))])
    cross = cdist(q, x)
    ids, distances = ga._source15(d)
    expected, ed = GraphEmbedding.from_layout(
        x, fitted.embedding_, source_knn=(ids, distances)).transform(q, return_diagnostics=True)
    calls = dict(weights=0, normalization=0)
    for name in ('weights', 'normalization'):
        original = getattr(ga._graph, '_' + name)
        def wrapped(*args, _name=name, _original=original):
            calls[_name] += 1
            return _original(*args)
        monkeypatch.setattr(ga._graph, '_' + name, wrapped)
    actual, diag = fitted.transform(cross, return_diagnostics=True)
    assert calls == dict(weights=len(q), normalization=1)
    assert actual.tobytes() == expected.tobytes()
    assert diag['objective_calls'] == sum(r['calls'] for r in ed['queries'])
    assert diag['mapper_options'] == ga._graph.OPTIONS
    assert not hasattr(fitted, 'save') and not hasattr(fitted, 'load')


def test_query_timeout_is_cooperative_and_preserves_fit(fitted):
    old = fitted.embedding_
    timeout = fitted._config['transform_timeout']
    try:
        fitted._config['transform_timeout'] = 1e-12
        with pytest.raises(TimeoutError, match='cooperative timeout'):
            fitted.transform(data()[1][:1])
        assert np.array_equal(old, fitted.embedding_)
    finally:
        fitted._config['transform_timeout'] = timeout


def test_default_hard_reference_cap():
    assert ga.PrecomputedGraphEmbedding().config['max_reference'] == 2000
    with pytest.raises(ValueError, match='reference budget'):
        ga.PrecomputedGraphEmbedding().fit(np.zeros((2001, 2001)))


def test_live_zero_duplicate_and_tied_query_arithmetic():
    # 17 coincident references: stable cutoff and all-zero override matter.
    x = np.zeros((17, 3))
    d = cdist(x, x)
    ids, distances = ga._source15(d)
    z = np.random.default_rng(3).normal(size=(17, 2))
    q = np.array([[0., 0., 0.], [1., 0., 0.]])
    expected = GraphEmbedding.from_layout(x, z, source_knn=(ids, distances)).transform(q)
    actual, _ = ga._map_queries(z, ids, cdist(q, x), 900)
    assert actual.tobytes() == expected.tobytes()


def test_success_cleanup_and_query_ownership(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    _, d = data(16)
    model = ga.PrecomputedGraphEmbedding(epochs=2).fit(d)
    assert not list(tmp_path.iterdir())
    q = d[:2].copy()
    before = q.copy()
    q.flags.writeable = False
    y, diag = model.transform(q, return_diagnostics=True)
    assert np.array_equal(q, before)
    assert not np.shares_memory(y, model._state['layout'])
    y[:] = 0
    diag['mapper_options']['maxiter'] = -1
    assert ga._graph.OPTIONS['maxiter'] == 100
    assert not list(tmp_path.iterdir())
