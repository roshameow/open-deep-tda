"""Experimental bounded dense precomputed-dissimilarity graph embedding.

Not a metric-learning, out-of-sample accuracy or persistent-homology guarantee.
The live standalone graph worker uses D rows only as distance-profile features
for disconnected-component centroid/PCA initialization, never for neighbors.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import graph_embedding as _graph
from ._graph_mapping import CompactMap
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS',
               'BLIS_NUM_THREADS')


def _integer(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= value <= np.iinfo(np.int64).max):
        raise ValueError('invalid ' + name)
    return int(value)


def _matrix(value, width, cfg, square=False):
    # Require ndarray metadata so shape/type/budgets precede conversion/allocation.
    # Views/read-only arrays are accepted, then privately copied; masks/subclasses
    # are refused rather than trusting custom conversion or missingness semantics.
    if type(value) is not np.ndarray or value.ndim != 2 or value.dtype.kind not in 'iuf':
        raise ValueError('expected ordinary real numeric ndarray matrix')
    n, m = value.shape
    if m != width or (square and (n != m or not 16 <= n <= cfg['max_reference'])):
        raise ValueError('invalid distance matrix shape / reference budget')
    if value.size > cfg['max_feature_entries']:
        raise ValueError('max_feature_entries exceeded')
    if max(value.nbytes, value.size * 8) > cfg['max_matrix_bytes']:
        raise ValueError('max_matrix_bytes exceeded')
    # Validate in bounded rows without a full-size boolean/symmetry temporary.
    for i, row in enumerate(value):
        if not np.isfinite(row).all() or np.any(row < 0):
            raise ValueError('distances must be finite and nonnegative')
        if square and (row[i] != 0 or not np.array_equal(row, value[:, i])):
            raise ValueError('distances must be exactly symmetric with zero diagonal')
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.array(value, dtype=np.float64, order='C', copy=True)
    if not np.isfinite(result).all():
        raise ValueError('distances overflow float64')
    return result


def _source15(d):
    ids = np.empty((len(d), 15), dtype=np.int64)
    distances = np.empty((len(d), 15), dtype=np.float64)
    for i, row in enumerate(d):
        # Exclude self by ID (not by zero value); retain all duplicate distances.
        order = np.argsort(row, kind='stable')
        ids[i] = order[order != i][:15]
        distances[i] = row[ids[i]]
    return ids, distances


class PrecomputedGraphEmbedding:
    """Experimental dense TRAIN dissimilarities and independent QxTRAIN queries.

    Fit requires an ordinary real ndarray, exactly symmetric, zero-diagonal,
    finite and nonnegative, with 16..2000 rows. Triangle inequalities are NOT
    required or certified; duplicates and nonmetric dissimilarities are allowed.
    Stable source15 neighbors come directly from D, excluding self by row ID.
    On connected graphs initialization depends on G/seed; disconnected component
    centers use PCA of mean D-row distance profiles (with the live worker's
    coincident-center rule), not raw-feature centroids or intrinsic geometry.

    Cross-distance columns MUST have exactly the fitted TRAIN row identity/order
    and use the same dissimilarity convention. Shape is checked, but numerical
    matrices alone cannot authenticate identity or provenance. No raw features
    are needed or retained. Matching TRAIN rows are still treated as new queries,
    not promised to interpolate their fitted coordinates. Mapping uses the live
    all-anchor conditional Cauchy KL and train-hull CompactMap, without an OOS,
    metric, triangle-inequality, topology/PH or global-optimum guarantee.

    Only defensive, read-only layout/source IDs and bounded diagnostics remain
    fitted: these are sensitive, not anonymous. No checkpoint save/load exists.
    Private worker scratch is deleted on success, failure and timeout (not secure
    erasure or protection from abrupt process death). Fit timeout covers the
    child, not parent validation/serialization; transform timeout is cooperative.
    Per-matrix/entry/event budgets are not a peak-RSS cap. Fit is transactional;
    concurrent calls or hostile input/private-state mutation are unsupported.
    """
    def __init__(self, *, max_reference=2000, max_matrix_bytes=128 * 1024**2,
                 max_feature_entries=4096**2, max_edges=122880,
                 max_sgd_events=500_000_000, epochs=300, negative_rate=5,
                 seed=0, fit_timeout=900, transform_timeout=900):
        cfg = {k: _integer(v, k, 0 if k in ('seed', 'negative_rate') else 1)
               for k, v in locals().copy().items() if k not in ('self', 'fit_timeout', 'transform_timeout')}
        if not 16 <= cfg['max_reference'] <= 2000 or cfg['seed'] > 2**32 - 1:
            raise ValueError('max_reference must be 16..2000; seed must fit uint32')
        cfg['fit_timeout'] = _graph._positive_real(fit_timeout, 'fit_timeout')
        cfg['transform_timeout'] = _graph._positive_real(transform_timeout, 'transform_timeout')
        self._config = cfg
        self._state = None

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def _validated_state(self):
        state = self._state
        if state is None:
            raise ValueError('fit required')
        z, ids = state['layout'], state['ids']
        for a, dtype in ((z, np.float64), (ids, np.int64)):
            if (type(a) is not np.ndarray or a.dtype != dtype or a.ndim != 2
                    or not a.flags.owndata or not a.flags.c_contiguous
                    or a.flags.writeable):
                raise ValueError('invalid owned fitted state')
        n = len(z)
        if not 16 <= n <= self._config['max_reference'] or z.shape != (n, 2) or ids.shape != (n, 15):
            raise ValueError('invalid fitted state shape')
        if not np.isfinite(z).all() or np.any(ids < 0) or np.any(ids >= n):
            raise ValueError('invalid fitted state values')
        for i, row in enumerate(ids):
            if i in row or len(set(row)) != 15:
                raise ValueError('invalid fitted source IDs')
        return state

    @property
    def embedding_(self):
        return self._validated_state()['layout'].copy()

    @property
    def source_ids_(self):
        return self._validated_state()['ids'].copy()

    @property
    def diagnostics_(self):
        return copy.deepcopy(self._validated_state()['diagnostics'])

    def fit(self, D_square):
        """Fit transactionally; failures leave any previous fitted state intact."""
        cfg = self._config
        if type(D_square) is not np.ndarray or D_square.ndim != 2:
            raise ValueError('expected square ndarray')
        n = len(D_square)
        # Conservative resource preflight before canonical matrix allocation.
        if 30 * n > cfg['max_edges']:
            raise ValueError('max_edges exceeded')
        if cfg['epochs'] * 30 * n * (2 + cfg['negative_rate']) > cfg['max_sgd_events']:
            raise ValueError('max_sgd_events conservative bound exceeded')
        d = _matrix(D_square, n, cfg, square=True)
        ids, distances = _source15(d)
        worker_cfg = dict(max_vertices=cfg['max_reference'],
                          **{k: cfg[k] for k in ('max_feature_entries', 'max_edges',
                             'max_sgd_events', 'epochs', 'negative_rate', 'seed')})
        env = os.environ.copy()
        env.update({k: '1' for k in THREAD_KEYS})
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        # TemporaryDirectory removes sensitive scratch on success/failure/timeout.
        # subprocess.run kills and reaps the sole worker on timeout.
        with tempfile.TemporaryDirectory(prefix='precomputed-graph-') as directory:
            out = Path(directory)
            np.save(out / 'features.npy', d, allow_pickle=False)
            np.savez(out / 'source.npz', ids=ids, distances=distances)
            (out / 'config.json').write_text(json.dumps(worker_cfg))
            with (out / 'worker.log').open('wb') as log:
                proc = subprocess.run([sys.executable, str(Path(__file__).with_name('_graph_worker.py').resolve()), directory],
                                      env=env, stdout=log, stderr=log, timeout=cfg['fit_timeout'])
            if proc.returncode:
                # Bounded error, no input matrices retained in exceptions.
                with (out / 'worker.log').open('rb') as log:
                    log.seek(max(0, (out / 'worker.log').stat().st_size - 4096))
                    tail = log.read(4096).decode(errors='replace')
                raise RuntimeError('graph worker failed: ' + tail)
            z = np.load(out / 'embedding.npy', allow_pickle=False)
            status = json.loads((out / 'worker_status.json').read_text())
            if (z.shape != (n, 2) or z.dtype != np.float64 or not np.isfinite(z).all()
                    or status.get('status') != 'completed' or not status.get('torch_absent')):
                raise RuntimeError('invalid worker output')
        # Check mapper normalization now, before committing any replacement state.
        _graph._normalization(z, ids)
        status.update(experimental=True, source_policy='exact stable direct-D source15',
                      component_features='D-row distance profiles (not raw features)',
                      mapping_domain='train_hull', resource_budget_is_rss_cap=False)
        z = np.array(z, copy=True, order='C')
        ids = np.array(ids, copy=True, order='C')
        z.flags.writeable = ids.flags.writeable = False
        self._state = dict(layout=z, ids=ids, diagnostics=status)
        return self

    def fit_transform(self, D_square):
        return self.fit(D_square).embedding_

    def transform(self, D_cross, *, return_diagnostics=False):
        """Map QxTRAIN with strict fitted-column identity; no query-query geometry.

        Identity/order is a caller obligation, not inferable from distances.
        Empty batches return (0, 2); optional diagnostics have bounded size.
        """
        state = self._validated_state()  # reject corrupted ownership before copying query
        cross = _matrix(D_cross, len(state['layout']), self._config)
        result, diag = _map_queries(state['layout'], state['ids'], cross,
                                    self._config['transform_timeout'])
        return (result, diag) if return_diagnostics else result


def _map_queries(z, ids, cross, timeout):
    """Map already validated, owned arrays using live shared query arithmetic."""
    started = time.perf_counter()
    n = len(z)
    center, unit, anchors = _graph._normalization(z, ids)
    compact = CompactMap(anchors, max_vertices=2000)
    result = np.empty((len(cross), 2), dtype=np.float64)
    diag = dict(status='completed', query_count=len(cross), full_anchor_count=n,
                mapping_domain='train_hull', distance_unit=unit,
                query_policy='exact stable top15 of supplied reference distances',
                nonconverged=0,
                unbounded_nonconverged=0, constrained_queries=0,
                constrained_nonconverged=0, objective_calls=0,
                max_final_violation=0.0, max_projected_gradient_l2=0.0,
                hull_tolerance=compact.tolerance, mapper_options=_graph.OPTIONS.copy())

    def deadline():
        if time.perf_counter() - started > timeout:
            raise TimeoutError('precomputed mapper cooperative timeout exceeded')

    for row in range(len(cross)):
        deadline()
        d = np.asarray(cross[row:row+1], dtype=np.float64)
        if not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError('cross distances must be finite and nonnegative')
        ii = np.argsort(d, axis=1, kind='stable')[:, :15]
        dd = np.take_along_axis(d, ii, axis=1)
        p = _graph._weights(dd)
        exact = dd == 0
        if exact.any():
            p[0] = exact[0]
        p /= p.sum(axis=1)[:, None]
        bary = np.einsum('ij,ijk->ik', p, anchors[ii])[0]
        positive_ids, p = ii[0], p[0]
        calls = 0

        def fg(y):
            nonlocal calls
            deadline()
            calls += 1
            # CompactMap.cauchy has the same operation ordering as production fg.
            return compact.cauchy(y, positive_ids, p)

        fg(bary)
        opt = _graph.minimize(fg, bary.copy(), jac=True, method='L-BFGS-B', options=_graph.OPTIONS.copy())
        fg(opt.x)
        chosen, rec = compact.solve(opt.x, positive_ids, p, bary, check=deadline)
        deadline()
        result[row] = chosen * unit + center
        if not np.isfinite(result[row]).all():
            raise RuntimeError('nonfinite query output')
        constrained = not rec['old_inside']
        success = rec['solver_success'] if constrained else bool(opt.success)
        diag['nonconverged'] += int(not success)
        diag['unbounded_nonconverged'] += int(not opt.success)
        diag['constrained_queries'] += int(constrained)
        diag['constrained_nonconverged'] += int(rec['solver_success'] is False)
        diag['objective_calls'] += calls + rec['objective_calls']
        diag['max_final_violation'] = max(diag['max_final_violation'], rec['final_violation'])
        diag['max_projected_gradient_l2'] = max(diag['max_projected_gradient_l2'], rec['projected_gradient_l2'])
    diag['objective_anchor_evaluations'] = n * diag['objective_calls']
    diag['seconds'] = time.perf_counter() - started
    return result, diag
