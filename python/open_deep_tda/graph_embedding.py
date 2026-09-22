"""Source15 fuzzy graph layout and fixed all-anchor conditional KL mapper.

NumPy/SciPy parent; isolated standalone graph worker. No preprocessing, topology
preservation claim, global optimizer guarantee or silent approximate fallback.
Features and all fitted anchors remain necessary for prediction (not anonymous).
"""
import ast
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
import zipfile

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .neighbors import build_neighbor_graph
from ._graph_mapping import CompactMap

OPTIONS = dict(maxiter=100, maxls=30, ftol=1e-12, gtol=1e-8)
_SCHEMA = 'graph-embedding-predictor'
_VERSION = 3
_ARRAY_KEYS = {'metadata', 'X', 'Z', 'source_ids', 'source_distances', 'center', 'unit'}
_CONFIG_KEYS = {'n_components', 'neighbor_backend', 'max_vertices', 'max_edges',
                'max_sgd_events', 'max_feature_entries', 'fit_timeout',
                'transform_timeout', 'seed', 'epochs', 'negative_rate', 'mapping_domain'}

def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    if value > np.iinfo(np.int64).max:
        raise ValueError(name + ' exceeds int64')
    return int(value)


def _matrix(value, name, *, width=None, max_rows=None, max_entries=None, empty=False):
    # np.asarray silently discards MaskedArray missingness; never reinterpret
    # masked payloads as observed features. This guard precedes canonical copy.
    if np.ma.isMaskedArray(value):
        raise ValueError(name + ' masked arrays / missing values are unsupported')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            a = np.asarray(value)
    except (TypeError, ValueError, Warning) as exc:
        raise ValueError(name + ' must be a rectangular numeric matrix') from exc
    if a.ndim != 2 or a.shape[1] == 0 or (not empty and a.shape[0] == 0):
        raise ValueError(name + ' must be a nonempty numeric matrix')
    if a.dtype.kind not in 'iuf':
        raise ValueError(name + ' must contain real numbers, not bool/object/complex')
    if width is not None and a.shape[1] != width:
        raise ValueError(name + ' feature dimension mismatch')
    if max_rows is not None and len(a) > max_rows:
        raise ValueError('max_vertices exceeded')
    if max_entries is not None and a.size > max_entries:
        raise ValueError('max_feature_entries exceeded')
    with np.errstate(over='ignore', invalid='ignore'):
        a = np.array(a, dtype=np.float64, order='C', copy=True)
    if not np.isfinite(a).all():
        raise ValueError(name + ' contains nonfinite values')
    return a


def _weights(d):
    target = np.log2(d.shape[1])
    rho = np.min(np.where(d>0,d,np.inf),axis=1); rho[~np.isfinite(rho)] = 0
    gap = np.maximum(d-rho[:,None],0)
    lo = np.zeros(len(d)); hi = np.maximum(d.max(axis=1),1e-12)
    saturated = (gap==0).sum(axis=1)>=target
    for _ in range(64):
        mid = (lo+hi)/2; mass = np.exp(-gap/mid[:,None]).sum(axis=1)
        lo = np.where(mass<target,mid,lo); hi = np.where(mass>=target,mid,hi)
    w = np.exp(-gap/hi[:,None]); w[saturated] = (gap[saturated]==0)
    return w


def _exact_knn(Q, X, k=15):
    ids = np.empty((len(Q),k),dtype=np.int64); dist = np.empty((len(Q),k))
    for start in range(0,len(Q),64):
        d = cdist(Q[start:start+64],X)
        if not np.isfinite(d).all(): raise ValueError('distance overflow')
        order = np.argsort(d,axis=1,kind='stable')[:,:k]
        ids[start:start+len(d)] = order
        dist[start:start+len(d)] = np.take_along_axis(d,order,axis=1)
    return ids, dist


def _validate_knn(pair, Q, X, *, nonself=False):
    if not isinstance(pair, (tuple,list)) or len(pair)!=2:
        raise ValueError('knn must be (ids, distances)')
    if np.ma.isMaskedArray(pair[0]) or np.ma.isMaskedArray(pair[1]):
        raise ValueError('masked knn IDs/distances / missing values are unsupported')
    raw = np.asarray(pair[0]); d = np.asarray(pair[1])
    if raw.shape!=(len(Q),15) or raw.dtype.kind not in 'iu':
        raise ValueError('knn IDs must be integer (rows,15)')
    if d.shape!=raw.shape or d.dtype.kind not in 'iuf':
        raise ValueError('knn distances must be numeric (rows,15)')
    if np.any(raw>=len(X)) or np.any(raw<0): raise ValueError('knn ID out of range')
    ids = np.array(raw,dtype=np.int64,copy=True)
    d = np.array(d,dtype=np.float64,copy=True)
    if not np.isfinite(d).all() or np.any(d<0): raise ValueError('invalid knn distance')
    for i in range(len(Q)):
        if len(np.unique(ids[i]))!=15: raise ValueError('duplicate knn IDs')
        if nonself and i in ids[i]: raise ValueError('source knn contains self')
        actual = cdist(Q[i:i+1], X[ids[i]])[0]
        if (not np.isfinite(actual).all() or not np.allclose(d[i],actual,rtol=1e-10,atol=1e-12)
                or not np.array_equal(d[i]==0,actual==0)):
            raise ValueError('knn distances do not match features')
        # Recompute rather than trusting approximate distances as fuzzy inputs.
        order = np.lexsort((ids[i],actual))
        ids[i] = ids[i,order]; d[i] = actual[order]
    return ids, d


def _dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + ' must be a finite positive real')
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a finite positive real') from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be a finite positive real')
    return value


def _normalization(z, ids):
    # Keep this fixed operation ordering for exact predictor round-trips.
    center = z.mean(axis=0)
    lengths = np.linalg.norm(z[:, None, :] - z[ids], axis=2)
    positive = lengths[lengths > 0]
    if not len(positive) or not np.isfinite(lengths).all():
        raise ValueError('invalid TRAIN layout distance unit')
    unit = float(np.median(positive))
    anchors = (z - center) / unit
    if not np.isfinite(center).all() or not np.isfinite(anchors).all():
        raise ValueError('nonfinite normalized anchors')
    return center, unit, anchors


def _check_index(index):
    if index is not None and not callable(getattr(index, 'query', None)):
        raise ValueError('query_index must implement query(Q,k)')


def _log_tail(out):
    try:
        with (out / 'worker.log').open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 8192))
            return stream.read(8192).decode('utf8', errors='replace')
    except OSError:
        return ''


def _json_read(path, limit=65536):
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('JSON metadata exceeds byte budget')
    return _strict_json(data)


def _strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(data.decode('utf8'), object_pairs_hook=pairs, parse_constant=invalid)


class GraphEmbedding:
    """Two-dimensional source15 fuzzy-graph layout and all-anchor query mapper.

    The default exact neighbor backend requires no precomputed graph. Explicit
    ``neighbor_backend='pynndescent'`` uses the optional isolated ANN backend;
    neither backend silently substitutes another. Graph fitting needs Numba in
    the worker interpreter. :meth:`from_layout` can instead configure the same
    query mapper from supplied coordinates without running graph SGD.

    mapping_domain='train_hull' (default) constrains predictions to the convex
    support of the TRAIN anchors. The historical L-BFGS phase is retained when
    feasible; otherwise a separate SLSQP phase starts at the barycenter and
    retains the best feasible visited point, including that barycenter. It is
    bounded interpolation, not post-hoc clipping, and neither optimizer is a
    global-optimum certificate. Explicit 'unbounded' reproduces historical
    extrapolation, including possible extreme finite query positions.

    The graph/component initialization and conditional Cauchy KL are heuristic
    dimensionality-reduction methods, not topology-preservation certificates.
    ``transform`` treats every row as a new query, including matching training
    rows: exact-match barycenters are initialization, not interpolation promises.

    work_dir is only a scratch parent, never a retention request. Default fit
    scratch is cleaned on success AND failure; default transform writes no disk.
    Explicit debug_dir warns and retains per-invocation raw data/coordinates/logs.
    Separate instances are independent; concurrent calls on one are unsupported.

    Exact fit search is O(N²d); transform uses every fitted anchor at each
    objective evaluation. Budgets are not an RSS cap. External indices must be
    deterministic and refer to fitted row order; candidate distances are checked
    but global nearest membership is not certified. No index is ever pickled.
    """
    def __init__(self, *, n_components=2, neighbor_backend='exact', work_dir=None,
                 debug_dir=None, max_vertices=60000, max_edges=1800000,
                 max_sgd_events=4000000000, max_feature_entries=16000000,
                 fit_timeout=600, transform_timeout=900, seed=0, epochs=300,
                 negative_rate=5, mapping_domain='train_hull'):
        if _integer(n_components, 'n_components') != 2:
            raise ValueError('only n_components=2 is supported')
        if not isinstance(neighbor_backend, str) or neighbor_backend not in ('exact', 'pynndescent'):
            raise ValueError("neighbor_backend must be 'exact' or 'pynndescent'")
        if not isinstance(mapping_domain, str) or mapping_domain not in ('train_hull', 'unbounded'):
            raise ValueError("mapping_domain must be 'train_hull' or 'unbounded'")
        self._config = dict(n_components=2, neighbor_backend=neighbor_backend, mapping_domain=mapping_domain,
            max_vertices=_integer(max_vertices, 'max_vertices'),
            max_edges=_integer(max_edges, 'max_edges'),
            max_sgd_events=_integer(max_sgd_events, 'max_sgd_events'),
            max_feature_entries=_integer(max_feature_entries, 'max_feature_entries'),
            seed=_integer(seed, 'seed', 0), epochs=_integer(epochs, 'epochs'),
            negative_rate=_integer(negative_rate, 'negative_rate', 0),
            fit_timeout=_positive_real(fit_timeout, 'fit_timeout'),
            transform_timeout=_positive_real(transform_timeout, 'transform_timeout'))
        if self._config['seed'] > 2**32 - 1:
            raise ValueError('seed exceeds uint32')
        self._root = Path(work_dir).resolve() if work_dir is not None else None
        self._debug_root = Path(debug_dir).resolve() if debug_dir is not None else None
        if self._debug_root is not None:
            warnings.warn('Privacy: debug_dir retains raw training/query features, coordinates '
                          'and worker logs in UUID directories; remove them explicitly.',
                          UserWarning, stacklevel=2)
        self._state = None
        self._transform_diagnostics = None
        self._fit_diagnostics = None
        self._artifact_dir = None

    @contextmanager
    def _invocation(self, kind):
        if self._debug_root is not None:
            self._debug_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            out = self._debug_root / (kind + '-' + uuid.uuid4().hex)
            out.mkdir(mode=0o700)
            yield out
        elif kind == 'transform':
            yield None
        else:
            if self._root is not None:
                self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tempfile.TemporaryDirectory(prefix='graph-fit-', dir=self._root) as name:
                out = Path(name)
                out.chmod(0o700)
                yield out

    def _compact_domain(self, anchors):
        return (CompactMap(anchors, max_vertices=self._config['max_vertices'])
                if self._config['mapping_domain'] == 'train_hull' else None)

    def _require(self):
        if self._state is None:
            raise RuntimeError('GraphEmbedding is not fitted')
        return self._state

    def fit(self, X, *, source_knn=None, query_index=None):
        self._state = None
        self._fit_diagnostics = None
        self._transform_diagnostics = None
        self._artifact_dir = None
        start = time.perf_counter()
        with self._invocation('fit') as out:
            retained = str(out) if self._debug_root is not None else None
            self._artifact_dir = out if retained else None
            try:
                def remaining():
                    left = self._config['fit_timeout'] - (time.perf_counter() - start)
                    if left <= 0:
                        raise TimeoutError('fit timeout exceeded')
                    return left
                x = _matrix(X, 'X', max_rows=self._config['max_vertices'],
                            max_entries=self._config['max_feature_entries'])
                if len(x) < 16:
                    raise ValueError('source15 requires at least 16 vertices')
                if 30 * len(x) > self._config['max_edges']:
                    raise ValueError('max_edges conservative preflight exceeded')
                _check_index(query_index)
                neighbor_diag = None
                if source_knn is None:
                    built = build_neighbor_graph(x, 15, backend=self._config['neighbor_backend'],
                        seed=self._config['seed'], timeout_seconds=remaining())
                    raw_ids = built['neighbors']
                    # The backend returns IDs, not distances. Recompute float64
                    # Euclidean distances from our detached original features.
                    if (not isinstance(raw_ids, np.ndarray) or raw_ids.shape != (len(x), 15)
                            or raw_ids.dtype.kind not in 'iu' or np.any(raw_ids < 0)
                            or np.any(raw_ids >= len(x))):
                        raise ValueError('neighbor backend returned invalid IDs')
                    distances = np.empty(raw_ids.shape, dtype=np.float64)
                    for row in range(len(x)):
                        distances[row] = cdist(x[row:row+1], x[raw_ids[row]])[0]
                    ids, d = _validate_knn((raw_ids, distances), x, x, nonself=True)
                    neighbor_diag = built['diagnostics']
                    source_policy = self._config['neighbor_backend'] + ' backend; float64 distances recalculated'
                else:
                    ids, d = _validate_knn(source_knn, x, x, nonself=True)
                    source_policy = 'precomputed candidates; pair distances checked; global recall NOT audited'
                remaining()
                np.savez_compressed(out / 'source.npz', ids=ids, distances=d)
                np.save(out / 'features.npy', x, allow_pickle=False)
                _dump(out / 'config.json', self._config)
                env = os.environ.copy()
                for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS',
                            'BLIS_NUM_THREADS'):
                    env[key] = '1'
                env['PYTHONDONTWRITEBYTECODE'] = '1'
                worker = Path(__file__).with_name('_graph_worker.py').resolve()
                with (out / 'worker.log').open('w+b') as log:
                    run = subprocess.run([sys.executable, str(worker), str(out)], env=env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=remaining())
                if run.returncode:
                    detail = ''
                    try:
                        status = _json_read(out / 'worker_status.json')
                        if isinstance(status, dict):
                            detail = str(status.get('error', ''))[:8192]
                    except (OSError, ValueError, UnicodeError):
                        pass
                    raise RuntimeError('graph worker exited ' + str(run.returncode) + ': ' + detail)
                remaining()
                z = _matrix(np.load(out / 'embedding.npy', allow_pickle=False), 'embedding', width=2,
                            max_rows=len(x))
                initial = _matrix(np.load(out / 'spectral_initial.npy', allow_pickle=False), 'initial',
                                  width=2, max_rows=len(x))
                if len(z) != len(x) or len(initial) != len(x):
                    raise RuntimeError('worker shape mismatch')
                with np.load(out / 'graph.npz', allow_pickle=False) as archive:
                    graph = {key: archive[key].copy() for key in archive.files}
                checked_ids, checked_d = _validate_knn(
                    (graph['neighbors'], graph['distances']), x, x, nonself=True)
                if not np.array_equal(ids, checked_ids) or not np.array_equal(d, checked_d):
                    raise RuntimeError('worker changed source neighbors')
                center, unit, anchors = _normalization(z, ids)
                diag = _json_read(out / 'worker_status.json')
                if not isinstance(diag, dict) or diag.get('status') != 'completed':
                    raise RuntimeError('worker did not complete')
                diag.update(mapping_domain=self._config['mapping_domain'], layout_origin='graph_sgd', source_policy=source_policy, neighbor_diagnostics=neighbor_diag,
                    query_policy=('external candidates; distances checked; recall NOT audited'
                                  if query_index is not None else 'exact stable blocked search'),
                    distance_unit=unit, full_anchor_count=len(x), reference_bytes=x.nbytes,
                    anchor_bytes=anchors.nbytes, graph_array_bytes=sum(v.nbytes for v in graph.values()),
                    fit_wall_seconds=time.perf_counter()-start, artifact_dir=retained,
                    exact_search_block_rows=64, mapper='A exact all-anchor normalized Cauchy KL',
                    mapper_options=OPTIONS.copy(), resource_budget_is_rss_cap=False,
                    graph_available=True, spectral_initial_available=True)
                if retained:
                    _dump(out / 'status.json', diag)
                self._state = dict(x=x, z=z, initial=initial, graph=graph, ids=ids, d=d,
                    center=center, unit=unit, anchors=anchors, index=query_index, diagnostics=diag,
                    compact=self._compact_domain(anchors))
                self._fit_diagnostics = copy.deepcopy(diag)
            except BaseException as exc:
                tail = _log_tail(out)
                diag = dict(status='failed', error=str(exc), worker_log_tail=tail,
                            artifact_dir=retained)
                self._fit_diagnostics = diag
                if retained:
                    try:
                        _dump(out / 'status.json', diag)
                    except OSError:
                        pass
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                error_type = ValueError if isinstance(exc, (ValueError, TypeError)) else RuntimeError
                message = 'GraphEmbedding fit failed: ' + str(exc)
                if isinstance(exc, subprocess.TimeoutExpired):
                    message = 'GraphEmbedding fit timed out: ' + str(exc)
                if tail:
                    message += '\nWorker log tail:\n' + tail
                if retained:
                    message += '\nDebug artifacts retained: ' + retained
                error = error_type(message)
                error.diagnostics = copy.deepcopy(diag)
                raise error from exc
        return self

    def fit_transform(self, X, *, source_knn=None, query_index=None):
        return self.fit(X, source_knn=source_knn, query_index=query_index).embedding_

    @classmethod
    def from_layout(cls, X, embedding, *, source_knn=None, query_index=None, **options):
        """Construct the fixed query mapper from supplied two-dimensional coordinates.

        Copies and validates ``X`` and ``embedding`` without running spectral
        initialization or graph SGD. Row ``embedding[i]`` must correspond to
        ``X[i]``; the caller supplies that alignment. The TRAIN distance unit and
        all-anchor Cauchy mapper are identical to those of :meth:`fit`. When source_knn is
        omitted, the explicitly configured neighbor backend builds source15;
        supplied candidates undergo the same identity/distance validation as fit.
        ``options`` are constructor options (including resource budgets).

        This factory makes no topology, provenance, training-quality or
        certification assumption about the supplied layout. Diagnostics mark
        ``layout_origin='external'``; graph/spectral arrays are unavailable.
        Prediction always treats rows as new queries, even exact training rows.
        No index is serialized. The resulting predictor supports save/load and
        necessarily retains detached copies of the training features/coordinates.
        No raw inputs or coordinates are written by this factory.
        """
        model = cls(**options)
        started = time.perf_counter()
        try:
            x = _matrix(X, 'X', max_rows=model._config['max_vertices'],
                        max_entries=model._config['max_feature_entries'])
            if len(x) < 16:
                raise ValueError('source15 requires at least 16 vertices')
            z = _matrix(embedding, 'embedding', width=2,
                        max_rows=model._config['max_vertices'])
            if len(z) != len(x):
                raise ValueError('embedding must have one row for every training row')
            if 30 * len(x) > model._config['max_edges']:
                raise ValueError('max_edges conservative preflight exceeded')
            _check_index(query_index)
            neighbor_diag = None
            if source_knn is None:
                built = build_neighbor_graph(x, 15, backend=model._config['neighbor_backend'],
                    seed=model._config['seed'], timeout_seconds=model._config['fit_timeout'])
                raw_ids = built['neighbors']
                if (not isinstance(raw_ids, np.ndarray) or raw_ids.shape != (len(x), 15)
                        or raw_ids.dtype.kind not in 'iu' or np.any(raw_ids < 0)
                        or np.any(raw_ids >= len(x))):
                    raise ValueError('neighbor backend returned invalid IDs')
                distances = np.empty(raw_ids.shape, dtype=np.float64)
                for row in range(len(x)):
                    distances[row] = cdist(x[row:row+1], x[raw_ids[row]])[0]
                ids, d = _validate_knn((raw_ids, distances), x, x, nonself=True)
                neighbor_diag = built['diagnostics']
                source_policy = model._config['neighbor_backend'] + ' backend; float64 distances recalculated'
            else:
                ids, d = _validate_knn(source_knn, x, x, nonself=True)
                source_policy = 'precomputed candidates; pair distances checked; global recall NOT audited'
            if time.perf_counter() - started > model._config['fit_timeout']:
                raise TimeoutError('from_layout source-neighbor timeout exceeded')
            center, unit, anchors = _normalization(z, ids)
            diag = dict(status='completed', layout_origin='external', mapping_domain=model._config['mapping_domain'],
                source_policy=source_policy, neighbor_diagnostics=neighbor_diag,
                query_policy=('external candidates; distances checked; recall NOT audited'
                              if query_index is not None else 'exact stable blocked search'),
                distance_unit=unit, full_anchor_count=len(x), n_features=x.shape[1],
                reference_bytes=x.nbytes, anchor_bytes=anchors.nbytes,
                artifact_dir=None, graph_available=False, spectral_initial_available=False,
                mapper='A exact all-anchor normalized Cauchy KL', mapper_options=OPTIONS.copy(),
                resource_budget_is_rss_cap=False, construction_seconds=time.perf_counter()-started)
            model._state = dict(x=x, z=z, initial=None, graph=None, ids=ids, d=d,
                center=center, unit=unit, anchors=anchors, index=query_index, diagnostics=diag,
                compact=model._compact_domain(anchors))
            model._fit_diagnostics = copy.deepcopy(diag)
            return model
        except (ValueError, TypeError, RuntimeError, TimeoutError) as exc:
            error_type = ValueError if isinstance(exc, (ValueError, TypeError)) else RuntimeError
            error = error_type('GraphEmbedding.from_layout failed: ' + str(exc))
            error.diagnostics = dict(status='failed', layout_origin='external', error=str(exc), artifact_dir=None)
            raise error from exc

    def transform(self, Q, *, query_knn=None, return_diagnostics=False):
        """Map independent new queries; never replace matching rows with anchors.

        Optional ``query_knn=(ids, distances)`` has shape (len(Q),15), uses fit
        row IDs, and takes precedence over a live index/default exact search.
        All candidates are copied, checked against query/source geometry and
        canonically sorted; global top15 recall is not implied. Exact coordinate
        duplicates affect barycenter initialization only. The fixed optimization
        and all-anchor normalization are unchanged for every query route.
        """
        s = self._require()
        start = time.perf_counter(); records = []
        self._transform_diagnostics = None
        with self._invocation('transform') as out:
            try:
                q = _matrix(Q,'Q',width=s['x'].shape[1],empty=True,
                            max_entries=self._config['max_feature_entries'])
                if out is not None:
                    np.save(out/'queries.npy',q,allow_pickle=False)
                supplied = (_validate_knn(query_knn, q, s['x'])
                            if query_knn is not None else None)
                result = np.empty((len(q),2))
                def deadline():
                    if time.perf_counter()-start>self._config['transform_timeout']:
                        raise TimeoutError('transform cooperative timeout exceeded')
                for row in range(len(q)):
                    deadline(); one = q[row:row+1]
                    if supplied is not None:
                        ii, dd = supplied[0][row:row+1], supplied[1][row:row+1]
                    elif s['index'] is None:
                        ii, dd = _exact_knn(one,s['x'])
                    else:
                        # Never expose our retained features or the caller Q to index mutation.
                        pair = s['index'].query(one.copy(),15)
                        ii, dd = _validate_knn(pair,one,s['x'])
                    p = _weights(dd)
                    exact = dd==0
                    if exact.any(): p[0] = exact[0]
                    p /= p.sum(axis=1)[:,None]
                    bary = np.einsum('ij,ijk->ik',p,s['anchors'][ii])[0]
                    ids = ii[0]; p = p[0]; calls = 0
                    def fg(y):
                        nonlocal calls
                        deadline(); calls += 1
                        r = y-s['anchors']; ds = 1+np.einsum('ij,ij->i',r,r)
                        w = 1/ds; normalized = w/w.sum()
                        f = float(np.dot(p,np.log(ds[ids]))+np.log(w.sum())+
                                  np.dot(p[p>0],np.log(p[p>0])))
                        g = 2*((p*w[ids])@r[ids]-(normalized*w)@r)
                        if not np.isfinite(f) or not np.isfinite(g).all():
                            raise RuntimeError('nonfinite mapper objective/gradient')
                        return f,g
                    initial,_ = fg(bary)
                    opt = minimize(fg,bary.copy(),jac=True,method='L-BFGS-B',options=OPTIONS.copy())
                    f,g = fg(opt.x)
                    rec = dict(success=bool(opt.success),status=int(opt.status),message=str(opt.message),
                        iterations=int(opt.nit),calls=calls,initial_objective=initial,final_objective=f,
                        objective_delta=f-initial,gradient_linf=float(np.max(abs(g))),finite=True,
                        mapping_domain=self._config['mapping_domain'])
                    chosen = opt.x
                    if s['compact'] is not None:
                        deadline()
                        chosen, compact_record = s['compact'].solve(opt.x, ids, p, bary, check=deadline)
                        deadline()
                        rec['unbounded_phase'] = dict(success=bool(opt.success), status=int(opt.status),
                            iterations=int(opt.nit), calls=calls, objective=f, gradient_linf=float(np.max(abs(g))))
                        rec['compact_phase'] = compact_record
                        rec['calls'] = calls + compact_record['objective_calls']
                        rec['final_objective'] = compact_record['final_objective']
                        rec['objective_delta'] = rec['final_objective'] - initial
                        rec['gradient_linf'] = compact_record['gradient_linf']
                        rec['projected_gradient_l2'] = compact_record['projected_gradient_l2']
                        if not compact_record['old_inside']:
                            rec.update(success=compact_record['solver_success'], status=compact_record['solver_status'],
                                       message=compact_record['message'])
                    result[row] = chosen*s['unit']+s['center']
                    if not np.isfinite(result[row]).all(): raise RuntimeError('nonfinite query output')
                    records.append(rec)
                if out is not None:
                    np.save(out/'embedding.npy',result,allow_pickle=False)
                diag = dict(status='completed',queries=records,query_count=len(q),
                    query_policy=('precomputed candidates; distances checked; global recall NOT audited'
                                  if supplied is not None else
                                  'external candidates; distances checked; global recall NOT audited'
                                  if s['index'] is not None else 'exact stable blocked search'),
                    full_anchor_count=len(s['x']),objective_anchor_evaluations=len(s['x'])*sum(r['calls'] for r in records),
                    nonconverged=sum(not r['success'] for r in records),
                    mapping_domain=self._config['mapping_domain'],
                    unbounded_nonconverged=sum(not r.get('unbounded_phase', r)['success'] for r in records),
                    constrained_queries=sum(not r['compact_phase']['old_inside'] for r in records if 'compact_phase' in r),
                    constrained_nonconverged=sum(r['compact_phase']['solver_success'] is False for r in records if 'compact_phase' in r),
                    seconds=time.perf_counter()-start,artifact_dir=str(out) if out is not None else None)
                self._transform_diagnostics = diag
                if out is not None: _dump(out/'status.json',diag)
                return (result,copy.deepcopy(diag)) if return_diagnostics else result
            except BaseException as exc:
                diag = dict(status='failed',error=repr(exc),completed_queries=len(records),
                            queries=records,artifact_dir=str(out) if out is not None else None)
                self._transform_diagnostics = diag
                if out is not None:
                    try:
                        _dump(out/'status.json',diag)
                    except OSError:
                        pass
                if isinstance(exc, (KeyboardInterrupt, SystemExit)): raise
                error_type = ValueError if isinstance(exc,(ValueError,TypeError)) else RuntimeError
                error = error_type('GraphEmbedding transform failed: '+str(exc))
                error.diagnostics = copy.deepcopy(diag)
                raise error from exc

    def save(self, path, *, include_training_data=True, overwrite=False):
        """Save a data-only NPZ predictor, including all original features.

        Version3 persists mapping_domain. Legacy version2 loads as unbounded.
        This is NOT a compact or anonymized model. No external index, graph,
        optimizer, or executable Python object is serialized. Destination is
        published atomically, mode600, without replacement unless opted in.
        """
        s = self._require()
        if include_training_data is not True:
            raise ValueError('include_training_data=False is unsupported: prediction requires X')
        if not isinstance(overwrite, bool):
            raise ValueError('overwrite must be bool')
        warnings.warn('Privacy: saved model retains all training features and coordinates; '
                      'it is not compact or anonymized.', UserWarning, stacklevel=2)
        target = Path(path)
        metadata = dict(schema=_SCHEMA, version=_VERSION, config=self._config.copy(),
                        n_samples=len(s['x']), n_features=s['x'].shape[1],
                        layout_origin=s['diagnostics'].get('layout_origin', 'unknown'))
        payload = json.dumps(metadata, sort_keys=True, allow_nan=False).encode('utf8')
        fd, temporary = tempfile.mkstemp(prefix='.' + target.name + '-', suffix='.npz',
                                         dir=target.parent)
        try:
            with os.fdopen(fd, 'w+b') as stream:
                os.fchmod(stream.fileno(), 0o600)
                np.savez_compressed(stream, metadata=np.frombuffer(payload, dtype=np.uint8),
                    X=s['x'], Z=s['z'], source_ids=s['ids'], source_distances=s['d'],
                    center=s['center'], unit=np.array([s['unit']], dtype=np.float64))
                stream.flush()
                os.fsync(stream.fileno())
            # Validate exactly the bytes being published, using budgets derived
            # from the already-owned predictor rather than default load limits.
            size = Path(temporary).stat().st_size
            raw_size = sum(s[key].nbytes for key in ('x', 'z', 'ids', 'd', 'center'))
            type(self).load(temporary, max_vertices=self._config['max_vertices'],
                max_feature_entries=self._config['max_feature_entries'],
                max_file_bytes=max(1, size), max_uncompressed_bytes=raw_size + 131072)
            if overwrite:
                os.replace(temporary, target)
            else:
                # Atomic exclusive publication; an existing file or symlink is
                # never replaced, including in the presence of competing saves.
                os.link(temporary, target)
            return target
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @classmethod
    def load(cls, path, *, query_index=None, work_dir=None, debug_dir=None,
             max_vertices=60000, max_feature_entries=16000000,
             max_file_bytes=268435456, max_uncompressed_bytes=536870912):
        """Load strict data-only predictor; exact queries unless given a live index.

        Version2 files lacking a domain retain historical unbounded mapping;
        version3 requires the explicit stored domain.
        Header/payload lengths, shapes, dtypes, metadata, finite values and all
        archive budgets are checked before allocating any full model array.
        Bounded streaming chunks are used for the finite-value preflight. ZIP
        member names are never extracted as filesystem paths. Saved graph and
        spectral arrays are intentionally absent; their properties raise.
        """
        _check_index(query_index)
        limits = dict(max_vertices=_integer(max_vertices, 'max_vertices'),
            max_feature_entries=_integer(max_feature_entries, 'max_feature_entries'),
            max_file_bytes=_integer(max_file_bytes, 'max_file_bytes'),
            max_uncompressed_bytes=_integer(max_uncompressed_bytes, 'max_uncompressed_bytes'))
        try:
            # Use one file/ZIP handle for preflight and reads (no pathname reopen
            # race between checking a header and allocating its payload).
            with open(path, 'rb') as stream:
                if os.fstat(stream.fileno()).st_size > limits['max_file_bytes']:
                    raise ValueError('compressed file byte budget exceeded')
                with zipfile.ZipFile(stream) as archive:
                    arrays, metadata, config = _read_predictor(archive, limits)
            model = cls(**config, work_dir=work_dir, debug_dir=debug_dir)
            x, z = arrays['X'], arrays['Z']
            ids, distances = _validate_knn((arrays['source_ids'], arrays['source_distances']),
                                            x, x, nonself=True)
            if (not np.array_equal(ids, arrays['source_ids'])
                    or not np.array_equal(distances, arrays['source_distances'])):
                raise ValueError('noncanonical source neighbors/distances')
            center, unit, anchors = _normalization(z, ids)
            if not np.array_equal(center, arrays['center']) or unit != arrays['unit'][0]:
                raise ValueError('saved center/unit inconsistent with canonical TRAIN normalization')
            diag = dict(status='loaded', schema=_SCHEMA, version=metadata['version'], mapping_domain=config['mapping_domain'],
                layout_origin=metadata.get('layout_origin', 'unknown'), full_anchor_count=len(x), n_features=x.shape[1], reference_bytes=x.nbytes,
                anchor_bytes=anchors.nbytes, distance_unit=unit, artifact_dir=None,
                graph_available=False, spectral_initial_available=False,
                query_policy='exact stable blocked search' if query_index is None else
                    'external candidates; distances checked; recall NOT audited',
                mapper='A exact all-anchor normalized Cauchy KL', mapper_options=OPTIONS.copy(),
                resource_budget_is_rss_cap=False)
            model._state = dict(x=x, z=z, initial=None, graph=None, ids=ids, d=distances,
                center=center, unit=unit, anchors=anchors, index=query_index, diagnostics=diag,
                compact=model._compact_domain(anchors))
            model._fit_diagnostics = copy.deepcopy(diag)
            return model
        except (OSError, ValueError, TypeError, KeyError, EOFError, OverflowError,
                zipfile.BadZipFile, UnicodeError, RecursionError, struct.error, SyntaxError) as exc:
            raise ValueError('Invalid GraphEmbedding predictor: ' + str(exc)) from exc

    @property
    def embedding_(self): return self._require()['z'].copy()
    @property
    def reference_(self): return self._require()['x'].copy()
    @property
    def source_knn_(self):
        s = self._require(); return s['ids'].copy(), s['d'].copy()
    @property
    def spectral_initial_(self):
        value = self._require()['initial']
        if value is None: raise AttributeError('spectral_initial_ is unavailable for external layouts or loaded predictors')
        return value.copy()
    @property
    def graph_(self):
        value = self._require()['graph']
        if value is None: raise AttributeError('graph_ is unavailable for external layouts or loaded predictors')
        return copy.deepcopy(value)
    @property
    def center_(self): return self._require()['center'].copy()
    @property
    def distance_unit_(self): return self._require()['unit']
    @property
    def diagnostics_(self): return copy.deepcopy(self._require()['diagnostics'])
    @property
    def artifact_dir_(self): return self._artifact_dir
    @property
    def transform_diagnostics_(self): return copy.deepcopy(self._transform_diagnostics)
    @property
    def fit_diagnostics_(self): return copy.deepcopy(self._fit_diagnostics)


def _npy_header(stream):
    # Do NOT let a malicious uint32 header length cause an unbounded read.
    if stream.read(6) != b'\x93NUMPY':
        raise ValueError('invalid NPY magic')
    version = stream.read(2)
    if version == b'\x01\x00':
        length_bytes, fmt = 2, '<H'
    elif version == b'\x02\x00':
        length_bytes, fmt = 4, '<I'
    else:
        raise ValueError('unsupported NPY version')
    raw = stream.read(length_bytes)
    if len(raw) != length_bytes:
        raise ValueError('truncated NPY header')
    length = struct.unpack(fmt, raw)[0]
    if not 0 < length <= 4096:
        raise ValueError('NPY header exceeds byte budget')
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError('truncated NPY header')
    header = ast.literal_eval(raw.decode('latin1').strip())
    if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'}:
        raise ValueError('invalid NPY header fields')
    if header['fortran_order'] is not False or not isinstance(header['descr'], str):
        raise ValueError('only C-order primitive dtypes are supported')
    shape = header['shape']
    if (not isinstance(shape, tuple) or len(shape) not in (1, 2)
            or any(type(v) is not int or v <= 0 for v in shape)):
        raise ValueError('invalid array shape')
    dtype = np.dtype(header['descr'])
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError('object/structured arrays are forbidden')
    return shape, dtype, stream.tell()


def _read_predictor(archive, limits):
    infos = archive.infolist()
    expected = {key + '.npy' for key in _ARRAY_KEYS}
    if len(infos) != len(expected) or {info.filename for info in infos} != expected:
        raise ValueError('archive requires exact keys; duplicate/unknown/traversal members forbidden')
    if archive.comment:
        raise ValueError('archive comments are not part of the schema')
    if sum(info.file_size for info in infos) > limits['max_uncompressed_bytes']:
        raise ValueError('uncompressed ZIP byte budget exceeded')
    headers = {}
    for info in infos:
        if (info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or info.file_size < 0 or info.compress_size < 0):
            raise ValueError('unsupported ZIP member')
        key = info.filename[:-4]
        with archive.open(info) as stream:
            shape, dtype, offset = _npy_header(stream)
        wanted = np.dtype('uint8' if key == 'metadata' else 'int64' if key == 'source_ids' else 'float64')
        if dtype != wanted:
            raise ValueError('wrong dtype for ' + key)
        entries = math.prod(shape)
        # Python ints deliberately avoid overflow in forged shape products.
        if entries * dtype.itemsize + offset != info.file_size:
            raise ValueError('NPY claimed shape disagrees with payload size')
        headers[key] = (shape, dtype, offset)
    if headers['metadata'][0][0] > 65536 or len(headers['metadata'][0]) != 1:
        raise ValueError('metadata must be a bounded uint8 vector')
    with archive.open('metadata.npy') as stream:
        stream.read(headers['metadata'][2])
        metadata = _strict_json(stream.read(65537))
    required_metadata = {'schema', 'version', 'config', 'n_samples', 'n_features'}
    if (not isinstance(metadata, dict) or set(metadata) not in
            (required_metadata, required_metadata | {'layout_origin'})):
        raise ValueError('invalid metadata schema keys')
    # Optional diagnostic declaration is backwards-compatible with version2;
    # it does not authenticate provenance or establish any structural property.
    if metadata.get('layout_origin', 'unknown') not in ('external', 'graph_sgd', 'unknown'):
        raise ValueError('invalid layout_origin declaration')
    if metadata['schema'] != _SCHEMA or type(metadata['version']) is not int or metadata['version'] not in (2, 3):
        raise ValueError('unsupported schema/version')
    config = metadata['config']
    expected_keys = _CONFIG_KEYS - {'mapping_domain'} if metadata['version'] == 2 else _CONFIG_KEYS
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise ValueError('invalid saved configuration keys')
    config = dict(config)
    if metadata['version'] == 2:
        # Legacy version2 always meant unbounded. Never silently change the
        # predictions of a historical checkpoint when the runtime default changes.
        config['mapping_domain'] = 'unbounded'
    # Constructor validation is cheap and has no disk side effects.
    canonical_config = GraphEmbedding(**config)._config
    n = _integer(metadata['n_samples'], 'n_samples', 16)
    features = _integer(metadata['n_features'], 'n_features')
    if n > min(limits['max_vertices'], canonical_config['max_vertices']):
        raise ValueError('max_vertices exceeded')
    if n * features > min(limits['max_feature_entries'], canonical_config['max_feature_entries']):
        raise ValueError('max_feature_entries exceeded')
    if 30 * n > canonical_config['max_edges']:
        raise ValueError('saved max_edges conservative preflight exceeded')
    shapes = dict(X=(n, features), Z=(n, 2), source_ids=(n, 15),
                  source_distances=(n, 15), center=(2,), unit=(1,))
    for key, shape in shapes.items():
        if headers[key][0] != shape:
            raise ValueError('invalid shape for ' + key)
    # Scan each payload before any full array allocation. A forged nonfinite
    # last element cannot induce allocation of preceding full model arrays.
    for key, (shape, dtype, offset) in headers.items():
        with archive.open(key + '.npy') as stream:
            stream.read(offset)
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                values = np.frombuffer(chunk, dtype=dtype)
                if dtype.kind == 'f' and not np.isfinite(values).all():
                    raise ValueError('nonfinite payload in ' + key)
                if key == 'source_ids' and (np.any(values < 0) or np.any(values >= n)):
                    raise ValueError('source ID out of range')
                if key == 'source_distances' and np.any(values < 0):
                    raise ValueError('negative source distances')
    arrays = {}
    for key in shapes:
        with archive.open(key + '.npy') as stream:
            arrays[key] = np.lib.format.read_array(stream, allow_pickle=False)
    return arrays, metadata, canonical_config
