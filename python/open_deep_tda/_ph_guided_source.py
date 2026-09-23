"""Experimental, bounded source-only structural teacher (not an estimator).

Every row is an original source ID. The four K4 outer-face choices are finite
combinatorial choices, not a trained model or an infeasibility test. Only fresh
full-domain same-ID H0 and selected sparse GF(2) H1 checks accept a layout.
No labels, held-out samples, Ripser, networkx or private data are accessed.
"""
from dataclasses import dataclass, field
from itertools import combinations
from numbers import Integral
import math
import time

import numpy as np
from scipy.optimize import minimize

from .structural_constructive import (LayoutLimits, construct_global_layout,
                                      full_hierarchy, pairwise_planar, _mst, _radius)
from .structural_h0 import compare_h0
from .structural_sparse_h1 import (H1Limits, ResourceLimitError,
                                   analyze_sparse_h1, validate_cycles,
                                   validate_distance_matrix)

__all__ = ['TeacherLimits', 'TeacherResult', 'construct_source_teacher']


@dataclass(frozen=True)
class TeacherLimits:
    max_vertices: int = 300
    max_pairs: int = 44_850
    max_iterations: int = 400
    max_distance_workspace_bytes: int = 268_435_456
    max_seconds: float = 60.


@dataclass(frozen=True)
class TeacherResult:
    certified: bool
    embedding: object = None
    reason: str = ''
    diagnostics: dict = field(default_factory=dict)
    source: object = None
    target: object = None
    h0_error: object = None


class _TimedOut(RuntimeError):
    pass


def _check_limits(limits):
    if not isinstance(limits, TeacherLimits):
        raise ValueError('limits must be TeacherLimits')
    for name, maximum in [('max_vertices', 300), ('max_pairs', 44_850),
                          ('max_iterations', 400),
                          ('max_distance_workspace_bytes', 268_435_456)]:
        value = getattr(limits, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value <= maximum:
            raise ValueError('%s must be an integer in [0, %d]' % (name, maximum))
    seconds = limits.max_seconds
    if isinstance(seconds, (bool, np.bool_)) or not isinstance(seconds, (int, float, np.floating)) or not math.isfinite(seconds) or not 0 <= seconds <= 3600:
        raise ValueError('max_seconds must be finite in [0, 3600]')


def _source(D, X, limits):
    if not isinstance(D, np.ndarray) or D.dtype != np.float64 or D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError('D must be a square float64 ndarray')
    n = len(D)
    if n > limits.max_vertices or n*(n-1)//2 > limits.max_pairs:
        raise ResourceLimitError('source vertex/pair limit exceeded before scanning')
    if not isinstance(X, np.ndarray) or X.dtype != np.float64 or X.ndim != 2 or X.shape[0] != n or not 2 <= X.shape[1] <= 4096:
        raise ValueError('X must be (N,2<=F<=4096) float64 original source features')
    # Subtraction/norm can retain multiple dense broadcast temporaries; this
    # bounds an estimated three float64 feature cubes, not whole-process RSS.
    if 3*n*n*X.shape[1]*8 > limits.max_distance_workspace_bytes:
        raise ResourceLimitError('source feature-distance workspace budget exceeded')
    D = validate_distance_matrix(D, limits=H1Limits(max_vertices=limits.max_vertices))
    if not np.isfinite(X).all():
        raise ValueError('X must be finite')
    # The source metric is Euclidean in X, in unchanged source units. Reject
    # inconsistent distances rather than quietly mixing unrelated geometries.
    try:
        with np.errstate(over='raise', invalid='raise'):
            metric = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    except FloatingPointError as exc:
        raise ValueError('source feature distances unrepresentable') from exc
    scale = max(float(metric.max(initial=0.)), np.finfo(float).tiny)
    if not np.isfinite(metric).all() or not np.allclose(D, metric, rtol=1e-12, atol=1e-12*scale):
        raise ValueError('D must equal Euclidean distances of X in source units (roundoff only)')
    return D.copy(), X.copy()


def _paths(chains):
    adjacency = {}
    edges = set()
    for chain in chains:
        for u, v in chain:
            edges.add((u, v))
    for u, v in sorted(edges):
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    branch = sorted(u for u, neighbors in adjacency.items() if len(neighbors) == 3)
    if len(branch) != 4 or any(len(v) not in (2, 3) for v in adjacency.values()):
        return None
    paths, used = {}, set()
    for u in branch:
        for neighbor in sorted(adjacency[u]):
            edge = tuple(sorted((u, neighbor)))
            if edge in used:
                continue
            chain = [u, neighbor]
            used.add(edge)
            previous, current = u, neighbor
            while current not in branch:
                nxt = next(v for v in adjacency[current] if v != previous)
                edge = tuple(sorted((current, nxt)))
                if edge in used or len(chain) > len(adjacency):
                    return None
                used.add(edge)
                chain.append(nxt)
                previous, current = current, nxt
            key = tuple(sorted((u, current)))
            if u == current or key in paths:
                return None
            paths[key] = chain if u < current else list(reversed(chain))
    if used != edges or set(paths) != set(combinations(branch, 2)):
        return None
    return adjacency, branch, paths


def _scaffold(X, graph, branch, paths, center, side):
    outer = sorted(set(branch) - {center})
    height = side * np.sqrt(3.) / 2.
    P = {outer[0]: np.array([0., 0.]), outer[1]: np.array([side, 0.]),
         outer[2]: np.array([side/2., height]), center: np.array([side/2., height/3.])}
    Y = np.empty((len(X), 2), dtype=np.float64)
    axes = {}
    for (u, v), chain in sorted(paths.items()):
        steps = np.linalg.norm(np.diff(X[chain], axis=0), axis=1)
        delta = X[v]-X[u]
        norm = float(delta @ delta)
        if np.any(steps <= 0) or norm <= 0:
            raise ValueError('degenerate source K4 path')
        # Fixed SOURCE-witness graph-geodesic spacing, not proportional to
        # noisy high-dimensional chords: one long source edge must not push a
        # required target birth edge beyond a after planarization.
        fractions = np.linspace(0., 1., len(chain), dtype=np.float64)
        for vertex, t in zip(chain, fractions):
            Y[vertex] = (1-t)*P[u]+t*P[v]
        residual = X[chain]-(X[u]+(((X[chain]-X[u]) @ delta)/norm)[:, None]*delta)
        _, _, vt = np.linalg.svd(residual, full_matrices=False)
        axis = vt[0].copy()
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
        axes[u, v] = axis
    for vertex in sorted(set(range(len(X))) - set(graph)):
        choices = []
        for u, v in sorted(paths):
            delta = X[v]-X[u]
            t = float(np.clip(((X[vertex]-X[u]) @ delta)/(delta @ delta), 0., 1.))
            foot = X[u]+t*delta
            choices.append((float(np.linalg.norm(X[vertex]-foot)), (u, v), t, foot))
        _, (u, v), t, foot = min(choices, key=lambda item: (item[0], item[1]))
        tangent = P[v]-P[u]
        normal = np.array([-tangent[1], tangent[0]])/np.linalg.norm(tangent)
        Y[vertex] = (1-t)*P[u]+t*P[v]+normal*((X[vertex]-foot) @ axes[u, v])
    if not np.isfinite(Y).all():
        raise FloatingPointError('nonfinite scaffold')
    return Y


def _repair(Y0, lower, upper, tree, pairs, iterations, deadline):
    ii, jj = pairs
    tu, tv = tree
    n = len(Y0)
    def objective(flat):
        if time.monotonic() >= deadline:
            raise _TimedOut('source-only repair timeout')
        Y = flat.reshape(n, 2)
        diff = Y[ii]-Y[jj]
        length = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        gap = np.minimum(length-lower, 0.)
        grad = np.zeros_like(Y)
        np.add.at(grad, ii, (2*gap/length)[:, None]*diff)
        np.add.at(grad, jj, -(2*gap/length)[:, None]*diff)
        diff = Y[tu]-Y[tv]
        length = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        excess = np.maximum(length-upper, 0.)
        np.add.at(grad, tu, (2*excess/length)[:, None]*diff)
        np.add.at(grad, tv, -(2*excess/length)[:, None]*diff)
        drift = Y-Y0
        grad += .0002*drift
        return float(gap @ gap + excess @ excess + .0001*np.sum(drift*drift)), grad.ravel()
    fit = minimize(objective, Y0.ravel().copy(), jac=True, method='L-BFGS-B',
                   options={'maxiter': iterations, 'maxls': 20, 'ftol': 1e-14, 'gtol': 1e-10})
    return fit.x.reshape(n, 2).copy(), dict(iterations=int(fit.nit), optimizer_success=bool(fit.success),
                                            objective=float(fit.fun), message=str(fit.message))


def construct_source_teacher(D, X, cycles, a, b, h0_tol, *, strategy='single',
                             limits=TeacherLimits(), h1_limits=H1Limits()):
    """Try a source-only teacher; return no embedding unless independently certified.

    D is owned float64 Euclidean distances of the original float64 X. `single`
    feeds an untrained centered PCA2 source guide to construct_global_layout.
    `subdivided_k4` uses four sorted branch-ID outer-face choices, side length
    2.5 times the median positive source pair distance (NOT historical raw 2.5),
    uniform graph-geodesic spacing along each source witness path, and a
    source-only H0 contact objective. No target
    H1 enters optimization. Failure is unresolved, not proof of infeasibility.
    Caps bound logical work; timeout is checked at objective calls, not a hard
    bound on an individual BLAS/SciPy call. No external Ripser is invoked.
    """
    _check_limits(limits)
    started=time.monotonic()
    deadline=started+float(limits.max_seconds)
    def check_deadline():
        if time.monotonic() >= deadline:
            raise ResourceLimitError('source teacher max_seconds exceeded')
    check_deadline()
    if strategy not in ('single', 'subdivided_k4'):
        raise ValueError('unknown strategy')
    if not isinstance(h1_limits, H1Limits):
        raise ValueError('h1_limits must be H1Limits')
    D, X = _source(D, X, limits)
    check_deadline()
    a, b, tolerance = _radius(a, 'a'), _radius(b, 'b'), _radius(h0_tol, 'h0_tol')
    if a > b:
        raise ValueError('a must not exceed b')
    chains = validate_cycles(cycles, len(D), limits=h1_limits)
    source = analyze_sparse_h1(D, chains, a, b, limits=h1_limits)
    check_deadline()
    info = dict(strategy=strategy, n_vertices=len(D), scope='all original source rows; selected H1 family only',
                metric='Euclidean X; original source units', candidates=[])
    def fail(reason):
        return TeacherResult(False, None, reason, info, source)
    if not chains or not source.certified:
        return fail('source selected H1 family not certified')
    if strategy == 'single':
        try:
            centered = X-X.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            guide = np.asarray(centered @ vt[:2].T, dtype=np.float64)
            check_deadline()
            result = construct_global_layout(D, guide, chains, a, b, tolerance,
                                             limits=LayoutLimits(max_vertices=limits.max_vertices,
                                                                 max_matrix_entries=limits.max_vertices**2),
                                             h1_limits=h1_limits)
            check_deadline()
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            return fail('single numerical failure: '+str(exc))
        info['candidates'].append(dict(certified=result.certified, reason=result.reason,
                                       diagnostics=result.diagnostics))
        if not result.certified:
            return fail(result.reason)
        embedding = result.embedding.copy()
        embedding.setflags(write=False)
        return TeacherResult(True, embedding, result.reason, info, source, result.target, result.h0_error)
    shape = _paths(chains)
    if shape is None:
        return fail('unsupported witness union: require a subdivided K4 with six disjoint paths')
    graph, branch, paths = shape
    positive = D[np.triu_indices(len(D), 1)]
    positive = positive[positive > 0]
    if not len(positive):
        return fail('no positive source pair distance')
    side = 2.5 * float(np.median(positive))
    if not math.isfinite(side) or side <= 0:
        return fail('unrepresentable source-derived side')
    info.update(branch_ids=branch, side=side, side_rule='2.5 * median positive source pair distance',
                witness_path_spacing='uniform per original-ID graph edge',
                epsilon=.9*tolerance, prior=.0001, max_iterations=limits.max_iterations)
    hierarchy = full_hierarchy(D)
    pairs = np.triu_indices(len(D), 1)
    lower = np.maximum(0., hierarchy[pairs]-.9*tolerance)
    edges = _mst(D)
    tree = np.array(edges, dtype=np.int64).reshape(-1, 2).T
    upper = np.array([D[u, v]+.9*tolerance for u, v in edges])
    check_deadline()
    winner = None
    info['candidates'] = [dict(interior_branch=center, status='not_attempted') for center in branch]
    for record in info['candidates']:
        center = record['interior_branch']
        try:
            if time.monotonic() >= deadline:
                raise _TimedOut('constructor timeout')
            initial = _scaffold(X, graph, branch, paths, center, side)
            record['status'] = 'attempted'
            # Always audit each initial and final candidate, but only a fresh
            # certificate can authorize a returned embedding.
            for phase, Y in [('initial', initial), ('repaired', None)]:
                if phase == 'repaired':
                    if not limits.max_iterations:
                        record['repair'] = 'iteration cap zero'
                        break
                    Y, optimization = _repair(initial, lower, upper, tree, pairs,
                                              limits.max_iterations, deadline)
                    record['optimization'] = optimization
                if not np.isfinite(Y).all():
                    raise FloatingPointError('nonfinite candidate')
                target_D = pairwise_planar(Y)
                h0 = compare_h0(D, target_D, tolerance=tolerance, max_vertices=limits.max_vertices)
                error = h0['max_merge_error']
                h1 = analyze_sparse_h1(target_D, chains, a, b, limits=h1_limits)
                accepted = bool(h0['certified_within_tolerance'] and h1.certified)
                record[phase] = dict(h0_error=error, h0_certified=h0['certified_within_tolerance'],
                                     selected_rank=h1.rank, certified=accepted, target_reason=h1.reason)
                if accepted and winner is None:
                    result = Y.copy()
                    result.setflags(write=False)
                    winner = (result, h1, error, center, phase)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError, ValueError, _TimedOut) as exc:
            record['failure'] = str(exc)
            record['status'] = 'failed'
            if isinstance(exc, _TimedOut):
                info['timeout'] = True
                break
    if winner is not None:
        result, h1, error, center, phase = winner
        info['selected_branch'] = center
        info['selected_phase'] = phase
        return TeacherResult(True, result, 'certified full-domain H0 and selected source/target H1',
                             info, source, h1, error)
    return fail('no independently certified K4 candidate; unresolved, not infeasible')
