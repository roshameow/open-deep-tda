"""Bounded, paired diagnostics, not a guarantee of global topological fidelity.

Verified design invariants are exercised in tests/test_analysis.py: ties exclude
self-neighbors, PH uses identical sample IDs, and resource errors propagate.
Geometry is exact *within* at most 512 uniformly sampled points; PH uses at most
128 of those same points. Neither estimates full-data nearest neighbors or PH.
"""

import time

import numpy as np
from scipy.spatial.distance import cdist

from .datasets import _integer

GEOMETRY_SAMPLE_CAP = 512
TOPOLOGY_SAMPLE_CAP = 128


def _matrix(X, name):
    X = np.asarray(X)
    if X.dtype.kind not in 'biuf':
        raise ValueError(f'{name} must contain real numeric values')
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 1 or not np.all(np.isfinite(X)):
        raise ValueError(f'{name} must be a finite numeric matrix with at least one feature')
    return X


def _paired(reference, embedding):
    reference, embedding = _matrix(reference, 'reference'), _matrix(embedding, 'embedding')
    if len(reference) != len(embedding):
        raise ValueError('reference and embedding must have the same number of rows')
    return reference, embedding


def _distances(X):
    D = cdist(X, X)
    if not np.all(np.isfinite(D)):
        raise ValueError('distance computation overflowed; rescale input coordinates')
    return D


def _neighbors(D):
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    # Stable ties follow the shared, sorted sample IDs; self is always last.
    order = np.argsort(D, axis=1, kind='stable')[:, :-1]
    ranks = np.zeros(D.shape, dtype=np.int64)
    if len(D):
        ranks[np.arange(len(D))[:, None], order] = np.arange(1, len(D))
    return order, ranks


def _geometry(A, B, k):
    n = len(A)
    k_overlap = min(k, max(0, n - 1))
    k_rank = min(k, max(0, (n - 1) // 2))
    oa, ra = _neighbors(A)
    ob, rb = _neighbors(B)
    overlap = (float(np.mean([len(set(a[:k_overlap]) & set(b[:k_overlap])) / k_overlap
                             for a, b in zip(oa, ob)])) if k_overlap else 1.0)
    trust, continuity = 1.0, 1.0
    if k_rank:
        target_ranks = ra[np.arange(n)[:, None], ob[:, :k_rank]]
        source_ranks = rb[np.arange(n)[:, None], oa[:, :k_rank]]
        norm = 2.0 / (n * k_rank * (2 * n - 3 * k_rank - 1))
        trust = float(np.clip(1 - norm * np.maximum(target_ranks - k_rank, 0).sum(), 0, 1))
        continuity = float(np.clip(1 - norm * np.maximum(source_ranks - k_rank, 0).sum(), 0, 1))
    upper = np.triu_indices(n, 1)
    a, b = A[upper], B[upper]
    # Scaling both vectors first avoids overflow in least-squares dot products.
    unit = max(float(a.max()) if len(a) else 0., float(b.max()) if len(b) else 0.)
    unit = unit if unit > 0 else 1.
    a_unit, b_unit = a / unit, b / unit
    denominator = float(b_unit @ b_unit)
    scale = float((a_unit @ b_unit) / denominator) if denominator > 0 else 1.0
    if not np.isfinite(scale):
        raise ValueError('scale alignment overflowed; rescale input coordinates')
    source_energy = float(a_unit @ a_unit)

    def errors(target):
        difference = a_unit - target
        energy = float(difference @ difference)
        rmse = float(np.sqrt(energy / len(a)) * unit) if len(a) else 0.0
        stress = float(np.sqrt(energy / source_energy)) if source_energy > 0 else (0.0 if energy == 0 else None)
        if not np.isfinite(rmse) or (stress is not None and not np.isfinite(stress)):
            raise ValueError('distance error overflowed; rescale input coordinates')
        return {'distance_rmse': rmse, 'normalized_stress': stress}

    return {
        'knn_overlap': overlap, 'trustworthiness': trust, 'continuity': continuity,
        'k_requested': k, 'k_overlap': k_overlap, 'k_rank': k_rank,
        'rank_metric_status': 'defined' if k_rank else 'vacuous: fewer than 3 samples',
        'raw': errors(b_unit), 'scale_aligned': errors(scale * b_unit),
        'target_scale': scale,
        'scale_alignment': 'one global nonnegative least-squares alpha = dot(d_source,d_target)/dot(d_target,d_target), on geometry sample; alpha=1 if target distances vanish',
        'stress_convention': 'sqrt(sum squared distance errors / sum squared source distances); null if source is constant but error is nonzero',
        'source_constant': bool(not np.any(a)), 'target_constant': bool(not np.any(b)),
    }


def evaluate_embedding(reference, embedding, topology_size=64, seed=0, k=10, budgets=None):
    """Return finite JSON-compatible geometry and H0/H1 transport diagnostics.

    Matching costs use squared L2 ground cost with diagonal lifetime²/2 and
    normalization by max(1, source positive bar count). Also reports exact
    L-infinity bottleneck distances, raw Betti curves and long-bar diagnostics.
    ``budgets`` forwards native PH limits and optionally ``max_matching_size``
    (default 512). Budget errors and censored/truncated PH are never swallowed.
    Raw and aligned PH use the same single geometry-derived target multiplier.
    Essential H0 is counted but excluded from finite diagram comparisons.
    """
    started = time.perf_counter()
    reference, embedding = _paired(reference, embedding)
    topology_size = _integer(topology_size, 'topology_size', 1)
    k = _integer(k, 'k', 1)
    rng = np.random.default_rng(seed)
    n = len(reference)
    count = min(n, GEOMETRY_SAMPLE_CAP)
    ids = np.sort(rng.choice(n, count, replace=False)) if count < n else np.arange(n)
    topo_count = min(count, topology_size, TOPOLOGY_SAMPLE_CAP)
    local_ids = np.sort(rng.choice(count, topo_count, replace=False)) if topo_count < count else np.arange(count)
    topo_ids = ids[local_ids]
    A, B = _distances(reference[ids]), _distances(embedding[ids])
    geometry = _geometry(A, B, k)
    geometry_seconds = time.perf_counter() - started
    from . import topology
    limits = dict(budgets or {})
    matching_limit = _integer(limits.pop('max_matching_size', 512), 'max_matching_size')
    ph_start = time.perf_counter()
    source = topology.persistence(A[np.ix_(local_ids, local_ids)], **limits)
    target = topology.persistence(B[np.ix_(local_ids, local_ids)], **limits)
    for result in (source, target):
        if result.get('truncated', False) or any(p.get('censored', False) for p in result['pairs']):
            raise ValueError('evaluation requires complete, uncensored PH; increase max_radius')
    ph_seconds = time.perf_counter() - ph_start
    match_start = time.perf_counter()
    topo_metrics = {}
    for dimension in (0, 1):
        P = topology.diagram(source, dimension=dimension, include_essential=False, positive_only=True)
        Q = topology.diagram(target, dimension=dimension, include_essential=False, positive_only=True)
        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(Q)):
            raise ValueError('finite diagram comparison received nonfinite endpoints')
        raw = topology.diagram_matching(P, Q, max_matching_size=matching_limit)
        aligned = topology.diagram_matching(P, Q * geometry['target_scale'], max_matching_size=matching_limit)
        normalizer = max(1, len(P))
        raw_cost, aligned_cost = float(raw['cost']), float(aligned['cost'])
        if not np.isfinite(raw_cost) or not np.isfinite(aligned_cost):
            raise ValueError('diagram matching cost overflowed; rescale input coordinates')
        source_essential = sum(bool(p.get('essential', False)) for p in source['pairs'] if p['dimension'] == dimension)
        target_essential = sum(bool(p.get('essential', False)) for p in target['pairs'] if p['dimension'] == dimension)
        upper = max(float(P.max()) if P.size else 0., float(Q.max()) if Q.size else 0., 1e-12)
        grid = np.linspace(0., upper * 1.01, 64)
        def betti(diag, essential):
            return ((diag[:, 0, None] <= grid) & (grid < diag[:, 1, None])).sum(axis=0) + essential
        lifetimes = P[:, 1] - P[:, 0]
        threshold = float(np.quantile(lifetimes, .75)) if len(P) else None
        long_ids = set(np.flatnonzero(lifetimes >= threshold).tolist()) if len(P) else set()
        unmatched_long = long_ids.intersection(raw['unmatched_source'].tolist())
        topo_metrics[f'h{dimension}'] = {
            'source_diagram': P.tolist(), 'target_diagram': Q.tolist(),
            'raw_bottleneck': topology.bottleneck_distance(P, Q, max_matching_size=matching_limit),
            'scale_aligned_bottleneck': topology.bottleneck_distance(P, Q * geometry['target_scale'], max_matching_size=matching_limit),
            'betti_curve': {'radius': grid.tolist(), 'source': betti(P, source_essential).tolist(),
                            'target': betti(Q, target_essential).tolist(), 'scale': 'raw'},
            'long_bar_threshold': threshold,
            'long_bar_rule': 'source lifetime >= source 75th percentile, diagnostic not a significance test',
            'raw_long_source_unmatched_fraction': len(unmatched_long) / max(1, len(long_ids)),
            'source_bars': len(P), 'target_bars': len(Q),
            'raw_transport_squared': raw_cost,
            'scale_aligned_transport_squared': aligned_cost,
            'raw_normalized_cost': raw_cost / normalizer,
            'scale_aligned_normalized_cost': aligned_cost / normalizer,
            'raw_source_unmatched_fraction': len(raw['unmatched_source']) / normalizer,
            'raw_target_unmatched_fraction': len(raw['unmatched_target']) / max(1, len(Q)),
            'source_essential_bars': sum(bool(p.get('essential', False)) for p in source['pairs'] if p['dimension'] == dimension),
            'target_essential_bars': sum(bool(p.get('essential', False)) for p in target['pairs'] if p['dimension'] == dimension),
        }
    for name, result in [('source', source), ('target', target)]:
        topo_metrics[name + '_computation'] = {
            key: int(result[key]) for key in ('n_vertices', 'n_simplices', 'reduction_operations', 'peak_reduction_entries') if key in result
        }
    # Use JSON null for the native default unbounded radius, rather than Infinity.
    recorded_limits = dict(max_simplices=1000000, max_reduction_entries=10000000,
                           max_reduction_operations=100000000, max_radius=None)
    recorded_limits.update({key: (None if key == 'max_radius' and np.isposinf(value) else
                                  value.item() if isinstance(value, np.generic) else value)
                            for key, value in limits.items()})
    return {
        'geometry': geometry, 'topology': topo_metrics,
        'sampling': {
            'n_samples': n, 'seed': int(seed) if seed is not None else None,
            'method': 'uniform without replacement; topology nested in geometry sample',
            'geometry_size': count, 'geometry_cap': GEOMETRY_SAMPLE_CAP,
            'geometry_ids': ids.tolist(), 'topology_size_requested': topology_size,
            'topology_size': topo_count, 'topology_cap': TOPOLOGY_SAMPLE_CAP,
            'topology_ids': topo_ids.tolist(), 'same_ids_source_target': True,
            'reference_dimension': reference.shape[1], 'embedding_dimension': embedding.shape[1],
        },
        'budgets': dict(recorded_limits, max_matching_size=matching_limit),
        'timing_seconds': {'geometry': geometry_seconds, 'persistence': ph_seconds,
                           'matching': time.perf_counter() - match_start,
                           'total': time.perf_counter() - started},
        'limitations': 'Exact Euclidean geometry and VR H0/H1 only on the reported subsets; no full-data, H2, semantic, or universal fidelity guarantee. Ties use shared sample-ID order.',
    }


def evaluate_stability(reference, embedding, sample_sizes=(16, 32, 64), repeats=5,
                       seed=0, bootstrap=False, budgets=None):
    """Paired multi-size resampling audit; not a confidence guarantee for true topology.

    bootstrap=True samples IDs with replacement (duplicate rows are retained).
    Otherwise each run is a uniform subset without replacement. Exact sample IDs
    are recorded. At most 128 points per replicate and 100 total replicates.
    """
    reference, embedding = _paired(reference, embedding)
    if not len(reference):
        raise ValueError('stability requires at least one sample')
    sizes = [_integer(s, 'sample_size', 1) for s in sample_sizes]
    repeats = _integer(repeats, 'repeats', 1)
    if not sizes or any(s > TOPOLOGY_SAMPLE_CAP for s in sizes) or len(sizes) * repeats > 100:
        raise ValueError('stability budget: sizes must be <=128 and total replicates <=100')
    rng = np.random.default_rng(seed)
    results = []
    for requested in sizes:
        count = requested if bootstrap else min(requested, len(reference))
        for replicate in range(repeats):
            ids = rng.choice(len(reference), count, replace=bool(bootstrap))
            report = evaluate_embedding(reference[ids], embedding[ids], topology_size=count,
                seed=int(rng.integers(2**31)), budgets=budgets)
            results.append({'size_requested': requested, 'size': count, 'replicate': replicate,
                'original_sample_ids': ids.tolist(), 'distinct_samples': len(np.unique(ids)),
                'trustworthiness': report['geometry']['trustworthiness'],
                'h0_raw_cost': report['topology']['h0']['raw_normalized_cost'],
                'h1_raw_cost': report['topology']['h1']['raw_normalized_cost'],
                'h1_aligned_cost': report['topology']['h1']['scale_aligned_normalized_cost'],
                'h1_bottleneck': report['topology']['h1']['raw_bottleneck']})
    summary = []
    for size in sizes:
        group = [row for row in results if row['size_requested'] == size]
        metrics = {}
        for key in ('trustworthiness', 'h0_raw_cost', 'h1_raw_cost', 'h1_aligned_cost', 'h1_bottleneck'):
            values = [row[key] for row in group]
            metrics[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
        summary.append({'size_requested': size, 'metrics': metrics})
    return {'seed': int(seed), 'bootstrap': bool(bootstrap), 'repeats': repeats,
            'results': results, 'summary': summary,
            'limitations': 'Paired resampling sensitivity only; not proof of full-data or semantic topology.'}
