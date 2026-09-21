"""Query-sampled geometry against the ENTIRE supplied population.

Unlike evaluation.evaluate_embedding's bounded subcloud diagnostics, neighbors
and ranks here include all N rows. Queries alone are subsampled, so reported
trustworthiness/continuity are estimates of the full-population average when
queries are uniform, or diagnostics for the declared query group (e.g. test).
Memory is O(block_size*N), not an N-by-N distance matrix.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .evaluation import _paired
from .datasets import _integer


def population_geometry(reference, embedding, query_indices=None, max_queries=256,
                        k=15, seed=0, block_size=16):
    X, Z = _paired(reference, embedding)
    n = len(X)
    if not n:
        raise ValueError('population must be nonempty')
    max_queries = _integer(max_queries, 'max_queries', 1)
    k = _integer(k, 'k', 1)
    block_size = _integer(block_size, 'block_size', 1)
    rng = np.random.default_rng(seed)
    if query_indices is None:
        ids = np.sort(rng.choice(n, min(n, max_queries), replace=False))
    else:
        raw = np.asarray(query_indices)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu' or not len(raw):
            raise ValueError('query_indices must be a nonempty integer vector')
        ids = raw.astype(np.int64)
        if np.any(ids < 0) or np.any(ids >= n) or len(np.unique(ids)) != len(ids):
            raise ValueError('query_indices must be distinct valid population IDs')
        if len(ids) > max_queries:
            raise ValueError('query count exceeds max_queries; no silent query removal')
    overlap_k = min(k, n - 1)
    rank_k = min(k, (n - 1) // 2)
    missing_source, missing_target, overlaps = 0., 0., []
    # Distances are float64; rows converted to rank vectors one at a time.
    for start in range(0, len(ids), block_size):
        query = ids[start:start + block_size]
        DX, DZ = cdist(X[query], X), cdist(Z[query], Z)
        if not np.isfinite(DX).all() or not np.isfinite(DZ).all():
            raise ValueError('population distances overflowed; rescale inputs')
        DX[np.arange(len(query)), query] = np.inf
        DZ[np.arange(len(query)), query] = np.inf
        for a, b in zip(DX, DZ):
            order_x = np.argsort(a, kind='stable')
            order_z = np.argsort(b, kind='stable')
            if overlap_k:
                overlaps.append(len(np.intersect1d(order_x[:overlap_k], order_z[:overlap_k])) / overlap_k)
            if rank_k:
                ranks = np.empty(n, dtype=np.int64)
                ranks[order_x] = np.arange(1, n + 1)
                missing_source += np.maximum(ranks[order_z[:rank_k]] - rank_k, 0).sum()
                ranks[order_z] = np.arange(1, n + 1)
                missing_target += np.maximum(ranks[order_x[:rank_k]] - rank_k, 0).sum()
    denominator = len(ids) * rank_k * (2*n - 3*rank_k - 1)
    return {
        'trustworthiness': float(1 - 2*missing_source/denominator) if rank_k else None,
        'continuity': float(1 - 2*missing_target/denominator) if rank_k else None,
        'knn_overlap': float(np.mean(overlaps)) if overlaps else None,
        'population_size': n, 'query_count': len(ids), 'query_indices': ids.tolist(),
        'requested_k': k, 'overlap_k': overlap_k, 'rank_k': rank_k,
        'scope': 'query rows versus all supplied population rows; self excluded; stable sample-ID ties',
        'rank_metric_status': 'defined' if rank_k else 'undefined for population <3',
        'memory_scope': 'two block_size by N float64 distance blocks plus per-row rank/order arrays',
        'block_size': block_size,
        'limitations': 'Only query rows are sampled. Semantic relevance and global topology are not implied.'}
