"""Mapper: connected cover preimages in a fixed, symmetric reference kNN graph.

This is a graph-based Mapper summary, NOT an exact Reeb graph. Its cycles are
not automatically the H1 classes of the underlying point cloud.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .datasets import _integer
from .evaluation import _matrix


def mapper_graph(X, lens=None, n_intervals=8, overlap=0.3, n_neighbors=8):
    """Build an interval-cover nerve's 1-skeleton with explicit sample members.

    Default lens: first reference coordinate. ``lens`` can be a finite vector
    or a callable returning one. Closed intervals have overlap/width equal to
    ``overlap``; even zero-overlap closed intervals can share boundary points.
    Constant lenses use one interval. kNN is exact Euclidean cKDTree (ties use
    the backend's deterministic ordering, not a canonical ordering across
    SciPy versions); the graph is fixed BEFORE cover restrictions. No dense
    N-by-N distance matrix is constructed. Storage is O(N * n_neighbors).
    """
    X = _matrix(X, 'X')
    n_intervals = _integer(n_intervals, 'n_intervals', 1)
    n_neighbors = _integer(n_neighbors, 'n_neighbors', 1)
    if not np.isscalar(overlap) or not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('overlap must lie in [0, 1)')
    n = len(X)
    values = X[:, 0] if lens is None else np.asarray(lens(X) if callable(lens) else lens, dtype=float)
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError('lens must have one finite scalar per sample')
    effective_k = min(n_neighbors, max(0, n - 1))
    rows, columns = [], []
    if effective_k:
        tree = cKDTree(X)
        for start in range(0, n, 1024):
            distances, neighbors = tree.query(X[start:start + 1024], k=effective_k + 1)
            if not np.all(np.isfinite(distances)):
                raise ValueError('nearest-neighbor distances overflowed; rescale X')
            for offset, candidates in enumerate(neighbors):
                i = start + offset
                selected = candidates[candidates != i][:effective_k]
                rows.extend([i] * len(selected))
                columns.extend(selected.tolist())
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)), shape=(n, n))
    graph = graph.maximum(graph.T)
    intervals = []
    if n:
        low, high = float(values.min()), float(values.max())
        if low == high:
            intervals = [[low, high]]
        else:
            width = (high - low) / (n_intervals - (n_intervals - 1) * overlap)
            if not np.isfinite(width):
                raise ValueError('lens cover overflowed; rescale lens')
            step = width * (1 - overlap)
            intervals = [[float(low + i * step), float(low + i * step + width)] for i in range(n_intervals)]
            intervals[0][0], intervals[-1][1] = low, high
    nodes, memberships = [], [[] for _ in range(n)]
    for interval_id, (low, high) in enumerate(intervals):
        members = np.flatnonzero((values >= low) & (values <= high))
        if not len(members):
            continue
        count, components = connected_components(graph[members][:, members], directed=False)
        for component in range(count):
            ids = members[components == component]
            node_id = len(nodes)
            unit = max(1., float(np.max(np.abs(values[ids]))))
            mean = float(np.mean(values[ids] / unit) * unit)
            nodes.append({'id': node_id, 'interval': interval_id, 'members': ids.tolist(),
                          'size': len(ids), 'lens_mean': mean})
            for point in ids:
                memberships[point].append(node_id)
    intersections = {}
    for point, node_ids in enumerate(memberships):
        for i, left in enumerate(node_ids):
            for right in node_ids[i + 1:]:
                intersections.setdefault((left, right), []).append(point)
    edges = [{'source': left, 'target': right, 'members': members, 'size': len(members)}
             for (left, right), members in sorted(intersections.items())]
    return {
        'kind': 'Mapper graph (not an exact Reeb graph)', 'nodes': nodes, 'edges': edges,
        'cover': {'intervals': intervals, 'n_intervals_requested': n_intervals,
                  'n_intervals_effective': len(intervals), 'overlap': float(overlap),
                  'boundary_convention': 'closed intervals',
                  'lens': 'first reference coordinate' if lens is None else 'user-supplied scalar lens'},
        'graph': {'n_samples': n, 'n_neighbors_requested': n_neighbors,
                  'n_neighbors_effective': effective_k, 'n_edges': int(graph.nnz // 2),
                  'metric': 'euclidean', 'method': 'exact cKDTree kNN; union symmetrization; fixed before cover'},
        'limitations': 'Cover resolution, lens and reference graph affect the summary. Mapper cycles are not a claim of point-cloud H1 or an exact Reeb graph.',
    }
