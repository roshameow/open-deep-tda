"""Standalone, bounded same-ID H0 verification on complete supplied domains.

No sampling, ID matching, distance normalization, or estimator integration is
performed. Row i must denote the same vertex in both inputs.
"""

import numbers

import numpy as np

from . import topology


_SCOPE = "complete supplied vertex domain; no assertion about omitted points"


def _preflight_size(value, name, budget):
    """Inspect shape/sequence lengths without materializing a square array."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("{} must be a square matrix".format(name))
        n = int(shape[0])
    else:
        try:
            n = len(value)
        except TypeError as exc:
            raise ValueError("{} must be a square matrix".format(name)) from exc
    if n > budget:
        raise RuntimeError(
            "{} has {} vertices, exceeding max_vertices={}".format(name, n, budget)
        )
    if shape is None:
        try:
            square = all(len(row) == n for row in value)
        except TypeError as exc:
            raise ValueError("{} must be a square matrix".format(name)) from exc
        if not square:
            raise ValueError("{} must be a square matrix".format(name))
    return n


def _tree(D):
    n = len(D)
    edges = topology.mst(D).tolist() if n > 1 else []
    weights = [float(D[i, j]) for i, j in edges]
    adjacency = [[] for _ in range(n)]
    for (i, j), weight in zip(edges, weights):
        adjacency[i].append((j, weight))
        adjacency[j].append((i, weight))
    return {"edges": edges, "weights": weights}, adjacency


def _path_maxima(adjacency, root):
    maxima = [0.0] * len(adjacency)
    stack = [(root, -1)]
    while stack:
        vertex, parent = stack.pop()
        for neighbor, weight in adjacency[vertex]:
            if neighbor != parent:
                maxima[neighbor] = max(maxima[vertex], weight)
                stack.append((neighbor, vertex))
    return maxima


def compare_h0(D_source, D_target, *, tolerance=0.0, max_vertices=2048):
    """Return a JSON-compatible certificate for same-ID minimax merge distances.

    Inputs are equally sized, finite, nonnegative, symmetric distance matrices
    with zero diagonal, validated/converted to float64 as in ``topology.mst``.
    Triangle inequalities are not required. All supplied vertices participate;
    the caller is responsible for identical row IDs/order in both matrices.
    ``tolerance`` is a finite nonnegative real (not bool), in distance units.
    ``max_vertices`` is a nonnegative integer (not bool). Both domain sizes are
    checked before array conversion, distance validation, or MST computation;
    exceeding the budget raises RuntimeError, never a partial certificate.

    For each pair, its minimax merge distance is the maximum edge weight on
    the path in a global MST. ``max_merge_error`` is the exact maximum absolute
    difference of these distances, computed in O(n^2) tree traversal work with
    O(n) auxiliary storage beyond the input matrices. The definitive predicate
    ``certified_within_tolerance`` tests this error, without numerical slack.
    ``max_merge_error_witness`` records the lexicographically first maximizing
    pair and both merge distances; it is None for n < 2 (error zero).

    ``max_tree_edge_error`` instead compares ORIGINAL distances on the union
    of the two MST edge sets. It bounds the merge error: using a source MST
    path in the target gives one inequality, and the target MST gives the
    reverse inequality. ``tree_edge_bound_within_tolerance`` is consequently
    sufficient, NOT necessary. In particular, zero symmetric H0 tree-edge
    stress is sufficient but not necessary for identical merge hierarchies.
    A failed bound alone must not reject a hierarchy. Sorted H0 bar lengths
    alone cannot establish same-ID equality either.

    ``source_tree`` and ``target_tree`` contain edge ID pairs and corresponding
    weights in their own matrix. Native/backend errors propagate explicitly;
    no fallback or partial success is returned. This certifies only H0 on the
    complete supplied domain, not omitted vertices or higher homology.
    """
    budget = topology._integer_budget(max_vertices, "max_vertices")
    if (isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, numbers.Real) or tolerance < 0):
        raise ValueError("tolerance must be a finite nonnegative real")
    try:
        tolerance = float(tolerance)
    except OverflowError as exc:
        raise ValueError("tolerance must be a finite nonnegative real") from exc
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative real")

    n = _preflight_size(D_source, "D_source", budget)
    target_n = _preflight_size(D_target, "D_target", budget)
    if n != target_n:
        raise ValueError("D_source and D_target must have equal shape")
    source = topology._distance_array(D_source)
    target = topology._distance_array(D_target)
    source_tree, source_adjacency = _tree(source)
    target_tree, target_adjacency = _tree(target)

    union = {tuple(edge) for edge in source_tree["edges"] + target_tree["edges"]}
    edge_error = max(
        (abs(float(source[i, j]) - float(target[i, j])) for i, j in union),
        default=0.0,
    )
    merge_error = 0.0
    witness = None
    for i in range(n - 1):
        source_merges = _path_maxima(source_adjacency, i)
        target_merges = _path_maxima(target_adjacency, i)
        for j in range(i + 1, n):
            error = abs(source_merges[j] - target_merges[j])
            if witness is None or error > merge_error:
                merge_error = error
                witness = {
                    "pair": [i, j],
                    "source_merge_distance": source_merges[j],
                    "target_merge_distance": target_merges[j],
                    "absolute_error": error,
                }

    return {
        "n_vertices": n,
        "scope": _SCOPE,
        "criterion": "max_merge_error <= tolerance (same-ID minimax merge distances)",
        "tolerance": tolerance,
        "max_tree_edge_error": edge_error,
        "tree_edge_bound_within_tolerance": edge_error <= tolerance,
        "max_merge_error": merge_error,
        "max_merge_error_witness": witness,
        "certified_within_tolerance": merge_error <= tolerance,
        "source_tree": source_tree,
        "target_tree": target_tree,
    }
