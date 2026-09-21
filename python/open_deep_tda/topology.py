"""Bounded exact VR topology and squared-L2 augmented diagram matching.

The native backend is loaded only by :func:`mst`, :func:`persistence`, and
:func:`h1_persistence`.
No approximate or substitute persistence implementation is used on failure.
"""

import importlib
import numbers

import numpy as np


_BUDGET_DEFAULTS = {
    "max_simplices": 1000000,
    "max_reduction_entries": 10000000,
    "max_reduction_operations": 100000000,
    "max_radius": float("inf"),
}


def _native():
    try:
        return importlib.import_module("open_deep_tda._core")
    except ImportError as exc:
        raise ImportError(
            "Exact topology requires the compiled open_deep_tda._core backend. "
            "Build/install the project's C++ extension before calling mst, persistence, or h1_persistence."
        ) from exc


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError("{} must contain real numeric values".format(name))
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values".format(name))
    return array


def _distance_array(D):
    D = _real_array(D, "distances")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distances must be a square matrix")
    if np.any(D < 0):
        raise ValueError("distances must be nonnegative")
    if not np.array_equal(D, D.T):
        raise ValueError("distances must be symmetric")
    if np.any(np.diag(D) != 0):
        raise ValueError("distances must have a zero diagonal")
    return np.ascontiguousarray(D)


def _integer_budget(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError("{} must be a finite nonnegative integer".format(name))
    return int(value)


def _budgets(budgets, full_filtration=False):
    unknown = set(budgets) - set(_BUDGET_DEFAULTS)
    if unknown:
        raise ValueError("Unknown persistence budget(s): {}".format(", ".join(sorted(unknown))))
    result = dict(_BUDGET_DEFAULTS, **budgets)
    for name in _BUDGET_DEFAULTS:
        if name != "max_radius":
            result[name] = _integer_budget(result[name], name)
    radius = result["max_radius"]
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, numbers.Real):
        raise ValueError("max_radius must be nonnegative or +inf")
    radius = float(radius)
    if np.isnan(radius) or radius < 0:
        raise ValueError("max_radius must be nonnegative or +inf")
    if full_filtration and radius != float("inf"):
        raise ValueError("Training losses require full filtration (max_radius=inf), not censored bars")
    result["max_radius"] = radius
    return result


def distance_matrix(X):
    """Return finite float64 Euclidean distances for a real (n, d) matrix."""
    X = _real_array(X, "X")
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional point matrix")
    # pdist avoids an n*n*d temporary and cancellation from Gram identities.
    from scipy.spatial.distance import pdist, squareform
    if X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    D = squareform(pdist(X, metric="euclidean"))
    if not np.isfinite(D).all():
        raise ValueError("Euclidean distances overflowed; rescale X")
    return np.ascontiguousarray(D, dtype=np.float64)


def persistence(D, **budgets):
    """Compute exact native VR H0/H1; budget exhaustion is never partial success."""
    D = _distance_array(D)
    options = _budgets(budgets)
    return _native().persistence(D, **options)


def h1_persistence(D, **budgets):
    """Compute exact VR, serializing only positive H1 and censored/essential H1.

    Same budgets, result/pair schema, order, critical IDs, and full-computation
    statistics as :func:`persistence`. H0 and finite zero-length H1 are omitted
    before creating Python objects; native simplex/reduction work is unchanged.
    Truncated results remain marked and are not suitable for training losses.
    """
    D = _distance_array(D)
    options = _budgets(budgets)
    return _native().persistence_h1(D, **options)


def mst(D):
    """Return deterministic native Prim edges, shape (max(n-1, 0), 2)."""
    D = _distance_array(D)
    return np.asarray(_native().mst(D), dtype=np.int64).reshape(-1, 2)


def diagram(result, dimension=1, include_essential=False, positive_only=True):
    """Extract bars; censored bars are rejected rather than presented as finite.

    Essential bars are excluded by default. Explicitly including them produces
    infinity, which :func:`diagram_matching` deliberately does not accept.
    """
    if isinstance(dimension, bool) or dimension not in (0, 1):
        raise ValueError("dimension must be 0 or 1")
    bars = []
    for pair in result["pairs"]:
        if pair["dimension"] != dimension:
            continue
        if pair.get("censored", False):
            raise ValueError("Censored bars cannot form an ordinary persistence diagram")
        if pair["essential"] and not include_essential:
            continue
        birth, death = float(pair["birth"]), float(pair["death"])
        if not np.isfinite(birth) or np.isnan(death) or death < birth:
            raise ValueError("Invalid persistence endpoints")
        if not np.isfinite(death) and not pair["essential"]:
            raise ValueError("Nonessential bars must be finite")
        if not positive_only or death > birth:
            bars.append((birth, death))
    return np.asarray(bars, dtype=np.float64).reshape(-1, 2)


def _diagram_array(P, name):
    P = _real_array(P, name)
    if P.ndim == 1 and P.size == 0:
        P = P.reshape(0, 2)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("{} must have shape (n, 2)".format(name))
    if np.any(P[:, 1] < P[:, 0]):
        raise ValueError("{} has death before birth".format(name))
    return P


def diagram_matching(P, Q, max_matching_size=512):
    """Exact partial matching with squared L2 and lifetime**2 / 2 deletion.

    The size budget bounds the augmented assignment dimension len(P)+len(Q),
    before allocating its quadratic cost matrix. Costs are not normalized.
    Inputs are finite diagrams, never right-censored or essential bars.
    """
    P, Q = _diagram_array(P, "P"), _diagram_array(Q, "Q")
    budget = _integer_budget(max_matching_size, "max_matching_size")
    m, n = len(P), len(Q)
    if m + n > budget:
        raise RuntimeError("Diagram matching size {} exceeds max_matching_size={}".format(m + n, budget))
    if m + n == 0:
        return {"matched": np.empty((0, 2), dtype=np.int64),
                "unmatched_source": np.empty(0, dtype=np.int64),
                "unmatched_target": np.empty(0, dtype=np.int64), "cost": 0.0}
    with np.errstate(over="ignore", invalid="ignore"):
        cross = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(axis=2)
        source_diag = (P[:, 1] - P[:, 0]) ** 2 / 2
        target_diag = (Q[:, 1] - Q[:, 0]) ** 2 / 2
    if not all(np.isfinite(x).all() for x in (cross, source_diag, target_diag)):
        raise ValueError("Diagram costs overflowed; rescale diagram endpoints")
    costs = np.full((m + n, m + n), np.inf)
    costs[:m, :n] = cross
    costs[np.arange(m), n + np.arange(m)] = source_diag
    costs[m + np.arange(n), np.arange(n)] = target_diag
    costs[m:, n:] = 0.0
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(costs)
    real = (rows < m) & (cols < n)
    matched = np.column_stack((rows[real], cols[real])).astype(np.int64)
    total = float(costs[rows, cols].sum())
    if not np.isfinite(total):
        raise ValueError("Total diagram cost overflowed; rescale diagram endpoints")
    return {"matched": matched.reshape(-1, 2),
            "unmatched_source": rows[(rows < m) & (cols >= n)].astype(np.int64),
            "unmatched_target": cols[(rows >= m) & (cols < n)].astype(np.int64),
            "cost": total}


def bottleneck_distance(P, Q, max_matching_size=512):
    """Exact finite-diagram bottleneck distance with L-infinity ground metric.

    Binary-search the finite candidate edge costs in the augmented bipartite
    assignment. This is distinct from diagram_matching's squared-L2 transport.
    """
    P, Q = _diagram_array(P, "P"), _diagram_array(Q, "Q")
    budget = _integer_budget(max_matching_size, "max_matching_size")
    m, n = len(P), len(Q)
    if m + n > budget:
        raise RuntimeError("Bottleneck matching size exceeds max_matching_size")
    if m + n == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        cross = np.abs(P[:, None, :] - Q[None, :, :]).max(axis=2)
        a = (P[:, 1] - P[:, 0]) / 2
        b = (Q[:, 1] - Q[:, 0]) / 2
    if not all(np.isfinite(x).all() for x in (cross, a, b)):
        raise ValueError("Bottleneck costs overflowed; rescale endpoints")
    costs = np.full((m + n, m + n), np.inf)
    costs[:m, :n] = cross
    costs[np.arange(m), n + np.arange(m)] = a
    costs[m + np.arange(n), np.arange(n)] = b
    costs[m:, n:] = 0
    candidates = np.unique(costs[np.isfinite(costs)])
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    left, right = 0, len(candidates) - 1
    while left < right:
        middle = (left + right) // 2
        adjacency = csr_matrix(costs <= candidates[middle])
        feasible = np.all(maximum_bipartite_matching(adjacency, perm_type="column") >= 0)
        if feasible:
            right = middle
        else:
            left = middle + 1
    return float(candidates[left])
