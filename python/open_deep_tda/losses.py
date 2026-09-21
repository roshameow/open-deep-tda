"""Piecewise-differentiable topology losses with a fixed, detached reference.

Native PH/MST and assignment determine indices only; target lengths are always
read from live torch tensors. No target bar count normalization is applied.
"""

import math
import numbers

import numpy as np
import torch

from . import topology


_CRITICAL_EPS = 1e-12


def _tensor(X, name, ndim=None):
    if not isinstance(X, torch.Tensor) or not X.is_floating_point():
        raise ValueError("{} must be a floating-point torch tensor".format(name))
    if ndim is not None and X.ndim != ndim:
        raise ValueError("{} must have {} dimensions".format(name, ndim))
    if not bool(torch.isfinite(X).all()):
        raise ValueError("{} must be finite".format(name))
    return X


def _zero(X):
    # Empty sum stays connected, without overflowing a sum of large distances.
    return X.reshape(-1)[:0].sum()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("{} must be finite and positive".format(name))
    return float(value)


def _numpy(D):
    return D.detach().to(device="cpu", dtype=torch.float64).numpy()


def _distances(D_source, D_target):
    _tensor(D_source, "D_source", 2)
    _tensor(D_target, "D_target", 2)
    if D_source.shape != D_target.shape:
        raise ValueError("Source and target must have the same shape and sample IDs")
    # Validation applies even when cached source metadata avoids native work.
    source = topology._distance_array(_numpy(D_source))
    target = topology._distance_array(_numpy(D_target))
    reference = D_source.detach().to(device=D_target.device, dtype=D_target.dtype)
    return reference, source, target


def _edges(edges, n, name):
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("{} must be an integer array of shape (k, 2)".format(name))
    if np.any(edges < 0) or np.any(edges >= n) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("{} contains invalid vertex IDs".format(name))
    return edges.astype(np.int64, copy=False)


def _gather(D, edges):
    ids = torch.as_tensor(edges, dtype=torch.long, device=D.device)
    return D[ids[:, 0], ids[:, 1]]


def pairwise_distances(X):
    """Live Euclidean distances, including safe zero gradients at coincidences.

    Avoid cdist's Gram-matrix fast path: cancellation there can produce nonzero
    self-distances, violating the native filtration's zero-diagonal contract.
    """
    _tensor(X, "X", 2)
    D = torch.cdist(X, X, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    if not bool(torch.isfinite(D).all()):
        raise ValueError("Euclidean distances overflowed; rescale X")
    return D


def h0_loss(D_source, D_target, source_edges=None):
    """Symmetric identity-aware MST stress, divided by 2*(n-1)."""
    reference, source, target = _distances(D_source, D_target)
    n = len(source)
    if source_edges is not None:
        source_edges = _edges(source_edges, n, "source_edges")
        if len(source_edges) != max(0, n - 1):
            raise ValueError("source_edges must contain n-1 MST edges")
        # Cached trees are trusted to be minimal, but must actually be trees.
        parent = list(range(n))
        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for i, j in source_edges:
            a, b = root(i), root(j)
            if a == b:
                raise ValueError("source_edges must be an acyclic tree")
            parent[a] = b
    if n < 2:
        return _zero(D_target)
    if source_edges is None:
        source_edges = topology.mst(source)
    target_edges = topology.mst(target)
    difference = reference - D_target
    return (_gather(difference, source_edges).square().sum()
            + _gather(difference, target_edges).square().sum()) / (2 * (n - 1))


def _h1_pairs(result, n):
    if result.get("truncated", False):
        raise ValueError("Training losses reject truncated/censored persistence")
    if result["n_vertices"] != n:
        raise ValueError("Cached persistence has incompatible n_vertices")
    pairs = []
    for pair in result["pairs"]:
        if pair.get("censored", False):
            raise ValueError("Training losses reject censored persistence")
        if pair["dimension"] != 1:
            continue
        b, d = float(pair["birth"]), float(pair["death"])
        if pair["essential"] or not np.isfinite([b, d]).all() or d < b:
            raise ValueError("Full-filtration H1 bars must have finite ordered endpoints")
        if d > b:
            _edges(np.asarray([pair["birth_edge"], pair["death_edge"]]), n, "critical edges")
            pairs.append(pair)
    return pairs


def _pair_edges(pairs, key):
    return np.asarray([p[key] for p in pairs], dtype=np.int64).reshape(-1, 2)


def _live_diagram(D, pairs):
    return torch.stack((_gather(D, _pair_edges(pairs, "birth_edge")),
                        _gather(D, _pair_edges(pairs, "death_edge"))), dim=1)


def h1_loss(D_source, D_target, source_result=None, tau=0.01,
            budgets=None, max_matching_size=512):
    """Return exact matched H1 loss and reference-critical-edge regularization.

    The reference PH may be cached for these exact source distances and vertex
    IDs, using either full persistence or h1_persistence results. Target PH is
    recomputed via h1_persistence on every invocation. Finite-radius training is
    rejected rather than treating right-censored deaths as observed endpoints.
    Critical loss uses epsilon=1e-12 in its fixed reference-weight denominator.
    """
    tau = _positive(tau, "tau")
    options = topology._budgets({} if budgets is None else budgets, full_filtration=True)
    topology._integer_budget(max_matching_size, "max_matching_size")
    reference, source, target = _distances(D_source, D_target)
    if source_result is None:
        source_result = topology.h1_persistence(source, **options)
    source_pairs = _h1_pairs(source_result, len(source))
    target_result = topology.h1_persistence(target, **options)
    target_pairs = _h1_pairs(target_result, len(target))
    P = np.asarray([(p["birth"], p["death"]) for p in source_pairs], dtype=np.float64).reshape(-1, 2)
    Q = np.asarray([(p["birth"], p["death"]) for p in target_pairs], dtype=np.float64).reshape(-1, 2)
    matching = topology.diagram_matching(P, Q, max_matching_size=max_matching_size)
    live_source = _live_diagram(reference, source_pairs)
    live_target = _live_diagram(D_target, target_pairs)
    # A cache may be computed in numpy float64 and its reference distance
    # tensor subsequently stored in float32. Permit dtype rounding, not a
    # stale rescaled cache; comparing exact float64 endpoints would reject
    # this normal training path.
    tolerance = 8 * torch.finfo(D_source.dtype).eps
    if not np.allclose(_numpy(_live_diagram(D_source.detach(), source_pairs)), P,
                       rtol=tolerance, atol=0):
        raise ValueError("Cached source persistence endpoints do not match source critical edges")
    pd1 = _zero(D_target)
    matched = matching["matched"]
    if len(matched):
        i = torch.as_tensor(matched[:, 0], dtype=torch.long, device=D_target.device)
        j = torch.as_tensor(matched[:, 1], dtype=torch.long, device=D_target.device)
        pd1 = pd1 + (live_source[i] - live_target[j]).square().sum()
    for bars, ids in ((live_source, matching["unmatched_source"]),
                      (live_target, matching["unmatched_target"])):
        indices = torch.as_tensor(ids, dtype=torch.long, device=D_target.device)
        pd1 = pd1 + (bars[indices, 1] - bars[indices, 0]).square().sum() / 2
    pd1 = pd1 / max(1, len(source_pairs))
    crit1 = _zero(D_target)
    if source_pairs:
        lifetime = live_source[:, 1] - live_source[:, 0]
        weights = lifetime / (lifetime + tau)
        current = _live_diagram(D_target, source_pairs)
        crit1 = ((current - live_source).square().sum(dim=1) * weights).sum() / (
            2 * weights.sum() + _CRITICAL_EPS)
    return {"pd1": pd1, "crit1": crit1,
            "source_bars": len(source_pairs), "target_bars": len(target_pairs)}


def _lengths(source_lengths, target_lengths):
    _tensor(source_lengths, "source_lengths")
    _tensor(target_lengths, "target_lengths")
    if source_lengths.shape != target_lengths.shape:
        raise ValueError("Source and target lengths must have the same shape")
    if bool((source_lengths < 0).any()) or bool((target_lengths < 0).any()):
        raise ValueError("Lengths must be nonnegative")
    return source_lengths.detach().to(device=target_lengths.device, dtype=target_lengths.dtype)


def near_loss(source_lengths, target_lengths, delta=1.0):
    """Mean Huber loss (not smooth-L1's delta-rescaled convention)."""
    delta = _positive(delta, "delta")
    source = _lengths(source_lengths, target_lengths)
    if target_lengths.numel() == 0:
        return _zero(target_lengths)
    error = (target_lengths - source).abs()
    return torch.where(error <= delta, 0.5 * error.square(),
                       delta * (error - 0.5 * delta)).mean()


def separation_loss(source_lengths, target_lengths, margin=1.0):
    """Mean squared hinge toward min(reference length, margin)."""
    margin = _positive(margin, "margin")
    source = _lengths(source_lengths, target_lengths)
    if target_lengths.numel() == 0:
        return _zero(target_lengths)
    return torch.relu(source.clamp(max=margin) - target_lengths).square().mean()
