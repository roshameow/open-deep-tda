"""Fuzzy neighbor cross-entropy, not an exact (Parametric) UMAP reproduction.

Local exponential neighbor probabilities and fuzzy union are existing ideas:
McInnes, Healy & Melville (2018), https://arxiv.org/abs/1802.03426.
We use a fixed-scale Cauchy kernel and an explicit sampled nonedge loss, without
UMAP's fitted curve, optimizer, or sampling/weighting equivalence claims.
"""
import math
import numbers

import numpy as np
from scipy.spatial.distance import cdist
import torch


_REPULSION_EPS = 1e-8


def build_fuzzy_weights(reference, neighbors, edges):
    """Return float32 weights aligned with ``edges`` and JSON diagnostics.

    ``reference`` is finite real (N,D); ``neighbors`` is integer (N,K), with
    distinct nonself IDs per row (row ordering is irrelevant). ``edges`` is an
    integer (E,2) array of nonself IDs: subsets, either orientation, repeated
    edges and nonedges are allowed. Nonedges receive zero, missing reverse
    neighbors contribute zero. No N-by-N distances are allocated.

    rho is the smallest strictly positive neighbor distance, or zero for an
    all-duplicate row. sigma solves sum(exp(-max(d-rho,0)/sigma))=max(1,log2(K)).
    When duplicates/ties make this target unreachable, sigma=0 denotes the
    limiting hard weights 1[d<=rho]; this is reported, not hidden jitter. K=0
    gives zero weights; K=1 gives weight one for its sole directed neighbor.
    """
    try:
        raw = np.asarray(reference)
        nbr = np.asarray(neighbors)
        edge = np.asarray(edges)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected rectangular numeric arrays") from exc
    if raw.ndim != 2 or raw.shape[1] == 0 or raw.dtype.kind not in "iuf":
        raise ValueError("reference must be a real matrix with at least one feature")
    with np.errstate(over="ignore", invalid="ignore"):
        X = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("reference must be finite and float64-representable")
    n = len(X)
    if n > math.isqrt(np.iinfo(np.int64).max):
        raise ValueError("population exceeds int64 edge-key capacity")
    if nbr.ndim != 2 or nbr.shape[0] != n or nbr.dtype.kind not in "iu":
        raise ValueError("neighbors must be an integer (N,K) array")
    k = nbr.shape[1]
    if k > max(0, n - 1) or np.any(nbr < 0) or np.any(nbr >= n):
        raise ValueError("invalid neighbor IDs or K")
    if edge.ndim != 2 or edge.shape[1] != 2 or edge.dtype.kind not in "iu":
        raise ValueError("edges must be an integer (E,2) array")
    if np.any(edge < 0) or np.any(edge >= n) or np.any(edge[:, 0] == edge[:, 1]):
        raise ValueError("invalid edge IDs")
    nbr, edge = nbr.astype(np.int64, copy=False), edge.astype(np.int64, copy=False)
    directed = np.empty((n, k), dtype=np.float64)
    rhos, sigmas, masses = [], [], []
    saturated = unattainable = duplicate_rows = 0
    target = max(1., math.log2(k)) if k else 0.
    for i, row in enumerate(nbr):
        if np.any(row == i) or len(np.unique(row)) != k:
            raise ValueError("neighbors contain self or duplicate IDs")
        if not k:
            continue
        d = cdist(X[i:i + 1], X[row])[0]
        if not np.isfinite(d).all():
            raise ValueError("neighbor distances overflowed; rescale reference")
        positive = d[d > 0]
        rho = float(positive.min()) if len(positive) else 0.
        gap = np.maximum(d - rho, 0.)
        minimum_mass = int(np.count_nonzero(gap == 0))
        duplicate_rows += int(np.any(d == 0))
        if minimum_mass >= target:
            w = (gap == 0).astype(np.float64)
            sigma = 0.
            saturated += 1
            unattainable += int(minimum_mass > target)
        else:
            # Search in dimensionless units to preserve scale equivariance,
            # including for very small distances. Fixed 64 iterations.
            unit = float(gap.max())
            delta = gap / unit
            low, high = 0., 1.
            while np.exp(-delta / high).sum() < target:
                high *= 2.
            for _ in range(64):
                mid = (low + high) / 2
                if np.exp(-delta / mid).sum() < target:
                    low = mid
                else:
                    high = mid
            w = np.exp(-delta / high)
            sigma = high * unit
        directed[i] = w
        rhos.append(rho)
        sigmas.append(sigma)
        masses.append(float(w.sum()))
    keys = (np.arange(n, dtype=np.int64)[:, None] * n + nbr).ravel()
    order = np.argsort(keys)
    keys, values = keys[order], directed.ravel()[order]

    def lookup(a, b):
        query = a * n + b
        ids = np.searchsorted(keys, query)
        result = np.zeros(len(query), dtype=np.float64)
        valid = np.flatnonzero(ids < len(keys))
        valid = valid[keys[ids[valid]] == query[valid]]
        result[valid] = values[ids[valid]]
        return result

    forward = lookup(edge[:, 0], edge[:, 1])
    reverse = lookup(edge[:, 1], edge[:, 0])
    weights = np.clip(forward + reverse - forward * reverse, 0, 1).astype(np.float32)

    def bounds(values):
        return [float(min(values)), float(max(values))] if values else [None, None]

    diagnostics = {
        "n_samples": n, "n_neighbors": k, "edge_count": len(edge),
        "target_directed_mass": target, "directed_mass_range": bounds(masses),
        "rho_range": bounds(rhos), "sigma_range": bounds(sigmas),
        "sigma_zero_rows": saturated, "unattainable_target_rows": unattainable,
        "rows_with_duplicate_points": duplicate_rows,
        "weight_range": bounds(weights.tolist()),
        "metric": "euclidean", "symmetrization": "a+b-a*b",
        "degenerate_rule": "sigma=0 hard limit when minimum mass >= target; no jitter",
        "claim": "existing fuzzy-neighbor probability idea; not exact UMAP reproduction",
    }
    return weights, diagnostics


def fuzzy_losses(positive_lengths_tensor, negative_lengths_tensor, weights_tensor, scale=1.0):
    """Return positive CE and sampled-negative repulsion (each a scalar mean).

    t=(d/scale)^2; attraction=log1p(t); repulsion=log1p(1/(t+1e-8)).
    Positive CE averages w*attraction+(1-w)*repulsion; negatives average
    repulsion. Scale is fixed, positive and finite; outputs are NOT normalized.
    Inputs are nonnegative finite floating vectors on the same device/dtype,
    weights in [0,1] aligned with positives. Empty terms are connected zeros.
    Log-domain evaluation avoids squared-length overflow. Half precision is
    promoted to float32; distances below dtype tiny have zero branch gradient.
    """
    if (isinstance(scale, (bool, np.bool_)) or not isinstance(scale, numbers.Real)
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("scale must be finite and positive")
    pos, neg, weights = positive_lengths_tensor, negative_lengths_tensor, weights_tensor
    for name, tensor in (("positive lengths", pos), ("negative lengths", neg), ("weights", weights)):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.ndim != 1:
            raise ValueError(f"{name} must be a floating tensor vector")
        if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
            raise ValueError(f"{name} must be finite and nonnegative")
    if weights.shape != pos.shape or bool((weights > 1).any()):
        raise ValueError("weights must match positive lengths and lie in [0,1]")
    if any(t.device != pos.device or t.dtype != pos.dtype for t in (neg, weights)):
        raise ValueError("all tensors must share device and dtype")
    if pos.dtype in (torch.float16, torch.bfloat16):
        pos, neg, weights = pos.float(), neg.float(), weights.float()

    def terms(d):
        log_t = 2 * (d.clamp_min(torch.finfo(d.dtype).tiny).log() - math.log(scale))
        log_t = torch.where(d == 0, log_t.new_tensor(-math.inf), log_t)
        attraction = torch.logaddexp(log_t, torch.zeros_like(log_t))
        log_denominator = torch.logaddexp(log_t, log_t.new_tensor(math.log(_REPULSION_EPS)))
        repulsion = torch.logaddexp(-log_denominator, torch.zeros_like(log_denominator))
        return attraction, repulsion

    if pos.numel():
        attraction, repulsion = terms(pos)
        positive = (weights * attraction + (1 - weights) * repulsion).mean()
    else:
        positive = pos.sum() + weights.sum()
    negative = terms(neg)[1].mean() if neg.numel() else neg.sum()
    return positive, negative
