"""Anchored sampled neighbor NCE; not exact t-SNE or UMAP.

The graph supplies positives and excludes negatives. Feature-duplicate exclusion
is the caller's responsibility: intersect the returned mask with that policy.
"""
import math
import numbers
from typing import Optional

import numpy as np
import torch


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Integral) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class AnchoredNeighborSampler:
    """Sample uniformly from the unique undirected source-union edges.

    Edges must be integer (E,2), nonself and in [0,n); n >= 2. Reversed and
    repeated edges are canonicalized, not given extra probability mass. Storage
    is O(E), with O(B*C) sampling workspace, never a dense population matrix.
    """

    def __init__(self, edges, n):
        self.n = _integer(n, "n", 2)
        if self.n > math.isqrt(np.iinfo(np.int64).max):
            raise ValueError("n exceeds int64 canonical edge-key capacity")
        edge = np.asarray(edges)
        if edge.ndim != 2 or edge.shape[1] != 2 or edge.dtype.kind not in "iu":
            raise ValueError("edges must be an integer (E,2) array")
        if (np.any(edge < 0) or np.any(edge >= self.n)
                or np.any(edge[:, 0] == edge[:, 1])):
            raise ValueError("edges contain invalid or self indices")
        self.edges = np.unique(np.sort(edge.astype(np.int64), axis=1), axis=0)
        self.keys = self.edges[:, 0] * self.n + self.edges[:, 1]
        # Prevent accidental edits that would desynchronize the exclusion keys.
        self.edges.setflags(write=False)
        self.keys.setflags(write=False)

    def sample(self, batch_size, candidates, rng):
        """Return oriented positives (B,2), negative IDs and bool mask (B,C).

        Positive edges are drawn with replacement, each orientation with equal
        probability. Each negative slot is one uniform draw from [0,n), also
        with replacement: repeated valid IDs contribute repeated NCE terms.
        Self and ALL source-union edges are masked, without retries or forced
        filling. Dense/complete graphs can therefore have no valid negatives.
        Masked IDs are still in range; ignore them using the mask. Zero batch
        size or an empty graph returns B=0; C=0 is supported. rng must be a
        numpy.random.Generator; no global random state is used.
        """
        batch_size = _integer(batch_size, "batch_size")
        candidates = _integer(candidates, "candidates")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        if not len(self.edges) or not batch_size:
            return (np.empty((0, 2), dtype=np.int64),
                    np.empty((0, candidates), dtype=np.int64),
                    np.empty((0, candidates), dtype=bool))
        pos = self.edges[rng.integers(len(self.edges), size=batch_size)].copy()
        reverse = rng.integers(2, size=batch_size).astype(bool)
        pos[reverse] = pos[reverse, ::-1]
        neg = rng.integers(self.n, size=(batch_size, candidates), dtype=np.int64)
        anchors = pos[:, :1]
        query = np.minimum(anchors, neg) * self.n + np.maximum(anchors, neg)
        indices = np.searchsorted(self.keys, query)
        # Clip only for safe lookup; a query past the last key is not an edge.
        is_edge = ((indices < len(self.keys))
                   & (self.keys[np.minimum(indices, len(self.keys) - 1)] == query))
        valid = (neg != anchors) & ~is_edge
        return pos, neg, valid


def sample_anchor_pairs(edges, n, batch_size, negative_candidates, rng):
    """Return anchors (B,), positives (B,), negatives (B,C), valid (B,C).

    Convenience wrapper; reuse AnchoredNeighborSampler to avoid rebuilding
    canonical keys each batch. Sampling and empty semantics are identical.
    """
    pos, neg, valid = AnchoredNeighborSampler(edges, n).sample(
        batch_size, negative_candidates, rng)
    return pos[:, 0], pos[:, 1], neg, valid


def _log_kernel(d, scale):
    # Substitute only exact zeros: clamping subnormals changes the formula when
    # scale is similarly small. The zero branch has an exact zero derivative.
    log_d = torch.where(d > 0, d, torch.ones_like(d)).log()
    log_t = 2 * (log_d - math.log(scale))
    log_t = log_t.masked_fill(d == 0, -math.inf)
    return torch.logaddexp(log_t, torch.zeros_like(log_t))


class _RelativeScores(torch.autograd.Function):
    """Fuse temperature with the kernel derivative in log space.

    Ordinary chaining can overflow 1/temperature or underflow the kernel's
    sigmoid derivative even when their product with 1/d is representable.
    This analytical VJP combines all factors before exponentiation.
    """

    @staticmethod
    def forward(ctx, pos, neg, valid, scale, temperature):
        ctx.save_for_backward(pos, neg, valid)
        ctx.scale, ctx.temperature = scale, temperature
        relative = (_log_kernel(pos, scale)[:, None] - _log_kernel(neg, scale))
        return (relative.masked_fill(~valid, 0) / temperature).masked_fill(~valid, -math.inf)

    @staticmethod
    def backward(ctx, grad):
        pos, neg, valid = ctx.saved_tensors
        grad = grad.masked_fill(~valid, 0)

        def product(d, upstream):
            log_d = torch.where(d > 0, d, torch.ones_like(d)).log()
            log_den = torch.logaddexp(2 * log_d, log_d.new_tensor(2 * math.log(ctx.scale)))
            log_derivative = math.log(2) + log_d - log_den - math.log(ctx.temperature)
            magnitude = upstream.abs()
            log_upstream = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude)).log()
            log_product = (log_derivative + log_upstream).masked_fill(
                (d == 0) | (magnitude == 0), -math.inf)
            return upstream.sign() * log_product.exp()

        return product(pos, grad.sum(dim=1)), -product(neg, grad), None, None, None


def neighbor_nce_loss(pos_lengths, neg_lengths, weights, valid, scale: float,
                      temperature: float = 1.0,
                      hard_negatives: Optional[int] = None):
    """Self-normalized weighted mean of anchored, sampled neighbor NCE.

    score(d) = -log1p((d/scale)^2)/temperature; each row's loss is
    logsumexp([positive score, valid negative scores]) - positive score.
    This is a sampled contrastive objective, NOT exact t-SNE/UMAP. Nonnegative
    weights are normalized within the batch (not an unbiased population ratio).
    Rows without negatives contribute zero but retain their weight in the mean.
    Empty batches, zero weight mass or zero candidates give connected zeros.

    Lengths and weights must be finite, nonnegative floating tensors sharing
    dtype/device, of shapes (B,), (B,C), (B,). valid must be bool (B,C) on the
    same device; even masked lengths must be finite/nonnegative. scale and
    temperature are fixed finite positive real scalars. hard_negatives is None
    (all valid) or a nonnegative integer: keep up to K highest DETACHED negative
    scores per row. K=0 disables negatives; tie ordering follows torch.topk.
    Selection is detached, selected scores are live; masked/unselected lengths
    have zero gradient. Duplicate zero lengths are finite, with zero derivative.
    Half/bfloat16 computation is promoted to float32. Temperatures that risk
    overflowing float32 logits promote computation to float64. Log-domain
    scores and an analytical log-domain backward avoid intermediate squared-
    length overflow and derivative underflow, without clamping subnormal
    lengths. Truly unrepresentable results/gradients can still overflow.
    """
    for name, value in (("scale", scale), ("temperature", temperature)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if hard_negatives is not None:
        hard_negatives = _integer(hard_negatives, "hard_negatives")
    pos, neg = pos_lengths, neg_lengths
    for name, tensor, ndim in (("pos_lengths", pos, 1), ("neg_lengths", neg, 2),
                               ("weights", weights, 1)):
        if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
                or tensor.ndim != ndim):
            raise ValueError(f"{name} must be a floating tensor with ndim={ndim}")
        if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
            raise ValueError(f"{name} must be finite and nonnegative")
    if weights.shape != pos.shape or neg.shape[0] != len(pos):
        raise ValueError("lengths and weights must have matching batch dimensions")
    if any(t.device != pos.device or t.dtype != pos.dtype for t in (neg, weights)):
        raise ValueError("lengths and weights must share dtype and device")
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != neg.shape or valid.device != pos.device):
        raise ValueError("valid must be a bool tensor matching neg_lengths shape/device")
    if pos.dtype in (torch.float16, torch.bfloat16):
        pos, neg, weights = pos.float(), neg.float(), weights.float()
    # A bound on |log kernel| for finite float32 lengths, with spare headroom
    # for centering. Promote before forming logits, not after an overflow.
    float32_max = torch.finfo(torch.float32).max
    logit_bound = 2 * (math.log(float32_max) + abs(math.log(scale))) + 2
    if pos.dtype == torch.float32 and not (
            logit_bound / float32_max <= temperature <= float32_max):
        pos, neg, weights = pos.double(), neg.double(), weights.double()
    # Empty sums connect every input without overflowing a sum of huge lengths.
    zero = sum(t.reshape(-1)[:0].sum() for t in (pos, neg, weights))
    if (not pos.numel() or not neg.shape[1] or hard_negatives == 0
            or not bool(valid.any()) or not bool((weights > 0).any())):
        return zero

    if hard_negatives is not None and hard_negatives < neg.shape[1]:
        # Positive temperature preserves ranking; defer division until after
        # centering so equal, enormous-magnitude scores do not become inf-inf.
        negative = -_log_kernel(neg.detach(), scale)
        ids = negative.masked_fill(~valid, -math.inf).topk(hard_negatives, dim=1).indices
        neg = neg.gather(1, ids)
        valid = valid.gather(1, ids)
    # Shift by the positive first: avoids subtracting two large, nearly equal
    # log partitions and makes all-masked rows exactly zero with zero gradient.
    relative = _RelativeScores.apply(pos, neg, valid, scale, temperature)
    per_row = torch.logsumexp(torch.cat((torch.zeros_like(pos[:, None]), relative), dim=1), dim=1)
    # Scaling by detached max leaves the ratio/gradient unchanged and prevents
    # weight-mass overflow. It also avoids imposing an arbitrary epsilon floor.
    normalized = weights / weights.detach().max()
    return (normalized / normalized.sum() * per_row).sum() + zero
