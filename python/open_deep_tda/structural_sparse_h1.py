"""Bounded, standalone exact selected-family H1 analysis over GF(2).

The complex is the flag (Vietoris--Rips) complex on ALL rows of D, with
closed thresholds D[i, j] <= radius. No triangle inequality is assumed.
Only the supplied family is certified, NOT all H1 or a persistence diagram.
The analyzer has no dependency on other structural modules. The integer
reduction is an original implementation of ordinary Gaussian elimination over
GF(2). For comparison across complexes, row i must denote the SAME vertex and
cycles must use those shared global IDs; edge coordinate indices need not match.

Proof: triangle boundaries span B1 of the flag complex's 2-skeleton. Encoding a
closed chain in the full survival-edge basis and eliminating all boundary pivots
computes its class in C1/B1. The rank of the reduced selected columns is their
rank modulo B1 (equivalently rank([B1, cycles])-rank(B1)). Birth-edge membership
places them in Z1 at a. Independence at b implies independence throughout [a,b],
because any earlier boundary relation would persist at b. Higher-dimensional
simplices do not change H1. This certifies only the supplied family, not a map
between all source/target chains or an equality of whole homology groups.

Verified methods/pitfalls (regression-tested in test_structural_sparse_h1.py):
* Use survival edges, not choose(N, 2), as bit coordinates.
* Count triangles in a streaming pass before allocating any reduction basis.
* A quotient map must eliminate EVERY boundary pivot, including pivots below
  a free leading coordinate; merely stopping at that coordinate is incorrect.
* Vertices outside the chains can fill them; individually surviving classes
  can be dependent. Birth membership is checked on EVERY supplied edge,
  including repetitions that later cancel over GF(2).

Synthetic tests include a 960-row complex with 748660 streamed triangles and an
independent surviving square; no private dataset is required. Dense small-case
oracles and the existing exact checker independently exercise quotient rank.

Limits bound logical full-width 64-bit reduction words, not Python RSS or
wall time. Matrix/graph/chain bookkeeping is separately bounded by the shape,
edge, chain and triangle caps. Callers must not mutate inputs concurrently.
"""

from dataclasses import dataclass
from numbers import Integral
import math

import numpy as np


__all__ = ["H1Limits", "ResourceLimitError", "H1Result", "CycleStatus",
           "validate_distance_matrix", "validate_cycles", "analyze_sparse_h1"]


class ResourceLimitError(ValueError):
    """Resource-specific ValueError; no partial rank/certificate is returned."""


@dataclass(frozen=True)
class H1Limits:
    """Nonnegative integer budgets; zero is a real limit.

    Vertex/edge/triangle ceilings cannot exceed 1024/100000/2000000.
    Chain ceilings cannot exceed 1024 cycles / 1000000 supplied edges.
    Word-work/storage budgets may be increased explicitly. Each integer
    operation is conservatively charged at W=max(1, ceil(E_survival/64))
    words, even if its operands are shorter. Storage charges W per stored
    pivot plus 3W scratch (working vector, temporary mask, XOR output).
    """

    max_vertices: int = 1024
    max_edges: int = 100_000
    max_triangles: int = 2_000_000
    max_word_ops: int = 20_000_000_000
    max_storage_words: int = 25_000_000
    max_cycles: int = 1024
    max_chain_edges: int = 1_000_000


@dataclass(frozen=True)
class CycleStatus:
    """Input-order status; open/malformed chains raise rather than return.

    ``edges_present`` means ALL supplied edges exist at birth. ``survives``
    means this holds and the individual quotient class is nonzero. It does
    not assert independence from the other requested classes.
    """

    closed: bool
    edges_present: bool
    survives: bool
    reason: str


@dataclass(frozen=True)
class H1Result:
    certified: bool
    rank: int
    reason: str
    work: dict
    statuses: tuple


_HARD_CAPS = {"max_vertices": 1024, "max_edges": 100_000,
              "max_triangles": 2_000_000, "max_cycles": 1024,
              "max_chain_edges": 1_000_000}


def _check_limits(limits):
    if not isinstance(limits, H1Limits):
        raise ValueError("limits must be H1Limits")
    for name in H1Limits.__dataclass_fields__:
        value = getattr(limits, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        if name in _HARD_CAPS and value > _HARD_CAPS[name]:
            raise ValueError(f"{name} exceeds hard ceiling {_HARD_CAPS[name]}")


def validate_distance_matrix(D, *, limits=H1Limits()):
    """Return D unchanged after strict validation; never coerce or copy it.

    D must be a float64 numpy ndarray, square, finite, exactly symmetric,
    nonnegative and exactly zero on the diagonal. Empty matrices are allowed.
    Oversize inputs raise ResourceLimitError before scanning their contents.
    Malformed values, shapes, dtypes and limits raise ValueError.
    """
    _check_limits(limits)
    if not isinstance(D, np.ndarray) or D.dtype != np.dtype(np.float64):
        raise ValueError("D must be a float64 ndarray")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be square and two-dimensional")
    if len(D) > limits.max_vertices:
        raise ResourceLimitError("max_vertices exceeded")
    if (not np.isfinite(D).all() or np.any(D < 0)
            or np.any(np.diag(D) != 0) or not np.array_equal(D, D.T)):
        raise ValueError("D must be finite, symmetric, nonnegative and zero-diagonal")
    return D


def _sequence(value, name):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sized list, tuple or ndarray")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        raise ValueError(f"{name} must be a sequence")


def validate_cycles(cycles, n, *, limits=H1Limits()):
    """Validate chains on vertex IDs [0,n); return immutable canonical pairs.

    The return is a tuple of chains, each a tuple of pairs (min(u,v),max(u,v)).
    Repetitions are RETAINED for birth-edge checking, but cancel for closure
    and homology. IDs must be nonboolean integers, distinct within an edge.
    Chains must have zero GF(2) vertex boundary; open chains raise ValueError.
    Empty families and zero chains are valid (zero chains cannot certify).
    Input containers must be sized list/tuple/ndarray, not unbounded iterators.
    The total raw edge count is checked before allocating normalized chains.
    """
    _check_limits(limits)
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if n > limits.max_vertices:
        raise ResourceLimitError("max_vertices exceeded")
    _sequence(cycles, "cycles")
    if len(cycles) > limits.max_cycles:
        raise ResourceLimitError("max_cycles exceeded")
    total = 0
    for chain in cycles:
        _sequence(chain, "chain")
        if isinstance(chain, np.ndarray) and (chain.ndim != 2 or chain.shape[1] != 2):
            raise ValueError("chain arrays must have shape (E,2)")
        total += len(chain)
        if total > limits.max_chain_edges:
            raise ResourceLimitError("max_chain_edges exceeded")
    normalized = []
    for index, chain in enumerate(cycles):
        edges, boundary = [], set()
        for edge in chain:
            _sequence(edge, "edge")
            if len(edge) != 2 or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                or not 0 <= v < n for v in edge
            ):
                raise ValueError("each edge needs two integer vertex IDs in range")
            u, v = map(int, edge)
            if u == v:
                raise ValueError("self edges are not allowed")
            edges.append((min(u, v), max(u, v)))
            boundary.symmetric_difference_update((u, v))
        if boundary:
            raise ValueError(f"cycle {index} is not closed over GF(2)")
        normalized.append(tuple(edges))
    return tuple(normalized)


def _radius(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    if isinstance(value, Integral):
        # Python float/int comparisons preserve the exact integer threshold.
        value = int(value)
    elif isinstance(value, (float, np.floating)):
        if isinstance(value, np.floating) and value.dtype.itemsize > 8:
            raise ValueError(f"{name} cannot have precision wider than float64")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    else:
        raise ValueError(f"{name} must be an integer or float of at most 64 bits")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _survival_graph(D, b, limits):
    forward = [set() for _ in range(len(D))]
    edge_ids = {}
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            if float(D[i, j]) <= b:
                if len(edge_ids) >= limits.max_edges:
                    raise ResourceLimitError("max_edges exceeded")
                edge_ids[i, j] = len(edge_ids)
                forward[i].add(j)
    return forward, edge_ids


def _triangles(forward):
    # At most N common neighbours in temporary storage; no triangle list.
    for i, neighbours in enumerate(forward):
        for j in sorted(neighbours):
            for k in sorted(neighbours.intersection(forward[j])):
                yield i, j, k


class _WordBudget:
    def __init__(self, width, limits):
        self.width, self.limits = width, limits
        self.word_ops = self.storage_words = self.peak_storage_words = 0

    def charge(self, count=1):
        amount = count * self.width
        if amount > self.limits.max_word_ops - self.word_ops:
            raise ResourceLimitError("max_word_ops exceeded during reduction")
        self.word_ops += amount

    def reserve(self, count=1):
        amount = count * self.width
        if amount > self.limits.max_storage_words - self.storage_words:
            raise ResourceLimitError("max_storage_words exceeded during reduction")
        self.storage_words += amount
        self.peak_storage_words = max(self.peak_storage_words, self.storage_words)

    def encode(self, edges, edge_ids):
        self.charge()  # initialization
        bits = 0
        for edge in edges:
            self.charge(2)  # shift and XOR, before allocation
            bits ^= 1 << edge_ids[edge]
        return bits


def _insert(bits, pivots, budget):
    while True:
        budget.charge()  # zero / leading-coordinate inspection
        if bits == 0:
            return
        leading = bits.bit_length() - 1
        if leading not in pivots:
            budget.reserve()  # reserve before retaining the vector
            pivots[leading] = bits
            return
        budget.charge()
        bits ^= pivots[leading]


def analyze_sparse_h1(D, cycles, a, b, *, limits=H1Limits()):
    """Analyze selected closed chains at birth a and survival b, 0 <= a <= b.

    Returns H1Result(certified, rank, reason, work, statuses). Rank is the
    dimension of the span of supplied classes modulo ALL triangle boundaries
    at b, excluding chains with ANY supplied edge absent at a. Certification
    means rank == number of supplied chains; an empty family certifies rank 0.
    A filled or dependent family gives certified=False, not a resource error.
    Missing birth edges yield explicit per-chain failure, never KeyError.

    Matrix and cycle contracts are those of the public validation helpers.
    There is no source-contract check or claimed equality of full H1 groups.
    Triangle count is preflighted by streaming before any integer reduction.
    Preflight also checks necessary minimum word work (8W per triangle and
    (2*raw_edges+3)W per birth-present chain) and 3W scratch storage. These
    are lower bounds for admission, not a promise that reduction will fit:
    every actual operation/insertion is checked again at runtime, and any
    exhaustion raises ResourceLimitError with NO partial result. Dictionary
    access and bounded graph enumeration are not integer-vector word work.
    """
    D = validate_distance_matrix(D, limits=limits)
    a, b = _radius(a, "a"), _radius(b, "b")
    if a > b:
        raise ValueError("a must not exceed b")
    chains = validate_cycles(cycles, len(D), limits=limits)
    present = tuple(all(float(D[u, v]) <= a for u, v in chain) for chain in chains)
    forward, edge_ids = _survival_graph(D, b, limits)
    triangle_count = 0
    for _ in _triangles(forward):
        triangle_count += 1
        if triangle_count > limits.max_triangles:
            raise ResourceLimitError("max_triangles exceeded in preflight")
    width = max(1, (len(edge_ids) + 63) // 64)
    minimum_ops = width * (8 * triangle_count + sum(
        2 * len(chain) + 3 for chain, valid in zip(chains, present) if valid))
    if minimum_ops > limits.max_word_ops:
        raise ResourceLimitError("max_word_ops exceeded in preflight")
    if 3 * width > limits.max_storage_words:
        raise ResourceLimitError("max_storage_words exceeded in preflight")
    budget = _WordBudget(width, limits)
    budget.reserve(3)
    boundaries = {}
    for i, j, k in _triangles(forward):
        _insert(budget.encode(((i, j), (i, k), (j, k)), edge_ids), boundaries, budget)
    ordered_pivots = sorted(boundaries, reverse=True)
    selected, statuses = {}, []
    for chain, valid in zip(chains, present):
        if not valid:
            statuses.append(CycleStatus(True, False, False, "missing_birth_edges"))
            continue
        bits = budget.encode(chain, edge_ids)
        for pivot in ordered_pivots:
            budget.charge(2)  # right shift and bit test
            if (bits >> pivot) & 1:
                budget.charge()
                bits ^= boundaries[pivot]
        budget.charge()  # zero test
        survives = bool(bits)
        statuses.append(CycleStatus(True, True, survives,
                                    "survives" if survives else "zero_at_survival"))
        _insert(bits, selected, budget)
        del bits
    rank = len(selected)
    certified = rank == len(chains)
    reason = ("certified_selected_family" if certified else
              "missing_birth_edges" if not all(present) else
              "filled_or_dependent_selected_family")
    work = {"vertices": len(D), "edges": len(edge_ids), "triangles": triangle_count,
            "cycles": len(chains), "chain_edges": sum(map(len, chains)),
            "basis_width": len(edge_ids), "words_per_vector": width,
            "preflight_min_word_ops": minimum_ops, "word_ops": budget.word_ops,
            "peak_storage_words": budget.peak_storage_words,
            "boundary_rank": len(boundaries), "selected_rank": rank}
    return H1Result(certified, rank, reason, work, tuple(statuses))
