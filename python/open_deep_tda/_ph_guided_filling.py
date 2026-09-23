"""Bounded exact fixed-family H1 filling oracle and persistent cut loss.

Private, independent implementation. Closed Rips thresholds, GF(2), all row IDs;
only the specified source classes are checked (not the full barcode). Distances
are symmetric float64 dissimilarities; no triangle inequality is required.
The oracle operates on detached NumPy matrices; the loss recomputes live torch
lengths, so discrete certificates are never differentiated. Do not mutate inputs
concurrently. Budgets count logical 64-bit bitvector words, not Python RSS.
"""

from dataclasses import dataclass
from numbers import Integral
from time import monotonic

import numpy as np


class FillingResourceError(RuntimeError):
    """An exact answer was not computed; never interpret as certification."""


@dataclass(frozen=True)
class FillingLimits:
    max_vertices: int = 300
    max_edges: int = 12_000
    max_triangles: int = 250_000
    max_cycles: int = 128
    max_chain_edges: int = 100_000
    max_word_ops: int = 100_000_000_000
    max_peak_words: int = 25_000_000
    deadline_seconds: float = 45.0


@dataclass(frozen=True)
class FillingRelation:
    cycle_indices: tuple
    filling_triangles: tuple
    cheapest_edge_to_cut: tuple
    # Proof: boundary(filling_triangles) = XOR of the indexed input chains.


@dataclass(frozen=True)
class FillingResult:
    accepted: bool
    surviving_rank: int
    edges_present_at_birth: tuple
    edges_present_at_survival: tuple
    missing_birth_edges: tuple
    missing_survival_edges: tuple
    relations: tuple
    target_edges: int
    target_triangles: int
    word_ops: int
    peak_words: int


_HARD = dict(max_vertices=300, max_edges=12_000, max_triangles=250_000,
             max_cycles=128, max_chain_edges=100_000)


def _limits(limits):
    if not isinstance(limits, FillingLimits):
        raise ValueError("limits must be FillingLimits")
    for key in _HARD.keys() | {"max_word_ops", "max_peak_words"}:
        v = getattr(limits, key)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0:
            raise ValueError(key + " must be a nonnegative integer")
        if key in _HARD and v > _HARD[key]:
            raise ValueError(key + " exceeds hard ceiling")
    if (isinstance(limits.deadline_seconds, (bool, np.bool_))
            or not isinstance(limits.deadline_seconds, (int, float))
            or not np.isfinite(limits.deadline_seconds) or limits.deadline_seconds <= 0):
        raise ValueError("deadline_seconds must be finite and positive")


class _Budget:
    def __init__(self, limits):
        self.limits = limits
        self.end = monotonic() + limits.deadline_seconds
        self.work = self.live = self.peak = 0

    def tick(self, count=1):
        if count > self.limits.max_word_ops - self.work:
            raise FillingResourceError("max_word_ops exceeded")
        self.work += count
        if monotonic() >= self.end:
            raise FillingResourceError("deadline_seconds exceeded")

    def reserve(self, count):
        self.tick()
        if count > self.limits.max_peak_words - self.live:
            raise FillingResourceError("max_peak_words exceeded")
        self.live += count
        self.peak = max(self.peak, self.live)

    def release(self, count):
        self.live -= count


def _matrix(value, nmax, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float64"):
        raise ValueError(name + " must be a float64 ndarray")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(name + " must be square")
    if len(value) > nmax:
        raise FillingResourceError("max_vertices exceeded")
    if (not np.isfinite(value).all() or np.any(value < 0)
            or not np.array_equal(value, value.T) or np.any(np.diag(value) != 0)):
        raise ValueError(name + " must be finite, nonnegative, symmetric, zero-diagonal")
    return value


def _chains(cycles, n, budget):
    if not isinstance(cycles, (list, tuple)):
        raise ValueError("cycles must be a sized sequence")
    if len(cycles) > budget.limits.max_cycles:
        raise FillingResourceError("max_cycles exceeded")
    if not cycles:
        raise ValueError("invalid_source_contract: empty family")
    result, raw = [], 0
    for cycle in cycles:
        if not isinstance(cycle, (list, tuple)):
            raise ValueError("each cycle must be a sized sequence")
        raw += len(cycle)
        if raw > budget.limits.max_chain_edges:
            raise FillingResourceError("max_chain_edges exceeded")
        parity = set()
        for edge in cycle:
            budget.tick()
            if (not isinstance(edge, (tuple, list)) or len(edge) != 2
                    or any(isinstance(u, (bool, np.bool_)) or not isinstance(u, Integral)
                           or u < 0 or u >= n for u in edge) or edge[0] == edge[1]):
                raise ValueError("invalid_source_contract: invalid edge")
            pair = tuple(sorted((int(edge[0]), int(edge[1]))))
            if pair in parity:
                parity.remove(pair)
            else:
                parity.add(pair)
        boundary = set()
        for u, v in parity:
            boundary.symmetric_difference_update((u, v))
        if not parity or boundary:
            raise ValueError("invalid_source_contract: zero or non-closed cycle")
        result.append(tuple(sorted(parity)))
    return tuple(result)


def _complex(D, radius, budget):
    n = len(D)
    neighbors = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        budget.tick()
        for j in range(i + 1, n):
            if D[i, j] <= radius:
                if len(edges) >= budget.limits.max_edges:
                    raise FillingResourceError("max_edges exceeded")
                edges.append((i, j))
                neighbors[i].add(j)
                neighbors[j].add(i)
    ids = {e: k for k, e in enumerate(edges)}
    triangles = []
    # Enumerate each clique once; charge intersection probes and check deadline
    # on every graph edge, even if no triangles are emitted.
    for i, j in edges:
        budget.tick()
        for k in neighbors[i].intersection(neighbors[j]):
            budget.tick()
            if k > j:
                if len(triangles) >= budget.limits.max_triangles:
                    raise FillingResourceError("max_triangles exceeded")
                triangles.append((i, j, k))
    triangles.sort()
    return edges, ids, triangles


def _reduce(value, provenance, coefficients, basis, width, budget, insert):
    # Invariant: value = XOR(coefficients * input cycles) XOR boundary(provenance).
    while value:
        budget.tick(width)
        pivot = value.bit_length() - 1
        other = basis.get(pivot)
        if other is None:
            if insert:
                budget.reserve(width)
                basis[pivot] = (value, provenance, coefficients)
            return value, provenance, coefficients, insert
        budget.tick(3 * width)
        value ^= other[0]
        provenance ^= other[1]
        coefficients ^= other[2]
    return 0, provenance, coefficients, False


def _boundary_basis(ids, triangles, width, budget, provenance):
    basis = {}
    for index, (i, j, k) in enumerate(triangles):
        budget.tick(4 * width)
        vector = (1 << ids[i, j]) ^ (1 << ids[i, k]) ^ (1 << ids[j, k])
        _reduce(vector, (1 << index) if provenance else 0, 0,
                basis, width, budget, True)
    return basis


def _quotient(value, basis, width, budget):
    # Visit EVERY boundary pivot, including ones below free coordinates.
    fill = 0
    for pivot in sorted(basis, reverse=True):
        budget.tick(width)
        if value & (1 << pivot):
            budget.tick(2 * width)
            row = basis[pivot]
            value ^= row[0]
            fill ^= row[1]
    return value, fill


def _vector(chain, ids, width, budget):
    value = 0
    for edge in chain:
        budget.tick(2 * width)
        value ^= 1 << ids[edge]
    return value


def find_filling_obstructions(source_D, target_D, cycles, birth_radius,
                              survival_radius, *, limits=FillingLimits()):
    """Return full-domain target status plus independent, verified GF2 fillings.

    Source must contain each supplied nonzero closed chain at birth and the
    family must be independent modulo all source triangles at survival. Target
    cycles missing survival edges are excluded from target quotient relations;
    their missing edges are explicitly returned (not a certification). The
    source is verified HERE, not delegated to the caller. Radius ties are in.
    Exhaustion raises FillingResourceError without returning partial results.
    """
    _limits(limits)
    budget = _Budget(limits)
    S = _matrix(source_D, limits.max_vertices, "source_D")
    D = _matrix(target_D, limits.max_vertices, "target_D")
    if S.shape != D.shape:
        raise ValueError("source and target must share row IDs and shape")
    if (isinstance(birth_radius, (bool, np.bool_)) or
            isinstance(survival_radius, (bool, np.bool_)) or
            not all(isinstance(r, (int, float, np.integer, np.floating))
                    and np.isfinite(r) for r in (birth_radius, survival_radius)) or
            birth_radius < 0 or birth_radius > survival_radius):
        raise ValueError("require finite 0 <= birth_radius <= survival_radius")
    chains = _chains(cycles, len(S), budget)
    # Every ORIGINAL supplied edge must already exist at source birth. F2
    # cancellation is valid for chain algebra, not permission to smuggle an
    # unavailable diagonal in twice and call it a source-born witness.
    for chain in cycles:
        for edge in chain:
            budget.tick()
            if S[tuple(sorted((int(edge[0]), int(edge[1]))))] > birth_radius:
                raise ValueError('invalid_source_contract: raw supplied edge missing at birth')
    if any(S[e] > birth_radius for chain in chains for e in chain):
        raise ValueError("invalid_source_contract: edge missing at source birth")

    _, src_ids, src_triangles = _complex(S, survival_radius, budget)
    sw = max(1, (len(src_ids) + 63) // 64)
    budget.reserve(3 * sw + 3 * len(src_triangles))
    source_basis = _boundary_basis(src_ids, src_triangles, sw, budget, False)
    classes = {}
    for index, chain in enumerate(chains):
        value, _ = _quotient(_vector(chain, src_ids, sw, budget), source_basis, sw, budget)
        value, _, _, _ = _reduce(value, 0, 1 << index, classes, sw, budget, True)
        if not value:
            raise ValueError("invalid_source_contract: dependent or filled source class")
    budget.release(3 * sw + 3 * len(src_triangles) + sw * (len(source_basis) + len(classes)))
    del source_basis, classes, src_triangles

    edges, ids, triangles = _complex(D, survival_radius, budget)
    width = (max(1, (len(edges) + 63) // 64)
             + max(1, (len(triangles) + 63) // 64)
             + max(1, (len(chains) + 63) // 64))
    # Conservatively account for three scratch vectors and retained triangle IDs.
    budget.reserve(3 * width + 3 * len(triangles))
    basis = _boundary_basis(ids, triangles, width, budget, True)
    birth, survival, missing_birth, missing_survival = [], [], [], []
    classes = {}
    relations = []
    for index, chain in enumerate(chains):
        budget.tick()
        mb = tuple(e for e in chain if D[e] > birth_radius)
        ms = tuple(e for e in chain if D[e] > survival_radius)
        missing_birth.append(mb)
        missing_survival.append(ms)
        birth.append(not mb)
        survival.append(not ms)
        if ms:
            continue
        value, fill = _quotient(_vector(chain, ids, width, budget), basis, width, budget)
        value, fill, coeff, _ = _reduce(value, fill, 1 << index,
                                         classes, width, budget, True)
        if value:
            continue
        # Verify the boundary equation independently against actual triangles.
        boundary = 0
        faces, candidates = [], set()
        for t, (i, j, k) in enumerate(triangles):
            budget.tick()
            if fill & (1 << t):
                budget.tick(4 * width)
                faces.append((i, j, k))
                boundary ^= (1 << ids[i, j]) ^ (1 << ids[i, k]) ^ (1 << ids[j, k])
                for e in ((i, j), (i, k), (j, k)):
                    if S[e] > survival_radius:
                        candidates.add(e)
        expected = 0
        indices = []
        for q, other in enumerate(chains):
            budget.tick()
            if coeff & (1 << q):
                indices.append(q)
                expected ^= _vector(other, ids, width, budget)
        if not indices or expected != boundary or not candidates:
            raise RuntimeError("internal GF2 filling or source-independence invariant failed")
        # Cheapest positive hinge to move above b; lexicographic tie break.
        chosen = min(candidates, key=lambda e: (max(0.0, survival_radius - D[e]), e))
        budget.reserve(3 * len(faces) + len(indices) + 2)
        relations.append(FillingRelation(tuple(indices), tuple(faces), chosen))
    rank = len(classes)
    return FillingResult(all(birth) and all(survival) and rank == len(chains), rank,
                         tuple(birth), tuple(survival), tuple(missing_birth),
                         tuple(missing_survival), tuple(relations), len(edges),
                         len(triangles), budget.work, budget.peak)


class FillingCutPool:
    """Monotone cut set; configure one instance per fixed source Γ and radius.

    add(result) is transactional: over-capacity raises without changing the pool.
    A new oracle result may add cuts but never deletes existing constraints.
    """

    def __init__(self, cycles, birth_radius, survival_radius, *, margin=0.0, max_cuts=500):
        if (isinstance(max_cuts, bool) or not isinstance(max_cuts, Integral)
                or max_cuts < 0 or max_cuts > 500):
            raise ValueError("max_cuts must be an integer in [0,500]")
        if (not np.isfinite(birth_radius) or not np.isfinite(survival_radius)
                or not np.isfinite(margin) or birth_radius < 0
                or survival_radius < birth_radius or margin < 0):
            raise ValueError("invalid radii or margin")
        with np.errstate(over='ignore', invalid='ignore'):
            cutoff = float(survival_radius) + float(margin)
        if not np.isfinite(cutoff):
            raise ValueError('survival + margin must be representable and finite')
        self.birth = float(birth_radius)
        self.survival = float(survival_radius)
        self.margin = float(margin)
        self.max_cuts = int(max_cuts)
        self.cycle_edges = tuple(sorted({tuple(sorted(e)) for cycle in cycles for e in cycle}))
        self.cuts = set()

    def add(self, result):
        if not isinstance(result, FillingResult):
            raise ValueError("expected a completed FillingResult")
        proposed = self.cuts | {r.cheapest_edge_to_cut for r in result.relations}
        if len(proposed) > self.max_cuts:
            raise FillingResourceError("max_cuts exceeded")
        self.cuts = proposed
        return tuple(sorted(proposed))

    def loss(self, Z, *, include_survival_upper=True):
        """Squared live distance hinges; source-cycle birth upper is dedicated.

        Returns (birth_upper, survival_upper, cut_lower), unweighted scalar
        tensors. The optional survival upper prevents missing edges at b from
        being ignored even when a caller disables the stricter birth objective.
        Z is a 2D floating torch tensor; zero norm gives a finite subgradient.
        """
        import torch
        if not isinstance(Z, torch.Tensor) or Z.ndim != 2 or not Z.is_floating_point() or not bool(torch.isfinite(Z).all()):
            raise ValueError("Z must be a finite floating 2D torch tensor")
        if max(self.birth, self.survival, self.survival + self.margin) > torch.finfo(Z.dtype).max:
            raise ValueError('cycle/cut radius unrepresentable in live target dtype')
        if any(u < 0 or v >= len(Z) for u, v in self.cycle_edges) or any(
                u < 0 or v >= len(Z) for u, v in self.cuts):
            raise ValueError("edge vertex outside Z")
        def lengths(edges):
            if not edges:
                return Z.reshape(-1)[:0]
            pairs = torch.as_tensor(sorted(edges), device=Z.device, dtype=torch.long)
            return torch.linalg.vector_norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], dim=1)
        born = lengths(self.cycle_edges)
        cut = lengths(self.cuts)
        birth_loss = torch.relu(born - self.birth).square().sum()
        survival_loss = torch.relu(born - self.survival).square().sum() if include_survival_upper else born.sum() * 0
        cut_loss = torch.relu(self.survival + self.margin - cut).square().sum()
        return birth_loss, survival_loss, cut_loss
