"""Exact, small-instance, same-row-identity H1 interval witness verification.

This is neither an optimizer nor barcode matching/global scalable PH. A supplied
chain must exist at ``birth_radius`` and remain nonzero at ``survival_radius``;
it need not be born at the first radius or die at the second. Filtrations use
closed thresholds (distance <= radius), coefficients F2, and ALL supplied rows.
Triangle inequality is not required for the symmetric nonnegative dissimilarities.
Only the supplied cycle representatives are checked: this does not certify a
simplicial/chain map on the entire source complex, equality of full persistence
diagrams, or preservation of every source homology class.

Invalid source contracts raise ValueError with ``invalid_source_contract`` in
the message; they are never reported as target failures. Malformed inputs raise
ValueError. Resource exhaustion raises H1BudgetExceeded, never a partial result.

Verified method/pitfall: quotient representatives require eliminating *every*
triangle pivot, not just leading pivots until the first free coordinate. External
vertices can fill a witness, and two individually nonzero classes can merge.
The accompanying tests retain independent dense-GF2 checks of these facts.
NumPy scalar promotion can change threshold membership; Python scalar comparisons
avoid both float32 threshold rounding and int64-to-float64 loss of precision.
"""

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
import warnings

import numpy as np


class H1BudgetExceeded(RuntimeError):
    """An exact computation could not complete within the requested budgets."""


@dataclass(frozen=True)
class H1WitnessResult:
    """Target status, in input order, after F2 parity cancellation.

    ``survives`` means edges are present at birth AND the class is nonzero at
    survival. Missing-at-birth witnesses are excluded from the joint rank.
    """

    edges_present: bool
    survives: bool


@dataclass(frozen=True)
class H1WitnessCheck:
    witnesses: tuple
    surviving_rank: int
    all_classes_independent: bool
    n_simplices: int
    reduction_operations: int
    peak_reduction_entries: int

    @property
    def accepted(self):
        """Every requested source class has an independent target witness."""
        return self.all_classes_independent


def _integer_budget(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _radius(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and nonnegative")
    if isinstance(value, Integral):
        value = int(value)  # Do not round integers through float64.
    elif isinstance(value, (float, np.floating)):
        if isinstance(value, np.floating) and value.dtype.itemsize > 8:
            raise ValueError(f"{name}: floating precision wider than float64 is unsupported")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite and nonnegative")
    else:
        raise ValueError(f"{name} must be an integer or a float of at most 64 bits")
    if value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _matrix(value, name, max_vertices):
    try:
        with warnings.catch_warnings():
            # Older NumPy warns (newer NumPy raises) for ragged matrices.
            warnings.filterwarnings("error", message="Creating an ndarray from ragged")
            raw = np.asarray(value)
    except (TypeError, ValueError, Warning) as exc:
        raise ValueError(f"{name} must be a real square matrix") from exc
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real square matrix")
    if raw.dtype.kind == "f" and raw.dtype.itemsize > 8:
        raise ValueError(f"{name}: floating precision wider than float64 is unsupported")
    if raw.shape[0] > min(max_vertices, 128):
        raise H1BudgetExceeded("max_vertices exceeded (hard prototype ceiling: 128)")
    if (not np.isfinite(raw).all() or np.any(raw < 0)
            or np.any(np.diag(raw) != 0) or not np.array_equal(raw, raw.T)):
        raise ValueError(f"{name} must be finite, nonnegative, symmetric, with zero diagonal")
    # Python scalar comparisons preserve large integer thresholds and the exact
    # represented values of float16/32/64. NumPy's scalar promotion can instead
    # round a threshold down to float32 or round an int64 edge up/down to float64.
    return raw.astype(object)


def _chains(cycles, n):
    result = []
    try:
        for index, cycle in enumerate(cycles):
            parity = set()
            for edge in cycle:
                edge = tuple(edge)
                if len(edge) != 2 or any(
                    isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                    or not 0 <= v < n for v in edge
                ) or edge[0] == edge[1]:
                    raise ValueError("edges must have two distinct integer vertex IDs in range")
                pair = tuple(sorted(map(int, edge)))
                if pair in parity:
                    parity.remove(pair)
                else:
                    parity.add(pair)
            if not parity:
                raise ValueError(f"invalid_source_contract: witness {index} is empty after parity")
            boundary = set()
            for edge in parity:
                boundary.symmetric_difference_update(edge)
            if boundary:
                raise ValueError(f"invalid_source_contract: witness {index} is not closed")
            result.append(tuple(sorted(parity)))
            # An independent family cannot exceed the ambient edge dimension.
            if len(result) > n * (n - 1) // 2:
                raise ValueError("invalid_source_contract: too many witnesses for independence")
    except TypeError as exc:
        raise ValueError("cycles must be an iterable of edge-list chains") from exc
    if not result:
        raise ValueError("invalid_source_contract: supply at least one witness")
    return result


def _triangles(adjacency):
    # Stream triples: never allocate a list of triangles or boundary columns.
    for i, j in combinations(range(len(adjacency)), 2):
        if adjacency[i, j]:
            for k in range(j + 1, len(adjacency)):
                if adjacency[i, k] and adjacency[j, k]:
                    yield i, j, k


class _ReductionBudget:
    def __init__(self, n_edges, operations, entries):
        self.words = max(1, (n_edges + 63) // 64)
        self.max_operations = operations
        self.max_entries = entries
        self.operations = 0
        self.live = 0
        self.peak = 0

    def work(self, units=1):
        amount = units * self.words
        if amount > self.max_operations - self.operations:
            raise H1BudgetExceeded("max_reduction_operations exceeded")
        self.operations += amount

    def reserve(self, vectors=1):
        amount = vectors * self.words
        if amount > self.max_entries - self.live:
            raise H1BudgetExceeded("max_reduction_entries exceeded")
        self.live += amount
        self.peak = max(self.peak, self.live)

    def vector(self, edges, edge_ids):
        self.work()
        value = 0
        for edge in edges:
            self.work(2)  # shift and XOR, charged at full ambient width
            value ^= 1 << edge_ids[edge]
        return value


def _insert(value, basis, budget):
    while True:
        budget.work()  # zero/leading-pivot inspection
        if not value:
            return False
        pivot = value.bit_length() - 1
        if pivot not in basis:
            budget.reserve()
            basis[pivot] = value
            return True
        budget.work()
        value ^= basis[pivot]


def _evaluate(D, adjacency, chains, birth, edge_ids, budget):
    # Three full-width scratch slots cover the working value, mask and XOR
    # output, in addition to all stored basis vectors (conservative accounting).
    budget.reserve(3)
    boundaries = {}
    for i, j, k in _triangles(adjacency):
        value = budget.vector(((i, j), (i, k), (j, k)), edge_ids)
        _insert(value, boundaries, budget)
        del value
    pivots = sorted(boundaries, reverse=True)
    classes = {}
    statuses = []
    for chain in chains:
        present = all(D[i, j] <= birth for i, j in chain)
        if not present:
            statuses.append(H1WitnessResult(False, False))
            continue
        value = budget.vector(chain, edge_ids)
        # A canonical, linear quotient map: do not stop at a free leading bit.
        for pivot in pivots:
            budget.work(2)  # shift and bit test
            if (value >> pivot) & 1:
                budget.work()
                value ^= boundaries[pivot]
        budget.work()
        statuses.append(H1WitnessResult(True, bool(value)))
        _insert(value, classes, budget)
        del value
    return tuple(statuses), len(classes)


def check_h1_witnesses(
    D_source, D_target, cycles, birth_radius, survival_radius, *,
    max_vertices=128, max_simplices=1000000,
    max_reduction_operations=10000000, max_reduction_entries=1000000,
):
    """Verify a nonempty independent source family and its same-ID target image.

    ``cycles`` is an iterable of edge lists; undirected repeats cancel over F2.
    IDs refer to the same rows of equally sized source and target matrices.
    Inputs are not mutated. Symmetry and zero diagonals are checked exactly;
    radii must satisfy 0 <= birth_radius <= survival_radius < infinity.
    Matrix integer dtypes and floating dtypes up to float64 are supported;
    radii are integers or floats up to float64. Wider floats are rejected rather
    than rounded. Comparisons use Python scalars, retaining exact represented
    values across float32/float64 and integer/float thresholds.
    Source edges must exist at birth, and source classes must be independent
    modulo ALL source triangle boundaries at survival, or ValueError is raised.

    Return H1WitnessCheck with per-witness target statuses, joint surviving
    rank, and ``all_classes_independent`` (also ``accepted``). Rank includes only
    witnesses present at target birth. All supplied rows enter both complexes,
    even rows not incident to any witness. Nonzero at the closed survival
    threshold certifies persistence throughout this interval by inclusion.

    Budgets (nonnegative integers, zero is a real limit):
    * max_vertices: per matrix, with a hard ceiling of 128 in this prototype.
    * max_simplices: combined vertices + edges + triangles in BOTH survival
      complexes. A counting pass checks this before any boundary allocation.
    * max_reduction_operations: cumulative conservative full-width word-work
      units across both reductions. With W = max(1, ceil(choose(n,2)/64)),
      vector initialization/pivot inspection/zero test costs W, setting a bit
      costs 2W, pivot testing costs 2W, and a row XOR costs W.
    * max_reduction_entries: peak logical 64-bit word slots: W per stored
      triangle/class basis vector, plus 3W scratch. Source storage is released
      before target reduction. Reservation precedes every basis insertion.

    These bound integer-bitvector work/storage, not total Python RSS or input
    parsing/triangle-enumeration time. Matrix/adjacency arrays, edge labels,
    normalized chains, dictionaries and Python integer headers are additional
    bounded-width bookkeeping. No dense boundary matrix is constructed.
    """
    limits = {name: _integer_budget(name, value) for name, value in (
        ("max_vertices", max_vertices), ("max_simplices", max_simplices),
        ("max_reduction_operations", max_reduction_operations),
        ("max_reduction_entries", max_reduction_entries),
    )}
    birth = _radius("birth_radius", birth_radius)
    survival = _radius("survival_radius", survival_radius)
    if birth > survival:
        raise ValueError("birth_radius must not exceed survival_radius")
    source = _matrix(D_source, "D_source", limits["max_vertices"])
    target = _matrix(D_target, "D_target", limits["max_vertices"])
    if source.shape != target.shape:
        raise ValueError("source and target must have equal sizes and the same vertex domain")
    n = len(source)
    chains = _chains(cycles, n)
    for index, chain in enumerate(chains):
        if any(source[i, j] > birth for i, j in chain):
            raise ValueError(f"invalid_source_contract: witness {index} has missing birth edges")
    adjacencies = (source <= survival, target <= survival)
    count = 0
    for adjacency in adjacencies:
        count += n + int(np.count_nonzero(np.triu(adjacency, 1)))
        if count > limits["max_simplices"]:
            raise H1BudgetExceeded("max_simplices exceeded")
        for _ in _triangles(adjacency):
            count += 1
            if count > limits["max_simplices"]:
                raise H1BudgetExceeded("max_simplices exceeded")
    edge_ids = {edge: index for index, edge in enumerate(combinations(range(n), 2))}
    budget = _ReductionBudget(len(edge_ids), limits["max_reduction_operations"],
                              limits["max_reduction_entries"])
    _, source_rank = _evaluate(source, adjacencies[0], chains, birth, edge_ids, budget)
    if source_rank != len(chains):
        raise ValueError("invalid_source_contract: source classes are filled or dependent")
    # _evaluate's local bases are gone; only statuses/rank escape the function.
    budget.live = 0
    statuses, target_rank = _evaluate(target, adjacencies[1], chains, birth, edge_ids, budget)
    return H1WitnessCheck(statuses, target_rank, target_rank == len(chains), count,
                          budget.operations, budget.peak)
