"""Bounded, exact F2 dual certificates for selected same-row-identity H1 classes.

All supplied rows participate, including off-witness vertices. Closed thresholds
and numeric validation are exactly those of :mod:`structural_h1`. Cochains are
chosen from the source alone, never adapted to the target. Success is sufficient,
NOT necessary: a valid target can violate this particular choice of duals.

This is not an optimizer, a full-complex isomorphism/chain-map certificate, full
barcode matching, a global big-data guarantee, or a semantic-preservation claim.
Separation candidates do not imply simultaneous Euclidean feasibility. Changes
to a layout require a new target triangle scan before certifying that layout.

Verified method/pitfall: reduce both edge vectors AND cycle labels, then solve
highest-pivot equations in ascending order with free coordinates zero. Explicitly
check original source boundaries and the full identity pairing. Attached target
triangles can violate a fixed dual without killing H1; off-witness cones and
relations between individually surviving classes must not be ignored. The tests
retain independent dense GF2 and primal-checker oracles for these distinctions.
Logical output budgets must charge pairing-cell emission as well as extraction,
and count repeated output vertex/index IDs separately from retained bitsets.
"""

from itertools import combinations

import numpy as np

from .structural_h1 import (
    H1BudgetExceeded, _chains, _integer_budget, _matrix, _radius, _triangles,
)


MAX_VERTICES = 64


class _DualBudget:
    """Conservative word-work and peak live logical storage, not Python RSS."""

    def __init__(self, edges, cycles, operations, entries):
        self.edge_words = max(1, (edges + 63) // 64)
        self.label_words = max(1, (cycles + 63) // 64)
        self.words = self.edge_words + self.label_words
        self.max_operations = operations
        self.max_entries = entries
        self.operations = self.live = self.peak = 0

    def work(self, units=1):
        amount = units * self.words
        if amount > self.max_operations - self.operations:
            raise H1BudgetExceeded("max_reduction_operations exceeded")
        self.operations += amount

    def reserve_words(self, amount):
        if amount > self.max_entries - self.live:
            raise H1BudgetExceeded("max_reduction_entries exceeded")
        self.live += amount
        self.peak = max(self.peak, self.live)

    def reserve(self, records=1):
        self.reserve_words(records * self.words)

    def vector(self, chain, edge_ids):
        self.work()
        vector = 0
        for edge in chain:
            self.work(2)  # Shift and XOR at full conservative record width.
            vector ^= 1 << edge_ids[edge]
        return vector

    def edge(self, edge):
        self.work(2)
        self.reserve_words(2)  # Output vertex IDs, before list allocation.
        return list(edge)


def _insert_labeled(vector, label, basis, budget):
    while True:
        budget.work(2)  # Zero test and leading-pivot inspection.
        if not vector:
            return False
        pivot = vector.bit_length() - 1
        if pivot not in basis:
            budget.reserve()  # BOTH edge vector and cycle-label words.
            basis[pivot] = (vector, label)
            return True
        row, row_label = basis[pivot]
        budget.work(2)
        vector ^= row
        label ^= row_label


def _evaluate_mask(chain, masks, edge_ids, budget):
    budget.work()
    value = 0
    for edge in chain:
        budget.work()
        value ^= masks[edge_ids[edge]]
    budget.work()  # Inspection of the returned parity, including zero.
    return value


def _source_masks(adjacency, chains, edge_ids, budget):
    """Source-only labeled elimination; caller releases basis after return."""
    basis = {}
    for triangle in _triangles(adjacency):
        vector = budget.vector(combinations(triangle, 2), edge_ids)
        _insert_labeled(vector, 0, basis, budget)
    boundary_rank = len(basis)
    budget.work()
    required = 0
    for j, chain in enumerate(chains):
        vector = budget.vector(chain, edge_ids)
        budget.work(2)  # Union and label shift.
        required |= vector
        if not _insert_labeled(vector, 1 << j, basis, budget):
            raise ValueError("invalid_source_contract: source classes are filled or dependent")

    # One separately rounded k-bit mask per ambient edge, including zero masks.
    budget.reserve_words(len(edge_ids) * budget.label_words)
    budget.work(len(edge_ids))
    masks = [0] * len(edge_ids)
    for pivot in sorted(basis):
        row, label = basis[pivot]
        budget.work(2)
        lower = row ^ (1 << pivot)
        value = label
        while True:
            budget.work()
            if not lower:
                break
            budget.work(5)  # Negate/AND, bit_length, mask XOR, lower XOR.
            bit = lower & -lower
            value ^= masks[bit.bit_length() - 1]
            lower ^= bit
        masks[pivot] = value
    return masks, boundary_rank, required


def dual_h1_certificate(
    D_source, D_target, cycles, birth_radius, survival_radius, *,
    max_vertices=64, max_simplices=200_000,
    max_reduction_operations=20_000_000, max_reduction_entries=2_000_000,
):
    """Return source dual cochains and all current forbidden target triangles.

    Matrices have identical row identities and sizes, are exactly symmetric,
    finite, nonnegative, and zero-diagonal. Triangle inequality is not required.
    Integer and up-to-float64 dtypes/radii retain their represented precision via
    the existing structural_h1 validators; wider floats are rejected. Radii obey
    0 <= birth <= survival, with distance <= radius defining membership.
    ``cycles`` is a nonempty iterable of edge-list chains. Undirected repeats
    cancel over F2. Every normalized chain must be closed, present in the source
    at birth, and independent modulo ALL source triangles at survival. Invalid
    source contracts raise ValueError containing ``invalid_source_contract``;
    malformed inputs raise ValueError. Inputs are not mutated.

    Result is a deterministic JSON-compatible dict. ``cochains[i]`` contains
    ``index`` and lexicographic ``support_edges`` over ALL ambient vertex pairs,
    even source-absent edges. These cochains depend only on the source, radii and
    ordered normalized cycles, not the target. ``source_pairing[i][j]`` is the
    explicitly checked alpha_i(Gamma_j) = delta_ij. ``source_boundary_rank`` and
    both ``source_triangle_count`` and ``target_triangle_count`` are reported.
    ``required_birth_edges`` is the normalized union; ``missing_target_birth_edges``
    lists its target birth violations. Each ``forbidden_triangles`` record has
    ``vertices``, zero-based ``violated_cocycles``, and ALL
    ``eligible_source_absent_edges`` (source distance STRICTLY > survival).
    Each forbidden triangle has such an edge, none required at birth.

    ``certificate_satisfied`` means birth edges exist and every target triangle
    pairs to zero with every fixed cochain. Identity pairing then proves that
    all selected classes stay independent over the interval. False is NOT proof
    of invalid target H1: this certificate is sufficient, not necessary. No
    target homology reduction, exact birth/death assertion, or preservation of
    all source classes is claimed. No partial result is returned on exhaustion.

    Budgets are nonnegative integers (bool is invalid; zero is a real limit):
    * max_vertices: per matrix, hard ceiling 64 even if a larger limit is given.
    * max_simplices: combined vertices + survival edges + survival triangles in
      BOTH complexes. Streaming preflight precedes all basis/mask allocation.
    * max_reduction_operations: cumulative conservative word-work through source
      elimination, substitution, explicit checks, target scan and output. Let
      E=choose(n,2), k=cycle count, We=max(1,ceil(E/64)), Wk=max(1,ceil(k/64)),
      W=We+Wk. Each initialization, inspection, bit_length, unary/binary bitset
      operation or emitted scalar costs W; a shift plus bit test costs 2W.
      Pivot inspection costs 2W even on zero. Mask initialization costs E*W.
    * max_reduction_entries: peak live logical 64-bit words. Every stored basis
      record reserves We+Wk, including zero labels; masks reserve E*Wk; pairing
      masks reserve k*Wk. Six W-word scratch records cover working values,
      required-edge union and intermediate bitset results throughout. The source
      basis is released after substitution; masks and outputs remain live.
      Every emitted triangle vertex, edge endpoint, cocycle index and pairing
      cell reserves one additional word BEFORE allocation (including duplicates).
      ``n_simplices``, ``reduction_operations`` and ``peak_reduction_entries``
      report the actual counters for exact-boundary reproducibility.

    These bound logical bitvector work/storage, NOT total RSS, parsing time or
    wall time. Matrix/adjacency arrays, normalized input chains, ambient edge
    maps, pivot keys, Python/container headers and fixed report metadata are
    bounded-domain bookkeeping outside these budgets. Triangle enumeration is
    streamed; no dense boundary matrix or complete triangle table is allocated.
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
    matrices = []
    for value, name in ((D_source, "D_source"), (D_target, "D_target")):
        try:
            matrices.append(_matrix(value, name, min(MAX_VERTICES, limits["max_vertices"])))
        except H1BudgetExceeded as exc:
            raise H1BudgetExceeded("max_vertices exceeded (hard ceiling: 64)") from exc
    source, target = matrices
    if source.shape != target.shape:
        raise ValueError("source and target must have equal sizes and the same vertex domain")
    n = len(source)
    chains = _chains(cycles, n)
    for j, chain in enumerate(chains):
        if any(source[u, v] > birth for u, v in chain):
            raise ValueError(f"invalid_source_contract: witness {j} has missing birth edges")

    adjacencies = (source <= survival, target <= survival)
    count, triangle_counts = 0, []
    for adjacency in adjacencies:
        count += n + int(np.count_nonzero(np.triu(adjacency, 1)))
        if count > limits["max_simplices"]:
            raise H1BudgetExceeded("max_simplices exceeded")
        triangles = 0
        for _ in _triangles(adjacency):
            count += 1
            triangles += 1
            if count > limits["max_simplices"]:
                raise H1BudgetExceeded("max_simplices exceeded")
        triangle_counts.append(triangles)

    edges = list(combinations(range(n), 2))
    edge_ids = {edge: i for i, edge in enumerate(edges)}
    k = len(chains)
    budget = _DualBudget(len(edges), k, limits["max_reduction_operations"],
                         limits["max_reduction_entries"])
    budget.reserve(6)
    masks, boundary_rank, required = _source_masks(adjacencies[0], chains, edge_ids, budget)
    budget.live -= (boundary_rank + k) * budget.words  # Helper's basis is gone.

    # Verify ORIGINAL obligations, not just equations in the reduced basis.
    for triangle in _triangles(adjacencies[0]):
        if _evaluate_mask(combinations(triangle, 2), masks, edge_ids, budget):
            raise RuntimeError("internal dual construction error: source triangle parity")
    budget.reserve_words(k * budget.label_words)
    pairing_masks = []
    for j, chain in enumerate(chains):
        mask = _evaluate_mask(chain, masks, edge_ids, budget)
        budget.work(2)  # Shift and comparison.
        if mask != 1 << j:
            raise RuntimeError("internal dual construction error: source cycle pairing")
        pairing_masks.append(mask)

    budget.work(3 * k * k)  # Shift, bit test AND, and emitted pairing cell.
    budget.reserve_words(k * k)
    pairing = [[(mask >> i) & 1 for mask in pairing_masks] for i in range(k)]
    cochains = []
    for i in range(k):
        budget.work()
        budget.reserve_words(1)
        support = []
        for edge, mask in zip(edges, masks):
            budget.work(2)
            if (mask >> i) & 1:
                support.append(budget.edge(edge))
        cochains.append({"index": i, "support_edges": support})

    required_edges, missing = [], []
    for index, edge in enumerate(edges):
        budget.work(2)
        if (required >> index) & 1:
            required_edges.append(budget.edge(edge))
            if target[edge] > birth:
                missing.append(budget.edge(edge))

    forbidden = []
    for triangle in _triangles(adjacencies[1]):
        parity = _evaluate_mask(combinations(triangle, 2), masks, edge_ids, budget)
        if not parity:
            continue
        violated, eligible = [], []
        for i in range(k):
            budget.work(2)
            if (parity >> i) & 1:
                budget.work()
                budget.reserve_words(1)
                violated.append(i)
        for edge in combinations(triangle, 2):
            if source[edge] > survival:
                eligible.append(budget.edge(edge))
        if not eligible:
            raise RuntimeError("internal dual construction error: forbidden source triangle")
        budget.work(3)
        budget.reserve_words(3)
        forbidden.append({"vertices": list(triangle), "violated_cocycles": violated,
                          "eligible_source_absent_edges": eligible})

    return {
        "scope": "sufficient_selected_h1_certificate_not_necessary_not_whole_complex_map",
        "n_vertices": n, "n_cocycles": k,
        "birth_radius": birth, "survival_radius": survival,
        "source_boundary_rank": boundary_rank,
        "source_triangle_count": triangle_counts[0], "source_pairing": pairing,
        "cochains": cochains, "required_birth_edges": required_edges,
        "missing_target_birth_edges": missing, "target_triangle_count": triangle_counts[1],
        "forbidden_triangles": forbidden, "certificate_satisfied": not missing and not forbidden,
        "n_simplices": count, "reduction_operations": budget.operations,
        "peak_reduction_entries": budget.peak,
    }
