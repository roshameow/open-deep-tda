"""Constructive same-row-ID H1 obstructions for bounded F2 flag complexes.

Verified method/pitfall: quotient reduction must visit EVERY triangle pivot,
including pivots below a free leading edge. Carry triangle provenance through
both quotient and class elimination; otherwise a class dependency need not
come with a valid filling. Off-witness vertices and merged nonzero classes are
essential. Tests independently check dense GF2 ranks and certificate boundaries.
"""

from itertools import combinations

import numpy as np

from .structural_h1 import (
    H1BudgetExceeded, _ReductionBudget, _chains, _evaluate, _integer_budget,
    _matrix, _radius, _triangles,
)


class _CertificateBudget(_ReductionBudget):
    def __init__(self, edges, triangles, cycles, operations, entries):
        super().__init__(edges, operations, entries)
        # Separate Python bitsets need separately rounded word widths, including
        # a slot for zero. Charge even edge-only source work at this larger width.
        self.words += max(1, (triangles + 63) // 64) + max(1, (cycles + 63) // 64)

    def reserve_words(self, amount):
        if amount > self.max_entries - self.live:
            raise H1BudgetExceeded("max_reduction_entries exceeded")
        self.live += amount
        self.peak = max(self.peak, self.live)


def _insert_augmented(value, filling, coefficients, basis, budget):
    """Invariant: value = XOR(coefficients * cycles) + boundary(filling)."""
    while True:
        budget.work()
        if not value:
            return filling, coefficients
        pivot = value.bit_length() - 1
        if pivot not in basis:
            budget.reserve()  # Before allocating the stored record.
            basis[pivot] = (value, filling, coefficients)
            return None
        other_value, other_filling, other_coefficients = basis[pivot]
        budget.work(3)
        value ^= other_value
        filling ^= other_filling
        coefficients ^= other_coefficients


def _certificate(coefficients, filling, chains, triangles, edge_ids, budget):
    """Decode and independently reconstruct both sides of the boundary equation."""
    budget.work(3)
    if not coefficients:
        raise RuntimeError("internal error: empty witness relation")
    expected = actual = 0
    indices, faces = [], []
    for index, chain in enumerate(chains):
        budget.work(2)
        if (coefficients >> index) & 1:
            budget.reserve_words(1)
            indices.append(index)
            vector = budget.vector(chain, edge_ids)
            budget.work()
            expected ^= vector
    for index, triangle in enumerate(triangles):
        budget.work(2)
        if (filling >> index) & 1:
            budget.reserve_words(3)  # Emitted vertex IDs, before list allocation.
            faces.append(list(triangle))
            vector = budget.vector(combinations(triangle, 2), edge_ids)
            budget.work()
            actual ^= vector
    budget.work()
    if expected != actual:
        raise RuntimeError("internal error: target filling boundary mismatch")
    return {"kind": "filled" if len(indices) == 1 else "merged",
            "cycle_indices": indices, "filling_triangles": faces}


def _target_obstructions(target, adjacency, chains, birth, triangle_count,
                         edge_ids, budget):
    # Six augmented scratch records cover working inputs, shift/XOR outputs,
    # quotient/class provenance, and independent certificate reconstruction.
    budget.reserve(6)
    budget.reserve_words(3 * triangle_count)
    triangles = list(_triangles(adjacency))  # Counted before any list allocation.
    boundaries = {}
    for index, triangle in enumerate(triangles):
        value = budget.vector(combinations(triangle, 2), edge_ids)
        budget.work(2)
        _insert_augmented(value, 1 << index, 0, boundaries, budget)
    pivots = sorted(boundaries, reverse=True)
    classes, witnesses, relations = {}, [], []
    for index, chain in enumerate(chains):
        # Use parity-normalized edges, not the raw input sequence.
        missing = []
        for i, j in chain:
            if target[i, j] > birth:
                budget.reserve_words(2)
                missing.append([i, j])
        witness = {"cycle_index": index, "missing_birth_edges": missing,
                   "edges_present": not missing, "survives": False}
        witnesses.append(witness)
        if missing:
            continue
        value = budget.vector(chain, edge_ids)
        budget.work(3)
        filling, coefficients = 0, 1 << index
        for pivot in pivots:
            budget.work(2)
            if (value >> pivot) & 1:
                budget.work(2)
                value ^= boundaries[pivot][0]
                filling ^= boundaries[pivot][1]
        budget.work()
        witness["survives"] = bool(value)  # Before eliminating other classes.
        dependent = _insert_augmented(value, filling, coefficients, classes, budget)
        if dependent is not None:
            filling, coefficients = dependent
            relations.append(_certificate(coefficients, filling, chains, triangles,
                                          edge_ids, budget))
    return witnesses, relations, len(classes)


def find_h1_obstructions(
    D_source, D_target, cycles, birth_radius, survival_radius, *,
    max_vertices=64, max_simplices=200000,
    max_reduction_operations=20000000, max_reduction_entries=2000000,
):
    """Return JSON-compatible exact target obstructions, never partial success.

    The nonempty source family must consist of closed, nonzero, independent
    classes at survival, with every edge present at birth. Invalid contracts
    raise ValueError (``invalid_source_contract``). Undirected repeated edges
    cancel over F2. Matrices use identical row IDs, must be equally sized,
    symmetric, finite, nonnegative, and zero-diagonal; triangle inequality is
    unnecessary. Closed thresholds satisfy 0 <= birth <= survival. Numeric
    validation and exact scalar comparisons match ``check_h1_witnesses``.

    Result keys:
      * source_validated (True), source_rank (number of input cycles).
      * witnesses: input-ordered dictionaries with cycle_index,
        missing_birth_edges (sorted [i,j] vertex-ID pairs), edges_present,
        survives (present at birth and individually nonzero at survival).
      * relations: an independent basis of target relations among ONLY
        birth-present witnesses. Each has kind (``filled`` for one cycle,
        ``merged`` for two or more), cycle_indices (zero-based input indices),
        and filling_triangles (sorted [i,j,k] vertex-ID triples). The F2 boundary
        is exactly the XOR of those source chains. Every certificate is checked
        before return. All supplied target vertices participate, not just those
        on cycles. Relations need not enumerate every possible dependency.
      * surviving_rank, all_classes_independent (including the birth condition),
        n_simplices, reduction_operations, peak_reduction_entries.

    No optimizer, critical-edge matching, global PH, or chain-map claim is made.
    This is an interval witness certificate, not an assertion of exact birth
    or death times. Inputs are not mutated; output contains only JSON types.

    Budgets are nonnegative integers; exhaustion raises H1BudgetExceeded
    (RuntimeError), never a partial result. max_vertices has a HARD ceiling 64.
    max_simplices counts vertices + survival edges + survival triangles in BOTH
    complexes, with a streaming counting pass before reduction/list allocation.
    Work and storage budgets are genuinely cumulative/peak across source and
    target, including certificate checks, not independent per-stage budgets.

    Let E=choose(n,2), T=target survival triangle count, C=cycle count, and
    W=max(1,ceil(E/64))+max(1,ceil(T/64))+max(1,ceil(C/64)). Every stored reduction
    record reserves W logical 64-bit words BEFORE insertion, including full
    triangle and witness provenance widths. Even the source's edge-only
    reduction is conservatively charged at W. A vector initialization, pivot
    inspection, comparison or bitset XOR costs W; a shift/bit test or bit
    insertion costs 2W. Three source and six target scratch records are reserved
    before use. Triangle tables, filling-triangle IDs, missing-edge IDs and
    relation cycle-index lists additionally reserve one word per ID before
    allocation; certificate outputs remain live.
    Source reduction storage is released before target allocation.

    These are conservative logical bitvector work/storage limits, NOT total
    Python RSS or wall-time limits. Input parsing, matrix/adjacency arrays,
    normalized chains, edge maps, pivot labels, container headers, status
    dictionaries and enumeration scans are bounded-domain bookkeeping outside
    these budgets. No dense boundary matrix is allocated.
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
            matrices.append(_matrix(value, name, min(64, limits["max_vertices"])))
        except H1BudgetExceeded as exc:
            raise H1BudgetExceeded("max_vertices exceeded (hard prototype ceiling: 64)") from exc
    source, target = matrices
    if source.shape != target.shape:
        raise ValueError("source and target must have equal sizes and the same vertex domain")
    n = len(source)
    chains = _chains(cycles, n)
    for index, chain in enumerate(chains):
        if any(source[i, j] > birth for i, j in chain):
            raise ValueError(f"invalid_source_contract: witness {index} has missing birth edges")
    adjacencies = (source <= survival, target <= survival)
    count, triangle_counts = 0, []
    for adjacency in adjacencies:
        count += n + int(np.count_nonzero(np.triu(adjacency, 1)))
        if count > limits["max_simplices"]:
            raise H1BudgetExceeded("max_simplices exceeded")
        triangle_count = 0
        for _ in _triangles(adjacency):
            count += 1
            triangle_count += 1
            if count > limits["max_simplices"]:
                raise H1BudgetExceeded("max_simplices exceeded")
        triangle_counts.append(triangle_count)
    edge_ids = {edge: index for index, edge in enumerate(combinations(range(n), 2))}
    budget = _CertificateBudget(len(edge_ids), triangle_counts[1], len(chains),
                                limits["max_reduction_operations"],
                                limits["max_reduction_entries"])
    _, source_rank = _evaluate(source, adjacencies[0], chains, birth, edge_ids, budget)
    if source_rank != len(chains):
        raise ValueError("invalid_source_contract: source classes are filled or dependent")
    budget.live = 0  # _evaluate's bases and scratch have been released.
    witnesses, relations, rank = _target_obstructions(
        target, adjacencies[1], chains, birth, triangle_counts[1], edge_ids, budget)
    return {"source_validated": True, "source_rank": source_rank,
            "witnesses": witnesses, "relations": relations,
            "surviving_rank": rank, "all_classes_independent": rank == len(chains),
            "n_simplices": count, "reduction_operations": budget.operations,
            "peak_reduction_entries": budget.peak}
