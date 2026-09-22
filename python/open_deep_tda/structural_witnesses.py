"""Label-blind, exact bounded H1 image-basis selection on supplied row IDs.

This is a small-domain structural diagnostic, not a semantic certificate,
barcode matcher, layout solver, or scalable/global-data PH algorithm. No labels
are consumed, no rows are sampled or omitted, and distances/IDs are unchanged.

Method/pitfall: fundamental GRAPH cycles span the birth cycle space, including
cycles already filled at birth. Quotient by ALL survival triangle boundaries,
then retain an independent image basis. Eliminate every triangle pivot (not just
leading pivots); return the original birth chains, never their quotient vectors,
which can involve later edges. Tests exercise these distinctions with dense F2
oracles and same-ID witness verification.
"""

from itertools import combinations

import numpy as np

from .structural_h1 import (
    H1BudgetExceeded,
    _integer_budget,
    _matrix,
    _radius,
    _triangles,
    _ReductionBudget,
    _insert,
)


_SCOPE = (
    "complete supplied vertex domain within the bounded prototype; "
    "no assertion about omitted points, semantic structure, or global big-data topology"
)


def _fundamental_cycles(D, birth):
    """Stream canonical chains from a deterministic lexicographic forest.

    Add an edge iff its endpoints are disconnected in the current forest.
    Every remaining edge closes its unique forest path; later forest edges
    cannot change that path. Only one candidate chain is materialized at a time.
    """
    forest = [[] for _ in range(len(D))]
    for a, b in combinations(range(len(D)), 2):
        if D[a, b] > birth:
            continue
        parents = {a: None}
        queue = [a]
        for vertex in queue:
            for neighbor in forest[vertex]:
                if neighbor not in parents:
                    parents[neighbor] = vertex
                    queue.append(neighbor)
        if b not in parents:
            forest[a].append(b)
            forest[b].append(a)
            continue
        chain = [(a, b)]
        vertex = b
        while vertex != a:
            parent = parents[vertex]
            chain.append((min(vertex, parent), max(vertex, parent)))
            vertex = parent
        yield sorted(chain)


def select_h1_witnesses(
    D, birth_radius, survival_radius, *,
    max_cycles=8, max_vertices=64, max_simplices=200_000,
    max_reduction_operations=20_000_000, max_reduction_entries=2_000_000,
):
    """Return a JSON-compatible exact image-rank report and birth-edge witnesses.

    Computes im(H1(VR(D, birth)) -> H1(VR(D, survival))) over F2, on ALL
    supplied rows. Thresholds are closed (distance <= radius). D is a finite,
    symmetric, nonnegative square dissimilarity matrix with zero diagonal;
    triangle inequalities are not required. Shared structural_h1 validators
    preserve represented integer/float threshold comparisons without float32
    rounding or int64-to-float64 precision loss. Radii obey 0 <= birth <=
    survival < infinity. Inputs are not mutated.

    Birth edges are visited in lexicographic row-ID order to build a spanning
    forest. Fundamental graph cycles, in chord order, are reduced modulo all
    triangle boundaries at survival, then greedily tested for independence.
    ``cycles`` contains the ORIGINAL chains of the first ``max_cycles``
    independent classes, as sorted lists of canonical [i, j] edges (i < j),
    without repeated edges. These use the same row IDs as D and only birth
    edges. ``image_rank`` is always the full image dimension, even with a zero
    output cap. ``selected_count`` is len(cycles); ``complete_basis`` means
    selected_count == image_rank, including the valid empty rank-zero family.
    A nonempty returned family can be passed to check_h1_witnesses(D, D, ...).
    The verifier deliberately rejects empty families; the selector accepts them.

    All budgets are nonnegative integers, not bool. ``max_vertices`` defaults
    to 64 and shares the verifier's hard prototype ceiling of 128.
    ``max_simplices`` counts vertices + edges + triangles of the ONE survival
    complex. A streaming counting pass checks it before any boundary vectors
    or graph cycles are allocated. No dense boundary matrix is constructed.
    Resource exhaustion raises H1BudgetExceeded, never a partial rank/report.

    Reduction accounting reuses structural_h1: with W = max(1, ceil(C(n,2)/64)),
    initialization/pivot inspection costs W, a bit insertion or pivot test 2W,
    and XOR W. ``reduction_operations`` counts this conservative word-work.
    ``peak_reduction_entries`` counts logical 64-bit slots: W per stored
    triangle/class basis vector plus 3W reserved scratch, even for an empty
    complex. Reservation/checking precedes insertion/work. Returned chains,
    matrix/adjacency arrays, edge labels, graph traversal, integer headers and
    dictionaries are bounded-domain bookkeeping, NOT included in these word
    budgets; these are not total RSS or wall-time guarantees.

    Other metadata: n_vertices, birth_radius, survival_radius, scope, n_edges
    and n_triangles (at survival), n_simplices, graph_cycle_count (at birth),
    and reduction_words_per_vector. No semantic/global-data claims are made.
    """
    limits = {name: _integer_budget(name, value) for name, value in (
        ("max_cycles", max_cycles), ("max_vertices", max_vertices),
        ("max_simplices", max_simplices),
        ("max_reduction_operations", max_reduction_operations),
        ("max_reduction_entries", max_reduction_entries),
    )}
    birth = _radius("birth_radius", birth_radius)
    survival = _radius("survival_radius", survival_radius)
    if birth > survival:
        raise ValueError("birth_radius must not exceed survival_radius")
    source = _matrix(D, "D", limits["max_vertices"])
    n = len(source)
    adjacency = source <= survival
    n_edges = int(np.count_nonzero(np.triu(adjacency, 1)))
    n_simplices = n + n_edges
    if n_simplices > limits["max_simplices"]:
        raise H1BudgetExceeded("max_simplices exceeded")
    n_triangles = 0
    for _ in _triangles(adjacency):
        n_triangles += 1
        n_simplices += 1
        if n_simplices > limits["max_simplices"]:
            raise H1BudgetExceeded("max_simplices exceeded")

    edge_ids = {edge: index for index, edge in enumerate(combinations(range(n), 2))}
    budget = _ReductionBudget(len(edge_ids), limits["max_reduction_operations"],
                              limits["max_reduction_entries"])
    budget.reserve(3)
    boundaries = {}
    for i, j, k in _triangles(adjacency):
        value = budget.vector(((i, j), (i, k), (j, k)), edge_ids)
        _insert(value, boundaries, budget)
        del value
    pivots = sorted(boundaries, reverse=True)
    classes = {}
    selected = []
    graph_cycle_count = 0
    for chain in _fundamental_cycles(source, birth):
        graph_cycle_count += 1
        value = budget.vector(chain, edge_ids)
        # Canonical linear quotient map: even a free leading bit must not
        # prevent elimination of triangle pivots at lower coordinates.
        for pivot in pivots:
            budget.work(2)
            if (value >> pivot) & 1:
                budget.work()
                value ^= boundaries[pivot]
        if _insert(value, classes, budget) and len(selected) < limits["max_cycles"]:
            selected.append([list(edge) for edge in chain])
        del value

    return {
        "cycles": selected,
        "image_rank": len(classes),
        "selected_count": len(selected),
        "complete_basis": len(selected) == len(classes),
        "n_vertices": n,
        "birth_radius": birth,
        "survival_radius": survival,
        "scope": _SCOPE,
        "n_edges": n_edges,
        "n_triangles": n_triangles,
        "n_simplices": n_simplices,
        "graph_cycle_count": graph_cycle_count,
        "reduction_operations": budget.operations,
        "peak_reduction_entries": budget.peak,
        "reduction_words_per_vector": budget.words,
    }
