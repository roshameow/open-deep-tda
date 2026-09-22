"""Exact selector contract, with an independent dense GF(2) nullspace oracle.

Regression pitfalls: survival triangles may use vertices outside a returned
chain; two nonzero cycles may represent the same class; quotient reduction may
introduce edges absent at birth, so return the ORIGINAL chains. A family cap
must not truncate rank computation or hide budget exhaustion. Precision tests
retain represented integer/float32 thresholds rather than rounded comparisons.
No native extension, external data, or optional persistence library is needed.
"""

import json
from itertools import combinations

import numpy as np
import pytest

from open_deep_tda import structural_h1
from open_deep_tda.structural_h1 import H1BudgetExceeded, check_h1_witnesses
from open_deep_tda.structural_witnesses import select_h1_witnesses


SQUARE = [[0, 1], [0, 3], [1, 2], [2, 3]]
POINTS = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
LIMITS = ("max_vertices", "max_simplices", "max_reduction_operations",
          "max_reduction_entries")
FIELDS = {
    "cycles", "image_rank", "selected_count", "complete_basis", "n_vertices",
    "birth_radius", "survival_radius", "scope", "n_edges", "n_triangles",
    "n_simplices", "graph_cycle_count", "reduction_operations",
    "peak_reduction_entries", "reduction_words_per_vector",
}


def distances(points):
    points = np.asarray(points)
    return np.linalg.norm(points[:, None] - points[None, :], axis=-1)


def graph_matrix(n, edges, *, edge_value=1., missing=2.):
    D = np.full((n, n), missing)
    np.fill_diagonal(D, 0)
    for i, j in edges:
        D[i, j] = D[j, i] = edge_value
    return D


def separate_squares(count):
    cycles = [[[i + 4 * k, j + 4 * k] for i, j in SQUARE] for k in range(count)]
    return graph_matrix(4 * count, [e for c in cycles for e in c]), cycles


def annulus():
    D, cycles = separate_squares(2)
    for i in range(4):
        for j in (i + 4, (i + 1) % 4 + 4):
            D[i, j] = D[j, i] = 1.2
    return D, cycles


def dense_rref(matrix):
    """Row elimination on uint8 arrays; independent of production bitsets."""
    A = np.array(matrix, dtype=np.uint8, copy=True)
    pivots = []
    for col in range(A.shape[1]):
        row = len(pivots)
        candidates = np.flatnonzero(A[row:, col])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        A[[row, pivot]] = A[[pivot, row]]
        for other in range(A.shape[0]):
            if other != row and A[other, col]:
                A[other] ^= A[row]
        pivots.append(col)
        if len(pivots) == A.shape[0]:
            break
    return A, pivots


def dense_rank(matrix):
    return len(dense_rref(matrix)[1])


def dense_nullspace(matrix):
    """Kernel from RREF free variables, NOT a spanning-forest implementation."""
    reduced, pivots = dense_rref(matrix)
    free = [j for j in range(reduced.shape[1]) if j not in pivots]
    basis = np.zeros((reduced.shape[1], len(free)), dtype=np.uint8)
    for k, col in enumerate(free):
        basis[col, k] = 1
        for row, pivot in enumerate(pivots):
            basis[pivot, k] = reduced[row, col]
    return basis


def dense_oracle(D, birth, survival, cycles=()):
    """Compute image H1(K_birth)->H1(K_survival) from dense chain matrices.

    im H1 has dimension rank([boundary_2(survival), ker boundary_1(birth)])
    minus rank(boundary_2(survival)); no production selector/helpers are used.
    """
    D = np.asarray(D).astype(object)  # Python scalar comparisons retain precision.
    n = len(D)
    edges = list(combinations(range(n), 2))
    edge_ids = {e: i for i, e in enumerate(edges)}
    birth_ids = [k for k, (i, j) in enumerate(edges) if D[i, j] <= birth]
    incidence = np.zeros((n, len(birth_ids)), dtype=np.uint8)
    for col, k in enumerate(birth_ids):
        incidence[list(edges[k]), col] = 1
    kernel = dense_nullspace(incidence)
    graph_cycles = np.zeros((len(edges), kernel.shape[1]), dtype=np.uint8)
    graph_cycles[birth_ids, :] = kernel
    triangles = [list(combinations(t, 2)) for t in combinations(range(n), 3)
                 if all(D[i, j] <= survival for i, j in combinations(t, 2))]

    def columns(chains):
        A = np.zeros((len(edges), len(chains)), dtype=np.uint8)
        for col, chain in enumerate(chains):
            for i, j in chain:
                A[edge_ids[tuple(sorted((i, j)))], col] ^= 1
        return A

    boundary = columns(triangles)
    base_rank = dense_rank(boundary)
    image_rank = dense_rank(np.concatenate((boundary, graph_cycles), axis=1)) - base_rank
    selected_rank = dense_rank(np.concatenate((boundary, columns(cycles)), axis=1)) - base_rank
    return {
        "image_rank": image_rank,
        "selected_rank": selected_rank,
        "graph_cycle_count": kernel.shape[1],
        "n_edges": sum(D[i, j] <= survival for i, j in edges),
        "n_triangles": len(triangles),
    }


def assert_certificate(result, D, birth, survival, cap=8):
    assert type(result) is dict
    assert set(result) == FIELDS
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["n_vertices"] == len(D)
    assert result["birth_radius"] == birth
    assert result["survival_radius"] == survival
    assert isinstance(result["scope"], str)
    assert "complete supplied vertex domain" in result["scope"]
    assert "no assertion about omitted points" in result["scope"]
    for key in FIELDS - {"cycles", "complete_basis", "birth_radius", "survival_radius", "scope"}:
        assert type(result[key]) is int and result[key] >= 0
    assert type(result["cycles"]) is list
    for chain in result["cycles"]:
        assert type(chain) is list and chain
        assert chain == sorted(chain)
        assert len({tuple(e) for e in chain}) == len(chain)
        parity = np.zeros(len(D), dtype=np.uint8)
        for edge in chain:
            assert type(edge) is list and len(edge) == 2
            i, j = edge
            assert type(i) is type(j) is int
            assert 0 <= i < j < len(D)
            assert np.asarray(D).astype(object)[i, j] <= birth
            parity[i] ^= 1
            parity[j] ^= 1
        assert not parity.any()
    oracle = dense_oracle(D, birth, survival, result["cycles"])
    for key in ("image_rank", "graph_cycle_count", "n_edges", "n_triangles"):
        assert result[key] == oracle[key]
    assert result["selected_count"] == len(result["cycles"]) == min(cap, oracle["image_rank"])
    assert oracle["selected_rank"] == result["selected_count"]
    assert result["complete_basis"] is (result["selected_count"] == oracle["image_rank"])
    assert result["n_simplices"] == len(D) + oracle["n_edges"] + oracle["n_triangles"]
    words = max(1, (len(D) * (len(D) - 1) // 2 + 63) // 64)
    assert result["reduction_words_per_vector"] == words
    assert result["peak_reduction_entries"] >= 3 * words
    assert result["peak_reduction_entries"] % words == 0
    assert result["reduction_operations"] % words == 0
    # The verifier intentionally rejects empty source contracts, unlike selection.
    if result["cycles"]:
        checked = check_h1_witnesses(D, D, result["cycles"], birth, survival)
        assert checked.accepted
        assert checked.surviving_rank == result["selected_count"]


def test_square_identity_json_and_single_survival_complex():
    D = distances(POINTS)
    before = D.copy()
    result = select_h1_witnesses(D, 1, 1.2)
    assert result["cycles"] == [SQUARE]
    assert result["image_rank"] == result["graph_cycle_count"] == 1
    assert result["n_simplices"] == 8  # Not the verifier's combined source/target 16.
    assert result["peak_reduction_entries"] == 4  # Three scratch plus one class.
    # W=1: initialization 1 + four edge-bit insertions 8 + rank inspection 1.
    assert result["reduction_operations"] == 10
    assert_certificate(result, D, 1, 1.2)
    assert result == select_h1_witnesses(D.tolist(), np.int64(1), np.float64(1.2))
    np.testing.assert_array_equal(D, before)


def test_two_independent_squares_and_caps_are_prefixes():
    D, cycles = separate_squares(2)
    full = select_h1_witnesses(D, 1, 1.2)
    assert full["cycles"] == cycles
    assert full["image_rank"] == 2
    for cap in (0, 1, 2, 3, np.int64(1)):
        result = select_h1_witnesses(D, 1, 1.2, max_cycles=cap)
        assert_certificate(result, D, 1, 1.2, cap)
        assert result["cycles"] == cycles[:cap]
        for key in FIELDS - {"cycles", "selected_count", "complete_basis"}:
            assert result[key] == full[key], key


def test_output_cap_counts_independent_classes_not_examined_candidates():
    D, cycles = separate_squares(2)
    D[0, 2] = D[2, 0] = 1.2  # Fill the first candidate, not the second.
    result = select_h1_witnesses(D, 1, 1.2, max_cycles=1)
    assert result["cycles"] == cycles[1:]
    assert result["image_rank"] == 1 and result["graph_cycle_count"] == 2
    assert_certificate(result, D, 1, 1.2, 1)


def test_default_family_cap_is_eight_not_a_rank_cap():
    D, cycles = separate_squares(9)
    result = select_h1_witnesses(D, 1, 1.2)
    assert result["cycles"] == cycles[:8]
    assert result["image_rank"] == result["graph_cycle_count"] == 9
    assert not result["complete_basis"]
    full = select_h1_witnesses(D, 1, 1.2, max_cycles=9)
    assert full["cycles"] == cycles
    assert full["complete_basis"]
    assert full["reduction_operations"] == result["reduction_operations"]
    assert full["peak_reduction_entries"] == result["peak_reduction_entries"]


def test_center_vertex_fills_cycle_using_entire_supplied_domain():
    D = graph_matrix(5, SQUARE)
    D[4, :4] = D[:4, 4] = 1.2
    result = select_h1_witnesses(D, 1, 1.2)
    assert_certificate(result, D, 1, 1.2)
    assert result["graph_cycle_count"] == 1
    assert result["image_rank"] == 0
    assert result["cycles"] == [] and result["complete_basis"]
    assert result["n_triangles"] == 4
    restricted = select_h1_witnesses(D[:4, :4], 1, 1.2)
    assert restricted["cycles"] == [SQUARE]


@pytest.mark.parametrize("birth,survival,rank,graph_rank", [
    (0.9, 1.2, 0, 0), (1., 1., 1, 1), (1., 1.2, 1, 1),
    (1., 2., 0, 1), (2., 2., 0, 3),
])
def test_birth_presence_and_inclusive_death(birth, survival, rank, graph_rank):
    D = graph_matrix(4, SQUARE)
    result = select_h1_witnesses(D, birth, survival)
    assert_certificate(result, D, birth, survival)
    assert result["image_rank"] == rank
    assert result["graph_cycle_count"] == graph_rank


def test_annulus_greedily_keeps_first_class_not_both_nonzero_cycles():
    """Leading-only triangle elimination incorrectly reports rank two here."""
    D, cycles = annulus()
    result = select_h1_witnesses(D, 1, 1.2)
    assert_certificate(result, D, 1, 1.2)
    assert result["graph_cycle_count"] == 2
    assert result["image_rank"] == 1
    assert result["cycles"] == cycles[:1]
    for chain in cycles:
        assert dense_oracle(D, 1, 1.2, [chain])["selected_rank"] == 1


def test_returns_original_birth_chain_not_later_edge_quotient():
    """A survival triangle replaces edge (1,2) by absent-at-birth (0,1),(0,2).

    The square's free leading edge (3,4) does not excuse skipping the lower
    triangle pivot; either way the returned witness must be the original square.
    """
    square = [[i + 1, j + 1] for i, j in SQUARE]
    D = graph_matrix(5, square)
    D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = 1.2
    result = select_h1_witnesses(D, 1, 1.2)
    assert_certificate(result, D, 1, 1.2)
    assert result["n_triangles"] == 1
    assert result["cycles"] == [square]


@pytest.mark.parametrize("weighted", [False, True])
def test_lexicographic_forest_not_distance_sorted_forest(weighted):
    edges = [[i, j] for i in (0, 1) for j in (2, 3, 4)]  # Triangle-free K2,3.
    D = graph_matrix(5, edges)
    if weighted:
        for (i, j), weight in zip(edges, [1., .4, .5, .6, .7, .8]):
            D[i, j] = D[j, i] = weight
    expected = [[[0, 2], [0, 3], [1, 2], [1, 3]],
                [[0, 2], [0, 4], [1, 2], [1, 4]]]
    result = select_h1_witnesses(D, 1, 1)
    assert result["cycles"] == expected
    assert_certificate(result, D, 1, 1)
    assert select_h1_witnesses(D, 1, 1, max_cycles=1)["cycles"] == expected[:1]
    assert select_h1_witnesses(D.copy(), 1, 1) == result


def test_duplicate_points_ties_and_zero_radius():
    D = distances(np.vstack([POINTS, POINTS[0]]))
    assert D[0, 4] == 0
    result = select_h1_witnesses(D, 1, 1)
    assert_certificate(result, D, 1, 1)
    assert result["cycles"] == [SQUARE]
    # A zero-distance clique has graph cycles, but all are triangle boundaries.
    D = np.zeros((4, 4))
    result = select_h1_witnesses(D, 0, 0)
    assert_certificate(result, D, 0, 0)
    assert result["graph_cycle_count"] == 3 and result["image_rank"] == 0
    # Triangle inequality is not required: a zero-weight square need not fill.
    D = graph_matrix(4, SQUARE, edge_value=0.)
    result = select_h1_witnesses(D, 0, 0)
    assert result["cycles"] == [SQUARE]
    assert_certificate(result, D, 0, 0)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_empty_and_acyclic_domains_still_reserve_three_scratch_slots(n):
    D = graph_matrix(n, [])
    result = select_h1_witnesses(D, 0, 0, max_vertices=n, max_simplices=n,
                                 max_reduction_operations=0, max_reduction_entries=3)
    assert_certificate(result, D, 0, 0)
    assert result["complete_basis"] and result["cycles"] == []
    assert result["reduction_operations"] == 0
    assert result["peak_reduction_entries"] == 3
    for entries in (0, 1, 2):
        with pytest.raises(H1BudgetExceeded, match="max_reduction_entries"):
            select_h1_witnesses(D, 0, 0, max_cycles=0, max_reduction_entries=entries)


def test_tree_has_no_graph_cycles():
    D = graph_matrix(5, [[0, 1], [1, 2], [1, 3], [3, 4]])
    result = select_h1_witnesses(D, 1, 1.2)
    assert_certificate(result, D, 1, 1.2)
    assert result["graph_cycle_count"] == result["image_rank"] == 0


@pytest.mark.parametrize("birth,graph_rank,operations", [(0, 0, 8), (1, 1, 19)])
def test_filled_triangle_accounting_even_without_returned_cycles(birth, graph_rank, operations):
    """Boundary work is required even without birth cycles or with a zero cap.

    W=1: triangle initialization/three bits/insertion cost 1+6+1=8.
    A birth graph cycle additionally costs 1+6+2 (pivot test)+1 (XOR)+1
    (zero rank inspection)=11; it adds no stored class or returned witness.
    """
    D = graph_matrix(3, [[0, 1], [0, 2], [1, 2]])
    limits = dict(max_vertices=3, max_simplices=7,
                  max_reduction_operations=operations, max_reduction_entries=4)
    result = select_h1_witnesses(D, birth, 1, max_cycles=0, **limits)
    assert_certificate(result, D, birth, 1, 0)
    assert result["graph_cycle_count"] == graph_rank
    assert result["reduction_operations"] == operations
    assert result["peak_reduction_entries"] == 4
    for key, limit in limits.items():
        with pytest.raises(H1BudgetExceeded, match=key):
            select_h1_witnesses(D, birth, 1, max_cycles=0,
                                **dict(limits, **{key: limit - 1}))


@pytest.mark.parametrize("cap", [0, 1, 8])
def test_exact_budget_boundaries_with_multword_triangles(cap):
    D, _ = annulus()
    padded = graph_matrix(13, [])
    padded[5:, 5:] = D  # Active high-ID edges cross the 64-bit word boundary.
    result = select_h1_witnesses(padded, 1, 1.2, max_cycles=cap)
    assert_certificate(result, padded, 1, 1.2, cap)
    assert result["reduction_words_per_vector"] == 2
    assert result["n_edges"] == 16 and result["n_triangles"] == 8
    assert result["n_simplices"] == 37
    limits = dict(max_vertices=13, max_simplices=result["n_simplices"],
                  max_reduction_operations=result["reduction_operations"],
                  max_reduction_entries=result["peak_reduction_entries"])
    assert select_h1_witnesses(padded, 1, 1.2, max_cycles=cap, **limits) == result
    for key, limit in limits.items():
        below = dict(limits, **{key: limit - 1})
        with pytest.raises(H1BudgetExceeded, match=key):
            select_h1_witnesses(padded, 1, 1.2, max_cycles=cap, **below)


@pytest.mark.parametrize("cap", [0, 1])
def test_cap_does_not_hide_later_independent_class_budget_exhaustion(cap):
    D, _ = separate_squares(2)
    result = select_h1_witnesses(D, 1, 1.2)
    assert result["peak_reduction_entries"] == 5
    for key, bound in (("max_reduction_entries", 4),
                       ("max_reduction_operations", result["reduction_operations"] - 1)):
        with pytest.raises(H1BudgetExceeded, match=key):
            select_h1_witnesses(D, 1, 1.2, max_cycles=cap, **{key: bound})


def test_single_complex_simplex_preflight_precedes_any_reduction(monkeypatch):
    D = np.zeros((8, 8))
    count = 8 + 28 + 56
    full = select_h1_witnesses(D, 0, 0, max_simplices=count)
    assert full["n_simplices"] == count

    def forbidden(*args, **kwargs):
        pytest.fail("reduction began before survival-simplex budget preflight")

    monkeypatch.setattr(structural_h1._ReductionBudget, "vector", forbidden)
    monkeypatch.setattr(structural_h1._ReductionBudget, "reserve", forbidden)
    with pytest.raises(H1BudgetExceeded, match="max_simplices"):
        select_h1_witnesses(D, 0, 0, max_cycles=0, max_simplices=count - 1)


def test_default_vertex_limit_and_hard_ceiling_even_with_zero_cap():
    D = graph_matrix(65, [])
    with pytest.raises(H1BudgetExceeded, match="max_vertices"):
        select_h1_witnesses(D, 0, 0, max_cycles=0)
    result = select_h1_witnesses(D, 0, 0, max_vertices=65)
    assert result["n_vertices"] == 65 and result["image_rank"] == 0
    D = graph_matrix(128, SQUARE)
    result = select_h1_witnesses(D, 1, 1.2, max_vertices=128)
    assert result["cycles"] == [SQUARE]
    assert result["reduction_words_per_vector"] == 127
    assert result["peak_reduction_entries"] == 4 * 127
    D = graph_matrix(129, [])
    with pytest.raises(H1BudgetExceeded, match="128"):
        select_h1_witnesses(D, 0, 0, max_vertices=1000, max_cycles=0)


@pytest.mark.parametrize("key", LIMITS)
def test_zero_budget_is_a_real_limit(key):
    with pytest.raises(H1BudgetExceeded, match=key):
        select_h1_witnesses(graph_matrix(4, SQUARE), 1, 1.2, **{key: 0})


@pytest.mark.parametrize("key", ("max_cycles",) + LIMITS)
@pytest.mark.parametrize("value", [-1, True, np.bool_(False), 1.5, np.inf, np.nan, None, "8"])
def test_malformed_integer_limits(key, value):
    with pytest.raises(ValueError, match=key):
        select_h1_witnesses(graph_matrix(4, SQUARE), 1, 1.2, **{key: value})


@pytest.mark.parametrize("bad", [
    None, [], [0], np.ones((4, 3)), np.zeros((2, 2, 2)), [[0j]], [["0"]], [[False]],
    np.array([[0]], dtype=object), [[np.nan]], [[np.inf]], [[1]], [[0, -1], [-1, 0]],
    [[0, 1], [2, 0]], [[0, 1], [1 + 1e-12, 0]], [[0], [1, 2]],
])
def test_malformed_matrices_rejected_even_when_family_cap_is_zero(bad):
    with pytest.raises(ValueError):
        select_h1_witnesses(bad, 1, 1.2, max_cycles=0)


@pytest.mark.parametrize("birth,survival", [
    (-1, 1), (1, -1), (1.2, 1), (np.nan, 2), (1, np.inf), (np.inf, np.inf),
    (1, np.nan), (True, 1), (0, np.bool_(False)), ("1", 2), (1, 1j), (0, object()),
])
def test_malformed_radii(birth, survival):
    with pytest.raises(ValueError):
        select_h1_witnesses(graph_matrix(4, SQUARE), birth, survival)


def test_precision_safe_large_integer_and_mixed_float_thresholds():
    side = 2**53 + 1
    D = np.full((4, 4), side + 1, dtype=np.int64)
    np.fill_diagonal(D, 0)
    for i, j in SQUARE:
        D[i, j] = D[j, i] = side
    for radius in (side - 1, float(side - 1)):
        result = select_h1_witnesses(D, radius, radius)
        assert result["cycles"] == [] and result["graph_cycle_count"] == 0
        assert_certificate(result, D, radius, radius)
    result = select_h1_witnesses(D, np.int64(side), side)
    assert result["cycles"] == [SQUARE]
    assert type(result["birth_radius"]) is int and result["birth_radius"] == side
    assert_certificate(result, D, side, side)
    D32 = graph_matrix(4, SQUARE).astype(np.float32)
    diagonal = np.nextafter(np.float32(1), np.float32(2))
    D32[0, 2] = D32[2, 0] = D32[1, 3] = D32[3, 1] = diagonal
    radius = 1.0000000894069672
    assert float(diagonal) > radius
    for D in (D32, D32.astype(np.float64)):
        result = select_h1_witnesses(D, 1, radius)
        assert result["cycles"] == [SQUARE]
        assert_certificate(result, D, 1, radius)
        assert select_h1_witnesses(D, 1, float(diagonal))["image_rank"] == 0
    huge = 10**1000  # Finite integers need not fit float64.
    result = select_h1_witnesses(D32, huge, huge)
    assert result["cycles"] == [] and result["image_rank"] == 0
    assert result["birth_radius"] == result["survival_radius"] == huge
    assert_certificate(result, D32, huge, huge)


def test_wider_float_precision_is_rejected_instead_of_rounded():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("platform has no floating type wider than float64")
    D = distances(POINTS)
    with pytest.raises(ValueError, match="precision"):
        select_h1_witnesses(D.astype(np.longdouble), 1, 1.2)
    for birth, survival in ((np.longdouble(1), 1.2), (1, np.longdouble(1.2))):
        with pytest.raises(ValueError, match="precision"):
            select_h1_witnesses(D, birth, survival)


def test_random_small_matrices_against_dense_nullspace_oracle():
    rng = np.random.default_rng(271828)
    nonzero = merged_or_filled = 0
    for _ in range(100):
        n = int(rng.integers(0, 9))
        upper = np.triu(rng.choice([0., 1., 1.2, 2.], size=(n, n),
                                  p=[.05, .4, .1, .45]), 1)
        D = upper + upper.T
        cap = int(rng.integers(0, 4))
        result = select_h1_witnesses(D, 1, 1.2, max_cycles=cap)
        assert_certificate(result, D, 1, 1.2, cap)
        full = select_h1_witnesses(D, 1, 1.2, max_cycles=n * n)
        assert_certificate(full, D, 1, 1.2, n * n)
        assert result["cycles"] == full["cycles"][:cap]
        assert result["reduction_operations"] == full["reduction_operations"]
        assert result["peak_reduction_entries"] == full["peak_reduction_entries"]
        nonzero += full["image_rank"] > 0
        merged_or_filled += full["graph_cycle_count"] > full["image_rank"]
        # Changing edge-bit positions preserves ranks, not canonical chains.
        permutation = rng.permutation(n)
        permuted = D[np.ix_(permutation, permutation)]
        relabeled = select_h1_witnesses(permuted, 1, 1.2, max_cycles=cap)
        assert_certificate(relabeled, permuted, 1, 1.2, cap)
        assert relabeled["image_rank"] == result["image_rank"]
        assert relabeled["graph_cycle_count"] == result["graph_cycle_count"]
    assert nonzero >= 10
    assert merged_or_filled >= 20
