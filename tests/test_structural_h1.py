"""Standalone witness verifier regressions; the oracle uses dense uint8 GF2.

Reproducible pitfalls retained here: equal critical lengths or barcodes do not
preserve row identities; off-witness vertices and relations between surviving
classes must be included. No optimization/convergence assertion is involved.
"""

from itertools import combinations

import numpy as np
import pytest

from open_deep_tda.structural_h1 import H1BudgetExceeded, check_h1_witnesses


SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]
POINTS = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


def distances(points):
    points = np.asarray(points)
    return np.linalg.norm(points[:, None] - points[None, :], axis=-1)


def graph_matrix(n, edges):
    D = np.full((n, n), 2.)
    np.fill_diagonal(D, 0.)
    for i, j in edges:
        D[i, j] = D[j, i] = 1.
    return D


def dense_rank(columns, dimension):
    """Independent row elimination on an explicit GF2 matrix, not bitsets."""
    if not columns:
        return 0
    A = np.array(columns, dtype=np.uint8).T.copy()
    rank = 0
    for column in range(A.shape[1]):
        candidates = np.flatnonzero(A[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + candidates[0]
        A[[rank, pivot]] = A[[pivot, rank]]
        for row in range(dimension):
            if row != rank and A[row, column]:
                A[row] ^= A[rank]
        rank += 1
        if rank == dimension:
            break
    return rank


def oracle(D, cycles, birth, survival):
    """rank([boundary_2, cycles])-rank(boundary_2), using all vertices."""
    edges = list(combinations(range(len(D)), 2))
    indices = {edge: i for i, edge in enumerate(edges)}

    def vector(chain):
        v = np.zeros(len(edges), dtype=np.uint8)
        for edge in chain:
            v[indices[tuple(sorted(edge))]] ^= 1
        return v

    triangles = []
    for triple in combinations(range(len(D)), 3):
        boundary = list(combinations(triple, 2))
        if all(D[i, j] <= survival for i, j in boundary):
            triangles.append(vector(boundary))
    base_rank = dense_rank(triangles, len(edges))
    present, survives, eligible = [], [], []
    for cycle in cycles:
        v = vector(cycle)
        ok = all(D[i, j] <= birth for (i, j), bit in zip(edges, v) if bit)
        present.append(ok)
        survives.append(ok and dense_rank(triangles + [v], len(edges)) > base_rank)
        if ok:
            eligible.append(v)
    return present, survives, dense_rank(triangles + eligible, len(edges)) - base_rank


def fundamental_cycles(D, birth):
    """Build closed graph chains using paths in a spanning forest."""
    n = len(D)
    forest = [[] for _ in range(n)]
    cycles = []
    for a, b in combinations(range(n), 2):
        if D[a, b] > birth:
            continue
        paths = {a: []}
        queue = [a]
        for vertex in queue:
            for neighbor in forest[vertex]:
                if neighbor not in paths:
                    paths[neighbor] = paths[vertex] + [(vertex, neighbor)]
                    queue.append(neighbor)
        if b in paths:
            cycles.append(paths[b] + [(b, a)])
        else:
            forest[a].append(b)
            forest[b].append(a)
    return cycles


def assert_oracle(result, D, cycles, birth=1., survival=1.2):
    present, survives, rank = oracle(D, cycles, birth, survival)
    assert [w.edges_present for w in result.witnesses] == present
    assert [w.survives for w in result.witnesses] == survives
    assert result.surviving_rank == rank
    assert result.all_classes_independent == (rank == len(cycles))
    assert result.accepted == result.all_classes_independent


def test_square_identity_interval_and_inclusive_thresholds():
    D = distances(POINTS)
    before = D.copy()
    result = check_h1_witnesses(D, D, [SQUARE], 1, 1.2)
    assert result.accepted
    assert result.n_simplices == 16  # 2 * (4 vertices + 4 edges)
    # W=1. Per complex: vector init 1 + four bit insertions 8 + nonzero
    # test 1 + rank pivot inspection 1 = 11. Storage is 3 scratch + 1 basis;
    # source storage must be released before processing the target.
    assert result.reduction_operations == 22
    assert result.peak_reduction_entries == 4
    assert_oracle(result, D, [SQUARE])
    assert check_h1_witnesses(D, D, [SQUARE], 1, 1).accepted
    np.testing.assert_array_equal(D, before)
    # At the exact death threshold the source no longer satisfies the contract.
    with pytest.raises(ValueError, match="invalid_source_contract"):
        check_h1_witnesses(D, D, [SQUARE], 1, D[0, 2])


def test_precision_safe_integer_and_mixed_float_thresholds():
    side = 2**53 + 1
    D = np.full((4, 4), side + 1, dtype=np.int64)
    np.fill_diagonal(D, 0)
    for i, j in SQUARE:
        D[i, j] = D[j, i] = side
    with pytest.raises(ValueError, match="invalid_source_contract.*missing"):
        check_h1_witnesses(D, D, [SQUARE], side - 1, side - 1)
    with pytest.raises(ValueError, match="invalid_source_contract.*missing"):
        check_h1_witnesses(D, D, [SQUARE], float(side - 1), float(side - 1))
    # The odd integer radius must not itself be rounded to an even float.
    assert check_h1_witnesses(D, D, [SQUARE], np.int64(side), side).accepted

    D32 = graph_matrix(4, SQUARE).astype(np.float32)
    D32[0, 2] = D32[2, 0] = D32[1, 3] = D32[3, 1] = np.nextafter(
        np.float32(1), np.float32(2))
    D64 = D32.astype(np.float64)
    radius = 1.0000000894069672  # Strictly below both identical represented diagonals.
    assert float(D32[0, 2]) > radius
    for source, target in [(D64, D32), (D32, D64)]:
        assert check_h1_witnesses(source, target, [SQUARE], 1, radius).accepted
    # Finite Python integer radii need not fit float64 at all. Validation must
    # reach the (filled) source contract, rather than overflow on the radius.
    with pytest.raises(ValueError, match="invalid_source_contract.*filled"):
        check_h1_witnesses(D64, D64, [SQUARE], 10**1000, 10**1000)


def test_wider_float_precision_is_rejected_not_silently_rounded():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("platform has no floating type wider than float64")
    D = distances(POINTS)
    with pytest.raises(ValueError, match="precision"):
        check_h1_witnesses(D.astype(np.longdouble), D, [SQUARE], 1, 1.2)
    with pytest.raises(ValueError, match="precision"):
        check_h1_witnesses(D, D, [SQUARE], 1, np.longdouble(1.2))


def test_equal_source_critical_lengths_noncollinear_missing_loop_rejected():
    source = distances(POINTS)
    target_points = np.array([[0., 0.], [0., 3.], [1., 1.], [2., 1.]])
    assert np.linalg.matrix_rank(target_points - target_points[0]) == 2
    target = distances(target_points)
    # Square, lexicographic tie ordering: positive H1 birth edge (2,3),
    # death critical edge (0,2). Matching just these lengths misses the loop.
    assert source[2, 3] == target[2, 3] == 1
    assert source[0, 2] == target[0, 2] == np.sqrt(2)
    result = check_h1_witnesses(source, target, [SQUARE], 1, 1.2)
    assert not result.accepted
    assert not result.witnesses[0].edges_present
    assert_oracle(result, target, [SQUARE])
    # This target has no positive H1 at any of its critical radii.
    for radius in np.unique(target):
        chains = fundamental_cycles(target, radius)
        assert oracle(target, chains, radius, radius)[2] == 0


def test_identical_barcode_wrong_sample_permutation_rejects_edges():
    source = distances(POINTS)
    permutation = [0, 2, 1, 3]
    target = source[np.ix_(permutation, permutation)]
    # Exact simultaneous row/column permutation is a filtered isomorphism,
    # hence has exactly the same barcode, but not the same sample-ID witness.
    inverse = np.argsort(permutation)
    np.testing.assert_array_equal(target[np.ix_(inverse, inverse)], source)
    result = check_h1_witnesses(source, target, [SQUARE], 1, 1.2)
    assert not result.witnesses[0].edges_present
    assert not result.accepted


def test_external_center_fills_target_cycle_not_source():
    source = distances(np.vstack([POINTS, [10., 10.]]))
    target = distances(np.vstack([POINTS, [.5, .5]]))
    result = check_h1_witnesses(source, target, [SQUARE], 1, 1.2)
    assert result.witnesses[0].edges_present
    assert not result.witnesses[0].survives
    assert result.surviving_rank == 0
    assert not result.accepted
    assert_oracle(result, target, [SQUARE])
    assert check_h1_witnesses(source[:4, :4], target[:4, :4], [SQUARE], 1, 1.2).accepted


def annulus_fixture():
    cycles = [SQUARE, [(i + 4, j + 4) for i, j in SQUARE]]
    source = graph_matrix(8, cycles[0] + cycles[1])
    target = source.copy()
    for i in range(4):
        for j in (i + 4, (i + 1) % 4 + 4):
            target[i, j] = target[j, i] = 1.2
    return source, target, cycles


def test_individually_surviving_classes_merge_in_triangulated_annulus():
    source, target, cycles = annulus_fixture()
    result = check_h1_witnesses(source, target, cycles, 1, 1.2)
    assert all(w.edges_present and w.survives for w in result.witnesses)
    assert result.surviving_rank == 1
    assert not result.all_classes_independent
    assert_oracle(result, target, cycles)
    with pytest.raises(ValueError, match="invalid_source_contract"):
        check_h1_witnesses(target, target, cycles, 1, 1.2)


def test_parity_orientation_numpy_ids_and_canceled_missing_edges():
    D = distances(POINTS)
    chain = [(np.int64(j), np.int64(i)) for i, j in SQUARE]
    chain += [(0, 2), (2, 0), (0, 1), (1, 0)]
    before = list(chain)
    result = check_h1_witnesses(D, D, [chain], 1, 1.2)
    assert result.accepted
    assert chain == before
    assert result == check_h1_witnesses(D.tolist(), D.tolist(), [SQUARE], 1, 1.2)


def test_closed_disconnected_chain_allowed():
    source, _, cycles = annulus_fixture()
    assert check_h1_witnesses(source, source, [cycles[0] + cycles[1]], 1, 1.2).accepted


def test_missing_at_birth_but_present_at_survival_excluded_from_rank():
    source = distances(POINTS)
    target = source.copy()
    target[0, 1] = target[1, 0] = 1.1
    result = check_h1_witnesses(source, target, [SQUARE], 1, 1.2)
    assert not result.witnesses[0].edges_present
    assert not result.witnesses[0].survives
    assert result.surviving_rank == 0


def test_joint_rank_retains_valid_members_when_another_has_missing_edges():
    source, _, cycles = annulus_fixture()
    target = source.copy()
    target[0, 1] = target[1, 0] = 2.
    result = check_h1_witnesses(source, target, cycles, 1, 1.2)
    assert [w.edges_present for w in result.witnesses] == [False, True]
    assert [w.survives for w in result.witnesses] == [False, True]
    assert result.surviving_rank == 1
    assert not result.accepted
    assert_oracle(result, target, cycles)


def test_random_small_matrices_against_independent_dense_gf2_oracle():
    rng = np.random.default_rng(631)
    valid_families = 0
    for _ in range(100):
        n = int(rng.integers(4, 9))

        def random_matrix():
            upper = np.triu(rng.choice([1., 1.2, 2.], size=(n, n), p=[.45, .1, .45]), 1)
            return upper + upper.T

        source, target = random_matrix(), random_matrix()
        candidates = fundamental_cycles(source, 1.)
        independent = []
        for chain in candidates:
            _, _, rank = oracle(source, independent + [chain], 1., 1.2)
            if rank > len(independent):
                independent.append(chain)
        if not independent:
            if candidates:
                with pytest.raises(ValueError, match="invalid_source_contract"):
                    check_h1_witnesses(source, target, candidates, 1, 1.2)
            continue
        valid_families += 1
        for D in (source, target):
            result = check_h1_witnesses(source, D, independent, 1, 1.2)
            assert_oracle(result, D, independent)
        # Change edge-bit ordering without changing the underlying complexes.
        perm = rng.permutation(n)
        inv = np.argsort(perm)
        relabeled = [[(int(inv[i]), int(inv[j])) for i, j in c] for c in independent]
        result = check_h1_witnesses(source[np.ix_(perm, perm)], target[np.ix_(perm, perm)],
                                    relabeled, 1, 1.2)
        assert_oracle(result, target, independent)
    assert valid_families >= 15


@pytest.mark.parametrize("cycles", [[], [[]], [[(0, 1), (1, 0)]], [[(0, 1)]],
                                     [SQUARE, SQUARE]])
def test_invalid_source_chains_never_report_target_failure(cycles):
    D = distances(POINTS)
    with pytest.raises(ValueError, match="invalid_source_contract"):
        check_h1_witnesses(D, D, cycles, 1, 1.2)


def test_invalid_source_missing_edges_or_filled_even_if_target_is_valid():
    D = distances(POINTS)
    source = D.copy()
    source[0, 1] = source[1, 0] = 1.1
    with pytest.raises(ValueError, match="invalid_source_contract.*missing"):
        check_h1_witnesses(source, D, [SQUARE], 1, 1.2)
    with pytest.raises(ValueError, match="invalid_source_contract.*filled"):
        check_h1_witnesses(np.zeros((4, 4)), D, [SQUARE], 1, 1.2)


@pytest.mark.parametrize("edge", [(0, 0), (-1, 2), (0, 4), (0., 1), (False, 1),
                                  (0, 1, 2), (0,), None, 2, ("0", 1)])
def test_invalid_edges_rejected_even_if_repeated(edge):
    D = distances(POINTS)
    with pytest.raises(ValueError):
        check_h1_witnesses(D, D, [SQUARE + [edge, edge]], 1, 1.2)


@pytest.mark.parametrize("bad", [np.ones((4, 3)), np.zeros((3, 3)), [[0j]], [["0"]],
                                 [[np.nan]], [[np.inf]], [[1]],
                                 [[0, -1], [-1, 0]], [[0, 1], [2, 0]],
                                 [[0, 1], [1 + 1e-12, 0]], [[0], [1, 2]], None])
@pytest.mark.parametrize("which", ["source", "target"])
def test_invalid_matrices(bad, which):
    D = distances(POINTS)
    args = (bad, D) if which == "source" else (D, bad)
    with pytest.raises(ValueError):
        check_h1_witnesses(*args, [SQUARE], 1, 1.2)


@pytest.mark.parametrize("birth,survival", [(-1, 1), (1, -1), (1.2, 1), (np.nan, 2),
                                             (1, np.inf), (np.inf, np.inf), (1, np.nan),
                                             (True, 1), ("1", 2), (1, 1j),
                                             (0, object())])
def test_invalid_radii(birth, survival):
    D = distances(POINTS)
    with pytest.raises(ValueError):
        check_h1_witnesses(D, D, [SQUARE], birth, survival)


@pytest.mark.parametrize("key", ["max_vertices", "max_simplices", "max_reduction_operations",
                                 "max_reduction_entries"])
@pytest.mark.parametrize("value", [-1, True, 1.5, np.inf, None])
def test_invalid_budget_values(key, value):
    D = distances(POINTS)
    with pytest.raises(ValueError, match=key):
        check_h1_witnesses(D, D, [SQUARE], 1, 1.2, **{key: value})


@pytest.mark.parametrize("key", ["max_vertices", "max_simplices", "max_reduction_operations",
                                 "max_reduction_entries"])
def test_exhausted_budgets_raise_not_partial_success(key):
    D = distances(POINTS)
    with pytest.raises(H1BudgetExceeded, match=key):
        check_h1_witnesses(D, D, [SQUARE], 1, 1.2, **{key: 0})


def test_budget_exact_boundaries_with_triangle_and_multword_reduction():
    source, target, cycles = annulus_fixture()
    # 13 vertices -> 78 ambient edges -> two 64-bit words per vector.
    def pad(D):
        out = graph_matrix(13, [])
        out[:8, :8] = D
        return out
    source, target = pad(source), pad(target)
    # Move the annulus to IDs 5..12, putting active edges/triangles above bit 63.
    permutation = list(range(8, 13)) + list(range(8))
    source = source[np.ix_(permutation, permutation)]
    target = target[np.ix_(permutation, permutation)]
    cycles = [[(i + 5, j + 5) for i, j in cycle] for cycle in cycles]
    edge_ids = {edge: index for index, edge in enumerate(combinations(range(13), 2))}
    assert max(edge_ids[tuple(sorted(e))] for c in cycles for e in c) >= 64
    full = check_h1_witnesses(source, target, cycles, 1, 1.2)
    assert_oracle(full, target, cycles)
    assert full.peak_reduction_entries % 2 == 0
    limits = dict(max_vertices=13, max_simplices=full.n_simplices,
                  max_reduction_operations=full.reduction_operations,
                  max_reduction_entries=full.peak_reduction_entries)
    assert check_h1_witnesses(source, target, cycles, 1, 1.2, **limits) == full
    for key, value in limits.items():
        with pytest.raises(H1BudgetExceeded, match=key):
            check_h1_witnesses(source, target, cycles, 1, 1.2, **{key: value - 1})


def test_simplices_preflight_precedes_boundary_allocation(monkeypatch):
    import open_deep_tda.structural_h1 as module
    source = graph_matrix(8, SQUARE)
    target = np.zeros((8, 8))  # 8 + 28 + 56 target simplices
    def forbidden(*args, **kwargs):
        pytest.fail("reduction started before combined simplex-budget preflight")
    monkeypatch.setattr(module._ReductionBudget, "vector", forbidden)
    with pytest.raises(H1BudgetExceeded, match="max_simplices"):
        check_h1_witnesses(source, target, [SQUARE], 1, 1.2, max_simplices=103)


def test_hard_vertex_scope_and_zero_radius_duplicate_points():
    D = graph_matrix(128, SQUARE)
    result = check_h1_witnesses(D, D, [SQUARE], 1, 1.2)
    assert result.accepted
    assert result.peak_reduction_entries == 4 * 127  # one basis + three scratch vectors
    D = graph_matrix(129, SQUARE)
    with pytest.raises(H1BudgetExceeded, match="128"):
        check_h1_witnesses(D, D, [SQUARE], 1, 1.2, max_vertices=1000)
    D = graph_matrix(4, SQUARE)
    D[D == 1] = 0
    assert check_h1_witnesses(D, D, [SQUARE], 0, 0).accepted
    with pytest.raises(ValueError, match="invalid_source_contract"):
        check_h1_witnesses(np.zeros((4, 4)), D, [SQUARE], 0, 0)
