"""Constructive certificates checked independently with sets and dense GF2.

Verified regressions: matching critical lengths can lose same-ID birth edges;
external vertices can fill a square; duplicate target squares can each survive
while their XOR bounds. Missing-at-birth chains must never enter relations.
"""

from itertools import combinations
import json

import numpy as np
import pytest

from open_deep_tda.structural_h1 import H1BudgetExceeded, check_h1_witnesses
from open_deep_tda.structural_obstructions import find_h1_obstructions


SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]
POINTS = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


def distances(points):
    points = np.asarray(points)
    return np.linalg.norm(points[:, None] - points[None, :], axis=-1)


def graph_matrix(n, edges):
    result = np.full((n, n), 2.)
    np.fill_diagonal(result, 0.)
    for i, j in edges:
        result[i, j] = result[j, i] = 1.
    return result


def parity(edges):
    result = set()
    for edge in edges:
        result.symmetric_difference_update([tuple(sorted(edge))])
    return result


def dense_rank(columns, dimension):
    """Independent explicit uint8 row elimination, without integer bitsets."""
    if not columns:
        return 0
    matrix = np.array(columns, dtype=np.uint8).T.copy()
    rank = 0
    for column in range(matrix.shape[1]):
        candidates = np.flatnonzero(matrix[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + candidates[0]
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(dimension):
            if row != rank and matrix[row, column]:
                matrix[row] ^= matrix[rank]
        rank += 1
        if rank == dimension:
            break
    return rank


def oracle(D, cycles, birth=1., survival=1.2):
    edges = list(combinations(range(len(D)), 2))

    def vector(chain):
        support = parity(chain)
        return np.array([edge in support for edge in edges], dtype=np.uint8)

    boundaries = [vector(combinations(triangle, 2))
                  for triangle in combinations(range(len(D)), 3)
                  if all(D[i, j] <= survival for i, j in combinations(triangle, 2))]
    base_rank = dense_rank(boundaries, len(edges))
    present, survives, eligible = [], [], []
    for chain in cycles:
        ok = all(D[i, j] <= birth for i, j in parity(chain))
        present.append(ok)
        v = vector(chain)
        survives.append(ok and dense_rank(boundaries + [v], len(edges)) > base_rank)
        if ok:
            eligible.append(v)
    return present, survives, dense_rank(boundaries + eligible, len(edges)) - base_rank


def fundamental_cycles(D, birth=1.):
    forest = [[] for _ in D]
    cycles = []
    for a, b in combinations(range(len(D)), 2):
        if D[a, b] > birth:
            continue
        paths, queue = {a: []}, [a]
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


def verify(result, target, cycles, birth=1., survival=1.2):
    # Also forbids NumPy scalars, dataclasses, sets, NaN, and non-JSON results.
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["source_validated"] is True
    assert result["source_rank"] == len(cycles)
    present, survives, rank = oracle(target, cycles, birth, survival)
    assert result["surviving_rank"] == rank
    assert result["all_classes_independent"] == (rank == len(cycles))
    assert len(result["witnesses"]) == len(cycles)
    for index, witness in enumerate(result["witnesses"]):
        assert witness == {
            "cycle_index": index,
            "missing_birth_edges": [list(e) for e in sorted(parity(cycles[index]))
                                    if target[e] > birth],
            "edges_present": present[index], "survives": survives[index],
        }
    coefficient_columns = []
    assert len(result["relations"]) == sum(present) - rank
    for relation in result["relations"]:
        indices = relation["cycle_indices"]
        assert indices and indices == sorted(set(indices))
        assert all(0 <= i < len(cycles) and present[i] for i in indices)
        assert relation["kind"] == ("filled" if len(indices) == 1 else "merged")
        faces = relation["filling_triangles"]
        assert faces == [list(t) for t in sorted(set(map(tuple, faces)))]
        expected = parity(e for i in indices for e in cycles[i])
        actual = set()
        for face in faces:
            assert len(face) == 3 and face == sorted(set(face))
            assert all(0 <= v < len(target) for v in face)
            for i, j in combinations(face, 2):
                assert target[i, j] <= survival
                actual.symmetric_difference_update([(i, j)])
        assert actual == expected
        coefficient_columns.append([i in indices for i in range(len(cycles))])
    # Independent relations plus the oracle's nullity proves completeness.
    assert dense_rank(coefficient_columns, len(cycles)) == len(result["relations"])


def test_intact_square_no_obstruction_and_generator_inputs_not_mutated():
    source = distances(POINTS)
    before = source.copy()
    result = find_h1_obstructions(source, source, (iter(SQUARE) for _ in range(1)), 1, 1.2)
    verify(result, source, [SQUARE])
    assert result["relations"] == []
    assert result["n_simplices"] == 16
    np.testing.assert_array_equal(source, before)
    assert find_h1_obstructions(source.tolist(), source.tolist(), [SQUARE], 1, 1.2) == result
    verify(find_h1_obstructions(source, source, [SQUARE], 1, 1), source, [SQUARE], 1, 1)


def test_external_center_filling_uses_full_target_domain():
    source = distances(np.vstack([POINTS, [10., 10.]]))
    target = distances(np.vstack([POINTS, [.5, .5]]))
    result = find_h1_obstructions(source, target, [SQUARE], 1, 1.2)
    verify(result, target, [SQUARE])
    assert result["relations"] == [{"kind": "filled", "cycle_indices": [0],
                                     "filling_triangles": [[0, 1, 4], [0, 3, 4],
                                                           [1, 2, 4], [2, 3, 4]]}]
    assert find_h1_obstructions(source[:4, :4], target[:4, :4], [SQUARE], 1, 1.2)[
        "all_classes_independent"]


def duplicated_squares():
    cycles = [SQUARE, [(i + 4, j + 4) for i, j in SQUARE]]
    source = distances(np.vstack([POINTS, POINTS + [10., 0.]]))
    target = distances(np.vstack([POINTS, POINTS]))
    return source, target, cycles


def test_duplicated_target_squares_each_survives_but_xor_fills():
    source, target, cycles = duplicated_squares()
    result = find_h1_obstructions(source, target, cycles, 1, 1.2)
    verify(result, target, cycles)
    assert all(w["survives"] for w in result["witnesses"])
    assert result["surviving_rank"] == 1
    assert len(result["relations"]) == 1
    assert result["relations"][0]["cycle_indices"] == [0, 1]
    assert result["relations"][0]["kind"] == "merged"
    with pytest.raises(ValueError, match="invalid_source_contract"):
        find_h1_obstructions(target, source, cycles, 1, 1.2)


def test_multiple_independent_relations_distinguish_filled_from_merged():
    cycles = [[(i + shift, j + shift) for i, j in SQUARE] for shift in (0, 4, 8)]
    source = graph_matrix(12, [e for c in cycles for e in c])
    target = source.copy()
    for i in range(4):
        for j in (i + 4, (i + 1) % 4 + 4):
            target[i, j] = target[j, i] = 1.2
    target[8, 10] = target[10, 8] = 1.2
    result = find_h1_obstructions(source, target, cycles, 1, 1.2)
    verify(result, target, cycles)
    assert [(r["kind"], r["cycle_indices"]) for r in result["relations"]] == [
        ("merged", [0, 1]), ("filled", [2])]
    assert [w["survives"] for w in result["witnesses"]] == [True, True, False]


def test_wrong_critical_edge_noncollinear_target_reports_missing_ids():
    source = distances(POINTS)
    points = np.array([[0., 0.], [0., 3.], [1., 1.], [2., 1.]])
    target = distances(points)
    assert np.linalg.matrix_rank(points - points[0]) == 2
    assert source[2, 3] == target[2, 3] == 1
    assert source[0, 2] == target[0, 2] == np.sqrt(2)
    result = find_h1_obstructions(source, target, [SQUARE], 1, 1.2)
    verify(result, target, [SQUARE])
    assert result["witnesses"][0]["missing_birth_edges"] == [[0, 1], [0, 3], [1, 2]]
    assert result["relations"] == []


def test_missing_birth_edges_excluded_even_if_filled_at_survival():
    source, _, cycles = duplicated_squares()
    target = graph_matrix(8, list(combinations(range(8), 2)))
    target[0, 1] = target[1, 0] = 1.1
    result = find_h1_obstructions(source, target, cycles, 1, 1.2)
    verify(result, target, cycles)
    assert result["witnesses"][0]["missing_birth_edges"] == [[0, 1]]
    assert [r["cycle_indices"] for r in result["relations"]] == [[1]]


def test_row_permutation_does_not_preserve_same_id_edges():
    source = distances(POINTS)
    target = source[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])]
    result = find_h1_obstructions(source, target, [SQUARE], 1, 1.2)
    verify(result, target, [SQUARE])
    assert result["witnesses"][0]["missing_birth_edges"] == [[0, 1], [2, 3]]


def test_parity_orientation_and_zero_distance_ties():
    source = graph_matrix(4, SQUARE)
    source[source == 1] = 0
    chain = [(np.int64(j), np.int64(i)) for i, j in SQUARE]
    chain += [(0, 2), (2, 0), (0, 1), (1, 0)]
    before = list(chain)
    result = find_h1_obstructions(source, source, [chain], 0, 0)
    verify(result, source, [chain], 0, 0)
    assert result == find_h1_obstructions(source, source, [SQUARE], 0, 0)
    assert chain == before
    target = np.zeros((4, 4))
    verify(find_h1_obstructions(source, target, [chain], 0, 0), target, [chain], 0, 0)
    with pytest.raises(ValueError, match="invalid_source_contract"):
        find_h1_obstructions(target, source, [SQUARE], 0, 0)


@pytest.mark.parametrize("cycles", [[], [[]], [[(0, 1)]], [[(0, 1), (1, 0)]],
                                    [SQUARE, SQUARE], [SQUARE, list(reversed(SQUARE))]])
def test_invalid_source_cycles_raise(cycles):
    source = distances(POINTS)
    with pytest.raises(ValueError, match="invalid_source_contract"):
        find_h1_obstructions(source, source, cycles, 1, 1.2)


def test_invalid_source_missing_edges_filled_and_dependent():
    D = distances(POINTS)
    source = D.copy()
    source[0, 1] = source[1, 0] = 1.1
    with pytest.raises(ValueError, match="invalid_source_contract.*missing"):
        find_h1_obstructions(source, D, [SQUARE], 1, 1.2)
    with pytest.raises(ValueError, match="invalid_source_contract.*filled"):
        find_h1_obstructions(D, D, [SQUARE], 1, D[0, 2])
    source, target, cycles = duplicated_squares()
    # A non-identical but source-dependent XOR representative must also fail.
    with pytest.raises(ValueError, match="invalid_source_contract.*dependent"):
        find_h1_obstructions(source, target, cycles + [cycles[0] + cycles[1]], 1, 1.2)
    verify(find_h1_obstructions(source, target, [cycles[0] + cycles[1]], 1, 1.2),
           target, [cycles[0] + cycles[1]])


@pytest.mark.parametrize("edge", [(0, 0), (-1, 2), (0, 4), (0., 1), (False, 1),
                                   (0, 1, 2), (0,), None, 2, ("0", 1)])
def test_invalid_edges_rejected_before_parity_cancellation(edge):
    D = distances(POINTS)
    with pytest.raises(ValueError):
        find_h1_obstructions(D, D, [SQUARE + [edge, edge]], 1, 1.2)


@pytest.mark.parametrize("bad", [np.ones((4, 3)), np.zeros((3, 3)), [[0j]], [["0"]],
                                 [[np.nan]], [[np.inf]], [[1]], [[0, -1], [-1, 0]],
                                 [[0, 1], [2, 0]], [[0], [1, 2]], None])
@pytest.mark.parametrize("which", ["source", "target"])
def test_invalid_matrices(bad, which):
    D = distances(POINTS)
    args = (bad, D) if which == "source" else (D, bad)
    with pytest.raises(ValueError):
        find_h1_obstructions(*args, [SQUARE], 1, 1.2)


@pytest.mark.parametrize("birth,survival", [(-1, 1), (1, -1), (1.2, 1), (np.nan, 2),
                                           (1, np.inf), (True, 1), ("1", 2), (1, 1j)])
def test_invalid_radii(birth, survival):
    D = distances(POINTS)
    with pytest.raises(ValueError):
        find_h1_obstructions(D, D, [SQUARE], birth, survival)


@pytest.mark.parametrize("key", ["max_vertices", "max_simplices", "max_reduction_operations",
                                 "max_reduction_entries"])
@pytest.mark.parametrize("value", [-1, True, 1.5, np.inf, None])
def test_invalid_budgets(key, value):
    D = distances(POINTS)
    with pytest.raises(ValueError, match=key):
        find_h1_obstructions(D, D, [SQUARE], 1, 1.2, **{key: value})


@pytest.mark.parametrize("key", ["max_vertices", "max_simplices", "max_reduction_operations",
                                 "max_reduction_entries"])
def test_zero_budgets_never_return_partial_success(key):
    D = distances(POINTS)
    with pytest.raises(RuntimeError, match=key):
        find_h1_obstructions(D, D, [SQUARE], 1, 1.2, **{key: 0})


def test_exact_cumulative_budget_boundaries_with_multiword_provenance():
    source = graph_matrix(13, SQUARE)
    target = np.zeros((13, 13))  # 286 triangles, five provenance words.
    result = find_h1_obstructions(source, target, [SQUARE], 1, 1.2)
    verify(result, target, [SQUARE])
    limits = {"max_vertices": 13, "max_simplices": result["n_simplices"],
              "max_reduction_operations": result["reduction_operations"],
              "max_reduction_entries": result["peak_reduction_entries"]}
    assert find_h1_obstructions(source, target, [SQUARE], 1, 1.2, **limits) == result
    for key, value in limits.items():
        with pytest.raises(H1BudgetExceeded, match=key):
            find_h1_obstructions(source, target, [SQUARE], 1, 1.2,
                                 **dict(limits, **{key: value - 1}))


def test_source_target_work_is_summed_not_reset():
    D = distances(POINTS)
    result = find_h1_obstructions(D, D, [SQUARE], 1, 1.2)
    # W=3; source=11W, target=14W. Storage source=4W, target=7W.
    assert result["reduction_operations"] == 75
    assert result["peak_reduction_entries"] == 21
    with pytest.raises(H1BudgetExceeded, match="max_reduction_operations"):
        find_h1_obstructions(D, D, [SQUARE], 1, 1.2, max_reduction_operations=74)


def test_simplex_preflight_before_any_reduction(monkeypatch):
    import open_deep_tda.structural_obstructions as module
    source, target = graph_matrix(8, SQUARE), np.zeros((8, 8))

    def forbidden(*args, **kwargs):
        pytest.fail("reduction allocated before simplex preflight")

    monkeypatch.setattr(module, "_CertificateBudget", forbidden)
    with pytest.raises(H1BudgetExceeded, match="max_simplices"):
        find_h1_obstructions(source, target, [SQUARE], 1, 1.2, max_simplices=103)


def test_triangle_table_storage_reserved_before_allocation(monkeypatch):
    import open_deep_tda.structural_obstructions as module
    original = module._triangles
    calls = []

    def counted(adjacency):
        calls.append(1)
        assert len(calls) <= 2, "target triangle table allocated without reservation"
        yield from original(adjacency)

    monkeypatch.setattr(module, "_triangles", counted)
    with pytest.raises(H1BudgetExceeded, match="max_reduction_entries"):
        find_h1_obstructions(graph_matrix(8, SQUARE), np.zeros((8, 8)), [SQUARE], 1, 1.2,
                             max_reduction_entries=18)
    assert len(calls) == 2


def test_hard_cap_64_even_when_caller_requests_more():
    D = graph_matrix(64, SQUARE)
    result = find_h1_obstructions(D, D, [SQUARE], 1, 1.2, max_vertices=1000)
    assert result["all_classes_independent"]
    D = graph_matrix(65, SQUARE)
    with pytest.raises(H1BudgetExceeded, match="64"):
        find_h1_obstructions(D, D, [SQUARE], 1, 1.2, max_vertices=1000)


def test_multiword_witness_coefficients_and_high_vertex_ids():
    source = graph_matrix(19, [(i, j) for i in range(9) for j in range(9, 19)])
    cycles = [[(0, 9), (9, i), (i, j), (j, 0)] for i in range(1, 9)
              for j in range(10, 19)]
    assert len(cycles) == 72
    target = source.copy()
    target[0, 8] = target[8, 0] = 1.2
    result = find_h1_obstructions(source, target, cycles, 1, 1.2)
    verify(result, target, cycles)
    assert [r["cycle_indices"] for r in result["relations"]] == [[i] for i in range(63, 72)]


def test_precision_safe_thresholds():
    D = graph_matrix(4, SQUARE).astype(np.float32)
    D[0, 2] = D[2, 0] = D[1, 3] = D[3, 1] = np.nextafter(np.float32(1), np.float32(2))
    assert find_h1_obstructions(D.astype(float), D, [SQUARE], 1, 1.0000000894069672)[
        "all_classes_independent"]
    side = 2**53 + 1
    D = np.full((4, 4), side + 1, dtype=np.int64)
    np.fill_diagonal(D, 0)
    for i, j in SQUARE:
        D[i, j] = D[j, i] = side
    assert find_h1_obstructions(D, D, [SQUARE], np.int64(side), side)["all_classes_independent"]
    with pytest.raises(ValueError, match="invalid_source_contract.*missing"):
        find_h1_obstructions(D, D, [SQUARE], float(side - 1), side)


def test_random_small_dense_gf2_correctness_and_certificates():
    rng = np.random.default_rng(59317)
    families = certificates = 0
    for _ in range(160):
        n = int(rng.integers(4, 9))

        def random_matrix():
            upper = np.triu(rng.choice([1., 1.2, 2.], size=(n, n),
                                       p=[.45, .1, .45]), 1)
            return upper + upper.T

        source, target = random_matrix(), random_matrix()
        independent = []
        for cycle in fundamental_cycles(source):
            if oracle(source, independent + [cycle])[2] > len(independent):
                independent.append(cycle)
        if not independent:
            continue
        families += 1
        # Force some targets to retain birth edges, giving nontrivial relations.
        if families % 2:
            for cycle in independent:
                for i, j in cycle:
                    target[i, j] = target[j, i] = 1
        for D in (source, target):
            result = find_h1_obstructions(source, D, independent, 1, 1.2)
            verify(result, D, independent)
            checked = check_h1_witnesses(source, D, independent, 1, 1.2)
            assert checked.surviving_rank == result["surviving_rank"]
            certificates += len(result["relations"])
        permutation = rng.permutation(n)
        inverse = np.argsort(permutation)
        relabeled = [[(int(inverse[i]), int(inverse[j])) for i, j in c] for c in independent]
        relabeled_target = target[np.ix_(permutation, permutation)]
        result = find_h1_obstructions(source[np.ix_(permutation, permutation)],
                                      relabeled_target, relabeled, 1, 1.2)
        verify(result, relabeled_target, relabeled)
    assert families >= 25 and certificates >= 10
    # Explicit duplicate-square fixture covers merged relations even if random
    # targets happen to fill all eligible classes individually for a given seed.
