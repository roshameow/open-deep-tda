"""Offline standalone sparse-H1 regressions and an independent dense GF(2) oracle.

Run: python -m pytest -q tests/test_structural_sparse_h1.py
Tests import the installed public module; the optional cross-check reports a
skip only if the existing public exact-checker API is unavailable. Full-width accounting boundaries,
external fillings and a free-leading-coordinate quotient trap are retained as
reproducible implementation experience, not just happy-path examples.
NumPy longdouble has only binary64 precision on Apple arm64; precision tests
must inspect dtype.itemsize rather than assume extended precision by name.
"""

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from open_deep_tda import structural_sparse_h1 as sparse

H1Limits = sparse.H1Limits
ResourceLimitError = sparse.ResourceLimitError
analyze_sparse_h1 = sparse.analyze_sparse_h1


SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]


def graph_matrix(n, edges):
    matrix = np.full((n, n), 3., dtype=np.float64)
    np.fill_diagonal(matrix, 0.)
    for u, v in edges:
        matrix[u, v] = matrix[v, u] = 1.
    return matrix


def rank_dense(columns, dimension):
    """Row elimination of a dense byte matrix; no integer bitsets/pivot dicts."""
    matrix = np.array(columns, dtype=np.uint8).reshape((-1, dimension)).T.copy()
    row = 0
    for column in range(matrix.shape[1]):
        choices = np.flatnonzero(matrix[row:, column])
        if not len(choices):
            continue
        pivot = row + int(choices[0])
        matrix[[row, pivot], :] = matrix[[pivot, row], :]
        for other in range(row + 1, dimension):
            if matrix[other, column]:
                matrix[other, :] = np.bitwise_xor(matrix[other, :], matrix[row, :])
        row += 1
        if row == dimension:
            break
    return row


def oracle(matrix, cycles, a, b):
    """Deliberately uses the dense ALL-pairs basis, independent of sparse code."""
    edges = list(combinations(range(len(matrix)), 2))
    ids = {edge: j for j, edge in enumerate(edges)}

    def column(chain):
        result = np.zeros(len(edges), dtype=np.uint8)
        for edge in chain:
            result[ids[tuple(sorted(edge))]] ^= 1
        return result

    boundaries = [column(list(combinations(triple, 2)))
                  for triple in combinations(range(len(matrix)), 3)
                  if all(matrix[u, v] <= b for u, v in combinations(triple, 2))]
    initial = rank_dense(boundaries, len(edges))
    eligible, presence, survives = [], [], []
    for chain in cycles:
        valid = all(matrix[u, v] <= a for u, v in chain)
        presence.append(valid)
        c = column(chain)
        survives.append(valid and rank_dense(boundaries + [c], len(edges)) > initial)
        if valid:
            eligible.append(c)
    rank = rank_dense(boundaries + eligible, len(edges)) - initial
    return rank, presence, survives


def assert_oracle(matrix, cycles, a=1., b=1.):
    expected, presence, survives = oracle(matrix, cycles, a, b)
    result = analyze_sparse_h1(matrix, cycles, a, b)
    assert type(result.certified) is bool
    assert type(result.rank) is int
    assert result.rank == expected
    assert result.certified == (expected == len(cycles))
    assert [status.edges_present for status in result.statuses] == presence
    assert [status.survives for status in result.statuses] == survives
    assert all(status.closed for status in result.statuses)
    assert result.work["basis_width"] == np.count_nonzero(np.triu(matrix <= b, 1))
    return result


def forest_cycles(matrix, radius):
    """Independent BFS fundamental-cycle enumeration for random test inputs."""
    forest = [[] for _ in matrix]
    cycles = []
    for u, v in combinations(range(len(matrix)), 2):
        if matrix[u, v] > radius:
            continue
        paths, queue = {u: []}, [u]
        for node in queue:
            for neighbour in forest[node]:
                if neighbour not in paths:
                    paths[neighbour] = paths[node] + [(node, neighbour)]
                    queue.append(neighbour)
        if v in paths:
            cycles.append(paths[v] + [(v, u)])
        else:
            forest[u].append(v)
            forest[v].append(u)
    return cycles


def test_square_birth_survival_and_closed_thresholds():
    matrix = graph_matrix(4, SQUARE)
    result = assert_oracle(matrix, [SQUARE])
    assert result.certified and result.rank == 1
    assert result.statuses[0].reason == "survives"
    assert not assert_oracle(matrix, [SQUARE], np.nextafter(1., 0.), 1.).certified
    assert not assert_oracle(matrix, [SQUARE], 1., 3.).certified
    assert result.work["triangles"] == 0
    assert result.work["edges"] == result.work["basis_width"] == 4


def test_all_rows_can_fill_cycle_and_permutation_keeps_global_ids():
    matrix = graph_matrix(5, SQUARE + [(i, 4) for i in range(4)])
    assert analyze_sparse_h1(matrix[:4, :4], [SQUARE], 1, 1).certified
    result = assert_oracle(matrix, [SQUARE])
    assert not result.certified and result.rank == 0
    assert result.work["triangles"] == 4
    order = np.array([4, 2, 0, 3, 1])
    inverse = np.argsort(order)
    chain = [(int(inverse[u]), int(inverse[v])) for u, v in SQUARE]
    permuted = assert_oracle(matrix[np.ix_(order, order)], [chain])
    assert permuted.rank == result.rank


def test_quotient_eliminates_pivots_below_free_leading_coordinate():
    # The triangle (0,1,4) relates the two chains, but its pivot lies BELOW
    # their free leading edge (2,3). Leading-only quotient reduction fails.
    matrix = graph_matrix(5, SQUARE + [(0, 4), (1, 4)])
    detour = [(0, 4), (4, 1), (1, 2), (2, 3), (3, 0)]
    result = assert_oracle(matrix, [SQUARE, detour])
    assert result.rank == 1 and not result.certified
    assert all(status.survives for status in result.statuses)
    assert result.reason == "filled_or_dependent_selected_family"


def test_independent_family_rank_not_dimension_of_all_h1():
    second = [(u + 4, v + 4) for u, v in SQUARE]
    matrix = graph_matrix(8, SQUARE + second)
    assert assert_oracle(matrix, [SQUARE]).rank == 1
    assert assert_oracle(matrix, [SQUARE, second]).rank == 2
    result = assert_oracle(matrix, [SQUARE, second, SQUARE + second])
    assert result.rank == 2 and all(s.survives for s in result.statuses)


def test_missing_birth_edges_excluded_even_if_present_at_survival():
    matrix = graph_matrix(8, SQUARE + [(u + 4, v + 4) for u, v in SQUARE])
    matrix[0, 1] = matrix[1, 0] = 1.5
    second = [(u + 4, v + 4) for u, v in SQUARE]
    result = assert_oracle(matrix, [SQUARE, second], 1, 2)
    assert result.rank == 1 and not result.certified
    assert result.statuses[0].reason == "missing_birth_edges"


def test_parity_orientation_zero_chains_and_raw_birth_membership():
    matrix = graph_matrix(4, SQUARE)
    reversed_chain = [(v, u) for u, v in reversed(SQUARE)]
    assert assert_oracle(matrix, [reversed_chain]).certified
    assert not assert_oracle(matrix, [SQUARE + reversed_chain]).certified
    assert not assert_oracle(matrix, [[]]).certified
    chain = SQUARE + [(0, 2), (2, 0)]
    result = assert_oracle(matrix, [chain], 1, 1)
    assert not result.statuses[0].edges_present  # raw canceled edges still checked
    assert not result.certified
    assert sparse.validate_cycles([chain], 4)[0][-2:] == ((0, 2), (0, 2))


@pytest.mark.parametrize("seed", range(40))
def test_independent_dense_random_small(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 10))
    upper = rng.choice([0., .5, 1., 1.5, 2.], size=(n, n))
    matrix = np.triu(upper, 1)
    matrix = matrix + matrix.T
    a, b = [(0., .5), (.5, 1.), (1., 1.), (1., 1.5), (1.5, 2.)][seed % 5]
    cycles = forest_cycles(matrix, a)
    if cycles:
        cycles += [cycles[0], cycles[0] + cycles[-1]]
    vertices = rng.choice(n, size=3, replace=False).tolist()
    cycles.append(list(zip(vertices, vertices[1:] + vertices[:1])))
    assert_oracle(matrix, cycles, a, b)


def test_empty_family_and_empty_matrix_are_explicit_vacuous_success():
    result = analyze_sparse_h1(np.zeros((0, 0), dtype=np.float64), [], 0, 0)
    assert result.certified and result.rank == 0 and result.statuses == ()
    assert result.work["edges"] == 0
    assert analyze_sparse_h1(np.zeros((1, 1)), [], 0, 0).certified


@pytest.mark.parametrize("bad", [
    [[0., 1.], [1., 0.]], np.zeros((2, 2), dtype=np.float32),
    np.zeros((2, 2), dtype=np.int64), np.zeros((2, 3)), np.zeros(2),
    np.array([[0., 1.], [2., 0.]]), np.array([[1., 0.], [0., 0.]]),
    np.array([[0., -1.], [-1., 0.]]), np.array([[0., np.inf], [np.inf, 0.]]),
    np.array([[0., np.nan], [np.nan, 0.]]), np.zeros((1, 1), dtype=complex),
])
def test_strict_matrix_validation(bad):
    with pytest.raises(ValueError):
        sparse.validate_distance_matrix(bad)


def test_exact_symmetry_no_tolerance_and_float64_strided_readonly_input():
    matrix = graph_matrix(4, SQUARE)
    matrix[0, 1] = np.nextafter(1., 2.)
    with pytest.raises(ValueError):
        sparse.validate_distance_matrix(matrix)
    matrix = graph_matrix(8, [(2*u, 2*v) for u, v in SQUARE])[::2, ::2]
    before = matrix.copy()
    matrix.flags.writeable = False
    assert sparse.validate_distance_matrix(matrix) is matrix
    assert analyze_sparse_h1(matrix, [SQUARE], 1, 1).certified
    np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize("cycles", [None, iter([]), [None], [[(0, 0)]],
    [[(0, 4)]], [[(-1, 0)]], [[(True, 1)]], [[(0., 1)]], [[(0, 1, 2)]],
    [[(0, 1)]], [[0]], [np.zeros((2, 3), dtype=int)], np.array(0),
])
def test_invalid_cycles_raise_valueerror(cycles):
    with pytest.raises(ValueError):
        sparse.validate_cycles(cycles, 4)


@pytest.mark.parametrize("a,b", [(-1, 1), (2, 1), (0, float('inf')),
    (float('nan'), 1), (True, 1), (0, "1"), (0, complex(1))])
def test_invalid_radii(a, b):
    with pytest.raises(ValueError):
        analyze_sparse_h1(graph_matrix(4, SQUARE), [SQUARE], a, b)


def test_longdouble_follows_actual_platform_precision():
    # On Apple arm64, longdouble is binary64 rather than extended precision.
    matrix = graph_matrix(4, SQUARE)
    if np.dtype(np.longdouble).itemsize > 8:
        with pytest.raises(ValueError):
            analyze_sparse_h1(matrix, [SQUARE], 0, np.longdouble(1))
    else:
        assert analyze_sparse_h1(matrix, [SQUARE], 1, np.longdouble(1)).certified


def test_integer_threshold_is_not_rounded_through_numpy_float_promotion():
    matrix = graph_matrix(4, SQUARE)
    matrix[matrix == 1.] = float(2**53 + 2)
    matrix[matrix == 3.] = float(2**54)
    result = analyze_sparse_h1(matrix, [SQUARE], 2**53 + 1, 2**53 + 2)
    assert not result.statuses[0].edges_present


@pytest.mark.parametrize("limits", [None, H1Limits(max_vertices=True),
    H1Limits(max_edges=-1), H1Limits(max_triangles=1.5), H1Limits(max_cycles=np.bool_(1)),
    H1Limits(max_vertices=1025), H1Limits(max_edges=100001),
    H1Limits(max_triangles=2000001), H1Limits(max_chain_edges=1000001)])
def test_invalid_limit_configuration(limits):
    with pytest.raises(ValueError):
        analyze_sparse_h1(graph_matrix(4, SQUARE), [SQUARE], 1, 1, limits=limits)


@pytest.mark.parametrize("limits", [H1Limits(max_vertices=3), H1Limits(max_edges=3),
    H1Limits(max_cycles=0), H1Limits(max_chain_edges=3),
    H1Limits(max_word_ops=0), H1Limits(max_storage_words=0)])
def test_resource_refusal_is_exception_not_partial_success(limits):
    with pytest.raises(ResourceLimitError):
        analyze_sparse_h1(graph_matrix(4, SQUARE), [SQUARE], 1, 1, limits=limits)


def test_triangle_preflight_happens_before_any_reduction(monkeypatch):
    def forbidden(*args):
        pytest.fail("reduction ran before triangle preflight completed")
    monkeypatch.setattr(sparse, "_insert", forbidden)
    with pytest.raises(ResourceLimitError, match="max_triangles.*preflight"):
        analyze_sparse_h1(np.zeros((8, 8)), [], 0, 0,
                          limits=H1Limits(max_triangles=55))


def test_word_and_storage_preflight_happen_before_reduction(monkeypatch):
    def forbidden(*args):
        pytest.fail("reduction ran before work/storage preflight completed")
    monkeypatch.setattr(sparse, "_insert", forbidden)
    for limits in (H1Limits(max_word_ops=7), H1Limits(max_storage_words=2)):
        with pytest.raises(ResourceLimitError, match="preflight"):
            analyze_sparse_h1(np.zeros((3, 3)), [], 0, 0, limits=limits)


def test_exact_budget_boundaries_and_runtime_refusals():
    matrix = graph_matrix(5, SQUARE + [(0, 4), (1, 4)])
    result = analyze_sparse_h1(matrix, [SQUARE], 1, 1)
    limits = H1Limits(max_word_ops=result.work["word_ops"],
                      max_storage_words=result.work["peak_storage_words"],
                      max_vertices=5, max_edges=6, max_triangles=1,
                      max_cycles=1, max_chain_edges=4)
    assert analyze_sparse_h1(matrix, [SQUARE], 1, 1, limits=limits) == result
    assert result.work["word_ops"] > result.work["preflight_min_word_ops"]
    with pytest.raises(ResourceLimitError, match="during reduction"):
        analyze_sparse_h1(matrix, [SQUARE], 1, 1,
                          limits=replace(limits, max_word_ops=limits.max_word_ops - 1))
    with pytest.raises(ResourceLimitError, match="during reduction"):
        analyze_sparse_h1(matrix, [SQUARE], 1, 1,
                          limits=replace(limits, max_storage_words=limits.max_storage_words - 1))


def test_hard_vertex_ceiling_and_all_1024_rows_without_all_pairs_basis():
    with pytest.raises(ResourceLimitError, match="max_vertices"):
        sparse.validate_distance_matrix(np.zeros((1025, 1025)))
    ring = [(i, (i + 1) % 1024) for i in range(1024)]
    matrix = graph_matrix(1024, ring)
    result = analyze_sparse_h1(matrix, [ring], 1, 1)
    assert result.certified and result.work["vertices"] == 1024
    assert result.work["basis_width"] == 1024
    assert result.work["words_per_vector"] == 16
    assert result.work["triangles"] == 0
    assert result.work["peak_storage_words"] == 64


def test_default_limits_support_full_960_with_about_745k_triangles_offline():
    # Synthetic size regression, independent of any private dataset/cache.
    # A 166-clique contributes 748660 triangles; a disjoint square survives.
    matrix = np.full((960, 960), 3., dtype=np.float64)
    matrix[:166, :166] = 1.
    np.fill_diagonal(matrix, 0.)
    ring = [(u + 956, v + 956) for u, v in SQUARE]
    for u, v in ring:
        matrix[u, v] = matrix[v, u] = 1.
    result = analyze_sparse_h1(matrix, [ring], 1, 1)
    assert result.certified and result.rank == 1
    assert result.work["triangles"] == 748660
    assert result.work["basis_width"] == 166 * 165 // 2 + 4
    assert result.work["boundary_rank"] == 166 * 165 // 2 - 165
    assert result.work["word_ops"] <= H1Limits().max_word_ops
    assert result.work["peak_storage_words"] <= H1Limits().max_storage_words


def test_conservative_word_charges_for_single_triangle():
    # Three shifts + three XORs + initialization + pivot inspection = 8W.
    result = analyze_sparse_h1(np.zeros((3, 3)), [], 0, 0)
    assert result.work["word_ops"] == result.work["preflight_min_word_ops"] == 8
    assert result.work["peak_storage_words"] == 4  # one pivot + three scratch
    assert result.work["boundary_rank"] == 1


def test_hard_edge_ceiling():
    with pytest.raises(ResourceLimitError, match="max_edges"):
        analyze_sparse_h1(np.zeros((448, 448)), [], 0, 0)


def test_no_production_or_private_imports():
    import ast
    tree = ast.parse(Path(sparse.__file__).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not node.level
            names.append(node.module)
    assert set(names) <= {"dataclasses", "numbers", "math", "numpy"}


def test_optional_existing_public_exact_checker():
    try:
        from open_deep_tda.structural_h1 import check_h1_witnesses
    except ImportError:
        pytest.skip("optional production exact checker is not importable")
    source = graph_matrix(5, SQUARE)
    for target in (source, graph_matrix(5, SQUARE + [(i, 4) for i in range(4)])):
        existing = check_h1_witnesses(source, target, [SQUARE], 1, 1)
        result = analyze_sparse_h1(target, [SQUARE], 1, 1)
        assert result.certified == existing.accepted
        assert result.rank == existing.surviving_rank
        assert [(s.edges_present, s.survives) for s in result.statuses] == [
            (s.edges_present, s.survives) for s in existing.witnesses]


@pytest.mark.parametrize('seed', range(16))
def test_random_small_comparison_existing_exact_checker(seed):
    try:
        from open_deep_tda.structural_h1 import check_h1_witnesses
    except ImportError:
        pytest.skip('optional production exact checker is not importable')
    rng = np.random.default_rng(5000+seed)
    n = int(rng.integers(4, 13))
    source = graph_matrix(n, SQUARE)
    upper = np.triu(rng.choice([.5, 1., 1.5, 3.], size=(n, n)), 1)
    target = upper + upper.T
    old = check_h1_witnesses(source, target, [SQUARE], 1., 1.5)
    new = assert_oracle(target, [SQUARE], 1., 1.5)
    assert new.certified == old.accepted
    assert new.rank == old.surviving_rank
    assert [(s.edges_present, s.survives) for s in new.statuses] == [
        (s.edges_present, s.survives) for s in old.witnesses]
