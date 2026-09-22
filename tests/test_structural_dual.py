"""Public dual-H1 regressions with an independent dense uint8 GF(2) oracle.

Methods/pitfalls: evaluate every source triangle and the full identity pairing,
including off-witness rows, instead of checking just reduction pivots. Fixed
source duals give sufficient, not necessary, target certificates: an attached
triangle can fail the certificate without killing H1. Individually surviving
classes can merge, and repairing a stale triangle batch requires a fresh scan.
Parity normalization and Python-scalar thresholds preserve represented values.
The seeded 160-instance oracle and all 1,024 five-vertex target graphs remain
explicit, deterministic tests, independent of production integer-bitset algebra.
Budget boundaries use reported totals rather than guessed operation counts;
word-width, output-storage lower bounds and preflight sentinels independently
check that those totals cover the documented work and retained payload. Pairing
cell emission costs work in addition to shift/AND extraction; the hand-counted
square regression prevents omitting that cost while retaining self-consistent
reported-boundary tests. Corrupted-dual sentinels exercise both explicit checks.

Run: PYTHONPATH=python python -m pytest -q tests/test_structural_dual.py
No optional PH library, external dataset, network, or timing assertion is needed.
"""

from itertools import combinations
import inspect
import json

import numpy as np
import pytest

from open_deep_tda import structural_dual, structural_h1
from open_deep_tda.structural_dual import dual_h1_certificate
from open_deep_tda.structural_h1 import H1BudgetExceeded, check_h1_witnesses


SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]
LIMITS = {
    "max_vertices": 64,
    "max_simplices": 200000,
    "max_reduction_operations": 20000000,
    "max_reduction_entries": 2000000,
}
FIELDS = {
    "scope", "n_vertices", "n_cocycles", "birth_radius", "survival_radius",
    "source_boundary_rank", "source_triangle_count", "source_pairing", "cochains",
    "required_birth_edges", "missing_target_birth_edges", "target_triangle_count",
    "forbidden_triangles", "certificate_satisfied", "n_simplices",
    "reduction_operations", "peak_reduction_entries",
}


def graph_matrix(n, edges):
    D = np.full((n, n), 2.)
    np.fill_diagonal(D, 0.)
    for u, v in edges:
        D[u, v] = D[v, u] = 1.
    return D


def dense_rank(matrix):
    """Independent scalar/array GF2 row elimination, no integer bitsets."""
    A = np.array(matrix, dtype=np.uint8, copy=True)
    rank = 0
    for column in range(A.shape[1]):
        candidates = np.flatnonzero(A[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + candidates[0]
        A[[rank, pivot]] = A[[pivot, rank]]
        for row in range(rank + 1, len(A)):
            if A[row, column]:
                A[row] ^= A[rank]
        rank += 1
    return rank


def dense_complex(D, cycles, survival):
    edges = list(combinations(range(len(D)), 2))
    indices = {edge: i for i, edge in enumerate(edges)}

    def vector(chain):
        row = np.zeros(len(edges), dtype=np.uint8)
        for edge in chain:
            row[indices[tuple(sorted(edge))]] ^= 1
        return row

    triangles = [t for t in combinations(range(len(D)), 3)
                 if all(D[u, v] <= survival for u, v in combinations(t, 2))]
    B = np.array([vector(combinations(t, 2)) for t in triangles],
                 dtype=np.uint8).reshape(len(triangles), len(edges))
    G = np.array([vector(c) for c in cycles], dtype=np.uint8).reshape(len(cycles), len(edges))
    return edges, triangles, B, G


def fundamental_cycles(D, birth):
    """Graph spanning forest, independently generates closed candidate chains."""
    tree = [[] for _ in D]
    cycles = []
    for u, v in combinations(range(len(D)), 2):
        if D[u, v] > birth:
            continue
        paths = {u: []}
        queue = [u]
        for node in queue:
            for neighbor in tree[node]:
                if neighbor not in paths:
                    paths[neighbor] = paths[node] + [(node, neighbor)]
                    queue.append(neighbor)
        if v in paths:
            cycles.append(paths[v] + [(v, u)])
        else:
            tree[u].append(v)
            tree[v].append(u)
    return cycles


def verify_dense(report, source, target, cycles, birth=1, survival=1.2):
    assert type(report) is dict and set(report) == FIELDS
    assert report['scope'] == (
        'sufficient_selected_h1_certificate_not_necessary_not_whole_complex_map')
    assert report['n_vertices'] == len(source)
    assert report['n_cocycles'] == len(cycles)
    assert report['birth_radius'] == birth
    assert report['survival_radius'] == survival
    assert type(report['certificate_satisfied']) is bool
    for key in ('n_vertices', 'n_cocycles', 'source_boundary_rank',
                'source_triangle_count', 'target_triangle_count', 'n_simplices',
                'reduction_operations', 'peak_reduction_entries'):
        assert type(report[key]) is int and report[key] >= 0
    expected_simplices = 0
    for D in (source, target):
        expected_simplices += len(D)
        expected_simplices += sum(D[u, v] <= survival
                                 for u, v in combinations(range(len(D)), 2))
        expected_simplices += sum(all(D[u, v] <= survival
                                     for u, v in combinations(t, 2))
                                 for t in combinations(range(len(D)), 3))
    assert report['n_simplices'] == expected_simplices
    edges, _, B, G = dense_complex(source, cycles, survival)
    k = len(cycles)
    assert len(report['cochains']) == k
    A = np.zeros((k, len(edges)), dtype=np.uint8)
    for i, cochain in enumerate(report['cochains']):
        assert set(cochain) == {'index', 'support_edges'}
        assert type(cochain['index']) is int and cochain['index'] == i
        support = cochain['support_edges']
        assert support == sorted(support)
        assert len(support) == len(set(map(tuple, support)))
        for edge in cochain['support_edges']:
            assert type(edge) is list and len(edge) == 2
            assert all(type(v) is int for v in edge)
            assert 0 <= edge[0] < edge[1] < len(source)
            A[i, edges.index(tuple(edge))] = 1
    np.testing.assert_array_equal((A @ B.T) % 2, np.zeros((k, len(B)), dtype=np.uint8))
    np.testing.assert_array_equal((A @ G.T) % 2, np.eye(k, dtype=np.uint8))
    assert report['source_pairing'] == np.eye(k, dtype=int).tolist()
    assert report['source_triangle_count'] == len(B)
    assert report['source_boundary_rank'] == dense_rank(B)
    assert dense_rank(np.vstack([B, G])) - dense_rank(B) == k
    _, triangles, target_B, _ = dense_complex(target, cycles, survival)
    parities = (target_B @ A.T) % 2
    expected = []
    for triangle, parity in zip(triangles, parities):
        if not parity.any():
            continue
        eligible = [[u, v] for u, v in combinations(triangle, 2) if source[u, v] > survival]
        assert eligible
        assert not set(map(tuple, eligible)) & set(map(tuple, report['required_birth_edges']))
        expected.append(dict(vertices=list(triangle),
                             violated_cocycles=np.flatnonzero(parity).tolist(),
                             eligible_source_absent_edges=eligible))
    assert report['forbidden_triangles'] == expected
    assert report['target_triangle_count'] == len(target_B)
    required = [e for e, used in zip(edges, G.any(axis=0)) if used]
    assert report['required_birth_edges'] == [list(e) for e in required]
    assert report['missing_target_birth_edges'] == [
        [u, v] for u, v in required if target[u, v] > birth]
    present = all(target[u, v] <= birth for u, v in report['required_birth_edges'])
    assert report['certificate_satisfied'] == (present and not expected)
    if report['certificate_satisfied']:
        assert dense_rank(np.vstack([target_B, G])) - dense_rank(target_B) == k
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_square_identity_and_closed_threshold():
    source = graph_matrix(4, SQUARE)
    before = source.copy()
    report = dual_h1_certificate(source, source, [SQUARE], 1, 1)
    assert report['certificate_satisfied']
    assert report['cochains'] == [{'index': 0, 'support_edges': [[2, 3]]}]
    verify_dense(report, source, source, [SQUARE], survival=1)
    np.testing.assert_array_equal(source, before)
    with pytest.raises(ValueError, match='invalid_source_contract'):
        dual_h1_certificate(source, source, [SQUARE], 1, 2)


def test_target_center_fill_all_triangles_including_external_vertex():
    source = graph_matrix(5, SQUARE)
    target = source.copy()
    target[:4, 4] = target[4, :4] = 1.2  # exact survival threshold
    report = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
    assert not report['certificate_satisfied']
    assert not report['missing_target_birth_edges']
    assert report['target_triangle_count'] == 4
    assert report['forbidden_triangles'] == [dict(
        vertices=[2, 3, 4], violated_cocycles=[0],
        eligible_source_absent_edges=[[2, 4], [3, 4]])]
    assert not check_h1_witnesses(source, target, [SQUARE], 1, 1.2).accepted
    verify_dense(report, source, target, [SQUARE])
    # Either suggested edge removes this obstruction and breaks the cone fill.
    for u, v in report['forbidden_triangles'][0]['eligible_source_absent_edges']:
        repaired = target.copy()
        repaired[u, v] = repaired[v, u] = np.nextafter(1.2, np.inf)
        assert dual_h1_certificate(source, repaired, [SQUARE], 1, 1.2)['certificate_satisfied']
        assert check_h1_witnesses(source, repaired, [SQUARE], 1, 1.2).accepted


def test_two_rings_with_duplicate_rows_pairing_and_target_relation():
    cycles = [SQUARE, [(u + 4, v + 4) for u, v in SQUARE]]
    base = graph_matrix(8, cycles[0] + cycles[1])
    # Duplicate a point in EACH ring, introducing source triangles at zero
    # duplicate distance. All 10 rows must be included in the dual equations.
    ids = list(range(8)) + [0, 4]
    source = base[np.ix_(ids, ids)]
    target = source.copy()
    for u in range(4):
        for v in (u + 4, (u + 1) % 4 + 4):
            target[u, v] = target[v, u] = 1.2
    # Mixed input basis forces labeled cycle-row elimination, not just B2 XOR.
    mixed = [cycles[0] + cycles[1], cycles[0]]
    for family in (cycles, mixed):
        intact = dual_h1_certificate(source, source, family, 1, 1.2)
        assert intact['certificate_satisfied']
        assert intact['source_triangle_count'] > 0
        verify_dense(intact, source, source, family)
        failed = dual_h1_certificate(source, target, family, 1, 1.2)
        assert not failed['certificate_satisfied']
        assert failed['forbidden_triangles']
        verify_dense(failed, source, target, family)
    checked = check_h1_witnesses(source, target, cycles, 1, 1.2)
    assert all(w.survives for w in checked.witnesses)
    assert checked.surviving_rank == 1
    assert not checked.accepted


def test_batch_multiple_alternative_cone_fillings():
    source = graph_matrix(6, SQUARE)
    target = source.copy()
    target[:4, 4:] = target[4:, :4] = 1
    report = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
    assert [t['vertices'] for t in report['forbidden_triangles']] == [[2, 3, 4], [2, 3, 5]]
    verify_dense(report, source, target, [SQUARE])
    target[2, 4] = target[4, 2] = 2
    assert not check_h1_witnesses(source, target, [SQUARE], 1, 1.2).accepted
    assert len(dual_h1_certificate(source, target, [SQUARE], 1, 1.2)['forbidden_triangles']) == 1
    target[2, 5] = target[5, 2] = 2
    assert dual_h1_certificate(source, target, [SQUARE], 1, 1.2)['certificate_satisfied']


def test_certificate_not_necessary_for_valid_target():
    source = graph_matrix(5, SQUARE)
    target = source.copy()
    target[2, 4] = target[4, 2] = target[3, 4] = target[4, 3] = 1
    # A single triangle attached along an edge does not fill the square, but
    # this fixed zero-free-coordinate extension need not annihilate it.
    report = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
    assert not report['certificate_satisfied']
    assert check_h1_witnesses(source, target, [SQUARE], 1, 1.2).accepted
    verify_dense(report, source, target, [SQUARE])


def test_missing_birth_edges_cannot_pass_empty_triangle_scan():
    source = graph_matrix(4, SQUARE)
    target = source.copy()
    target[0, 1] = target[1, 0] = 1.1
    report = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
    assert report['missing_target_birth_edges'] == [[0, 1]]
    assert not report['forbidden_triangles']
    assert not report['certificate_satisfied']


def test_orientation_parity_generators_and_duplicate_obligation_rejection():
    source = graph_matrix(4, SQUARE)
    dirty = [(np.int64(v), np.int64(u)) for u, v in SQUARE] + [(0, 2), (2, 0)]
    report = dual_h1_certificate(source.tolist(), source, (iter(c) for c in [dirty]), 1, 1.2)
    assert report == dual_h1_certificate(source, source, [SQUARE], 1, 1.2)
    for family in ([], [[]], [[(0, 1)]], [SQUARE, dirty]):
        with pytest.raises(ValueError, match='invalid_source_contract'):
            dual_h1_certificate(source, source, family, 1, 1.2)


def test_precision_safe_thresholds_and_zero_distance_edges():
    side = 2**53 + 1
    source = np.full((4, 4), side + 1, dtype=np.int64)
    np.fill_diagonal(source, 0)
    for u, v in SQUARE:
        source[u, v] = source[v, u] = side
    assert dual_h1_certificate(source, source, [SQUARE], np.int64(side), side)['certificate_satisfied']
    with pytest.raises(ValueError, match='missing birth edges'):
        dual_h1_certificate(source, source, [SQUARE], float(side - 1), side)
    source = graph_matrix(4, SQUARE).astype(np.float32)
    source[source == 2] = np.nextafter(np.float32(1), np.float32(2))
    report = dual_h1_certificate(source, source, [SQUARE], 1, 1.0000000894069672)
    assert report['certificate_satisfied']
    source = graph_matrix(4, SQUARE)
    source[source == 1] = 0
    assert dual_h1_certificate(source, source, [SQUARE], 0, 0)['certificate_satisfied']


def test_domain64_and_labels_wider_than_machine_word():
    source = graph_matrix(64, SQUARE)
    assert dual_h1_certificate(source, source, [SQUARE], 1, 1.2)['n_vertices'] == 64
    with pytest.raises(H1BudgetExceeded, match='max_vertices'):
        dual_h1_certificate(graph_matrix(65, SQUARE), graph_matrix(65, SQUARE), [SQUARE], 1, 1.2)
    source = graph_matrix(20, [(u, v) for u in range(10) for v in range(10, 20)])
    cycles = fundamental_cycles(source, 1)
    assert len(cycles) == 81  # K10,10: E - V + 1, no triangle boundaries
    report = dual_h1_certificate(source, source, cycles, 1, 1.2)
    assert report['certificate_satisfied']
    verify_dense(report, source, source, cycles)


@pytest.mark.parametrize('birth,survival', [(2, 1), (-1, 1), (True, 1), (1, np.inf)])
def test_bad_radii(birth, survival):
    source = graph_matrix(4, SQUARE)
    with pytest.raises(ValueError):
        dual_h1_certificate(source, source, [SQUARE], birth, survival)


@pytest.mark.parametrize('which', ['source', 'target'])
@pytest.mark.parametrize('bad', [np.zeros((3, 3)), np.zeros((4, 3)), [[np.nan]],
                                 [[0, 1], [2, 0]], [[0j]], [[True]]])
def test_bad_matrix(which, bad):
    source = graph_matrix(4, SQUARE)
    args = (bad, source) if which == 'source' else (source, bad)
    with pytest.raises(ValueError):
        dual_h1_certificate(*args, [SQUARE], 1, 1.2)


def test_seeded_small_dense_oracle_and_permuted_edge_order():
    rng = np.random.default_rng(7391)
    valid = 0
    source_triangles = 0
    for _ in range(160):
        n = int(rng.integers(4, 9))

        def random_matrix():
            upper = np.triu(rng.choice([1., 1.2, 2.], (n, n), p=[.45, .1, .45]), 1)
            return upper + upper.T

        source, target = random_matrix(), random_matrix()
        candidates = fundamental_cycles(source, 1)
        _, _, B, G = dense_complex(source, candidates, 1.2)
        rows = B.copy()
        rank = dense_rank(rows)
        selected = []
        for candidate, row in zip(candidates, G):
            augmented = np.vstack([rows, row])
            if dense_rank(augmented) > rank:
                selected.append(candidate)
                rows = augmented
                rank += 1
        if not selected:
            if candidates:
                with pytest.raises(ValueError, match='invalid_source_contract'):
                    dual_h1_certificate(source, target, candidates, 1, 1.2)
            continue
        valid += 1
        source_triangles += bool(len(B))
        # Invertible triangular change of selected cycle basis stresses labels.
        family = [sum(selected[:j + 1], []) for j in range(len(selected))]
        for D in (source, target):
            report = dual_h1_certificate(source, D, family, 1, 1.2)
            verify_dense(report, source, D, family)
            primal = assert_primal_dense(source, D, family)
            if report['certificate_satisfied']:
                assert primal.accepted
        permutation = rng.permutation(n)
        inverse = np.argsort(permutation)
        relabeled = [[(int(inverse[u]), int(inverse[v])) for u, v in c] for c in family]
        perm_source = source[np.ix_(permutation, permutation)]
        perm_target = target[np.ix_(permutation, permutation)]
        report = dual_h1_certificate(perm_source, perm_target, relabeled, 1, 1.2)
        verify_dense(report, perm_source, perm_target, relabeled)
        assert_primal_dense(perm_source, perm_target, relabeled)
    assert valid >= 15
    assert source_triangles >= 5


def test_exhaustive_five_vertex_target_graphs_have_no_false_acceptance():
    source = graph_matrix(5, SQUARE)
    edges = list(combinations(range(5), 2))
    certified = false_negatives = 0
    for bits in range(1 << len(edges)):
        target = graph_matrix(5, [edge for j, edge in enumerate(edges) if (bits >> j) & 1])
        report = dual_h1_certificate(source, target, [SQUARE], 1, 1)
        verify_dense(report, source, target, [SQUARE], survival=1)
        primal = assert_primal_dense(source, target, [SQUARE], 1, 1).accepted
        if report['certificate_satisfied']:
            certified += 1
            assert primal
        elif primal:
            false_negatives += 1
    assert certified == 12
    assert false_negatives == 3  # Sufficient, deliberately not necessary.


def test_layout_change_requires_rescan_for_new_forbidden_triangles():
    source = graph_matrix(6, SQUARE)
    target = source.copy()
    target[:4, 4] = target[4, :4] = 1
    initial = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
    assert [t['vertices'] for t in initial['forbidden_triangles']] == [[2, 3, 4]]
    target[2, 4] = target[4, 2] = 2
    assert dual_h1_certificate(source, target, [SQUARE], 1, 1.2)['certificate_satisfied']
    # A subsequent change closes a different cone. The old triangle stays
    # absent, so satisfying a stale batch alone cannot certify this layout.
    target[:4, 5] = target[5, :4] = 1
    rescanned = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
    assert not rescanned['certificate_satisfied']
    assert [t['vertices'] for t in rescanned['forbidden_triangles']] == [[2, 3, 5]]
    assert not check_h1_witnesses(source, target, [SQUARE], 1, 1.2).accepted


def assert_primal_dense(source, target, cycles, birth=1, survival=1.2):
    """Check the primal verifier independently, including certificate failures."""
    _, _, B, G = dense_complex(target, cycles, survival)
    edges = list(combinations(range(len(target)), 2))
    rank = dense_rank(B)
    present = [all(target[u, v] <= birth for (u, v), used in zip(edges, row) if used)
               for row in G]
    survives = [ok and dense_rank(np.vstack([B, row])) > rank
                for ok, row in zip(present, G)]
    joint_rank = dense_rank(np.vstack([B, G[present]])) - rank
    result = check_h1_witnesses(source, target, cycles, birth, survival)
    assert [w.edges_present for w in result.witnesses] == present
    assert [w.survives for w in result.witnesses] == survives
    assert result.surviving_rank == joint_rank
    assert result.accepted == result.all_classes_independent == (joint_rank == len(cycles))
    return result


def test_public_signature_defaults_and_shared_exact_validators():
    parameters = inspect.signature(dual_h1_certificate).parameters
    for name, default in LIMITS.items():
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default == default
    assert structural_dual.H1BudgetExceeded is H1BudgetExceeded
    for name in ('_matrix', '_radius', '_chains', '_integer_budget'):
        assert getattr(structural_dual, name) is getattr(structural_h1, name)


@pytest.mark.parametrize('key', LIMITS)
@pytest.mark.parametrize('value', [-1, True, False, np.bool_(False), 1.0, 1.5,
                                  np.nan, np.inf, None, '64', 1j, np.array(64)])
def test_invalid_budget_types_and_values(key, value):
    source = graph_matrix(4, SQUARE)
    with pytest.raises(ValueError, match=key):
        dual_h1_certificate(source, source, [SQUARE], 1, 1.2, **{key: value})


@pytest.mark.parametrize('key', LIMITS)
def test_zero_budget_raises_instead_of_returning_partial_certificate(key):
    source = graph_matrix(4, SQUARE)
    with pytest.raises(H1BudgetExceeded, match=key):
        dual_h1_certificate(source, source, [SQUARE], 1, 1.2, **{key: 0})


@pytest.mark.parametrize('which', ['source', 'target'])
@pytest.mark.parametrize('bad', [
    None, [], [0], 0, np.zeros((2, 2, 2)), [[0], [1, 2]],
    np.array([[0]], dtype=object), [['0']], [[False]], [[np.inf]], [[-np.inf]],
    [[1]], [[0, -1], [-1, 0]], [[0, 1], [1 + 1e-12, 0]],
])
def test_more_invalid_matrix_shapes_and_types(which, bad):
    source = graph_matrix(4, SQUARE)
    args = (bad, source) if which == 'source' else (source, bad)
    with pytest.raises(ValueError):
        dual_h1_certificate(*args, [SQUARE], 1, 1.2)


@pytest.mark.parametrize('kind', ['nan', 'infinite', 'negative', 'diagonal',
                                  'asymmetric', 'complex', 'bool', 'object', 'str'])
@pytest.mark.parametrize('which', ['source', 'target'])
def test_invalid_matrix_values_on_matching_vertex_domain(kind, which):
    source = graph_matrix(4, SQUARE)
    bad = source.copy()
    if kind in ('nan', 'infinite', 'negative'):
        bad[0, 1] = bad[1, 0] = {'nan': np.nan, 'infinite': np.inf, 'negative': -1}[kind]
    elif kind == 'diagonal':
        bad[0, 0] = np.nextafter(0., 1.)
    elif kind == 'asymmetric':
        bad[0, 1] = np.nextafter(1., 2.)
    else:
        bad = bad.astype({'complex': complex, 'bool': bool, 'object': object, 'str': str}[kind])
    args = (bad, source) if which == 'source' else (source, bad)
    with pytest.raises(ValueError):
        dual_h1_certificate(*args, [SQUARE], 1, 1.2)


@pytest.mark.parametrize('birth,survival', [
    (1, -1), (np.nan, 2), (1, np.nan), (np.inf, np.inf), (0, -np.inf),
    (np.bool_(False), 1), (0, True), (0, np.bool_(True)), ('1', 2),
    (1, 1j), (None, 1), (0, object()), (np.array(1), 2),
])
def test_more_invalid_radii(birth, survival):
    source = graph_matrix(4, SQUARE)
    with pytest.raises(ValueError):
        dual_h1_certificate(source, source, [SQUARE], birth, survival)


@pytest.mark.parametrize('cycles', [None, 2, [None], [2], ['01'], [[(0, 1), (1, 0)]]])
def test_malformed_chain_containers(cycles):
    source = graph_matrix(4, SQUARE)
    with pytest.raises(ValueError):
        dual_h1_certificate(source, source, cycles, 1, 1.2)


@pytest.mark.parametrize('edge', [
    (0, 0), (-1, 2), (0, 4), (0., 1), (False, 1), (0, np.bool_(True)),
    (0, 1, 2), (0,), (), None, 2, ('0', 1), (0, 1j),
])
def test_invalid_edges_rejected_even_if_they_would_cancel(edge):
    source = graph_matrix(4, SQUARE)
    with pytest.raises(ValueError):
        dual_h1_certificate(source, source, [SQUARE + [edge, edge]], 1, 1.2)


@pytest.mark.parametrize('n', [0, 1, 2, 3])
def test_empty_source_contract_is_not_a_vacuous_certificate(n):
    source = graph_matrix(n, [])
    with pytest.raises(ValueError, match='invalid_source_contract'):
        dual_h1_certificate(source, source, [], 1, 1.2)


def test_invalid_source_contract_not_reported_as_target_failure():
    source = graph_matrix(4, SQUARE)
    missing = source.copy()
    missing[0, 1] = missing[1, 0] = np.nextafter(1., 2.)
    with pytest.raises(ValueError, match='invalid_source_contract.*missing'):
        dual_h1_certificate(missing, source, [SQUARE], 1, 1.2)
    with pytest.raises(ValueError, match='invalid_source_contract.*filled'):
        dual_h1_certificate(np.zeros((4, 4)), source, [SQUARE], 1, 1.2)
    with pytest.raises(ValueError, match='invalid_source_contract'):
        dual_h1_certificate(source, source, [SQUARE] * 7, 1, 1.2)


@pytest.mark.parametrize('which', ['source', 'target'])
def test_hard64_ceiling_cannot_be_overridden(which):
    small, large = graph_matrix(64, SQUARE), graph_matrix(65, SQUARE)
    args = (large, small) if which == 'source' else (small, large)
    with pytest.raises(H1BudgetExceeded, match='max_vertices'):
        dual_h1_certificate(*args, [SQUARE], 1, 1.2, max_vertices=1000)
    report = dual_h1_certificate(small, small, [SQUARE], 1, 1.2, max_vertices=np.int64(64))
    assert report['certificate_satisfied'] and report['n_vertices'] == 64


def test_mixed_precision_matrices_and_huge_finite_integer_radius():
    source = graph_matrix(4, SQUARE).astype(np.float32)
    diagonal = np.nextafter(np.float32(1), np.float32(2))
    source[source == 2] = diagonal
    for left, right in ((source, source.astype(np.float64)),
                        (source.astype(np.float64), source)):
        assert dual_h1_certificate(left, right, [SQUARE], 1, 1.0000000894069672)[
            'certificate_satisfied']
        with pytest.raises(ValueError, match='invalid_source_contract'):
            dual_h1_certificate(left, right, [SQUARE], 1, float(diagonal))
        with pytest.raises(ValueError, match='invalid_source_contract.*filled'):
            dual_h1_certificate(left, right, [SQUARE], 10**1000, 10**1000)


def test_wider_float_precision_rejected_not_silently_rounded():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip('platform has no floating type wider than float64')
    source = graph_matrix(4, SQUARE)
    for left, right in ((source.astype(np.longdouble), source),
                        (source, source.astype(np.longdouble))):
        with pytest.raises(ValueError, match='precision'):
            dual_h1_certificate(left, right, [SQUARE], 1, 1.2)
    for birth, survival in ((np.longdouble(1), 1.2), (1, np.longdouble(1.2))):
        with pytest.raises(ValueError, match='precision'):
            dual_h1_certificate(source, source, [SQUARE], birth, survival)


def budget_fixture(kind):
    if kind == 'multiword_labels':
        source = graph_matrix(20, [(u, v) for u in range(10) for v in range(10, 20)])
        return source, np.zeros_like(source), fundamental_cycles(source, 1)
    if kind == 'multiword_edges':
        # Active high-ID edges, not just padding, cross the 64-bit boundary.
        cycle = [(u + 9, v + 9) for u, v in SQUARE]
        source = graph_matrix(13, cycle)
        target = source.copy()
        target[0, 9:] = target[9:, 0] = 1.2
        return source, target, [cycle]
    if kind == 'dense_source':
        # Duplicate each square row three times: many dependent boundaries,
        # but the original square remains nonzero.
        ids = np.tile(np.arange(4), 3)
        source = graph_matrix(4, SQUARE)[np.ix_(ids, ids)]
        return source, source.copy(), [SQUARE]
    n = 16 if kind == 'dense_target' else 4
    source = graph_matrix(n, SQUARE)
    target = np.zeros_like(source) if kind == 'dense_target' else source.copy()
    return source, target, [SQUARE]


@pytest.mark.parametrize('kind', ['square', 'multiword_edges', 'multiword_labels',
                                  'dense_source', 'dense_target'])
def test_exact_reported_budget_boundaries_and_output_lower_bounds(kind):
    source, target, cycles = budget_fixture(kind)
    source_before, target_before = source.copy(), target.copy()
    report = dual_h1_certificate(source, target, cycles, 1, 1.2)
    verify_dense(report, source, target, cycles)
    n, k = len(source), len(cycles)
    E = n * (n - 1) // 2
    words, label_words = max(1, (E + 63) // 64), max(1, (k + 63) // 64)
    # Independent lower bounds: live source labeled rows, complete edge masks,
    # and emitted JSON scalar IDs/pairing cells must not escape accounting.
    rows = report['source_boundary_rank'] + k
    scratch = 6 * (words + label_words)
    assert report['peak_reduction_entries'] >= (
        scratch + rows * (words + label_words) + E * label_words)
    emitted = k * k + k  # Full pairing, plus cochain indices.
    emitted += 2 * sum(len(c['support_edges']) for c in report['cochains'])
    emitted += 2 * (len(report['required_birth_edges']) + len(report['missing_target_birth_edges']))
    emitted += sum(3 + len(t['violated_cocycles']) + 2 * len(t['eligible_source_absent_edges'])
                   for t in report['forbidden_triangles'])
    assert report['peak_reduction_entries'] >= (
        scratch + (E + k) * label_words + emitted)
    assert report['reduction_operations'] >= rows * (words + label_words) + emitted
    assert report['reduction_operations'] % (words + label_words) == 0
    if kind == 'multiword_labels':
        assert k == 81 and label_words == 2
        assert any(i >= 64 for t in report['forbidden_triangles'] for i in t['violated_cocycles'])
    if kind == 'multiword_edges':
        assert words == 2
        edge_ids = {edge: i for i, edge in enumerate(combinations(range(n), 2))}
        assert max(edge_ids[tuple(sorted(e))] for c in cycles for e in c) >= 64
    if kind in ('dense_target', 'multiword_labels'):
        assert report['target_triangle_count'] == n * (n - 1) * (n - 2) // 6
        assert report['forbidden_triangles'] and not report['certificate_satisfied']
    limits = dict(max_vertices=n, max_simplices=report['n_simplices'],
                  max_reduction_operations=report['reduction_operations'],
                  max_reduction_entries=report['peak_reduction_entries'])
    assert dual_h1_certificate(source, target, cycles, 1, 1.2, **limits) == report
    numpy_limits = {key: np.int64(value) for key, value in limits.items()}
    assert dual_h1_certificate(source, target, cycles, 1, 1.2, **numpy_limits) == report
    for key, value in limits.items():
        with pytest.raises(H1BudgetExceeded, match=key):
            dual_h1_certificate(source, target, cycles, 1, 1.2,
                                **dict(limits, **{key: value - 1}))
        # Raising halfway through output/reduction must not damage inputs or
        # contaminate the next invocation with partially accumulated state.
        np.testing.assert_array_equal(source, source_before)
        np.testing.assert_array_equal(target, target_before)
    assert dual_h1_certificate(source, target, cycles, 1, 1.2) == report


@pytest.mark.parametrize('stage', ['source', 'target_vertices_edges', 'target_triangles'])
def test_combined_simplex_preflight_before_budget_or_labeled_insertion(monkeypatch, stage):
    source = graph_matrix(8, SQUARE)
    target = np.zeros((8, 8))
    source_count = 8 + 4
    target_vertices_edges = 8 + 28
    total = source_count + target_vertices_edges + 56
    limit = {'source': source_count - 1,
             'target_vertices_edges': source_count + target_vertices_edges - 1,
             'target_triangles': total - 1}[stage]

    def forbidden(*args, **kwargs):
        pytest.fail('dual reduction/storage began before combined simplex preflight')

    monkeypatch.setattr(structural_dual, '_DualBudget', forbidden)
    monkeypatch.setattr(structural_dual, '_insert_labeled', forbidden)
    with pytest.raises(H1BudgetExceeded, match='max_simplices'):
        dual_h1_certificate(source, target, [SQUARE], 1, 1.2, max_simplices=limit,
                            max_reduction_operations=0, max_reduction_entries=0)


def test_fixed_source_duals_do_not_depend_on_target_or_call_primal_verifier(monkeypatch):
    source = graph_matrix(6, SQUARE)
    targets = [source.copy(), np.zeros_like(source)]
    attached = source.copy()
    attached[2, 4] = attached[4, 2] = attached[3, 4] = attached[4, 3] = 1
    targets.append(attached)
    missing = source.copy()
    missing[0, 1] = missing[1, 0] = 2
    targets.append(missing)
    # This verifier result is deliberately independent of dual satisfaction.
    assert check_h1_witnesses(source, attached, [SQUARE], 1, 1.2).accepted

    def forbidden(*args, **kwargs):
        pytest.fail('dual construction must not use target primal verification')

    monkeypatch.setattr(structural_h1, 'check_h1_witnesses', forbidden)
    if hasattr(structural_dual, 'check_h1_witnesses'):
        monkeypatch.setattr(structural_dual, 'check_h1_witnesses', forbidden)
    fixed = ('cochains', 'source_pairing', 'source_boundary_rank',
             'source_triangle_count', 'required_birth_edges', 'n_cocycles')
    baseline = None
    outcomes = []
    for target in targets:
        target_before = target.copy()
        report = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
        verify_dense(report, source, target, [SQUARE])
        snapshot = {key: report[key] for key in fixed}
        if baseline is None:
            baseline = snapshot
        assert snapshot == baseline
        outcomes.append(report['certificate_satisfied'])
        np.testing.assert_array_equal(target, target_before)
    assert outcomes == [True, False, False, False]


def test_primal_verifier_is_independent_of_dual_certificate(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('primal verifier must not delegate to a merely sufficient dual certificate')

    monkeypatch.setattr(structural_dual, 'dual_h1_certificate', forbidden)
    if hasattr(structural_h1, 'dual_h1_certificate'):
        monkeypatch.setattr(structural_h1, 'dual_h1_certificate', forbidden)
    source = graph_matrix(5, SQUARE)
    attached = source.copy()
    attached[2, 4] = attached[4, 2] = attached[3, 4] = attached[4, 3] = 1
    assert assert_primal_dense(source, attached, [SQUARE]).accepted
    cone = source.copy()
    cone[:4, 4] = cone[4, :4] = 1.2
    assert not assert_primal_dense(source, cone, [SQUARE]).accepted
    cycles = [SQUARE, [(u + 4, v + 4) for u, v in SQUARE]]
    source = graph_matrix(8, cycles[0] + cycles[1])
    target = source.copy()
    for u in range(4):
        for v in (u + 4, (u + 1) % 4 + 4):
            target[u, v] = target[v, u] = 1.2
    result = assert_primal_dense(source, target, cycles)
    assert all(w.survives for w in result.witnesses) and result.surviving_rank == 1
    target = source.copy()
    target[0, 1] = target[1, 0] = 1.1
    result = assert_primal_dense(source, target, cycles)
    assert [w.edges_present for w in result.witnesses] == [False, True]
    assert result.surviving_rank == 1 and not result.accepted


def test_source_triangle_simplex_preflight_precedes_any_budget_allocation(monkeypatch):
    source, target, cycles = budget_fixture('dense_source')
    _, triangles, _, _ = dense_complex(source, cycles, 1.2)
    assert len(triangles) > 1
    source_count = len(source) + len(triangles) + sum(
        source[u, v] <= 1.2 for u, v in combinations(range(len(source)), 2))

    def forbidden(*args, **kwargs):
        pytest.fail('source triangle preflight must complete before dual allocation')

    monkeypatch.setattr(structural_dual, '_DualBudget', forbidden)
    monkeypatch.setattr(structural_dual, '_insert_labeled', forbidden)
    with pytest.raises(H1BudgetExceeded, match='max_simplices'):
        dual_h1_certificate(source, target, cycles, 1, 1.2,
                            max_simplices=int(source_count) - 1,
                            max_reduction_operations=0, max_reduction_entries=0)


@pytest.mark.parametrize('kind', ['dense_target', 'multiword_labels'])
def test_work_and_storage_include_later_dense_target_scan_and_output(kind):
    source, target, cycles = budget_fixture(kind)
    intact = dual_h1_certificate(source, source, cycles, 1, 1.2)
    dense = dual_h1_certificate(source, target, cycles, 1, 1.2)
    assert intact['certificate_satisfied'] and not dense['certificate_satisfied']
    assert dense['cochains'] == intact['cochains']
    # Holding the source reduction fixed must still charge the additional scan
    # and retained target obstruction payload, cumulatively rather than reset.
    extra_triangles = dense['target_triangle_count'] - intact['target_triangle_count']
    label_words = max(1, (len(cycles) + 63) // 64)
    assert dense['reduction_operations'] >= intact['reduction_operations'] + extra_triangles * label_words
    assert dense['peak_reduction_entries'] > intact['peak_reduction_entries']
    for key, field in (('max_reduction_operations', 'reduction_operations'),
                       ('max_reduction_entries', 'peak_reduction_entries')):
        with pytest.raises(H1BudgetExceeded, match=key):
            dual_h1_certificate(source, target, cycles, 1, 1.2, **{key: intact[field]})


def test_target_birth_and_survival_membership_use_exact_closed_boundaries():
    source = graph_matrix(5, SQUARE)
    target = source.copy()
    target[0, 1] = target[1, 0] = np.nextafter(1., np.inf)
    report = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
    assert report['missing_target_birth_edges'] == [[0, 1]]
    assert not report['certificate_satisfied']
    target[0, 1] = target[1, 0] = 1
    target[:4, 4] = target[4, :4] = 1.2
    at = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
    assert at['target_triangle_count'] == 4 and not at['certificate_satisfied']
    target[:4, 4] = target[4, :4] = np.nextafter(1.2, np.inf)
    after = dual_h1_certificate(source, target, [SQUARE], 1, 1.2)
    assert after['target_triangle_count'] == 0 and after['certificate_satisfied']
    assert after['cochains'] == at['cochains']


def test_readonly_matrices_and_mutable_chain_inputs_are_not_modified():
    source = graph_matrix(5, SQUARE)
    target = source.copy()
    target[:4, 4] = target[4, :4] = 1.2
    source_before, target_before = source.copy(), target.copy()
    source.flags.writeable = target.flags.writeable = False
    chains = [[list(e) for e in SQUARE] + [[0, 2], [2, 0]]]
    chains_before = json.loads(json.dumps(chains))
    report = dual_h1_certificate(source, target, chains, 1, 1.2)
    verify_dense(report, source, target, [SQUARE])
    assert chains == chains_before
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(target, target_before)


def test_square_hand_counted_work_and_peak_storage_include_output_emission():
    source = graph_matrix(4, SQUARE)
    report = dual_h1_certificate(source, source, [SQUARE], 1, 1.2)
    # W=We+Wk=2. Required init, vector, union/label, insertion, mask init,
    # substitution (three lower bits), pairing check, expected-label comparison,
    # pairing extraction AND emission, cochain output, required-edge output.
    work_units = [1, 9, 2, 2, 6, 21, 6, 2, 3, 15, 20]
    assert report['reduction_operations'] == 2 * sum(work_units) == 174
    # Output peak: six scratch records, six masks, one pairing mask, one
    # pairing cell, one cochain index, one support edge and four required edges.
    # The earlier source peak (12+2+6=20) is lower; its basis has been released.
    assert report['peak_reduction_entries'] == 12 + 6 + 1 + 1 + 1 + 2 + 8 == 31
    assert report['n_simplices'] == 16
    for key, value in (('max_reduction_operations', 173), ('max_reduction_entries', 30)):
        with pytest.raises(H1BudgetExceeded, match=key):
            dual_h1_certificate(source, source, [SQUARE], 1, 1.2, **{key: value})


@pytest.mark.parametrize('obligation', ['source triangle parity', 'source cycle pairing'])
def test_explicit_original_obligations_checked_after_labeled_elimination(monkeypatch, obligation):
    source = graph_matrix(4, SQUARE)
    if obligation == 'source triangle parity':
        # Duplicate row zero introduces triangles through (0,4), outside the
        # requested square's edges. Corrupting that mask preserves its pairing.
        ids = [0, 1, 2, 3, 0]
        source = source[np.ix_(ids, ids)]
        corrupt_edge = (0, 4)
    else:
        corrupt_edge = (2, 3)
    original = structural_dual._source_masks

    def corrupt(adjacency, chains, edge_ids, budget):
        masks, rank, required = original(adjacency, chains, edge_ids, budget)
        masks[edge_ids[corrupt_edge]] ^= 1
        return masks, rank, required

    monkeypatch.setattr(structural_dual, '_source_masks', corrupt)
    with pytest.raises(RuntimeError, match=obligation):
        dual_h1_certificate(source, source, [SQUARE], 1, 1.2)
