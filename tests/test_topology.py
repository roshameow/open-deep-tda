"""Analytic/oracle checks; native-only cases skip when the extension is absent."""

import importlib
import itertools

import numpy as np
import pytest

from open_deep_tda import topology


@pytest.fixture
def native():
    pytest.importorskip("open_deep_tda._core", reason="Native C++ extension not built")


def square():
    return np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


def test_lazy_native_error(monkeypatch):
    original = importlib.import_module
    def unavailable(name, *args, **kwargs):
        if name == "open_deep_tda._core":
            raise ImportError("not built")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", unavailable)
    np.testing.assert_array_equal(topology.distance_matrix([[0], [2]]), [[0, 2], [2, 0]])
    assert topology.diagram_matching([], [])['cost'] == 0
    with pytest.raises(ImportError, match="compiled.*_core"):
        topology.persistence(np.zeros((1, 1)))


@pytest.mark.parametrize("X", [[], [1, 2], [[[0]]], [[np.nan]], [[np.inf]], [[1j]], [["1"]]])
def test_distance_matrix_bad_input(X):
    with pytest.raises(ValueError):
        topology.distance_matrix(X)


def test_distance_matrix_shape_precision_and_duplicates():
    assert topology.distance_matrix(np.empty((0, 3))).shape == (0, 0)
    np.testing.assert_array_equal(topology.distance_matrix(np.empty((3, 0))), np.zeros((3, 3)))
    np.testing.assert_array_equal(topology.distance_matrix([[2, 3]]), [[0]])
    X = np.array([[1e10, 1e10], [1e10 + 3, 1e10 + 4], [1e10, 1e10]])
    D = topology.distance_matrix(X)
    assert D.dtype == np.float64 and D.flags.c_contiguous
    np.testing.assert_array_equal(D, [[0, 5, 0], [5, 0, 5], [0, 5, 0]])


@pytest.mark.parametrize("D", [np.ones((2, 3)), [[1]], [[0, -1], [-1, 0]],
                                    [[0, 1], [2, 0]], [[0, np.inf], [np.inf, 0]],
                                    [[np.nan]], [[0j]]])
def test_invalid_matrix_before_native(D):
    with pytest.raises(ValueError):
        topology.persistence(D)
    with pytest.raises(ValueError):
        topology.mst(D)


@pytest.mark.parametrize("budgets", [{"max_simplices": np.inf}, {"max_simplices": -1},
    {"max_reduction_entries": 1.5}, {"max_reduction_operations": True},
    {"max_radius": np.nan}, {"max_radius": -np.inf}, {"max_radius": -1}, {"bogus": 3}])
def test_invalid_budgets_before_native(budgets):
    with pytest.raises(ValueError):
        topology.persistence(np.zeros((1, 1)), **budgets)


def test_diagram_extraction_and_censoring():
    def pair(b, d, essential=False, censored=False, dim=1):
        return dict(birth=b, death=d, essential=essential, censored=censored, dimension=dim)
    result = {"pairs": [pair(1, 2), pair(2, 2), pair(0, np.inf, True, dim=0)]}
    np.testing.assert_array_equal(topology.diagram(result), [[1, 2]])
    np.testing.assert_array_equal(topology.diagram(result, positive_only=False), [[1, 2], [2, 2]])
    assert topology.diagram(result, dimension=0).shape == (0, 2)
    assert np.isinf(topology.diagram(result, dimension=0, include_essential=True)[0, 1])
    with pytest.raises(ValueError, match="Censored"):
        topology.diagram({"pairs": [pair(1, np.inf, censored=True)]})
    with pytest.raises(ValueError):
        topology.diagram({"pairs": [pair(2, 1)]})


def test_exact_diagonal_cost_and_shapes():
    result = topology.diagram_matching([[1, 3]], [])
    assert result["cost"] == 2
    assert result["matched"].shape == (0, 2)
    np.testing.assert_array_equal(result["unmatched_source"], [0])
    assert result["unmatched_target"].shape == (0,)
    reverse = topology.diagram_matching([], [[1, 3]])
    assert reverse["cost"] == 2
    np.testing.assert_array_equal(reverse["unmatched_target"], [0])
    empty = topology.diagram_matching([], [], max_matching_size=0)
    assert empty["cost"] == 0 and empty["matched"].dtype.kind == "i"
    # Real-to-real cost is huge: neither bar may be forced to the other.
    apart = topology.diagram_matching([[1, 2]], [[10, 11]])
    assert apart["cost"] == 1
    assert apart["matched"].shape == (0, 2)


def test_matching_mixed_and_permutation():
    P, Q = np.array([[1, 4], [20, 21]]), np.array([[1.1, 3.9], [40, 42]])
    result = topology.diagram_matching(P, Q)
    np.testing.assert_array_equal(result["matched"], [[0, 0]])
    np.testing.assert_array_equal(result["unmatched_source"], [1])
    np.testing.assert_array_equal(result["unmatched_target"], [1])
    assert result["cost"] == pytest.approx(.02 + .5 + 2)
    assert topology.diagram_matching(Q[::-1], P[::-1])["cost"] == pytest.approx(result["cost"])
    assert topology.diagram_matching(3 * P, 3 * Q)["cost"] == pytest.approx(9 * result["cost"])
    assert topology.diagram_matching(P, P)["cost"] == 0


def brute_partial_cost(P, Q):
    """Independent enumeration, not a second augmented assignment solver."""
    best = float("inf")
    for k in range(min(len(P), len(Q)) + 1):
        for source in itertools.combinations(range(len(P)), k):
            for target in itertools.permutations(range(len(Q)), k):
                cost = sum(float(np.sum((P[i] - Q[j]) ** 2)) for i, j in zip(source, target))
                cost += sum(float((p[1] - p[0]) ** 2 / 2) for i, p in enumerate(P) if i not in source)
                cost += sum(float((q[1] - q[0]) ** 2 / 2) for j, q in enumerate(Q) if j not in target)
                best = min(best, cost)
    return best


def test_augmented_assignment_against_all_partial_matchings():
    rng = np.random.default_rng(32)
    for m, n in itertools.product(range(4), repeat=2):
        for _ in range(3):
            P = np.sort(rng.uniform(0, 8, size=(m, 2)), axis=1)
            Q = np.sort(rng.uniform(0, 8, size=(n, 2)), axis=1)
            result = topology.diagram_matching(P, Q)
            assert result["cost"] == pytest.approx(brute_partial_cost(P, Q))
            assert len(result["matched"]) + len(result["unmatched_source"]) == m
            assert len(result["matched"]) + len(result["unmatched_target"]) == n


@pytest.mark.parametrize("P", [[[0, np.inf]], [[np.nan, 2]], [[2, 1]], [[1, 2, 3]], [1, 2], [[1j, 2]]])
def test_matching_invalid_diagrams(P):
    with pytest.raises(ValueError):
        topology.diagram_matching(P, [])


def test_matching_budget_and_overflow():
    with pytest.raises(RuntimeError, match="max_matching_size"):
        topology.diagram_matching([[1, 2]], [[1, 2]], max_matching_size=1)
    for budget in [np.inf, -1, 1.5, True]:
        with pytest.raises(ValueError):
            topology.diagram_matching([], [], max_matching_size=budget)
    with pytest.raises(ValueError, match="overflow"):
        topology.diagram_matching([[0, 1e300]], [])


@pytest.mark.parametrize("X,h0", [(np.empty((0, 2)), []), ([[0, 0]], []),
    ([[0, 0], [0, 0]], [0]), ([[0], [1], [4]], [1, 3])])
def test_small_clouds(native, X, h0):
    D = topology.distance_matrix(X)
    result = topology.persistence(D)
    assert result["n_vertices"] == len(X)
    bars = topology.diagram(result, dimension=0, positive_only=False)
    np.testing.assert_allclose(np.sort(bars[:, 1]), h0)
    assert topology.diagram(result).shape == (0, 2)
    assert topology.mst(D).shape == (max(0, len(X) - 1), 2)
    essential = [p for p in result["pairs"] if p["essential"]]
    assert len(essential) == int(len(X) > 0)
    assert not result["truncated"]


def test_equilateral_is_rips_not_cech(native):
    D = np.ones((3, 3)) - np.eye(3)
    result = topology.persistence(D)
    assert result["n_simplices"] == 7
    assert topology.diagram(result).shape == (0, 2)
    np.testing.assert_allclose(topology.diagram(result, positive_only=False), [[1, 1]])


def test_square_critical_edges_zero_bars_and_determinism(native):
    D = topology.distance_matrix(square())
    result = topology.persistence(D)
    np.testing.assert_allclose(topology.diagram(result), [[1, np.sqrt(2)]])
    assert result == topology.persistence(D)
    assert result["n_simplices"] == 4 + 6 + 4
    assert {p["dimension"] for p in result["pairs"]} == {0, 1}
    for p in result["pairs"]:
        if p["dimension"] == 1 and not p["essential"]:
            assert D[tuple(p["birth_edge"])] == p["birth"]
            assert D[tuple(p["death_edge"])] == p["death"]
            tri_edges = list(itertools.combinations(p["death_simplex"], 2))
            longest = sorted(tri_edges, key=lambda e: (-D[e], e))[0]
            assert tuple(p["death_edge"]) == longest
    np.testing.assert_array_equal(topology.mst(D), topology.mst(D))


def test_h0_matches_mst_and_geometric_invariance(native):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(9, 3))
    D = topology.distance_matrix(X)
    result = topology.persistence(D)
    edges = topology.mst(D)
    np.testing.assert_allclose(np.sort(topology.diagram(result, dimension=0)[:, 1]),
                               np.sort(D[edges[:, 0], edges[:, 1]]))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    for Y, scale in [(X @ rotation + 13, 1), (X[rng.permutation(len(X))], 1), (X * 2.7, 2.7)]:
        transformed = topology.persistence(topology.distance_matrix(Y))
        for dim in [0, 1]:
            P = topology.diagram(result, dimension=dim)
            Q = topology.diagram(transformed, dimension=dim)
            assert topology.diagram_matching(scale * P, Q)["cost"] == pytest.approx(0, abs=1e-20)


def test_native_budgets_and_censoring(native):
    D = topology.distance_matrix(square())
    for options in [{"max_simplices": 1}, {"max_reduction_entries": 0}, {"max_reduction_operations": 0}]:
        with pytest.raises(RuntimeError):
            topology.persistence(D, **options)
    result = topology.persistence(D, max_radius=1.1)
    assert result["truncated"]
    assert any(p["dimension"] == 1 and p["censored"] for p in result["pairs"])
    with pytest.raises(ValueError, match="Censored"):
        topology.diagram(result)


def test_optional_ripser_oracle(native):
    ripser = pytest.importorskip("ripser")
    rng = np.random.default_rng(42)
    for X in [square(), rng.normal(size=(12, 3)), rng.normal(size=(10, 2))]:
        D = topology.distance_matrix(X)
        ours = topology.persistence(D)
        oracle = ripser.ripser(D, distance_matrix=True, maxdim=1)["dgms"]
        for dim in [0, 1]:
            expected = oracle[dim]
            expected = expected[np.isfinite(expected[:, 1]) & (expected[:, 1] > expected[:, 0])]
            actual = topology.diagram(ours, dimension=dim)
            assert len(actual) == len(expected)
            # Ripser internally uses single precision even for double input.
            assert topology.diagram_matching(actual, expected)["cost"] < 1e-10
