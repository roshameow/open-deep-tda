"""Synthetic structural-H0 checks, including an independent threshold oracle.

Regression pitfalls: equal sorted bars do not imply same-ID merges; comparing
original distances on MST edges can reject equal hierarchies. The threshold
oracle intentionally uses neither MSTs nor tree-path maxima.
"""

import json
from fractions import Fraction

import numpy as np
import pytest

from open_deep_tda import structural_h0, topology
from open_deep_tda.structural_h0 import compare_h0


@pytest.fixture
def native():
    pytest.importorskip("open_deep_tda._core", reason="Native C++ extension not built")


def line(values):
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - values[None, :])


def threshold_partition(D, threshold):
    """Full-graph connected components at a threshold, with minimum-ID labels."""
    labels = [-1] * len(D)
    for root in range(len(D)):
        if labels[root] != -1:
            continue
        labels[root] = root
        stack = [root]
        while stack:
            i = stack.pop()
            for j in range(len(D)):
                if labels[j] == -1 and D[i, j] <= threshold:
                    labels[j] = root
                    stack.append(j)
    return labels


def threshold_merges(D):
    merges = np.full(D.shape, np.inf)
    np.fill_diagonal(merges, 0)
    for threshold in np.unique(D):
        labels = threshold_partition(D, threshold)
        for i in range(len(D)):
            for j in range(i + 1, len(D)):
                if labels[i] == labels[j] and np.isinf(merges[i, j]):
                    merges[i, j] = merges[j, i] = threshold
    return merges


def check_certificate(result, source, target, tolerance):
    n = len(source)
    assert result["n_vertices"] == n
    assert result["scope"] == (
        "complete supplied vertex domain; no assertion about omitted points"
    )
    assert result["criterion"] == (
        "max_merge_error <= tolerance (same-ID minimax merge distances)"
    )
    assert result["tolerance"] == tolerance
    # Strict JSON: no NumPy scalars/arrays, NaNs, or infinities.
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    source_merges, target_merges = threshold_merges(source), threshold_merges(target)
    errors = np.abs(source_merges - target_merges)
    expected = float(errors.max()) if n else 0.0
    assert result["max_merge_error"] == expected
    assert result["certified_within_tolerance"] is bool(expected <= tolerance)
    witness = result["max_merge_error_witness"]
    if n < 2:
        assert witness is None
    else:
        pair = next([i, j] for i in range(n) for j in range(i + 1, n)
                    if errors[i, j] == expected)
        assert witness == {
            "pair": pair,
            "source_merge_distance": source_merges[tuple(pair)],
            "target_merge_distance": target_merges[tuple(pair)],
            "absolute_error": expected,
        }
    union = set()
    for name, D in (("source_tree", source), ("target_tree", target)):
        tree = result[name]
        assert len(tree["edges"]) == len(tree["weights"]) == max(n - 1, 0)
        tree_distances = np.full(D.shape, np.inf)
        np.fill_diagonal(tree_distances, 0)
        for (i, j), weight in zip(tree["edges"], tree["weights"]):
            assert 0 <= i < j < n
            assert weight == D[i, j]
            union.add((i, j))
            tree_distances[i, j] = tree_distances[j, i] = weight
        # Verify connectivity and full threshold partitions, not just edge count.
        for threshold in np.unique(D):
            assert threshold_partition(tree_distances, threshold) == threshold_partition(
                D, threshold
            )
    bound = max((abs(float(source[i, j]) - float(target[i, j])) for i, j in union),
                default=0.0)
    assert result["max_tree_edge_error"] == bound
    assert result["tree_edge_bound_within_tolerance"] is bool(bound <= tolerance)
    assert expected <= bound


def test_local_pairs_miss_global_bridge(native):
    source, target = line([0, 1, 10, 11]), line([0, 1, 100, 101])
    for ids in ([0, 1], [2, 3]):
        local = compare_h0(source[np.ix_(ids, ids)], target[np.ix_(ids, ids)])
        assert local["certified_within_tolerance"]
        assert local["max_merge_error"] == local["max_tree_edge_error"] == 0
    result = compare_h0(source, target)
    check_certificate(result, source, target, 0)
    assert not result["certified_within_tolerance"]
    assert result["max_merge_error"] == result["max_tree_edge_error"] == 90
    assert result["max_merge_error_witness"] == {
        "pair": [0, 2], "source_merge_distance": 9.0,
        "target_merge_distance": 99.0, "absolute_error": 90.0,
    }
    for tolerance in (np.nextafter(90.0, 0), 90.0, 91.0):
        result = compare_h0(source, target, tolerance=tolerance)
        check_certificate(result, source, target, tolerance)


def test_equal_sorted_bars_wrong_ids_rejected(native):
    source = line([0, 1, 3])
    permutation = [2, 1, 0]
    target = source[np.ix_(permutation, permutation)]
    result = compare_h0(source, target)
    assert sorted(result["source_tree"]["weights"]) == sorted(result["target_tree"]["weights"])
    assert result["max_merge_error"] == 1
    assert not result["certified_within_tolerance"]
    check_certificate(result, source, target, 0)


def test_equal_hierarchies_can_have_nonzero_tree_edge_bound(native):
    source = line([0, 1, 2])
    target = np.ones((3, 3)) - np.eye(3)
    for a, b in ((source, target), (target, source)):
        result = compare_h0(a, b)
        assert result["max_tree_edge_error"] == 1
        assert not result["tree_edge_bound_within_tolerance"]
        assert result["max_merge_error"] == 0
        assert result["certified_within_tolerance"]
        check_certificate(result, a, b, 0)


@pytest.mark.parametrize("seed", range(5))
def test_random_full_threshold_partition_oracle(native, seed):
    rng = np.random.default_rng(seed)
    for n in range(2, 9):
        # Discrete dissimilarities deliberately include ties and zero edges.
        matrices = []
        for _ in range(2):
            upper = np.triu(rng.integers(0, 6, size=(n, n)), k=1)
            matrices.append((upper + upper.T).astype(float))
        # Integer-coordinate distances additionally exercise duplicate points.
        cases = [matrices, [line(rng.integers(0, 5, n)), line(rng.integers(0, 5, n))]]
        for source, target in cases:
            for tolerance in (0.0, 1.0, 3.0):
                result = compare_h0(source, target, tolerance=tolerance, max_vertices=n)
                check_certificate(result, source, target, tolerance)
            for threshold in np.unique(np.concatenate((source.ravel(), target.ravel()))):
                if threshold_partition(source, threshold) != threshold_partition(target, threshold):
                    assert compare_h0(source, target)["max_merge_error"] > 0
                    break
            else:
                assert compare_h0(source, target)["max_merge_error"] == 0


def test_duplicates_all_zero_and_float_inputs(native):
    for source, target in (
        (line([0, 0, 1, 1]), line([0, 1, 0, 1])),
        (np.zeros((4, 4)), np.zeros((4, 4))),
        (np.zeros((2, 2)), line([0, 1e-12])),
        (line([0, 0.125, 2.5]), line([0, 0.25, 2.75])),
        (line([0, np.finfo(float).max]), np.zeros((2, 2))),
    ):
        result = compare_h0(source, target)
        check_certificate(result, source, target, 0)


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_no_backend_needed(monkeypatch, n):
    def forbidden(*args, **kwargs):
        pytest.fail("Degenerate matrices do not need a native MST")
    monkeypatch.setattr(topology, "mst", forbidden)
    D = np.zeros((n, n))
    result = compare_h0(D, D, max_vertices=n)
    check_certificate(result, D, D, 0)
    assert result["certified_within_tolerance"]
    assert result["tree_edge_bound_within_tolerance"]


@pytest.mark.parametrize("tolerance", [True, np.bool_(False), -1, np.nan, np.inf,
                                       -np.inf, 1j, "0", None, [0], np.array(0), 10**1000,
                                       Fraction(-1, 10**1000)])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        compare_h0(np.zeros((0, 0)), np.zeros((0, 0)), tolerance=tolerance)


@pytest.mark.parametrize("budget", [True, np.bool_(True), -1, 1.5, 2.0, np.inf,
                                    np.nan, "2", None])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match="max_vertices"):
        compare_h0(np.zeros((0, 0)), np.zeros((0, 0)), max_vertices=budget)


@pytest.mark.parametrize("bad", [np.ones((2, 3)), np.zeros(2), 3, [], [[0], [0, 0]],
                                 [[[0]]], [[1]], [[0, -1], [-1, 0]],
                                 [[0, 1], [2, 0]], [[np.nan]], [[np.inf]],
                                 [[0, np.inf], [np.inf, 0]], [[0j]], [["0"]]])
def test_invalid_distances_no_backend(monkeypatch, bad):
    def forbidden(*args, **kwargs):
        pytest.fail("Validation must precede native computation")
    monkeypatch.setattr(topology, "mst", forbidden)
    with pytest.raises(ValueError):
        compare_h0(bad, bad)
    with pytest.raises(ValueError):
        compare_h0(np.zeros((1, 1)), bad)


def test_shapes_checked_before_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Shape mismatch must precede distance conversion")
    monkeypatch.setattr(topology, "_distance_array", forbidden)
    with pytest.raises(ValueError, match="equal shape"):
        compare_h0(np.zeros((1, 1)), np.zeros((2, 2)))


@pytest.mark.parametrize("oversized_source", [True, False])
@pytest.mark.parametrize("representation", ["array", "list", "shape_only", "default"])
def test_budget_precedes_square_conversions(monkeypatch, oversized_source, representation):
    class ShapeOnly:
        def __init__(self, n):
            self.shape = (n, n)

        def __array__(self, *args, **kwargs):
            pytest.fail("Oversized input must not be converted")

    budget = 2048 if representation == "default" else 2
    if representation == "array":
        large = np.zeros((3, 3), dtype=np.float32).T
    elif representation == "list":
        large = [[0] * 3 for _ in range(3)]
    else:
        large = ShapeOnly(budget + 1)

    def forbidden(*args, **kwargs):
        pytest.fail("Both budgets must be checked before validation/MST")
    monkeypatch.setattr(topology, "_distance_array", forbidden)
    monkeypatch.setattr(topology, "mst", forbidden)
    small = np.zeros((1, 1))
    a, b = (large, small) if oversized_source else (small, large)
    options = {} if representation == "default" else {"max_vertices": budget}
    with pytest.raises(RuntimeError, match="max_vertices"):
        compare_h0(a, b, **options)


def test_exact_budget_boundary_and_input_immutability(native):
    source = line([0, 1, 4]).astype(np.float32).T
    target = line([0, 2, 4]).astype(np.int64)
    before_source, before_target = source.copy(), target.copy()
    source.flags.writeable = target.flags.writeable = False
    result = compare_h0(source, target, tolerance=np.float64(1), max_vertices=np.int64(3))
    check_certificate(result, source, target, 1)
    assert compare_h0(source.tolist(), target.tolist(), tolerance=1) == result
    np.testing.assert_array_equal(source, before_source)
    np.testing.assert_array_equal(target, before_target)
    with pytest.raises(RuntimeError, match="max_vertices"):
        compare_h0(source, target, max_vertices=2)
    with pytest.raises(RuntimeError, match="max_vertices"):
        compare_h0(np.zeros((1, 1)), np.zeros((1, 1)), max_vertices=0)


@pytest.mark.parametrize("failed_call", [1, 2])
def test_native_failure_propagates_without_partial_success(monkeypatch, failed_call):
    calls = []

    def fail(D):
        calls.append(D)
        if len(calls) == failed_call:
            raise RuntimeError("native MST failure")
        return np.array([[0, 1]])

    monkeypatch.setattr(structural_h0.topology, "mst", fail)
    with pytest.raises(RuntimeError, match="native MST failure"):
        compare_h0(line([0, 1]), line([0, 2]))
    assert len(calls) == failed_call
