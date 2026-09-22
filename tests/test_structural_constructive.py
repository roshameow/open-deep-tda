"""Independent constructor regressions; synthetic, offline, and full-domain.

Run explicitly: python -m pytest -q tests/test_structural_constructive.py
The acceptance oracle uses scalar Floyd minimax and dense uint8 F2 elimination,
not the constructor's MST or sparse reducer. Verified methods/pitfalls retained
here: an external vertex can fill a cycle; minimax closure must precede restriction
to cycle rows; a deliberately injected H0 excess smaller than 1e-10 must still
fail (test_final_h0_rejects_tiny_excess_without_proposal_slack). Full source
validation must also precede unsupported-family rejection, or malformed indices
in an unsupported family are silently masked. ResourceLimitError's ValueError
base is an explicit requested contract, not an inferred implementation detail.
No production code or benchmark artifacts are modified by these tests.

Promotion verification checkpoint: the complete new public structural test set
passed on Python 3.9.16 and an isolated Python 3.13.2 environment (source imports
via PYTHONPATH=python): 365 passed, one optional real-Ripser test skipped because
Ripser was absent, with its opt-in explicitly enabled. Existing native CTest:
2/2 passed. The fixed 12-row public example certified in both environments.
This is not a wheel/distribution or whole-repository test claim.

Verified cross-version pitfall: BLAS/LAPACK can return nonfinite Procrustes
results without honoring np.errstate. The constructor checks derived covariance
and factors explicitly and returns no embedding on this unsupported numerical
case. Regressions below simulate each nonfinite factor without relying on a
particular BLAS implementation's exception behavior.
"""
import copy
import importlib
from itertools import combinations
import random

import numpy as np
import pytest


@pytest.fixture(scope="module")
def layout():
    # Import at fixture setup so standalone independent-oracle tests remain runnable
    # without importing a private draft. A missing implementation FAILS;
    # it is deliberately not silently converted to a skip.
    return importlib.import_module("open_deep_tda.structural_constructive")


def distances(points):
    points = np.asarray(points, dtype=np.float64)
    return np.hypot(points[:, None, 0] - points[None, :, 0],
                    points[:, None, 1] - points[None, :, 1])


def floyd_minimax(matrix):
    """Independent all-pairs bottleneck paths, including every input row."""
    result = np.array(matrix, dtype=np.float64, copy=True)
    for k in range(len(result)):
        for i in range(len(result)):
            for j in range(len(result)):
                result[i, j] = min(result[i, j], max(result[i, k], result[k, j]))
    return result


def dense_rank(columns, rows):
    if not columns:
        return 0
    matrix = np.asarray(columns, dtype=np.uint8).T.copy()
    rank = 0
    for column in range(matrix.shape[1]):
        candidates = np.flatnonzero(matrix[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(matrix.shape[0]):
            if row != rank and matrix[row, column]:
                matrix[row] ^= matrix[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def dense_h1(matrix, cycles, a, b):
    """Selected chains independent modulo ALL triangle boundaries at b."""
    n = len(matrix)
    edges = list(combinations(range(n), 2))
    index = {edge: row for row, edge in enumerate(edges)}
    boundaries = []
    for i, j, k in combinations(range(n), 3):
        if max(matrix[i, j], matrix[i, k], matrix[j, k]) <= b:
            column = np.zeros(len(edges), dtype=np.uint8)
            for edge in ((i, j), (i, k), (j, k)):
                column[index[edge]] = 1
            boundaries.append(column)
    witnesses = []
    for cycle in cycles:
        column = np.zeros(len(edges), dtype=np.uint8)
        for i, j in cycle:
            if matrix[i, j] > a:
                return False
            column[index[tuple(sorted((i, j)))]] ^= 1
        boundary = np.zeros(n, dtype=np.uint8)
        for present, (i, j) in zip(column, edges):
            if present:
                boundary[i] ^= 1
                boundary[j] ^= 1
        if boundary.any():
            return False
        witnesses.append(column)
    return bool(witnesses) and (
        dense_rank(boundaries + witnesses, len(edges))
        - dense_rank(boundaries, len(edges)) == len(witnesses)
    )


def square(extra=False):
    guide = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]], dtype=np.float64)
    if extra:
        guide = np.vstack((guide, [8., 7.]))
    cycles = [[(0, 1), (1, 2), (2, 3), (3, 0)]]
    return distances(guide), guide, cycles


def assert_failure(result):
    assert not result.certified
    assert result.embedding is None, "failure must never expose a partial candidate"
    assert isinstance(result.reason, str) and result.reason
    assert isinstance(result.diagnostics, dict)


def assert_certificate(result, matrix, cycles, a=2.01, b=2.1, tolerance=.05):
    assert result.certified, result.reason
    assert isinstance(result.reason, str)
    assert isinstance(result.diagnostics, dict)
    assert isinstance(result.embedding, np.ndarray)
    assert result.embedding.shape == (len(matrix), 2)
    assert np.isfinite(result.embedding).all()
    target = distances(result.embedding)
    error = float(np.max(np.abs(floyd_minimax(matrix) - floyd_minimax(target))))
    # This is intentionally NOT allclose/approx: no additional acceptance slack.
    assert error <= tolerance
    assert result.h0_error is not None and np.isfinite(result.h0_error)
    assert result.h0_error <= tolerance
    # Reporting may differ by a norm's rounding; acceptance above remains exact.
    assert result.h0_error == pytest.approx(error, rel=0., abs=2e-14)
    assert dense_h1(matrix, cycles, a, b), "invalid source fixture"
    assert dense_h1(target, cycles, a, b), "full target has a filling or absent edge"
    assert result.source.certified
    assert result.target.certified
    assert result.planar.certified
    assert isinstance(result.hole, np.ndarray)
    assert result.hole.size == 2 and np.isfinite(result.hole).all()


def test_oracle_analytic_square_and_external_filling():
    matrix, guide, cycles = square()
    expected = np.full((4, 4), 2.)
    np.fill_diagonal(expected, 0.)
    np.testing.assert_array_equal(floyd_minimax(matrix), expected)
    assert dense_h1(matrix, cycles, 2.01, 2.1)
    assert not dense_h1(matrix, cycles, 1.99, 2.1)
    assert not dense_h1(matrix, cycles, 2.01, 3.)
    filled = distances(np.vstack((guide, [1., 1.])))
    assert dense_h1(filled[:4, :4], cycles, 2.01, 2.1)
    assert not dense_h1(filled, cycles, 2.01, 2.1)


@pytest.mark.parametrize("extra", [False, True], ids=["square", "far-extra-row"])
def test_positive_full_domain_certificate(layout, extra):
    matrix, guide, cycles = square(extra)
    result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    assert_certificate(result, matrix, cycles)


def test_repeatability_readonly_inputs_and_global_rng(layout):
    matrix, guide, cycles = square(extra=True)
    original = matrix.copy(), guide.copy(), copy.deepcopy(cycles)
    matrix.flags.writeable = guide.flags.writeable = False
    np_state, py_state = np.random.get_state(), random.getstate()
    first = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    second = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    assert_certificate(first, matrix, cycles)
    assert_certificate(second, matrix, cycles)
    np.testing.assert_array_equal(first.embedding, second.embedding)
    np.testing.assert_array_equal(first.hole, second.hole)
    assert first.reason == second.reason
    assert first.h0_error == second.h0_error
    np.testing.assert_array_equal(matrix, original[0])
    np.testing.assert_array_equal(guide, original[1])
    assert cycles == original[2]
    current = np.random.get_state()
    assert current[0] == np_state[0] and current[2:] == np_state[2:]
    np.testing.assert_array_equal(current[1], np_state[1])
    assert random.getstate() == py_state


def test_source_external_filling_cannot_be_certified(layout):
    _, guide, cycles = square()
    guide = np.vstack((guide, [1., 1.]))
    matrix = distances(guide)
    assert not dense_h1(matrix, cycles, 2.01, 2.1)
    result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    assert_failure(result)
    assert result.source is not None and not result.source.certified


def test_source_edge_missing_at_birth(layout):
    matrix, guide, cycles = square()
    result = layout.construct_global_layout(matrix, guide, cycles, 1.99, 2.1)
    assert_failure(result)
    assert result.source is not None and not result.source.certified


def test_zero_chord_is_unsupported_without_partial_embedding(layout):
    matrix, guide, cycles = square()
    matrix[0, 1] = matrix[1, 0] = 0.
    assert dense_h1(matrix, cycles, 2.01, 2.1)
    assert_failure(layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1))


def test_multiple_independent_cycles_are_unsupported(layout):
    _, guide, cycles = square()
    guide = np.vstack((guide, guide + [10., 0.]))
    matrix = distances(guide)
    cycles += [[(i + 4, j + 4) for i, j in cycles[0]]]
    assert dense_h1(matrix, cycles, 2.01, 2.1)
    assert_failure(layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1))


@pytest.mark.parametrize("kind", ["nonsquare", "rank1", "nan", "inf", "negative",
                                  "asymmetric", "diagonal", "complex", "ragged"])
def test_malformed_source_raises_value_error(layout, kind):
    matrix, guide, cycles = square()
    if kind == "nonsquare":
        matrix = matrix[:, :3]
    elif kind == "rank1":
        matrix = matrix.ravel()
    elif kind == "nan":
        matrix[0, 1] = matrix[1, 0] = np.nan
    elif kind == "inf":
        matrix[0, 1] = matrix[1, 0] = np.inf
    elif kind == "negative":
        matrix[0, 1] = matrix[1, 0] = -1.
    elif kind == "asymmetric":
        matrix[0, 1] += .1
    elif kind == "diagonal":
        matrix[0, 0] = .1
    elif kind == "complex":
        matrix = matrix.astype(complex) + 1j
    elif kind == "ragged":
        matrix = [[0.], [1., 0.]]
    with pytest.raises(ValueError):
        layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)


@pytest.mark.parametrize("kind", ["wrong_rows", "wrong_columns", "rank1", "nan", "inf",
                                  "complex", "ragged"])
def test_malformed_guide_raises_value_error(layout, kind):
    matrix, guide, cycles = square()
    if kind == "wrong_rows":
        guide = guide[:3]
    elif kind == "wrong_columns":
        guide = np.zeros((4, 3))
    elif kind == "rank1":
        guide = guide.ravel()
    elif kind == "nan":
        guide[0, 0] = np.nan
    elif kind == "inf":
        guide[0, 0] = np.inf
    elif kind == "complex":
        guide = guide.astype(complex) + 1j
    elif kind == "ragged":
        guide = [[0.], [1., 0.]]
    with pytest.raises(ValueError):
        layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)


@pytest.mark.parametrize("name", ["a", "b", "h0_tolerance"])
@pytest.mark.parametrize("value", [-1., np.nan, np.inf, True, 1j, "2", None])
def test_invalid_radii_and_tolerance(layout, name, value):
    matrix, guide, cycles = square()
    kwargs = dict(a=2.01, b=2.1, h0_tolerance=.05)
    kwargs[name] = value
    with pytest.raises(ValueError):
        layout.construct_global_layout(matrix, guide, cycles, **kwargs)


def test_reversed_radii_raise_value_error(layout):
    matrix, guide, cycles = square()
    with pytest.raises(ValueError):
        layout.construct_global_layout(matrix, guide, cycles, 2.2, 2.1)


@pytest.mark.parametrize("cycles", [None, [None], [[(0, 4)]], [[(-1, 1)]],
                                    [[(0, 1, 2)]], [[(0.5, 1)]], [[(True, 1)]]])
def test_malformed_cycle_indices_raise_value_error(layout, cycles):
    matrix, guide, _ = square()
    with pytest.raises(ValueError):
        layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)


def test_extreme_finite_guide_is_never_falsely_certified(layout):
    matrix, guide, cycles = square(extra=True)
    guide *= 8e307 / np.max(guide)
    assert np.isfinite(guide).all()
    with np.errstate(over="ignore", invalid="ignore"):
        result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    # Rescaling/ignoring a guide is permissible; a success still needs full checks.
    if result.certified:
        assert_certificate(result, matrix, cycles)
    else:
        assert_failure(result)


def test_overflow_scale_cannot_return_a_bogus_embedding(layout):
    matrix, guide, cycles = square()
    matrix *= 4e307
    guide *= 4e307
    assert np.isfinite(matrix).all() and np.isfinite(guide).all()
    with np.errstate(over="ignore", invalid="ignore"):
        result = layout.construct_global_layout(
            matrix, guide, cycles, 8.04e307, 8.4e307, h0_tolerance=.05)
    # The interval-planar certificate cannot prove this squared-distance scale.
    assert_failure(result)


def test_valid_non_simple_chain_is_unsupported(layout):
    # Two squares sharing just one vertex form a nonzero but nonsimple 1-cycle.
    cycles = [[(0, 1), (1, 2), (2, 3), (3, 0),
               (0, 4), (4, 5), (5, 6), (6, 0)]]
    matrix = np.full((7, 7), 5., dtype=np.float64)
    np.fill_diagonal(matrix, 0.)
    for i, j in cycles[0]:
        matrix[i, j] = matrix[j, i] = 2.
    guide = np.zeros((7, 2), dtype=np.float64)
    assert dense_h1(matrix, cycles, 2.01, 2.1)
    assert_failure(layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1))


def test_cycle_edge_orientation_and_order_are_not_geometry(layout):
    matrix, guide, cycles = square()
    # Same undirected chain, deliberately neither path-ordered nor consistently oriented.
    reordered = [[(3, 2), (1, 0), (3, 0), (1, 2)]]
    assert dense_h1(matrix, reordered, 2.01, 2.1)
    result = layout.construct_global_layout(matrix, guide, reordered, 2.01, 2.1)
    assert_certificate(result, matrix, cycles)


def test_nonmetric_external_shortcut_uses_full_h0_domain(layout):
    matrix, guide, cycles = square(extra=True)
    matrix[4, :] = matrix[:, 4] = 4.
    matrix[4, 4] = 0.
    matrix[0, 4] = matrix[4, 0] = .8
    matrix[2, 4] = matrix[4, 2] = .8
    assert dense_h1(matrix, cycles, 2.01, 2.1)
    assert floyd_minimax(matrix)[0, 2] == .8
    assert floyd_minimax(matrix[:4, :4])[0, 2] == 2.
    result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    if result.certified:
        assert_certificate(result, matrix, cycles)
    else:
        assert_failure(result)


def test_zero_h0_tolerance_has_no_hidden_acceptance_slack(layout):
    matrix, guide, cycles = square()
    result = layout.construct_global_layout(
        matrix, guide, cycles, 2.01, 2.1, h0_tolerance=0.)
    if result.certified:
        assert_certificate(result, matrix, cycles, tolerance=0.)
    else:
        assert_failure(result)


@pytest.mark.parametrize("n", [0, 1, 2, 4, 9])
def test_full_hierarchy_matches_floyd_with_ties_and_zero_edges(layout, n):
    rng = np.random.default_rng(171 + n)
    weights = rng.integers(0, 8, size=(n, n)).astype(np.float64)
    matrix = np.triu(weights, 1)
    matrix += matrix.T
    original = matrix.copy()
    matrix.flags.writeable = False
    expected = floyd_minimax(matrix)
    first = layout.full_hierarchy(matrix)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(layout.full_hierarchy(matrix), first)
    np.testing.assert_array_equal(matrix, original)
    assert not np.shares_memory(matrix, first)


def test_full_hierarchy_uses_external_shortcuts_before_restriction(layout):
    matrix, _, _ = square(extra=True)
    matrix[4, :] = matrix[:, 4] = 4.
    matrix[4, 4] = 0.
    matrix[0, 4] = matrix[4, 0] = .8
    matrix[2, 4] = matrix[4, 2] = .8
    np.testing.assert_array_equal(layout.full_hierarchy(matrix), floyd_minimax(matrix))
    assert layout.full_hierarchy(matrix)[0, 2] == .8
    assert layout.full_hierarchy(matrix[:4, :4])[0, 2] == 2.


@pytest.mark.parametrize("tolerance", [.01, .05, .1])
def test_square_contraction_uses_point98_of_requested_tolerance(layout, tolerance):
    matrix, guide, cycles = square()
    result = layout.construct_global_layout(
        matrix, guide, cycles, 2.01, 2.1, h0_tolerance=tolerance)
    assert_certificate(result, matrix, cycles, tolerance=tolerance)
    expected = np.maximum(floyd_minimax(matrix) - .98*tolerance, 0.)
    # Candidate coordinates require trig/SVD; exact acceptance is checked above.
    np.testing.assert_allclose(floyd_minimax(distances(result.embedding)), expected,
                               rtol=0., atol=2e-14)


def test_full_source_analysis_precedes_unsupported_family(layout):
    matrix, guide, cycles = square()
    # The family is numerically well-formed but dependent in source homology.
    cycles = cycles + copy.deepcopy(cycles)
    assert not dense_h1(matrix, cycles, 2.01, 2.1)
    result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    assert_failure(result)
    assert result.source is not None, "source H1 must be checked before family rejection"
    assert not result.source.certified


def test_malformed_multicycle_family_is_not_silently_unsupported(layout):
    matrix, guide, cycles = square()
    cycles.append([(0, 99)])
    with pytest.raises(ValueError):
        layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)


def assert_budget_refusal(layout, matrix, guide, cycles, **kwargs):
    try:
        result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1, **kwargs)
    except layout.ResourceLimitError:
        return
    assert_failure(result)


def test_resource_error_is_a_value_error(layout):
    assert issubclass(layout.ResourceLimitError, ValueError)


@pytest.mark.parametrize("name,value", [("max_vertices", 3), ("max_matrix_entries", 24),
                                        ("max_parent_tests", 0),
                                        ("max_candidate_points", 0),
                                        ("max_contact_pairs", 0)])
def test_layout_limits_never_publish_partial_embeddings(layout, name, value):
    from dataclasses import replace
    matrix, guide, cycles = square(extra=True)
    limits = replace(layout.LayoutLimits(), **{name: value})
    assert_budget_refusal(layout, matrix, guide, cycles, limits=limits)


def test_matrix_entry_budget_exact_boundary(layout):
    from dataclasses import replace
    matrix, guide, cycles = square()
    limits = replace(layout.LayoutLimits(), max_matrix_entries=16)
    result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1, limits=limits)
    assert_certificate(result, matrix, cycles)
    assert_budget_refusal(layout, matrix, guide, cycles,
                         limits=replace(limits, max_matrix_entries=15))


@pytest.mark.parametrize("name", ["max_vertices", "max_matrix_entries", "max_parent_tests",
                                  "max_candidate_points", "max_contact_pairs", "candidate_chunk"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_layout_limits_raise_value_error(layout, name, value):
    from dataclasses import replace
    matrix, guide, cycles = square()
    with pytest.raises(ValueError):
        limits = replace(layout.LayoutLimits(), **{name: value})
        layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1, limits=limits)


@pytest.mark.parametrize("name", ["max_vertices", "max_edges", "max_word_ops",
                                  "max_storage_words", "max_cycles", "max_chain_edges"])
def test_source_h1_limits_are_forwarded(layout, name):
    from dataclasses import replace
    matrix, guide, cycles = square()
    limits = replace(layout.H1Limits(), **{name: 0})
    assert_budget_refusal(layout, matrix, guide, cycles, h1_limits=limits)


def test_source_h1_triangle_budget_includes_noncycle_vertices(layout):
    from dataclasses import replace
    _, guide, cycles = square()
    guide = np.vstack((guide, [-.3, 0.]))
    matrix = distances(guide)
    assert max(matrix[0, 3], matrix[0, 4], matrix[3, 4]) <= 2.1
    assert dense_h1(matrix, cycles, 2.01, 2.1)
    limits = replace(layout.H1Limits(), max_triangles=0)
    assert_budget_refusal(layout, matrix, guide, cycles, h1_limits=limits)


def test_final_h0_rejects_tiny_excess_without_proposal_slack(layout, monkeypatch):
    matrix, guide, cycles = square(extra=True)
    baseline = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    assert_certificate(baseline, matrix, cycles)
    seed = baseline.embedding[:4]
    parent = int(np.argmin(distances(np.vstack((seed, baseline.embedding[4])))[4, :4]))
    direction = baseline.embedding[4] - seed[parent]
    direction /= np.hypot(*direction)
    level = floyd_minimax(matrix)[4, parent]
    # Just above the contractual threshold, much smaller than an allclose atol.
    candidate = seed[parent] + (level + .05 + 8*np.spacing(level))*direction
    target = distances(np.vstack((seed, candidate)))
    expected_error = float(np.max(np.abs(floyd_minimax(matrix) - floyd_minimax(target))))
    assert .05 < expected_error < .05 + 1e-10

    # This is an inspected internal hook, not an assumed public contact API.
    def injected_contact(center, radius, centers, radii, supplied_guide, slack, budget):
        return 0., candidate.copy()

    monkeypatch.setattr(layout, "_closest_contact", injected_contact)
    result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    assert_failure(result)
    assert result.h0_error is not None and result.h0_error > .05
    assert result.hole is None


@pytest.mark.parametrize("which", ["target_sparse_h1", "planar_interval"])
def test_final_certificate_failure_cannot_publish_candidate(layout, monkeypatch, which):
    from dataclasses import replace
    matrix, guide, cycles = square(extra=True)
    if which == "target_sparse_h1":
        actual = layout.analyze_sparse_h1
        calls = []

        def reject_target(*args, **kwargs):
            result = actual(*args, **kwargs)
            calls.append(result)
            if len(calls) == 2:
                return replace(result, certified=False, reason="injected target failure")
            return result

        monkeypatch.setattr(layout, "analyze_sparse_h1", reject_target)
    else:
        actual = layout.planar_certify
        calls = []

        def reject_planar(*args, **kwargs):
            result = actual(*args, **kwargs)
            calls.append(result)
            return replace(result, certified=False, reason="injected planar failure")

        monkeypatch.setattr(layout, "planar_certify", reject_planar)
    result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    assert len(calls) == (2 if which == "target_sparse_h1" else 1)
    assert_failure(result)
    assert result.hole is None


@pytest.mark.parametrize('slot', [0, 1, 2], ids=['left', 'singular', 'right'])
def test_silent_nonfinite_lapack_result_is_unsupported_not_invalid_input(layout, monkeypatch, slot):
    # NumPy/BLAS versions differ in whether overflow respects np.errstate.
    # Explicit guards must preserve the valid-input failure lifecycle even if
    # LAPACK returns a nonfinite factor without raising an exception.
    matrix, guide, cycles = square()
    actual = layout.np.linalg.svd

    def nonfinite_factor(*args, **kwargs):
        factors = list(actual(*args, **kwargs))
        factors[slot] = np.full_like(factors[slot], np.nan)
        return tuple(factors)

    monkeypatch.setattr(layout.np.linalg, 'svd', nonfinite_factor)
    result = layout.construct_global_layout(matrix, guide, cycles, 2.01, 2.1)
    assert_failure(result)
    assert result.source.certified
    assert result.hole is None and result.target is None
    assert 'Procrustes' in result.reason
