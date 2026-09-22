"""Bounded source-dual/adaptive-H0 repair regressions, not a capacity claim.

Verified methods/pitfalls are persisted here (only this file is in task scope):
recheck ACTUAL returned coordinates with independent H1/H0 checkers, including
failed restart candidates; a small objective or SLSQP success is not evidence.
Cover every current odd triangle, not only a single filling. Verify adaptive
merge trees by independent Floyd minimax closure, not original MST identities.
Tied equidistant source rows admit a planar octagon H0 realization even when
original MST-star edge lengths are wrong; this says nothing about general
planar feasibility or nonconvex search convergence. All random tests use local
seeds and <=9 vertices; no real dataset, optional PH library, or network.

Run: PYTHONPATH=python python -m pytest -q tests/test_structural_dual_layout.py
"""

from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations
import json

import numpy as np
import pytest

from open_deep_tda import structural_dual_layout as layout
from open_deep_tda import topology
from open_deep_tda.structural_dual import dual_h1_certificate
from open_deep_tda.structural_h0 import compare_h0
from open_deep_tda.structural_h1 import H1BudgetExceeded, check_h1_witnesses


SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]
POINTS = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
BIRTH, SURVIVAL = 1., 1.2
LIMITS = dict(max_rounds=8, max_iterations=100, restarts=1, seed=42)


def distances(points):
    """Do not reuse the repair module's distance conversion."""
    points = np.asarray(points)
    return np.linalg.norm(points[:, None] - points[None, :], axis=-1)


def fixture(name):
    cycles = [deepcopy(SQUARE)]
    source = POINTS.copy()
    if name == "missing":
        initial = np.array([[0., 0.], [0., 3.], [1., 1.], [2., 1.]])
    elif name == "wrong_ids":
        initial = POINTS[[0, 2, 1, 3]].copy()
    elif name == "center_fill":
        source = np.vstack([POINTS, [10., 10.]])
        initial = np.vstack([POINTS, [.5, .5]])
    elif name == "duplicate_rings":
        source = np.vstack([POINTS, POINTS + [10., 0.]])
        initial = np.vstack([POINTS, POINTS])
        cycles += [[(i + 4, j + 4) for i, j in SQUARE]]
    elif name in ("collinear", "coincident"):
        initial = np.zeros((4, 2))
        if name == "collinear":
            initial[:, 0] = np.arange(4)
    else:
        raise AssertionError(name)
    return distances(source), initial, cycles


def repair(D, initial, cycles, tolerance=None, **overrides):
    return layout.solve_structural_layout(
        D, initial, cycles, BIRTH, SURVIVAL, h0_tolerance=tolerance,
        **dict(LIMITS, **overrides))


def verify_result(result, D, cycles, tolerance=None, **overrides):
    """Fresh existing checkers, not optimizer success or cached certificates."""
    limits = dict(LIMITS, **overrides)
    assert result["status"] in ("certified", "unresolved")
    candidate = result["last_candidate"]
    assert candidate.shape == (len(D), 2)
    assert np.isfinite(candidate).all()
    target = topology.distance_matrix(candidate)
    np.testing.assert_allclose(target, distances(candidate), rtol=1e-14, atol=0)
    accepted = True
    if cycles:
        checked = check_h1_witnesses(D, target, cycles, BIRTH, SURVIVAL)
        assert result["certificate"]["h1"]["independent_check"] == asdict(checked)
        assert result["certificate"]["h1"]["accepted"] == checked.accepted
        assert result["certificate"]["h1"]["surviving_rank"] == checked.surviving_rank
        assert result["certificate"]["h1"]["all_classes_independent"] == checked.all_classes_independent
        accepted = checked.accepted
        assert result["charged_oracle_operations"] > 0
    else:
        assert result["certificate"]["h1"] is None
        assert result["charged_oracle_operations"] == 0
    if tolerance is not None:
        checked_h0 = compare_h0(D, target, tolerance=tolerance)
        assert result["certificate"]["h0"] == checked_h0
        accepted = accepted and checked_h0["certified_within_tolerance"]
    else:
        assert result["certificate"]["h0"] is None  # No unrequested H0 claim.
    assert (result["status"] == "certified") == accepted
    if accepted:
        np.testing.assert_array_equal(result["embedding"], candidate)
    else:
        assert result["embedding"] is None
    assert 1 <= result["verification_calls"] <= 1 + limits["restarts"] + (
        limits["restarts"] + 1) * limits["max_rounds"]
    assert len(result["history"]) <= (limits["restarts"] + 1) * limits["max_rounds"]
    for record in result["history"]:
        assert 0 <= record["restart"] <= limits["restarts"]
        assert 0 <= record["round"] < limits["max_rounds"]
        for cut in record.get("cuts", []):
            i, j = cut["edge"]
            assert 0 <= i < j < len(D)
            assert D[i, j] > SURVIVAL
            assert SURVIVAL < cut["minimum_distance"] <= D[i, j]
            assert cut["triangles_covered"] > 0
    # Metadata must be finite and serializable even on unresolved searches.
    metadata = {k: v for k, v in result.items() if k not in ("embedding", "last_candidate")}
    json.dumps(metadata, allow_nan=False)
    assert "not an infeasibility certificate" in result["search"]


@pytest.mark.parametrize("name", ["missing", "wrong_ids", "center_fill", "duplicate_rings",
                                  "collinear", "coincident"])
@pytest.mark.parametrize("tolerance", [None, .05], ids=["h1_only", "joint_h0"])
def test_small_adversarial_repairs_checked_independently(name, tolerance):
    if tolerance is not None:
        pytest.importorskip("open_deep_tda._core")
    D, initial, cycles = fixture(name)
    before_D, before_initial, before_cycles = D.copy(), initial.copy(), deepcopy(cycles)
    failed = check_h1_witnesses(D, distances(initial), cycles, BIRTH, SURVIVAL)
    assert not failed.accepted
    if name == "missing":
        assert np.linalg.matrix_rank(initial - initial[0]) == 2
        assert distances(initial)[2, 3] == D[2, 3]
        assert distances(initial)[0, 2] == D[0, 2]
    if name == "wrong_ids":
        # A simultaneous permutation preserves the barcode, not sample IDs.
        np.testing.assert_array_equal(distances(initial), D[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])])
    if name == "center_fill":
        assert failed.witnesses[0].edges_present and not failed.witnesses[0].survives
    if name == "duplicate_rings":
        assert all(w.survives for w in failed.witnesses)
        assert failed.surviving_rank == 1
    D.flags.writeable = initial.flags.writeable = False
    result = repair(D, initial, cycles, tolerance)
    verify_result(result, D, cycles, tolerance)
    # Regression expectations for these tiny fixtures, not a general guarantee.
    assert result["status"] == "certified"
    assert result["history"]
    np.testing.assert_array_equal(D, before_D)
    np.testing.assert_array_equal(initial, before_initial)
    assert cycles == before_cycles


@pytest.mark.parametrize("tolerance", [None, .05])
def test_intact_layout_returned_exactly_untouched_without_optimizer(monkeypatch, tolerance):
    if tolerance is not None:
        pytest.importorskip("open_deep_tda._core")

    def forbidden(*args, **kwargs):
        pytest.fail("an already checked layout must not be optimized")

    monkeypatch.setattr(layout, "_optimize", forbidden)
    initial = POINTS + [7., -3.]
    D = distances(POINTS)
    result = repair(D, initial, [SQUARE], tolerance, max_iterations=0, max_rounds=0)
    verify_result(result, D, [SQUARE], tolerance, max_iterations=0, max_rounds=0)
    np.testing.assert_array_equal(result["embedding"], initial)
    assert not np.shares_memory(result["embedding"], initial)
    assert result["history"] == [] and result["verification_calls"] == 1


def test_h0_global_bridge_and_conservative_envelope_is_not_necessary(monkeypatch):
    pytest.importorskip("open_deep_tda._core")
    source = np.column_stack(([0., 1., 10., 11.], np.zeros(4)))
    initial = np.column_stack(([0., 1., 100., 101.], np.zeros(4)))
    D, target = distances(source), distances(initial)
    for ids in ([0, 1], [2, 3]):
        assert compare_h0(D[np.ix_(ids, ids)], target[np.ix_(ids, ids)],
                          tolerance=.05)["certified_within_tolerance"]
    assert not compare_h0(D, target, tolerance=.05)["certified_within_tolerance"]
    result = layout.solve_structural_layout(D, initial, h0_tolerance=.05, **LIMITS)
    verify_result(result, D, [], .05)
    assert result["status"] == "certified"

    # Both squares have identical same-ID merge distances (=1), but the source
    # MST's particular edge (0,1) is too long in this equally valid target.
    D, initial, _ = fixture("wrong_ids")
    _, upper = layout._envelope(D, .05)
    assert any(distances(initial)[edge] > bound for edge, bound in upper.items())
    assert compare_h0(D, distances(initial), tolerance=.05)["certified_within_tolerance"]

    def forbidden(*args, **kwargs):
        pytest.fail("sufficient search envelope must not replace the H0 checker")

    monkeypatch.setattr(layout, "_optimize", forbidden)
    result = layout.solve_structural_layout(D, initial, h0_tolerance=.05, **LIMITS)
    verify_result(result, D, [], .05)
    np.testing.assert_array_equal(result["embedding"], initial)


@pytest.mark.parametrize("zero_limit", ["max_rounds", "max_iterations"])
def test_zero_search_budget_is_unresolved_not_false_success(monkeypatch, zero_limit):
    D, initial, cycles = fixture("center_fill")

    def forbidden(*args, **kwargs):
        pytest.fail("zero search budget must not run SLSQP")

    monkeypatch.setattr(layout, "_optimize", forbidden)
    result = repair(D, initial, cycles, **{zero_limit: 0})
    verify_result(result, D, cycles, **{zero_limit: 0})
    assert result["status"] == "unresolved"
    assert result["history"] == [] and result["verification_calls"] == 1
    np.testing.assert_array_equal(result["last_candidate"], initial)


@pytest.mark.parametrize("coincident", [False, True], ids=["collinear", "coincident"])
@pytest.mark.parametrize("tolerance", [None, .05])
def test_seeded_restarts_escape_degenerate_starts_without_global_rng_changes(coincident, tolerance):
    if tolerance is not None:
        pytest.importorskip("open_deep_tda._core")
    initial = np.zeros((4, 2))
    if not coincident:
        initial[:, 0] = np.arange(4)
    D = distances(POINTS)
    before = deepcopy(np.random.get_state())
    first = repair(D, initial, [SQUARE], tolerance)
    second = repair(D, initial, [SQUARE], tolerance)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    verify_result(first, D, [SQUARE], tolerance)
    assert first["status"] == second["status"] == "certified"
    assert any(record["restart"] == 1 for record in first["history"])
    np.testing.assert_array_equal(first["embedding"], second["embedding"])
    assert first["history"] == second["history"]


def optimizer_record(success=True):
    return dict(optimizer_success=success, optimizer_status=0 if success else 9,
                iterations=1, displacement_objective=0., constraint_count=1)


@pytest.mark.parametrize("valid_candidate", [False, True])
def test_acceptance_uses_oracles_not_slsqp_status(monkeypatch, valid_candidate):
    pytest.importorskip("open_deep_tda._core")
    D, initial, cycles = fixture("missing")
    unit = max(float(np.median(D[D > 0])), SURVIVAL)

    def misleading_optimizer(anchor, start, lower, upper, scale, iterations):
        assert scale == unit
        candidate = (POINTS - initial.mean(axis=0)) / unit if valid_candidate else anchor.copy()
        return candidate, optimizer_record(success=not valid_candidate)

    monkeypatch.setattr(layout, "_optimize", misleading_optimizer)
    result = repair(D, initial, cycles, .05, max_rounds=2, restarts=0)
    verify_result(result, D, cycles, .05, max_rounds=2, restarts=0)
    assert result["status"] == ("certified" if valid_candidate else "unresolved")
    if not valid_candidate:
        assert len(result["history"]) == 2
        assert all(row["optimizer_success"] for row in result["history"])
        assert all(row["displacement_objective"] == 0. for row in result["history"])


def test_h1_only_does_not_claim_unrequested_h0():
    pytest.importorskip("open_deep_tda._core")
    D, initial, cycles = fixture("center_fill")
    initial[-1] = [3., 3.]  # No filling, but wrong global H0 bridge.
    assert not compare_h0(D, distances(initial), tolerance=.05)["certified_within_tolerance"]
    result = repair(D, initial, cycles)
    verify_result(result, D, cycles)
    assert result["status"] == "certified" and result["certificate"]["h0"] is None
    np.testing.assert_array_equal(result["embedding"], initial)


def test_invalid_source_contracts_fail_before_optimization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid source must never reach optimization")

    monkeypatch.setattr(layout, "_optimize", forbidden)
    D = distances(POINTS)
    missing = D.copy()
    missing[0, 1] = missing[1, 0] = 1.1
    for source, cycles in [(missing, [SQUARE]), (np.zeros((4, 4)), [SQUARE]),
                           (D, [SQUARE, SQUARE]), (D, [[(0, 1)]]),
                           (D, [SQUARE + [(0, 4), (4, 0)]])]:
        with pytest.raises(ValueError):
            repair(source, POINTS, cycles)


def test_invalid_source_matrices_and_initial_coordinates():
    D = distances(POINTS)
    for bad in (np.zeros((4, 3)), D + np.eye(4), -D, np.triu(D),
                np.full((4, 4), np.nan), np.full((4, 4), np.inf),
                D.astype(complex), D.astype(str), [[0], [1, 0]], None):
        with pytest.raises(ValueError):
            repair(bad, POINTS, [SQUARE])
    for bad in (np.zeros((4, 3)), np.zeros((3, 2)), np.zeros(8),
                np.full((4, 2), np.nan), np.full((4, 2), np.inf),
                POINTS.astype(complex), POINTS.astype(str), POINTS.astype(bool)):
        with pytest.raises(ValueError):
            repair(D, bad, [SQUARE])


def test_invalid_radii_and_missing_obligations():
    D = distances(POINTS)
    for birth, survival in ((None, 1.2), (1, None), (-1, 1.2), (1.2, 1),
                            (np.nan, 1.2), (1, np.inf), (True, 1.2), (1, "1.2"),
                            (1, 10**1000)):
        with pytest.raises(ValueError):
            layout.solve_structural_layout(D, POINTS, [SQUARE], birth, survival)
    with pytest.raises(ValueError, match="at least one"):
        layout.solve_structural_layout(D, POINTS)
    with pytest.raises(ValueError, match="radii must be omitted"):
        layout.solve_structural_layout(D, POINTS, h0_tolerance=.05, birth_radius=1)


def test_invalid_search_limits_and_scalar_options():
    D = distances(POINTS)
    integer_limits = ("max_vertices", "max_rounds", "max_iterations", "restarts", "seed",
                      "max_simplices", "max_reduction_operations", "max_reduction_entries")
    for key in integer_limits:
        for bad in (-1, True, 1.5, np.inf, None):
            with pytest.raises(ValueError, match=key):
                repair(D, POINTS, [SQUARE], **{key: bad})
    for key in ("jitter", "separation_margin", "h0_tolerance"):
        for bad in (-1, True, np.nan, np.inf, "0", 10**1000):
            with pytest.raises(ValueError, match=key):
                layout.solve_structural_layout(D, POINTS, [SQUARE], BIRTH, SURVIVAL, **{key: bad})
    with pytest.raises(ValueError, match="separation_margin"):
        repair(D, POINTS, [SQUARE], separation_margin=0)
    # H1 budgets are validated even when only H0 was requested.
    with pytest.raises(ValueError, match="max_simplices"):
        layout.solve_structural_layout(D, POINTS, h0_tolerance=.05, max_simplices=-1)


def test_resource_exhaustion_and_hard_cap_fail_closed_before_conversion():
    D = distances(POINTS)
    for key in ("max_vertices", "max_simplices", "max_reduction_operations", "max_reduction_entries"):
        with pytest.raises(RuntimeError, match=key):
            repair(D, POINTS, [SQUARE], **{key: 0})

    class Oversized:
        shape = (65, 65)

        def __array__(self, *args, **kwargs):
            pytest.fail("hard cap must be checked before source conversion")

    with pytest.raises(RuntimeError, match="max_vertices=64"):
        repair(Oversized(), np.zeros((65, 2)), [SQUARE], max_vertices=1000)

    # Source validation fits, but target center introduces extra triangles.
    D, initial, cycles = fixture("center_fill")
    source_cost = dual_h1_certificate(D, D, cycles, BIRTH, SURVIVAL)["n_simplices"]
    with pytest.raises(H1BudgetExceeded, match="max_simplices"):
        repair(D, initial, cycles, max_simplices=source_cost)


def test_checker_budget_and_backend_errors_propagate(monkeypatch):
    D = distances(POINTS)

    def budget_failure(*args, **kwargs):
        raise H1BudgetExceeded("independent checker budget exhausted")

    monkeypatch.setattr(layout, "check_h1_witnesses", budget_failure)
    with pytest.raises(H1BudgetExceeded, match="independent checker"):
        repair(D, POINTS, [SQUARE])

    def backend_failure(*args, **kwargs):
        raise RuntimeError("H0 backend unavailable")

    monkeypatch.setattr(layout, "compare_h0", backend_failure)
    with pytest.raises(RuntimeError, match="H0 backend unavailable"):
        layout.solve_structural_layout(D, POINTS, h0_tolerance=.05)


def minimax_closure(D):
    """Independent Floyd-Warshall oracle; no native MST or production traversal."""
    result = np.array(D, dtype=float, copy=True)
    for k in range(len(result)):
        result = np.minimum(result, np.maximum(result[:, k, None], result[None, k, :]))
    return result


def assert_batch_cover(dual, source, lower, cuts, survival=SURVIVAL):
    """Every forbidden triangle is hit, with exact newly-covered accounting."""
    pending = list(dual['forbidden_triangles'])
    selected = set()
    for cut in cuts:
        edge = tuple(cut['edge'])
        assert edge not in selected
        selected.add(edge)
        assert 0 <= edge[0] < edge[1] < len(source)
        assert survival < cut['minimum_distance'] <= source[edge]
        assert source[edge] > survival
        assert survival < lower[edge] <= source[edge]
        assert lower[edge] >= cut['minimum_distance']
        hit = [t for t in pending if list(edge) in t['eligible_source_absent_edges']]
        assert len(hit) == cut['triangles_covered'] > 0
        pending = [t for t in pending if t not in hit]
    assert not pending
    for triangle in dual['forbidden_triangles']:
        faces = set(combinations(triangle['vertices'], 2))
        eligible = set(map(tuple, triangle['eligible_source_absent_edges']))
        assert eligible == {e for e in faces if source[e] > survival}
        assert faces & eligible & selected  # ALL triangles, not just one relation.


@pytest.mark.parametrize('name', ['center_fill', 'duplicate_rings'])
def test_full_batch_and_birth_bounds_reach_optimizer(monkeypatch, name):
    D, initial, cycles = fixture(name)
    dual_calls, captured = [], []
    optimize = layout._optimize

    def observe_dual(*args, **kwargs):
        report = dual_h1_certificate(*args, **kwargs)
        dual_calls.append(report)
        return report

    def observe_optimizer(anchor, start, lower, upper, unit, iterations):
        captured.append((deepcopy(dual_calls[-1]), dict(lower), dict(upper)))
        return optimize(anchor, start, lower, upper, unit, iterations)

    monkeypatch.setattr(layout, 'dual_h1_certificate', observe_dual)
    monkeypatch.setattr(layout, '_optimize', observe_optimizer)
    result = repair(D, initial, cycles)
    verify_result(result, D, cycles)
    assert result['status'] == 'certified'
    assert len(captured) == len(result['history']) > 0
    assert any(row['cuts'] for row in result['history'])
    for row, (dual, lower, upper) in zip(result['history'], captured):
        assert row['forbidden_triangles'] == len(dual['forbidden_triangles'])
        assert_batch_cover(dual, D, lower, row['cuts'])
        for cycle in cycles:
            assert all(upper[tuple(sorted(edge))] <= BIRTH for edge in cycle)


@pytest.mark.parametrize('margin', [np.nextafter(0., 1.), .01, 10.])
def test_batch_cuts_seeded_random_targets_ties_and_one_ulp_source_gap(margin):
    rng = np.random.default_rng(481)
    covered = 0
    for iteration in range(24):
        # Square plus duplicate source rows and two off-witness isolated rows.
        # Duplicate source rows introduce triangles without killing the cycle.
        ids = list(range(4)) + rng.integers(0, 4, size=2).tolist()
        base = distances(POINTS)
        source = np.full((8, 8), 3.)
        source[:6, :6] = base[np.ix_(ids, ids)]
        np.fill_diagonal(source, 0.)
        for edge in combinations(range(8), 2):
            if source[edge] > SURVIVAL:
                source[edge] = source[edge[::-1]] = rng.choice(
                    [np.nextafter(SURVIVAL, np.inf), 1.25, 2., 3.])
        target = np.zeros_like(source)
        for edge in combinations(range(8), 2):
            target[edge] = target[edge[::-1]] = rng.choice([0., 1., SURVIVAL, 2.])
        if iteration == 0:
            target.fill(0.)  # Dense target: many simultaneous alternative fillings.
        report = dual_h1_certificate(source, target, [SQUARE], BIRTH, SURVIVAL)
        # Direct parity scan, independent of the batch selector's bookkeeping.
        supports = [set(map(tuple, c['support_edges'])) for c in report['cochains']]
        forbidden = [list(t) for t in combinations(range(8), 3)
                     if all(target[e] <= SURVIVAL for e in combinations(t, 2))
                     and any(len(set(combinations(t, 2)) & s) % 2 for s in supports)]
        assert [t['vertices'] for t in report['forbidden_triangles']] == forbidden
        before_source, before_target, before_report = source.copy(), target.copy(), deepcopy(report)
        # Preserve a stronger existing source-consistent cut and unrelated edge.
        lower = {(0, 6): float(source[0, 6]), (0, 1): .75}
        before_lower = dict(lower)
        cuts = layout._batch_cuts(report, source, target, SURVIVAL, margin, 1., lower)
        assert_batch_cover(report, source, lower, cuts)
        assert all(lower[e] >= bound for e, bound in before_lower.items())
        assert all(value <= source[e] for e, value in lower.items())
        selected = {tuple(c['edge']) for c in cuts}
        assert set(lower) == set(before_lower) | selected
        for edge in set(before_lower) - selected:
            assert lower[edge] == before_lower[edge]
        again_lower = dict(before_lower)
        assert layout._batch_cuts(report, source, target, SURVIVAL, margin, 1., again_lower) == cuts
        assert again_lower == lower
        np.testing.assert_array_equal(source, before_source)
        np.testing.assert_array_equal(target, before_target)
        assert report == before_report
        covered += len(forbidden)
    assert covered > 24


def test_batch_empty_and_invalid_certificates_fail_closed():
    D = distances(POINTS)
    lower = {(0, 1): .5}
    assert layout._batch_cuts({'forbidden_triangles': []}, D, D, SURVIVAL, .01, 1., lower) == []
    assert lower == {(0, 1): .5}
    for eligible, message in [([], 'source-absent'), ([[0, 1]], 'invalid source-dual')]:
        with pytest.raises(RuntimeError, match=message):
            layout._batch_cuts({'forbidden_triangles': [
                {'eligible_source_absent_edges': eligible}]}, D, D, SURVIVAL, .01, 1., {})


@pytest.mark.parametrize('tied', [False, True])
def test_merge_upper_exact_hierarchy_random_source_oracle(tied):
    pytest.importorskip('open_deep_tda._core')
    rng = np.random.default_rng(951)
    epsilon = .125  # Binary-exact offsets make exact subtraction meaningful.
    changed = 0
    for iteration in range(48):
        n = int(rng.integers(2, 10))
        raw = rng.integers(0, 5, size=(n, n)).astype(float) if tied else (
            rng.permutation(n * n).reshape(n, n).astype(float) / 16)
        D = np.triu(raw, 1)
        D += D.T
        target = np.triu(rng.integers(0, 6, size=(n, n)).astype(float), 1)
        target += target.T
        source_merges = minimax_closure(D)
        lower = {(i, j): max(0., source_merges[i, j] - epsilon)
                 for i, j in combinations(range(n), 2)}
        # Some preferred cross pairs become incompatible; source edges always
        # remain available. This exercises filtering without assuming necessity.
        for edge in lower:
            if D[edge] > source_merges[edge] and rng.random() < .5:
                lower[edge] = float(D[edge])
        before = D.copy(), target.copy(), dict(lower)
        upper = layout._merge_upper(D, target, lower, epsilon)
        assert len(upper) == n - 1
        assert all(lower[e] <= b for e, b in upper.items())
        assert all(0 <= i < j < n for i, j in upper)
        weighted_tree = np.full_like(D, np.inf)
        np.fill_diagonal(weighted_tree, 0.)
        for (i, j), bound in upper.items():
            weighted_tree[i, j] = weighted_tree[j, i] = bound - epsilon
        # Exact full same-ID minimax hierarchy, not sorted edge weights/bars.
        np.testing.assert_array_equal(minimax_closure(weighted_tree), source_merges)
        assert layout._merge_upper(D, target, lower, epsilon) == upper
        np.testing.assert_array_equal(D, before[0])
        np.testing.assert_array_equal(target, before[1])
        assert lower == before[2]
        changed += set(upper) != set(map(tuple, topology.mst(D).tolist()))
    assert changed > 5  # We actually exercised adaptive, non-original edges.


def test_merge_upper_uses_shortest_compatible_edge_and_rejects_inconsistent_cuts():
    pytest.importorskip('open_deep_tda._core')
    D = np.array([[0., 1., 3., 3.], [1., 0., 3., 3.],
                  [3., 3., 0., 1.], [3., 3., 1., 0.]])
    target = np.array([[0., 1., 9., 4.], [1., 0., 2., 3.],
                       [9., 2., 0., 1.], [4., 3., 1., 0.]])
    assert layout._merge_upper(D, target, {}, .125) == {
        (0, 1): 1.125, (2, 3): 1.125, (1, 2): 3.125}
    lower = {(1, 2): 3.25}
    assert layout._merge_upper(D, target, lower, .125) == {
        (0, 1): 1.125, (2, 3): 1.125, (1, 3): 3.125}
    lower = {(i, j): 4. for i in (0, 1) for j in (2, 3)}
    with pytest.raises(RuntimeError, match='inconsistent cuts'):
        layout._merge_upper(D, target, lower, .125)


def octagon(side):
    angles = np.arange(8) * (np.pi / 4)
    return np.column_stack([np.cos(angles), np.sin(angles)]) * (side / (2 * np.sin(np.pi / 8)))


def test_eight_equidistant_source_has_octagon_h0_despite_wrong_original_star():
    pytest.importorskip('open_deep_tda._core')
    source = np.ones((8, 8)) - np.eye(8)
    initial = octagon(1.)
    target = distances(initial)
    report = compare_h0(source, target, tolerance=.05)
    assert report['source_tree']['edges'] == [[0, i] for i in range(1, 8)]
    assert report['certified_within_tolerance']
    assert report['max_merge_error'] < 1e-14
    assert not report['tree_edge_bound_within_tolerance']
    assert report['max_tree_edge_error'] > 1.
    _, original_upper = layout._envelope(source, .05)
    assert any(target[e] > bound for e, bound in original_upper.items())
    lower, _ = layout._envelope(source, .05)
    adaptive = layout._merge_upper(source, target, lower, .045)
    assert all(target[e] <= bound for e, bound in adaptive.items())
    result = layout.solve_structural_layout(source, initial, h0_tolerance=.05, **LIMITS)
    verify_result(result, source, [], .05)
    assert result['status'] == 'certified' and result['verification_calls'] == 1
    np.testing.assert_array_equal(result['embedding'], initial)


def test_h0_octagon_side_1point2_repaired_with_declared_small_budget():
    pytest.importorskip('open_deep_tda._core')
    source = np.ones((8, 8)) - np.eye(8)
    initial = octagon(1.2)
    assert not compare_h0(source, distances(initial), tolerance=.05)['certified_within_tolerance']
    # The side-one construction above proves THIS instance feasible. This
    # fixed-budget regression was observed to succeed, not a capacity theorem.
    result = layout.solve_structural_layout(source, initial, h0_tolerance=.05, **LIMITS)
    verify_result(result, source, [], .05)
    assert result['status'] == 'certified' and result['history']


def test_empty_h1_family_is_h0_only_and_never_calls_h1_or_dual(monkeypatch):
    pytest.importorskip('open_deep_tda._core')

    def forbidden(*args, **kwargs):
        pytest.fail('empty H1 obligation must not construct/check a dual certificate')

    monkeypatch.setattr(layout, 'dual_h1_certificate', forbidden)
    monkeypatch.setattr(layout, 'check_h1_witnesses', forbidden)
    D = distances(POINTS)
    result = layout.solve_structural_layout(D, 1.2 * POINTS, cycles=iter(()),
                                             h0_tolerance=.05, **LIMITS)
    verify_result(result, D, [], .05)
    assert result['status'] == 'certified' and result['history']
    for key in ('birth_radius', 'survival_radius'):
        with pytest.raises(ValueError, match='radii must be omitted'):
            layout.solve_structural_layout(D, POINTS, cycles=[], h0_tolerance=.05, **{key: 1.})


def test_independent_h1_disagreement_with_satisfied_dual_fails_closed(monkeypatch):
    calls = []

    def disagree(*args, **kwargs):
        result = check_h1_witnesses(*args, **kwargs)
        calls.append(result)
        return replace(result, all_classes_independent=False)

    monkeypatch.setattr(layout, 'check_h1_witnesses', disagree)
    with pytest.raises(RuntimeError, match='disagrees'):
        repair(distances(POINTS), POINTS, [SQUARE])
    assert len(calls) == 1


def test_every_actual_candidate_checks_both_oracles_and_charges_work(monkeypatch):
    pytest.importorskip('open_deep_tda._core')
    D, initial, cycles = fixture('coincident')
    dual_calls, checker_calls, h0_calls = [], [], []

    def observe(function, calls):
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            calls.append(result)
            return result
        return wrapped

    monkeypatch.setattr(layout, 'dual_h1_certificate', observe(dual_h1_certificate, dual_calls))
    monkeypatch.setattr(layout, 'check_h1_witnesses', observe(check_h1_witnesses, checker_calls))
    monkeypatch.setattr(layout, 'compare_h0', observe(compare_h0, h0_calls))
    result = repair(D, initial, cycles, .05)
    verify_result(result, D, cycles, .05)
    checks = result['verification_calls']
    assert checks > 1 and result['status'] == 'certified'
    assert any(row['restart'] == 1 for row in result['history'])
    assert len(checker_calls) == len(h0_calls) == checks
    assert len(dual_calls) <= 1 + (LIMITS['restarts'] + 1) * LIMITS['max_rounds']
    assert dual_calls[0]['certificate_satisfied']  # Source validation, not a target pass.
    assert result['charged_oracle_operations'] == (
        sum(r['reduction_operations'] for r in dual_calls)
        + sum(r.reduction_operations for r in checker_calls))


@pytest.mark.parametrize('mode', ['h1_only', 'joint', 'h0_only'])
@pytest.mark.parametrize('first_finite', [False, True])
@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_failed_last_candidate_certificate_matches_actual_coords_after_restarts(
        monkeypatch, mode, first_finite, bad):
    if mode != 'h1_only':
        pytest.importorskip('open_deep_tda._core')
    D, initial, cycles = fixture('center_fill')
    tolerance = None if mode == 'h1_only' else .05
    seen = []

    def fail_optimizer(anchor, start, lower, upper, unit, iterations):
        seen.append(start * unit + initial.mean(axis=0))
        if first_finite and len(seen) == 1:
            return start + .001 * np.arange(start.size).reshape(start.shape), optimizer_record()
        return np.full_like(start, bad), optimizer_record()

    monkeypatch.setattr(layout, '_optimize', fail_optimizer)
    overrides = dict(max_rounds=2, restarts=2)
    if mode == 'h0_only':
        cycles = []
        result = layout.solve_structural_layout(D, initial, h0_tolerance=tolerance,
                                                 **dict(LIMITS, **overrides))
    else:
        result = repair(D, initial, cycles, tolerance, **overrides)
    verify_result(result, D, cycles, tolerance, **overrides)
    assert result['status'] == 'unresolved' and result['embedding'] is None
    assert result['verification_calls'] == 1 + overrides['restarts'] + int(first_finite)
    assert sum(r.get('status') == 'nonfinite_candidate' for r in result['history']) == 3
    np.testing.assert_array_equal(result['last_candidate'], seen[-1])
    assert not np.array_equal(result['last_candidate'], initial)


def test_feasible_jitter_restart_is_checked_before_nonfinite_optimizer(monkeypatch):
    D, initial, cycles = fixture('wrong_ids')
    unit = max(float(np.median(D[D > 0])), SURVIVAL)
    optimize_calls = []

    class FixedJitter:
        def normal(self, *, size):
            assert size == initial.shape
            return (POINTS - initial) / unit

    def fail_optimizer(anchor, start, lower, upper, scale, iterations):
        optimize_calls.append(start.copy())
        return np.full_like(start, np.nan), optimizer_record()

    monkeypatch.setattr(layout.np.random, 'default_rng', lambda seed: FixedJitter())
    monkeypatch.setattr(layout, '_optimize', fail_optimizer)
    result = repair(D, initial, cycles, jitter=1., max_rounds=2)
    verify_result(result, D, cycles, jitter=1., max_rounds=2)
    assert result['status'] == 'certified' and result['verification_calls'] == 2
    assert len(optimize_calls) == 1
    np.testing.assert_allclose(result['embedding'], POINTS, atol=1e-15)


@pytest.mark.parametrize('failed_obligation', ['h0', 'h1'])
def test_joint_acceptance_requires_both_checks_not_just_one(monkeypatch, failed_obligation):
    pytest.importorskip('open_deep_tda._core')
    if failed_obligation == 'h0':
        D, initial, cycles = fixture('center_fill')
        initial[-1] = [3., 3.]  # Intact H1, wrong global H0 bridge.
    else:
        D, initial, cycles = fixture('wrong_ids')  # Exact H0, absent H1 birth edges.
    assert check_h1_witnesses(D, distances(initial), cycles, BIRTH, SURVIVAL).accepted == (
        failed_obligation == 'h0')
    assert compare_h0(D, distances(initial), tolerance=.05)['certified_within_tolerance'] == (
        failed_obligation == 'h1')

    def misleading_optimizer(anchor, start, lower, upper, unit, iterations):
        return anchor.copy(), optimizer_record(success=True)

    monkeypatch.setattr(layout, '_optimize', misleading_optimizer)
    result = repair(D, initial, cycles, .05, max_rounds=2, restarts=0)
    verify_result(result, D, cycles, .05, max_rounds=2, restarts=0)
    assert result['status'] == 'unresolved'
    assert all(row['optimizer_success'] and row['displacement_objective'] == 0.
               for row in result['history'])
    if failed_obligation == 'h0':
        assert all(row['cuts'] == [] and row['forbidden_triangles'] is None
                   for row in result['history'])


def test_sufficient_dual_is_not_required_to_accept_independently_valid_h1(monkeypatch):
    D, initial, cycles = fixture('center_fill')
    initial[-1] = [.5, 1.6]  # Triangle on edge (2,3), but not a filling of the square.
    assert not dual_h1_certificate(D, distances(initial), cycles, BIRTH, SURVIVAL)[
        'certificate_satisfied']
    assert check_h1_witnesses(D, distances(initial), cycles, BIRTH, SURVIVAL).accepted
    source_calls = []

    def source_validation_only(source, target, *args, **kwargs):
        np.testing.assert_array_equal(source, target)
        source_calls.append(1)
        return dual_h1_certificate(source, target, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('fixed source duals are sufficient, not necessary for acceptance')

    monkeypatch.setattr(layout, 'dual_h1_certificate', source_validation_only)
    monkeypatch.setattr(layout, '_optimize', forbidden)
    result = repair(D, initial, cycles)
    verify_result(result, D, cycles)
    assert result['status'] == 'certified' and source_calls == [1]
    assert result['history'] == [] and result['verification_calls'] == 1
    np.testing.assert_array_equal(result['embedding'], initial)
