"""Bounded planar REPAIR contracts, not a reducer or convergence theorem.

Verified methods/pitfalls retained here: independently check same-ID H0 and H1
on every returned candidate, including off-cycle vertices and joint class rank.
A source-consistent cut breaks one exhibited filling, not all possible fillings;
a chosen branch or conservative H0 envelope cannot prove global infeasibility.
Small fixed fixtures and explicit search limits keep these regressions bounded.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations
import json

import numpy as np
import pytest

from open_deep_tda import structural_layout as layout
from open_deep_tda.structural_h0 import compare_h0
from open_deep_tda.structural_h1 import H1BudgetExceeded, check_h1_witnesses
from open_deep_tda.structural_obstructions import find_h1_obstructions


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
    else:
        raise AssertionError(name)
    return distances(source), initial, cycles


def repair(D, initial, cycles, tolerance=None, **overrides):
    return layout.repair_layout(
        D, initial, cycles, BIRTH, SURVIVAL, h0_tolerance=tolerance,
        **dict(LIMITS, **overrides))


def verify_result(result, D, cycles, tolerance=None, **overrides):
    """Fresh existing checkers, not optimizer success or cached certificates."""
    limits = dict(LIMITS, **overrides)
    assert result["status"] in ("certified", "unresolved")
    candidate = result["last_candidate"]
    assert candidate.shape == (len(D), 2)
    assert np.isfinite(candidate).all()
    target = distances(candidate)
    accepted = True
    if cycles:
        checked = check_h1_witnesses(D, target, cycles, BIRTH, SURVIVAL)
        assert result["certificate"]["h1"]["independent_check"] == asdict(checked)
        assert result["certificate"]["h1"]["accepted"] == checked.accepted
        assert result["certificate"]["h1"]["surviving_rank"] == checked.surviving_rank
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
    assert 1 <= result["verification_calls"] <= 1 + (
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
            assert cut["relation_cycles"]
    # Metadata must be finite and serializable even on unresolved searches.
    metadata = {k: v for k, v in result.items() if k not in ("embedding", "last_candidate")}
    json.dumps(metadata, allow_nan=False)
    assert "not an infeasibility certificate" in result["search"]


@pytest.mark.parametrize("name", ["missing", "wrong_ids", "center_fill", "duplicate_rings"])
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
    # Use the documented default budget for the coincident-ring regression.
    # The shortened smoke budget is not a cross-SciPy convergence promise.
    search = (dict(max_rounds=16, max_iterations=300, restarts=2, seed=0)
              if name == 'duplicate_rings' else {})
    result = repair(D, initial, cycles, tolerance, **search)
    verify_result(result, D, cycles, tolerance, **search)
    # This retained reference is a bounded nonconvex search, not a convergence
    # contract—even these feasible fixtures can exhaust their budget on another
    # SciPy/BLAS version. verify_result above requires the truthful certificate
    # and embedding=None on failure. Deterministic acceptance/rejection paths
    # are tested separately with prescribed proposals; quality failures remain
    # in the published real-data diagnostics, not hidden as test passes.
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
    result = layout.repair_layout(D, initial, h0_tolerance=.05, **LIMITS)
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
    result = layout.repair_layout(D, initial, h0_tolerance=.05, **LIMITS)
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


@pytest.mark.parametrize("name", ["center_fill", "duplicate_rings"])
def test_cuts_break_exhibited_fillings_and_are_actually_passed_to_optimizer(monkeypatch, name):
    D, initial, cycles = fixture(name)
    previous = find_h1_obstructions(D, distances(initial), cycles, BIRTH, SURVIVAL)
    captured = []
    real_optimize = layout._optimize

    def observe(anchor, start, lower, upper, unit, iterations):
        candidate, record = real_optimize(anchor, start, lower, upper, unit, iterations)
        captured.append((dict(lower), dict(upper), candidate * unit + initial.mean(axis=0)))
        return candidate, record

    monkeypatch.setattr(layout, "_optimize", observe)
    result = repair(D, initial, cycles)
    verify_result(result, D, cycles)
    assert any(row["cuts"] for row in result["history"])
    previous_target = distances(initial)
    for row, (lower, upper, candidate) in zip(result["history"], captured):
        assert len(row["cuts"]) == len(previous["relations"])
        for cut, relation in zip(row["cuts"], previous["relations"]):
            edges, boundary = set(), set()
            for triangle in relation["filling_triangles"]:
                faces = set(combinations(triangle, 2))
                assert all(previous_target[edge] <= SURVIVAL for edge in faces)
                edges.update(faces)
                boundary.symmetric_difference_update(faces)
            expected = set()
            for index in relation["cycle_indices"]:
                for edge in cycles[index]:
                    expected.symmetric_difference_update([tuple(sorted(edge))])
            assert boundary == expected  # Exact F2 closure, not a barcode match.
            edge = tuple(cut["edge"])
            assert edge in edges and D[edge] > SURVIVAL
            assert SURVIVAL < cut["minimum_distance"] <= D[edge]
            assert lower[edge] >= cut["minimum_distance"]
            assert cut["relation_cycles"] == relation["cycle_indices"]
        for cycle in cycles:
            assert all(upper[tuple(sorted(edge))] <= BIRTH for edge in cycle)
        previous_target = distances(candidate)
        previous = find_h1_obstructions(D, previous_target, cycles, BIRTH, SURVIVAL)
    assert len(captured) == len(result["history"])


def test_independent_h1_checker_is_called_and_disagreement_fails_closed(monkeypatch):
    D = distances(POINTS)
    calls = []

    def disagree(*args, **kwargs):
        checked = check_h1_witnesses(*args, **kwargs)
        calls.append(checked)
        return replace(checked, all_classes_independent=False)

    monkeypatch.setattr(layout, "check_h1_witnesses", disagree)
    with pytest.raises(RuntimeError, match="disagrees"):
        repair(D, POINTS, [SQUARE])
    assert len(calls) == 1


def test_every_candidate_checks_both_requested_oracles_and_charges_work(monkeypatch):
    pytest.importorskip("open_deep_tda._core")
    D, initial, cycles = fixture("missing")
    finder_calls, checker_calls, h0_calls = [], [], []

    def observe(function, calls):
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            calls.append(result)
            return result
        return wrapped

    monkeypatch.setattr(layout, "find_h1_obstructions", observe(find_h1_obstructions, finder_calls))
    monkeypatch.setattr(layout, "check_h1_witnesses", observe(check_h1_witnesses, checker_calls))
    monkeypatch.setattr(layout, "compare_h0", observe(compare_h0, h0_calls))
    result = repair(D, initial, cycles, .05)
    verify_result(result, D, cycles, .05)
    checks = result["verification_calls"]
    assert checks > 1
    assert len(finder_calls) == checks + 1  # Separate initial source validation.
    assert len(checker_calls) == len(h0_calls) == checks
    assert result["charged_oracle_operations"] == (
        sum(r["reduction_operations"] for r in finder_calls)
        + sum(r.reduction_operations for r in checker_calls))


def test_h1_only_does_not_claim_unrequested_h0():
    pytest.importorskip("open_deep_tda._core")
    D, initial, cycles = fixture("center_fill")
    initial[-1] = [3., 3.]  # No filling, but wrong global H0 bridge.
    assert not compare_h0(D, distances(initial), tolerance=.05)["certified_within_tolerance"]
    result = repair(D, initial, cycles)
    verify_result(result, D, cycles)
    assert result["status"] == "certified" and result["certificate"]["h0"] is None
    np.testing.assert_array_equal(result["embedding"], initial)


def test_nonfinite_optimizer_candidate_never_escapes_as_embedding(monkeypatch):
    D, initial, cycles = fixture("center_fill")

    def nonfinite(anchor, start, lower, upper, unit, iterations):
        return np.full_like(start, np.nan), optimizer_record()

    monkeypatch.setattr(layout, "_optimize", nonfinite)
    result = repair(D, initial, cycles, max_rounds=2)
    verify_result(result, D, cycles, max_rounds=2)
    assert result["status"] == "unresolved" and result["verification_calls"] == 1
    assert len(result["history"]) == LIMITS["restarts"] + 1
    assert all(row["status"] == "nonfinite_candidate" for row in result["history"])
    np.testing.assert_array_equal(result["last_candidate"], initial)


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
            layout.repair_layout(D, POINTS, [SQUARE], birth, survival)
    with pytest.raises(ValueError, match="at least one"):
        layout.repair_layout(D, POINTS)
    with pytest.raises(ValueError, match="radii must be omitted"):
        layout.repair_layout(D, POINTS, h0_tolerance=.05, birth_radius=1)


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
                layout.repair_layout(D, POINTS, [SQUARE], BIRTH, SURVIVAL, **{key: bad})
    with pytest.raises(ValueError, match="separation_margin"):
        repair(D, POINTS, [SQUARE], separation_margin=0)
    # H1 budgets are validated even when only H0 was requested.
    with pytest.raises(ValueError, match="max_simplices"):
        layout.repair_layout(D, POINTS, h0_tolerance=.05, max_simplices=-1)


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
    source_cost = find_h1_obstructions(D, D, cycles, BIRTH, SURVIVAL)["n_simplices"]
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
        layout.repair_layout(D, POINTS, h0_tolerance=.05)


def test_short_coincident_ring_search_never_promises_convergence():
    D, initial, cycles = fixture('duplicate_rings')
    result = repair(D, initial, cycles)
    # SciPy 1.13 can exhaust this short budget on an explicitly feasible case.
    # Neither platform-dependent convergence nor an optimizer flag is a proof.
    verify_result(result, D, cycles)
