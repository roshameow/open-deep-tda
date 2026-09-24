"""Synthetic, source-only opt-in beam regression; never a general OOS claim."""
import json
import time

import numpy as np
import pytest

from open_deep_tda import DeepTDA
from open_deep_tda import _ph_guided_beam as beam
from open_deep_tda import _ph_guided_training as guided
from open_deep_tda._ph_guided_source import TeacherLimits, construct_source_teacher
from open_deep_tda._ph_guided_training import GuidedLimits
from open_deep_tda.structural_constructive import pairwise_planar
from open_deep_tda.structural_h0 import compare_h0
from open_deep_tda.structural_sparse_h1 import ResourceLimitError, analyze_sparse_h1
from open_deep_tda.structural_planar import ResourceLimitError as PlanarResourceLimitError


CYCLE = [[(0, 1), (1, 2), (2, 3), (3, 0)]]


def square(extra=()):
    X = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]] + list(extra))
    return np.linalg.norm(X[:, None] - X[None, :], axis=-1), X


def run(D, X, *, a=2.01, b=2.1, tol=.05, limits=TeacherLimits()):
    return construct_source_teacher(D, X, CYCLE, a, b, tol,
                                    strategy='single_beam', limits=limits)


def test_square_source_and_independent_complete_certificates():
    D, X = square()
    before = (D.copy(), X.copy())
    result = run(D, X)
    assert result.certified and result.source.certified and result.target.certified
    assert result.embedding.flags.owndata and not result.embedding.flags.writeable
    T = pairwise_planar(result.embedding)
    assert compare_h0(D, T, tolerance=.05, max_vertices=4)['certified_within_tolerance']
    assert analyze_sparse_h1(T, CYCLE, 2.01, 2.1).certified
    # Birth is inclusive at the exact source edge, not at the preceding float.
    assert run(D, X, a=2., b=2.1).certified
    assert not run(D, X, a=np.nextafter(2., 0.), b=2.1).certified
    # At the square diagonal the source cycle is filled at the closed survival threshold.
    assert not run(D, X, a=2., b=np.sqrt(8.)).certified
    assert result.diagnostics['beam_work']['expanded_states'] == 0
    assert result.diagnostics['beam_events'][-1]['status'] == 'certified'
    np.testing.assert_array_equal(D, before[0]); np.testing.assert_array_equal(X, before[1])


def test_sorted_parent_ties_first_complete_and_caps():
    D, X = square([[4., 0.]])
    one, two = run(D, X), run(D, X)
    assert one.certified and two.certified
    np.testing.assert_array_equal(one.embedding, two.embedding)
    events = one.diagnostics['beam_events']
    assert events[0]['parents'] == [1, 2, 0, 3]
    assert events[-1]['path'] == ((1, 0),)
    assert one.diagnostics['beam_work']['complete_checked'] == 1
    for field, message in [('max_beam_expansions', 'expansion'),
                           ('max_beam_parent_tests', 'parent'),
                           ('max_beam_candidate_points', 'contact'),
                           ('max_beam_contact_pairs', 'contact')]:
        with pytest.raises(ResourceLimitError, match=message):
            run(D, X, limits=TeacherLimits(**{field: 0}))
    with pytest.raises(ValueError, match='max_beam_expansions'):
        run(D, X, limits=TeacherLimits(max_beam_expansions=4097))
    with pytest.raises(ResourceLimitError, match='workspace'):
        run(D, X, limits=TeacherLimits(max_distance_workspace_bytes=1))
    with pytest.raises(ResourceLimitError, match='vertex/pair'):
        run(D, X, limits=TeacherLimits(max_vertices=4))
    with pytest.raises(ResourceLimitError, match='max_seconds'):
        run(D, X, limits=TeacherLimits(max_seconds=0))


def test_invalid_source_and_unsupported_fail_closed(monkeypatch):
    D, X = square()
    original_seed = beam.seed
    monkeypatch.setattr(beam, 'seed', lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('uncertified source must never enter beam')))
    assert not run(D, X, a=1.99).certified  # missing birth edges at source
    monkeypatch.setattr(beam, 'seed', original_seed)
    changed = D.copy(); changed[0, 1] += .01; changed[1, 0] += .01
    with pytest.raises(ValueError, match='Euclidean'):
        run(changed, X)
    # Both disconnected squares have source H1, but the beam accepts exactly one.
    double = np.vstack([X, X + [10., 0.]])
    metric = np.linalg.norm(double[:, None]-double[None, :], axis=-1)
    family = CYCLE + [[(u+4, v+4) for u, v in CYCLE[0]]]
    assert analyze_sparse_h1(metric, family, 2.01, 2.1).certified
    unsupported = construct_source_teacher(metric, double, family, 2.01, 2.1,
                                           .05, strategy='single_beam')
    assert not unsupported.certified and unsupported.embedding is None
    assert 'exactly one' in unsupported.reason
    for a, b in [(1.99, 2.1), (2.01, 3.0)]:
        r = run(D, X, a=a, b=b)
        assert not r.certified and r.embedding is None


def test_target_h1_or_planar_absence_never_accepted(monkeypatch):
    D, X = square()
    original = beam.analyze_sparse_h1
    def missing(*args, **kwargs):
        from dataclasses import replace
        return replace(original(*args, **kwargs), certified=False, reason='missing target H1')
    monkeypatch.setattr(beam, 'analyze_sparse_h1', missing)
    r = run(D, X)
    assert not r.certified and r.embedding is None and r.target is None
    assert 'h1_failed' in r.diagnostics['beam_events'][-1]['status']
    monkeypatch.setattr(beam, 'analyze_sparse_h1', original)
    from dataclasses import replace
    original_planar = beam.planar_certify
    monkeypatch.setattr(beam, 'planar_certify', lambda *args, **kwargs:
                        replace(original_planar(*args, **kwargs), certified=False))
    r = run(D, X)
    assert not r.certified and r.embedding is None
    assert r.diagnostics['beam_events'][-1]['status'].startswith('planar_failed')
    def exhausted(*args, **kwargs):
        raise PlanarResourceLimitError('forced planar work cap')
    monkeypatch.setattr(beam, 'planar_certify', exhausted)
    with pytest.raises(PlanarResourceLimitError, match='forced planar work cap'):
        run(D, X)


def test_birth_threshold_discrepancy_refused_for_both_source_teachers():
    D, X = square()
    changed = X.copy()
    changed[1, 0] = np.nextafter(2., np.inf)
    for strategy in ('single', 'single_beam'):
        with pytest.raises(ValueError, match='birth/survival threshold'):
            construct_source_teacher(D, changed, CYCLE, 2., 2.1, .05,
                                     strategy=strategy)


def test_beam_h0_outward_boundary_and_contact_budget_before_trigonometry(monkeypatch):
    D, X = square()
    result = run(D, X, tol=2.**-49)
    assert not result.certified and result.embedding is None
    assert result.diagnostics['beam_events'][-1]['status'] == 'h0_failed'
    counts = dict(parent_tests=0, candidate_points=0, contact_pairs=0)
    def forbidden(*args, **kwargs):
        raise AssertionError('candidate trig must not precede its work cap')
    monkeypatch.setattr(beam.np, 'cos', forbidden)
    with pytest.raises(ResourceLimitError, match='contact work exhausted'):
        beam.contacts(np.zeros(2), 1., np.array([[1., 0.]]),
                      np.array([.9]), np.array([0., 1.]), 1e-12,
                      counts, time.monotonic()+10.,
                      TeacherLimits(max_beam_candidate_points=0))
    assert counts['candidate_points'] == 0


def test_static_budget_and_unbounded_inner_edge_refused_before_native_fit(monkeypatch):
    D, X = square()
    m = DeepTDA(steps=6, warmup_steps=0, h0_size=4, h1_size=4,
                topology_interval=1, subset_bank_size=1,
                standardize=False, missing_indicators=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('native fit must not begin')
    monkeypatch.setattr(DeepTDA, 'fit', forbidden)
    for teacher in (TeacherLimits(max_pairs=0),
                    TeacherLimits(max_distance_workspace_bytes=1)):
        with pytest.raises(ResourceLimitError, match='budget'):
            m.fit_with_topology_guidance(X, cycles=CYCLE, birth_radius=2.01,
                                         survival_radius=2.1, strategy='single_beam',
                                         teacher_limits=teacher)
    # Four high-dimensional rows have a small 3*n*n*d distance cube, but
    # native fit's 1,024 sampled pair differences exceed one megabyte.
    with pytest.raises(ResourceLimitError, match='pair-sample workspace'):
        m.fit_with_topology_guidance(np.zeros((4, 2048)), cycles=CYCLE,
            birth_radius=2.01, survival_radius=2.1, strategy='single_beam',
            limits=GuidedLimits(max_feature_workspace_bytes=1_000_000))
    consumed = []
    def infinite_edge():
        while True:
            consumed.append(1)
            yield 0
    with pytest.raises(ValueError, match='finite family of edge lists') as error:
        m.fit_with_topology_guidance(X, cycles=[[infinite_edge()]], birth_radius=2.01,
                                     survival_radius=2.1, strategy='single_beam')
    assert 'exactly two' in str(error.value.__cause__)
    assert len(consumed) == 3 and not m._fitted


def test_timeout_propagates_without_partial_embedding(monkeypatch):
    D, X = square([[3., 0.]])
    def timeout(*args, **kwargs):
        raise TimeoutError('forced beam deadline')
    monkeypatch.setattr(beam, 'contacts', timeout)
    with pytest.raises(TimeoutError, match='forced beam deadline'):
        run(D, X)


def test_terminal_beam_certificate_crossing_teacher_clock_is_refused(monkeypatch):
    D, X = square()
    original = beam.planar_certify
    calls = []
    def slow_terminal(*args, **kwargs):
        calls.append(1)
        time.sleep(.6)
        return original(*args, **kwargs)
    monkeypatch.setattr(beam, 'planar_certify', slow_terminal)
    with pytest.raises(TimeoutError, match='beam deadline'):
        run(D, X, limits=TeacherLimits(max_seconds=.5))
    assert calls


def test_certificate_wall_timeout_is_transactional(monkeypatch):
    theta = 2*np.pi*np.arange(16)/16
    X = np.column_stack((np.cos(theta), np.sin(theta)))
    cycle = [[(i, (i+1)%16) for i in range(16)]]
    m = DeepTDA(steps=6, warmup_steps=0, topology_interval=1,
                h0_size=16, h1_size=16, subset_bank_size=1,
                standardize=False, missing_indicators=False, seed=0)
    original = guided.compare_h0
    calls = []
    def slow_check(*args, **kwargs):
        calls.append(1)
        time.sleep(.25)
        return original(*args, **kwargs)
    monkeypatch.setattr(guided, 'compare_h0', slow_check)
    plan = GuidedLimits(max_vertices=16, max_feature_workspace_bytes=100000,
                        teacher_steps=0, contact_steps=0, cut_steps=0, max_seconds=.20)
    with pytest.raises(TimeoutError, match='guided PH training wall limit'):
        m.fit_with_topology_guidance(X, cycles=cycle, birth_radius=.5,
            survival_radius=.9, strategy='single_beam', limits=plan,
            teacher_limits=TeacherLimits(max_vertices=16, max_pairs=120, max_seconds=30))
    assert calls and not m._fitted and not hasattr(m, 'embedding_')


def test_post_certificate_report_wall_timeout_preserves_prior_fit(monkeypatch):
    from open_deep_tda import evaluation
    theta = 2*np.pi*np.arange(16)/16
    X = np.column_stack((np.cos(theta), np.sin(theta)))
    cycle = [[(i, (i+1)%16) for i in range(16)]]
    m = DeepTDA(steps=6, warmup_steps=0, topology_interval=1,
                h0_size=16, h1_size=16, subset_bank_size=1,
                standardize=False, missing_indicators=False, seed=0)
    m.fit(X)
    prior = m.transform(X).copy()
    original = evaluation.evaluate_embedding
    calls = []
    def slow_postfit(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            time.sleep(1.1)
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluation, 'evaluate_embedding', slow_postfit)
    plan = GuidedLimits(max_vertices=16, max_feature_workspace_bytes=100000,
                        teacher_steps=0, contact_steps=0, cut_steps=0, max_seconds=1.)
    with pytest.raises(TimeoutError, match='guided fit/report wall limit'):
        m.fit_with_topology_guidance(X, cycles=cycle, birth_radius=.5,
            survival_radius=.9, strategy='single_beam', limits=plan,
            teacher_limits=TeacherLimits(max_vertices=16, max_pairs=120, max_seconds=30))
    assert len(calls) == 2
    np.testing.assert_array_equal(prior, m.transform(X))


def test_same_ph_mlp_fit_reload_and_transactional_failure(tmp_path, monkeypatch):
    theta = 2*np.pi*np.arange(16)/16
    X = np.column_stack((np.cos(theta), np.sin(theta)))
    cycle = [[(i, (i+1)%16) for i in range(16)]]
    m = DeepTDA(steps=6, warmup_steps=0, h0_size=16, h1_size=16,
                topology_interval=1, subset_bank_size=1, missing_indicators=False,
                standardize=False, seed=0)
    plan = GuidedLimits(max_vertices=16, max_feature_workspace_bytes=100000,
                        teacher_steps=0, contact_steps=0, cut_steps=0, max_seconds=90)
    source = TeacherLimits(max_vertices=16, max_pairs=120, max_seconds=30)
    assert m.fit_with_topology_guidance(X, cycles=cycle, birth_radius=.5,
        survival_radius=.9, source_scale=1.1, strategy='single_beam',
        limits=plan, teacher_limits=source) is m
    report = m.report_['guided_training']
    assert report['certificate']['accepted'] and report['source_teacher']['beam_work']['complete_checked']
    assert 'single-cycle beam' in report['phase_scopes']
    assert m.training_coverage_['h1_updates'] > 0
    json.dumps(m.report_, allow_nan=False)
    path = tmp_path/'beam.pt'; m.save(path)
    loaded = DeepTDA.load(path)
    np.testing.assert_array_equal(loaded.transform(X), loaded.embedding_)
    factor = loaded.reference_scale_/report['source_scale']
    D = pairwise_planar(loaded.reference_.astype(np.float64)*factor)
    T = pairwise_planar(loaded.embedding_.astype(np.float64)*factor)
    assert compare_h0(D, T, tolerance=.05, max_vertices=16)['certified_within_tolerance']
    assert analyze_sparse_h1(T, cycle, .5, .9).certified
    old = m.transform(X).copy()
    with pytest.raises(ValueError, match='exactly 1'):
        m.fit_with_topology_guidance(X, cycles=[], birth_radius=.5,
                                     survival_radius=.9, strategy='single_beam')
    np.testing.assert_array_equal(old, m.transform(X))
    with pytest.raises(ResourceLimitError, match='budget'):
        m.fit_with_topology_guidance(X, cycles=cycle, birth_radius=.5,
            survival_radius=.9, strategy='single_beam', limits=plan,
            teacher_limits=TeacherLimits(max_vertices=15))
    np.testing.assert_array_equal(old, m.transform(X))
    # A failure AFTER the candidate's native PH fit must not publish its state.
    def timed_out(*args, **kwargs):
        raise TimeoutError('forced beam timeout')
    monkeypatch.setattr(beam, 'seed', timed_out)
    with pytest.raises(TimeoutError, match='forced beam timeout'):
        m.fit_with_topology_guidance(X, cycles=cycle, birth_radius=.5,
            survival_radius=.9, strategy='single_beam', limits=plan, teacher_limits=source)
    np.testing.assert_array_equal(old, m.transform(X))
