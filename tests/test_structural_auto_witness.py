"""Public optional-helper regressions; ordinary tests never run real Ripser.

Verified reusable checks: mock the lazy dependency through sys.modules, record
all-row float64 values, protective copies and solver options, and validate the
resulting family with the real sparse analyzer. Filled external rows and float32-rounded bar
endpoints must not be trusted. Only these synthetic tests establish behavior;
no real external-solver performance or resource bound is claimed.

Run: python -m pytest -q tests/test_structural_auto_witness.py
Optional real integration (still skips if Ripser is not installed):
OPEN_DEEP_TDA_TEST_RIPSER=1 python -m pytest -q \
    tests/test_structural_auto_witness.py -k real_ripser
The opt-in gate is separate from importorskip: installing an optional dependency
alone must not turn the ordinary mock suite into an external-solver workload.
Verified on Python 3.9: both the opt-in gate and, separately, missing-package
importorskip report a skip rather than counting external integration as passed.
The ordinary example regressions check full 12-row construction, deterministic
JSON, unchanged RNG state, failure-without-embedding and no adaptive retries.
"""
from dataclasses import FrozenInstanceError, replace
import importlib
import os
import sys
import types

import numpy as np
import pytest

from open_deep_tda import structural_auto_witness as aw
from open_deep_tda.structural_sparse_h1 import H1Limits, ResourceLimitError


def square(n=4):
    D = np.full((n, n), 10., dtype=np.float64)
    np.fill_diagonal(D, 0.)
    for u, v in ((0, 1), (1, 2), (2, 3), (0, 3)):
        D[u, v] = D[v, u] = 1.
    D[0, 2] = D[2, 0] = D[1, 3] = D[3, 1] = 3.
    return D


def install(monkeypatch, bars=((1., 3.),), cochains=None):
    bars = np.asarray(bars, dtype=np.float64).reshape(-1, 2)
    if cochains is None:
        cochains = [np.array([[0, 1, 1]], dtype=np.int64) for _ in bars]
    output = {'dgms': [np.empty((0, 2)), bars], 'cocycles': [[], cochains]}
    calls = []

    def ripser(D, **kwargs):
        calls.append((D, kwargs))
        return output

    monkeypatch.setitem(sys.modules, 'ripser', types.SimpleNamespace(ripser=ripser))
    return calls, output


def rejected(result, reason):
    assert not result.certified
    assert result.cycles == ()
    assert reason in result.diagnostics['reason']


def test_explicit_consent_and_lazy_import(monkeypatch):
    # Reloading helper must not attempt to import this missing dependency.
    monkeypatch.setitem(sys.modules, 'ripser', None)
    importlib.reload(aw)
    for consent in (False, None, 1, np.bool_(True), 'yes'):
        with pytest.raises(ValueError, match='allow_external=True'):
            aw.auto_global_witnesses(square(), allow_external=consent)
    with pytest.raises(ValueError, match='allow_external=True'):
        aw.auto_global_witnesses(square())
    rejected(aw.auto_global_witnesses(square(), allow_external=True), 'unavailable')


@pytest.mark.parametrize('n', [4, 7, 19])
def test_all_rows_fixed_interval_immutable_family_fresh_analysis(monkeypatch, n):
    calls, _ = install(monkeypatch)
    D = square(n)
    D.flags.writeable = False
    original = D.copy()
    analyzed = []
    real = aw.analyze_sparse_h1

    def observe(matrix, cycles, a, b, **kwargs):
        analyzed.append((matrix, cycles, a, b, kwargs))
        return real(matrix, cycles, a, b, **kwargs)

    monkeypatch.setattr(aw, 'analyze_sparse_h1', observe)
    result = aw.auto_global_witnesses(D, allow_external=True)
    assert result.certified and result.source.certified and result.source.rank == 1
    assert result.birth == 1.4 and result.survival == 2.6
    assert result.cycles == (((0, 1), (0, 3), (1, 2), (2, 3)),)
    assert result.diagnostics['fundamental_edge'] == (2, 3)
    assert len(calls) == len(analyzed) == 1
    assert analyzed[0][0] is D
    assert calls[0][0] is not D
    np.testing.assert_array_equal(calls[0][0], D)
    assert calls[0][1] == dict(distance_matrix=True, maxdim=1, coeff=2, do_cocycles=True)
    assert analyzed[0][1:4] == (result.cycles, result.birth, result.survival)
    assert result.source.work['vertices'] == n
    np.testing.assert_array_equal(D, original)
    with pytest.raises(FrozenInstanceError):
        result.cycles = ()
    with pytest.raises(TypeError):
        result.cycles[0][0] = (0, 2)


@pytest.mark.parametrize('bars,expected', [
    ([(1., 2.), (1., 3.)], 1),  # longest first
    ([(1., 3.), (.5, 2.5)], 1),  # same length, earlier birth
    ([(1., 3.), (1., 3.)], 0),  # exact tie, original index
    ([(0., np.inf), (1., 1.), (np.nan, 4.), (1., 3.)], 3),
])
def test_fixed_bar_selection(monkeypatch, bars, expected):
    install(monkeypatch, bars)
    result = aw.auto_global_witnesses(square(), allow_external=True)
    assert result.diagnostics['bar_index'] == expected
    assert result.diagnostics['bar'] == bars[expected]


def test_no_finite_positive_bar(monkeypatch):
    install(monkeypatch, [(1., 1.), (0., np.inf), (-1., 3.), (3., 2.)])
    result = aw.auto_global_witnesses(square(), allow_external=True)
    rejected(result, 'no_finite_positive')
    assert result.birth is result.survival is result.source is None


def test_never_fall_back_to_shorter_bar(monkeypatch):
    install(monkeypatch, [(1., 3.), (1., 2.)],
            [np.empty((0, 3), dtype=np.int64), np.array([[0, 1, 1]])])
    result = aw.auto_global_witnesses(square(), allow_external=True)
    rejected(result, 'no_pairing_one')
    assert result.diagnostics['bar_index'] == 0


def test_external_row_triangle_rejects_cochain(monkeypatch):
    install(monkeypatch)
    D = square(5)
    D[4, :4] = D[:4, 4] = 1.
    result = aw.auto_global_witnesses(D, allow_external=True)
    rejected(result, 'cochain_not_closed')
    assert result.diagnostics['survival_triangles'] == 4
    assert result.source is None


def test_float32_endpoints_do_not_override_exact_birth(monkeypatch):
    install(monkeypatch, [(1., 1. + 2.**-24)])
    D = square()
    # float32 collapses this edge to 1; exact birth lies below the real edge.
    D[2, 3] = D[3, 2] = 1. + 2.**-25
    assert np.float32(D[2, 3]) == np.float32(1.)
    result = aw.auto_global_witnesses(D, allow_external=True)
    rejected(result, 'no_pairing_one')
    assert result.birth < D[2, 3] < result.survival


def test_float32_endpoints_do_not_override_exact_survival(monkeypatch):
    install(monkeypatch, [(1., 1. + 2.**-24)])
    D = square()
    # A diagonal rounds UP to the external death, but is present at exact b.
    D[0, 2] = D[2, 0] = 1. + 0.7 * 2.**-24
    assert np.float32(D[0, 2]) == np.float32(1. + 2.**-24)
    rejected(aw.auto_global_witnesses(D, allow_external=True), 'cochain_not_closed')


def test_fresh_analysis_rejection_never_exposes_family(monkeypatch):
    install(monkeypatch)
    real = aw.analyze_sparse_h1

    def reject(*args, **kwargs):
        return replace(real(*args, **kwargs), certified=False, reason='test_refusal')

    monkeypatch.setattr(aw, 'analyze_sparse_h1', reject)
    result = aw.auto_global_witnesses(square(), allow_external=True)
    rejected(result, 'exact_source_rejected')
    assert result.source is not None and not result.source.certified


@pytest.mark.parametrize('field,value', [
    ('max_vertices', 3), ('max_edges', 3), ('max_cycles', 0),
    ('max_chain_edges', 3), ('max_word_ops', 0), ('max_storage_words', 0),
])
def test_resource_refusal(monkeypatch, field, value):
    calls, _ = install(monkeypatch)
    with pytest.raises(ResourceLimitError, match=field):
        aw.auto_global_witnesses(square(), allow_external=True,
                                 limits=replace(H1Limits(), **{field: value}))
    if field == 'max_vertices':
        assert calls == []


def test_triangle_preflight_cap(monkeypatch):
    install(monkeypatch)
    D = square(5)
    D[4, :4] = D[:4, 4] = 1.
    with pytest.raises(ResourceLimitError, match='max_triangles'):
        aw.auto_global_witnesses(D, allow_external=True,
                                 limits=replace(H1Limits(), max_triangles=3))


@pytest.mark.parametrize('cochain', [np.array([[0., 1., 1.]]), np.array([[0, 4, 1]]),
                                     np.array([[0, 0, 1]]), np.array([0, 1, 1])])
def test_malformed_cochain_no_family(monkeypatch, cochain):
    install(monkeypatch, cochains=[cochain])
    rejected(aw.auto_global_witnesses(square(), allow_external=True), 'malformed')


def test_cochain_orientation_duplicates_and_parity(monkeypatch):
    install(monkeypatch, cochains=[np.array([[1, 0, 3], [2, 3, 1], [3, 2, 1]])])
    assert aw.auto_global_witnesses(square(), allow_external=True).certified


def test_input_validation_precedes_external_call(monkeypatch):
    calls, _ = install(monkeypatch)
    for D in (square().astype(np.float32), np.zeros((4, 3)), [[0.]],
              np.full((4, 4), np.nan)):
        with pytest.raises(ValueError):
            aw.auto_global_witnesses(D, allow_external=True)
    assert calls == []


def test_first_lex_fundamental_edge(monkeypatch):
    D = square(8)
    for u, v in ((4, 5), (5, 6), (6, 7), (4, 7)):
        D[u, v] = D[v, u] = 1.
    install(monkeypatch, cochains=[np.array([[0, 1, 1], [4, 5, 1]])])
    result = aw.auto_global_witnesses(D, allow_external=True)
    assert result.certified
    assert result.diagnostics['fundamental_edge'] == (2, 3)


def test_external_backend_cannot_mutate_caller_matrix(monkeypatch):
    _, output = install(monkeypatch)
    matrix = square()
    original = matrix.copy()
    def mutating_backend(received, **kwargs):
        received[:] = 0.
        return output
    monkeypatch.setitem(sys.modules, 'ripser', types.SimpleNamespace(ripser=mutating_backend))
    result = aw.auto_global_witnesses(matrix, allow_external=True)
    np.testing.assert_array_equal(matrix, original)
    assert result.certified and result.source.certified


@pytest.mark.skipif(os.environ.get('OPEN_DEEP_TDA_TEST_RIPSER') != '1',
                    reason='real Ripser integration requires explicit test opt-in')
def test_real_ripser_all_rows_optional():
    # A missing optional package must skip this test, not the offline module.
    pytest.importorskip('ripser', reason='optional Ripser is not installed')
    demo = example_module()
    D, _, _, _, _ = demo.synthetic_case()
    original = D.copy()
    result = aw.auto_global_witnesses(D, allow_external=True)
    assert result.certified, result.diagnostics
    assert result.source.certified and result.source.rank == 1
    assert result.source.work['vertices'] == 12
    assert result.birth < result.survival
    verified = aw.analyze_sparse_h1(D, result.cycles, result.birth, result.survival)
    assert verified.certified and verified.rank == 1
    np.testing.assert_array_equal(D, original)


def example_module():
    """Load only the owned public example, without running its CLI entry point."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'examples' / 'construct_global_layout.py'
    spec = importlib.util.spec_from_file_location('public_layout_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_example_all_rows_deterministic_no_external_or_rng(monkeypatch, capsys):
    import builtins
    import json
    import random

    real_import = builtins.__import__

    def no_optional_import(name, *args, **kwargs):
        if name in ('ripser', 'open_deep_tda.structural_auto_witness'):
            raise AssertionError('default example must not import the optional helper or solver')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_optional_import)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    demo = example_module()
    D, guide, cycles, a, b = demo.synthetic_case()
    assert D.dtype == guide.dtype == np.float64
    assert D.shape == (12, 12) and guide.shape == (12, 2)
    assert len(cycles) == 1 and len(cycles[0]) == 8
    assert aw.analyze_sparse_h1(D, cycles, a, b).work['vertices'] == 12
    first = demo.run_example()
    assert first == demo.run_example()
    assert first['certified'] and first['source_certified']
    assert first['target_certified'] and first['planar_certified']
    assert first['embedding_shape'] == [12, 2]
    assert first['diagnostics']['inserted_vertices'] == 12
    assert first['source_rank'] == first['target_rank'] == 1
    assert first['h0_error'] <= first['h0_tolerance'] == .05
    assert demo.main([]) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and after[2:] == numpy_state[2:]
    np.testing.assert_array_equal(after[1], numpy_state[1])


def test_example_unsupported_json_no_embedding_no_retry(monkeypatch, capsys):
    import json

    demo = example_module()
    calls = []

    def unsupported(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(certified=False, embedding=None,
                                     reason='unsupported test fixture', diagnostics={})

    monkeypatch.setattr(demo, 'construct_global_layout', unsupported)
    assert demo.main([]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report['certified'] and report['embedding'] is None
    assert report['reason'] == 'unsupported test fixture'
    assert len(calls) == 1


def test_example_exception_json_nonzero_no_retry(monkeypatch, capsys):
    import json

    demo = example_module()
    calls = []

    def exhausted(*args, **kwargs):
        calls.append(1)
        raise ResourceLimitError('test resource refusal')

    monkeypatch.setattr(demo, 'construct_global_layout', exhausted)
    assert demo.main([]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report['certified'] and report['embedding'] is None
    assert report['stage'] == 'exception'
    assert 'ResourceLimitError' in report['reason']
    assert len(calls) == 1


def test_example_auto_explicit_consent_no_fallback(monkeypatch, capsys):
    import json

    demo = example_module()
    calls = []

    def refuse(D, *, allow_external):
        calls.append((D.shape, allow_external))
        return aw.WitnessResult(False, (), None, None,
                                {'reason': 'fixed proposal refused'}, None)

    def forbidden(*args, **kwargs):
        pytest.fail('auto refusal must not retry the default family')

    monkeypatch.setattr(aw, 'auto_global_witnesses', refuse)
    monkeypatch.setattr(demo, 'construct_global_layout', forbidden)
    assert demo.main(['--auto']) == 1
    report = json.loads(capsys.readouterr().out)
    assert calls == [((12, 12), True)]
    assert not report['certified'] and report['embedding'] is None
    assert report['stage'] == 'proposal' and report['reason'] == 'fixed proposal refused'
