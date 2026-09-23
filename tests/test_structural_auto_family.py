"""Offline synthetic source-only family checks; private integration is opt-in.

Run: python -m pytest -q tests/test_structural_auto_family.py
Opt-in private external workload (prefer OS-isolated process; never co-import
Torch and Numba): TDA_PRIVATE_K4_FAMILY=1 python -m pytest -q \
    tests/test_structural_auto_family.py -k private_k4
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from open_deep_tda import structural_auto_family as af
from open_deep_tda.structural_sparse_h1 import ResourceLimitError, analyze_sparse_h1


def three_squares():
    D = np.full((12, 12), 10., dtype=np.float64)
    np.fill_diagonal(D, 0.)
    for base in (0, 4, 8):
        for u, v in ((0, 1), (1, 2), (2, 3), (0, 3)):
            D[base+u, base+v] = D[base+v, base+u] = 1.
        for u, v in ((0, 2), (1, 3)):
            D[base+u, base+v] = D[base+v, base+u] = 3.
    return D


def install(monkeypatch, bars=None, cochains=None):
    if bars is None:
        bars = [(1., 3.)] * 3
    bars = np.asarray(bars, dtype=np.float64).reshape(-1, 2)
    if cochains is None:
        cochains = [np.array([[4*i, 4*i+1, 1]], dtype=np.int64)
                    for i in range(len(bars))]
    calls = []

    def ripser(D, **kwargs):
        calls.append((D.copy(), kwargs))
        return {'dgms': [np.empty((0, 2)), bars], 'cocycles': [[], cochains]}

    monkeypatch.setitem(sys.modules, 'ripser', types.SimpleNamespace(ripser=ripser))
    return calls


def refused(result, reason):
    assert not result.certified and result.cycles == () and result.source is None
    assert reason in result.diagnostics['reason']


def test_three_original_row_cycles_and_independent_exact_source(monkeypatch):
    calls = install(monkeypatch)
    D = three_squares()
    original = D.copy()
    result = af.auto_source_h1_family(D, allow_external=True)
    assert result.certified, result.diagnostics
    assert result.birth == 1.4 and result.survival == 2.6
    assert result.source.certified and result.source.rank == 3
    assert result.source.work['vertices'] == len(D)
    assert len(result.cycles) == 3
    assert result.cycles == tuple(tuple(sorted(((base, base+1), (base+1, base+2),
                                                  (base+2, base+3), (base, base+3))))
                                  for base in (0, 4, 8))
    assert analyze_sparse_h1(D, result.cycles, result.birth, result.survival).rank == 3
    assert len(calls) == 1 and calls[0][1] == dict(distance_matrix=True, maxdim=1,
                                                  coeff=2, do_cocycles=True)
    np.testing.assert_array_equal(calls[0][0], original)
    np.testing.assert_array_equal(D, original)


def test_no_consent_input_caps_before_external(monkeypatch):
    calls = install(monkeypatch)
    with pytest.raises(ValueError, match='allow_external=True'):
        af.auto_source_h1_family(three_squares())
    with pytest.raises(ResourceLimitError, match='max_vertices'):
        af.auto_source_h1_family(three_squares(), allow_external=True,
                                 limits=replace(af.FamilyLimits(), max_vertices=11))
    with pytest.raises(ValueError, match='max_vertices'):
        af.auto_source_h1_family(three_squares(), allow_external=True,
                                 limits=replace(af.FamilyLimits(), max_vertices=301))
    assert calls == []


def test_preflight_refuses_unconditional_caps_without_external_call(monkeypatch):
    calls = install(monkeypatch)
    D = three_squares()
    with pytest.raises(ResourceLimitError, match='max_edges'):
        af.auto_source_h1_family(D, allow_external=True,
                                 limits=replace(af.FamilyLimits(), max_edges=5))
    with pytest.raises(ResourceLimitError, match='max_operations'):
        af.auto_source_h1_family(D, allow_external=True,
                                 limits=replace(af.FamilyLimits(), max_operations=143))
    assert calls == []


def test_triangle_limit_precedes_early_bad_cochain_refusal(monkeypatch):
    calls = install(monkeypatch)
    D = three_squares()
    D[0, 2] = D[2, 0] = 1.  # First triangle pairs oddly; more follow it.
    with pytest.raises(ResourceLimitError, match='max_triangles'):
        af.auto_source_h1_family(D, allow_external=True,
                                 limits=replace(af.FamilyLimits(), max_triangles=1))
    assert len(calls) == 1


def test_fixed_top_three_and_empty_common_interval_never_retry(monkeypatch):
    # Three individually valid 20%-80% interiors with no triple intersection.
    calls = install(monkeypatch, bars=[(1., 11.), (2., 10.), (9., 12.), (1., 2.)])
    result = af.auto_source_h1_family(three_squares(), allow_external=True)
    assert result.diagnostics['bar_indices'] == (0, 1, 2)
    refused(result, 'no_common_fixed_interior')
    assert len(calls) == 1


def test_longest_ties_and_no_alternate_bar_after_invalid_top_cochain(monkeypatch):
    install(monkeypatch, bars=[(1., 3.), (.5, 2.5), (1., 3.), (1., 2.)],
            cochains=[np.array([[0, 1, 1]]), np.empty((0, 3), dtype=np.int64),
                      np.array([[8, 9, 1]]), np.array([[4, 5, 1]])])
    result = af.auto_source_h1_family(three_squares(), allow_external=True)
    assert result.diagnostics['bar_indices'] == (1, 0, 2)
    refused(result, 'no_three_independent')


def test_exact_source_rejection_never_exposes_partial_family(monkeypatch):
    install(monkeypatch)
    real = af.analyze_sparse_h1
    def reject(*args, **kwargs):
        return replace(real(*args, **kwargs), certified=False, reason='forced_rejection')
    monkeypatch.setattr(af, 'analyze_sparse_h1', reject)
    result = af.auto_source_h1_family(three_squares(), allow_external=True)
    assert not result.certified and result.cycles == ()
    assert result.source is not None and not result.source.certified
    assert result.diagnostics['reason'] == 'exact_source_rejected: forced_rejection'


def test_shortage_and_malformed_cochain_fail_closed(monkeypatch):
    install(monkeypatch, bars=[(1., 3.), (1., 3.), (2., np.inf)])
    refused(af.auto_source_h1_family(three_squares(), allow_external=True), 'fewer_than_three')
    install(monkeypatch, cochains=[np.array([[0, 1, 1]]),
                                  np.array([[4, 99, 1]]), np.array([[8, 9, 1]])])
    refused(af.auto_source_h1_family(three_squares(), allow_external=True), 'malformed_selected_cochain')


def test_outside_row_triangle_annihilation(monkeypatch):
    install(monkeypatch)
    D = three_squares()
    # The extra original row closes an active edge into a survival triangle.
    D[0, 4] = D[4, 0] = 1.
    D[1, 4] = D[4, 1] = 1.
    refused(af.auto_source_h1_family(D, allow_external=True), 'cochain_not_closed')


def test_dependent_cochains_no_partial_family(monkeypatch):
    install(monkeypatch, cochains=[np.array([[0, 1, 1]])] * 3)
    refused(af.auto_source_h1_family(three_squares(), allow_external=True),
            'no_three_independent')


@pytest.mark.parametrize('name,value', [('max_edges', 3), ('max_triangles', 0),
                                        ('max_operations', 0), ('max_word_ops', 0),
                                        ('max_storage_words', 0), ('max_seconds', 0)])
def test_budget_refusals(monkeypatch, name, value):
    install(monkeypatch)
    D = three_squares()
    if name == 'max_triangles':
        D[0, 2] = D[2, 0] = 1.
    with pytest.raises(ResourceLimitError, match=name):
        af.auto_source_h1_family(D, allow_external=True,
                                 limits=replace(af.FamilyLimits(), **{name: value}))


@pytest.mark.skipif(os.getenv('TDA_PRIVATE_K4_FAMILY') != '1',
                    reason='ignored private K4 + external Ripser require explicit opt-in')
def test_private_k4_original_id_source_certificate():
    pytest.importorskip('ripser', reason='external Ripser is not installed')
    from scipy.spatial.distance import pdist, squareform
    root = Path(__file__).resolve().parents[1]
    # Ignore-only contract locates source coordinates. Never inspect target,
    # labels, fixed cycle IDs, or private implementation artifacts.
    contract = json.loads((root / 'docs/research/k4_same_cycle_audit/source_contract.json').read_text())
    source = np.load(contract['paths']['source'], allow_pickle=False)
    if isinstance(source, np.lib.npyio.NpzFile):
        with source:
            X = source['X']
    else:
        X = source
    D = squareform(pdist(X)).astype(np.float64)
    result = af.auto_source_h1_family(D, allow_external=True)
    assert result.certified, result.diagnostics
    assert len(result.cycles) == 3 and result.source.rank == 3
    assert result.source.work['vertices'] == len(X)
    assert analyze_sparse_h1(D, result.cycles, result.birth, result.survival,
                             limits=af.H1Limits(max_vertices=300, max_edges=12_000,
                                                max_triangles=250_000,
                                                max_word_ops=500_000_000,
                                                max_storage_words=2_000_000,
                                                max_cycles=3, max_chain_edges=900)).certified
