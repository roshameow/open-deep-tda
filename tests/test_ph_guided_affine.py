"""Source-only affine head: original TRAIN IDs only, no OOS assertion."""
import json

import numpy as np
import pytest
import torch

from open_deep_tda import DeepTDA
from open_deep_tda import _ph_guided_affine as affine
from open_deep_tda import _ph_guided_beam as beam
from open_deep_tda._ph_guided_source import TeacherLimits
from open_deep_tda._ph_guided_training import GuidedLimits, GuidedTrainingUnresolved
from open_deep_tda.structural_constructive import full_hierarchy, pairwise_planar
from open_deep_tda.structural_h0 import compare_h0
from open_deep_tda.structural_sparse_h1 import ResourceLimitError, analyze_sparse_h1
from scipy.spatial.distance import pdist, squareform


def fixture():
    theta = 2*np.pi*np.arange(16)/16
    rng = np.random.default_rng(12)
    X = np.column_stack((np.cos(theta), np.sin(theta),
                         rng.normal(size=(16, 16))*1e-4)).astype(np.float64)
    cycle = [[(i, (i+1)%16) for i in range(16)]]
    model = DeepTDA(steps=6, warmup_steps=0, h0_size=16, h1_size=16,
                    topology_interval=1, subset_bank_size=1, seed=0,
                    standardize=False, missing_indicators=False)
    limits = GuidedLimits(max_vertices=16, max_feature_workspace_bytes=2_000_000,
                          max_seconds=90)
    teacher = TeacherLimits(max_vertices=16, max_pairs=120,
                            max_distance_workspace_bytes=300000, max_seconds=30)
    kw = dict(cycles=cycle, birth_radius=.5, survival_radius=.9,
              strategy='single_beam', teacher_realization='affine_min_norm',
              limits=limits, teacher_limits=teacher)
    return X, model, kw


def test_full_row_rank_affine_fit_reload_and_determinism(tmp_path):
    X, first, kw = fixture()
    first.fit_with_topology_guidance(X, **kw)
    A = np.column_stack((first.reference_.astype(np.float64), np.ones(len(X))))
    assert np.linalg.matrix_rank(A, tol=np.linalg.svd(A, compute_uv=False)[0]*1e-12) == 16
    second = DeepTDA(first.config).fit_with_topology_guidance(X, **kw)
    info = first.report_['guided_training']
    assert info['rank'] == 16 and info['condition'] <= 1e12
    assert info['svd_rcond'] == 1e-12 and info['max_fitted_teacher_error'] < .01
    assert info['native_sampled_ph_activation']['h1_updates'] > 0
    assert 'no native H1 causal or OOS guarantee' in info['phase_scopes']
    np.testing.assert_array_equal(first.embedding_, second.embedding_)
    path = tmp_path/'affine.pt'; first.save(path)
    assert torch.load(path, weights_only=True)['report']['guided_training']['rank'] == 16
    loaded = DeepTDA.load(path)
    np.testing.assert_array_equal(first.embedding_, loaded.embedding_)
    np.testing.assert_array_equal(loaded.transform(X), loaded.embedding_)
    factor = loaded.reference_scale_/info['source_scale']
    source = squareform(pdist(loaded.reference_.astype(np.float64)*factor))
    target = pairwise_planar(loaded.embedding_.astype(np.float64)*factor)
    assert compare_h0(source, target, tolerance=.05, max_vertices=16)['certified_within_tolerance']
    assert analyze_sparse_h1(target, kw['cycles'], .5, .9).certified
    json.dumps(loaded.report_, allow_nan=False)


def test_preflight_and_unchanged_fitted_estimator(monkeypatch):
    X, model, kw = fixture()
    old = X[:, :2].copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('native fit must not begin')
    with monkeypatch.context() as patch:
        patch.setattr(DeepTDA, 'fit', forbidden)
        with pytest.raises(ValueError, match='dimension'):
            model.fit_with_topology_guidance(old, **kw)
        with pytest.raises(ValueError, match='explicit single_beam'):
            model.fit_with_topology_guidance(X, **dict(kw, strategy='single'))
        with pytest.raises(ResourceLimitError, match='workspace'):
            model.fit_with_topology_guidance(X, **dict(kw, limits=GuidedLimits(
                max_feature_workspace_bytes=1)))
        with pytest.raises(ValueError, match='rank/condition'):
            model.fit_with_topology_guidance(np.zeros_like(X), **kw)
        invalid = X.copy(); invalid[0, 0] = np.inf
        with pytest.raises(ValueError, match='infinite'):
            model.fit_with_topology_guidance(invalid, **kw)
        with pytest.raises(ValueError, match='h0_tolerance'):
            model.fit_with_topology_guidance(X, **dict(kw, h0_tolerance=np.nan))
        with pytest.raises(ResourceLimitError, match='max_seconds'):
            model.fit_with_topology_guidance(X, **dict(kw,
                teacher_limits=TeacherLimits(max_seconds=0)))
        with pytest.raises(ValueError, match='source unit conversion'):
            model.fit_with_topology_guidance(X, **dict(kw,
                source_scale=np.nextafter(0., 1.)))
        with pytest.raises(ValueError, match='source distances overflow'):
            model.fit_with_topology_guidance(X, **dict(kw, source_scale=1e-300))
        with pytest.raises(ResourceLimitError, match='pair-sample workspace'):
            model.fit_with_topology_guidance(X, **dict(kw, limits=GuidedLimits(
                max_feature_workspace_bytes=300000)))
        high = DeepTDA(steps=6, warmup_steps=0, h0_size=4, h1_size=4,
                       topology_interval=1, subset_bank_size=1,
                       standardize=False, missing_indicators=True)
        with pytest.raises(ValueError, match='4096 feature ceiling'):
            high.fit_with_topology_guidance(np.zeros((4, 2049)),
                cycles=[[(0,1),(1,2),(2,3),(3,0)]],birth_radius=2.,
                survival_radius=2.1, strategy='single_beam',
                teacher_realization='affine_min_norm')
    model.fit(X)
    prior = model.transform(X).copy()
    with pytest.raises(ValueError, match='dimension'):
        model.fit_with_topology_guidance(old, **kw)
    np.testing.assert_array_equal(prior, model.transform(X))


def test_native_residual_parameters_remain_bitwise_unchanged(monkeypatch):
    X, model, kw = fixture()
    native_fit = DeepTDA.fit
    snapshot = {}
    def capture(candidate, *args, **kwargs):
        result = native_fit(candidate, *args, **kwargs)
        snapshot.update({key: val.detach().clone()
                         for key, val in candidate.model_.residual.state_dict().items()})
        return result
    monkeypatch.setattr(DeepTDA, 'fit', capture)
    model.fit_with_topology_guidance(X, **kw)
    assert snapshot and all(torch.equal(val, model.model_.residual.state_dict()[key])
                            for key, val in snapshot.items())


def test_final_certificate_refusal_is_atomic(monkeypatch):
    X, model, kw = fixture()
    model.fit(X)
    prior = model.transform(X).copy()
    original = affine.analyze_sparse_h1
    from dataclasses import replace
    calls = []
    def fail_only_final(*args, **kwargs):
        calls.append(1)
        result = original(*args, **kwargs)
        return replace(result, certified=False, reason='forced') if len(calls) == 2 else result
    monkeypatch.setattr(affine, 'analyze_sparse_h1', fail_only_final)
    with pytest.raises(GuidedTrainingUnresolved, match='certificate refused') as caught:
        model.fit_with_topology_guidance(X, **kw)
    assert caught.value.diagnostics['stage'] == 'final_certificate'
    np.testing.assert_array_equal(prior, model.transform(X))


def test_affine_rechecks_source_even_if_teacher_flag_is_forged(monkeypatch):
    X, model, kw = fixture()
    model.fit(X)
    prior = model.transform(X).copy()
    def forged(*args, **kwargs):
        raise AssertionError('an invalid source must be refused before teacher access')
    monkeypatch.setattr(affine, 'construct_source_teacher', forged)
    with pytest.raises(GuidedTrainingUnresolved, match='source selected H1 family') as caught:
        model.fit_with_topology_guidance(X, **dict(kw, birth_radius=.3))
    assert caught.value.diagnostics['stage'] == 'source_contract'
    np.testing.assert_array_equal(prior, model.transform(X))


def test_one_sided_seed_shortcut_monotonicity():
    # A source-only shortcut through IDs 3 and 4 lowers U[0,2] from 4 to 2.
    source = np.array([[0.,0.], [4.,0.], [8.,0.], [2.,0.], [6.,0.]])
    ids = [0,1,2]
    U = full_hierarchy(pairwise_planar(source))
    V = full_hierarchy(pairwise_planar(source[ids]))
    delta = .98*.05
    slack = 128*np.finfo(float).eps*8
    assert U[0,2] == 2 and V[0,2] == 4
    assert beam._seed_compatible(V, U, ids, delta, slack)
    collapsed = full_hierarchy(pairwise_planar(np.array([[0.,0.],[.1,0.],[.2,0.]])))
    assert not beam._seed_compatible(collapsed, U, ids, delta, slack)
    assert not np.allclose(V, np.maximum(0., U[np.ix_(ids,ids)]-delta), atol=slack)
