"""The guided path is the SAME PH-trained DeepTDA MLP, accepted by checks only."""
import json

import numpy as np
import pytest

from open_deep_tda import DeepTDA
from open_deep_tda._ph_guided_training import GuidedLimits, GuidedTrainingUnresolved
from open_deep_tda._ph_guided_source import TeacherLimits, TeacherResult
from open_deep_tda.structural_sparse_h1 import ResourceLimitError
from open_deep_tda.structural_h0 import compare_h0
from open_deep_tda.structural_h1 import check_h1_witnesses
from open_deep_tda.topology import distance_matrix


def fixture():
    theta=2*np.pi*np.arange(16)/16
    X=np.column_stack((np.cos(theta),np.sin(theta))).astype(np.float64)
    gamma=[(i,(i+1)%16) for i in range(16)]
    model=DeepTDA(steps=6,warmup_steps=0,h0_size=16,h1_size=16,
                  topology_interval=1,subset_bank_size=1,
                  missing_indicators=False,standardize=False,seed=0)
    limits=GuidedLimits(max_vertices=16,max_feature_workspace_bytes=100000,
                        teacher_steps=0,contact_steps=0,cut_steps=0,max_seconds=90)
    tl=TeacherLimits(max_vertices=16,max_pairs=120,max_iterations=10,max_seconds=30)
    return X,gamma,model,limits,tl


def test_opt_in_same_estimator_exact_full_domain_and_safe_checkpoint(tmp_path):
    X,gamma,m,limits,tl=fixture()
    assert m.fit_with_topology_guidance(X,cycles=[gamma],birth_radius=.5,
                 survival_radius=.9,strategy='single',limits=limits,teacher_limits=tl) is m
    info=m.report_['guided_training']
    assert info['status']=='certified_selected_TRAIN_family'
    assert info['certificate']['accepted'] and info['source_witness_lengths']==[16]
    assert m.report_['status']=='guided_certified_train'
    assert m.report_['guided_training']['requested_native_ph_subcloud_size']==16
    assert m.report_['guided_training']['effective_native_ph_subcloud_size']==16
    assert m.training_coverage_['h1_updates']>0
    # The guided source metric is the frozen reference in physical source units.
    source=distance_matrix(np.asarray(m.reference_,dtype=np.float64)*m.reference_scale_)
    target=distance_matrix(np.asarray(m.embedding_,dtype=np.float64)*m.reference_scale_)
    assert compare_h0(source,target,max_vertices=16,tolerance=.05)['certified_within_tolerance']
    assert check_h1_witnesses(source,target,[gamma],.5,.9,max_vertices=16).accepted
    json.dumps(m.report_,allow_nan=False)
    path=tmp_path/'model.pt';m.save(path)
    loaded=DeepTDA.load(path)
    np.testing.assert_allclose(loaded.transform(X),m.embedding_,rtol=0,atol=1e-6)
    assert loaded.report_['guided_training']['certificate']['accepted']


def test_nonbinary_source_scale_certifies_represented_saved_model(tmp_path):
    X,gamma,m,limits,tl=fixture()
    m.fit_with_topology_guidance(X,cycles=[gamma],birth_radius=.47,
                                 survival_radius=.82,source_scale=1.1,
                                 strategy='single',limits=limits,teacher_limits=tl)
    path=tmp_path/'scaled.pt';m.save(path)
    loaded=DeepTDA.load(path)
    factor=loaded.reference_scale_/loaded.report_['guided_training']['source_scale']
    reference=np.asarray(loaded.reference_,dtype=np.float64)*factor
    source=distance_matrix(reference)
    represented=loaded.transform(X).astype(np.float64)*factor
    actual=distance_matrix(represented)
    assert check_h1_witnesses(source,actual,[gamma],.47,.82,max_vertices=16).accepted
    h=compare_h0(source,actual,max_vertices=16)['max_merge_error']
    assert h<=.05
    assert abs(h-loaded.report_['guided_training']['certificate']['h0_error'])<1e-6


def test_oversized_input_and_unbounded_cycle_refused_before_native_fit(monkeypatch):
    X,gamma,m,limits,tl=fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('must preflight before native DeepTDA.fit')
    monkeypatch.setattr(DeepTDA,'fit',forbidden)
    with pytest.raises(ResourceLimitError,match='budget'):
        m.fit_with_topology_guidance(np.zeros((17,2)),cycles=[gamma],birth_radius=.5,
                                    survival_radius=.9,limits=limits,teacher_limits=tl)
    def oversized_edges():
        for i in range(1000000):
            yield (i % 16,(i+1) % 16)
    with pytest.raises(ResourceLimitError,match='cycle exceeds'):
        m.fit_with_topology_guidance(X,cycles=[oversized_edges()],birth_radius=.5,
                                    survival_radius=.9,limits=limits,teacher_limits=tl)
    assert not m._fitted


def test_failure_does_not_replace_previously_fitted_estimator():
    X,gamma,m,limits,tl=fixture();m.fit(X)
    before=m.transform(X).copy()
    with pytest.raises(ValueError,match='requires exactly 1 source cycle'):
        m.fit_with_topology_guidance(X,cycles=[],birth_radius=.5,survival_radius=.9)
    np.testing.assert_array_equal(m.transform(X),before)
    with pytest.raises(ResourceLimitError,match='budget'):
        m.fit_with_topology_guidance(X,cycles=[gamma],birth_radius=.5,
          survival_radius=.9,limits=GuidedLimits(max_vertices=4,max_feature_workspace_bytes=100000,
                                                teacher_steps=0,contact_steps=0,cut_steps=0,max_seconds=90),
          teacher_limits=tl)
    np.testing.assert_array_equal(m.transform(X),before)


def test_unresolved_source_constructor_leaves_public_estimator_untouched(monkeypatch):
    X,gamma,m,limits,tl=fixture()
    import open_deep_tda._ph_guided_training as guided
    monkeypatch.setattr(guided, 'construct_source_teacher',
                        lambda *args, **kwargs: TeacherResult(False, None, 'unsupported source pattern', {}))
    with pytest.raises(GuidedTrainingUnresolved, match='unsupported source pattern') as caught:
        m.fit_with_topology_guidance(X,cycles=[gamma],birth_radius=.5,
                                    survival_radius=.9,limits=limits,teacher_limits=tl)
    assert caught.value.diagnostics['stage']=='source_teacher'
    assert not m._fitted and not hasattr(m,'embedding_')


def test_refuse_configuration_without_an_active_native_h1_update():
    X,gamma,_,limits,tl=fixture()
    m=DeepTDA(steps=1,warmup_steps=1,h0_size=16,h1_size=16,subset_bank_size=1,
              missing_indicators=False,standardize=False)
    with pytest.raises(ValueError,match='active PH'):
        m.fit_with_topology_guidance(X,cycles=[gamma],birth_radius=.5,
                                    survival_radius=.9,limits=limits,teacher_limits=tl)
    assert not m._fitted
