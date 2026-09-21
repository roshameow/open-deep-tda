import numpy as np
import pytest
import torch
from open_deep_tda import DeepTDA, TDAConfig
from open_deep_tda.datasets import make_dataset


def small(**kwargs):
    args=dict(steps=5,warmup_steps=1,h0_size=12,h1_size=12,evaluation_size=12,
              subset_bank_size=2,validation_interval=20)
    args.update(kwargs);return DeepTDA(**args)


@pytest.mark.parametrize('mode',['geometry','semantic'])
def test_compact_checkpoint_preserves_inference_not_training_data(tmp_path,mode):
    X=make_dataset('circle',30,5,seed=4)[0]
    model=small(mode=mode,semantic_steps=3,semantic_dim=4).fit(X)
    path=model.save(tmp_path/'compact.pt',include_training_data=False)
    state=torch.load(path,weights_only=True)
    assert state['reference'] is None and state['embedding'] is None
    assert state['history']==[] and state['validation_history']==[]
    assert 'topology' not in state['report']
    loaded=DeepTDA.load(path)
    np.testing.assert_array_equal(model.transform(X),loaded.transform(X))
    assert loaded.reference_ is None and loaded.embedding_ is None
    with pytest.raises(RuntimeError,match='omitted'):loaded.ood_scores(X)
    with pytest.raises(ValueError,match='unavailable'):loaded.save(tmp_path/'wrong.pt')
    loaded.save(tmp_path/'compact-again.pt',include_training_data=False)


def test_compact_coordinates_rejected(tmp_path):
    X=make_dataset('circle',20,4,seed=1)[0]
    model=small(optimizer_mode='coordinates').fit(X)
    with pytest.raises(ValueError,match='coordinate'):model.save(tmp_path/'bad.pt',include_training_data=False)


def test_inference_restores_module_mode_and_batching():
    X=make_dataset('circle',40,5,seed=2)[0]
    model=small().fit(X)
    model.model_.train()
    model.config.inference_batch_size=7
    a=model.transform(X)
    assert model.model_.training
    model.config.inference_batch_size=4096
    b=model.transform(X)
    np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)


def test_steps_zero_skips_graph_and_ph_training_cache(monkeypatch):
    import open_deep_tda.neighbors as module
    monkeypatch.setattr(module,'build_neighbor_graph',lambda *a,**k:pytest.fail('unneeded graph'))
    X=make_dataset('circle',20,4,seed=3)[0]
    model=small(steps=0).fit(X)
    assert model.neighbor_diagnostics_['skipped']
    assert model.training_coverage_['optimizer_steps']==0
    assert model.timings_['source_cache_seconds']==0


def test_fuzzy_fit_and_roundtrip(tmp_path):
    X=make_dataset('circle',36,5,seed=7)[0]
    model=small(geometry_objective='fuzzy',fuzzy_repulsion=1,lambda_h0=.1,lambda_h1=.01).fit(X)
    assert model.report_['geometry_objective']['objective']=='fuzzy'
    assert all(np.isfinite(r['loss']) for r in model.history_)
    restored=DeepTDA.load(model.save(tmp_path/'fuzzy.pt'))
    np.testing.assert_array_equal(model.transform(X),restored.transform(X))


@pytest.mark.parametrize('kw',[{'geometry_objective':'bad'},{'fuzzy_scale':0},{'fuzzy_scale':float('nan')},
    {'fuzzy_repulsion':-1},{'inference_batch_size':0}])
def test_release_config_validation(kw):
    with pytest.raises(ValueError):TDAConfig(**kw).validate()
