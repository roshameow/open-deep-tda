import numpy as np
import pytest
from open_deep_tda import DeepTDA, TDAConfig
from open_deep_tda.datasets import make_dataset
from open_deep_tda import losses


def test_online_refresh_and_paired_h0_streams(monkeypatch,tmp_path):
    X=make_dataset('circle',80,6,seed=3)[0]
    config=dict(steps=8,warmup_steps=0,topology_interval=1,h0_size=12,h1_size=12,
                subset_bank_size=6,landmark_size=20,h0_refresh_every=1,h1_refresh_every=1,
                evaluation_size=12,validation_interval=20,n_neighbors=5)
    observed=[];original=losses.h0_loss
    def spy(source,*args,**kwargs):
        observed.append(source.cpu().numpy().copy())
        return original(source,*args,**kwargs)
    monkeypatch.setattr(losses,'h0_loss',spy)
    full=DeepTDA(**config).fit(X)
    source_full=observed.copy();observed.clear()
    ablated=DeepTDA(**config,lambda_h1=0).fit(X)
    assert len(observed)==len(source_full)==8
    for A,B in zip(observed,source_full):np.testing.assert_array_equal(A,B)
    c=full.training_coverage_
    assert c['h1_initial_subsets']==1
    assert c['h1_online_refreshes']==c['h0_online_refreshes']==7
    assert c['h1_unique_samples']>12
    assert ablated.training_coverage_['h1_online_refreshes']==0
    assert full.neighbor_diagnostics_['backend']=='exact'
    restored=DeepTDA.load(full.save(tmp_path/'model.pt'))
    assert restored.neighbor_diagnostics_==full.neighbor_diagnostics_
    assert restored.training_coverage_==full.training_coverage_
    np.testing.assert_array_equal(restored.transform(X),full.transform(X))


@pytest.mark.parametrize('kwargs',[{'h0_refresh_every':-1},{'h1_refresh_every':1.5},
    {'neighbor_backend':'unknown'},{'neighbor_min_recall':1.1},{'neighbor_min_recall':float('nan')},
    {'neighbor_working_memory_mb':0}])
def test_new_config_validation(kwargs):
    with pytest.raises(ValueError):TDAConfig(**kwargs).validate()
