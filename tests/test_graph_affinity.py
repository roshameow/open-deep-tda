"""Numerical source-affinity regression, not a clustering acceptance test."""
import numpy as np
import pytest

from open_deep_tda._graph_affinity import weights


def historical(d):
    target=np.log2(d.shape[1]);rho=np.min(np.where(d>0,d,np.inf),axis=1);rho[~np.isfinite(rho)]=0
    gap=np.maximum(d-rho[:,None],0);lo=np.zeros(len(d));hi=np.maximum(d.max(axis=1),1e-12)
    saturated=(gap==0).sum(axis=1)>=target
    for _ in range(64):
        mid=(lo+hi)/2;mass=np.exp(-gap/mid[:,None]).sum(axis=1)
        lo=np.where(mass<target,mid,lo);hi=np.where(mass>=target,mid,hi)
    result=np.exp(-gap/hi[:,None]);result[saturated]=gap[saturated]==0
    return result


def test_ordinary_scale_computation_is_preserved_bitwise():
    rng=np.random.default_rng(821)
    d=np.sort(rng.uniform(.01,12,size=(1000,15)),axis=1)
    assert np.array_equal(weights(d),historical(d))


@pytest.mark.parametrize('scale',[1e-300,1e-200,1e-100,1e-30,1.,1e30,1e100,1e200,1e300])
def test_uniform_scale_no_longer_turns_query_affinities_uniform(scale):
    d=np.arange(1,16,dtype=float)[None,:]
    got=weights(d*scale)
    np.testing.assert_allclose(got,weights(d),rtol=1e-10,atol=1e-12)
    assert abs(got.sum()-np.log2(15))<1e-11


def test_large_dynamic_range_inside_row_has_resolved_bandwidth():
    d=np.r_[np.arange(1,10)*1e-250,np.arange(1,7)*1e100][None,:]
    w=weights(d)
    assert np.isfinite(w).all() and abs(w.sum()-np.log2(15))<1e-11
    assert w[0,0]==1 and np.all(w[0,-6:]==0)


def test_ties_zero_rows_and_empty_queries():
    d=np.array([[0.,0.,0.,0.,1.],[1.,1.,1.,1.,2.],[0.,0.,0.,0.,0.]])
    np.testing.assert_array_equal(weights(d),historical(d))
    assert weights(np.empty((0,15))).shape==(0,15)


@pytest.mark.parametrize('bad',[np.array([[np.nan,1.]]),np.array([[np.inf,1.]]),
                               np.array([[-1.,1.]]),np.empty((2,0)),np.ones(3),
                               np.array([[True,False]]),np.array([['1','2']]),
                               np.ma.array([[1.,2.]],mask=[[False,True]])])
def test_invalid_distance_arrays_rejected(bad):
    with pytest.raises(ValueError):weights(bad)


def test_parent_query_affinities_use_shared_scale_safe_rule():
    from open_deep_tda.graph_embedding import _weights
    d=np.arange(1,16,dtype=float)[None,:]*1e-100
    np.testing.assert_array_equal(_weights(d),weights(d))


def test_standalone_worker_uses_same_rule_without_parent_torch(tmp_path):
    import subprocess,sys
    from pathlib import Path
    worker=Path(__file__).resolve().parents[1]/'python/open_deep_tda/_graph_worker.py'
    code='''
import importlib.util,sys,numpy as np
spec=importlib.util.spec_from_file_location('worker_under_test',sys.argv[1])
w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
d=np.arange(1,16,dtype=float)[None,:]*1e-100
assert abs(w.weights(d).sum()-np.log2(15))<1e-11
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code,str(worker)],check=True,capture_output=True,timeout=30)
