"""Source-only benchmark protocol/privacy checks; no real cache or fitting required."""
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.graph_core import common, kernels, publish, worker


def test_contract_counts_and_controls():
    assert common.PLAN['seeds']==[0,1,2]
    assert common.PLAN['methods']==['strong','directA','umap']
    assert len([(d,m,s) for d in common.DATASETS for m in ('pca','strong','directA','umap') for s in ([0] if m=='pca' else [0,1,2])])==30
    configs=json.loads((common.CODE/'strong_configs.json').read_text())
    assert (configs['fashion']['geometry_objective'],configs['fashion']['steps'])==('neighbor_nce',7200)
    assert (configs['har']['geometry_objective'],configs['har']['steps'])==('fuzzy_graph',2400)
    assert (configs['coil']['geometry_objective'],configs['coil']['steps'])==('stress',600)
    assert kernels.OPTIONS==dict(maxiter=100,maxls=30,ftol=1e-12,gtol=1e-8)
    assert common.umap_options(2)['n_neighbors']==16
    assert common.umap_options(2)['force_approximation_algorithm'] is True


def _barrier(tmp_path,status='completed'):
    a=tmp_path/'returned.npy';np.save(a,np.zeros((3,2)))
    obj=dict(terminal_fit_jobs=30,fit_statuses={str(i):dict(status=status) for i in range(30)},embeddings={str(a):common.sha(a)})
    (tmp_path/'embeddings_frozen.json').write_text(json.dumps(obj))
    return a,obj


def test_labels_barrier_missing_running_and_mutation(tmp_path):
    with pytest.raises(FileNotFoundError):common.require_label_barrier(tmp_path)
    path,meta=_barrier(tmp_path,'running')
    with pytest.raises(RuntimeError,match='active'):common.require_label_barrier(tmp_path)
    path,meta=_barrier(tmp_path)
    assert common.require_label_barrier(tmp_path)['terminal_fit_jobs']==30
    np.save(path,np.ones((3,2)))
    with pytest.raises(RuntimeError,match='changed'):common.require_label_barrier(tmp_path)


def test_label_loader_never_opens_labels_without_barrier(monkeypatch):
    calls=[]
    monkeypatch.setattr(worker,'require_label_barrier',lambda:(_ for _ in ()).throw(RuntimeError('not frozen')))
    monkeypatch.setattr(worker,'member',lambda *args:calls.append(args))
    with pytest.raises(RuntimeError,match='not frozen'):worker.get_labels('fashion',{})
    assert not calls


def test_registration_reads_train_member_only(tmp_path,monkeypatch):
    cache=tmp_path/'cache';cache.mkdir();run=tmp_path/'run';run.mkdir()
    train=np.arange(1200,dtype=np.float32).reshape(600,2)
    np.savez(cache/'features.npz',train=train,test=np.zeros((16,2)),train_labels=np.zeros(600))
    np.savez(cache/'reference_pca.npz',components=np.eye(2),mean=np.zeros(2),distance_scale=np.array(1.))
    (run/'synthetic_preflight.json').write_text(json.dumps(dict(status='passed')))
    (run/'strong_preflight.json').write_text(json.dumps(dict(status='passed',resolved_configs={'fashion':{'steps':1}})))
    tiny={'fashion':dict(path='cache/features.npz',train='train.npy',test='test.npy',annotations='cache/features.npz',n=600,m=16,d=2)}
    monkeypatch.setattr(worker,'OUT',run);monkeypatch.setattr(worker,'ROOT',tmp_path);monkeypatch.setattr(worker,'DATASETS',tiny)
    monkeypatch.setattr(worker,'sources',lambda:{'fixture/source':'a'*64})
    monkeypatch.setattr(worker,'environment_snapshot',lambda:{'platform':'fixture','packages':{}})
    import subprocess
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'fixture-head\n')
    seen=[];original=common.member
    def guarded(path,key):
        seen.append(key)
        assert key=='train.npy','TEST or labels read before registration'
        return original(path,key)
    monkeypatch.setattr(worker,'member',guarded)
    worker.freeze()
    reg=json.loads((run/'preregistration.json').read_text())
    assert seen==['train.npy']
    assert reg['inputs']['fashion']['train_array_sha256']==common.ah(train)
    assert len(reg['query_ids']['fashion']['train'])==512
    assert reg['query_ids']['fashion']['test']==list(range(16))
    with pytest.raises(AssertionError):worker.freeze()


def test_source_mismatch_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'OUT',tmp_path)
    (tmp_path/'preregistration.json').write_text(json.dumps(dict(plan=common.PLAN,sources={'fixture':'old'})))
    monkeypatch.setattr(common,'sources',lambda:{'fixture':'changed'})
    with pytest.raises(RuntimeError,match='source/plan'):common.registration()


def _public_fixture():
    records=[]
    for d in publish.DATASETS:
        for m in publish.METHODS:
            for s in ([0] if m=='pca' else [0,1,2]):
                records.append(dict(dataset=d,method=m,seed=s,status='completed',**{k:.5 for k in publish.METRIC_KEYS}))
    return dict(schema='graph-core-confirmation-summary-v1',records=records,summary=[dict(dataset=d,method=m,status='complete',completed=1 if m=='pca' else 3,expected=1 if m=='pca' else 3,statistics={}) for d in publish.DATASETS for m in publish.METHODS])


def test_public_aggregate_complete_counts():
    data=_public_fixture();assert publish.validate_public_aggregate(data)
    data['records'].pop()
    with pytest.raises(ValueError,match='all30'):publish.validate_public_aggregate(data)


@pytest.mark.parametrize('key',['query_ids','sample_ids','nonconverged_ids','coordinates','predictions','raw_logs','model_path','traceback'])
def test_public_aggregate_rejects_raw_fields(key):
    data=_public_fixture();data['leak']={key:[1,2]}
    with pytest.raises(ValueError):publish.validate_public_aggregate(data)


# Construct the synthetic rejected path without putting an identifying-path
# literal in the source distribution (the archive scanner also inspects tests).
@pytest.mark.parametrize('value',['/Users/'+'example/model.pt','/home/example/data','docs/research/example','/tmp/model','../model.npz'])
def test_public_aggregate_rejects_paths(value):
    data=_public_fixture();data['leak']=value
    with pytest.raises(ValueError):publish.validate_public_aggregate(data)


def test_source_has_no_research_or_personal_imports():
    for path in list(common.CODE.glob('*.py'))+[ROOT/'benchmarks/confirm_graph_core.py']:
        source=path.read_text()
        for forbidden in ('docs/','/Users/','/tmp/open-deep','sparse_graph_lane','inductive_mapping_lane'):
            # Privacy validators intentionally name forbidden tokens, never imports.
            if path.name in ('publish.py','compact_publish.py') and forbidden in ('docs/','/Users/'):continue
            assert forbidden not in source,(path.name,forbidden)


def test_output_rejects_public_trees(tmp_path):
    for path in (ROOT/'assets/run',ROOT/'benchmarks/results/raw',ROOT/'python/outputs'):
        with pytest.raises(ValueError):common.validate_output(path)
    assert common.validate_output(tmp_path)==tmp_path.resolve()


def test_exact_metrics_and_gradient_synthetic():
    from sklearn.manifold import trustworthiness
    rng=np.random.default_rng(931);X=rng.normal(size=(120,5));Z=rng.normal(size=(120,2))
    row,_=kernels.metrics(X,X,Z,Z,np.arange(120))
    assert abs(row['15']['trustworthiness']-trustworthiness(X,Z,n_neighbors=15))<1e-12
    ids=np.arange(15);p=np.ones(15)/15;y=np.array([.4,.7]);mapper=kernels.FrozenMap(Z);_,g=mapper.cauchy(y,ids,p);eps=1e-6
    numerical=np.array([(mapper.cauchy(y+eps*np.eye(2)[i],ids,p)[0]-mapper.cauchy(y-eps*np.eye(2)[i],ids,p)[0])/(2*eps) for i in range(2)])
    assert np.max(abs(g-numerical))<1e-8


def test_unknown_public_root_rejected():
    data=_public_fixture();data['unreviewed_payload']=[[1.,2.]]
    with pytest.raises(ValueError,match='root'):publish.validate_public_aggregate(data)


def test_no_invented_pca_variance_or_seed_mean():
    data=_public_fixture();row=data['summary'][0]
    row['statistics']={'test_overlap_15':dict(mean=.5,sample_std=0.,values=[.5])}
    with pytest.raises(ValueError,match='variance'):publish.validate_public_aggregate(data)
    row['statistics']['test_overlap_15']['sample_std']=None
    assert publish.validate_public_aggregate(data)
    row['statistics']['test_overlap_15']['mean']=.9
    with pytest.raises(ValueError,match='Mean'):publish.validate_public_aggregate(data)


def test_environment_change_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'OUT',tmp_path)
    (tmp_path/'preregistration.json').write_text(json.dumps(dict(plan=common.PLAN,sources={},environment={'packages':{'numpy':'before'}})))
    monkeypatch.setattr(common,'sources',lambda:{})
    monkeypatch.setattr(common,'environment_snapshot',lambda:{'packages':{'numpy':'after'}})
    with pytest.raises(RuntimeError,match='environment'):common.registration()
