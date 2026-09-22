"""Synthetic protocol/privacy checks for the separate compact mapping followup."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.graph_core import compact,compact_publish,kernels,support
from benchmarks import confirm_graph_core


def test_historical_kernel_is_explicitly_unbounded():
    text=(ROOT/'benchmarks/graph_core/kernels.py').read_text()
    assert 'CompactMap' not in text and 'train_hull' not in text
    assert kernels.OPTIONS==dict(maxiter=100,maxls=30,ftol=1e-12,gtol=1e-8)
    assert compact.PLAN['mapping_domain']=='train_hull'
    assert len(compact.ARMS)==9


def test_production_mapping_helper_loaded_without_package_import(monkeypatch):
    before=set(sys.modules)
    m=support.production_graph_mapping()
    assert m.__file__==str(ROOT/'python/open_deep_tda/_graph_mapping.py')
    assert callable(m.CompactMap.solve)
    assert not any(n.startswith('open_deep_tda') for n in set(sys.modules)-before)


def test_refinement_retains_inside_bytes_and_every_query():
    anchors=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    old=np.array([[.25,.25],[100.,-30.],[.6,.4]])
    physical=old*3+np.array([2.,-4.]);ids=np.tile(np.arange(4),(3,1));p=np.full((3,4),.25)
    final,norm,inside,records=compact.refine_cached(old,physical,anchors,np.array([2.,-4.]),3.,ids,p)
    assert final.shape==physical.shape and final[inside].tobytes()==physical[inside].tobytes()
    assert inside.tolist()==[True,False,True] and len(records)==1
    mapper=support.production_graph_mapping().CompactMap(anchors)
    assert all(mapper.contains(y) for y in norm)
    assert records[0]['phase']=='constrained_phase_II'


def test_finite_failed_incumbent_retained_not_retried():
    class Mapper:
        calls=0
        def contains(self,y):return bool(np.max(abs(y))<=1)
        def solve(self,old,ids,p,bary):
            self.calls+=1
            return bary.copy(),dict(solver_success=False,phase='constrained_phase_II')
    obj=Mapper();N=np.array([[0.,0.],[1.,0.],[0.,1.]])
    old=np.array([[99.,99.]]);ids=np.array([[0,1,2]]);p=np.full((1,3),1/3)
    Y,norm,inside,records=compact.refine_cached(old,old.copy(),N,np.zeros(2),1.,ids,p,obj)
    assert obj.calls==1 and len(Y)==1 and not records[0]['solver_success'] and np.isfinite(Y).all()


def test_compact_barrier_missing_incomplete_changed(tmp_path):
    with pytest.raises(FileNotFoundError):compact.require_barrier(tmp_path)
    data=tmp_path/'test.npy';np.save(data,np.ones((4,2)))
    b=dict(terminal_arms=8,complete_finite_arms=8,outputs={'test.npy':compact.sha(data)})
    (tmp_path/'embeddings_frozen.json').write_text(json.dumps(b))
    with pytest.raises(RuntimeError,match='All9'):compact.require_barrier(tmp_path)
    b.update(terminal_arms=9,complete_finite_arms=9);(tmp_path/'embeddings_frozen.json').write_text(json.dumps(b))
    assert compact.require_barrier(tmp_path)['terminal_arms']==9
    np.save(data,np.zeros((4,2)))
    with pytest.raises(RuntimeError,match='changed'):compact.require_barrier(tmp_path)


def test_compact_score_cannot_read_labels_before_barrier(monkeypatch):
    calls=[]
    monkeypatch.setattr(compact,'require_barrier',lambda:(_ for _ in ()).throw(RuntimeError('not ready')))
    monkeypatch.setattr(compact,'member',lambda *a:calls.append(a))
    with pytest.raises(RuntimeError,match='not ready'):compact.score('unused','fashion',0,{})
    assert not calls


def fixture():
    records=[]
    for d,n in [('fashion',10000),('har',2947),('coil',480)]:
        for seed in (0,1,2):
            metrics={str(k):dict(overlap=.5,trustworthiness=.9,continuity=.9) for k in (5,15,50)}
            records.append(dict(dataset=d,seed=seed,status='completed',mapping_domain='train_hull',train_metrics=metrics,test_metrics=metrics,query_count=n,outside_count=1,inside_count=n-1,inside_bytes_unchanged=True))
    return dict(schema='graph-core-compact-confirmation-v2',records=records,summary=[dict(dataset=d,completed=3,expected=3,statistics={}) for d in ('fashion','har','coil')],totals=dict(queries=40281,outside_refined=9))


def test_compact_counts_and_no_relabel():
    v=fixture();assert compact_publish.validate_compact_public(v)
    v['records'][0]['mapping_domain']='unbounded'
    with pytest.raises(ValueError,match='relabel'):compact_publish.validate_compact_public(v)
    v=fixture();v['records'].pop()
    with pytest.raises(ValueError,match='nine'):compact_publish.validate_compact_public(v)


@pytest.mark.parametrize('key',['outside_ids','query_index','raw_outputs','nonconverged_ids','path'])
def test_compact_privacy_fields(key):
    v=fixture();v['records'][0][key]=[1,2]
    with pytest.raises(ValueError):compact_publish.validate_compact_public(v)


# Keep the synthetic rejecting case without an identifying-path literal in archives.
@pytest.mark.parametrize('path',['/Users/'+'example/file','docs/research/run','/tmp/data'])
def test_compact_privacy_paths(path):
    v=fixture();v['protocol']={'leak':path}
    with pytest.raises(ValueError):compact_publish.validate_compact_public(v)


def test_cli_requires_prior_for_compact_registration(tmp_path):
    with pytest.raises(SystemExit):confirm_graph_core.main(['--preregister','--protocol','compact-v2','--output',str(tmp_path/'new')])
    assert not (tmp_path/'new').exists()


def test_cli_historical_run_remains_legacy_default(tmp_path,monkeypatch):
    (tmp_path/'preregistration.json').write_text(json.dumps({'plan':{'protocol':'graph-core-official-confirmation-public-port-v1'}}))
    calls=[]
    monkeypatch.setattr(confirm_graph_core.subprocess,'call',lambda args,**kw:calls.append(args) or 0)
    assert confirm_graph_core.main(['--run','--output',str(tmp_path)])==0
    assert calls[0][1].endswith('/runner.py')
    with pytest.raises(SystemExit):confirm_graph_core.main(['--run','--protocol','compact-v2','--output',str(tmp_path)])


def test_cli_compact_run_uses_separate_supervisor(tmp_path,monkeypatch):
    (tmp_path/'preregistration.json').write_text(json.dumps({'plan':{'protocol':'graph-core-compact-followup-public-v2'}}))
    calls=[];monkeypatch.setattr(confirm_graph_core.subprocess,'call',lambda args,**kw:calls.append(args) or 0)
    assert confirm_graph_core.main(['--run','--protocol','compact-v2','--output',str(tmp_path)])==0
    assert calls[0][1].endswith('/compact.py') and calls[0][2]=='supervise'


def test_compact_registration_hashes_old_arrays_without_label_reads(tmp_path,monkeypatch):
    prior=tmp_path/'prior';out=tmp_path/'followup';prior.mkdir();out.mkdir()
    queries={d:dict(train=[0],test=[0]) for d in ('fashion','har','coil')}
    (prior/'preregistration.json').write_text(json.dumps(dict(query_ids=queries)))
    hashes={}
    for d,s in compact.ARMS:
        directory=prior/d/f'directA-seed{s}';directory.mkdir(parents=True)
        for filename in ('fit.npy','test.npy','A_normalized.npy'):np.save(directory/filename,np.zeros((2,2)))
        np.savez(directory/'model.npz',normalized_fit=np.zeros((2,2)),center=np.zeros(2),unit=1.)
        (directory/'query_diagnostics.jsonl').write_text('{}\n')
        (directory/'query_metrics.json').write_text('{}')
        result={'status':'completed'}
        for name in ('fit','test'):
            h=compact.sha(directory/(name+'.npy'));hashes[str(directory/(name+'.npy'))]=h;result[name+'_sha256']=h
        (directory/'result.json').write_text(json.dumps(result))
        graph=prior/d/f'graph{s}';graph.mkdir();np.savez(graph/'neighbors.npz',query_ids=np.zeros((2,1),int))
    for d in ('fashion','har','coil'):
        np.save(prior/d/'train.npy',np.zeros((2,2)));np.save(prior/d/'test.npy',np.zeros((2,2)))
    (prior/'embeddings_frozen.json').write_text(json.dumps(dict(terminal_fit_jobs=30,embeddings=hashes)))
    monkeypatch.setattr(compact,'OUT',out);monkeypatch.setattr(compact,'sources',lambda:{'fixture':'a'*64});monkeypatch.setattr(compact,'environment_snapshot',lambda:{'fixture':True})
    monkeypatch.setattr(compact,'descriptor',lambda p:{'metadata_only':True})
    monkeypatch.setattr(compact,'member',lambda *a:(_ for _ in ()).throw(AssertionError('No label/member access during cached registration')))
    (out/'compact_preflight.json').write_text(json.dumps(dict(status='passed',source_hashes=compact.sources())))
    compact.freeze(prior)
    reg=json.loads((out/'preregistration.json').read_text());assert reg['labels_accessed'] is False
    assert 'fashion/directA-seed0/test.npy' in reg['original_inputs']
    assert not any('evaluation_predictions' in p for p in reg['original_inputs'])
    assert compact.check(reg)['unchanged']
    np.save(prior/'fashion/directA-seed0/test.npy',np.ones((2,2)))
    with pytest.raises(RuntimeError,match='cached'):compact.check(reg)


def test_compact_renderer_has_no_axis_cropping_or_point_clipping():
    import inspect
    from benchmarks.graph_core import compact_figures
    text=inspect.getsource(compact_figures.render_compact)
    assert 'set_xlim' not in text and 'set_ylim' not in text and 'np.clip' not in text
    assert len(compact_figures.LABEL_NAMES['fashion'])==10
    assert len(compact_figures.LABEL_NAMES['har'])==6
    assert len(compact_figures.LABEL_NAMES['coil'])==20


def test_unknown_compact_record_payload_rejected():
    v=fixture();v['records'][0]['unreviewed_matrix']=[[0.,1.]]
    with pytest.raises(ValueError,match='record'):compact_publish.validate_compact_public(v)
