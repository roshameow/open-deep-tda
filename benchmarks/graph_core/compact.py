"""Separate fixed nine-arm TRAIN-hull mapping-only followup; never refits layouts."""
import os
os.environ['BLIS_NUM_THREADS']='1'
if __package__:
    from .common import ROOT,CODE,OUT,DATASETS,sha,dump,now,descriptor,member,environment_snapshot
    from . import kernels
    from .support import production_graph_mapping
else:
    from common import ROOT,CODE,OUT,DATASETS,sha,dump,now,descriptor,member,environment_snapshot
    import kernels
    from support import production_graph_mapping
import sys,json,time,traceback,shutil,subprocess,signal
from pathlib import Path
import numpy as np

ARMS=[(d,s) for d in ('fashion','har','coil') for s in (0,1,2)]
PLAN=dict(protocol='graph-core-compact-followup-public-v2',mapping_domain='train_hull',arms=[dict(dataset=d,seed=s) for d,s in ARMS],
    prior='Completed unbounded-v1 directA arrays/normalization/source15; no original record overwritten, no layout/graph fitting or phase-I rerun',
    selection='Every outside-TRAIN-hull cached normalized phase-I candidate refined by exact production CompactMap.solve(old,ids,p,bary); all inside physical rows copied byte-for-byte. Fixed source probabilities and TRAIN unit. No seed selection, retries, query deletion, clipping or display restriction.',
    solver='Exact production helper pinned by hash. SLSQP maxiter100,ftol1e-12,maxcalls2000, constrained to closed TRAIN convex hull; original barycenter is a feasible incumbent. Tolerance64*eps*(1+max TRAIN anchor norm). Local solution, not global optimum. Finite feasible incumbents retained on solver failure.',
    labels='No annotation or saved classification/cluster prediction access before new all9 complete finite output/hash barrier. Prior arrays hashed before refinement; source/input hashes rechecked before barrier and closeout.',
    metrics='Reuse unchanged TRAIN geometry. Same prior512 TRAIN/1024 TEST queries (COIL480) vsFULLTRAIN, k5/15/50; allTEST15NN. Reuse original TRAIN KMeans centers and verify every old TEST cluster assignment before scoring. COIL physical pose adjacency unchanged.',
    timing='Original unbounded transform timings are historical, plus outside-only refinement separately. Validation/scoring/I/O separate. NOT a fresh full train_hull transform timing.',
    failure='All9 generation/scoring records including failures. No scoring if any output arm incomplete/nonfinite. No retry. 600s outer wall cap,1 numerical thread; no RSS-cap claim.',
    scope='Authorized well-posedness followup motivated by TRAIN-internal validation; official TEST previously seen, no virgin-test/topology/global-optimum claim.')

def arm(d,s):return f'{d}-seed{s}'
def prior_arm(prior,d,s):return Path(prior)/d/f'directA-seed{s}'
def sources():
    files={f'benchmark/{n}':CODE/n for n in ('compact.py','support.py','common.py','kernels.py')}
    files.update({'entrypoint':ROOT/'benchmarks/confirm_graph_core.py','production/mapping':ROOT/'python/open_deep_tda/_graph_mapping.py','production/predictor':ROOT/'python/open_deep_tda/graph_embedding.py'})
    return {k:sha(p) for k,p in files.items()}

def refine_cached(normalized,physical,anchors,center,unit,ids,p,mapper=None):
    """Exact outside-only routing; returns every row, never clips or drops a point."""
    mapper=production_graph_mapping().CompactMap(anchors) if mapper is None else mapper
    inside=np.array([mapper.contains(y) for y in normalized]);outside=np.flatnonzero(~inside)
    final=physical.copy();norm=normalized.copy();records=[]
    bary=np.einsum('ij,ijk->ik',p,anchors[ids])
    for i in outside:
        started=time.perf_counter()
        y,record=mapper.solve(normalized[i],ids[i],p[i],bary[i])
        record['seconds']=time.perf_counter()-started
        if not np.isfinite(y).all() or not mapper.contains(y):raise RuntimeError('No finite feasible candidate; arm fails without substitution')
        norm[i]=y;final[i]=y*unit+center;records.append(dict(record,query_index=int(i)))
    if final[inside].tobytes()!=physical[inside].tobytes():raise RuntimeError('Inside physical bytes changed')
    return final,norm,inside,records

def preflight():
    helper=production_graph_mapping()
    assert helper.MAXITER==100 and helper.MAXCALLS==2000,'Phase-II settings differ from fixed protocol'
    anchors=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    old=np.array([[.25,.25],[100.,-40.],[.8,.6]]);ids=np.tile(np.arange(4),(3,1));p=np.full((3,4),.25)
    physical=old*2+np.array([3.,-2.]);new,norm,inside,records=refine_cached(old,physical,anchors,np.array([3.,-2.]),2.,ids,p)
    assert inside.tolist()==[True,False,True] and np.array_equal(new[inside],physical[inside])
    assert all(helper.CompactMap(anchors).contains(y) for y in norm) and len(records)==1
    assert records[0]['phase']=='constrained_phase_II'
    dump(OUT/'compact_preflight.json',dict(status='passed',synthetic_only=True,inside_bitwise=True,all_rows_retained=True,exact_production_helper=True,source_hashes=sources(),utc=now()))

def freeze(prior):
    prior=Path(prior).resolve()
    if prior==OUT or prior in OUT.parents or OUT in prior.parents:raise ValueError('Followup must be a separate sibling/unrelated output directory')
    assert not (OUT/'preregistration.json').exists()
    pre=json.loads((OUT/'compact_preflight.json').read_text());assert pre['status']=='passed' and pre['source_hashes']==sources()
    oldreg=json.loads((prior/'preregistration.json').read_text());oldbarrier=json.loads((prior/'embeddings_frozen.json').read_text())
    assert oldbarrier['terminal_fit_jobs']==30
    paths=[prior/'preregistration.json',prior/'embeddings_frozen.json']
    for d in DATASETS:paths.extend([prior/d/'train.npy',prior/d/'test.npy'])
    for d,s in ARMS:
        a=prior_arm(prior,d,s)
        paths.extend(a/f for f in ('model.npz','fit.npy','test.npy','A_normalized.npy','query_diagnostics.jsonl','result.json','query_metrics.json'))
        paths.append(prior/d/f'graph{s}'/'neighbors.npz')
        fit_result=json.loads((a/'result.json').read_text());assert fit_result['status']=='completed'
        for name in ('fit','test'):
            path=a/(name+'.npy');actual=sha(path)
            assert actual==fit_result[name+'_sha256']
            # Barrier may have been written under a different absolute path.
            suffix=f'{d}/directA-seed{s}/{name}.npy'
            matches=[h for p,h in oldbarrier['embeddings'].items() if Path(p).as_posix().endswith(suffix)]
            assert matches==[actual],'Original embedding-barrier provenance mismatch'
    inputs={str(p.relative_to(prior)):sha(p) for p in sorted(set(paths))}
    annotations={d:descriptor(ROOT/spec['annotations']) for d,spec in DATASETS.items()}
    registration=dict(utc=now(),plan=PLAN,prior_output=str(prior),sources=sources(),original_inputs=inputs,annotation_descriptors=annotations,query_ids=oldreg['query_ids'],environment=environment_snapshot(),labels_accessed=False)
    with (OUT/'preregistration.json').open('x') as f:json.dump(registration,f,indent=2);f.write('\n')

def check(reg):
    if reg['plan']!=PLAN or reg['sources']!=sources():raise RuntimeError('Frozen compact source/plan changed')
    if reg['environment']!=environment_snapshot():raise RuntimeError('Frozen numerical environment changed')
    prior=Path(reg['prior_output'])
    changed=[p for p,h in reg['original_inputs'].items() if sha(prior/p)!=h]
    if changed:raise RuntimeError('Original cached arrays/source metadata changed')
    return dict(unchanged=True,source_hashes=sources(),original_inputs_checked=len(reg['original_inputs']),utc=now())

def generate(prior,d,s):
    a=prior_arm(prior,d,s);out=OUT/arm(d,s);out.mkdir();start=time.perf_counter()
    Z=np.load(a/'fit.npy');physical=np.load(a/'test.npy');old=np.load(a/'A_normalized.npy')
    with np.load(a/'model.npz') as f:N,center,unit=f['normalized_fit'],f['center'],float(f['unit'])
    assert np.array_equal(N,(Z-center)/unit) and np.array_equal(physical,old*unit+center)
    with np.load(Path(prior)/d/f'graph{s}'/'neighbors.npz') as f:ids,dist=f['query_ids'],f['query_distances']
    p=kernels.weights(dist);zero=dist==0;p[zero.any(1)]=zero[zero.any(1)];p/=p.sum(1)[:,None]
    mapper=production_graph_mapping().CompactMap(N);original=[json.loads(l) for l in (a/'query_diagnostics.jsonl').read_text().splitlines()];assert len(original)==len(old)
    t=time.perf_counter();errors=[]
    with (out/'phase_I_validation.jsonl').open('x') as log:
        for i,r in enumerate(original):
            f,g=mapper.cauchy(old[i],ids[i],p[i]);df=abs(f-r['final_objective']);dg=abs(float(max(abs(g)))-r['gradient_linf']);errors.append((df,dg));log.write(json.dumps(dict(query_index=i,objective_difference=df,gradient_linf_difference=dg))+'\n')
    validation_seconds=time.perf_counter()-t
    if np.max(errors)>1e-8:raise RuntimeError('Cached phase-I objective/gradient mismatch; no repair')
    t=time.perf_counter();final,norm,inside,records=refine_cached(old,physical,N,center,unit,ids,p,mapper);routing_seconds=time.perf_counter()-t
    refinement_seconds=sum(r['seconds'] for r in records)
    assert np.isfinite(final).all() and all(mapper.contains(y) for y in norm)
    shutil.copyfile(a/'fit.npy',out/'fit.npy');np.save(out/'test.npy',final);np.save(out/'normalized.npy',norm);np.save(out/'outside_ids.npy',np.flatnonzero(~inside))
    with (out/'refinement.jsonl').open('x') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    result=dict(dataset=d,seed=s,status='completed',query_count=len(old),outside_count=int((~inside).sum()),inside_bytes_unchanged=True,all_final_finite=True,all_final_feasible=True,old_max_norm=float(np.linalg.norm(old,axis=1).max()),final_max_norm=float(np.linalg.norm(norm,axis=1).max()),max_final_violation=max(mapper.violation(y) for y in norm),tolerance=mapper.tolerance,
        validation=dict(all_queries=len(old),seconds=validation_seconds,max_objective_difference=float(np.max(errors,axis=0)[0]),max_gradient_linf_difference=float(np.max(errors,axis=0)[1]),passed=True),
        original_unbounded_transform_seconds=json.loads((a/'result.json').read_text())['transform_seconds'],original_nonconverged_ids=[r['query_index'] for r in original if not r['success']],refinement_seconds=refinement_seconds,routing_and_refinement_seconds=routing_seconds,refinement_timing_scope='sum of outside-only production solve calls; routing/copy overhead reported separately',refinement_nonconverged_ids=[r['query_index'] for r in records if not r['solver_success']],refinement_objective_calls=sum(r['objective_calls'] for r in records),generation_seconds=time.perf_counter()-start)
    dump(out/'generation.json',result);return result

def make_barrier(records,reg):
    expected={(d,s) for d,s in ARMS}
    assert len(records)==9 and {(r['dataset'],r['seed']) for r in records}==expected
    assert all(r['status']=='completed' and r['all_final_finite'] for r in records),'No label access for incomplete outputs'
    check(reg);outputs={}
    for d,s in ARMS:
        for name in ('fit.npy','test.npy','normalized.npy','outside_ids.npy'):
            p=OUT/arm(d,s)/name;outputs[str(p.relative_to(OUT))]=sha(p)
    barrier=dict(utc=now(),terminal_arms=9,complete_finite_arms=9,labels_accessed=False,outputs=outputs,registration_sha256=sha(OUT/'preregistration.json'))
    dump(OUT/'embeddings_frozen.json',barrier);return barrier

def require_barrier(directory=OUT):
    directory=Path(directory);b=json.loads((directory/'embeddings_frozen.json').read_text())
    if b.get('terminal_arms')!=9 or b.get('complete_finite_arms')!=9:raise RuntimeError('All9 finite output arms required before labels')
    for p,h in b['outputs'].items():
        path=Path(p);path=path if path.is_absolute() else directory/path
        if sha(path)!=h:raise RuntimeError('Compact output changed after label barrier')
    return b

def pose(prior,Z,Y):
    spec=DATASETS['coil'];angles,_=member(ROOT/spec['annotations'],'angles_degrees.npy');objects,_=member(ROOT/spec['annotations'],'object_ids.npy')
    from scipy.spatial.distance import cdist
    def measure(A):
        recalls=[];diffs=[]
        for o in np.unique(objects):
            ids=np.flatnonzero(objects==o);theta=angles[ids]%360;order=np.argsort(theta,kind='stable');adj={int(order[i]):{int(order[(i-1)%len(ids)]),int(order[(i+1)%len(ids)])} for i in range(len(ids))};D=cdist(A[ids],A[ids]);np.fill_diagonal(D,np.inf);nn=np.argsort(D,axis=1,kind='stable')[:,:2]
            for i in np.flatnonzero(ids>=len(Z)):
                recalls.append(len(set(nn[i])&adj[int(i)])/2);dt=abs(theta[nn[i]]-theta[i]);diffs.extend(np.minimum(dt,360-dt).tolist())
        return dict(adjacent_view_recall=float(np.mean(recalls)),mean_neighbor_angle_degrees=float(np.mean(diffs)),queries=len(recalls))
    X=np.load(Path(prior)/'coil/train.npy');V=np.load(Path(prior)/'coil/test.npy')
    return dict(reference=measure(np.vstack((X,V))),embedding=measure(np.vstack((Z,Y))))

def score(prior,d,s,reg):
    require_barrier();spec=DATASETS[d]
    assert descriptor(ROOT/spec['annotations'])==reg['annotation_descriptors'][d]
    a=prior_arm(prior,d,s);out=OUT/arm(d,s);Z=np.load(out/'fit.npy');Y=np.load(out/'test.npy');oldY=np.load(a/'test.npy');X=np.load(Path(prior)/d/'train.npy');V=np.load(Path(prior)/d/'test.npy')
    access=dict(utc=now(),barrier_sha256=sha(OUT/'embeddings_frozen.json'));dump(out/'label_access.json',access)
    yt,train_label_hash=member(ROOT/spec['annotations'],spec['train_labels']);yv,test_label_hash=member(ROOT/spec['annotations'],spec['test_labels'])
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,adjusted_rand_score,normalized_mutual_info_score
    from sklearn.metrics.pairwise import pairwise_distances_argmin
    pred=KNeighborsClassifier(n_neighbors=15,weights='uniform',p=2,n_jobs=1).fit(Z,yt).predict(Y)
    with np.load(a/'evaluation_predictions.npz') as f:centers,oldpred=f['cluster_centers'],f['cluster']
    matches=bool(np.array_equal(pairwise_distances_argmin(oldY,centers),oldpred));assert matches,'Old TRAIN-center assignments not reproduced'
    cluster=pairwise_distances_argmin(Y,centers);np.savez_compressed(out/'evaluation_predictions.npz',classification=pred,cluster=cluster,cluster_centers=centers)
    q=json.loads((a/'query_metrics.json').read_text());assert q['train_ids']==reg['query_ids'][d]['train'] and q['test_ids']==reg['query_ids'][d]['test'];vi=np.asarray(q['test_ids'])
    test,pq=kernels.metrics(X,V[vi],Z,Y[vi]);baselines={m:dict(record=json.loads((Path(prior)/d/f'{m}-seed{s}/evaluation.json').read_text())) for m in ('directA','strong','umap')}
    for item in baselines.values():
        previous=item['record']['label_access']
        assert previous['train_labels_sha256']==train_label_hash and previous['test_labels_sha256']==test_label_hash,'Labels differ from original benchmark'
    train=baselines['directA']['record']['train_metrics'];dump(out/'query_metrics.json',dict(train_ids=q['train_ids'],test_ids=q['test_ids'],train=q['train'],test=pq))
    row=dict(dataset=d,seed=s,status='completed',train_metrics=train,test_metrics=test,classification=dict(accuracy=float(accuracy_score(yv,pred)),balanced_accuracy=float(balanced_accuracy_score(yv,pred)),all_test_rows=len(Y)),kmeans=dict(ari=float(adjusted_rand_score(yv,cluster)),nmi=float(normalized_mutual_info_score(yv,cluster)),n_clusters=len(centers),original_prediction_match=matches),label_access=access,readonly_baselines=baselines)
    if d=='coil':row['pose']=pose(prior,Z,Y)
    dump(out/'evaluation.json',row);return row

def run():
    reg=json.loads((OUT/'preregistration.json').read_text());check(reg);prior=Path(reg['prior_output']);generation=[];scores=[]
    try:
        for d,s in ARMS:
            try:r=generate(prior,d,s)
            except Exception:r=dict(dataset=d,seed=s,status='failed',traceback=traceback.format_exc())
            generation.append(r);dump(OUT/'generation_results.json',generation)
        make_barrier(generation,reg)
        for d,s in ARMS:
            start=time.perf_counter()
            try:r=score(prior,d,s,reg)
            except Exception:r=dict(dataset=d,seed=s,status='failed',traceback=traceback.format_exc())
            r['seconds']=time.perf_counter()-start;scores.append(r);dump(OUT/'scoring_results.json',scores)
    finally:
        integrity=check(reg);dump(OUT/'integrity.json',integrity);dump(OUT/'results.json',dict(protocol=PLAN['protocol'],generation=generation,evaluations=scores,integrity=integrity))
    if any(r['status']!='completed' for r in generation+scores):raise SystemExit(1)

def supervise():
    with (OUT/'run_started.json').open('x') as f:json.dump(dict(utc=now(),budget_seconds=600),f)
    start=time.perf_counter()
    with (OUT/'run.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(CODE/'compact.py'),'execute'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=600);status='completed' if code==0 else 'failed'
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode;status='timeout'
    dump(OUT/'runner_status.json',dict(status=status,returncode=code,seconds=time.perf_counter()-start))
    if status!='completed':
        for filename in ('generation_results.json','scoring_results.json'):
            path=OUT/filename;rows=json.loads(path.read_text()) if path.exists() else [];seen={(r['dataset'],r['seed']) for r in rows}
            rows.extend(dict(dataset=d,seed=s,status='not_completed',reason=status) for d,s in ARMS if (d,s) not in seen);dump(path,rows)
    return 0 if status=='completed' else 1

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='preflight':preflight()
    elif mode=='freeze':freeze(sys.argv[2])
    elif mode=='execute':run()
    elif mode=='supervise':raise SystemExit(supervise())
    else:raise SystemExit('Unknown compact followup phase')
