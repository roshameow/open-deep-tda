#!/usr/bin/env python3
if __package__:
    from .common import *
else:
    from common import *
import time,traceback
from scipy import sparse
from scipy.spatial.distance import cdist
if __package__:
    from .kernels import FrozenMap,OPTIONS
else:
    from kernels import FrozenMap,OPTIONS

def index_for(X,seed):
    assert 'torch' not in sys.modules
    from pynndescent import NNDescent
    obj=NNDescent(np.asarray(X,np.float32),n_neighbors=31,metric='euclidean',random_state=seed,n_jobs=1,low_memory=True,compressed=False,parallel_batch_queries=False);obj.prepare();return obj

def make_graph(X,ids,d):
    n=len(X);w=numeric.weights(d);A=sparse.csr_matrix((w.ravel(),(np.repeat(np.arange(n),15),ids.ravel())),shape=(n,n));G=(A+A.T-A.multiply(A.T)).tocsr();G.eliminate_zeros();return G

def preflight():
    # Isolated NumPy/Numba/UMAP process, synthetic only, no real input access.
    graph_worker=load_graph_worker();initialize=graph_worker.initialize
    rng=np.random.default_rng(19);X=rng.normal(size=(160,8));Q=rng.normal(size=(13,8));index=index_for(X,0);ids,d=numeric.knn(X,X,15,np.arange(len(X)));G=make_graph(X,ids,d)
    Z,meta=initialize(X,G,0);assert meta['components']==1 and np.isfinite(Z).all()
    from numba import njit
    coo=G.tocoo();zcheck,counts,negative,events=njit(graph_worker.sgd_impl)(Z.copy(),coo.row,coo.col,coo.data,2,5,0,4000000000)
    assert np.isfinite(zcheck).all() and counts.sum()>0 and events>0
    from scipy.sparse.linalg import eigsh
    degree=np.asarray(G.sum(1)).ravel();D=sparse.diags(1/np.sqrt(degree));v,e=eigsh(sparse.eye(len(X))-D@G@D,k=3,which='SM',v0=np.random.default_rng(0).normal(size=len(X)),tol=1e-6);oldZ=e[:,np.argsort(v)[1:3]]
    for c in range(2):
        if oldZ[np.argmax(abs(oldZ[:,c])),c]<0:oldZ[:,c]*=-1
    oldZ=oldZ*(10/np.max(abs(oldZ)))+np.random.default_rng(0).normal(0,1e-4,oldZ.shape)
    assert np.max(abs(oldZ-Z))<1e-10
    H=sparse.block_diag([np.ones((4,4))-np.eye(4),np.ones((3,3))-np.eye(3),np.zeros((1,1)),np.ones((2,2))-np.eye(2)]).tocsr()
    z,details=initialize(np.zeros((10,8)),H,0);assert np.isfinite(z).all() and details['components']==4 and 'IDcircle' in details['center_rule']
    assert np.array_equal(z,initialize(np.zeros((10,8)),H,0)[0])
    save_pickle(index,OUT/'synthetic_index.pkl');index=load_pickle(OUT/'synthetic_index.pkl')
    sys.modules['tensorflow']=None;import umap
    ii=np.column_stack((np.arange(len(X)),ids));dd=np.column_stack((np.zeros(len(X)),d));model=umap.UMAP(**umap_options(0),precomputed_knn=(ii,dd,index)).fit(X);Y=model.transform(Q);assert np.array_equal(model._knn_indices,ii) and np.isfinite(Y).all()
    save_pickle(model,OUT/'synthetic_umap_model.pkl');restored=load_pickle(OUT/'synthetic_umap_model.pkl');assert np.array_equal(restored.transform(Q),Y)
    mapper=FrozenMap(Z);p=np.ones(15)/15;bary=p@Z[ids[0]];y,r=mapper.solve('A',ids[0],p,bary);assert r['finite'];eps=1e-6;f,g=mapper.cauchy(y,ids[0],p);fd=np.array([(mapper.cauchy(y+eps*np.eye(2)[j],ids[0],p)[0]-mapper.cauchy(y-eps*np.eye(2)[j],ids[0],p)[0])/(2*eps) for j in range(2)]);assert max(abs(g-fd))<1e-7
    dump(OUT/'synthetic_preflight.json',dict(status='passed',connected_max_abs_difference=float(np.max(abs(oldZ-Z))),disconnected=details,author_supplied_neighbors_retained=True,author_model_pickle_roundtrip=True,author_test_output_shape=list(Y.shape),A_finite=True,synthetic_SGD_compiles=True,A_gradient_difference=float(max(abs(g-fd))),graph_worker_source_sha256=sha(ROOT/'python/open_deep_tda/_graph_worker.py'),torch_absent='torch' not in sys.modules,completed_utc=now()))

def preflight_strong():
    # Separate Torch process; verifies raw-HAR/common-standardization convention.
    sys.path.insert(0,str(ROOT/'python'));from open_deep_tda import DeepTDA
    from open_deep_tda.preprocessing import NumericPreprocessor
    import open_deep_tda.neighbors as nb
    rng=np.random.default_rng(33);X=rng.normal(size=(80,12)).astype(np.float32);standard=NumericPreprocessor(True,False).fit_transform(X) if hasattr(NumericPreprocessor,'fit_transform') else NumericPreprocessor(True,False).fit(X).transform(X)
    mean=X.astype(float).mean(0);sd=X.astype(float).std(0);sd[sd<1e-12]=1;expected=((X.astype(float)-mean)/sd).astype(np.float32);assert np.array_equal(expected,standard)
    ids,d=numeric.knn(expected,expected,15,np.arange(len(X)));n=len(X);keys=np.unique(np.minimum(np.arange(n)[:,None],ids)*n+np.maximum(np.arange(n)[:,None],ids));edges=np.c_[keys//n,keys%n]
    original=nb.build_neighbor_graph;calls=[]
    def provider(U,k,**kw):calls.append(1);return dict(neighbors=ids,edges=edges,diagnostics={'backend':'synthetic_shared'})
    nb.build_neighbor_graph=provider
    try:
        model=DeepTDA(standardize=True,missing_indicators=False,steps=2,h0_size=8,h1_size=8,landmark_size=32,subset_bank_size=1,evaluation_size=8,warmup_steps=0).fit(X)
    finally:nb.build_neighbor_graph=original
    assert len(calls)==1 and np.array_equal(model.preprocessor_.transform(X),expected) and 'numba' not in sys.modules
    from open_deep_tda import TDAConfig
    resolved={d:TDAConfig(**cfg).validate().to_dict() for d,cfg in json.loads((CODE/'strong_configs.json').read_text()).items()}
    dump(OUT/'strong_preflight.json',dict(status='passed',resolved_configs=resolved,train_only_standardization_bitwise=True,shared_graph_provider=True,Torch_separate=True,completed_utc=now()))

def freeze():
    assert not (OUT/'preregistration.json').exists()
    assert all(json.loads((OUT/f).read_text())['status']=='passed' for f in ('synthetic_preflight.json','strong_preflight.json'))
    configs=json.loads((OUT/'strong_preflight.json').read_text())['resolved_configs'];inputs={};query={};audits={}
    for di,(d,spec) in enumerate(DATASETS.items()):
        X,h=member(ROOT/spec['path'],spec['train']);assert X.shape==(spec['n'],spec['d']) and np.isfinite(X).all()
        inputs[d]=dict(train_member_sha256=h,train_array_sha256=ah(X),files={p:descriptor(ROOT/p) for p in set([spec['path'],spec['annotations']])})
        if d!='har':
            prep_path=(ROOT/spec['path']).with_name('reference_pca.npz')
            with zipfile.ZipFile(prep_path) as z:assert set(z.namelist())=={'components.npy','mean.npy','distance_scale.npy'}
            inputs[d]['TRAIN_fitted_PCA_artifact']=dict(path=str(prep_path),sha256=sha(prep_path),scope='TRAIN-fitted components/mean/distance_scale only, no TEST or labels')
        rng=np.random.default_rng(940);query[d]=dict(train=np.sort(rng.choice(spec['n'],512,replace=False)).tolist(),test=(list(range(spec['m'])) if spec['m']<=1024 else np.sort(rng.choice(spec['m'],1024,replace=False)).tolist()))
        rng=np.random.default_rng(941);audits[d]=dict(train=np.sort(rng.choice(spec['n'],min(256,spec['n']),replace=False)).tolist(),test=np.sort(rng.choice(spec['m'],min(256,spec['m']),replace=False)).tolist())
    # No test or label payloads loaded above; archive central directories only.
    r=dict(created_utc=now(),plan=PLAN,sources=sources(),configs=configs,inputs=inputs,query_ids=query,audit_ids=audits,A_options=OPTIONS,A_sha256=sha(CODE/'kernels.py'),python=sys.version,executable=sys.executable,environment=environment_snapshot(),git_head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),preflight_hashes={n:sha(OUT/n) for n in ('synthetic_preflight.json','strong_preflight.json')})
    with (OUT/'preregistration.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print('FROZEN BEFORE TEST/LABEL PAYLOAD ACCESS',flush=True)

def prepare(d,reg):
    spec=DATASETS[d];root=base(d);root.mkdir(parents=True,exist_ok=True)
    for p,desc in reg['inputs'][d]['files'].items():assert descriptor(ROOT/p)==desc
    if d!='har':
        pre=reg['inputs'][d]['TRAIN_fitted_PCA_artifact'];assert sha(pre['path'])==pre['sha256']
    X,h=member(ROOT/spec['path'],spec['train']);assert h==reg['inputs'][d]['train_member_sha256']
    V,vh=member(ROOT/spec['path'],spec['test']);assert X.shape==(spec['n'],spec['d']) and V.shape==(spec['m'],spec['d']) and np.isfinite(V).all()
    if d=='har':
        np.save(root/'raw_train.npy',X);np.save(root/'raw_test.npy',V);mean=X.astype(float).mean(0);sd=X.astype(float).std(0);sd[sd<1e-12]=1;X=((X.astype(float)-mean)/sd).astype(np.float32);V=((V.astype(float)-mean)/sd).astype(np.float32);np.savez(root/'preprocessing.npz',mean=mean,scale=sd)
        prep='raw original HAR official subject split; TRAIN mean/std only; common float32 standardization; strong uses raw input and identical native preprocessing'
    else:prep='cached unwhitened PCA64 and global distance scalar fitted on FULL official TRAIN, as image_protocol.py; no50k pilot preprocessing reuse'
    np.save(root/'train.npy',X);np.save(root/'test.npy',V)
    return dict(preprocessing=prep,preprocessing_artifact=(dict(path=str(root/'preprocessing.npz'),sha256=sha(root/'preprocessing.npz')) if d=='har' else reg['inputs'][d]['TRAIN_fitted_PCA_artifact']),train_member_sha256=h,test_member_sha256=vh,train_array_sha256=ah(X),test_array_sha256=ah(V),train_shape=list(X.shape),test_shape=list(V.shape),first_TEST_access_after_registration=now(),labels_accessed=False)

def graph(d,seed,reg):
    root=graphdir(d,seed);root.mkdir(parents=True,exist_ok=True);X,V=load_data(d);index=index_for(X,seed);raw,_=index.neighbor_graph;ids,dist=rerank(X,X,raw,15,np.arange(len(X)))
    save_pickle(index,root/'index.pkl');index=load_pickle(root/'index.pkl');query,_=index.query(V,k=31,epsilon=.12);vi,vd=rerank(X,V,query,15)
    ai=reg['audit_ids'][d];a=np.asarray(ai['train']);b=np.asarray(ai['test']);raw16,_=index.query(V[b],k=16,epsilon=.12)
    audits=dict(fit=audit(X,X[a],ids[a],a),A_query=audit(X,V[b],vi[b]),author_query=audit(X,V[b],raw16[:,:15]));dump(root/'audits.json',audits)
    assert all(v['tie_aware_recall']>=.9 for v in audits.values()),'frozen ANN recall gate failed; no repair'
    np.savez_compressed(root/'neighbors.npz',fit_ids=ids,fit_distances=dist,query_ids=vi,query_distances=vd)
    return dict(audits={k:{q:v for q,v in a.items() if q not in ('exact_query_ids','per_query_strict')} for k,a in audits.items()},index_sha256=sha(root/'index.pkl'),fit_ids_sha256=ah(ids),torch_absent='torch' not in sys.modules)

def fit(d,method,seed,reg):
    root=armdir(d,method,seed);root.mkdir(parents=True,exist_ok=True);X,V=load_data(d);t=time.perf_counter();details={}
    if method=='pca':
        from sklearn.decomposition import PCA
        model=PCA(n_components=2,svd_solver='full').fit(X);Z=model.transform(X);fit_s=time.perf_counter()-t;t=time.perf_counter();Y=model.transform(V);details=dict(fit_seconds=fit_s,transform_seconds=time.perf_counter()-t);np.savez(root/'model.npz',components=model.components_,mean=model.mean_)
    else:
        with np.load(graphdir(d,seed)/'neighbors.npz') as a:ids,dist,qids,qd=a['fit_ids'],a['fit_distances'],a['query_ids'],a['query_distances']
        if method=='strong':
            assert 'numba' not in sys.modules
            sys.path.insert(0,str(ROOT/'python'));from open_deep_tda import DeepTDA
            import open_deep_tda.neighbors as nb
            n=len(X);keys=np.unique(np.minimum(np.arange(n)[:,None],ids)*n+np.maximum(np.arange(n)[:,None],ids));edges=np.c_[keys//n,keys%n];config=dict(reg['configs'][d],seed=seed)
            pairs=np.random.default_rng(seed).integers(n,size=(min(max(1024,n*2),20000),2));length=np.linalg.norm(X[pairs[:,0]].astype(float)-X[pairs[:,1]],axis=1);scale=np.median(length[length>0]);expected=np.asarray(X/scale,np.float32);original=nb.build_neighbor_graph;calls=[]
            def provider(U,k,**kw):
                assert k==15 and np.array_equal(U,expected),'strong common-reference normalization mismatch';calls.append(1);return dict(neighbors=ids,edges=edges,diagnostics=dict(backend='frozen_shared_graph',audit=json.loads((graphdir(d,seed)/'result.json').read_text())['audits']['fit']))
            nb.build_neighbor_graph=provider
            rawX=np.load(base(d)/'raw_train.npy') if d=='har' else X;rawV=np.load(base(d)/'raw_test.npy') if d=='har' else V
            t=time.perf_counter()
            try:model=DeepTDA(config).fit(rawX)
            finally:nb.build_neighbor_graph=original
            fit_s=time.perf_counter()-t;Z=model.embedding_;t=time.perf_counter();Y=model.transform(rawV);trans=time.perf_counter()-t;assert len(calls)==1
            model.save(root/'model.pt',include_training_data=False);details=dict(fit_seconds=fit_s,transform_seconds=trans,config=model.config.to_dict(),coverage=model.training_coverage_,graph_provider_calls=1,numba_absent='numba' not in sys.modules)
        elif method=='umap':
            assert 'torch' not in sys.modules
            index=load_pickle(graphdir(d,seed)/'index.pkl');sys.modules['tensorflow']=None;import umap
            ii=np.c_[np.arange(len(X)),ids];dd=np.c_[np.zeros(len(X)),dist];t=time.perf_counter();model=umap.UMAP(**umap_options(seed),precomputed_knn=(ii,dd,index)).fit(X);fit_s=time.perf_counter()-t;assert np.array_equal(model._knn_indices,ii)
            Z=model.embedding_;np.save(root/'fit.npy',Z);t=time.perf_counter();Y=model.transform(V);trans=time.perf_counter()-t;save_pickle(model,root/'model.pkl');details=dict(fit_seconds=fit_s,transform_seconds=trans,options=umap_options(seed),version=umap.__version__,supplied_neighbors_retained=True,torch_absent='torch' not in sys.modules)
        elif method=='directA':
            assert 'torch' not in sys.modules
            graph_worker=load_graph_worker();initialize=graph_worker.initialize
            from numba import njit
            t=time.perf_counter();G=make_graph(X,ids,dist);Z,init=initialize(X,G,seed);np.save(root/'initial.npy',Z);coo=G.tocoo();s=time.perf_counter();Z,counts,neg,events=njit(graph_worker.sgd_impl)(Z,coo.row,coo.col,coo.data,300,5,seed,4000000000);opt=time.perf_counter()-s;fit_s=time.perf_counter()-t;np.save(root/'fit.npy',Z)
            edge_lengths=np.linalg.norm(Z[:,None]-Z[ids],axis=2);unit=float(np.median(edge_lengths[edge_lengths>0]));center=Z.mean(0);assert unit>0 and np.isfinite(unit);N=(Z-center)/unit
            p=numeric.weights(qd);exact=qd==0;mask=exact.any(1);p[mask]=exact[mask];p/=p.sum(1)[:,None];bary=np.einsum('ij,ijk->ik',p,N[qids]);model=FrozenMap(N)
            np.savez_compressed(root/'model.npz',normalized_fit=N,center=center,unit=unit);np.savez_compressed(root/'edge_exposure.npz',counts=counts,weights=coo.data)
            output=np.lib.format.open_memmap(root/'A_normalized.npy',mode='w+',dtype=np.float64,shape=(len(V),2));output[:]=np.nan;output.flush();t=time.perf_counter();rows=[]
            with (root/'query_diagnostics.jsonl').open('x') as log:
                for i in range(len(V)):
                    start=time.perf_counter()
                    try:
                        y,r=model.solve('A',qids[i],p[i],bary[i]);output[i]=y;output.flush();r.update(query_index=i,seconds=time.perf_counter()-start,record_status='returned')
                    except Exception:r=dict(query_index=i,finite=False,success=False,record_status='exception',traceback=traceback.format_exc())
                    for key,v in list(r.items()):
                        if isinstance(v,float) and not np.isfinite(v):r[key]=None
                    rows.append(r);log.write(json.dumps(r,allow_nan=False)+'\n');log.flush()
                    if (i+1)%500==0:dump(root/'mapping_progress.json',dict(queries=i+1,seconds=time.perf_counter()-t))
            Y=np.asarray(output)*unit+center;trans=time.perf_counter()-t
            details=dict(fit_seconds=fit_s,optimization_seconds=opt,transform_seconds=trans,initializer=init,unit=unit,center=center.tolist(),A_options=OPTIONS,query_count=len(V),optimizer_successes=sum(r.get('success',False) for r in rows),optimizer_nonconvergences=sum(not r.get('success',False) for r in rows),nonfinite_or_exceptions=sum(not r.get('finite',False) for r in rows),max_gradient_linf=max((r.get('gradient_linf') or 0 for r in rows)),positive_updates=int(counts.sum()),negative_updates=int(neg),directed_edges=len(counts),unvisited_edges=int((counts==0).sum()),torch_absent='torch' not in sys.modules)
    np.save(root/'fit.npy',Z);np.save(root/'test.npy',Y);dump(root/'fit_transform_details.json',details)
    assert Z.shape==(len(X),2) and Y.shape==(len(V),2) and np.isfinite(Z).all() and np.isfinite(Y).all(),'nonfinite outputs retained, arm not eligible for metrics'
    return dict(**details,fit_sha256=sha(root/'fit.npy'),test_sha256=sha(root/'test.npy'),labels_accessed=False)

def get_labels(d,reg):
    barrier=require_label_barrier()
    spec=DATASETS[d];y,hy=member(ROOT/spec['annotations'],spec['train_labels']);v,hv=member(ROOT/spec['annotations'],spec['test_labels']);meta=dict(train_labels_sha256=hy,test_labels_sha256=hv,loaded_utc=now(),embeddings_manifest_sha256=sha(OUT/'embeddings_frozen.json'))
    if d=='har':
        a,ha=member(ROOT/spec['path'],'subjects_train.npy');b,hb=member(ROOT/spec['path'],'subjects_test.npy');assert not set(a)&set(b) and len(set(a))==21 and len(set(b))==9;meta['subject_split_verified']=True;meta['subject_hashes']=[ha,hb]
    assert y.shape==(spec['n'],) and v.shape==(spec['m'],);return y,v,meta

def pose(d,Z,Y):
    spec=DATASETS[d];angles,ha=member(ROOT/spec['annotations'],'angles_degrees.npy');objects,ho=member(ROOT/spec['annotations'],'object_ids.npy');X,V=load_data(d)
    def measure(allZ):
        recall=[];diffs=[]
        for o in np.unique(objects):
            ids=np.flatnonzero(objects==o);theta=angles[ids]%360;order=np.argsort(theta,kind='stable');adj={int(order[i]):{int(order[(i-1)%len(ids)]),int(order[(i+1)%len(ids)])} for i in range(len(ids))};D=cdist(allZ[ids],allZ[ids]);np.fill_diagonal(D,np.inf);nn=np.argsort(D,axis=1,kind='stable')[:,:2]
            for local in np.flatnonzero(ids>=len(Z)):
                recall.append(len(set(nn[local])&adj[int(local)])/2);dt=abs(theta[nn[local]]-theta[local]);diffs.extend(np.minimum(dt,360-dt).tolist())
        assert len(recall)==480;return dict(adjacent_view_recall=float(np.mean(recall)),mean_neighbor_angle_degrees=float(np.mean(diffs)),queries=480,scope='object-conditioned k2 among72 full views; TESTqueries only, no PH')
    return dict(reference=measure(np.vstack((X,V))),embedding=measure(np.vstack((Z,Y))),annotation_hashes=[ha,ho])

def evaluate(d,method,seed,reg):
    root=armdir(d,method,seed);X,V=load_data(d);Z=np.load(root/'fit.npy');Y=np.load(root/'test.npy');fi=np.asarray(reg['query_ids'][d]['train']);vi=np.asarray(reg['query_ids'][d]['test']);f,fq=numeric.metrics(X,X[fi],Z,Z[fi],fi);v,vq=numeric.metrics(X,V[vi],Z,Y[vi]);dump(root/'query_metrics.json',dict(train_ids=fi.tolist(),test_ids=vi.tolist(),train=fq,test=vq))
    yt,yv,labelmeta=get_labels(d,reg)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,adjusted_rand_score,normalized_mutual_info_score
    from sklearn.cluster import KMeans
    predicted=KNeighborsClassifier(n_neighbors=15,weights='uniform',p=2,n_jobs=1).fit(Z,yt).predict(Y);classes=len(np.unique(yt));km=KMeans(n_clusters=classes,n_init=10,random_state=seed).fit(Z);pred=km.predict(Y);np.savez_compressed(root/'evaluation_predictions.npz',classification=predicted,cluster=pred,cluster_centers=km.cluster_centers_)
    result=dict(train_metrics=f,test_metrics=v,classification=dict(accuracy=float(accuracy_score(yv,predicted)),balanced_accuracy=float(balanced_accuracy_score(yv,predicted)),all_test_rows=len(Y)),kmeans=dict(ari=float(adjusted_rand_score(yv,pred)),nmi=float(normalized_mutual_info_score(yv,pred)),n_clusters=classes,fit='TRAIN only'),label_access=labelmeta)
    if d=='coil':result['pose']=pose(d,Z,Y)
    # Independent exact rank oracle on16 fixed train/test queries and saved means.
    errors=[]
    for domain,source,target,pop,measured in [('train',X,Z,fi,fq),('test',V,Y,vi,vq)]:
        for off,i in enumerate(pop[:8]):
            ds=cdist(source[i:i+1],X)[0];dt=cdist(target[i:i+1],Z)[0];M=len(X)
            if domain=='train':ds[i]=np.inf;dt[i]=np.inf;M-=1
            a=np.lexsort((np.arange(len(X)),ds));b=np.lexsort((np.arange(len(X)),dt));ra=np.empty(len(X),int);rb=ra.copy();ra[a]=np.arange(1,len(X)+1);rb[b]=np.arange(1,len(X)+1)
            for k in (5,15,50):
                vals=[len(set(a[:k])&set(b[:k]))/k,1-2*sum(max(ra[j]-k,0) for j in b[:k])/(k*(2*M-3*k+1)),1-2*sum(max(rb[j]-k,0) for j in a[:k])/(k*(2*M-3*k+1))];errors.extend(abs(np.asarray(vals)-measured[str(k)][off]).tolist())
    assert max(errors)<1e-12
    # Independent classification check on first32 TEST rows, not fit/model change.
    for i in range(min(32,len(Y))):
        dist=cdist(Y[i:i+1],Z)[0];ids=np.argsort(dist,kind='stable')[:15];values,cnt=np.unique(yt[ids],return_counts=True);expected=values[np.argmax(cnt)]
        # If boundary ties differ across sklearn tree and stable IDs, disclose rather than silently force equality.
        if expected!=predicted[i]:result.setdefault('classification_tie_audit_mismatches',[]).append(i)
    result['independent_rank_audit']=dict(queries=16,max_abs_difference=max(errors));result['sources_unchanged']=reg['sources']==sources()
    if method=='directA':
        with np.load(root/'model.npz') as a:N,unit,center=a['normalized_fit'],float(a['unit']),a['center']
        with np.load(graphdir(d,seed)/'neighbors.npz') as a:ids,dist=a['query_ids'],a['query_distances']
        p=numeric.weights(dist);eq=dist==0;mask=eq.any(1);p[mask]=eq[mask];p/=p.sum(1)[:,None];rows=[json.loads(l) for l in (root/'query_diagnostics.jsonl').read_text().splitlines()];out=(Y-center)/unit;records=[];err=[]
        for i,r in enumerate(rows):
            w=1/(1+np.sum((N-out[i])**2,axis=1));weight=p[i]*w[ids[i]];ww=w*w;grad=2*(weight.sum()*out[i]-weight@N[ids[i]]-(ww.sum()*out[i]-ww@N)/w.sum());res=float(max(abs(grad)));err.append(abs(res-r['gradient_linf']));records.append(res)
        assert max(err)<1e-8;np.save(root/'independent_full_gradient_linf.npy',records);result['A_audit']=dict(all_queries=len(rows),normalizer_population=len(X),max_residual_difference=max(err),gradient_median=float(np.median(records)),gradient_max=float(max(records)),gradient_gtol_count=int(np.sum(np.asarray(records)<=1e-8)),optimizer_success=sum(r['success'] for r in rows),nonconverged_ids=[r['query_index'] for r in rows if not r['success']],objective_increases=sum(r['objective_delta']>1e-10 for r in rows))
    return result

def main():
    mode=sys.argv[1]
    if mode=='preflight':preflight();return
    if mode=='preflight_strong':preflight_strong();return
    if mode=='freeze':freeze();return
    reg=registration();d=sys.argv[2];seed=int(sys.argv[3]) if len(sys.argv)>3 else 0;method=sys.argv[4] if len(sys.argv)>4 else None
    path=(base(d)/'preparation.json' if mode=='prepare' else graphdir(d,seed)/'result.json' if mode=='graph' else armdir(d,method,seed)/('result.json' if mode=='fit' else 'evaluation.json'));start=time.perf_counter()
    try:
        val={'prepare':lambda:prepare(d,reg),'graph':lambda:graph(d,seed,reg),'fit':lambda:fit(d,method,seed,reg),'evaluate':lambda:evaluate(d,method,seed,reg)}[mode]();val.update(status='completed')
    except BaseException:val=dict(status='failed',traceback=traceback.format_exc())
    val.update(dataset=d,seed=seed,method=method,stage=mode,seconds=time.perf_counter()-start,completed_utc=now());dump(path,val);print(json.dumps(val),flush=True)
    if val['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
