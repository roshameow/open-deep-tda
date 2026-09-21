"""Real-data, official subject-disjoint HAR benchmark (not a leaderboard).

Example:
 PYTHONPATH=/tmp/open-deep-tda-oracle:python python benchmarks/benchmark_har.py \
   --seeds 0 --upstream-root /tmp/tda-upstream-research/topological-autoencoders

No tuning on official test subjects. Configurations are fixed before execution.
Each reducer is isolated; checkpoints permit resumable evaluation. This tests
numeric features, not raw-signal self-supervised representation quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from open_deep_tda.real_datasets import load_uci_har
from open_deep_tda.population_metrics import population_geometry
from open_deep_tda.topology import distance_matrix, diagram_matching, bottleneck_distance


def write_json(path, content):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(content,indent=2,allow_nan=False));tmp.replace(path)


def label_probe(train,test,y_train,y_test,subjects):
    model=KNeighborsClassifier(n_neighbors=15,n_jobs=1).fit(train,y_train)
    pred=model.predict(test)
    return {'accuracy':float(accuracy_score(y_test,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y_test,pred)),
            'macro_f1':float(f1_score(y_test,pred,average='macro')),
            'per_subject_accuracy':{str(s):float(accuracy_score(y_test[subjects==s],pred[subjects==s])) for s in np.unique(subjects)},
            'scope':'15NN supervised probe on frozen coordinates, trained on official training labels AFTER unsupervised fitting; not a training objective'}


def finite_diagrams(D):
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError('This independent larger-PH audit requires ripser (pip install ripser)') from exc
    raw=ripser(D,distance_matrix=True,maxdim=1)['dgms']
    return [p[np.isfinite(p).all(axis=1) & (p[:,1]>p[:,0])].astype(float) for p in raw]


def ph_metrics(source,target,scale):
    result={}
    for dimension in (0,1):
        P,Q=source[dimension],target[dimension]
        raw=diagram_matching(P,Q,max_matching_size=4096)
        aligned=diagram_matching(P,Q*scale,max_matching_size=4096)
        longest=np.argsort(P[:,1]-P[:,0])[-min(20,len(P)):] if len(P) else np.empty(0,dtype=int)
        result['h'+str(dimension)]={
            'source_bars':len(P),'target_bars':len(Q),
            'raw_normalized_squared_transport':raw['cost']/max(1,len(P)),
            'aligned_normalized_squared_transport':aligned['cost']/max(1,len(P)),
            'aligned_top20_source_unmatched_fraction':len(set(longest)&set(aligned['unmatched_source']))/max(1,len(longest)),
            'largest_source_lifetimes':sorted((P[:,1]-P[:,0]).tolist(),reverse=True)[:5],
            'largest_target_lifetimes_aligned':sorted(((Q[:,1]-Q[:,0])*scale).tolist(),reverse=True)[:5],
            'source_diagram':P.tolist(),'target_diagram':Q.tolist()}
        if dimension==1:
            result['h1']['aligned_bottleneck']=bottleneck_distance(P,Q*scale,max_matching_size=4096)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='outputs/har-real')
    p.add_argument('--cache',default='data/uci_har')
    p.add_argument('--download',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--seeds',default='0,1,2,3,4')
    p.add_argument('--methods',default='pca,deep_tda,no_h1,upstream_topoae,umap')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--upstream-epochs',type=int,default=20)
    p.add_argument('--upstream-root')
    p.add_argument('--queries',type=int,default=256)
    p.add_argument('--ph-sizes',default='128,512,1024')
    p.add_argument('--timeout',type=int,default=600)
    args=p.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')]
    methods=args.methods.split(',')
    sizes=sorted(set(int(v) for v in args.ph_sizes.split(',')))
    if not set(methods)<={'pca','deep_tda','no_h1','upstream_topoae','umap'}:
        p.error('unknown method')
    if not sizes or min(sizes)<2 or max(sizes)>1024 or args.queries<1 or args.steps<0 or args.upstream_epochs<1:
        p.error('PH sizes must be 2..1024, queries positive, valid step/epoch budgets')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data=load_uci_har(args.cache,download=args.download)
    # Original features are already bounded [-1,1]. One train-only scalar avoids
    # unnecessary zero missing-indicator columns and fits official tanh decoders.
    rng=np.random.default_rng(2026)
    a=rng.integers(len(data['X_train']),size=(20000,2))
    d=np.linalg.norm(data['X_train'][a[:,0]].astype(float)-data['X_train'][a[:,1]],axis=1)
    scale=float(np.median(d[d>0]))
    train=np.asarray(data['X_train']/scale,dtype=np.float32)
    test=np.asarray(data['X_test']/scale,dtype=np.float32)
    population=np.concatenate([train,test])
    feature_hash=hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
    protocol={'dataset':data['metadata'],'feature_sha256':feature_hash,'train_only_distance_scale':scale,
        'reference':'all original 561 UCI features / fixed train-only median distance; no PCA, no learned reference',
        'train_size':len(train),'test_size':len(test),'steps':args.steps,'official_topoae_epochs':args.upstream_epochs,
        'queries':min(args.queries,len(test)),'ph_sizes':sizes,'evaluation_seed':2026,
        'selection':'fixed untuned configurations; no official-test-based model or parameter selection',
        'protocol_limitations':'different architectures and optimization schedules are not equal compute; not a paper reproduction or leaderboard'}
    protocol_file=out/'protocol.json'
    if protocol_file.exists():
        if not args.resume:raise ValueError('output exists; use a new directory or --resume')
        if json.loads(protocol_file.read_text())!=protocol:raise ValueError('resume protocol differs; use a new output directory')
    else:
        write_json(protocol_file,protocol)
    feature_file=out/'features.npz'
    if not feature_file.exists():np.savez_compressed(feature_file,train=train,test=test)
    eval_rng=np.random.default_rng(2026)
    query_ids=len(train)+np.sort(eval_rng.choice(len(test),protocol['queries'],replace=False))
    all_ids=eval_rng.permutation(len(population))[:max(sizes)]
    write_json(out/'evaluation_ids.json',{'query_ids':query_ids.tolist(),
        'landmark_ids':{str(size):all_ids[:size].tolist() for size in sizes},
        'landmark_strategy':'nested uniform subsets of full train+test population, chosen before reducer results, no labels'})
    # Reference computations are shared, not repeated/tuned for individual methods.
    with threadpool_limits(limits=1):
        started=time.perf_counter()
        source_D={size:distance_matrix(population[all_ids[:size]]) for size in sizes}
        source_PH={size:finite_diagrams(source_D[size]) for size in sizes}
        write_json(out/'reference_summary.json',{'seconds':time.perf_counter()-started,
            'label_probe':label_probe(train,test,data['y_train'],data['y_test'],data['subjects_test']),
            'ph_bar_counts':{str(size):[len(a) for a in source_PH[size]] for size in sizes}})
        records=[]
        for seed in seeds:
            for method in methods:
                dest=out/f'{method}-seed{seed}';dest.mkdir(exist_ok=True)
                result_path=dest/'result.json'
                if args.resume and result_path.exists():
                    records.append(json.loads(result_path.read_text()));continue
                if not (args.resume and (dest/'embedding.npz').exists() and (dest/'metadata.json').exists()):
                    command=[sys.executable,str(Path(__file__).with_name('har_worker.py').resolve()),
                        '--method',method,'--features',str(feature_file.resolve()),'--output',str(dest.resolve()),
                        '--seed',str(seed),'--steps',str(args.steps),'--upstream-epochs',str(args.upstream_epochs)]
                    if args.upstream_root:command+=['--upstream-root',str(Path(args.upstream_root).resolve())]
                    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
                    try:
                        process=subprocess.run(command,env=env,text=True,capture_output=True,timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        record={'method':method,'seed':seed,'status':'failed','reason':'worker timeout'}
                        write_json(result_path,record);records.append(record);continue
                    (dest/'worker.log').write_text(process.stdout+'\n'+process.stderr)
                    if process.returncode:
                        record={'method':method,'seed':seed,'status':'failed','returncode':process.returncode,'reason':process.stderr[-3000:]}
                        write_json(result_path,record);records.append(record);print(method,'FAILED',flush=True);continue
                record=json.loads((dest/'metadata.json').read_text())
                with np.load(dest/'embedding.npz',allow_pickle=False) as arrays:
                    ztrain,ztest=arrays['train'],arrays['test']
                Z=np.concatenate([ztrain,ztest])
                started=time.perf_counter()
                record['heldout_population_geometry']=population_geometry(population,Z,query_indices=query_ids,
                    max_queries=protocol['queries'],k=15,block_size=16)
                record['heldout_label_probe']=label_probe(ztrain,ztest,data['y_train'],data['y_test'],data['subjects_test'])
                # One scale fitted on the largest common landmark distance matrix,
                # then held fixed for every PH scale/dimension of this method.
                largest=distance_matrix(Z[all_ids])
                source_flat=source_D[max(sizes)].ravel();target_flat=largest.ravel()
                denom=float(target_flat@target_flat)
                alpha=float(source_flat@target_flat/denom) if denom else 1.
                record['diagnostic_scale_alignment']=alpha
                record['ph_audit']={}
                for size in sizes:
                    target_D=largest[:size,:size]
                    target_PH=finite_diagrams(target_D)
                    record['ph_audit'][str(size)]=ph_metrics(source_PH[size],target_PH,alpha)
                record['evaluation_seconds']=time.perf_counter()-started
                record['status']='ok'
                record['ph_scope']='independent Ripser on SAME explicitly reported 128/512/1024 subsets, NOT full-data PH; larger tests do not prove true semantic cycles'
                write_json(result_path,record);records.append(record)
                write_json(out/'results.json',{'protocol':protocol,'records':records})
                print(method,seed,'trust',record['heldout_population_geometry']['trustworthiness'],
                    'accuracy',record['heldout_label_probe']['accuracy'],'seconds',record['fit_seconds'],flush=True)
        write_json(out/'results.json',{'protocol':protocol,'records':records})
        if any(r['status']!='ok' for r in records):return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
