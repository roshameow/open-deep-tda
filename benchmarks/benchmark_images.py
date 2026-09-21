"""Real image experiments with frozen common features and explicit non-toy scope.

COIL-20 uses all 1440 views, held-out viewpoints, and full per-object cycles.
Fashion-MNIST uses all 60000/10000 official samples, no fitting subsample.
This evaluates reducers on shared train-only PCA64 features, NOT image SSL.
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
from sklearn import config_context
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from image_protocol import prepare_images
from benchmark_har import finite_diagrams, ph_metrics, write_json
from open_deep_tda import TDAConfig
from open_deep_tda.population_metrics import population_geometry
from open_deep_tda.topology import distance_matrix
from open_deep_tda.evaluation import evaluate_embedding
from open_deep_tda.cycle_metrics import evaluate_rotation_orbits, angle_neighbor_metrics


def label_probe(train,test,ytrain,ytest):
    # All test images. Config bounds pairwise chunks for the high-dimensional
    # reference classifier; 2D reducers can use sklearn's tree implementation.
    with config_context(working_memory=64):
        model=KNeighborsClassifier(n_neighbors=15,n_jobs=1).fit(train,ytrain)
        pred=model.predict(test)
    return {'accuracy':float(accuracy_score(ytest,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(ytest,pred)),
        'macro_f1':float(f1_score(ytest,pred,average='macro')),
        'n_train':len(train),'n_test':len(test),
        'scope':'exact 15NN probe fit AFTER unsupervised reduction using training labels only; ALL official/declared test images scored'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=['coil20','fashion_mnist'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--seeds',default='0,1,2')
    parser.add_argument('--methods',default='pca,deep_tda,no_h1,upstream_topoae,umap')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--upstream-epochs',type=int)
    parser.add_argument('--upstream-root')
    parser.add_argument('--queries',type=int,default=256)
    parser.add_argument('--ph-size',type=int,default=1024)
    parser.add_argument('--neighbor-backend',choices=['exact','pynndescent'])
    parser.add_argument('--timeout',type=int,default=900)
    args=parser.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')];methods=args.methods.split(',')
    allowed={'pca','deep_tda','no_h1','upstream_topoae','umap'}
    if not methods or not set(methods)<=allowed:parser.error('unknown/empty methods; no substitute TopoAE++/RTD baseline')
    if not seeds or any(s<0 or s>=2**32 for s in seeds):parser.error('invalid seeds')
    if not 2<=args.ph_size<=1024 or args.queries<1:parser.error('PH size 2..1024 and queries>=1 required')
    steps=args.steps if args.steps is not None else (600 if args.dataset=='coil20' else 1200)
    epochs=args.upstream_epochs if args.upstream_epochs is not None else (30 if args.dataset=='coil20' else 5)
    if steps<0 or epochs<1:parser.error('invalid budgets')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    prepared=prepare_images(args.dataset,out,download=args.download)
    with np.load(out/'features.npz',allow_pickle=False) as a:train,test=a['train'],a['test']
    with np.load(out/'annotations.npz',allow_pickle=False) as a:annotations={key:a[key] for key in a.files}
    actual_hash=hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
    if actual_hash!=prepared['feature_sha256']:raise ValueError('prepared feature cache digest mismatch')
    cfg=TDAConfig(standardize=False,missing_indicators=False,steps=steps,warmup_steps=min(60,steps//10),
        batch_size=256,h0_size=128,h1_size=64,subset_bank_size=6,landmark_size=512,
        h0_refresh_every=1,h1_refresh_every=1,evaluation_size=32,
        validation_interval=max(1,steps+1),log_interval=50,
        neighbor_backend=args.neighbor_backend or ('exact' if args.dataset=='coil20' else 'pynndescent'),
        neighbor_working_memory_mb=64,neighbor_audit_queries=64,neighbor_min_recall=.9,
        neighbor_timeout_seconds=min(600,args.timeout)).validate()
    config_path=out/'reducer_config.json'
    protocol={'prepared':prepared,'config':cfg.to_dict(),'official_topoae_epochs':epochs,
        'queries':min(args.queries,len(test)),'ph_size':min(args.ph_size,len(train)+len(test)),
        'evaluation_seed':2026,'target_dimension':2,
        'model_selection':'fixed untuned protocol; no best-of-runs or official-test tuning',
        'caveat':'different architectures/epochs/steps are not compute matched; not original-paper hyperparameter reproduction',
        'unavailable_official_methods':{'TopoAE++':'not a reported baseline here: optional upstream Boost/CGAL/runtime requirements were unavailable',
            'RTD-AE':'not a reported baseline: actual declared backend requires Linux/CUDA; GUDHI adapter smoke is separate'}}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        if not args.resume:raise ValueError('existing experiment: use --resume or a new directory')
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('resume protocol differs; use a new directory')
    else:write_json(protocol_path,protocol)
    write_json(config_path,cfg.to_dict())
    X=np.concatenate([train,test]);rng=np.random.default_rng(2026)
    query_ids=len(train)+np.sort(rng.choice(len(test),protocol['queries'],replace=False))
    ph_ids=rng.permutation(len(X))[:protocol['ph_size']]
    write_json(out/'evaluation_ids.json',{'query_ids':query_ids.tolist(),'ph_ids':ph_ids.tolist(),
        'scope':'queries are held-out images ranked against ALL train+test images; PH is the declared common subcloud only'})
    source_D=distance_matrix(X[ph_ids]);source_PH=finite_diagrams(source_D)
    with threadpool_limits(limits=1):
        if not (args.resume and (out/'reference_evaluation.json').exists()):
            base={'label_probe':label_probe(train,test,annotations['train_labels'],annotations['test_labels']),
                  'ph_bar_counts':[len(dgm) for dgm in source_PH]}
            with np.load(out/'preprocessing_audit.npz',allow_pickle=False) as a:
                base['preprocessing_subcloud_audit']=evaluate_embedding(a['raw_pixels'],a['pca_features'],
                    topology_size=64,seed=2026)
            write_json(out/'reference_evaluation.json',base)
        if args.dataset=='coil20' and not (args.resume and (out/'raw_pixel_angle_reference.json').exists()):
            from open_deep_tda.image_datasets import load_coil20
            original=load_coil20(download=False,image_size=32)
            pixel=original['images'].reshape(1440,-1).astype(np.float32)/255.
            original_queries=original['angles_degrees']%15==0
            rows=[]
            for obj in np.unique(original['object_ids']):
                ids=np.flatnonzero(original['object_ids']==obj)
                rows.append({'object_id':int(obj),'metrics':angle_neighbor_metrics(pixel[ids],
                    original['angles_degrees'][ids],original_queries[ids])})
            write_json(out/'raw_pixel_angle_reference.json',{'objects':rows,
                'mean_adjacent_recall':float(np.mean([r['metrics']['adjacent_view_recall'] for r in rows])),
                'scope':'native data downsampled to32x32 but BEFORE PCA; held-out views, within-object neighbors'})
        records=[]
        for seed in seeds:
            for method in methods:
                dest=out/f'{method}-seed{seed}';dest.mkdir(exist_ok=True)
                result_path=dest/'result.json'
                if args.resume and result_path.exists():
                    saved=json.loads(result_path.read_text())
                    if saved['status']=='ok' or not args.retry_failed:records.append(saved);continue
                if not (args.resume and (dest/'embedding.npz').exists() and (dest/'metadata.json').exists()):
                    command=[sys.executable,str(Path(__file__).with_name('image_reducer_worker.py').resolve()),
                        '--method',method,'--features',str((out/'features.npz').resolve()),'--output',str(dest.resolve()),
                        '--seed',str(seed),'--steps',str(steps),'--upstream-epochs',str(epochs),
                        '--config',str(config_path.resolve())]
                    if args.upstream_root:command+=['--upstream-root',str(Path(args.upstream_root).resolve())]
                    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
                    try:process=subprocess.run(command,env=env,capture_output=True,text=True,timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        error={'method':method,'seed':seed,'status':'failed','reason':'isolated worker timeout'}
                        write_json(result_path,error);records.append(error);continue
                    (dest/'worker.log').write_text(process.stdout+'\n'+process.stderr)
                    if process.returncode:
                        error={'method':method,'seed':seed,'status':'failed','returncode':process.returncode,'reason':process.stderr[-3000:]}
                        write_json(result_path,error);records.append(error);print(method,seed,'FAILED',flush=True);continue
                record=json.loads((dest/'metadata.json').read_text())
                with np.load(dest/'embedding.npz',allow_pickle=False) as a:ztrain,ztest=a['train'],a['test']
                Z=np.concatenate([ztrain,ztest]);started=time.perf_counter()
                record['heldout_population_geometry']=population_geometry(X,Z,query_indices=query_ids,max_queries=protocol['queries'],k=15,block_size=16)
                record['heldout_label_probe']=label_probe(ztrain,ztest,annotations['train_labels'],annotations['test_labels'])
                target_D=distance_matrix(Z[ph_ids]);denominator=float(np.sum(target_D*target_D))
                alpha=float(np.sum(source_D*target_D)/denominator) if denominator else 1.
                record['diagnostic_scale_alignment']=alpha
                record['ph_audit']=ph_metrics(source_PH,finite_diagrams(target_D),alpha)
                if args.dataset=='coil20':
                    query_mask=np.arange(len(X))>=len(train)
                    record['rotation_orbits']=evaluate_rotation_orbits(X,Z,annotations['object_ids'],
                        annotations['angles_degrees'],query_mask=query_mask,global_scale=alpha)
                record['evaluation_seconds']=time.perf_counter()-started;record['status']='ok'
                record['ph_scope']='independent Ripser on shared landmark IDs; COIL additionally full72-view PH for each of20objects, not whole-population PH'
                write_json(result_path,record);records.append(record)
                write_json(out/'results.json',{'protocol':protocol,'records':records})
                text=[method,str(seed),'trust',str(round(record['heldout_population_geometry']['trustworthiness'],5)),
                    'accuracy',str(round(record['heldout_label_probe']['accuracy'],5))]
                if 'rotation_orbits' in record:text+=['angle_recall',str(round(record['rotation_orbits']['summary']['embedding_adjacent_recall'],5))]
                print(' '.join(text),flush=True)
        write_json(out/'results.json',{'protocol':protocol,'records':records})
    return 0 if all(r['status']=='ok' for r in records) else 2


if __name__=='__main__':raise SystemExit(main())
