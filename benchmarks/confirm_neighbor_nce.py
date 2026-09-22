"""Fixed full-data confirmation of a TRAIN-only conditional-neighbor search.

Register after optimize_neighbor_nce.py completes, then run without changing the
plan or code. All methods use every official train/test row. Fashion settings
are fixed by its internal TRAIN validation; HAR is a direct dimension-rule
transfer. Existing TEST sets have been used before, not fresh external evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
from threadpoolctl import threadpool_limits

from open_deep_tda import DeepTDA, TDAConfig
from confirm_geometry_optimization import geometry, probe, summarize, write_json, implementation_digest, load_cache
from tune_fuzzy_geometry import digest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2]


def selection(path):
    result = json.loads(path.read_text())
    plan = result['plan']
    if plan['protocol'] != 'fashion-train-neighbor-nce-v1' or plan['seeds'] != [0]:
        raise ValueError('expected completed registered TRAIN-only NCE pilot')
    expected = [c['name'] for c in plan['candidates']]
    rows = {r['candidate']:r for r in result['runs']}
    if len(rows) != len(result['runs']) or set(rows) != set(expected):
        raise ValueError('incomplete or duplicated candidate results')
    for candidate in plan['candidates']:
        row = rows[candidate['name']]
        config = TDAConfig(**row['config']).validate().to_dict()
        declared = dict(plan['common'], **{k:v for k,v in candidate.items() if k!='name'})
        if row.get('status', 'ok') != 'ok' or row['seed'] != 0 or any(config[k] != v for k,v in declared.items()):
            raise ValueError('candidate disagrees with registration')
        for key in ('knn_overlap', 'trustworthiness'):
            value = row['population_geometry'][key]
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid selection metric')
    ordered = [rows[n] for n in expected]
    winner = max(ordered, key=lambda r:(r['population_geometry']['knn_overlap'], r['population_geometry']['trustworthiness']))
    if result['selected_candidate'] != winner['candidate']:
        raise ValueError('selection not reproducible from registered rule')
    return winner, rows['graph_2400']


def make_plan(args):
    winner, baseline = selection(args.pilot)
    arms = [dict(name='previous_graph', config=dict(baseline['config'], output_calibration='train_pairs')),
            dict(name='selected', config=dict(winner['config'], output_calibration='train_pairs'))]
    if args.dataset == 'har':
        for arm in arms:
            arm['config'] = dict(arm['config'], standardize=True,
                                 residual_input_scale=float(np.sqrt(561)), neighbor_backend='exact')
    data = ({'features_sha256':digest(args.features), 'annotations_sha256':digest(args.annotations)}
            if args.dataset == 'fashion' else {'cache_sha256':digest(args.cache)})
    return dict(protocol='neighbor-nce-confirmation-v1', dataset=args.dataset, seeds=SEEDS,
                selected_candidate=winner['candidate'], pilot_result_sha256=digest(args.pilot),
                implementation_sha256=implementation_digest(), script_sha256=digest(Path(__file__)),
                arms=arms, input_hashes=data,
                preprocessing='Fashion: existing full official TRAIN-only PCA64. HAR: train-only feature standardization. Same preprocessing for both arms.',
                selection='fixed maximum TRAIN validation k15 overlap then trustworthiness; no official TEST reselection',
                evaluation='256 fixed TEST queries against full TRAIN+TEST population, k15; three fixed64point TEST PH subclouds raw/aligned H0/H1 including unmatched bars; frozen 15NN probe on all TEST; KMeans trained TRAIN then predicted TEST ARI/NMI with fixed known class count',
                labels='only post-fit evaluation; no labels passed to reducer or model selection',
                budgets='all declared steps; no early stopping; sampled negative counts and runtime differ',
                calibration='Both arms use analytic TRAIN-pair least-squares scalar after fitting; no TEST scale fitting. Does not improve neighbor ranking or clustering. Report uncalibrated PH beside actual calibrated output; no structural-topology gain inferred from units alone.',
                local_artifacts='compact model and all train/test embeddings in ignored outputs only',
                limitations='Previously used benchmarks, not fresh external validation. Full population topology not claimed; report all topology regressions. HAR is direct transfer, not a new search.')


def load_data(args):
    if args.dataset == 'har':
        f = load_cache(args.cache)
        train, test, yt, yv = [f[k] for k in ('X_train','X_test','y_train','y_test')]
        if train.shape != (7352,561) or test.shape != (2947,561):
            raise ValueError('expected full official HAR split')
        return train, test, yt, yv, 6
    with np.load(args.features, allow_pickle=False) as f:
        train,test = f['train'],f['test']
    with np.load(args.annotations, allow_pickle=False) as f:
        yt,yv = f['train_labels'],f['test_labels']
    if train.shape != (60000,64) or test.shape != (10000,64):
        raise ValueError('expected complete official Fashion PCA64')
    if yt.shape != (60000,) or yv.shape != (10000,) or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError('invalid complete Fashion arrays')
    return train,test,yt,yv,10


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['har','fashion'], required=True)
    p.add_argument('--pilot', type=Path, default=Path('outputs/neighbor-nce-pilot/results.json'))
    p.add_argument('--features', type=Path, default=Path('outputs/v02-fashion_mnist/features.npz'))
    p.add_argument('--annotations', type=Path, default=Path('outputs/v02-fashion_mnist/annotations.npz'))
    p.add_argument('--cache', type=Path, default=Path('data/uci_har/features.npz'))
    p.add_argument('--output', type=Path, required=True)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--preregister', action='store_true'); a.add_argument('--run', action='store_true')
    args=p.parse_args()
    if (ROOT/'outputs').resolve() not in args.output.resolve().parents:
        p.error('local artifacts require ignored repository outputs subdirectory')
    plan=make_plan(args)
    args.output.mkdir(parents=True,exist_ok=True)
    reg=args.output/'preregistration.json'
    if args.preregister:
        write_json(reg,dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan),exclusive=True)
        return
    if json.loads(reg.read_text())['plan']!=plan:
        raise ValueError('plan/code/data changed since registration')
    write_json(args.output/'run_started.json',dict(registration_sha256=digest(reg)),exclusive=True)
    result=dict(plan=plan,preregistration_sha256=digest(reg),records=[],status='running')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        train,test,yt,yv,classes=load_data(args)
        for seed in SEEDS:
            for arm in plan['arms']:
                start=time.perf_counter()
                cfg=dict(arm['config'],seed=seed)
                row=dict(method=arm['name'],seed=seed,config=cfg,status='running')
                try:
                    model=DeepTDA(cfg).fit(train)
                    row['fit_seconds']=model.fit_seconds_
                    Ztrain,Ztest=model.embedding_,model.transform(test)
                    # Fashion reference is fixed common PCA64 units across seeds,
                    # matching the earlier image benchmark's scalar convention.
                    if args.dataset=='fashion':
                        Ztrain,Ztest=Ztrain*model.reference_scale_,Ztest*model.reference_scale_
                        U,V=train,test
                    else:
                        U,V=model.reference_,model.reference_transform(test)
                    row.update(geometry(U,V,Ztrain,Ztest))
                    row['output_calibration'] = model.calibration_diagnostics_
                    uncalibrated = geometry(U,V,Ztrain/model.output_scale_,Ztest/model.output_scale_)
                    row['uncalibrated_test_subset_mean'] = uncalibrated['test_subset_mean']
                    row['probe']=probe(Ztrain,Ztest,yt,yv)
                    km=KMeans(n_clusters=classes,n_init=10,random_state=seed).fit(Ztrain)
                    pred=km.predict(Ztest)
                    row['kmeans']=dict(test_ari=float(adjusted_rand_score(yv,pred)),
                                       test_nmi=float(normalized_mutual_info_score(yv,pred)))
                    row['coverage']=model.training_coverage_
                    out=args.output/'local'/arm['name']/('seed'+str(seed))
                    out.mkdir(parents=True,exist_ok=False)
                    model.save(out/'model.pt',include_training_data=False)
                    np.savez_compressed(out/'embeddings.npz',Z_train=Ztrain,Z_test=Ztest)
                    row['status']='ok'
                except Exception as error:
                    import traceback
                    traceback.print_exc()  # Local redirected log only; no raw trace in public JSON.
                    row.update(status='failed',error_type=type(error).__name__)
                row['total_seconds']=time.perf_counter()-start
                result['records'].append(row)
                write_json(args.output/'progress.json',result)
                print(json.dumps(dict(method=arm['name'],seed=seed,status=row['status'],probe=row.get('probe'))),flush=True)
    failed=any(r['status']!='ok' for r in result['records'])
    result['status']='completed_with_failures' if failed else 'completed'
    result['summary']=summarize(result['records'],[a['name'] for a in plan['arms']])
    for method, summary in result['summary'].items():
        rows = [r for r in result['records'] if r['method']==method and r['status']=='ok']
        for group in ('uncalibrated_test_subset_mean', 'output_calibration'):
            keys = sorted({k for r in rows for k,v in r[group].items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
            for key in keys:
                values=[r[group][key] for r in rows if key in r[group]]
                summary['metrics'][group+'.'+key]=dict(count=len(values),mean=float(np.mean(values)),std=float(np.std(values)))
    write_json(args.output/'results.json',result,exclusive=True)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
