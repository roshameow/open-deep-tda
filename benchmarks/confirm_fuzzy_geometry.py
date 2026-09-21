"""Fixed official-test confirmation AFTER the TRAIN-only fuzzy pilot.

No selection on official TEST: the pilot winner is frozen. Stress/no-topology
is an additional preregistered attribution control, not a new search candidate.
All arms share the exact pilot harness, initialization, features, draws and steps.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

import tune_fuzzy_geometry as pilot
from open_deep_tda.models import ParametricEmbedding
from open_deep_tda.real_datasets import HAR_SHA256


CANDIDATES=[
    {'name':'stress_none','objective':'stress','repulsion':None,'topology':'none'},
    next(copy.deepcopy(c) for c in pilot.PLAN['candidates'] if c['name']=='stress_weak'),
    next(copy.deepcopy(c) for c in pilot.PLAN['candidates'] if c['name']=='fuzzy_r1_none'),
    next(copy.deepcopy(c) for c in pilot.PLAN['candidates'] if c['name']=='fuzzy_r1_weak')]
PLAN={'kind':'fixed-confirmation-not-search','seeds':[0,1,2], 'candidates':CANDIDATES,
      'selected_before_test':'fuzzy_r1_none','harness':pilot.PLAN,
      'attribution_control':'stress_none added before TEST to separate geometry from removing topology',
      'official_test_rule':'evaluate every declared arm; never choose a new winner or retune on test',
      'labels_rule':'labels loaded for final frozen-coordinate probes only, after ALL model fits',
      'scope':'HAR official TRAIN7352/TEST2947; same pilot harness, not DeepTDA.fit latency; test already used in earlier project baselines, not globally untouched data'}


def read_features(archive):
    if pilot.digest(archive)!=HAR_SHA256:raise ValueError('archive digest mismatch')
    result=[]
    with zipfile.ZipFile(archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('UCI HAR Dataset.zip'))) as inner:
            for split,n in [('train',7352),('test',2947)]:
                with inner.open(f'UCI HAR Dataset/{split}/X_{split}.txt') as f:
                    X=np.loadtxt(f,dtype=np.float32)
                if X.shape!=(n,561) or not np.isfinite(X).all():raise ValueError('invalid official feature split')
                result.append(X)
    return result


def read_probe_labels(archive):
    labels=[]
    with zipfile.ZipFile(archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('UCI HAR Dataset.zip'))) as inner:
            for split in ('train','test'):
                with inner.open(f'UCI HAR Dataset/{split}/y_{split}.txt') as f:labels.append(np.loadtxt(f,dtype=np.int64))
    return labels


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregister',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--archive',type=Path,default=Path('data/uci_har/human_activity_recognition_using_smartphones.zip'))
    p.add_argument('--pilot',type=Path,default=Path('outputs/geometry-pilot/results.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/geometry-confirmation'));args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);registered=args.output/'preregistration.json'
    if args.preregister:
        if registered.exists():raise ValueError('preregistration exists')
        previous=json.loads(args.pilot.read_text())
        if previous['selected_candidate']!=PLAN['selected_before_test']:raise ValueError('pilot winner differs')
        pilot.write_json(registered,{'plan':PLAN,'pilot_result_sha256':pilot.digest(args.pilot)})
        print(registered);return
    if not args.run:p.error('choose --preregister or --run')
    if json.loads(registered.read_text())['plan']!=PLAN:raise ValueError('plan changed after registration')
    if (args.output/'results.json').exists():raise ValueError('completed results already exist')
    train,test=read_features(args.archive)
    records=[];encoded=[];started=time.perf_counter()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        U,V,T,edges,weights,delta,margin,b0,b1,prep=pilot.prepare(train,test)
        for seed in PLAN['seeds']:
            torch.manual_seed(seed);initial=ParametricEmbedding(U.shape[1],2);initial.initialize_pca(U)
            for candidate in CANDIDATES:
                model,training=pilot.fit(candidate,seed,initial,T,edges,weights,delta,margin,b0,b1)
                ZU,ZV=pilot.encode(model,U),pilot.encode(model,V)
                evaluation=pilot.evaluate(U,V,ZU,ZV)
                record={'candidate':candidate['name'],'seed':seed,'status':'ok','training':training,
                    'population_geometry':evaluation['population_geometry'],
                    'test_subsets':evaluation['validation_subsets'],'test_subset_mean':evaluation['validation_subset_mean']}
                records.append(record);encoded.append((ZU,ZV))
                print(candidate['name'],seed,record['population_geometry'],flush=True)
        ytrain,ytest=read_probe_labels(args.archive)
        for record,(ZU,ZV) in zip(records,encoded):
            predicted=KNeighborsClassifier(n_neighbors=15,n_jobs=1).fit(ZU,ytrain).predict(ZV)
            record['probe']={'test_accuracy':float(accuracy_score(ytest,predicted)),
                'test_macro_f1':float(f1_score(ytest,predicted,average='macro')),
                'scope':'labels first parsed after all fits; diagnostic, not selection'}
    result={'plan':PLAN,'preregistration_sha256':pilot.digest(registered),'preparation':prep,
            'records':records,'total_seconds':time.perf_counter()-started,
            'selected_before_test':PLAN['selected_before_test'],'no_test_based_reselection':True}
    pilot.write_json(args.output/'results.json',result)
    print(args.output/'results.json')


if __name__=='__main__':main()
