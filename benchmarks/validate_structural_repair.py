"""Fixed small TRAIN-only structural-repair diagnostics, NOT quality benchmarks.

COIL: first 48 TRAIN rows (the first object's training views in the published
object-ordered preparation, not a best-object search).
Fashion: 48 label-blind random TRAIN rows, seed402. Both use existing TRAIN-only
PCA64 features; no labels/TEST arrays are parsed. This does not fit a model or
validate out-of-sample reduction. All successes AND failures are retained.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'

import numpy as np
from threadpoolctl import threadpool_limits

from open_deep_tda import topology
from open_deep_tda.structural_h0 import compare_h0
from open_deep_tda.structural_h1 import check_h1_witnesses
from open_deep_tda.structural_witnesses import select_h1_witnesses
from open_deep_tda.structural_layout import repair_layout
from open_deep_tda.structural_dual_layout import solve_structural_layout

ROOT=Path(__file__).resolve().parents[1]
PLAN=dict(protocol='small-structural-repair-diagnostic-v1',
    data={'coil20':'first48 official TRAIN PCA64 rows; first object in fixed object ordering, not a best-object search','fashion':'48 TRAIN PCA64 rows without replacement, RNG402'},
    labels='No labels or TEST arrays opened. Existing PCA fitted on full TRAIN; not a held-out experiment.',
    source='Normalize full selected-domain distances by their positive median; all48 rows used in every certificate.',
    witnesses='Longest positive source H1 bar; interval[birth+.2*lifetime,death-.2*lifetime]; lex source image basis cap2; no contract if absent/unresolvable.',
    modes={'h1_only':None,'joint_h0_h1':.05},
    search=dict(max_rounds=16,max_iterations=300,restarts=2,seed=0,jitter=.01,separation_margin=.01),
    budgets=dict(max_vertices=64,max_simplices=200000,max_reduction_operations=500000000,max_reduction_entries=2000000),
    initialization='Exact two-component PCA of selected input, in normalized source units; no source-label hints.',
    interpretation='All results are small-instance structural feasibility/repair diagnostics, not full-population topology, semantic clustering, newDR quality, or convergence/infeasibility proofs.')


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def hashes():
    paths=sorted((ROOT/'python/open_deep_tda').glob('*.py'))+[Path(__file__)]
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_train(path):
    with zipfile.ZipFile(path) as f:
        payload=f.read('train.npy')
    return np.load(io.BytesIO(payload),allow_pickle=False),hashlib.sha256(payload).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/structural-repair-real'))
    p.add_argument('--solver',choices=['sequential','dual'],default='dual')
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--preregister',action='store_true');a.add_argument('--run',action='store_true')
    args=p.parse_args()
    if (ROOT/'outputs').resolve() not in args.output.resolve().parents:
        p.error('local arrays must stay under ignored outputs')
    args.output.mkdir(parents=True,exist_ok=True)
    reg=args.output/'preregistration.json'
    plan=dict(PLAN,solver=args.solver)
    registration=dict(plan=plan,source_hashes=hashes())
    if args.preregister:
        with reg.open('x') as f:json.dump(registration,f,indent=2)
        return
    if json.loads(reg.read_text())!=registration:raise ValueError('registered code/plan changed')
    with (args.output/'run_started.json').open('x') as f:json.dump({'started':True},f)
    report=dict(plan=plan,source_hashes=registration['source_hashes'],runs=[])
    solve=repair_layout if args.solver=='sequential' else solve_structural_layout
    with threadpool_limits(limits=1):
        for dataset,path in [('coil20',ROOT/'outputs/v02-coil20/features.npz'),
                             ('fashion',ROOT/'outputs/v02-fashion_mnist/features.npz')]:
            train,data_hash=load_train(path)
            expected=(960,64) if dataset=='coil20' else (60000,64)
            if train.shape!=expected or train.dtype.kind not in 'fiu' or not np.isfinite(train).all():
                raise ValueError('unexpected complete TRAIN feature array')
            ids=np.arange(48) if dataset=='coil20' else np.sort(np.random.default_rng(402).choice(len(train),48,replace=False))
            X=train[ids].astype(float);D=topology.distance_matrix(X)
            scale=float(np.median(D[D>0])); X=X/scale;D=D/scale
            _,_,vt=np.linalg.svd(X-X.mean(0),full_matrices=False)
            for row in vt[:2]:
                if row[np.argmax(abs(row))]<0:row*=-1
            initial=(X-X.mean(0))@vt[:2].T
            bars=topology.diagram(topology.h1_persistence(D),dimension=1)
            if not len(bars):
                report['runs'].append(dict(dataset=dataset,status='no_source_contract',train_hash=data_hash));continue
            b,d=max(bars.tolist(),key=lambda row:(row[1]-row[0],-row[0],-row[1]))
            lo,hi=b+.2*(d-b),d-.2*(d-b)
            selected=select_h1_witnesses(D,lo,hi,max_cycles=2,**PLAN['budgets'])
            if not selected['cycles']:
                report['runs'].append(dict(dataset=dataset,status='no_source_contract',train_hash=data_hash));continue
            before=check_h1_witnesses(D,topology.distance_matrix(initial),selected['cycles'],lo,hi,**PLAN['budgets'])
            for name,tolerance in PLAN['modes'].items():
                row=dict(dataset=dataset,mode=name,train_hash=data_hash,n_vertices=48,
                         birth_radius=lo,survival_radius=hi,image_rank=selected['image_rank'],
                         selected_count=selected['selected_count'],before_h1=asdict(before),before_accepted=before.accepted,
                         before_h0=compare_h0(D,topology.distance_matrix(initial),tolerance=.05))
                started=time.perf_counter()
                try:
                    result=solve(D,initial,selected['cycles'],lo,hi,h0_tolerance=tolerance,
                                         **PLAN['search'],**PLAN['budgets'])
                    row.update({k:v for k,v in result.items() if k not in ('embedding','last_candidate')})
                    np.savez_compressed(args.output/(dataset+'-'+name+'.npz'),reference=X,initial=initial,
                                        candidate=result['last_candidate'],source_row_ids=ids)
                except Exception as e:
                    import traceback
                    traceback.print_exc()  # ignored local log only
                    row.update(status='error',error_type=type(e).__name__)
                row['seconds']=time.perf_counter()-started
                report['runs'].append(row)
                write(args.output/'progress.json',report)
                print(json.dumps({k:row[k] for k in ['dataset','mode','status','seconds','before_accepted']}),flush=True)
    write(args.output/'results.json',report)


if __name__=='__main__':main()
