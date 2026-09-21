"""TRAIN-only production-estimator ablation; no test rows or class labels parsed.

Register the bounded candidate set before running. Uses the same subject-disjoint
HAR training-fold split as the earlier fuzzy pilot. This is iterative development
on an existing validation set, not an independent final generalization estimate.
Run from the repository with PYTHONPATH=python:benchmarks.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from open_deep_tda import DeepTDA
from tune_fuzzy_geometry import load_train, evaluate, digest, write_json

COMMON = dict(standardize=True, missing_indicators=False, seed=0,
              batch_size=256, n_neighbors=15, warmup_steps=60,
              h0_size=128, h1_size=64, topology_interval=5,
              h0_refresh_every=1, h1_refresh_every=1, landmark_size=512,
              subset_bank_size=6, evaluation_size=32, log_interval=100,
              validation_interval=100000, num_threads=1)
CANDIDATES = [
    dict(name='pca', steps=0),
    dict(name='stress_600', steps=600),
    dict(name='stress_2400', steps=2400),
    dict(name='stress_none_600', steps=600, lambda_h0=0., lambda_h1=0.),
    dict(name='fuzzy_ce_2400', steps=2400, geometry_objective='fuzzy',
         fuzzy_repulsion=1., lambda_h0=0., lambda_h1=0.),
    dict(name='graph_r1_600', steps=600, geometry_objective='fuzzy_graph',
         fuzzy_repulsion=1., lambda_h0=0., lambda_h1=0.),
    dict(name='graph_r1_2400', steps=2400, geometry_objective='fuzzy_graph',
         fuzzy_repulsion=1., lambda_h0=0., lambda_h1=0.),
    dict(name='graph_r5_2400', steps=2400, geometry_objective='fuzzy_graph',
         fuzzy_repulsion=5., lambda_h0=0., lambda_h1=0.),
    dict(name='graph_weak_2400', steps=2400, geometry_objective='fuzzy_graph',
         fuzzy_repulsion=1., lambda_h0=.1, lambda_h1=.01),
]
PLAN = dict(protocol='production-geometry-ablation-v1', common=COMMON,
            candidates=CANDIDATES, seeds=[0],
            data='Only official HAR TRAIN X and subject IDs; validation subjects 3,8,17,25',
            selection='maximum validation-query k=15 population overlap; tie by trustworthiness',
            restrictions='No class labels, official TEST rows, best-seed selection or early stopping',
            evaluation='256 fixed validation queries against all official TRAIN rows; 3 fixed 64-point validation PH subclouds; raw AND scale-aligned costs',
            caveat='Reused development split. Longer runs are explicitly not compute matched.')


def gradient_summary(history):
    active = [r for r in history if r['topology_ramp'] == 1.]
    keys = ('near', 'separation', 'h0', 'h1')
    return {k: float(np.median([r['component_gradient_norms'][k] for r in active]))
            if active else 0. for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/uci_har/human_activity_recognition_using_smartphones.zip'))
    parser.add_argument('--output', type=Path, default=Path('outputs/geometry-ablation'))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preregister', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / 'preregistration.json'
    if args.preregister:
        if plan_path.exists():
            raise FileExistsError(plan_path)
        write_json(plan_path, dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=PLAN))
        return
    if json.loads(plan_path.read_text())['plan'] != PLAN:
        raise ValueError('registered plan differs')
    if (args.output / 'results.json').exists():
        raise FileExistsError('completed results already exist')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        train, validation, members = load_train(args.archive)
        results = dict(plan=PLAN, preregistration_sha256=digest(plan_path), parsed_members=members,
                       n_train=len(train), n_validation=len(validation), runs=[])
        for candidate in CANDIDATES:
            name = candidate['name']
            cfg = dict(COMMON, **{k: v for k, v in candidate.items() if k != 'name'})
            start = time.perf_counter()
            model = DeepTDA(cfg).fit(train)
            U, V = model.reference_, model.reference_transform(validation)
            measured = evaluate(U, V, model.embedding_, model.transform(validation))
            row = dict(candidate=name, config=model.config.to_dict(), seed=0,
                       fit_seconds=model.fit_seconds_, total_seconds=time.perf_counter() - start,
                       weighted_gradient_medians=gradient_summary(model.history_),
                       coverage=model.training_coverage_, geometry_objective=model.geometry_diagnostics_,
                       **measured)
            results['runs'].append(row)
            write_json(args.output / (name + '.json'), row)
            write_json(args.output / 'progress.json', results)
            print(json.dumps(dict(candidate=name, **measured['population_geometry'])), flush=True)
        winner = max(results['runs'], key=lambda r: (r['population_geometry']['knn_overlap'],
                                                    r['population_geometry']['trustworthiness']))
        results['selected_candidate'] = winner['candidate']
        write_json(args.output / 'results.json', results)


if __name__ == '__main__':
    main()
