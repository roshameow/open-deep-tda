"""Bounded production-estimator Fashion TRAIN-only conditional-neighbor search.

Only train-images-idx3-ubyte.gz is parsed. No labels or official TEST members.
Split all official TRAIN rows into fixed 50k fit / 10k validation; PCA64 is fit
on the 50k fold only. Register first. Report every candidate, not best seeds.
PYTHONPATH=python:benchmarks python benchmarks/optimize_neighbor_nce.py --preregister
PYTHONPATH=python:benchmarks python benchmarks/optimize_neighbor_nce.py --run
Optional ANN dependencies must be installed in the chosen interpreter.
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
from sklearn.decomposition import PCA
import torch
from threadpoolctl import threadpool_limits

from open_deep_tda import DeepTDA
from open_deep_tda.image_datasets import _read_idx_gzip
from tune_fuzzy_geometry import evaluate, digest, write_json

COMMON = dict(standardize=False, missing_indicators=False, seed=0, batch_size=256,
              n_neighbors=15, neighbor_backend='pynndescent', neighbor_timeout_seconds=300,
              warmup_steps=60, h0_size=128, h1_size=64, topology_interval=5,
              h0_refresh_every=1, h1_refresh_every=1, landmark_size=512, subset_bank_size=6,
              evaluation_size=32, log_interval=200, validation_interval=100000,
              num_threads=1, residual_input_scale=8., fuzzy_repulsion=1.,
              lambda_h0=.1, lambda_h1=.01)
CANDIDATES = [
    dict(name='graph_2400', geometry_objective='fuzzy_graph', steps=2400),
    dict(name='graph_7200', geometry_objective='fuzzy_graph', steps=7200),
    dict(name='nce16_2400', geometry_objective='neighbor_nce', steps=2400),
    dict(name='nce16_7200', geometry_objective='neighbor_nce', steps=7200),
    dict(name='nce_hard4_2400', geometry_objective='neighbor_nce', steps=2400, contrastive_hard_negatives=4),
    dict(name='nce_t05_2400', geometry_objective='neighbor_nce', steps=2400, contrastive_temperature=.5),
]
PLAN = dict(protocol='fashion-train-neighbor-nce-v1', seeds=[0], common=COMMON,
            candidates=CANDIDATES, split_seed=731, n_fit=50000, n_validation=10000,
            data='Only full official Fashion TRAIN images, no labels; randomized split seed731',
            preprocessing='pixels /255; randomized PCA64 fit on 50000 fit rows only, seed2026 iterated_power4; then production train-only scalar',
            selection='highest validation population k15 overlap, then trustworthiness, then registered order',
            evaluation='256 fixed validation queries against all60000 TRAIN rows; three fixed64point validation PH subsets, raw and aligned, full-population metric not tiny-cloud overlap',
            restrictions='No class labels, TEST images, early stopping, hidden candidates or seed selection. 7200-step arms explicitly not compute matched.',
            attribution='paired previous graph control and budget control; NCE repulsion is conditional, not separately weighted by fuzzy_repulsion; hard mining changes objective',
            future_confirmation='Freeze winner before official TEST. Transfer to HAR by dimension-only residual scale sqrt561 and train standardization; do not retune on TEST.')


def load_training(path):
    if digest(path) != '3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84':
        raise ValueError('unexpected official Fashion TRAIN image archive checksum')
    images = _read_idx_gzip(path, expected_count=60000, magic=2051)
    ids = np.random.default_rng(PLAN['split_seed']).permutation(60000)
    train = images[ids[:50000]].reshape(50000, -1).astype('float32') / 255.
    validation = images[ids[50000:]].reshape(10000, -1).astype('float32') / 255.
    pca = PCA(n_components=64, svd_solver='randomized', iterated_power=4, random_state=2026)
    U = np.ascontiguousarray(pca.fit_transform(train), dtype='float32')
    V = np.ascontiguousarray(pca.transform(validation), dtype='float32')
    return U, V, dict(training_explained_variance=float(pca.explained_variance_ratio_.sum()),
                      feature_sha256=hashlib.sha256(U.tobytes()+V.tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('data/fashion_mnist/train-images-idx3-ubyte.gz'))
    parser.add_argument('--output', type=Path, default=Path('outputs/neighbor-nce-pilot'))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preregister', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reg = args.output / 'preregistration.json'
    if args.preregister:
        if reg.exists():
            raise FileExistsError(reg)
        write_json(reg, dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=PLAN))
        return
    if json.loads(reg.read_text())['plan'] != PLAN:
        raise ValueError('registration changed')
    marker = args.output / 'run_started.json'
    with marker.open('x') as f:
        json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat()), f)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        train, validation, preparation = load_training(args.images)
        result = dict(plan=PLAN, preregistration_sha256=digest(reg), preparation=preparation, runs=[])
        for candidate in CANDIDATES:
            name = candidate['name']
            cfg = dict(COMMON, **{k:v for k,v in candidate.items() if k!='name'})
            start = time.perf_counter()
            model = DeepTDA(cfg).fit(train)
            V = model.reference_transform(validation)
            measured = evaluate(model.reference_, V, model.embedding_, model.transform(validation))
            row = dict(candidate=name, seed=0, config=model.config.to_dict(),
                       fit_seconds=model.fit_seconds_, elapsed_seconds=time.perf_counter()-start,
                       coverage=model.training_coverage_, **measured)
            result['runs'].append(row)
            write_json(args.output/(name+'.json'), row)
            write_json(args.output/'progress.json', result)
            print(json.dumps(dict(candidate=name, fit_seconds=model.fit_seconds_, **measured['population_geometry'])), flush=True)
        best = max(result['runs'], key=lambda r:(r['population_geometry']['knn_overlap'], r['population_geometry']['trustworthiness']))
        result['selected_candidate'] = best['candidate']
        write_json(args.output/'results.json', result)


if __name__ == '__main__':
    main()
