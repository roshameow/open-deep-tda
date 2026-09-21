"""Bounded, preregistered HAR TRAIN-only fuzzy-geometry pilot (no labels).

Run --preregister first, then --run. Only two official TRAIN archive members
are parsed. No models, feature arrays, embeddings or per-person results are
saved. The public README summarizes scope and limitations.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
import zipfile

# Set before numpy/torch imports as well as threadpoolctl for loaded runtimes.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from open_deep_tda import topology
from open_deep_tda.evaluation import evaluate_embedding
from open_deep_tda.fuzzy import build_fuzzy_weights, fuzzy_losses
from open_deep_tda.losses import h0_loss, h1_loss, near_loss, pairwise_distances, separation_loss
from open_deep_tda.models import ParametricEmbedding
from open_deep_tda.neighbors import build_neighbor_graph
from open_deep_tda.population_metrics import population_geometry
from open_deep_tda.preprocessing import NumericPreprocessor
from open_deep_tda.real_datasets import HAR_SHA256, HAR_CITATION
from open_deep_tda.sampling import GeometrySampler, TopologySampler


PLAN = {
    'protocol': 'geometry-pilot-v1', 'seeds': [0], 'steps': 400,
    'validation_subject_ids': [3, 8, 17, 25],
    'data': 'Only official UCI HAR TRAIN; raw 561 columns; no class labels',
    'preprocessing': 'training-fold standardization, no missing indicators; fixed positive random-pair median scalar',
    'architecture': 'shared ParametricEmbedding: PCA linear skip + 561->128->64->2 SiLU residual',
    'pca': 'exact shared training-fold PCA initialization; no validation fit',
    'device': 'cpu', 'threads': 1, 'learning_rate': .001, 'optimizer': 'Adam',
    'gradient_clip': 10., 'n_neighbors': 15, 'batch_edges': 256,
    'negative_sampling': '256 uniform nonedge pairs per update; shared draws across all configurations',
    'fuzzy_scale': 'fixed median positive training kNN union-edge length',
    'stress_delta': 'same median positive training kNN union-edge length',
    'stress_margin': 'positive training kNN union-edge length 75th percentile',
    'lambda_near': 1., 'lambda_sep_stress': .1,
    'warmup_steps': 40, 'topology_interval': 5, 'h0_size': 64, 'h1_size': 32,
    'subset_bank_size': 24, 'landmark_size': 256,
    'topology_sampling': 'fixed shared bank, rotating local/cover/random',
    'weak_topology': {'lambda_h0': .1, 'lambda_h1': .01, 'lambda_critical': .1},
    'topology_ramp': 'min(1,(step-warmup+1)/warmup), multiplied by interval; both H0/H1 every interval',
    'candidates': [
        {'name': 'stress_weak', 'objective': 'stress', 'repulsion': None, 'topology': 'weak'},
        {'name': 'fuzzy_r01_none', 'objective': 'fuzzy', 'repulsion': .1, 'topology': 'none'},
        {'name': 'fuzzy_r01_weak', 'objective': 'fuzzy', 'repulsion': .1, 'topology': 'weak'},
        {'name': 'fuzzy_r1_none', 'objective': 'fuzzy', 'repulsion': 1., 'topology': 'none'},
        {'name': 'fuzzy_r1_weak', 'objective': 'fuzzy', 'repulsion': 1., 'topology': 'weak'},
    ],
    'selection': 'highest validation-query population kNN overlap k=15, then trustworthiness; all tradeoffs disclosed',
    'geometry_evaluation': '256 fixed validation queries against ALL official TRAIN rows (training fold + validation); seed=917',
    'topology_evaluation': 'validation-only paired 64-point PH subsets, evaluation seeds 917/918/919; raw and globally scale-aligned',
    'no_early_stopping': True,
    'restrictions': 'No official TEST feature access, no class labels, no final-test selection, no hidden extra candidates',
    'harness': 'standalone shared production models/losses/graph; not a DeepTDA estimator benchmark',
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_train(archive):
    if digest(archive) != HAR_SHA256:
        raise ValueError('official HAR archive checksum mismatch')
    members = ['UCI HAR Dataset/train/X_train.txt', 'UCI HAR Dataset/train/subject_train.txt']
    with zipfile.ZipFile(archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('UCI HAR Dataset.zip'))) as inner:
            with inner.open(members[0]) as f:
                X = np.loadtxt(f, dtype=np.float32)
            with inner.open(members[1]) as f:
                subjects = np.loadtxt(f, dtype=np.int64)
    if X.shape != (7352, 561) or subjects.shape != (7352,) or not np.isfinite(X).all():
        raise ValueError('invalid official TRAIN data')
    if len(np.unique(subjects)) != 21 or not set(PLAN['validation_subject_ids']) <= set(subjects):
        raise ValueError('unexpected TRAIN subjects')
    validation = np.isin(subjects, PLAN['validation_subject_ids'])
    return X[~validation], X[validation], members


def lengths(X, pairs):
    return torch.linalg.vector_norm(X[pairs[:, 0]] - X[pairs[:, 1]], dim=1)


def source_entry(U, ids, h1=False):
    D = topology.distance_matrix(U[ids])
    return {'ids': ids, 'D': torch.tensor(D, dtype=torch.float32),
            'source': topology.h1_persistence(D) if h1 and hasattr(topology, 'h1_persistence')
                      else topology.persistence(D) if h1 else topology.mst(D)}


def prepare(train, validation):
    started = time.perf_counter()
    prep = NumericPreprocessor(standardize=True, missing_indicators=False).fit(train)
    U, V = prep.transform(train), prep.transform(validation)
    rng = np.random.default_rng(0)
    pairs = rng.integers(len(U), size=(10000, 2))
    d = np.linalg.norm(U[pairs[:, 0]] - U[pairs[:, 1]], axis=1)
    scale = float(np.median(d[d > 0]))
    U, V = U / scale, V / scale
    graph_started = time.perf_counter()
    graph = build_neighbor_graph(U, PLAN['n_neighbors'], working_memory_mb=64)
    graph_seconds = time.perf_counter() - graph_started
    fuzzy_started = time.perf_counter()
    weights, diagnostics = build_fuzzy_weights(U, graph['neighbors'], graph['edges'])
    fuzzy_seconds = time.perf_counter() - fuzzy_started
    T = torch.from_numpy(U)
    edge_lengths = lengths(T, graph['edges']).numpy()
    positive = edge_lengths[edge_lengths > 0]
    delta, margin = float(np.median(positive)), float(np.quantile(positive, .75))
    sampler = TopologySampler(U, graph['neighbors'], PLAN['landmark_size'], seed=121)
    banks_started = time.perf_counter()
    h0bank = [source_entry(U, ids) for _, ids in sampler.bank(PLAN['subset_bank_size'], PLAN['h0_size'])]
    h1bank = [source_entry(U, ids, True) for _, ids in sampler.bank(PLAN['subset_bank_size'], PLAN['h1_size'])]
    banks_seconds = time.perf_counter() - banks_started
    metadata = {'training_rows': len(U), 'validation_rows': len(V), 'features': U.shape[1],
        'reference_pair_scale': scale, 'fuzzy_scale_and_stress_delta': delta,
        'stress_margin': margin, 'graph_edges': len(graph['edges']),
        'graph_seconds': graph_seconds, 'fuzzy_weights_seconds': fuzzy_seconds,
        'source_topology_cache_seconds': banks_seconds,
        'common_preparation_seconds': time.perf_counter() - started,
        'fuzzy_diagnostics': diagnostics}
    return U, V, T, graph['edges'], torch.from_numpy(weights), delta, margin, h0bank, h1bank, metadata


def fit(candidate, seed, initial, T, edges, weights, delta, margin, h0bank, h1bank):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=PLAN['learning_rate'])
    geometry = GeometrySampler(edges, len(T))
    rng = np.random.default_rng(seed)
    edge_keys = edges[:, 0] * len(T) + edges[:, 1]
    history, seen = [], np.zeros(len(T), dtype=bool)
    topology_seconds = 0.
    started = time.perf_counter()
    for step in range(PLAN['steps']):
        pos, neg = geometry.sample(PLAN['batch_edges'], rng)
        topo = (candidate['topology'] == 'weak' and step >= PLAN['warmup_steps']
                and step % PLAN['topology_interval'] == 0)
        bank_id = (step // PLAN['topology_interval']) % len(h0bank)
        a, b = h0bank[bank_id], h1bank[bank_id]
        chunks = [pos.ravel(), neg.ravel()]
        if topo:
            chunks.extend([a['ids'], b['ids']])
        ids = np.unique(np.concatenate(chunks))
        seen[ids] = True
        z = model(T[ids])
        lp, ln = lengths(z, np.searchsorted(ids, pos)), lengths(z, np.searchsorted(ids, neg))
        if candidate['objective'] == 'fuzzy':
            index = np.searchsorted(edge_keys, pos[:, 0] * len(T) + pos[:, 1])
            near, separation = fuzzy_losses(lp, ln, weights[index], delta)
            loss = near + candidate['repulsion'] * separation
        else:
            near = near_loss(lengths(T, pos), lp, delta)
            separation = separation_loss(lengths(T, neg), ln, margin)
            loss = near + PLAN['lambda_sep_stress'] * separation
        parts = {}
        if topo:
            top_start = time.perf_counter()
            h0 = h0_loss(a['D'], pairwise_distances(z[np.searchsorted(ids, a['ids'])]), a['source'])
            h1 = h1_loss(b['D'], pairwise_distances(z[np.searchsorted(ids, b['ids'])]),
                         source_result=b['source'], tau=.1 * delta)
            ramp = min(1., (step - PLAN['warmup_steps'] + 1) / PLAN['warmup_steps'])
            weak = PLAN['weak_topology']
            loss = loss + ramp * PLAN['topology_interval'] * (weak['lambda_h0'] * h0
                + weak['lambda_h1'] * (h1['pd1'] + weak['lambda_critical'] * h1['crit1']))
            topology_seconds += time.perf_counter() - top_start
            parts = {'h0': float(h0.detach()), 'h1_pd': float(h1['pd1'].detach()),
                     'h1_critical': float(h1['crit1'].detach())}
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), PLAN['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        if step % 50 == 0 or step == PLAN['steps'] - 1:
            history.append({'step': step + 1, 'loss': float(loss.detach()),
                            'near': float(near.detach()), 'separation': float(separation.detach()), **parts})
    return model, {'optimizer_seconds': time.perf_counter() - started,
        'topology_forward_seconds': topology_seconds, 'history': history,
        'unique_training_rows_seen': int(seen.sum())}


def encode(model, X):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(X[i:i + 512])).numpy()
                               for i in range(0, len(X), 512)])


def evaluate(U, V, ZU, ZV):
    started = time.perf_counter()
    ids = np.sort(np.random.default_rng(917).choice(len(V), min(256, len(V)), replace=False)) + len(U)
    geom = population_geometry(np.concatenate([U, V]), np.concatenate([ZU, ZV]),
                               query_indices=ids, max_queries=256, k=15)
    # Persist aggregate metrics only, not query/sample IDs, diagrams or curves.
    geometry = {key: geom[key] for key in ('trustworthiness', 'continuity', 'knn_overlap',
                'population_size', 'query_count', 'requested_k', 'scope')}
    rows = []
    for seed in (917, 918, 919):
        report = evaluate_embedding(V, ZV, topology_size=64, seed=seed, k=15)
        row = {'evaluation_seed': seed, 'scale_alignment': report['geometry']['target_scale'],
               'raw_stress': report['geometry']['raw']['normalized_stress'],
               'aligned_stress': report['geometry']['scale_aligned']['normalized_stress']}
        for dimension in (0, 1):
            metrics = report['topology'][f'h{dimension}']
            for key in ('raw_normalized_cost', 'scale_aligned_normalized_cost', 'raw_bottleneck',
                        'scale_aligned_bottleneck', 'source_bars', 'target_bars',
                        'raw_long_source_unmatched_fraction'):
                row[f'h{dimension}_{key}'] = metrics[key]
        rows.append(row)
    mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != 'evaluation_seed'}
    return {'population_geometry': geometry, 'validation_subsets': rows,
            'validation_subset_mean': mean, 'evaluation_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/uci_har/human_activity_recognition_using_smartphones.zip'))
    parser.add_argument('--output', type=Path, default=Path('outputs/geometry-pilot'))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preregister', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preregistration = args.output / 'preregistration.json'
    if args.preregister:
        if preregistration.exists():
            raise FileExistsError('refusing to overwrite preregistration')
        write_json(preregistration, {'created_utc': datetime.now(timezone.utc).isoformat(), 'plan': PLAN})
        print(f'Preregistered {len(PLAN["candidates"])} candidates in {preregistration}')
        return
    registered = json.loads(preregistration.read_text())
    if registered['plan'] != PLAN:
        raise ValueError('plan differs from preregistration; do not silently retune')
    if (args.output / 'results.json').exists():
        raise FileExistsError('refusing to overwrite completed results')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run_start = time.perf_counter()
        train, validation, members = load_train(args.archive)
        U, V, T, edges, weights, delta, margin, h0bank, h1bank, metadata = prepare(train, validation)
        results = {'preregistration_sha256': digest(preregistration), 'plan': PLAN,
            'input': {'archive_sha256': HAR_SHA256, 'parsed_members': members, 'citation': HAR_CITATION},
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'torch': torch.__version__, 'platform': platform.platform()},
            'preparation': metadata, 'runs': []}
        for seed in PLAN['seeds']:
            torch.manual_seed(seed)
            initial = ParametricEmbedding(U.shape[1], 2)
            pca_start = time.perf_counter()
            initial.initialize_pca(U)
            results['preparation']['pca_initialization_seconds'] = time.perf_counter() - pca_start
            for candidate in PLAN['candidates']:
                model, training = fit(candidate, seed, initial, T, edges, weights, delta, margin, h0bank, h1bank)
                infer_start = time.perf_counter()
                ZU, ZV = encode(model, U), encode(model, V)
                inference = time.perf_counter() - infer_start
                measured = evaluate(U, V, ZU, ZV)
                row = {'candidate': candidate['name'], 'seed': seed, **training,
                       'inference_seconds_train_and_validation': inference, **measured}
                results['runs'].append(row)
                write_json(args.output / (candidate['name'] + '.json'), row)
                print(json.dumps({'candidate': candidate['name'], 'seconds': training['optimizer_seconds'],
                                  **measured['population_geometry']}), flush=True)
        winner = max(results['runs'], key=lambda r: (r['population_geometry']['knn_overlap'],
                                                   r['population_geometry']['trustworthiness']))
        results['selected_candidate'] = winner['candidate']
        results['total_elapsed_seconds'] = time.perf_counter() - run_start
        write_json(args.output / 'results.json', results)


if __name__ == '__main__':
    main()
