"""Fixed HAR-to-Fashion transfer confirmation; this is NOT a Fashion search.

From the repository (optional dependencies already installed in the isolated target):
  PYTHONPATH=python:benchmarks:/tmp/open-deep-tda-oracle python benchmarks/confirm_geometry_fashion.py --preregister
  PYTHONPATH=python:benchmarks:/tmp/open-deep-tda-oracle python benchmarks/confirm_geometry_fashion.py --run

Only this new script is changed. All artifacts stay in ignored outputs. JSON is
aggregate-only; private checkpoints/coordinates and raw subprocess logs are local.
Each method/seed has one subprocess attempt, plus production's audited ANN child.
No import-time execution, downloads, fallback, retries, or TEST-based selection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS',
              'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
from threadpoolctl import threadpool_limits

from open_deep_tda import DeepTDA, topology
from open_deep_tda.config import TDAConfig
import confirm_geometry_optimization as har
import image_protocol

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
SELECTED = 'graph_weak_conditioned'
METHODS = ('original_stress_1200', 'transferred_graph_weak_2400',
           'baseline_pca2', 'baseline_pca64')
WORKER_TIMEOUT = 1800
AUDIT_KEYS = ('performed', 'query_count', 'population_size', 'strict_id_recall',
              'tie_aware_distance_recall', 'acceptance_metric', 'min_recall', 'accepted')


def digest(path):
    return har.pilot.digest(path)


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def write_json(path, value, exclusive=False):
    har.write_json(path, value, exclusive=exclusive)


def sources():
    # HAR's digest includes its harness, pilot and all production Python files.
    # Also pin the actual native binary used for every PH diagnostic.
    return {'script_sha256': digest(Path(__file__)),
            'har_implementation_sha256': har.implementation_digest(),
            'image_preparation_source_sha256': digest(Path(image_protocol.__file__)),
            'native_binary_sha256': digest(Path(topology._native().__file__))}


def configs_from_har(path):
    saved = json.loads(path.read_text())['plan']
    if saved['protocol'] != 'production-geometry-confirmation-v1':
        raise ValueError('unexpected HAR protocol')
    selection = saved['selection']
    if selection['selected_candidate'] != SELECTED or selection['class_free'] is not True:
        raise ValueError('HAR selection is not the required locked class-free winner')
    chosen = [a for a in saved['arms'] if 'selected' in a['roles']]
    if len(chosen) != 1 or chosen[0]['candidate'] != SELECTED:
        raise ValueError('HAR selected arm disagrees with locked selection')
    # Check the saved ranking as well, without opening any HAR TEST results.
    winner = max(selection['scores'], key=lambda r: (r['knn_overlap'], r['trustworthiness']))
    if winner['candidate'] != SELECTED:
        raise ValueError('HAR saved ranking disagrees with locked winner')
    if saved['implementation_sha256'] != har.implementation_digest():
        raise ValueError('hash-pinned HAR implementation changed')
    original = chosen[0]['config']
    if TDAConfig(**original).validate().to_dict() != original:
        raise ValueError('HAR configuration is incomplete or changed')
    required = dict(geometry_objective='fuzzy_graph', steps=2400, lambda_h0=.1,
                    lambda_h1=.01, fuzzy_repulsion=1., mode='geometry',
                    optimizer_mode='parametric', n_components=2, device='cpu',
                    num_threads=1, seed=0, n_neighbors=15, missing_indicators=False)
    if any(original[k] != v for k, v in required.items()):
        raise ValueError('unexpected locked graph objective, budget, weights or common settings')
    if not math.isclose(original['residual_input_scale'], math.sqrt(561), rel_tol=1e-14):
        raise ValueError('HAR scale is not the declared square-root dimension rule')
    common = dict(original, standardize=False, missing_indicators=False,
                  n_neighbors=15, neighbor_backend='pynndescent', h0_size=128,
                  h1_size=64, topology_interval=5, h0_refresh_every=1, h1_refresh_every=1)
    transfer = dict(common, residual_input_scale=math.sqrt(64))
    stress = dict(common, geometry_objective='stress', steps=1200, lambda_h0=1.,
                  lambda_h1=.1, fuzzy_repulsion=.1, residual_input_scale=1.)
    pca = dict(stress, steps=0)
    configs = dict(zip(METHODS, (stress, transfer, pca, pca)))
    for config in configs.values():
        TDAConfig(**config).validate()
    overrides = {name: {k: v for k, v in c.items() if original[k] != v}
                 for name, c in configs.items()}
    return original, configs, overrides


def load_features(path):
    with np.load(path, allow_pickle=False) as data:
        train, test = data['train'], data['test']
    for array, count in ((train, 60000), (test, 10000)):
        if array.shape != (count, 64) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('expected full finite float32 official PCA64 feature splits')
    return train, test


def load_labels(path):
    # Called only AFTER model and KMeans fitting; never passed to reducers.
    with np.load(path, allow_pickle=False) as data:
        train, test = data['train_labels'], data['test_labels']
    for array, count in ((train, 60000), (test, 10000)):
        if (array.shape != (count,) or array.dtype.kind not in 'iu'
                or not np.array_equal(np.unique(array), np.arange(10))):
            raise ValueError('expected complete official ten-class label vectors')
    return train, test


def make_plan(args):
    original, configs, overrides = configs_from_har(args.har_registration)
    metadata = json.loads((args.features.parent / 'prepared.json').read_text())
    if (metadata['dataset'], metadata['n_train'], metadata['n_test'], metadata['reference_dim']) != ('fashion_mnist', 60000, 10000, 64):
        raise ValueError('unexpected cached preprocessing provenance')
    train, test = load_features(args.features)
    content_hash = hashlib.sha256(train.tobytes() + test.tobytes()).hexdigest()
    if content_hash != metadata['feature_sha256']:
        raise ValueError('features do not match existing TRAIN-only preparation record')
    return {
        'protocol': 'fashion-fixed-har-geometry-transfer-v1',
        'source_hashes': sources(),
        'har_preregistration_sha256': digest(args.har_registration),
        'selection': {'selected_candidate': SELECTED, 'locked': True,
                      'rule': 'transfer locked HAR TRAIN-validation selection; no Fashion selection'},
        'har_selected_config': original, 'har_selected_config_sha256': object_digest(original),
        'overrides_from_har_selected': overrides,
        'arms': [{'method': name, 'config': configs[name],
                  'config_sha256': object_digest(configs[name])} for name in METHODS],
        'seeds': list(SEEDS), 'expected_rows': len(SEEDS) * len(METHODS),
        'seed_rule': 'only seed changes within each registered method configuration',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'torch': str(torch.__version__), 'sklearn': sklearn.__version__,
                        **{p: importlib.metadata.version(p) for p in ('scipy', 'pynndescent', 'numba', 'threadpoolctl')}},
        'data': {'name': 'Fashion-MNIST', 'citation': 'Xiao, Rasul and Vollgraf (2017), arXiv:1708.07747',
                 'license': 'MIT, Zalando SE', 'train_rows': 60000, 'test_rows': 10000,
                 'features': 64, 'features_file_sha256': digest(args.features),
                 'features_content_sha256': content_hash,
                 'annotations_sha256': digest(args.annotations),
                 'preparation_sha256': digest(args.features.parent / 'prepared.json'),
                 'pca_map_sha256': digest(args.features.parent / 'reference_pca.npz'),
                 'provenance': 'reuse existing full official split in original order; TRAIN-only PCA64 and positive-pair median scale; hashes pin cached bytes, not independent archive authentication; no PCA refit or feature tuning'},
        'fit_rule': 'DeepTDA.fit(all 60000 TRAIN features) without validation_data; all preprocessing, PCA initialization, scale, graph and PH targets TRAIN-only; no labels opened until model and KMeans fitting complete',
        'conditioning': 'residual input only: sqrt(64)=8 replaces sqrt(561); PCA skip and reference metric unchanged',
        'neighbors': 'k15 pynndescent in production subprocess, exact global TRAIN recall audit on 64 seed-fixed queries, tie-aware minimum 0.9; strict and tie-aware recall exported without IDs; saved HAR timeout 300 seconds; rejection fails, no exact fallback',
        'evaluation_units': 'same cached PCA64 reference for EVERY method and seed; multiply model coordinates by its TRAIN-only reference_scale to undo the estimator extra normalization, as existing image worker does; saved model.transform output requires this scalar for display units',
        'population_geometry': 'reuse HAR geometry: same 256 TEST queries seed917 versus all 70000 TRAIN+TEST rows, k15, self excluded; no IDs exported',
        'topology': 'reuse HAR geometry/pilot.evaluate: native complete PH on three paired 64-point TEST subclouds, seeds917/918/919, nested in 512-point geometry draws; raw and scale-aligned H0/H1 costs, bottlenecks, counts and unmatched-long-bar fractions; alignment diagnostic only',
        'probe': 'reuse HAR probe: uniform Euclidean 15NN fit TRAIN coordinates/labels, evaluate all 10000 TEST for every method; accuracy and balanced accuracy',
        'clustering': 'HAR clustering definition with only k changed: KMeans(n_clusters=10,n_init=10,random_state=seed), other sklearn defaults; fit all TRAIN, predict all TEST; ARI and arithmetic NMI; fixed known k, never estimated or tuned',
        'baselines': 'PCA2 is production exact PCA skip via steps=0, no ANN or optimization; PCA64 is the unchanged cached reference, KMeans plus probe only, no redundant identity geometry/PH; all three seeds, both baselines mandatory once registered',
        'test_rule': 'no Fashion tuning, early stopping, retries, test selection, reselection or best-seed reporting; direct transfer not compute-matched or a new search',
        'resources': {'torch_threads': 1, 'threadpool_limit': 1, 'sequential_method_workers': True,
                      'worker_timeout_seconds': WORKER_TIMEOUT,
                      'timing_scope': 'fit includes estimator setup and TRAIN diagnostics; worker total includes imports, loading, evaluation and local persistence; no total RSS guarantee'},
        'failures': 'one exclusive subprocess attempt per row; persist all queued/running/ok/failed rows; failures retain available aggregates with stage/type; raw traceback local only; no automatic retries or overwrite',
        'aggregation': 'HAR summarize: arithmetic mean and population std over successful model seeds with explicit failure counts; three PH draws are diagnostics, not independent model replicates',
        'publication': 'aggregate-only JSON: configs, hashes, counts, scalar metrics; no paths, sample IDs, arrays, labels, predictions, diagrams, histories or personal information',
        'local_artifacts': 'private inference models without training data and compressed train/test embedding NPZ for all 2D methods; KMeans centers NPZ for all methods; raw logs; ignored output tree only',
        'caveat': 'Fashion data were used in existing image benchmarks; this fixed transfer makes no globally untouched TEST claim, full-population PH guarantee, or attribution of gains solely to objective (steps, weights and conditioning also differ)',
    }


def row_template(arm, seed):
    config = dict(arm['config'], seed=seed)
    return {'method': arm['method'], 'seed': seed, 'config': config,
            'config_sha256': object_digest(config), 'status': 'queued'}


def job_directory(output, method, seed):
    return output / 'local' / method / ('seed' + str(seed))


def audited_summary(diagnostics):
    audit = diagnostics['audit']
    return {k: audit[k] for k in AUDIT_KEYS}


def worker(args, plan):
    arm = next(a for a in plan['arms'] if a['method'] == args.worker)
    row = row_template(arm, args.seed)
    directory = job_directory(args.output, args.worker, args.seed)
    write_json(directory / 'attempt.json', {'status': 'started'}, exclusive=True)
    started = time.perf_counter()
    stage = 'load_features'
    try:
        train, test = load_features(args.features)
        row.update(status='running', stage=stage)
        write_json(directory / 'row.json', row)
        with threadpool_limits(limits=1):
            torch.set_num_threads(1)
            model = None
            if args.worker == 'baseline_pca64':
                ztrain, ztest = train, test
                row['fit_seconds'] = 0.
                row['neighbor_graph'] = {'skipped': True}
            else:
                stage = 'fit'
                row['stage'] = stage
                write_json(directory / 'row.json', row)
                model = DeepTDA(row['config']).fit(train)
                row['fit_seconds'] = float(model.fit_seconds_)
                row['display_reference_scale'] = float(model.reference_scale_)
                if row['config']['steps']:
                    diag = model.neighbor_diagnostics_
                    row['neighbor_graph'] = {'backend': diag['backend'],
                        'ann_subprocess': diag['ann_subprocess'], 'edge_count': diag['edge_count'],
                        'elapsed_seconds': diag['elapsed_seconds'], 'audit': audited_summary(diag)}
                    if not (diag['ann_subprocess'] and diag['audit']['performed'] and diag['audit']['accepted']):
                        raise RuntimeError('required ANN subprocess/audit not performed')
                else:
                    row['neighbor_graph'] = {'skipped': True}
                # Only explicitly whitelisted scalar coverage; never sample IDs.
                row['coverage'] = {k: model.training_coverage_[k] for k in
                    ('n_training_samples', 'optimizer_steps', 'geometry_unique_samples',
                     'h0_unique_samples', 'h1_unique_samples', 'h0_updates', 'h1_updates')}
                stage = 'transform'
                before = time.perf_counter()
                ztrain = model.embedding_ * model.reference_scale_
                ztest = model.transform(test) * model.reference_scale_
                row['test_transform_seconds'] = time.perf_counter() - before
                if ztrain.shape != (60000, 2) or ztest.shape != (10000, 2):
                    raise ValueError('incorrect embedding shapes')
                if not np.isfinite(ztrain).all() or not np.isfinite(ztest).all():
                    raise ValueError('nonfinite embeddings')
                stage = 'local_artifacts'
                model.save(directory / 'model.pt', include_training_data=False)
                np.savez_compressed(directory / 'embeddings.npz', Z_train=ztrain, Z_test=ztest)
                stage = 'geometry_and_topology'
                row['stage'] = stage
                write_json(directory / 'row.json', row)
                row.update(har.geometry(train, test, ztrain, ztest))
            stage = 'kmeans_fit'
            row['stage'] = stage
            write_json(directory / 'row.json', row)
            before = time.perf_counter()
            fitted = KMeans(n_clusters=10, n_init=10, random_state=args.seed).fit(ztrain)
            predicted = fitted.predict(ztest)
            row['kmeans_seconds'] = time.perf_counter() - before
            np.savez_compressed(directory / 'kmeans.npz', centers=fitted.cluster_centers_)
            stage = 'post_fit_labels'
            ytrain, ytest = load_labels(args.annotations)
            row['kmeans'] = {'test_ari': float(adjusted_rand_score(ytest, predicted)),
                'test_nmi': float(normalized_mutual_info_score(ytest, predicted, average_method='arithmetic'))}
            stage = 'probe'
            row['stage'] = stage
            write_json(directory / 'row.json', row)
            row['probe'] = har.probe(ztrain, ztest, ytrain, ytest)
            row['status'] = 'ok'
    except Exception as error:
        import traceback
        traceback.print_exc()  # Captured in ignored local worker.log, never JSON.
        row.update(status='failed', failed_stage=stage, error_type=type(error).__name__)
        if hasattr(error, 'diagnostics'):
            row['rejected_neighbor_audit'] = {k: error.diagnostics[k] for k in AUDIT_KEYS if k in error.diagnostics}
    row.pop('stage', None)
    row['measurement_seconds'] = time.perf_counter() - started
    write_json(directory / 'row.json', row)
    return 0 if row['status'] == 'ok' else 1


def read_worker_row(path, expected):
    measured = json.loads(path.read_text())
    if any(measured[k] != expected[k] for k in
           ('method', 'seed', 'config', 'config_sha256')):
        raise ValueError('worker identity/config mismatch')
    if measured['status'] not in ('running', 'ok', 'failed'):
        raise ValueError('unexpected worker status')
    return measured


def run(args, plan, registration):
    started = time.perf_counter()
    write_json(args.output / 'run_started.json', {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'preregistration_sha256': digest(registration)}, exclusive=True)
    records = [row_template(arm, seed) for seed in SEEDS for arm in plan['arms']]
    result = {'plan': plan, 'preregistration_sha256': digest(registration),
              'status': 'running', 'no_test_based_reselection': True, 'records': records}
    progress = args.output / 'progress.json'
    write_json(progress, result)
    for index, row in enumerate(records):
        directory = job_directory(args.output, row['method'], row['seed'])
        directory.mkdir(parents=True, exist_ok=False)
        row['status'] = 'running'
        write_json(progress, result)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', row['method'],
                   '--seed', str(row['seed']), '--features', str(args.features),
                   '--annotations', str(args.annotations), '--output', str(args.output),
                   '--har-registration', str(args.har_registration)]
        before = time.perf_counter()
        try:
            with (directory / 'worker.log').open('x') as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=WORKER_TIMEOUT, check=False)
            measured = read_worker_row(directory / 'row.json', row)
            row.update(measured)
            row['worker_returncode'] = child.returncode
            if child.returncode != 0 or row['status'] != 'ok':
                row.update(status='failed', failed_stage=row.get('failed_stage', 'worker_exit'),
                           error_type=row.get('error_type', 'WorkerExitError'))
        except Exception as error:
            partial = directory / 'row.json'
            if partial.is_file():
                try:
                    row.update(read_worker_row(partial, row))
                except Exception as partial_error:
                    # Corrupt or mismatched output must not change registered identity
                    # or prevent the remaining predeclared jobs from being attempted.
                    row['partial_record_error_type'] = type(partial_error).__name__
            row.update(status='failed', failed_stage=row.pop('stage', 'worker_process'),
                       error_type=type(error).__name__)
        row['total_seconds'] = time.perf_counter() - before
        records[index] = row
        write_json(progress, result)
        print(json.dumps({k: row[k] for k in ('method', 'seed', 'status', 'total_seconds')}), flush=True)
    failed = any(r['status'] != 'ok' for r in records)
    result['status'] = 'completed_with_failures' if failed else 'completed'
    result['summary'] = har.summarize(records, METHODS)
    result['total_seconds'] = time.perf_counter() - started
    result['source_hashes_verified_at_completion'] = sources() == plan['source_hashes']
    if not result['source_hashes_verified_at_completion']:
        result['status'] = 'invalidated_source_change'
        failed = True
    write_json(args.output / 'results.json', result, exclusive=True)
    write_json(progress, result)
    print(result['status'], flush=True)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preregister', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--worker', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--har-registration', type=Path, default=Path('outputs/geometry-optimization-confirmation/preregistration.json'))
    parser.add_argument('--features', type=Path, default=Path('outputs/v02-fashion_mnist/features.npz'))
    parser.add_argument('--annotations', type=Path, default=Path('outputs/v02-fashion_mnist/annotations.npz'))
    parser.add_argument('--output', type=Path, default=Path('outputs/fashion-geometry-confirmation'))
    args = parser.parse_args()
    if (ROOT / 'outputs').resolve() not in args.output.resolve().parents:
        parser.error('all output must remain under repository ignored outputs')
    # Resolve symlinks too: neither raw logs nor coordinates may escape outputs.
    local = args.output / 'local'
    if local.exists() and (ROOT / 'outputs').resolve() not in local.resolve().parents:
        parser.error('local output symlink escapes ignored outputs')
    registration = args.output / 'preregistration.json'
    if not args.worker and any((args.output / name).exists() for name in
                              ('run_started.json', 'progress.json', 'results.json', 'local')):
        parser.error('run already started; no resume, retry or overwrite')
    plan = make_plan(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preregister:
        write_json(registration, {'created_utc': datetime.now(timezone.utc).isoformat(),
                                  'plan': plan}, exclusive=True)
        print('Registered 12 fixed method/seed rows; no fitting or TEST metrics performed.')
        return 0
    if json.loads(registration.read_text())['plan'] != plan:
        parser.error('registration differs: source/config/data/environment changed; refusing fit')
    if args.worker:
        if not (args.output / 'run_started.json').is_file():
            parser.error('worker requires an exclusive registered parent run')
        return worker(args, plan)
    return run(args, plan, registration)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Top-level setup failures are explicit but do not leak paths in stdout.
        print(json.dumps({'status': 'failed', 'failed_stage': 'setup',
                          'error_type': type(error).__name__}), flush=True)
        raise
