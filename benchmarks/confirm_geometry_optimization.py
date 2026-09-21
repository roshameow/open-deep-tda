"""Registered official-split confirmation of the class-free production pilot.

From the repository, with PYTHONPATH=python:benchmarks, run --preregister only
AFTER the ablation has produced results, then --run with the same options.
Neither action is performed on import. This is confirmation on reused HAR data,
not fresh generalization evidence. Official TEST must never select a model.

Only aggregate JSON is suitable for public review. --save-local optionally saves
private inference checkpoints and coordinate NPZs under ignored outputs/.
Source-inspected reuse: pilot.evaluate strips query IDs, diagrams and curves;
DeepTDA steps=0 supplies its actual train-fitted PCA initialization/reference.
No benchmark execution or empirical verification is implied by this module.
"""
import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             balanced_accuracy_score, normalized_mutual_info_score)
from sklearn.neighbors import KNeighborsClassifier
import torch
from threadpoolctl import threadpool_limits

from open_deep_tda import DeepTDA
from open_deep_tda.config import TDAConfig
from open_deep_tda.real_datasets import HAR_CITATION, _validate_har
import tune_fuzzy_geometry as pilot


SEEDS = (0, 1, 2)
ORIGINAL = 'stress_600'
TOPOLOGY_CONTROL = 'graph_weak_2400'
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value, exclusive=False):
    """Reject nonfinite metrics; atomic progress, exclusive immutable records."""
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(text)
        temporary.replace(path)


def implementation_digest():
    """Pin local Python implementations without exposing machine/source paths."""
    paths = [Path(__file__).resolve(), Path(pilot.__file__).resolve()]
    paths.extend(sorted((ROOT / 'python' / 'open_deep_tda').rglob('*.py')))
    checksum = hashlib.sha256()
    for path in paths:
        checksum.update(path.name.encode())
        checksum.update(bytes.fromhex(pilot.digest(path)))
    return checksum.hexdigest()


def load_selection(path, conditioning_path=None):
    """Recompute class-free selection over complete registered development stages."""
    sources = [path] + ([conditioning_path] if conditioning_path is not None else [])
    ordered, stages = [], []
    for source in sources:
        result = json.loads(source.read_text())
        plan = result['plan']
        if plan['protocol'] not in ('production-geometry-ablation-v1', 'residual-conditioning-ablation-v1') or plan['seeds'] != [0]:
            raise ValueError('expected registered seed-0 ablation')
        names = [candidate['name'] for candidate in plan['candidates']]
        runs = result['runs']
        if len(names) != len(set(names)) or len(runs) != len(names) or {r['candidate'] for r in runs} != set(names):
            raise ValueError('pilot must contain every registered candidate exactly once')
        by_name = {row['candidate']: row for row in runs}
        stage_rows = []
        for candidate in plan['candidates']:
            row = dict(by_name[candidate['name']])
            if row.get('status', 'ok') != 'ok' or row['seed'] != 0:
                raise ValueError('pilot contains failed or unexpected seed')
            # Stage one predates this optional, backwards-compatible scalar.
            config = TDAConfig(**row['config']).validate().to_dict()
            declared = dict(plan['common'], **{k: v for k, v in candidate.items() if k != 'name'})
            if any(config[key] != value for key, value in declared.items()):
                raise ValueError('saved config disagrees with registration')
            if (config['mode'], config['optimizer_mode'], config['n_components'], config['device'], config['num_threads'], config['seed']) != ('geometry', 'parametric', 2, 'cpu', 1, 0):
                raise ValueError('expected seed-0 CPU parametric 2D geometry')
            row['config'] = config
            for key in ('knn_overlap', 'trustworthiness'):
                score = row['population_geometry'][key]
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('invalid class-free selection metric')
            stage_rows.append(row)
        winner = max(stage_rows, key=rank)
        if result['selected_candidate'] != winner['candidate']:
            raise ValueError('saved selection disagrees with registered ranking')
        ordered.extend(stage_rows)
        stages.append(dict(protocol=plan['protocol'], result_sha256=pilot.digest(source)))
    by_name = {r['candidate']: r for r in ordered}
    if len(by_name) != len(ordered) or not {ORIGINAL, TOPOLOGY_CONTROL} <= set(by_name):
        raise ValueError('missing controls or overlapping candidate names')
    winner = max(ordered, key=rank)
    controls = [ORIGINAL, TOPOLOGY_CONTROL, winner['candidate']]
    if conditioning_path is not None:
        controls.extend(['graph_weak_conditioned', 'graph_r1_conditioned'])
    arms = []
    for name in dict.fromkeys(controls):
        config = by_name[name]['config']
        if any(config[k] != by_name[ORIGINAL]['config'][k] for k in ('standardize', 'missing_indicators')):
            raise ValueError('reference preprocessing must match')
        roles = ['selected'] if name == winner['candidate'] else ['control']
        arms.append(dict(candidate=name, roles=roles, config=config))
    selection = dict(stages=stages, class_free=True,
        data='subject-grouped validation within official TRAIN; no activity labels',
        rule='maximum validation k15 overlap across registered stages; then trustworthiness; then registered order',
        selected_candidate=winner['candidate'],
        scores=[dict(candidate=r['candidate'], knn_overlap=rank(r)[0], trustworthiness=rank(r)[1]) for r in ordered])
    return selection, arms


def rank(row):
    return row['population_geometry']['knn_overlap'], row['population_geometry']['trustworthiness']


def make_plan(args):
    selection, arms = load_selection(args.pilot, args.conditioning_pilot)
    pca_config = dict(arms[0]['config'], steps=0)
    return {
        'protocol': 'production-geometry-confirmation-v1',
        'implementation_sha256': implementation_digest(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'torch': str(torch.__version__), 'sklearn': sklearn.__version__},
        'selection': selection, 'arms': arms, 'seeds': list(SEEDS),
        'seed_rule': 'only seed is overridden in each full saved arm config',
        'data': {
            'name': 'UCI HAR original 561-feature dataset', 'citation': HAR_CITATION,
            'license': 'CC BY 4.0 per UCI dataset page',
            'cache_sha256': pilot.digest(args.cache),
            'provenance': 'existing local feature cache; hash pins bytes, not independent archive authentication',
            'split': 'full official subject-disjoint TRAIN 7352 / TEST 2947',
            'features': 561, 'known_class_count': 6,
            'subjects': 'validate official subject-disjointness only; never model inputs',
        },
        'fit_rule': 'DeepTDA.fit(X_train) only, no validation_data; preprocessing, scale, PCA, graph and topology targets fit TRAIN only',
        'test_rule': 'no tuning, early stopping, best-seed selection or reselection on official TEST; report all declared arms and seeds',
        'labels_rule': 'labels only for post-fit 15NN probe and clustering diagnostics, never unsupervised fitting or selection',
        'population_geometry': '256 fixed TEST queries, seed 917, k=15, versus all 10299 TRAIN+TEST rows; exclude self; same query IDs for every arm/seed; IDs not exported',
        'probe': '15NN uniform Euclidean, TRAIN labels, predictions on all TEST; accuracy and balanced accuracy',
        'clustering': 'KMeans(n_clusters=6,n_init=10,random_state=seed), remaining sklearn defaults; fit TRAIN coordinates, predict all TEST; ARI and NMI(average_method=arithmetic); k fixed from known dataset class count, not estimated/tuned',
        'baselines': {
            'pca_config': pca_config,
            'definition': 'original production config with steps=0; DeepTDA train-fitted preprocessing and positive-pair median scaling; high-dimensional reference KMeans and exact production PCA2 KMeans',
            'pca_15nn': args.pca_probe,
            'reference': 'same reference per seed for all arms; scalar may vary across training seeds; no baseline refit on TEST',
        },
        'topology': 'pilot.evaluate uses evaluate_embedding(TEST reference, TEST Z,topology_size=64,k=15,seed=917/918/919); three fixed paired 64-point PH subsets nested in 512-point geometry samples; raw and scale-aligned H0/H1 costs and bottlenecks, bar counts and unmatched-long-bar fractions; alignment estimated on each 512-point TEST diagnostic sample, never used in training',
        'aggregation': 'per-seed aggregates and arithmetic mean/population std across successful seeds; failure counts explicit; PH draws are diagnostics, not independent model seeds or confidence intervals',
        'failures': 'record stage and exception type only, retain available metrics; continue all declared tasks; no automatic retries or overwrites',
        'resources': 'CPU, one thread; unequal steps/compute explicitly not compute-matched; fit and total wall times reported, no total-RSS claim',
        'local_artifacts': args.save_local,
        'publication': 'aggregate metrics/configs/hashes only; no paths, sample IDs, predictions, labels, coordinates, diagrams, histories or subject-level results in JSON',
        'caveat': 'HAR including official TEST was reused from prior development; this is not a fresh or globally untouched generalization estimate. Subset H0/H1 is not full-population topology. The weak arm is an attribution control, not a new search.',
    }


def load_cache(path):
    keys = ('X_train', 'X_test', 'y_train', 'y_test', 'subjects_train', 'subjects_test')
    with np.load(path, allow_pickle=False) as cache:
        if not set(keys) <= set(cache.files):
            raise ValueError('cache lacks official feature, label or subject fields')
        data = {key: cache[key] for key in keys}
    _validate_har(data)
    for key in keys:
        if key.startswith('X_'):
            if data[key].dtype.kind not in 'fiu':
                raise ValueError('features must be real numeric values')
            data[key] = np.ascontiguousarray(data[key], dtype=np.float32)
            if not np.isfinite(data[key]).all():
                raise ValueError('features overflow float32')
        elif data[key].dtype.kind not in 'iu':
            raise ValueError('labels and subjects must be integer vectors')
    return data


def clustering(train, test, ytest, seed):
    fitted = KMeans(n_clusters=6, n_init=10, random_state=seed).fit(train)
    predicted = fitted.predict(test)
    return {'test_ari': float(adjusted_rand_score(ytest, predicted)),
            'test_nmi': float(normalized_mutual_info_score(ytest, predicted, average_method='arithmetic'))}


def probe(train, test, ytrain, ytest):
    predicted = KNeighborsClassifier(n_neighbors=15, weights='uniform', p=2, n_jobs=1).fit(train, ytrain).predict(test)
    return {'test_accuracy': float(accuracy_score(ytest, predicted)),
            'test_balanced_accuracy': float(balanced_accuracy_score(ytest, predicted))}


def geometry(reference_train, reference_test, train, test):
    # Reuse the pilot's fixed draws and strict aggregate whitelist, not report_.
    measured = pilot.evaluate(reference_train, reference_test, train, test)
    return {'population_geometry': measured['population_geometry'],
            'test_subsets': measured['validation_subsets'],
            'test_subset_mean': measured['validation_subset_mean'],
            'evaluation_seconds': measured['evaluation_seconds']}


def save_local(output, method, seed, model, train, test):
    # Caller restricts this to the repository's ignored outputs tree.
    directory = output / 'local' / method / ('seed' + str(seed))
    if (ROOT / 'outputs').resolve() not in directory.resolve().parents:
        raise ValueError('local artifacts must remain inside ignored outputs')
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / 'model.pt', include_training_data=False)
    np.savez_compressed(directory / 'embeddings.npz', Z_train=train, Z_test=test)


def measure(method, seed, config, data, output, save, with_probe=True, high_dim=False):
    started = time.perf_counter()
    row = dict(method=method, seed=seed, config=dict(config, seed=seed), status='running')
    stage = 'fit'
    try:
        model = DeepTDA(row['config']).fit(data['X_train'])
        row['fit_seconds'] = float(model.fit_seconds_)
        stage = 'transform'
        reference_train = model.reference_
        reference_test = model.reference_transform(data['X_test'])
        train = reference_train if high_dim else model.embedding_
        test = reference_test if high_dim else model.transform(data['X_test'])
        if not high_dim:
            stage = 'geometry_and_topology'
            row.update(geometry(reference_train, reference_test, train, test))
        stage = 'kmeans'
        row['kmeans'] = clustering(train, test, data['y_test'], seed)
        if with_probe:
            stage = 'probe'
            row['probe'] = probe(train, test, data['y_train'], data['y_test'])
        if save and not high_dim:
            stage = 'local_artifacts'
            save_local(output, method, seed, model, train, test)
        row['status'] = 'ok'
    except Exception as error:
        # Exception text/tracebacks can disclose paths or arrays: never export.
        row.update(status='failed', failed_stage=stage, error_type=type(error).__name__)
    row['total_seconds'] = time.perf_counter() - started
    return row


def summarize(records, methods):
    result = {}
    for method in methods:
        rows = [r for r in records if r['method'] == method]
        good = [r for r in rows if r['status'] == 'ok']
        summary = {'expected_seed_count': len(SEEDS), 'completed_seed_count': len(good),
                   'failed_seed_count': len(rows) - len(good), 'metrics': {}}
        for group in ('population_geometry', 'kmeans', 'probe', 'test_subset_mean'):
            keys = sorted({k for row in good for k, v in row.get(group, {}).items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)})
            for key in keys:
                values = [row[group][key] for row in good if key in row.get(group, {})]
                summary['metrics'][group + '.' + key] = {
                    'count': len(values), 'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=0)),
                }
        result[method] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preregister', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--pilot', type=Path, default=Path('outputs/geometry-ablation/results.json'))
    parser.add_argument('--conditioning-pilot', type=Path, help='optional completed residual-conditioning ablation')
    parser.add_argument('--cache', type=Path, default=Path('data/uci_har/features.npz'))
    parser.add_argument('--output', type=Path, default=Path('outputs/geometry-optimization-confirmation'))
    parser.add_argument('--pca-probe', action='store_true', help='also report PCA2 15NN; use the same flag for registration/run')
    parser.add_argument('--save-local', action='store_true', help='save private models/NPZs in ignored outputs; use the same flag for registration/run')
    args = parser.parse_args()
    if not args.pilot.is_file():
        parser.error('completed production ablation results are required; no fallback winner is allowed')
    if args.save_local and (ROOT / 'outputs').resolve() not in args.output.resolve().parents:
        parser.error('--save-local requires an output subdirectory of repository outputs')
    plan = make_plan(args)
    args.output.mkdir(parents=True, exist_ok=True)
    registered = args.output / 'preregistration.json'
    progress = args.output / 'progress.json'
    result_path = args.output / 'results.json'
    marker = args.output / 'run_started.json'
    if any(path.exists() for path in (progress, result_path, marker)):
        parser.error('a run already started here; no silent overwrite, resume or retry')
    if args.preregister:
        write_json(registered, {'created_utc': datetime.now(timezone.utc).isoformat(),
                                'plan': plan}, exclusive=True)
        print('Registered fixed confirmation; no fitting or TEST metrics performed.')
        return
    if json.loads(registered.read_text())['plan'] != plan:
        parser.error('registration differs: pilot/cache/code/config/options changed; refusing TEST evaluation')
    # Validate before taking the exclusive run lock. A killed run leaves the lock
    # and incremental progress intact rather than silently allowing a rerun.
    data = load_cache(args.cache)
    started = time.perf_counter()
    write_json(marker, {'started_utc': datetime.now(timezone.utc).isoformat(),
                        'preregistration_sha256': pilot.digest(registered)}, exclusive=True)
    result = {'plan': plan, 'preregistration_sha256': pilot.digest(registered),
              'status': 'running', 'no_test_based_reselection': True,
              'environment': plan['environment'],
              'records': []}
    jobs = [(arm['candidate'], arm['config'], True, False) for arm in plan['arms']]
    jobs.extend([('baseline_high_dim', plan['baselines']['pca_config'], False, True),
                 ('baseline_pca2', plan['baselines']['pca_config'], args.pca_probe, False)])
    torch.set_num_threads(1)
    write_json(progress, result)
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            for method, config, with_probe, high_dim in jobs:
                row = measure(method, seed, config, data, args.output, args.save_local,
                              with_probe=with_probe, high_dim=high_dim)
                result['records'].append(row)
                write_json(progress, result)
                print(json.dumps({'method': method, 'seed': seed, 'status': row['status']}), flush=True)
    failed = any(row['status'] != 'ok' for row in result['records'])
    result['status'] = 'completed_with_failures' if failed else 'completed'
    result['summary'] = summarize(result['records'], [job[0] for job in jobs])
    result['total_seconds'] = time.perf_counter() - started
    write_json(result_path, result, exclusive=True)
    write_json(progress, result)
    print(result['status'], flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
