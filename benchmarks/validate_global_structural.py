#!/usr/bin/env python3
"""Frozen, all-TRAIN public-API structural replay; no downloads or private imports.

An explicit NPZ member supplies already-prepared TRAIN features. Only that member
is read (allow_pickle=False); TEST, labels and other members are never loaded.
The caller is responsible for TRAIN-only upstream preprocessing and permissions.
Rows are converted to float64, without centering or standardizing their features.
The source scale is the median of ALL positive full-TRAIN cdist(X,X) entries.
Automatic selection uses cdist(X,X)/scale; construction recomputes cdist on
X/scale, matching the declared as-run arithmetic. Their threshold graphs are
checked explicitly and the constructor independently validates its source.
There is no subsampling, label selection or H1 subcloud.

Policy fixed before computation: exact GraphEmbedding, seed 0, 300 epochs,
negative rate 5; one longest global Ripser bar, 20%-80% interval; first global
MST fundamental cycle pairing one; one constructor attempt at H0 tolerance .05.
The guide is fit to the original supplied TRAIN features, mean-centered, then
scaled by median positive normalized source15 distances / median positive guide
source15 distances. Source row IDs from the public graph API define those pairs.
There is no retry, bar switching, tolerance adjustment or success-only filtering.

--preregister pins this script, public implementation sources, runtime versions,
input archive/member hashes and fixed settings. --run verifies the registration,
claims a NEW output with exclusive creation BEFORE invoking any estimator, and
retains failed/unsupported outcomes. A started/interrupted output is not reusable.
The registration is a prospective REPLAY freeze, not a retroactive assertion that
an already-known historical success was originally preregistered. --run explicitly
consents to the optional external Ripser call; its internal time/storage are not
covered by sparse-analysis budgets. No model, coordinates, cycles, point IDs,
trees, hole centers or raw exception/worker logs are written to public outputs.
Graph fit scratch is ephemeral; do not enable its debug artifact retention.

The reported H0 and selected source/target H1 apply to ALL supplied TRAIN rows,
with the same row IDs. The additional planar proof concerns real Euclidean
geometry of represented float64 coordinates, NOT exact equivalence to rounded
pdist/cdist. No all-H1, universal feasibility, OOS H0 or official-benchmark claim
is made. TEST induction and post-barrier label metrics are separate historical
ablations, not performed by this TRAIN-only reproducer.

Example (paths are supplied by the user, not discovered from an environment):
  python benchmarks/validate_global_structural.py --preregister --input features.npz \
      --member train --registration frozen.json --output replay.json
  python benchmarks/validate_global_structural.py --run --input features.npz \
      --registration frozen.json --output replay.json

Source-root discovery uses pathlib only. Numerical dependencies are imported
lazily after single-thread settings. Output metadata contains relative source
names and digests, never the machine's source root or input/output paths.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'global-structural-replay-v1'
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_FEATURE_ENTRIES = 1_048_576
SOURCE_FILES = (
    'benchmarks/validate_global_structural.py',
    'python/open_deep_tda/__init__.py',
    'python/open_deep_tda/graph_embedding.py',
    'python/open_deep_tda/_graph_worker.py',
    'python/open_deep_tda/neighbors.py',
    'python/open_deep_tda/structural_auto_witness.py',
    'python/open_deep_tda/structural_sparse_h1.py',
    'python/open_deep_tda/structural_constructive.py',
    'python/open_deep_tda/structural_planar.py',
)
SETTINGS = {
    'seed': 0, 'epochs': 300, 'negative_rate': 5, 'neighbor_backend': 'exact',
    'max_vertices': 1024, 'min_vertices': 16, 'max_feature_entries': MAX_FEATURE_ENTRIES,
    'h0_tolerance': .05, 'k': 15, 'external_ripser_consent': True,
    'source_scale': 'median of every strictly positive full-TRAIN cdist entry',
    'selection_distance': 'cdist(X,X)/scale',
    'construction_distance': 'cdist(X/scale,X/scale); freshly validate source family',
    'guide_scale': 'median positive source15 D / median positive guide source15 norm',
    'selection': 'fixed longest positive finite global bar; first lex global MST cycle pairing one',
    'interval': 'birth+.2*(death-birth), death-.2*(death-birth)',
    'attempts': 1, 'labels_read': False, 'test_rows_read': False,
}


class ProtocolError(ValueError):
    """Protocol refusal. Messages are bounded public codes, not exception dumps."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes():
    return {name: _file_hash(ROOT/name) for name in SOURCE_FILES}


def _runtime():
    versions = {}
    for name in ('numpy', 'scipy', 'numba', 'ripser', 'scikit-learn', 'threadpoolctl'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'packages': versions}


def _numeric_environment():
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '1'
    source = str(ROOT/'python')
    if source not in sys.path:
        sys.path.insert(0, source)


def _load_train(input_path, member):
    """Bounded header-first read of one member; no other array is inspected."""
    if not isinstance(member, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', member):
        raise ProtocolError('invalid_member_name')
    path = Path(input_path)
    if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ProtocolError('input_archive_missing_or_over_budget')
    before = _file_hash(path)
    import numpy as np
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            selected = [info for info in infos if info.filename == member+'.npy']
            if len(infos) > 4096 or len(selected) != 1:
                raise ProtocolError('missing_duplicate_or_unbounded_member')
            if selected[0].file_size > MAX_MEMBER_BYTES:
                raise ProtocolError('member_byte_budget')
            payload = archive.read(selected[0])
        stream = io.BytesIO(payload)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ProtocolError('unsupported_npy_header')
        if (len(shape) != 2 or not 16 <= shape[0] <= 1024 or not 1 <= shape[1]
                or shape[0]*shape[1] > MAX_FEATURE_ENTRIES
                or dtype.kind not in 'fiu' or dtype.itemsize > 8):
            raise ProtocolError('invalid_or_over_budget_train_matrix')
        if stream.tell()+math.prod(shape)*dtype.itemsize != len(payload):
            raise ProtocolError('invalid_member_size')
        original = np.load(io.BytesIO(payload), allow_pickle=False)
        with np.errstate(over='ignore', invalid='ignore'):
            X = np.array(original, dtype=np.float64, order='C', copy=True)
        if not np.isfinite(X).all():
            raise ProtocolError('nonfinite_train_features')
    except (OSError, zipfile.BadZipFile, KeyError, EOFError, UnicodeError) as exc:
        raise ProtocolError('invalid_npz_input') from exc
    if _file_hash(path) != before:
        raise ProtocolError('input_changed_during_read')
    return X, {'archive_sha256': before, 'member': member, 'member_sha256': _sha(payload),
               'shape': list(X.shape), 'original_dtype': dtype.str, 'analysis_dtype': '<f8'}


def _output_pin(path):
    return _sha(str(Path(path).resolve()).encode())


def _validate_distinct(input_path, registration_path, output_path):
    paths = [Path(p).resolve() for p in (input_path, registration_path, output_path)]
    if len(set(paths)) != 3:
        raise ProtocolError('input_registration_output_must_be_distinct')


def preregister(input_path, member, registration_path, output_path):
    """Freeze a new replay without fitting, choosing a bar, or reading labels."""
    _validate_distinct(input_path, registration_path, output_path)
    if Path(output_path).exists() or Path(registration_path).exists():
        raise ProtocolError('registration_and_output_must_be_new')
    _numeric_environment()
    _, metadata = _load_train(input_path, member)
    record = {'schema': SCHEMA, 'status': 'preregistered',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'fixed-policy prospective TRAIN replay of a known diagnostic; not original-study preregistration',
              'input': metadata, 'settings': dict(SETTINGS), 'source_sha256': _source_hashes(),
              'runtime': _runtime(), 'output_path_sha256': _output_pin(output_path)}
    Path(registration_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(registration_path).open('x', encoding='utf8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return record


def _array_hash(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    return _sha(_canonical([list(a.shape), a.dtype.str])+a.tobytes())


def _h1_summary(result):
    keys = ('vertices', 'edges', 'triangles', 'basis_width', 'words_per_vector',
            'word_ops', 'peak_storage_words', 'boundary_rank', 'selected_rank')
    return {'certified': bool(result.certified), 'rank': int(result.rank),
            'all_birth_edges_present': all(s.edges_present for s in result.statuses),
            'all_individually_survive': all(s.survives for s in result.statuses),
            'work': {key: int(result.work[key]) for key in keys}}


def _overlap(D, points):
    import numpy as np
    from scipy.spatial.distance import cdist
    target = cdist(points, points)
    source = D.copy()
    np.fill_diagonal(source, np.inf)
    np.fill_diagonal(target, np.inf)
    first = np.argsort(source, axis=1, kind='stable')[:, :15]
    second = np.argsort(target, axis=1, kind='stable')[:, :15]
    return float(np.mean([len(set(a)&set(b))/15 for a, b in zip(first, second)]))


def _construction_reason(reason):
    """Categorize known refusals without copying arbitrary exception text/IDs."""
    text = str(reason).lower()
    for fragment, category in (
        ('source selected', 'source_family_not_certified'),
        ('zero contracted', 'unsupported_zero_seed_chord'),
        ('minor-arc', 'unsupported_circle_chords'),
        ('circle-incompatible', 'unsupported_seed_hierarchy'),
        ('protected-hole', 'protected_hole_guard_failed'),
        ('contact search', 'contact_search_failed'),
        ('h0 tolerance', 'h0_tolerance_not_proved'),
        ('target selected h1', 'target_h1_not_certified'),
        ('planar certificate', 'planar_not_certified'),
        ('numerical', 'numerical_construction_unsupported'),
        ('procrustes', 'numerical_construction_unsupported'),
    ):
        if fragment in text:
            return category
    return 'fixed_construction_not_certified'


def _compute_replay(X, registration):
    """One public-API attempt; returns only bounded sanitized summary metadata."""
    import numpy as np
    from dataclasses import asdict
    from scipy.spatial.distance import cdist
    from threadpoolctl import threadpool_limits
    from open_deep_tda.graph_embedding import GraphEmbedding
    from open_deep_tda.structural_auto_witness import auto_global_witnesses
    from open_deep_tda.structural_constructive import construct_global_layout, full_hierarchy, pairwise_planar, LayoutLimits
    from open_deep_tda.structural_sparse_h1 import H1Limits, analyze_sparse_h1
    from open_deep_tda.structural_planar import Limits as PlanarLimits, certify
    with threadpool_limits(limits=1):
        raw_D = cdist(X, X)
        if not np.isfinite(raw_D).all():
            raise ProtocolError('source_distance_overflow')
        positive = raw_D[raw_D > 0]
        if not len(positive):
            raise ProtocolError('no_positive_source_scale')
        scale = float(np.median(positive))
        selection_D = raw_D/scale
        normalized = X/scale
        D = cdist(normalized, normalized)
        if not np.isfinite(D).all() or not np.isfinite(selection_D).all():
            raise ProtocolError('normalized_source_distance_overflow')
        selected = auto_global_witnesses(selection_D, allow_external=True)
        result = {'status': 'unsupported', 'vertices': len(X), 'features': X.shape[1],
                  'source_pair_median': scale, 'selection_certified': bool(selected.certified),
                  'limits': {'sparse_h1': asdict(H1Limits()), 'constructor': asdict(LayoutLimits()),
                             'planar': asdict(PlanarLimits())},
                  'source_distance_sha256': _array_hash(D),
                  'selection_distance_sha256': _array_hash(selection_D),
                  'normalized_features_sha256': _array_hash(normalized), 'attempts': 1,
                  'scope': 'all supplied TRAIN rows; selected same-ID family only',
                  'external_solver_budget': 'Ripser internal time/storage outside bounded core budgets',
                  'planar_scope': 'real Euclidean represented float64 coordinates; not rounded-pdist equivalence'}
        if not selected.certified:
            result.update(stage='automatic_selection', reason='fixed_candidate_not_certified')
            return result
        cycles, a, b = selected.cycles, selected.birth, selected.survival
        result.update(birth=a, survival=b, cycle_count=len(cycles),
                      cycle_edges=sum(map(len, cycles)), cycles_sha256=_sha(_canonical(cycles)),
                      selection_source=_h1_summary(selected.source),
                      source=_h1_summary(analyze_sparse_h1(D, cycles, a, b)),
                      selection_construction_threshold_graphs_equal=bool(
                          np.array_equal(D <= a, selection_D <= a)
                          and np.array_equal(D <= b, selection_D <= b)))
        if not result['source']['certified']:
            result.update(stage='construction_source', reason='source_family_not_certified')
            return result
        graph = GraphEmbedding(neighbor_backend='exact', seed=0, epochs=300, negative_rate=5,
                               max_vertices=1024, max_feature_entries=MAX_FEATURE_ENTRIES)
        raw_guide = graph.fit_transform(X)
        ids, _ = graph.source_knn_
        guide = raw_guide.copy()
        guide -= guide.mean(axis=0)
        source_nn = D[np.arange(len(D))[:, None], ids]
        guide_nn = np.linalg.norm(guide[:, None]-guide[ids], axis=2)
        if not np.any(source_nn > 0) or not np.any(guide_nn > 0):
            raise ProtocolError('no_positive_guide_scale')
        guide_scale = float(np.median(source_nn[source_nn > 0])/np.median(guide_nn[guide_nn > 0]))
        guide *= guide_scale
        if not np.isfinite(guide).all():
            raise ProtocolError('nonfinite_calibrated_guide')
        result.update(raw_guide_sha256=_array_hash(raw_guide), guide_sha256=_array_hash(guide),
                      guide_scale=guide_scale, graph_seed=0, graph_backend='exact')
        centered = normalized-normalized.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        pca = centered@axes[:2].T
        result['overlap15'] = {'unconstrained_exact_graph': _overlap(D, guide), 'pca2': _overlap(D, pca)}
        built = construct_global_layout(D, guide, cycles, a, b, h0_tolerance=.05)
        result['constructor_certified'] = bool(built.certified)
        if not built.certified:
            safe_work = {key: int(built.diagnostics[key]) for key in
                         ('inserted_vertices', 'parent_tests', 'candidate_points', 'contact_pairs')}
            result.update(stage='construction', reason=_construction_reason(built.reason),
                          h0_error=built.h0_error, construction_work=safe_work)
            return result
        # Recreate target matrix independently; external checks do not accept a
        # successful flag alone. No coordinates/IDs escape this function.
        Z = built.embedding
        target = np.hypot(Z[:, 0, None]-Z[None, :, 0], Z[:, 1, None]-Z[None, :, 1])
        verified_target = analyze_sparse_h1(target, cycles, a, b)
        h0 = float(np.max(np.abs(full_hierarchy(D)-full_hierarchy(target))))
        planar = certify(Z, cycles, a, b, built.hole.reshape(1, 2))
        result.update(status='certified' if verified_target.certified and h0 <= .05 and planar.certified else 'verification_failed',
                      stage='complete', target=_h1_summary(verified_target),
                      h0={'all_rows': len(X), 'max_merge_error': h0, 'tolerance': .05, 'certified': h0 <= .05},
                      planar={'certified': bool(planar.certified), 'rank': int(planar.rank),
                              'all_rows': len(X), 'distance_pairs': int(planar.work['distance_pairs'])},
                      embedding_sha256=_array_hash(Z))
        result['overlap15']['constructive'] = _overlap(D, Z)
        result['overlap15']['constructive_minus_graph'] = result['overlap15']['constructive']-result['overlap15']['unconstrained_exact_graph']
        result['overlap15']['constructive_minus_pca'] = result['overlap15']['constructive']-result['overlap15']['pca2']
        return result


def run_registered(input_path, registration_path, output_path):
    """Consume one frozen registration into an exclusive, fail-closed output."""
    _validate_distinct(input_path, registration_path, output_path)
    _numeric_environment()
    if Path(output_path).exists():
        raise ProtocolError('output_already_claimed_no_retry')
    if Path(registration_path).stat().st_size > 128*1024:
        raise ProtocolError('registration_too_large')
    registration_bytes = Path(registration_path).read_bytes()
    registration = json.loads(registration_bytes)
    if (registration.get('schema') != SCHEMA or registration.get('status') != 'preregistered'
            or registration.get('settings') != SETTINGS
            or registration.get('output_path_sha256') != _output_pin(output_path)):
        raise ProtocolError('registration_contract_mismatch')
    if registration.get('source_sha256') != _source_hashes():
        raise ProtocolError('source_hash_mismatch')
    if registration.get('runtime') != _runtime():
        raise ProtocolError('runtime_mismatch')
    X, metadata = _load_train(input_path, registration['input']['member'])
    if metadata != registration['input']:
        raise ProtocolError('input_hash_or_shape_mismatch')
    report = {'schema': SCHEMA, 'status': 'started', 'registration_sha256': _sha(registration_bytes),
              'source_sha256': registration['source_sha256'], 'runtime': registration['runtime'],
              'input': metadata, 'settings': dict(SETTINGS),
              'scope': 'TRAIN-only replay; no TEST/labels/official-benchmark or general-feasibility claim'}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Keep the exclusively-created descriptor: never reopen another writer's file.
    with Path(output_path).open('x+', encoding='utf8') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
        try:
            # Public output must not inherit external solver/worker raw logs.
            with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                outcome = _compute_replay(X, registration)
            if _source_hashes() != registration['source_sha256'] or _file_hash(input_path) != metadata['archive_sha256']:
                raise ProtocolError('source_or_input_changed_during_run')
            report.update(status=outcome['status'], result=outcome)
            _canonical(report)  # refuse nonfinite/unserializable values before write
        except Exception as exc:
            # Never publish str(exc), tracebacks, paths or a worker-log tail.
            report.update(status='failed', result={'status': 'failed', 'reason': 'computation_failed',
                          'error_type': type(exc).__name__[:64], 'attempts': 1})
        stream.seek(0); stream.truncate()
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preregister', action='store_true')
    action.add_argument('--run', action='store_true', help='one frozen replay; explicitly permits optional external Ripser')
    parser.add_argument('--input', type=Path, required=True, help='local NPZ supplied by the user; never downloaded')
    parser.add_argument('--member', help='explicit TRAIN member name, required for preregistration')
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='NEW sanitized result JSON; never overwritten/retried')
    args = parser.parse_args(argv)
    if args.preregister and not args.member:
        parser.error('--member is mandatory with --preregister')
    if args.run and args.member is not None:
        parser.error('--run uses the frozen member; do not override it')
    try:
        if args.preregister:
            result = preregister(args.input, args.member, args.registration, args.output)
        else:
            result = run_registered(args.input, args.registration, args.output)
    except Exception as exc:
        print(json.dumps({'status': 'refused', 'reason': 'protocol_validation_failed',
                          'error_type': type(exc).__name__[:64]}, sort_keys=True))
        return 2
    print(json.dumps({'status': result['status'], 'schema': SCHEMA}, sort_keys=True))
    return 0 if result['status'] in ('preregistered', 'certified') else 1


if __name__ == '__main__':
    raise SystemExit(main())
