"""Exact, bounded-workspace kNN and subprocess-isolated, audited NN-descent.

No optional ANN dependency is imported in this process. Memory accounting and audit semantics are exposed in returned diagnostics and covered by tests.
"""
import math
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile

import numpy as np
from scipy.spatial.distance import cdist


_MIB = 1024 * 1024


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be in [{minimum}, {maximum or 'infinity'}]")
    return value


def _real(value, name, minimum=0, maximum=None, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if (not math.isfinite(value) or value < minimum or (positive and value == minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} is outside its valid range")
    return value


def _workspace(n, k, budget, backend):
    # Conservative live array allowance for partition, tie masks/IDs, lexsort,
    # and per-row validation. This is not an allocator/RSS guarantee.
    scratch = 64 * n + 64 * k + 4096 if k else 0
    if not k:
        rows = 0
    else:
        rows = min(n - 1, (budget - scratch) // (8 * n))
        if rows < 1:
            required = scratch + 8 * n
            raise ValueError(
                f"working_memory_mb budget is too small: need at least {required} bytes "
                "for one full-population distance row plus selection scratch")
        if backend == "pynndescent":
            rows = 1  # The global audit processes one random query at a time.
    return {
        "budget_bytes": budget,
        "distance_block_rows": rows,
        "distance_block_bytes": 8 * rows * n,
        "selection_scratch_allowance_bytes": scratch,
        "estimated_workspace_bytes": 8 * rows * n + scratch,
        "scope": "parent distance/selection workspace, not total RSS or ANN worker memory",
    }


def _select(distances, k):
    """Lexicographic (distance, row ID) top-k, including partition-boundary ties."""
    boundary = np.partition(distances, k - 1)[k - 1]
    inside = np.flatnonzero(distances < boundary)
    # flatnonzero is ID ordered; sorting only an arbitrary argpartition prefix
    # would NOT resolve ties straddling the k-th boundary.
    tied = np.flatnonzero(distances == boundary)[:k - len(inside)]
    ids = np.concatenate((inside, tied))
    return ids[np.lexsort((ids, distances[ids]))]


def _distance_rows(X, query, out):
    cdist(query, X, metric="euclidean", out=out)
    for row in out:
        if not np.isfinite(row).all():
            raise ValueError("Euclidean distances overflowed; rescale X")


def _exact(X, k, block_rows):
    n = len(X)
    neighbors = np.empty((n, k), dtype=np.int64)
    # Reuse a single block: assigning a fresh cdist result can briefly keep TWO
    # blocks live. Never allocate an N-by-N distance matrix, even for tiny N.
    block = np.empty((block_rows, n), dtype=np.float64)
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        distances = block[:stop - start]
        _distance_rows(X, X[start:stop], distances)
        for offset, row in enumerate(distances):
            row[start + offset] = np.inf
            neighbors[start + offset] = _select(row, k)
    return neighbors


def _read_worker_output(path, n, k):
    """Check the NPY header BEFORE allocating an array from a worker archive."""
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["neighbors.npy"]:
                raise ValueError("expected only neighbors.npy")
            info = archive.getinfo("neighbors.npy")
            if info.file_size > n * k * 8 + 1024:
                raise ValueError("output exceeds the expected neighbor-array byte budget")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported NPY version")
                if shape != (n, k) or fortran or dtype.kind not in "iu" or dtype.itemsize > 8:
                    raise ValueError("wrong neighbor shape, layout, or integer dtype")
                if info.file_size != stream.tell() + n * k * dtype.itemsize:
                    raise ValueError("incorrect neighbor-array payload size")
        with np.load(path, allow_pickle=False) as archive:
            raw = archive["neighbors"]
        # Validate before casting unsigned integers, avoiding wraparound.
        for i, row in enumerate(raw):
            if (np.any(row < 0) or np.any(row >= n) or np.any(row == i)
                    or len(np.unique(row)) != k):
                raise ValueError(f"row {i} has out-of-range, self, or duplicate neighbor IDs")
        return raw.astype(np.int64, copy=False)
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Malformed ANN worker output: {exc}") from exc


def _ann(X, k, seed, timeout, diagnostics=None):
    worker = Path(__file__).with_name("_neighbor_worker.py").resolve()
    with tempfile.TemporaryDirectory(prefix="open-deep-tda-neighbors-") as directory:
        directory = Path(directory)
        input_path, output_path = directory / "input.npy", directory / "output.npz"
        np.save(input_path, X, allow_pickle=False)
        # Preserve PYTHONPATH (including optional dependency targets). Executing
        # a SCRIPT, never `-m open_deep_tda...`, bypasses __init__ and torch.
        env = os.environ.copy()
        for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            env[name] = "1"
        command = [sys.executable, str(worker), str(input_path), str(output_path),
                   str(k), str(seed)]
        # File-backed logs avoid unbounded capture_output pipes in the parent.
        with (directory / "worker.log").open("w+b") as log:
            try:
                result = subprocess.run(command, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"ANN worker timed out after {timeout:g} seconds") from exc
            except OSError as exc:
                raise RuntimeError(f"Could not start ANN worker: {exc}") from exc
            if result.returncode:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 8192))
                detail = log.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ANN worker failed (exit {result.returncode}): {detail}")
        neighbors = _read_worker_output(output_path, len(X), k)
        stats = output_path.with_suffix('.metrics.json')
        if diagnostics is not None:
            diagnostics['child_peak_rss_mib'] = None
            if stats.exists():
                if stats.stat().st_size > 8192:
                    raise RuntimeError('Malformed ANN worker memory metadata')
                values = json.loads(stats.read_text())
                peak = values.get('child_peak_rss_mib')
                if peak is not None and (isinstance(peak, bool) or not isinstance(peak, (int,float)) or not math.isfinite(peak) or peak < 0):
                    raise RuntimeError('Malformed ANN worker peak RSS')
                diagnostics.update(values)
        return neighbors


def _audit(X, neighbors, count, seed, min_recall):
    n, k = neighbors.shape
    queries = np.sort(np.random.default_rng(seed).choice(n, min(n, count), replace=False))
    block = np.empty((1, n), dtype=np.float64)
    strict_hits, tie_hits = 0, 0
    for query in queries:
        _distance_rows(X, X[query:query + 1], block)
        row = block[0]
        row[query] = np.inf
        truth = _select(row, k)
        proposed = neighbors[query]
        strict_hits += len(np.intersect1d(truth, proposed, assume_unique=True))
        boundary = row[truth[-1]]
        inside = truth[row[truth] < boundary]
        # Merely counting all returned distances <= boundary would forgive
        # missing strictly closer points when many points tie at the boundary.
        tie_hits += len(np.intersect1d(inside, proposed, assume_unique=True))
        tie_hits += min(k - len(inside), int(np.count_nonzero(row[proposed] == boundary)))
    denominator = len(queries) * k
    strict_recall, tie_recall = strict_hits / denominator, tie_hits / denominator
    audit = {
        "performed": True, "query_count": len(queries), "query_indices": queries.tolist(),
        "population_size": n, "strict_id_recall": strict_recall,
        "tie_aware_distance_recall": tie_recall,
        "acceptance_metric": "tie_aware_distance_recall", "min_recall": min_recall,
        "accepted": tie_recall >= min_recall,
        "scope": "uniform queries against ALL population rows; self excluded",
        "tie_rule": "exact float64 distance equality; strictly closer IDs remain mandatory",
    }
    if not audit["accepted"]:
        error = RuntimeError(
            f"ANN recall audit rejected graph: tie-aware recall {tie_recall:.6f} "
            f"< min_recall {min_recall:g} (strict ID recall {strict_recall:.6f}); "
            "no exact fallback was performed")
        error.diagnostics = audit
        raise error
    return audit


def _edges(neighbors):
    """Sorted unique undirected union, without Python sets or an N-by-N mask."""
    n, k = neighbors.shape
    if not k:
        return np.empty((0, 2), dtype=np.int64)
    keys = np.empty(n * k, dtype=np.int64)
    for i, row in enumerate(neighbors):
        keys[i * k:(i + 1) * k] = np.minimum(i, row) * n + np.maximum(i, row)
    keys.sort(kind="quicksort")
    keep = np.empty(len(keys), dtype=bool)
    keep[0] = True
    keep[1:] = keys[1:] != keys[:-1]
    unique = keys[keep]
    del keys, keep
    edges = np.empty((len(unique), 2), dtype=np.int64)
    edges[:, 0] = unique // n
    edges[:, 1] = unique % n
    return edges


def build_neighbor_graph(X, k, backend="exact", working_memory_mb=64, seed=0,
                         audit_queries=64, min_recall=0.9, timeout_seconds=300):
    """Return ``{edges: int64(E,2), neighbors: int64(N,K), diagnostics: dict}``.

    K=min(k,max(0,N-1)); k=0 and empty/singleton inputs yield empty graphs.
    X must be a finite real numeric matrix with at least one feature. Distances
    are Euclidean, ties use current input row IDs (not permutation-invariant
    identities). Edges are the sorted unique undirected neighbor union.

    working_memory_mb bounds an explicit estimate of parent distance/selection
    workspace, excluding input, O(N*K) output/edge assembly and the ANN process.
    An insufficient one-row budget raises ValueError. ANN always uses a separate
    script process, then a global exact audit. min_recall gates tie-aware recall;
    strict ID recall is also reported. Rejection raises RuntimeError with audit
    data in ``exception.diagnostics``, NEVER silently switches to exact.
    timeout_seconds limits the ANN subprocess only, not exact search/the audit.
    """
    started = time.perf_counter()
    if not isinstance(backend, str) or backend not in ("exact", "pynndescent"):
        raise ValueError("backend must be 'exact' or 'pynndescent'")
    k = _integer(k, "k")
    seed = _integer(seed, "seed", maximum=2**32 - 1)
    audit_queries = _integer(audit_queries, "audit_queries", minimum=1)
    working_memory_mb = _real(working_memory_mb, "working_memory_mb", positive=True,
                              maximum=sys.maxsize / _MIB)
    min_recall = _real(min_recall, "min_recall", maximum=1)
    timeout_seconds = _real(timeout_seconds, "timeout_seconds", positive=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            raw = np.asarray(X)
    except (TypeError, ValueError, Warning) as exc:
        raise ValueError('X must be a rectangular real numeric matrix') from exc
    if raw.ndim != 2 or raw.shape[1] == 0 or raw.dtype.kind not in "iuf":
        raise ValueError("X must be a real numeric matrix with at least one feature")
    n, dimensions = raw.shape
    if n > math.isqrt(np.iinfo(np.int64).max):
        raise ValueError("population too large for int64 edge IDs")
    effective_k = min(k, max(0, n - 1))
    memory = _workspace(n, effective_k, int(working_memory_mb * _MIB), backend)
    with np.errstate(over="ignore", invalid="ignore"):
        X = np.ascontiguousarray(raw, dtype=np.float64)
    flat = X.reshape(-1)
    for start in range(0, flat.size, 4096):
        if not np.isfinite(flat[start:start + 4096]).all():
            raise ValueError("X must contain only finite float64-representable values")
    audit = {
        "performed": False, "query_count": 0, "query_indices": [], "population_size": n,
        "strict_id_recall": None, "tie_aware_distance_recall": None,
        "acceptance_metric": "tie_aware_distance_recall", "min_recall": min_recall,
        "accepted": True, "reason": "exact by construction" if effective_k else "empty graph",
    }
    ann_stats = {}
    if not effective_k:
        neighbors = np.empty((n, 0), dtype=np.int64)
    elif backend == "exact":
        neighbors = _exact(X, effective_k, memory["distance_block_rows"])
    else:
        neighbors = _ann(X, effective_k, seed, timeout_seconds, diagnostics=ann_stats)
        audit = _audit(X, neighbors, audit_queries, seed, min_recall)
    edges = _edges(neighbors)
    memory.update({
        "float64_input_bytes": 8 * n * dimensions,
        "neighbors_bytes": int(neighbors.nbytes), "edges_bytes": int(edges.nbytes),
        "edge_assembly_extra_upper_estimate_bytes": 32 * n * effective_k + 64 * effective_k,
        "ann_worker_memory_bounded": False,
        "exclusions": "caller input/conversion, outputs/edge assembly, library/runtime/allocator "
                      "overhead, filesystem cache, and NN-descent/Numba child process",
    })
    diagnostics = {
        "backend": backend, "population_size": n, "n_features": dimensions,
        "requested_k": k, "effective_k": effective_k, "seed": seed,
        "metric": "euclidean", "tie_break": "distance then current input row ID",
        "edge_count": len(edges), "memory": memory, "audit": audit,
        "ann_subprocess": backend == "pynndescent" and effective_k > 0,
        "ann_worker": ann_stats,
        "candidate_neighbors": min(n, max(30, 2 * effective_k + 1))
                               if backend == "pynndescent" and effective_k else None,
        "timeout_seconds": timeout_seconds, "timeout_scope": "ANN subprocess only",
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {"edges": edges, "neighbors": neighbors, "diagnostics": diagnostics}
