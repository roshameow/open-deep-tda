"""Private SCRIPT entry point for NN-descent; never run through package -m.

Only this process imports pynndescent/Numba. Do not add imports from
open_deep_tda: its __init__ imports torch, unsafe with some native ANN stacks.
The parent supplies: input.npy output.npz k seed. No pickle or package import.
"""
import sys
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


def main():
    if "torch" in sys.modules or "open_deep_tda" in sys.modules:
        raise RuntimeError("ANN worker must start without torch or open_deep_tda imported")
    input_path, output_path, k_arg, seed_arg = sys.argv[1:]
    k, seed = int(k_arg), int(seed_arg)
    try:
        from pynndescent import NNDescent
    except ImportError as exc:
        raise RuntimeError(
            "pynndescent ANN dependency is unavailable in the worker interpreter; "
            "install pynndescent there or include its directory in parent PYTHONPATH"
        ) from exc
    X = np.load(input_path, mmap_mode="r", allow_pickle=False)
    n = len(X)
    if X.ndim != 2 or X.dtype != np.float64 or not 0 < k < n:
        raise ValueError("invalid ANN worker input shape, dtype, or k")
    # NNDescent uses float32 internally; reject overflow instead of silently
    # indexing infinities. The parent audit always uses original float64 X.
    with np.errstate(over="ignore", invalid="ignore"):
        index_data = np.asarray(X, dtype=np.float32)
    flat = index_data.reshape(-1)
    for start in range(0, flat.size, 4096):
        if not np.isfinite(flat[start:start + 4096]).all():
            raise ValueError("X is not finite/representable in ANN float32; rescale X")
    width = min(n, max(30, 2 * k + 1))
    index = NNDescent(index_data, n_neighbors=width, metric="euclidean",
                      random_state=seed, n_jobs=1, low_memory=True,
                      compressed=False, parallel_batch_queries=False)
    candidates, distances = index.neighbor_graph
    if candidates.shape != (n, width) or candidates.dtype.kind not in "iu":
        raise RuntimeError("NNDescent returned malformed candidate IDs")
    neighbors = np.empty((n, k), dtype=np.int64)
    for i, row in enumerate(candidates):
        if np.any(row < 0) or np.any(row >= n) or not np.isfinite(distances[i]).all():
            raise RuntimeError("NNDescent returned invalid candidates/distances; rescale X "
                               "or choose an explicitly different backend")
        ids = np.unique(row[row != i])
        if len(ids) < k:
            raise RuntimeError("NNDescent returned too few distinct non-self candidates")
        # Re-rank the candidate set in original float64 Euclidean geometry;
        # candidate discovery itself is approximate and may omit tied IDs.
        exact_distances = cdist(X[i:i + 1], X[ids], metric="euclidean")[0]
        if not np.isfinite(exact_distances).all():
            raise ValueError("ANN candidate distances overflowed; rescale X")
        order = np.lexsort((ids, exact_distances))
        neighbors[i] = ids[order[:k]]
    if "torch" in sys.modules or "open_deep_tda" in sys.modules:
        raise RuntimeError("ANN dependency imported torch/package; isolation violated")
    np.savez(output_path, neighbors=neighbors)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = float(rss / (1024**2 if sys.platform == 'darwin' else 1024))
    except ImportError:
        peak = None
    Path(output_path).with_suffix('.metrics.json').write_text(json.dumps({
        'child_peak_rss_mib': peak,
        'scope': 'isolated ANN worker high-water RSS including imports/JIT/index/data; not a hard memory cap'}))


if __name__ == "__main__":
    main()
