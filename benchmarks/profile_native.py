"""Reproducible native PH profile (logical entries, not process RSS)."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from open_deep_tda.topology import distance_matrix, persistence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='benchmarks/results/native_profile.json')
    args = parser.parse_args()
    persistence(np.zeros((1, 1)))  # exclude extension import from all measurements
    rows = []
    for n in (16, 32, 64, 128):
        D = distance_matrix(np.random.default_rng(17).normal(size=(n, 8)))
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            result = persistence(D)
            durations.append(time.perf_counter() - start)
        rows.append(dict(n=n, dimension=8, median_seconds=float(np.median(durations)),
                         seconds=durations, n_simplices=result['n_simplices'],
                         operations=result['reduction_operations'], peak_entries=result['peak_reduction_entries']))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(platform=platform.platform(), python=sys.version, seed=17,
        repeats=3, measurement='warm import; native wrapper wall time, input distances precomputed',
        measurements=rows), indent=2))
    print(path)


if __name__ == '__main__':
    main()
