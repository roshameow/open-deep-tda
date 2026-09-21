"""Isolated standard-UMAP worker, executed as a script (does not import torch).

UMAP's optional TensorFlow/ParametricUMAP plugin is deliberately unavailable in
this process: this benchmark evaluates standard UMAP, which does not need it.
Native dependency crashes/timeouts cannot terminate the training benchmark.
"""
import json
import sys
import numpy as np


def main():
    input_path, output_path, config_json = sys.argv[1:]
    options = json.loads(config_json)
    sys.modules["tensorflow"] = None
    import umap
    X = np.load(input_path, allow_pickle=False)
    Z = umap.UMAP(**options).fit_transform(X)
    if not np.isfinite(Z).all():
        raise ValueError("UMAP returned nonfinite coordinates")
    np.save(output_path, Z)


if __name__ == "__main__":
    main()
