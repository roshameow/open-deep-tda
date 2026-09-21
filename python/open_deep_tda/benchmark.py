"""Reproducible fixed-reference comparisons and topology ablations (no label tuning)."""
from dataclasses import replace
import time
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import TDAConfig
from .estimator import DeepTDA
from .evaluation import evaluate_embedding


def run_benchmark(X, config=None, seeds=(0, 1, 2, 3, 4), methods=None):
    base = config or TDAConfig()
    if isinstance(base, dict):
        base = TDAConfig(**base)
    base.validate()
    methods = list(methods or ["pca", "tsne", "umap", "ae", "topoae_h0", "deep_tda", "no_h0", "no_h1", "no_critical", "local_only"])
    allowed = {"pca", "tsne", "umap", "ae", "topoae_h0", "deep_tda", "no_h0", "no_h1", "no_critical", "local_only"}
    if not set(methods) <= allowed:
        raise ValueError(f"unknown benchmark methods: {set(methods) - allowed}")
    results = []
    for seed in seeds:
        cfg = replace(base, seed=int(seed))
        # Learn/fix one reference per seed, then all reducers receive the same U.
        start = time.perf_counter()
        reference_model = DeepTDA(replace(cfg, steps=0, lambda_h0=0, lambda_h1=0)).fit(X)
        U = reference_model.reference_
        reference_seconds = time.perf_counter() - start
        for method in methods:
            start = time.perf_counter()
            record = {"method": method, "seed": int(seed), "reference_seconds": reference_seconds,
                      "reference_mode": cfg.mode, "n_samples": len(U)}
            if method == "pca":
                q = min(cfg.n_components, len(U), U.shape[1])
                Z = PCA(n_components=q, svd_solver="full").fit_transform(U)
                Z = np.pad(Z, ((0, 0), (0, cfg.n_components - q)))
            elif method == "tsne":
                if len(U) < 3:
                    record.update(status="skipped", reason="t-SNE requires at least 3 points in this benchmark")
                    results.append(record)
                    continue
                kwargs = dict(n_components=cfg.n_components, random_state=int(seed),
                              perplexity=min(30.0, max(1.0, (len(U) - 1) / 3)),
                              init="pca" if min(U.shape) >= cfg.n_components else "random", learning_rate=200.0)
                Z = TSNE(**kwargs).fit_transform(U)
            elif method == "umap":
                if len(U) < 4:
                    record.update(status="skipped", reason="UMAP needs >=4 samples here")
                    results.append(record)
                    continue
                options = dict(n_components=cfg.n_components, n_neighbors=max(2, min(cfg.n_neighbors, len(U)-1)),
                               random_state=int(seed), n_jobs=1)
                env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with tempfile.TemporaryDirectory(prefix="deep-tda-umap-") as directory:
                    source_path, target_path = Path(directory) / "source.npy", Path(directory) / "target.npy"
                    np.save(source_path, U)
                    try:
                        completed = subprocess.run([sys.executable, str(Path(__file__).with_name("_baseline_worker.py")),
                            str(source_path), str(target_path), json.dumps(options)], env=env,
                            capture_output=True, text=True, timeout=180)
                    except subprocess.TimeoutExpired:
                        record.update(status="failed", reason="isolated UMAP exceeded 180-second timeout")
                        results.append(record)
                        continue
                    if completed.returncode != 0:
                        status = "skipped" if "No module named 'umap'" in completed.stderr else "failed"
                        record.update(status=status, returncode=completed.returncode, reason=completed.stderr[-3000:])
                        results.append(record)
                        continue
                    Z = np.load(target_path, allow_pickle=False)
                record["isolated_process_startup_included"] = True
            else:
                changes = dict(mode="geometry", standardize=False, missing_indicators=False,
                               optimizer_mode="parametric")
                if method == "ae":
                    changes.update(geometry_objective="stress", lambda_near=0, lambda_sep=0, lambda_h0=0, lambda_h1=0, lambda_reconstruction=1)
                elif method == "topoae_h0":
                    changes.update(geometry_objective="stress", lambda_near=0, lambda_sep=0, lambda_h0=1, lambda_h1=0, lambda_reconstruction=1)
                elif method == "no_h0":
                    changes["lambda_h0"] = 0
                elif method == "no_h1":
                    changes["lambda_h1"] = 0
                elif method == "no_critical":
                    changes["lambda_critical"] = 0
                elif method == "local_only":
                    changes["topology_sampling"] = "local"
                fitted = DeepTDA(replace(cfg, **changes)).fit(U)
                Z = fitted.embedding_ * fitted.reference_scale_
                record["training_config"] = fitted.config.to_dict()
                record["reference_rescale_correction"] = fitted.reference_scale_
            record["reduction_seconds"] = time.perf_counter() - start
            record["metrics"] = evaluate_embedding(U, Z, topology_size=cfg.evaluation_size,
                seed=int(seed) + 991, k=cfg.n_neighbors, budgets=cfg.evaluation_budgets())
            record["status"] = "ok"
            results.append(record)
    return {"protocol": "Identical frozen reference features per seed; labels unused. No automatic hyperparameter search.",
            "caveats": ["topoae_h0 is an independent H0-only reconstruction baseline, not the original paper implementation.",
                         "Runtime includes each reducer's configured training/diagnostics; reference extraction cost is separate. UMAP also includes isolated-process startup/JIT.",
                         "Standard UMAP is isolated from optional TensorFlow imports and native crashes; failed baselines are explicitly recorded.",
                         "Raw and scale-aligned topology metrics must be read together; no universal ranking is implied."],
            "config": base.to_dict(), "results": results}
