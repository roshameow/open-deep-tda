"""The public PHRE figures must not substitute a graph-only training result."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks/render_phre_comparison.py"


def module():
    spec = importlib.util.spec_from_file_location("phre_figure_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_full_row_contract_and_labels():
    m = module()
    with pytest.raises(ValueError, match="no point exclusion"):
        m.embedding(np.zeros((9, 2)), 10)
    with pytest.raises(ValueError, match="no point exclusion"):
        m.embedding(np.array([[np.nan, 1.0]]), 1)
    with pytest.raises(ValueError, match="annotation/class"):
        m.render("fashion", np.zeros(9999, dtype=int), [], "/tmp/not-written.png")


def test_frozen_comparison_provenance_and_scopes():
    summary = json.loads((ROOT / "benchmarks/results/phre_external_comparison.json").read_text())
    baseline = json.loads((ROOT / "benchmarks/results/graph_core_confirmation.json").read_text())
    assert summary["entrypoint"] == "open_deep_tda.DeepTDA"
    assert all(summary["ph_config_weights"][d][k] > 0
               for d in ("fashion", "har", "coil")
               for k in ("lambda_h0", "lambda_h1", "lambda_critical"))
    for row in summary["three_seed_test_kmeans"]:
        method = "strong" if "DeepTDA" in row["method"] else "umap"
        original = next(r for r in baseline["summary"]
                        if r["dataset"] == row["dataset"] and r["method"] == method)
        assert row["test_kmeans_train_fitted_ari_mean"] == original["statistics"]["ari"]["mean"]
    coil = summary["coil_fixed_seed0"]
    for method, rowname in (("PH-Regularized Embedding (DeepTDA)", "strong"),
                            ("UMAP", "umap")):
        record = next(r for r in baseline["records"]
                      if r["dataset"] == "coil" and r["method"] == rowname and r["seed"] == 0)
        assert coil["method_ari"][method] == record["ari"]
    assert coil["topoaepp_training_updates"] == 1000
    for figure in summary["visuals"]:
        assert figure["panels"][0] == "PH-Regularized Embedding (DeepTDA)"
        assert figure["full_test_extent"] and figure["no_point_removal"]
        image = ROOT / figure["file"]
        import hashlib
        assert hashlib.sha256(image.read_bytes()).hexdigest() == figure["sha256"]


def test_readmes_show_only_ph_main_and_external_baselines():
    for name in ("README.md", "README.zh-CN.md"):
        text = (ROOT / name).read_text()
        assert "PH-Regularized Embedding" in text and "TopoAE++" in text and "UMAP" in text
        assert "GraphEmbedding" not in text and "graph-core-" not in text
        assert "visual-contracts-" not in text and "docs/" not in text
        assert "v0.3.0" not in text
        assert all(f"assets/phre-vs-external-{d}.png" in text for d in ("fashion", "har", "coil"))
