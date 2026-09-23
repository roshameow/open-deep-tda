"""Offline publication contracts: bounded summaries, synthetic inputs, no real runs.

Do not turn local caches, optional baseline packages, or external executables into
ordinary-test prerequisites. PNG inspection uses only the standard library.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import zlib

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/results/visual_contracts.json"
DRIVER = ROOT / "benchmarks/visual_contracts/reproduce_baselines.py"
# Construct synthetic rejecting examples; do not publish identifying-path literals.
LOCAL_PREFIXES = tuple("/" + part + "/" for part in ("Users", "home", "tmp"))
PRIVATE_PREFIX = "/".join(("docs", "research"))
RAW_KEYS = {
    "coordinates", "coords", "embedding", "embeddings", "embedding_array",
    "raw_points", "raw_data", "raw_payload", "raw_outputs", "raw_logs", "raw_log",
    "model", "models", "model_state", "model_weights", "state_dict", "checkpoint",
    "train_labels", "test_labels", "sample_ids", "row_ids", "query_ids",
    "input_path", "output_path", "cache_path", "traceback",
}


def assert_public_summary(value):
    """Reject payload fields and local paths, not legitimate scalar metrics."""
    if isinstance(value, dict):
        for key, child in value.items():
            assert isinstance(key, str)
            assert key.lower() not in RAW_KEYS, f"raw payload key: {key}"
            assert_public_summary(key)
            assert_public_summary(child)
    elif isinstance(value, list):
        assert not (value and all(isinstance(row, list) and row
                                 and all(isinstance(x, (int, float)) for x in row)
                                 for row in value)), "raw numeric matrix"
        for child in value:
            assert_public_summary(child)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        assert not any(prefix in normalized for prefix in LOCAL_PREFIXES), "local path"
        assert PRIVATE_PREFIX not in normalized, "private research path"
        assert not re.search(r"\b[A-Za-z]:/", normalized), "local drive path"
        assert not normalized.startswith(("file://", "~/")), "local path"
    elif isinstance(value, float):
        assert math.isfinite(value), "nonfinite summary"
    else:
        assert value is None or isinstance(value, (bool, int)), "non-JSON payload"


def read_public_json():
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, f"duplicate JSON key: {key}"
            result[key] = value
        return result
    return json.loads(EVIDENCE.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def png_chunks(raw):
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    offset = 8
    chunks = []
    while offset < len(raw):
        assert offset + 12 <= len(raw), "truncated PNG chunk"
        length, kind = struct.unpack_from(">I4s", raw, offset)
        end = offset + 12 + length
        assert end <= len(raw), "truncated PNG payload"
        payload = raw[offset + 8:offset + 8 + length]
        crc, = struct.unpack_from(">I", raw, offset + 8 + length)
        assert crc == zlib.crc32(kind + payload) & 0xffffffff, "PNG checksum"
        chunks.append((kind, payload))
        offset = end
        if kind == b"IEND":
            assert length == 0 and offset == len(raw), "trailing PNG payload"
            break
    assert chunks and chunks[0][0] == b"IHDR" and chunks[-1][0] == b"IEND"
    assert len(chunks[0][1]) == 13
    assert any(kind == b"IDAT" for kind, _ in chunks)
    return chunks


@pytest.mark.parametrize("key", sorted(RAW_KEYS))
def test_privacy_guard_rejects_nested_raw_payload_fields(key):
    with pytest.raises(AssertionError, match="raw payload"):
        assert_public_summary({"protocols": [{"metrics": {key: [[1., 2.]]}}]})


@pytest.mark.parametrize("path", [
    *(prefix + "fixture/run" for prefix in LOCAL_PREFIXES),
    PRIVATE_PREFIX + "/fixture/result.json",
    "C:" + "\\" + "fixture" + "\\" + "data", "file:" + "///cache", "~/cache",
])
def test_privacy_guard_rejects_paths_in_values_and_keys(path):
    for data in ({"sources": [{"description": path}]}, {path: "summary"}):
        with pytest.raises(AssertionError, match="path"):
            assert_public_summary(data)


def test_privacy_guard_accepts_numerical_summary_metrics():
    assert_public_summary({"metrics": {"radius": 1.25, "cycles": 4, "counts": [0, 1, 4],
                                      "ari": -.1, "completed": False, "score": None},
                           "path": "assets/visual-contracts-example.png",
                           "source": "https://example.org/paper"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_privacy_guard_rejects_nonfinite_metrics(value):
    with pytest.raises(AssertionError, match="nonfinite"):
        assert_public_summary({"metrics": {"radius": value}})


def test_visual_followups_do_not_promote_a_new_core_default():
    # Inspect syntax, not package imports (which could load optional backends).
    tree = ast.parse((ROOT / "python/open_deep_tda/config.py").read_text())
    config = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "TDAConfig")
    defaults = {node.target.id: ast.literal_eval(node.value)
                for node in config.body if isinstance(node, ast.AnnAssign)}
    assert defaults["geometry_objective"] == "stress"
    assert defaults["output_calibration"] == "none"
    assert defaults["optimizer_mode"] == "parametric"
    tree = ast.parse((ROOT / "python/open_deep_tda/graph_embedding.py").read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "GraphEmbedding")
    init = next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    defaults = {arg.arg: ast.literal_eval(value)
                for arg, value in zip(init.args.kwonlyargs, init.args.kw_defaults)}
    assert defaults["mapping_domain"] == "train_hull"
    assert defaults["epochs"] == 300 and defaults["negative_rate"] == 5


def test_public_source_has_no_local_research_paths():
    sources = [EVIDENCE, DRIVER, ROOT / "benchmarks/visual_contracts/README.md",
               ROOT / "README.md", ROOT / "README.zh-CN.md", Path(__file__)]
    for path in sources:
        assert path.is_file(), f"missing public artifact: {path.relative_to(ROOT)}"
        text = path.read_text(encoding="utf-8")
        assert not any(prefix in text for prefix in LOCAL_PREFIXES)
        assert PRIVATE_PREFIX not in text


@pytest.mark.parametrize("relative", ["README.md", "README.zh-CN.md",
                                       "benchmarks/visual_contracts/README.md"])
def test_public_readme_local_links_resolve(relative):
    readme = ROOT / relative
    targets = re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", readme.read_text())
    assert targets, "reproduction guide must link its public evidence"
    for target in targets:
        if re.match(r"(?:https?://|mailto:|#)", target):
            continue
        target = target.split("#", 1)[0]
        # Commands may use explicit user-supplied placeholders, not local caches.
        if not target or any(part in target for part in ("<", ">", "YOUR_", "{", "}")):
            continue
        path = (readme.parent / target).resolve()
        assert path.is_relative_to(ROOT), "README link escapes repository"
        assert path.exists(), f"broken README link: {target}"


def assert_png_metadata_public(raw):
    chunks = png_chunks(raw)
    allowed = {b"IHDR", b"IDAT", b"IEND", b"PLTE", b"tRNS", b"pHYs",
               b"sRGB", b"gAMA", b"cHRM", b"sBIT", b"bKGD", b"tEXt"}
    for kind, payload in chunks:
        assert kind in allowed, f"unreviewed PNG metadata chunk: {kind!r}"
        if kind == b"tEXt":
            key, separator, value = payload.partition(b"\x00")
            assert separator and key == b"Software", "unreviewed PNG text payload"
            assert len(value) <= 256, "unbounded PNG text payload"
            assert_public_summary(value.decode("latin1"))
    return struct.unpack_from(">II", chunks[0][1])


def tiny_png(extra=()):
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    parts = [(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)),
             *extra, (b"IDAT", zlib.compress(b"\x00\x00\x00\x00")), (b"IEND", b"")]
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunk(*part) for part in parts)


def test_png_reader_accepts_clean_image():
    assert assert_png_metadata_public(tiny_png()) == (1, 1)
    assert assert_png_metadata_public(tiny_png([(b"tEXt", b"Software\x00Matplotlib")])) == (1, 1)


@pytest.mark.parametrize("mutation", ["truncated", "trailing", "checksum", "raw_chunk",
                                         "raw_text", "path", "compressed_text", "exif"])
def test_png_reader_rejects_corrupt_or_unreviewed_payload(mutation):
    raw = tiny_png()
    if mutation == "truncated":
        raw = raw[:-1]
    elif mutation == "trailing":
        raw += b"attached dataset"
    elif mutation == "checksum":
        raw = raw[:-1] + bytes([raw[-1] ^ 1])
    else:
        kind, payload = {
            "raw_chunk": (b"raWd", b"1,2,3"),
            "raw_text": (b"tEXt", b"coordinates\x00[[1,2]]"),
            "path": (b"tEXt", b"Software\x00" + (LOCAL_PREFIXES[0] + "fixture").encode()),
            "compressed_text": (b"zTXt", b"data\x00\x00" + zlib.compress(b"[[1,2]]")),
            "exif": (b"eXIf", b"unreviewed"),
        }[mutation]
        raw = tiny_png([(kind, payload)])
    with pytest.raises(AssertionError):
        assert_png_metadata_public(raw)


@pytest.fixture
def driver(monkeypatch):
    # Import and every test using this fixture are guarded against optional fits
    # and process launches. A runner test may replace the guard with its own stub.
    import builtins
    import subprocess

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"umap", "ripser", "torch", "pynndescent"}, (
            "ordinary protocol tests must not import optional fit/oracle packages"
        )
        return original_import(name, *args, **kwargs)

    def no_process(*args, **kwargs):
        pytest.fail("ordinary protocol tests must not execute external programs")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, no_process)
    spec = importlib.util.spec_from_file_location("_public_visual_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset", ["K4", "Twist", "COIL20-1", "Digits"])
def test_four_public_figures_are_bounded_clean_pngs(dataset):
    path = ROOT / "assets" / f"visual-contracts-{dataset}.png"
    assert path.is_file(), f"missing public figure: {path.name}"
    raw = path.read_bytes()
    assert len(raw) <= 20 * 1024 * 1024, "unbounded figure payload"
    width, height = assert_png_metadata_public(raw)
    assert 800 <= width <= 20000 and 400 <= height <= 20000


def indexed(records, key="id"):
    result = {row[key]: row for row in records}
    assert len(result) == len(records), f"duplicate {key}"
    return result


def assert_evidence_contract(document):
    assert document["schema_version"] == 1
    assert set(document) == {"schema_version", "scope", "sources", "datasets", "protocols",
                             "baseline_results", "candidate_outcomes", "same_cycle_gate",
                             "figures", "limitations"}
    assert_public_summary(document)
    scope = document["scope"]
    assert scope["core_unchanged"] is True
    for field in ("production_defaults_changed", "candidates_are_defaults", "paper_reproduction"):
        assert scope[field] is False, field
    assert scope["new_fits_for_publication"] == 0
    sources = indexed(document["sources"])
    assert "not an independent reimplementation" in sources["topoaepp"]["role"]
    assert re.fullmatch(r"[0-9a-f]{40}", sources["topoaepp"]["commit"])
    for source in sources.values():
        if "sha256" in source:
            assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
    assert document["limitations"]

    baselines = indexed(document["baseline_results"])
    for dataset, updates in (("3Clusters", 427), ("Digits", 0)):
        key = f"raw-{dataset}-TopoAE++-adapter"
        assert key in baselines, "missing author timeout"
        row = baselines[key]
        assert row["status"] == "timeout" and row["completed_updates"] == updates
        assert row["timeout_seconds"] == 150 and row["failure"]
        assert "metrics" not in row, "timeout must not be replaced with proxy scores"
    assert "source persistence" in baselines["raw-Digits-TopoAE++-adapter"]["failure"].lower()

    candidates = indexed(document["candidate_outcomes"])
    assert set(candidates) == {"multiscale_log_distance", "linear_tail", "oracle_critical_edges"}
    assert all(row["promoted_to_default"] is False for row in candidates.values())
    log = candidates["multiscale_log_distance"]
    assert log["decision"] == "rejected_catastrophic_local_and_clustering_regression"
    records = indexed(log["records"], "fit_scope")
    assert set(records) == {"split", "transductive"}
    for row in records.values():
        assert row["cohorts"]["fit"]["ari"] < .2
        assert row["cohorts"]["fit"]["knn_overlap15"] < .05
    assert records["transductive"]["optimizer_converged"] is False
    assert records["transductive"]["status"] == "iteration_cap_last_iterate"

    tail = candidates["linear_tail"]
    assert tail["decision"] == "not_confirmed_no_promotion"
    assert tail["development_seed"] == 0 and tail["confirmation_seeds"] == [1, 2]
    assert tail["positive_new_seed_fit_ari_pairs"] == 1 and tail["new_seed_fit_pairs"] == 4
    pairs = {(r["fit_scope"], r["cohort"], r["seed"], r["method"]): r for r in tail["records"]}
    expected = {(scope, cohort, seed, method)
                for scope, cohorts in (("split", ("fit", "heldout397", "combined1797_inductive")),
                                       ("transductive", ("fit",)))
                for cohort in cohorts for seed in (0, 1, 2)
                for method in ("original", "linear_tail", "author_umap")}
    assert set(pairs) == expected and len(pairs) == len(tail["records"])
    positive = 0
    for fit_scope in ("split", "transductive"):
        for seed in (1, 2):
            old = pairs[fit_scope, "fit", seed, "original"]
            new = pairs[fit_scope, "fit", seed, "linear_tail"]
            positive += new["metrics"]["ari"] > old["metrics"]["ari"]
            assert new["metrics"]["silhouette"] < old["metrics"]["silhouette"]
            for key in ("overlap", "trust"):
                assert new["metrics"]["neighbors15"][key] < old["metrics"]["neighbors15"][key]
            assert (new["h1_source_aligned"]["betti_L1_relative"]
                    > old["h1_source_aligned"]["betti_L1_relative"])
    assert positive == 1
    fashion = tail["fashion_transfer"]
    assert fashion["decision"] == "negative_transfer_for_local_structure_and_information_agreement"
    old, new = fashion["metrics"]["old_graph"], fashion["metrics"]["linear_tail"]
    assert new["ari"] > old["ari"]  # Mixed transfer, not an invented all-metric loss.
    for key in ("nmi", "silhouette_fixed2048", "overlap15_fixed512"):
        assert new[key] < old[key]

    oracle = candidates["oracle_critical_edges"]
    assert oracle["decision"] == "rejected_as_general_or_same_cycle_preservation"
    assert "not an independent TopoAE++ implementation" in oracle["protocol"]["oracle_role"]
    assert set(oracle["registered_arms"]) == {"source_only", "adaptive_mst", "adaptive_full"}
    rows = {(r["dataset"], r["arm"]): r for r in oracle["records"]}
    assert set(rows) == {(dataset, arm) for dataset in ("Twist", "K4", "COIL20-1")
                         for arm in ("source", "graph_init", *oracle["registered_arms"])}
    assert len(rows) == len(oracle["records"])
    timeout = rows["COIL20-1", "adaptive_full"]
    assert timeout["status"] == "failed"
    assert timeout["completed_outer_rounds"] == 14 and timeout["requested_outer_rounds"] == 20
    assert any("COIL20-1" in text and "times out" in text for text in oracle["failures"])
    assert oracle["completed_rounds"] == 174 and oracle["nonconverged_rounds"] == 138

    gate = document["same_cycle_gate"]
    assert gate["dataset"] == "K4" and gate["all_rows"] == 300
    assert 0 < gate["a"] < gate["b"]
    assert gate["source_triangle_count"] == 158561
    assert gate["source_triangle_cochain_violations"] == [0, 0, 0]
    assert gate["decision"] == "reject_all_saved_targets" and gate["paper_failure_claim"] is False
    rows = indexed(gate["records"], "method")
    targets = {"graph_init", "source_only", "adaptive_mst", "adaptive_full", "author_baseline"}
    assert set(rows) == targets | {"source"}, "missing same-cycle outcome"
    assert rows["source"]["eligible_selected_rank"] == 3
    assert rows["source"]["all_three_preserved"] is True
    for method in targets:
        assert rows[method]["all_three_preserved"] is False
        assert rows[method]["eligible_selected_rank"] == 0
        assert rows[method]["independent_reduction_agrees"] is True


def test_public_evidence_schema_privacy_and_required_failures():
    assert_evidence_contract(read_public_json())


@pytest.mark.parametrize("mutation", ["clusters_timeout", "digits_timeout", "log_distance",
                                         "tail", "fashion", "coil_timeout", "author_gate",
                                         "source_only_gate", "mst_gate", "full_gate",
                                         "default_promotion", "independent_relabel"])
def test_evidence_guard_rejects_missing_failures_or_overclaims(mutation):
    value = copy.deepcopy(read_public_json())
    candidates = indexed(value["candidate_outcomes"])
    if mutation in {"clusters_timeout", "digits_timeout"}:
        dataset = "3Clusters" if mutation == "clusters_timeout" else "Digits"
        value["baseline_results"] = [r for r in value["baseline_results"]
                                     if r["id"] != f"raw-{dataset}-TopoAE++-adapter"]
    elif mutation == "log_distance":
        candidates["multiscale_log_distance"]["records"].pop()
    elif mutation == "tail":
        candidates["linear_tail"]["positive_new_seed_fit_ari_pairs"] = 4
    elif mutation == "fashion":
        candidates["linear_tail"]["fashion_transfer"]["decision"] = "universal_improvement"
    elif mutation == "coil_timeout":
        row = next(r for r in candidates["oracle_critical_edges"]["records"]
                   if (r["dataset"], r["arm"]) == ("COIL20-1", "adaptive_full"))
        row["status"] = "completed"
    elif mutation.endswith("_gate"):
        method = {"author_gate": "author_baseline", "source_only_gate": "source_only",
                  "mst_gate": "adaptive_mst", "full_gate": "adaptive_full"}[mutation]
        value["same_cycle_gate"]["records"] = [r for r in value["same_cycle_gate"]["records"]
                                                if r["method"] != method]
    elif mutation == "default_promotion":
        value["scope"]["production_defaults_changed"] = True
    else:
        candidates["oracle_critical_edges"]["protocol"]["oracle_role"] = "independent TopoAE++ implementation"
    with pytest.raises(AssertionError):
        assert_evidence_contract(value)


def test_dataset_columns_and_distinct_umap_protocols_are_explicit():
    value = read_public_json()
    datasets = indexed(value["datasets"])
    assert datasets["3Clusters"]["selected_columns"]["names"] == ["x", "y", "z"]
    assert datasets["3Clusters"]["excluded_columns"] == ["ClusterId"]
    assert datasets["COIL20-1"]["shape"] == [72, 1024]
    assert datasets["COIL20-1"]["excluded_columns"] == ["1025"]
    assert all(row["labels_used_for_fit"] is False for row in datasets.values())
    raw = value["protocols"]["author_raw"]["umap"]
    matched = value["protocols"]["digits_matched"]["umap"]
    assert (raw["n_neighbors"], raw["n_epochs"]) == (15, None)
    assert (matched["n_neighbors"], matched["n_epochs"]) == (16, 300)
    assert matched["force_approximation_algorithm"] is True


def test_figure_hashes_dimensions_and_rendering_metadata_match_public_files():
    figures = indexed(read_public_json()["figures"], "dataset")
    assert set(figures) == {"K4", "Twist", "COIL20-1", "Digits"}
    for dataset, record in figures.items():
        relative = PurePosixPath(record["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert record["path"] == f"assets/visual-contracts-{dataset}.png"
        raw = (ROOT / relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        assert assert_png_metadata_public(raw) == (record["width"], record["height"])
        assert record["png_metadata_stripped"] is True
        assert record["embedded_raw_point_arrays"] is False
        assert len(record["panels"]) == (4 if dataset == "Digits" else 3)
        for panel in record["panels"]:
            for flag in ("full_range", "equal_aspect", "independent_axes", "all_points"):
                assert panel[flag] is True
            assert panel["smoothed"] is False and panel["coordinate_transform"] == "none"
            assert panel["points"] == ({"K4": 300, "Twist": 100, "COIL20-1": 72}.get(dataset)
                                       or {"fit1400": 1400, "transductive1797": 1797}[panel["cohort"]])


@pytest.mark.parametrize("dataset", ["Twist", "K4"])
def test_headerless_parser_keeps_first_row_and_raw_feature_order(driver, dataset):
    text = "1.25,-2,3,999\n4,5.5,-6,998\n"
    actual = driver.parse_author_csv(text, dataset)
    np.testing.assert_array_equal(actual, [[1.25, -2, 3], [4, 5.5, -6]])
    assert actual.dtype == np.dtype("<f8") and actual.flags.c_contiguous


def test_3clusters_parser_selects_header_names_and_excludes_labels(driver):
    # Reordered header prevents an accidental positional-label feature leak.
    text = "z,ClusterId,x,y\n3,999,1,2\n6,888,4,5\n"
    expected = [[1., 2., 3.], [4., 5., 6.]]
    np.testing.assert_array_equal(driver.parse_author_csv(text, "3Clusters"), expected)
    changed_labels = text.replace("999", "-1").replace("888", "0")
    np.testing.assert_array_equal(driver.parse_author_csv(changed_labels, "3Clusters"), expected)


def coil_csv():
    # The excluded column comes FIRST; selection must honor registered names.
    header = ["1025", *map(str, range(1024, 0, -1))]
    values = {str(i): i / 1024 for i in range(1, 1025)}
    rows = [["1", *(str(values[name] + offset) for name in header[1:])]
            for offset in (0, 2)]
    return "\n".join(",".join(row) for row in (header, *rows)) + "\n"


def test_coil_parser_selects_exact_1024_features_not_constant_1025(driver):
    actual = driver.parse_author_csv(coil_csv(), "COIL20-1")
    expected = np.arange(1, 1025, dtype=float) / 1024
    np.testing.assert_array_equal(actual, np.vstack((expected, expected + 2)))
    assert actual.shape == (2, 1024)


@pytest.mark.parametrize("dataset,text", [
    ("K4", ""), ("K4", "1,2\n3,4\n"), ("K4", "x,y,z\n1,2,3\n"),
    ("K4", "1,2,3\n4,5\n"), ("K4", "1,2,3\n\n4,5,6\n"),
    ("K4", '"1,2,3\n'), ("Twist", "1,2,nan\n"),
    ("Twist", "1,inf,3\n"), ("Twist", "-inf,2,3\n"),
    ("Twist", "1,2,1e999\n"), ("Twist", "1,,3\n"),
    ("3Clusters", "ClusterId,x,y,z\n"),
    ("3Clusters", "ClusterId,x,y,y\n0,1,2,3\n"),
    ("3Clusters", "ClusterId,x,y,extra\n0,1,2,3\n"),
    ("3Clusters", "ClusterId,x,y,z\n0,1,2,nan\n"),
    ("Digits", "1,2,3\n"),
])
def test_parser_rejects_malformed_or_nonfinite_inputs(driver, dataset, text):
    with pytest.raises(ValueError):
        driver.parse_author_csv(text, dataset)


@pytest.mark.parametrize("mutation", ["header", "excluded_constant", "nonfinite"])
def test_coil_parser_rejects_unregistered_columns_or_values(driver, mutation):
    lines = coil_csv().splitlines()
    if mutation == "header":
        lines[0] = lines[0].replace("1025", "1026", 1)
    else:
        row = lines[1].split(",")
        row[0 if mutation == "excluded_constant" else 1] = "2" if mutation == "excluded_constant" else "nan"
        lines[1] = ",".join(row)
    with pytest.raises(ValueError):
        driver.parse_author_csv("\n".join(lines), "COIL20-1")


def adapter_manifest(driver):
    return dict(copy.deepcopy(driver.ADAPTER_CONTRACT), executable_sha256="a" * 64)


def test_adapter_contract_is_explicit_fixed_seed_generic_cpu(driver):
    manifest = adapter_manifest(driver)
    assert driver.validate_adapter_manifest(manifest) == manifest
    assert manifest["device"] == "cpu" and manifest["seed"] == 0
    assert manifest["threads"] == manifest["interop_threads"] == 1
    assert manifest["generic_dimension"] is True and manifest["output_dimensions"] == 2
    assert manifest["updates"] == 1000 and manifest["output_mode"] == "train"
    assert manifest["topology_implementation"] == "upstream-unmodified"
    assert driver.CASE_TIMEOUT == 150 and driver.TOTAL_TIMEOUT == 900
    for key in driver.ADAPTER_CONTRACT:
        incomplete = adapter_manifest(driver)
        del incomplete[key]
        with pytest.raises(ValueError, match="contract mismatch"):
            driver.validate_adapter_manifest(incomplete)


@pytest.mark.parametrize("key,value", [
    ("seed", 1), ("seed", False), ("device", "cuda"), ("threads", True),
    ("threads", 2), ("generic_dimension", False), ("output_dimensions", 64),
    ("hidden_dimensions", [128, 64]), ("updates", 427), ("output_mode", "eval"),
    ("full_batch", False), ("source_ph_dtype", "float32"),
    ("topology_implementation", "independent"), ("learning_rate", float("nan")),
    ("executable_sha256", "not-a-hash"), ("executable_sha256", "A" * 64),
])
def test_adapter_manifest_rejects_wrong_or_implicit_contract(driver, key, value):
    manifest = adapter_manifest(driver)
    manifest[key] = value
    with pytest.raises(ValueError):
        driver.validate_adapter_manifest(manifest)


def test_native_input_uses_original_float64_and_exact_dimension_header(driver, tmp_path):
    X = np.array([[1.000000000001, 2., 3.], [4., 5., 6.]], dtype="<f8")
    path = tmp_path / "input.f64"
    driver.write_native_input(path, X)
    raw = path.read_bytes()
    assert raw[:24] == b"TDAF64LE" + struct.pack("<QQ", 2, 3)
    assert raw[24:] == X.tobytes()
    with pytest.raises(FileExistsError):
        driver.write_native_input(path, X)


@pytest.mark.parametrize("X", [np.zeros((1, 3)), np.zeros((2001, 1)), np.zeros((2, 4097)),
                               np.zeros((2, 0)), np.zeros(3), np.full((2, 3), np.nan),
                               np.full((2, 3), np.inf), np.full((2, 3), 1e40)])
def test_native_input_enforces_dimension_and_finiteness_bounds(driver, tmp_path, X):
    path = tmp_path / "invalid.f64"
    with pytest.raises(ValueError):
        driver.write_native_input(path, X)
    assert not path.exists()


def test_native_output_requires_exact_finite_two_dimensional_payload(driver, tmp_path):
    path = tmp_path / "output.f32"
    expected = np.arange(6, dtype="<f4").reshape(3, 2)
    path.write_bytes(expected.tobytes())
    np.testing.assert_array_equal(driver.read_native_output(path, 3), expected)
    for raw in (expected.tobytes()[:-1], expected.tobytes() + b"x",
                expected.astype("<f8").tobytes(), np.full((3, 2), np.nan, dtype="<f4").tobytes()):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            driver.read_native_output(path, 3)


def complete_progress():
    events = [{"kind": "source_ph_start"}, {"kind": "source_ph_ready"}]
    events.extend({"kind": "step", "step": step} for step in range(1000))
    events.append({"kind": "complete", "updates": 1000, "output_mode": "train"})
    return "\n".join("EVENT " + json.dumps(event) for event in events)


def test_author_progress_requires_every_update_and_train_mode_encode(driver):
    text = complete_progress()
    assert driver.validate_protocol(text) == {"updates": 1000, "output_mode": "train"}
    for bad in ("", "\n".join(text.splitlines()[:429]),
                text.replace('"step": 0', '"step": 1', 1),
                text.replace('"output_mode": "train"', '"output_mode": "eval"'),
                text + '\nEVENT {"kind":"step","step":1000}',
                text.replace('"step": 5}', '"step": 5, "total": NaN}', 1),
                text.replace('"step": 0}', '"step": 0, "step": 0}', 1)):
        with pytest.raises(ValueError):
            driver.validate_protocol(bad)


def test_help_is_offline_and_documents_opt_in_adapter_contract(driver, capsys):
    with pytest.raises(SystemExit) as error:
        driver.main(["--help"])
    assert error.value.code == 0
    text = capsys.readouterr().out
    for term in ("--check-only", "--adapter-manifest", "--adapter-sha256", "digits-matched",
                 "--upstream", "--cpu-adapter", "executable_sha256"):
        assert term in text


@pytest.fixture
def synthetic_run(driver, tmp_path, monkeypatch):
    import os

    upstream = tmp_path / "synthetic-upstream"
    data_dir = upstream / "data"
    data_dir.mkdir(parents=True)
    text = b"1,2,3\n4,5,6\n"
    (data_dir / "Twist.csv").write_bytes(text)
    X = np.array([[1., 2., 3.], [4., 5., 6.]], dtype="<f8")
    monkeypatch.setattr(driver, "DATA_SHA256", {"Twist": driver.sha256(text)})
    monkeypatch.setattr(driver, "X_SHA256", {"Twist": driver.sha256(X.tobytes())})
    monkeypatch.setattr(driver, "UPSTREAM_SOURCE_SHA256", {})
    checks = []
    monkeypatch.setattr(driver, "verify_upstream", lambda path: checks.append(path))
    monkeypatch.setattr(driver, "freeze_sources", lambda: ({}, {"synthetic": "fixture"}))
    # The test's own import guard remains active; avoid main's process-global
    # import-hook/preloaded-module policy affecting unrelated tests in the suite.
    monkeypatch.setattr(driver, "install_import_guard", lambda: None)
    for key in driver.THREAD_KEYS:
        monkeypatch.setenv(key, os.environ.get(key, "1"))
    executable = tmp_path / "trusted-fixture-native"
    executable.write_bytes(b"\x7fELFsynthetic test fixture; NEVER EXECUTE")
    executable.chmod(0o700)
    manifest = adapter_manifest(driver)
    manifest["executable_sha256"] = driver.sha256(executable)
    manifest_path = tmp_path / "adapter.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "new-output"
    args = ["--upstream", str(upstream), "--output", str(output), "--dataset", "Twist",
            "--methods", "AUTHOR", "--cpu-adapter", str(executable),
            "--adapter-sha256", manifest["executable_sha256"],
            "--adapter-manifest", str(manifest_path)]
    return dict(args=args, output=output, executable=executable, manifest=manifest_path,
                checks=checks, upstream=upstream, X=X)


def test_check_only_freezes_synthetic_inputs_without_executing_adapter(driver, synthetic_run):
    case = synthetic_run
    assert driver.main(case["args"] + ["--check-only"]) == 0
    registration = json.loads((case["output"] / "registration.json").read_text())
    result = json.loads((case["output"] / "check-result.json").read_text())
    assert result["status"] == "validated-no-fits" and result["adapter_execution_verified"] is False
    assert registration["seed"] == 0 and registration["labels_used"] is False
    assert registration["retries"] == 0 and registration["check_only"] is True
    assert registration["case_timeout_seconds"] == 150
    assert registration["total_child_budget_seconds"] == 900
    assert len(registration["inputs"]) == 1
    item = registration["inputs"][0]
    assert item["fit_shape"] == [2, 3] and item["query_shape"] == [0, 3]
    with np.load(item["input"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["fit"], case["X"])
        np.testing.assert_array_equal(saved["fit_ids"], [0, 1])
    assert case["checks"] == [case["upstream"], case["upstream"]]
    assert not list(case["output"].rglob("coordinates.npz"))
    assert not list(case["output"].rglob("stdout.log"))
    with pytest.raises(ValueError, match="already exists"):
        driver.main(case["args"] + ["--check-only"])


@pytest.mark.parametrize("flag", ["--cpu-adapter", "--adapter-sha256", "--adapter-manifest"])
def test_author_requires_each_explicit_trusted_adapter_argument(driver, synthetic_run, flag):
    case = synthetic_run
    args = case["args"].copy()
    index = args.index(flag)
    del args[index:index + 2]
    with pytest.raises(ValueError, match="AUTHOR requires"):
        driver.main(args + ["--check-only"])
    assert not case["output"].exists()


@pytest.mark.parametrize("change", ["cli", "manifest", "executable"])
def test_adapter_sha_mismatch_fails_before_creating_output_or_launch(driver, synthetic_run, change):
    case = synthetic_run
    args = case["args"].copy()
    if change == "cli":
        args[args.index("--adapter-sha256") + 1] = "0" * 64
    elif change == "manifest":
        manifest = json.loads(case["manifest"].read_text())
        manifest["executable_sha256"] = "0" * 64
        case["manifest"].write_text(json.dumps(manifest))
    else:
        case["executable"].write_bytes(case["executable"].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        driver.main(args + ["--check-only"])
    assert not case["output"].exists()


@pytest.mark.parametrize("timed_out", [False, True])
def test_process_budget_and_single_thread_environment_are_mocked(driver, tmp_path, monkeypatch, timed_out):
    calls = []
    waits = []
    signals = []

    class Child:
        pid = 999999

        def wait(self, timeout=None):
            waits.append(timeout)
            if timed_out and len(waits) == 1:
                raise driver.subprocess.TimeoutExpired("synthetic", timeout)
            return -9 if timed_out else 0

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Child()

    monkeypatch.setattr(driver.subprocess, "Popen", popen)
    monkeypatch.setattr(driver.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    command = ["synthetic-program-never-executed", "input.f64", "output.f32"]
    result = driver.run_bounded(command, tmp_path, 150)
    assert result["timed_out"] is timed_out and result["timeout_seconds"] == 150
    assert calls[0][0] == command and calls[0][1]["start_new_session"] is True
    assert calls[0][1]["cwd"] == tmp_path
    assert all(calls[0][1]["env"][key] == "1" for key in driver.THREAD_KEYS)
    assert waits == ([150, None] if timed_out else [150])
    assert signals and all(pid == Child.pid for pid, _ in signals)


@pytest.mark.parametrize("failure", ["timeout", "partial", "bad_coordinates"])
def test_mocked_author_failures_never_become_completed_outputs(driver, synthetic_run, monkeypatch, failure):
    case = synthetic_run
    calls = []

    def process(command, folder, timeout):
        calls.append((command, timeout))
        # A partial file is deliberately present even on timeout: existence alone
        # cannot turn an interrupted fit into a successful published baseline.
        np.zeros((2, 2), dtype="<f4").tofile(folder / "output.f32")
        text = complete_progress() if failure == "bad_coordinates" else "EVENT {\"kind\":\"source_ph_start\"}"
        (folder / "stdout.log").write_text(text)
        if failure == "bad_coordinates":
            np.full((2, 2), np.nan, dtype="<f4").tofile(folder / "output.f32")
        return dict(returncode=0, timed_out=failure == "timeout", wall_seconds=1., timeout_seconds=timeout)

    monkeypatch.setattr(driver, "run_bounded", process)
    assert driver.main(case["args"]) == 1
    result = json.loads((case["output"] / "results.json").read_text())
    assert result["status"] == "incomplete" and len(result["results"]) == 1
    row = result["results"][0]
    assert row["status"] == ("timeout" if failure == "timeout" else "failed")
    assert row["attempts"] == 1 and "coordinates" not in row
    assert len(calls) == 1 and calls[0][1] == 150
    assert calls[0][0][0] == str(case["executable"])
    assert not list(case["output"].rglob("coordinates.npz"))
