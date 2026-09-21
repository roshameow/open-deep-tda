"""Neighbor correctness, explicit budgets, and script-process ANN isolation."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import open_deep_tda.neighbors as module
from open_deep_tda.neighbors import build_neighbor_graph


def dense_neighbors(X, k):
    n = len(X)
    k = min(k, max(0, n - 1))
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def assert_graph(result, X, k, exact=True):
    n = len(X)
    K = min(k, max(0, n - 1))
    neighbors, edges = result["neighbors"], result["edges"]
    assert neighbors.shape == (n, K)
    assert neighbors.dtype == edges.dtype == np.int64
    assert edges.ndim == 2 and edges.shape[1] == 2
    expected_edges = sorted({tuple(sorted((i, int(j))))
                             for i, row in enumerate(neighbors) for j in row})
    np.testing.assert_array_equal(edges, np.array(expected_edges, dtype=np.int64).reshape(-1, 2))
    for i, row in enumerate(neighbors):
        assert i not in row
        assert len(set(row)) == K
        assert np.all((row >= 0) & (row < n))
    if exact:
        np.testing.assert_array_equal(neighbors, dense_neighbors(X, k))
    json.dumps(result["diagnostics"], allow_nan=False)


@pytest.mark.parametrize("n,d,k", [(0, 3, 5), (1, 3, 5), (2, 3, 8), (17, 2, 0),
                                    (17, 2, 1), (17, 2, 4), (17, 2, 50), (211, 7, 15)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_exact_dense_and_tiny(n, d, k, dtype):
    X = (np.random.default_rng(3).normal(size=(n, d)) * 3).astype(dtype)
    original = X.copy()
    result = build_neighbor_graph(X, k, working_memory_mb=0.05)
    assert_graph(result, X, k)
    np.testing.assert_array_equal(X, original)
    assert not result["diagnostics"]["audit"]["performed"]


def test_duplicates_boundary_ties_and_permutations():
    X = np.array([[0, 0]] * 20 + [[1, 0], [-1, 0], [0, 1], [0, -1]] * 8,
                 dtype=np.float64)
    rng = np.random.default_rng(14)
    for k in (1, 7, 21, len(X) - 1):
        for _ in range(3):
            # IDs are current row positions. Tied neighbors need not be
            # equivariant to reordering, but must match dense stable sorting.
            X = X[rng.permutation(len(X))]
            small = build_neighbor_graph(X, k, working_memory_mb=0.02)
            large = build_neighbor_graph(X, k, working_memory_mb=16)
            assert_graph(small, X, k)
            np.testing.assert_array_equal(small["neighbors"], large["neighbors"])


def test_noncontiguous_and_nearly_tied_floats():
    X = np.array([[0., 0.], [1., 0.], [1. + 1e-12, 0.], [-1., 0.], [3., 4.]])
    assert_graph(build_neighbor_graph(X[:, ::-1], 2), X[:, ::-1], 2)
    assert build_neighbor_graph(X, 2)["neighbors"][0].tolist() == [1, 3]


@pytest.mark.parametrize("X", [np.zeros(4), 1, [[1], [1, 2]], np.zeros((3, 0)),
                                [[np.nan]], [[np.inf]], [[-np.inf]], [[1j]],
                                [[True]], [["1"]], np.array([[1]], dtype=object)])
def test_bad_X(X):
    with pytest.raises(ValueError, match="X"):
        build_neighbor_graph(X, 2)


@pytest.mark.parametrize("options", [
    {"k": -1}, {"k": 1.2}, {"k": True}, {"backend": "auto"}, {"backend": []},
    {"working_memory_mb": 0}, {"working_memory_mb": float("inf")},
    {"working_memory_mb": "1"}, {"working_memory_mb": True},
    {"seed": -1}, {"seed": 2**32}, {"seed": 1.5}, {"seed": True},
    {"audit_queries": 0}, {"audit_queries": 1.5}, {"audit_queries": True},
    {"min_recall": -0.1}, {"min_recall": 1.1}, {"min_recall": float("nan")},
    {"timeout_seconds": 0}, {"timeout_seconds": -1}, {"timeout_seconds": float("inf")},
])
def test_bad_options(options):
    with pytest.raises(ValueError):
        build_neighbor_graph(np.zeros((4, 2)), **dict({"k": 2}, **options))


def test_overflowed_distances_fail():
    with pytest.raises(ValueError, match="distances overflowed"):
        build_neighbor_graph(np.array([[-1e308], [1e308]]), 1)


def test_workspace_reuses_buffer_and_never_allocates_square(monkeypatch):
    X = np.random.default_rng(1).normal(size=(101, 4))
    seen, buffers = [], []
    original = module.cdist

    def spy(A, B, **kwargs):
        out = kwargs["out"]
        seen.append(out.shape)
        buffers.append(out.__array_interface__["data"][0])
        assert A.shape[0] < len(X)
        assert B.shape == X.shape
        return original(A, B, **kwargs)

    monkeypatch.setattr(module, "cdist", spy)
    result = build_neighbor_graph(X, 7, working_memory_mb=0.02)
    memory = result["diagnostics"]["memory"]
    assert len(seen) > 1 and len(set(buffers)) == 1
    assert max(a * b * 8 for a, b in seen) <= memory["distance_block_bytes"]
    assert memory["estimated_workspace_bytes"] <= memory["budget_bytes"]
    assert_graph(result, X, 7)
    seen.clear()
    build_neighbor_graph(X, 7, working_memory_mb=64)
    assert max(a for a, b in seen) == len(X) - 1


def test_60k_memory_plan_without_running_quadratic_search():
    plan = module._workspace(60000, 15, 64 * 1024**2, "exact")
    assert plan["distance_block_rows"] == 131
    assert plan["estimated_workspace_bytes"] == 66725056
    assert plan["estimated_workspace_bytes"] <= plan["budget_bytes"]
    # One-row minimum is enforced, not silently rounded up past the budget.
    required = 72 * 60000 + 64 * 15 + 4096
    assert module._workspace(60000, 15, required, "exact")["distance_block_rows"] == 1
    with pytest.raises(ValueError, match="budget"):
        module._workspace(60000, 15, required - 1, "exact")


@pytest.mark.parametrize("backend", ["exact", "pynndescent"])
def test_insufficient_budget_fails_before_work(monkeypatch, backend):
    def forbidden(*args, **kwargs):
        raise AssertionError("budget failure should precede distance computation/subprocess")
    monkeypatch.setattr(module, "cdist", forbidden)
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="one full-population distance row"):
        build_neighbor_graph(np.zeros((100, 5)), 5, backend=backend, working_memory_mb=.001)


def fake_worker(monkeypatch, array):
    paths = []

    def run(command, **kwargs):
        paths.append(Path(command[2]).parent)
        assert command[0] == sys.executable
        assert Path(command[1]).name == "_neighbor_worker.py"
        assert "-m" not in command
        assert kwargs["env"].get("PYTHONPATH") == os.environ.get("PYTHONPATH")
        assert kwargs["env"]["NUMBA_NUM_THREADS"] == "1"
        assert kwargs["timeout"] > 0
        assert kwargs["check"] is False
        if callable(array):
            array(command)
        else:
            np.savez(command[3], neighbors=array)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    return paths


def test_mock_ann_global_audit_and_environment(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", "") + ":/dependency/target")
    X = np.random.default_rng(4).normal(size=(101, 4))
    truth = dense_neighbors(X, 5)
    paths = fake_worker(monkeypatch, truth)
    imported_before = "pynndescent" in sys.modules
    audit_shapes = []
    original_cdist = module.cdist

    def audit_cdist(A, B, **kwargs):
        audit_shapes.append((len(A), len(B)))
        return original_cdist(A, B, **kwargs)

    monkeypatch.setattr(module, "cdist", audit_cdist)
    result = build_neighbor_graph(X, 5, backend="pynndescent", audit_queries=17, seed=29,
                                  working_memory_mb=.02, min_recall=1)
    assert_graph(result, X, 5)
    audit = result["diagnostics"]["audit"]
    assert audit["population_size"] == 101 and audit["query_count"] == 17
    assert audit_shapes == [(1, 101)] * 17  # not an audit of a 17-point subcloud
    assert audit["strict_id_recall"] == audit["tie_aware_distance_recall"] == 1
    assert audit["query_indices"] == sorted(np.random.default_rng(29).choice(101, 17, False))
    assert all(not path.exists() for path in paths)
    assert ("pynndescent" in sys.modules) == imported_before


def test_ann_rejects_low_recall_without_fallback(monkeypatch):
    X = np.arange(100, dtype=float).reshape(-1, 1)
    wrong = ((np.arange(100) + 50) % 100).reshape(-1, 1)
    paths = fake_worker(monkeypatch, wrong)
    monkeypatch.setattr(module, "_exact", lambda *a: pytest.fail("silent exact fallback"))
    with pytest.raises(RuntimeError, match="recall audit rejected") as caught:
        build_neighbor_graph(X, 1, backend="pynndescent", audit_queries=64, min_recall=.9)
    audit = caught.value.diagnostics
    assert audit["query_count"] == 64 and audit["population_size"] == 100
    assert audit["strict_id_recall"] == audit["tie_aware_distance_recall"] == 0
    assert not audit["accepted"]
    assert all(not path.exists() for path in paths)


def test_ann_duplicate_tie_recall_not_strict_id_recall(monkeypatch):
    X = np.zeros((20, 3))
    alternative = np.array([sorted([(i + 9) % 20, (i + 10) % 20]) for i in range(20)])
    fake_worker(monkeypatch, alternative)
    result = build_neighbor_graph(X, 2, backend="pynndescent", min_recall=1)
    audit = result["diagnostics"]["audit"]
    assert audit["strict_id_recall"] < .2
    assert audit["tie_aware_distance_recall"] == 1
    assert audit["acceptance_metric"] == "tie_aware_distance_recall"


def test_tie_audit_does_not_forgive_missing_strictly_closer_ids(monkeypatch):
    X = np.array([[0.], [.1], [1.], [-1.]])
    proposal = dense_neighbors(X, 2)
    proposal[0] = [2, 3]  # both at kth radius, but mandatory closer ID 1 missing
    fake_worker(monkeypatch, proposal)
    with pytest.raises(RuntimeError) as caught:
        build_neighbor_graph(X, 2, backend="pynndescent", min_recall=1)
    assert caught.value.diagnostics["tie_aware_distance_recall"] == .875


@pytest.mark.parametrize("malformed", [
    np.zeros((4, 1), dtype=np.int64), np.zeros((4, 2), dtype=float),
    np.zeros((4, 2), dtype=object), np.array([[1, 2], [0, 2], [0, 1], [0, 3]]),
    np.array([[1, 1], [0, 2], [0, 1], [0, 1]]),
    np.array([[1, -1], [0, 2], [0, 1], [0, 1]]),
    np.array([[1, 4], [0, 2], [0, 1], [0, 1]]),
    np.full((4, 2), np.iinfo(np.uint64).max, dtype=np.uint64),
])
def test_malformed_ann_outputs(monkeypatch, malformed):
    paths = fake_worker(monkeypatch, malformed)
    with pytest.raises(RuntimeError, match="Malformed ANN worker output"):
        build_neighbor_graph(np.arange(4.).reshape(-1, 1), 2, backend="pynndescent")
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize("kind", ["missing", "garbage", "extra_member", "oversize", "huge_header"])
def test_ann_file_protocol_and_preallocation_budget(monkeypatch, kind):
    def write(command):
        path = command[3]
        if kind == "missing":
            return
        if kind == "garbage":
            Path(path).write_bytes(b"not npz")
        elif kind == "extra_member":
            np.savez(path, neighbors=np.ones((4, 2), dtype=np.int64), unexpected=np.zeros(1))
        elif kind == "oversize":
            np.savez(path, neighbors=np.zeros((1000, 2), dtype=np.int64))
        else:
            stream = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                stream, {"descr": "<i8", "fortran_order": False, "shape": (10**12, 2)})
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("neighbors.npy", stream.getvalue())
    fake_worker(monkeypatch, write)
    with pytest.raises(RuntimeError, match="Malformed ANN worker output"):
        build_neighbor_graph(np.zeros((4, 2)), 2, backend="pynndescent")


def test_ann_subprocess_timeout_and_cleanup(monkeypatch):
    paths = []

    def run(command, **kwargs):
        assert kwargs["timeout"] == .01
        paths.append(Path(command[2]).parent)
        raise subprocess.TimeoutExpired(command, .01)

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="timed out"):
        build_neighbor_graph(np.zeros((4, 2)), 2, backend="pynndescent", timeout_seconds=.01)
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize("exit_code,message", [(1, "pynndescent dependency unavailable"),
                                               (-11, "native crash")])
def test_ann_dependency_failure_and_native_crash(monkeypatch, exit_code, message):
    def run(command, **kwargs):
        kwargs["stdout"].write(message.encode())
        return SimpleNamespace(returncode=exit_code)
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=message):
        build_neighbor_graph(np.zeros((4, 2)), 2, backend="pynndescent")


@pytest.mark.parametrize("n,k", [(0, 5), (1, 5), (8, 0)])
def test_trivial_ann_needs_no_dependency_or_worker(monkeypatch, n, k):
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("not needed"))
    X = np.zeros((n, 3))
    result = build_neighbor_graph(X, k, backend="pynndescent")
    assert_graph(result, X, k)
    assert not result["diagnostics"]["ann_subprocess"]


def test_actual_script_bypasses_package_and_torch(tmp_path):
    # A stub ANN package asserts the isolation and fixed options INSIDE the
    # spawned interpreter. This test runs even when pynndescent is unavailable.
    (tmp_path / "pynndescent.py").write_text('''
import sys
import numpy as np
assert "torch" not in sys.modules
assert "open_deep_tda" not in sys.modules
class NNDescent:
    def __init__(self, X, **kw):
        assert kw["n_jobs"] == 1 and kw["random_state"] == 13
        assert kw["low_memory"] is True
        assert kw["n_neighbors"] == 5
        assert X.dtype == np.float32
        # Deliberately reverse IDs and make library distances uninformative;
        # the real worker must rerank by original float64 distances, then IDs.
        self.neighbor_graph = (np.tile(np.arange(4, -1, -1), (5, 1)), np.zeros((5, 5)))
''')
    X = np.array([[0.], [1. + 1e-12], [1.], [-1.], [2.]])
    np.save(tmp_path / "input.npy", X)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    worker = Path(module.__file__).with_name("_neighbor_worker.py")
    completed = subprocess.run([sys.executable, str(worker), str(tmp_path / "input.npy"),
                                str(tmp_path / "output.npz"), "2", "13"],
                               env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(module._read_worker_output(tmp_path / "output.npz", 5, 2),
                                  dense_neighbors(X, 2))


def test_actual_script_missing_package_message(tmp_path):
    (tmp_path / "pynndescent.py").write_text('raise ImportError("dependency absent")\n')
    np.save(tmp_path / "input.npy", np.zeros((4, 2)))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    worker = Path(module.__file__).with_name("_neighbor_worker.py")
    completed = subprocess.run([sys.executable, str(worker), str(tmp_path / "input.npy"),
                                str(tmp_path / "output.npz"), "2", "13"],
                               env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert "pynndescent ANN dependency is unavailable" in completed.stderr


@pytest.mark.skipif(importlib.util.find_spec("pynndescent") is None,
                    reason="optional pynndescent package unavailable; no parent import")
def test_actual_ann_if_available():
    X = np.random.default_rng(12).normal(size=(128, 5))
    first = build_neighbor_graph(X, 5, backend="pynndescent", seed=3, min_recall=.9)
    second = build_neighbor_graph(X, 5, backend="pynndescent", seed=3, min_recall=.9)
    assert_graph(first, X, 5, exact=False)
    np.testing.assert_array_equal(first["neighbors"], second["neighbors"])
    assert first["diagnostics"]["audit"]["query_count"] == 64
    assert first["diagnostics"]["audit"]["tie_aware_distance_recall"] >= .9
    # Tiny graphs still use the requested isolated backend, with width <= N.
    for X in (np.zeros((2, 3)), np.zeros((3, 2)), np.zeros((20, 2))):
        result = build_neighbor_graph(X, 7, backend="pynndescent", min_recall=1)
        assert_graph(result, X, 7, exact=len(X) <= 3)
        assert result["diagnostics"]["audit"]["tie_aware_distance_recall"] == 1
