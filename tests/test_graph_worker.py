"""Analytic geometry and isolated execution tests for the graph worker.

Only named function ASTs are executed in the parent: importing the worker in
this process would violate its Torch guard and defeat dependency isolation.
"""
import ast
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist

from open_deep_tda import graph_embedding, _graph_affinity
from open_deep_tda.graph_embedding import GraphEmbedding


WORKER = Path(graph_embedding.__file__).with_name("_graph_worker.py")


def extract_function(name):
    node = next(
        node
        for node in ast.parse(WORKER.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = dict(
        np=np,
        sparse=sparse,
        connected_components=connected_components,
        eigsh=eigsh,
        cdist=cdist,
        _affinity_module=_graph_affinity,
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(WORKER), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.fixture(scope="module")
def initialize():
    return extract_function("initialize")


@pytest.fixture(scope="module")
def requires_numba():
    if importlib.util.find_spec("numba") is None:
        pytest.skip(
            "optional Numba is not installed; real graph SGD runs in a subprocess"
        )


def component_fixture(kind):
    sizes = {
        "connected": [7],
        "disconnected": [5, 4],
        "tiny": [1, 2, 3],
        "isolated": [1, 1, 1, 1],
        "coincident": [2, 2, 2],
    }[kind]
    blocks = [sparse.csr_matrix(np.ones((n, n)) - np.eye(n)) for n in sizes]
    X = np.random.default_rng(44).normal(size=(sum(sizes), 3))
    if kind == "coincident":
        X = np.tile([[-1.0, 0, 0], [1.0, 0, 0]], (3, 1))
    return X, sparse.block_diag(blocks, format="csr"), sizes


@pytest.mark.parametrize(
    "kind", ["connected", "disconnected", "tiny", "isolated", "coincident"]
)
def test_component_geometry_and_rng(initialize, kind):
    X, G, sizes = component_fixture(kind)
    before = X.copy()
    gb = G.copy()
    ns, rs = np.random.get_state(), random.getstate()
    Z, diag = initialize(X, G, seed=31)
    np.testing.assert_array_equal(Z, initialize(X, G, seed=31)[0])
    np.testing.assert_array_equal(X, before)
    for name in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(G, name), getattr(gb, name))
    after = np.random.get_state()
    assert ns[0] == after[0] and ns[2:] == after[2:]
    np.testing.assert_array_equal(ns[1], after[1])
    assert rs == random.getstate()
    assert Z.shape == (len(X), 2) and Z.flags.c_contiguous and np.isfinite(Z).all()
    assert diag["component_sizes"] == sizes and diag["components"] == len(sizes)
    clean = Z - np.random.default_rng(31).normal(0, 1e-4, Z.shape)
    if kind == "connected":
        assert diag["center_rule"] == "connected original"
        assert np.max(abs(clean)) == 10
        np.testing.assert_allclose(clean.mean(0), 0, atol=1e-14)
    else:
        # Complete components have constant degree, hence centered nontrivial
        # Laplacian eigenvectors. Their means recover component centers.
        centers = np.array(
            [part.mean(0) for part in np.split(clean, np.cumsum(sizes)[:-1])]
        )
        np.testing.assert_allclose(centers.mean(0), 0, atol=1e-14)
        np.testing.assert_allclose(np.max(abs(centers)), 10, atol=1e-14)
        spacing = cdist(centers, centers)
        np.fill_diagonal(spacing, np.inf)
        scale = spacing.min() / 4
        for part, center, size in zip(
            np.split(clean, np.cumsum(sizes)[:-1]), centers, sizes
        ):
            if size == 1:
                np.testing.assert_allclose(part[0], center, atol=1e-14)
            else:
                np.testing.assert_allclose(
                    np.max(abs(part - center)), scale, atol=1e-14
                )
            if size == 2:
                np.testing.assert_allclose(
                    part - center, [[-scale, 0], [scale, 0]], atol=1e-14
                )
    if kind == "coincident":
        assert diag["center_rule"] == "explicit coincidentcentroid IDcircle"
        angle = np.arange(3) * 2 * np.pi / 3
        expected = 10 * np.c_[np.cos(angle), np.sin(angle)]
        np.testing.assert_allclose(
            clean.reshape(3, 2, 2).mean(1), expected, atol=2e-15, rtol=0
        )
    elif kind != "connected":
        assert diag["center_rule"] == "sourcecentroidPCA"


def test_tiny_components_analytic(initialize):
    X = np.array([[0.0], [8.0], [8.0]])
    G = sparse.block_diag(
        [sparse.csr_matrix((1, 1)), [[0.0, 1.0], [1.0, 0.0]]], format="csr"
    )
    Z, diag = initialize(X, G, seed=7)
    clean = Z - np.random.default_rng(7).normal(0, 1e-4, Z.shape)
    # SVD sign is arbitrary; component separation and local pair offsets are not.
    np.testing.assert_allclose(abs(clean[0, 0]), 10, atol=1e-14)
    np.testing.assert_allclose(clean[:, 1], 0, atol=1e-14)
    np.testing.assert_allclose(clean[2] - clean[1], [10, 0], atol=1e-14)
    np.testing.assert_allclose(clean[1:].mean(0), -clean[0], atol=1e-14)
    assert diag["component_sizes"] == [1, 2]


def test_initializer_seed_only_jitters_isolated_vertices(initialize):
    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    G = sparse.csr_matrix((3, 3))
    a, _ = initialize(X, G, seed=11)
    b, _ = initialize(X, G, seed=12)
    expected = np.random.default_rng(11).normal(
        0, 1e-4, a.shape
    ) - np.random.default_rng(12).normal(0, 1e-4, b.shape)
    np.testing.assert_allclose(a - b, expected, atol=2e-15, rtol=0)


def test_worker_knn_duplicate_ties_nonself():
    knn = extract_function("knn")
    X = np.zeros((20, 3))
    ids, d = knn(X, X, 15, np.arange(len(X)))
    for i, row in enumerate(ids):
        np.testing.assert_array_equal(row, np.delete(np.arange(20), i)[:15])
    np.testing.assert_array_equal(d, np.zeros_like(d))
    query_ids, _ = knn(X[:2], X, 15)
    np.testing.assert_array_equal(query_ids, np.tile(np.arange(15), (2, 1)))


@pytest.mark.parametrize("kind", ["ordinary", "zeros", "saturated"])
def test_worker_fuzzy_weights(kind):
    d = {
        "ordinary": np.arange(1, 16)[None, :],
        "zeros": np.zeros((1, 15)),
        "saturated": np.array([[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
    }[kind]
    expected = {"ordinary": np.log2(15), "zeros": 15, "saturated": 5}[kind]
    w = extract_function("weights")(d)
    assert np.isfinite(w).all() and np.all((w >= 0) & (w <= 1))
    np.testing.assert_allclose(w.sum(1), expected, atol=1e-12, rtol=0)


def test_worker_torch_guard_isolated():
    code = (
        'import runpy, sys; sys.modules["torch"] = object(); '
        'runpy.run_path(sys.argv[1], run_name="guard_test")'
    )
    run = subprocess.run(
        [sys.executable, "-B", "-c", code, str(WORKER)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode != 0
    assert "Torch preloaded in graph worker" in run.stderr


def test_real_disconnected_graph_and_sgd(requires_numba, initialize, tmp_path):
    X = np.vstack([np.zeros((16, 3)), np.full((16, 3), 100.0)])
    model = GraphEmbedding(epochs=2, work_dir=tmp_path).fit(X)
    g = model.graph_
    G = sparse.csr_matrix((g["weights"], (g["head"], g["tail"])), shape=(32, 32))
    expected, diag = initialize(X, G)
    assert diag["components"] == 2
    np.testing.assert_array_equal(model.spectral_initial_, expected)
    np.testing.assert_array_equal(G.toarray(), G.toarray().T)
    np.testing.assert_array_equal(G.diagonal(), np.zeros(32))
    assert G.nnz == 2 * 16 * 15
    np.testing.assert_array_equal(g["weights"], np.ones(G.nnz))
    np.testing.assert_array_equal(g["counts"], np.full(G.nnz, 2))
    assert np.isfinite(model.embedding_).all()
    assert not np.array_equal(model.embedding_, expected)
    status = model.fit_diagnostics_
    assert status["torch_absent"] is True and status["threads"] == 1
    assert status["positive_updates"] == 2 * G.nnz
    assert status["sgd_events"] == status["sgd_event_upper_bound"] == 2 * G.nnz * 7
    assert status["negative_updates"] <= status["positive_updates"] * 5
    assert not list(tmp_path.iterdir())
    assert "numba" not in sys.modules and "pynndescent" not in sys.modules
    # Copies, including graph arrays and nested diagnostics, are detached.
    g["weights"][:] = -1
    assert np.all(model.graph_["weights"] == 1)
    status["component_sizes"][0] = -1
    assert model.fit_diagnostics_["component_sizes"] == [16, 16]
    json.dumps(model.diagnostics_, allow_nan=False)


def test_connected_spectral_coordinates_match_dense_laplacian(initialize):
    # Irregular weighted path has distinct low eigenvalues, avoiding ambiguity
    # within repeated eigenspaces. This oracle uses a full dense eigensolve.
    weights = np.array([0.3, 0.8, 0.5, 1.0, 0.7])
    G = sparse.diags([weights, weights], [-1, 1], shape=(6, 6), format="csr")
    degree = np.asarray(G.sum(axis=1)).ravel()
    inv = 1 / np.sqrt(degree)
    L = np.eye(6) - inv[:, None] * G.toarray() * inv[None, :]
    values, vectors = np.linalg.eigh(L)
    expected = vectors[:, 1:3].copy()
    for axis in range(2):
        if expected[np.argmax(abs(expected[:, axis])), axis] < 0:
            expected[:, axis] *= -1
    expected *= 10 / np.max(abs(expected))
    Z, diag = initialize(np.arange(12.0).reshape(6, 2), G, seed=9)
    clean = Z - np.random.default_rng(9).normal(0, 1e-4, Z.shape)
    np.testing.assert_allclose(clean, expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(L @ clean, clean * values[1:3], atol=1e-8, rtol=0)
    assert diag["components"] == 1
