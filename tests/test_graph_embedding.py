"""Synthetic contracts for graph fitting, frozen-layout mapping and safe checkpoints."""
import io
import importlib.util
import json
from pathlib import Path
import random
import stat
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from open_deep_tda import graph_embedding as graph_api
from open_deep_tda.graph_embedding import GraphEmbedding


@pytest.fixture(scope="module")
def api():
    return graph_api


@pytest.fixture(scope="module")
def requires_numba():
    # The package parent may have Torch loaded. Never import Numba here.
    if importlib.util.find_spec("numba") is None:
        pytest.skip(
            "optional Numba is not installed; real graph SGD runs in a subprocess"
        )


def rng_equal(a, b):
    assert a[0] == b[0] and a[2:] == b[2:]
    np.testing.assert_array_equal(a[1], b[1])


def features():
    return np.random.default_rng(782).normal(size=(32, 4))


def knn(X):
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    ids = np.argsort(D, axis=1, kind="stable")[:, :15]
    return ids, np.take_along_axis(D, ids, axis=1)


@pytest.fixture(scope="module")
def fitted(api, tmp_path_factory, requires_numba):
    return api.GraphEmbedding(
        epochs=2, work_dir=tmp_path_factory.mktemp("scratch")
    ).fit(features())


@pytest.fixture
def mapper():
    X = features()
    Z = np.random.default_rng(103).normal(size=(len(X), 2))
    return GraphEmbedding.from_layout(X, Z, source_knn=knn(X))


def test_defaults(api):
    import inspect

    sig = inspect.signature(api.GraphEmbedding)
    expected = dict(
        n_components=2,
        neighbor_backend="exact",
        work_dir=None,
        debug_dir=None,
        max_vertices=60000,
        max_edges=1800000,
        max_sgd_events=4000000000,
        max_feature_entries=16000000,
        fit_timeout=600,
        transform_timeout=900,
        epochs=300,
    )
    for key, value in expected.items():
        assert sig.parameters[key].default == value


@pytest.mark.parametrize(
    "X",
    [
        None,
        1,
        [],
        [1, 2],
        [[1], [1, 2]],
        np.zeros((0, 4)),
        np.zeros((15, 4)),
        np.zeros((16, 0)),
        np.zeros((16, 2, 1)),
        np.full((16, 2), np.nan),
        np.full((16, 2), np.inf),
        np.ones((16, 2), complex),
        np.ones((16, 2), bool),
        np.full((16, 2), "2"),
        np.ones((16, 2), object),
        np.ma.array(np.ones((16, 2)), mask=True),
    ],
)
@pytest.mark.parametrize("factory", [False, True])
def test_input_rejected(api, X, tmp_path, factory):
    with pytest.raises((ValueError, TypeError)):
        if factory:
            GraphEmbedding.from_layout(X, layout())
        else:
            api.GraphEmbedding(epochs=1, work_dir=tmp_path).fit(X)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "kw",
    [
        dict(n_components=3),
        dict(n_components=True),
        dict(seed=-1),
        dict(seed=2**32),
        dict(seed=True),
        dict(epochs=0),
        dict(epochs=True),
        dict(epochs=1.2),
        dict(negative_rate=-1),
        dict(max_vertices=0),
        dict(max_edges=0),
        dict(max_sgd_events=0),
        dict(max_feature_entries=0),
        dict(fit_timeout=0),
        dict(transform_timeout=np.inf),
        dict(neighbor_backend="auto"),
    ],
)
def test_option_validation(api, kw):
    with pytest.raises((ValueError, TypeError)):
        api.GraphEmbedding(**kw)


@pytest.mark.parametrize(
    "limit", [dict(max_vertices=31), dict(max_edges=959), dict(max_feature_entries=127)]
)
def test_parent_budgets(api, limit, monkeypatch, tmp_path):
    def forbidden(*a, **k):
        raise AssertionError("must reject before child")

    monkeypatch.setattr(api.subprocess, "run", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        api.GraphEmbedding(epochs=1, work_dir=tmp_path, **limit).fit(features())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "kind",
    [
        "shape",
        "float",
        "negative",
        "large",
        "self",
        "duplicate",
        "nan",
        "wrong",
        "masked_ids",
        "masked_d",
    ],
)
@pytest.mark.parametrize("factory", [False, True])
def test_bad_source(api, kind, monkeypatch, tmp_path, factory):
    X = features()
    ids, d = knn(X)
    if kind == "shape":
        ids = ids[:, :14]
        d = d[:, :14]
    elif kind == "float":
        ids = ids.astype(float)
    elif kind == "negative":
        ids[0, 0] = -1
    elif kind == "large":
        ids[0, 0] = len(X)
    elif kind == "self":
        ids[0, 0] = 0
    elif kind == "duplicate":
        ids[0, 0] = ids[0, 1]
    elif kind == "nan":
        d[0, 0] = np.nan
    elif kind == "wrong":
        d[0, 0] += 1
    elif kind == "masked_ids":
        ids = np.ma.array(ids, mask=True)
    else:
        d = np.ma.array(d, mask=True)

    def forbidden(*a, **k):
        raise AssertionError("must reject before child")

    monkeypatch.setattr(api.subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        if factory:
            GraphEmbedding.from_layout(X, layout(), source_knn=(ids, d))
        else:
            api.GraphEmbedding(epochs=1, work_dir=tmp_path).fit(X, source_knn=(ids, d))
    assert not list(tmp_path.iterdir())


def test_fit_copies_rng_and_precomputed(api, fitted, tmp_path):
    X = features()[:, ::-1].copy()
    original = X.copy()
    ids, d = knn(X)
    before = (ids.copy(), d.copy())
    ns = np.random.get_state()
    rs = random.getstate()
    model = api.GraphEmbedding(epochs=2, work_dir=tmp_path).fit(
        X, source_knn=(ids[:, ::-1], d[:, ::-1])
    )
    rng_equal(ns, np.random.get_state())
    assert rs == random.getstate()
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(ids, before[0])
    np.testing.assert_array_equal(d, before[1])
    X[:] = 999
    np.testing.assert_array_equal(model.reference_, original)
    for name in ("embedding_", "reference_", "spectral_initial_", "center_"):
        a = getattr(model, name)
        b = a.copy()
        a.flat[0] += 1
        np.testing.assert_array_equal(getattr(model, name), b)
    assert model.artifact_dir_ is None and not list(tmp_path.iterdir())


def test_query_duplicates_batch_rng_no_anchor(mapper, api, monkeypatch):
    Q = np.vstack([features()[:1], np.ones((1, 4)), np.ones((1, 4))])
    before = Q.copy()
    Z = mapper.embedding_
    ns = np.random.get_state()
    rs = random.getstate()
    seen = []
    original = api.minimize

    def solve(*a, **kw):
        seen.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(api, "minimize", solve)
    Y = mapper.transform(Q)
    assert len(seen) == 3  # training IDs are new queries, not anchor overrides
    np.testing.assert_array_equal(Y[1], Y[2])
    np.testing.assert_array_equal(Y[::-1], mapper.transform(Q[::-1]))
    np.testing.assert_array_equal(Y, np.vstack([mapper.transform(q[None]) for q in Q]))
    np.testing.assert_array_equal(Q, before)
    np.testing.assert_array_equal(mapper.embedding_, Z)
    rng_equal(ns, np.random.get_state())
    assert rs == random.getstate()


@pytest.mark.parametrize(
    "Q",
    [
        np.zeros((2, 3)),
        np.zeros(4),
        np.ones((1, 4), complex),
        np.full((1, 4), np.nan),
        np.ma.array(np.ones((1, 4)), mask=True),
    ],
)
def test_bad_query(mapper, Q):
    with pytest.raises(ValueError):
        mapper.transform(Q)


@pytest.mark.parametrize("kind", ["ordinary", "zeros", "saturated"])
def test_fuzzy_weights(api, kind):
    d = {
        "ordinary": np.arange(1, 16)[None, :],
        "zeros": np.zeros((1, 15)),
        "saturated": np.array([[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
    }[kind]
    expected = {"ordinary": np.log2(15), "zeros": 15, "saturated": 5}[kind]
    np.testing.assert_allclose(api._weights(d).sum(1), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("failure", ["exit", "timeout", "interrupt", "missing"])
def test_failure_cleanup_bounded_stderr(api, monkeypatch, tmp_path, failure):
    seen = []

    def fake(cmd, **kw):
        folder = Path(cmd[-1])
        seen.append(folder)
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        assert Path(cmd[1]).name == "_graph_worker.py" and "-m" not in cmd
        assert kw["stderr"] is subprocess.STDOUT
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        if failure == "interrupt":
            raise KeyboardInterrupt()
        kw["stdout"].write(b"x" * 20000 + b"REVIEW_SENTINEL")
        kw["stdout"].flush()
        return subprocess.CompletedProcess(cmd, 17 if failure == "exit" else 0)

    monkeypatch.setattr(api.subprocess, "run", fake)
    model = api.GraphEmbedding(epochs=1, work_dir=tmp_path)
    with pytest.raises((RuntimeError, KeyboardInterrupt, FileNotFoundError)) as error:
        model.fit(features())
    assert seen and not seen[0].exists() and not list(tmp_path.iterdir())
    if failure == "exit":
        assert "REVIEW_SENTINEL" in str(error.value) and len(str(error.value)) < 12000


def test_parallel_unique_scratch(api, monkeypatch, tmp_path):
    from threading import Barrier

    barrier = Barrier(2)
    paths = []

    def fake(cmd, **kw):
        path = Path(cmd[-1])
        paths.append(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        barrier.wait(timeout=5)
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(api.subprocess, "run", fake)

    def one(_):
        with pytest.raises(RuntimeError):
            api.GraphEmbedding(epochs=1, work_dir=tmp_path).fit(features())

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one, range(2)))
    assert len(set(paths)) == 2 and not list(tmp_path.iterdir())


def test_debug_explicit_warning_retains(requires_numba, api, tmp_path):
    with pytest.warns(UserWarning):
        model = api.GraphEmbedding(epochs=1, debug_dir=tmp_path).fit(features())
    assert model.artifact_dir_ is not None
    assert list(tmp_path.rglob("features.npy"))


def test_default_exact_and_ann_routing(requires_numba, api, monkeypatch, tmp_path):
    original = api.build_neighbor_graph
    seen = []

    def route(X, k, **kw):
        seen.append(kw.get("backend", "exact"))
        kw["backend"] = "exact"
        return original(X, k, **kw)

    monkeypatch.setattr(api, "build_neighbor_graph", route)
    for backend in ("exact", "pynndescent"):
        api.GraphEmbedding(epochs=1, neighbor_backend=backend, work_dir=tmp_path).fit(
            features()
        )
    assert seen == ["exact", "pynndescent"]


class ExactIndex:
    def __init__(self, X):
        self.X = X.copy()
        self.bad = False

    def query(self, Q, k):
        assert Q.shape == (1, 4) and k == 15
        D = cdist(Q, self.X)
        ids = np.argsort(D, axis=1, kind="stable")[:, :15]
        d = np.take_along_axis(D, ids, axis=1)
        if self.bad:
            ids[0, 0] = ids[0, 1]
        Q[:] = 999
        return ids[:, ::-1], d[:, ::-1]

    def __reduce__(self):
        raise AssertionError("external index must never be serialized")


def test_save_load_bitwise_external_not_serialized(api, mapper, tmp_path):
    path = tmp_path / "predictor.npz"
    with pytest.raises(ValueError):
        mapper.save(path, include_training_data=False)
    with pytest.warns(UserWarning):
        mapper.save(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = api.GraphEmbedding.load(path)
    Q = features()[:3]
    np.testing.assert_array_equal(loaded.embedding_, mapper.embedding_)
    np.testing.assert_array_equal(loaded.transform(Q), mapper.transform(Q))
    index = ExactIndex(features())
    indexed = api.GraphEmbedding.load(path, query_index=index)
    np.testing.assert_array_equal(indexed.transform(Q), loaded.transform(Q))
    with pytest.warns(UserWarning):
        indexed.save(tmp_path / "indexed.npz")
    index.bad = True
    with pytest.raises((ValueError, RuntimeError)):
        indexed.transform(Q)
    with pytest.raises((FileExistsError, ValueError)):
        mapper.save(path)


def test_import_isolated_no_numba_or_pynndescent():
    code = (
        "from open_deep_tda.graph_embedding import GraphEmbedding; "
        "import sys; assert not any(k in sys.modules "
        "for k in ('numba', 'pynndescent'))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=True)
    return stream.getvalue()


@pytest.mark.parametrize(
    "kind",
    [
        "version",
        "dtype",
        "shape",
        "nan",
        "object",
        "duplicate",
        "unknown",
        "huge_shape",
        "huge_member",
        "bad_npy_version",
        "bad_unit",
        "bad_ids",
        "json_duplicate",
        "no_training",
    ],
)
def test_safe_load_malformed(api, mapper, tmp_path, kind, monkeypatch):
    good = tmp_path / "good.npz"
    with pytest.warns(UserWarning):
        mapper.save(good)
    with zipfile.ZipFile(good) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    if kind in ("version", "json_duplicate", "no_training"):
        metadata = json.loads(
            np.load(io.BytesIO(members["metadata.npy"]), allow_pickle=False)
            .tobytes()
            .decode()
        )
        if kind == "version":
            metadata["version"] = 999
        if kind == "no_training":
            metadata["include_training_data"] = False
        text = json.dumps(metadata)
        if kind == "json_duplicate":
            text = text[:-1] + ',"version":2}'
        members["metadata.npy"] = npy_bytes(
            np.frombuffer(text.encode(), dtype=np.uint8)
        )
    elif kind == "dtype":
        members["X.npy"] = npy_bytes(features().astype(np.float32))
    elif kind == "shape":
        members["Z.npy"] = npy_bytes(np.zeros((32, 3)))
    elif kind == "nan":
        members["Z.npy"] = npy_bytes(np.full((32, 2), np.nan))
    elif kind == "object":
        members["X.npy"] = npy_bytes(np.full((32, 4), object(), dtype=object))
    elif kind == "unknown":
        members["unknown.npy"] = npy_bytes(np.zeros(1))
    elif kind == "bad_unit":
        members["unit.npy"] = npy_bytes(np.array([-1.0]))
    elif kind == "bad_ids":
        members["source_ids.npy"] = npy_bytes(np.zeros((32, 15), dtype=np.int64))
    elif kind == "huge_shape":
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            stream, dict(descr="<f8", fortran_order=False, shape=(2**60, 4))
        )
        members["X.npy"] = stream.getvalue()
    elif kind == "huge_member":
        members["X.npy"] += b"\0" * 1000000
    elif kind == "bad_npy_version":
        members["X.npy"] = b"\x93NUMPY\x09\x00" + members["X.npy"][8:]
    path = tmp_path / (kind + ".npz")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        if kind == "duplicate":
            with pytest.warns(UserWarning):
                archive.writestr("X.npy", members["X.npy"])
    preflight = kind in (
        "dtype",
        "shape",
        "nan",
        "object",
        "duplicate",
        "unknown",
        "huge_shape",
        "huge_member",
        "bad_npy_version",
    )
    if preflight:

        def forbidden(*a, **kw):
            raise AssertionError("malformed header accepted before np.load allocation")

        monkeypatch.setattr(np.lib.format, "read_array", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        api.GraphEmbedding.load(path)


@pytest.mark.parametrize(
    "budget",
    [
        dict(max_file_bytes=1),
        dict(max_uncompressed_bytes=1),
        dict(max_vertices=31),
        dict(max_feature_entries=127),
    ],
)
def test_safe_load_budgets(api, mapper, tmp_path, budget, monkeypatch):
    path = tmp_path / "good.npz"
    with pytest.warns(UserWarning):
        mapper.save(path)

    def forbidden(*a, **kw):
        raise AssertionError("budget must precede array allocation")

    monkeypatch.setattr(np.lib.format, "read_array", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        api.GraphEmbedding.load(path, **budget)


def test_mapper_full_normalizer_gradient(mapper, api, monkeypatch):
    Q = np.array([[0.1, -0.2, 0.3, 0.8]])
    X = mapper.reference_
    anchors = (mapper.embedding_ - mapper.center_) / mapper.distance_unit_
    D = cdist(Q, X)
    ids = np.argsort(D, axis=1, kind="stable")[:, :15]
    p = api._weights(np.take_along_axis(D, ids, axis=1))[0]
    p /= p.sum()

    def oracle(y):
        s = 1 + np.sum((y - anchors) ** 2, axis=1)
        q = (1 / s) / (1 / s).sum()
        return float(np.sum(p * np.log(p / q[ids[0]])))

    seen = []
    original = api.minimize

    def inspect(fg, x0, **kw):
        assert kw["options"] == dict(maxiter=100, maxls=30, ftol=1e-12, gtol=1e-8)
        np.testing.assert_allclose(x0, p @ anchors[ids[0]], atol=1e-15, rtol=0)
        y = x0 + np.array([0.13, -0.07])
        f, g = fg(y)
        eps = 1e-5
        numerical = np.array(
            [(oracle(y + eps * e) - oracle(y - eps * e)) / (2 * eps) for e in np.eye(2)]
        )
        assert abs(f - oracle(y)) < 1e-12
        np.testing.assert_allclose(g, numerical, atol=1e-9, rtol=0)
        seen.append(True)
        return original(fg, x0, **kw)

    monkeypatch.setattr(api, "minimize", inspect)
    mapper.transform(Q)
    assert seen


def test_event_budget_failure_clean(api, tmp_path):
    model = api.GraphEmbedding(epochs=1, max_sgd_events=1, work_dir=tmp_path)
    with pytest.raises(RuntimeError, match="max_sgd_events"):
        model.fit(features())
    assert not list(tmp_path.iterdir()) and model.fit_diagnostics_["status"] == "failed"
    with pytest.raises(RuntimeError):
        _ = model.embedding_


def test_transform_no_disk_empty_and_feature_budget(api, mapper, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("default transform wrote disk")

    monkeypatch.setattr(np, "save", forbidden)
    monkeypatch.setattr(np, "savez_compressed", forbidden)
    Y, diag = mapper.transform(np.empty((0, 4)), return_diagnostics=True)
    assert Y.shape == (0, 2) and diag["query_count"] == 0
    assert mapper.transform(features()[:1]).shape == (1, 2)
    monkeypatch.setitem(mapper._config, "max_feature_entries", 3)
    with pytest.raises(ValueError, match="max_feature_entries"):
        mapper.transform(features()[:1])


def test_transform_timeout_and_nonconvergence(api, mapper, monkeypatch):
    from types import SimpleNamespace

    def not_converged(fg, x0, **kw):
        return SimpleNamespace(
            x=x0, success=False, status=2, message="synthetic", nit=0
        )

    monkeypatch.setattr(api, "minimize", not_converged)
    _, diag = mapper.transform(features()[:2], return_diagnostics=True)
    assert diag["nonconverged"] == 2
    clock = [0.0]
    monkeypatch.setattr(api.time, "perf_counter", lambda: clock[0])

    def timeout(fg, x0, **kw):
        clock[0] = 1000.0
        return not_converged(fg, x0, **kw)

    monkeypatch.setattr(api, "minimize", timeout)
    with pytest.raises(RuntimeError, match="timeout"):
        mapper.transform(features()[:2])
    assert mapper.transform_diagnostics_["completed_queries"] == 0


def test_fit_timeout_actual_and_failed_refit(requires_numba, api, tmp_path):
    model = api.GraphEmbedding(epochs=1, fit_timeout=1e-9, work_dir=tmp_path)
    with pytest.raises(RuntimeError, match="timeout"):
        model.fit(features())
    assert not list(tmp_path.iterdir())
    model = api.GraphEmbedding(epochs=1, work_dir=tmp_path).fit(features())
    with pytest.raises(ValueError):
        model.fit(np.full((32, 4), np.nan))
    with pytest.raises(RuntimeError):
        model.transform(features()[:1])
    assert not list(tmp_path.iterdir())


def test_source_nonself_query_includes_self_idties(api):
    X = np.zeros((20, 4))
    source, d = knn(X)
    ids, dist = api._validate_knn((source, d), X, X, nonself=True)
    assert all(i not in row for i, row in enumerate(ids))
    qi, qd = api._exact_knn(X[:2], X)
    np.testing.assert_array_equal(qi, np.tile(np.arange(15), (2, 1)))
    np.testing.assert_array_equal(qd, np.zeros((2, 15)))


def test_load_never_pickle_and_unavailable_graph(api, mapper, tmp_path, monkeypatch):
    path = tmp_path / "valid.npz"
    with pytest.warns(UserWarning):
        mapper.save(path)
    original = np.lib.format.read_array
    calls = []

    def inspect(*a, **kw):
        assert kw.get("allow_pickle") is False
        calls.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(np.lib.format, "read_array", inspect)
    loaded = api.GraphEmbedding.load(path)
    assert calls
    for name in ("graph_", "spectral_initial_"):
        with pytest.raises(AttributeError):
            getattr(loaded, name)


def test_duplicate_initialization_uses_lowest_retained15(api, tmp_path, monkeypatch):
    X = np.vstack([np.zeros((20, 4)), features()[20:]])
    Z = np.random.default_rng(94).normal(size=(len(X), 2))
    model = GraphEmbedding.from_layout(X, Z, source_knn=knn(X))
    anchors = (model.embedding_ - model.center_) / model.distance_unit_
    seen = []
    original = api.minimize

    def inspect(fg, x0, **kw):
        np.testing.assert_allclose(x0, anchors[:15].mean(0), atol=1e-14, rtol=0)
        s = 1 + np.sum((x0 - anchors) ** 2, axis=1)
        q = (1 / s) / (1 / s).sum()
        assert abs(fg(x0)[0] - np.mean(np.log((1 / 15) / q[:15]))) < 1e-12
        seen.append(True)
        return original(fg, x0, **kw)

    monkeypatch.setattr(api, "minimize", inspect)
    assert np.isfinite(model.transform(np.zeros((1, 4)))).all() and seen


@pytest.mark.parametrize(
    "kind", ["header_length", "truncated_header", "fortran", "zip_budget"]
)
def test_header_and_zip_preallocation(api, mapper, tmp_path, monkeypatch, kind):
    import struct

    good = tmp_path / "base.npz"
    with pytest.warns(UserWarning):
        mapper.save(good)
    with zipfile.ZipFile(good) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if kind == "header_length":
        members["X.npy"] = b"\x93NUMPY\x02\x00" + struct.pack("<I", 2**32 - 1)
    elif kind == "truncated_header":
        members["X.npy"] = b"\x93NUMPY\x01\x00" + struct.pack("<H", 4000) + b"{"
    elif kind == "fortran":
        members["X.npy"] = npy_bytes(np.asfortranarray(features()))
    else:
        members["X.npy"] += b"\0" * 2_000_000
    path = tmp_path / "malformed.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)

    def forbidden(*a, **kw):
        raise AssertionError("allocated malformed model")

    monkeypatch.setattr(np.lib.format, "read_array", forbidden)
    with pytest.raises(ValueError):
        api.GraphEmbedding.load(path, max_uncompressed_bytes=1_000_000)


def test_same_seed_fit_repeat_bitwise(api, fitted, tmp_path):
    model = api.GraphEmbedding(epochs=2, work_dir=tmp_path).fit(features())
    np.testing.assert_array_equal(model.embedding_, fitted.embedding_)
    np.testing.assert_array_equal(model.spectral_initial_, fitted.spectral_initial_)
    for name in ("head", "tail", "weights", "counts"):
        np.testing.assert_array_equal(model.graph_[name], fitted.graph_[name])


def layout():
    return np.random.default_rng(103).normal(size=(32, 2))


def query_neighbors(Q, X):
    distances = cdist(Q, X)
    ids = np.argsort(distances, axis=1, kind="stable")[:, :15]
    return ids, np.take_along_axis(distances, ids, axis=1)


def test_factory_detached_inputs_unit_rng_no_worker(api, monkeypatch, tmp_path):
    X = features()[:, ::-1]
    Z = layout()[:, ::-1]
    ids, d = knn(X)
    originals = [a.copy() for a in (X, Z, ids, d)]
    ns, rs = np.random.get_state(), random.getstate()

    def forbidden(*a, **kw):
        raise AssertionError(
            "precomputed factory must not build neighbors, run a worker or write data"
        )

    monkeypatch.setattr(api.subprocess, "run", forbidden)
    monkeypatch.setattr(api, "build_neighbor_graph", forbidden)
    monkeypatch.setattr(np, "save", forbidden)
    monkeypatch.setattr(np, "savez_compressed", forbidden)
    model = GraphEmbedding.from_layout(X, Z, source_knn=(ids, d), work_dir=tmp_path)
    rng_equal(ns, np.random.get_state())
    assert rs == random.getstate()
    expected_center = Z.mean(axis=0)
    edge_lengths = np.linalg.norm(Z[:, None, :] - Z[ids], axis=2)
    expected_unit = np.median(edge_lengths[edge_lengths > 0])
    assert model.distance_unit_ == expected_unit
    np.testing.assert_array_equal(model.center_, expected_center)
    assert model.diagnostics_["layout_origin"] == "external"
    assert model.diagnostics_["graph_available"] is False
    assert model.diagnostics_["spectral_initial_available"] is False
    assert model.artifact_dir_ is None and not list(tmp_path.iterdir())
    for a, original in zip((X, Z, ids, d), originals):
        np.testing.assert_array_equal(a, original)
        a[:] = 999
    for actual, expected in zip(
        (model.reference_, model.embedding_, *model.source_knn_), originals
    ):
        np.testing.assert_array_equal(actual, expected)
        actual.flat[0] = -999
    for actual, expected in zip(
        (model.reference_, model.embedding_, *model.source_knn_), originals
    ):
        np.testing.assert_array_equal(actual, expected)
    center = model.center_
    center[:] = 999
    np.testing.assert_array_equal(model.center_, expected_center)
    diagnostics = model.diagnostics_
    diagnostics["mapper_options"]["maxiter"] = -1
    assert model.diagnostics_["mapper_options"]["maxiter"] == 100
    for name in ("graph_", "spectral_initial_"):
        with pytest.raises(AttributeError):
            getattr(model, name)


def test_factory_matches_fitted_bitwise(fitted):
    external = GraphEmbedding.from_layout(
        fitted.reference_, fitted.embedding_, source_knn=fitted.source_knn_
    )
    np.testing.assert_array_equal(external.embedding_, fitted.embedding_)
    np.testing.assert_array_equal(external.center_, fitted.center_)
    assert external.distance_unit_ == fitted.distance_unit_
    Q = np.vstack([features()[:2], [[0.3, -0.2, 0.8, 1.1]]])
    np.testing.assert_array_equal(external.transform(Q), fitted.transform(Q))
    assert external.diagnostics_["layout_origin"] == "external"
    assert fitted.diagnostics_["layout_origin"] == "graph_sgd"


@pytest.mark.parametrize(
    "Z",
    [
        None,
        1,
        [],
        [1, 2],
        [[1], [1, 2]],
        np.zeros((31, 2)),
        np.zeros((33, 2)),
        np.zeros((32, 1)),
        np.zeros((32, 3)),
        np.zeros((32, 2, 1)),
        np.zeros((32, 2)),
        np.full((32, 2), np.nan),
        np.full((32, 2), np.inf),
        np.ones((32, 2), complex),
        np.ones((32, 2), bool),
        np.ones((32, 2), object),
        np.full((32, 2), "1"),
        np.ma.array(np.ones((32, 2)), mask=True),
        np.ma.array(layout(), mask=False),
    ],
)
def test_factory_invalid_embedding(Z, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("factory may not invoke a worker")

    monkeypatch.setattr(graph_api.subprocess, "run", forbidden)
    with pytest.raises((ValueError, TypeError)):
        GraphEmbedding.from_layout(features(), Z, source_knn=knn(features()))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_vertices=31),
        dict(max_edges=959),
        dict(max_feature_entries=127),
        dict(query_index=object()),
        dict(n_components=3),
    ],
)
def test_factory_invalid_budgets_and_options(kwargs, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("factory must validate before launching a worker")

    monkeypatch.setattr(graph_api.subprocess, "run", forbidden)
    with pytest.raises((ValueError, TypeError)):
        GraphEmbedding.from_layout(
            features(), layout(), source_knn=knn(features()), **kwargs
        )


def test_factory_exact_backend_matches_precomputed_and_boundary_budgets():
    X, Z = features(), layout()
    model = GraphEmbedding.from_layout(
        X, Z, max_vertices=32, max_edges=960, max_feature_entries=128, max_sgd_events=1
    )
    supplied = GraphEmbedding.from_layout(X, Z, source_knn=knn(X))
    for a, b in zip(model.source_knn_, supplied.source_knn_):
        np.testing.assert_array_equal(a, b)
    assert model.distance_unit_ == supplied.distance_unit_
    np.testing.assert_array_equal(model.transform(X[:2]), supplied.transform(X[:2]))


def test_factory_minimum_sixteen_rows():
    X, Z = features()[:16], layout()[:16]
    model = GraphEmbedding.from_layout(X, Z, source_knn=knn(X))
    assert model.embedding_.shape == (16, 2)
    assert np.isfinite(model.transform(X[:1])).all()
    with pytest.raises(ValueError, match="16"):
        GraphEmbedding.from_layout(X[:15], Z[:15])


def test_factory_checkpoint_origin_and_layout(mapper, tmp_path):
    path = tmp_path / "external.npz"
    with pytest.warns(UserWarning):
        mapper.save(path)
    loaded = GraphEmbedding.load(path)
    assert loaded.diagnostics_["layout_origin"] == "external"
    assert loaded.distance_unit_ == mapper.distance_unit_
    np.testing.assert_array_equal(loaded.center_, mapper.center_)
    np.testing.assert_array_equal(loaded.embedding_, mapper.embedding_)
    np.testing.assert_array_equal(
        loaded.transform(features()[:3]), mapper.transform(features()[:3])
    )
    for name in ("graph_", "spectral_initial_"):
        with pytest.raises(AttributeError):
            getattr(loaded, name)


def test_seen_rows_return_solver_result_not_hidden_anchor(mapper, monkeypatch):
    from types import SimpleNamespace

    target = np.array([0.123, -0.987])
    seen = []

    def solve(fg, x0, **kw):
        seen.append(x0.copy())
        return SimpleNamespace(
            x=target.copy(), success=True, status=0, message="test", nit=0
        )

    monkeypatch.setattr(graph_api, "minimize", solve)
    expected = target * mapper.distance_unit_ + mapper.center_
    Q = features()[:3]
    result = mapper.transform(Q, query_knn=query_neighbors(Q, mapper.reference_))
    assert len(seen) == len(Q)
    np.testing.assert_array_equal(result, np.tile(expected, (len(Q), 1)))
    assert not np.array_equal(result, mapper.embedding_[:3])


def test_query_knn_exact_bitwise_precedence_copies(mapper, monkeypatch):
    Q = np.vstack([features()[:1], np.ones((2, 4))])
    original = Q.copy()
    ids, d = query_neighbors(Q, mapper.reference_)
    before = ids.copy(), d.copy()
    expected = mapper.transform(Q)

    class ForbiddenIndex:
        def query(self, *a, **kw):
            raise AssertionError(
                "explicit query_knn must take precedence over the live index"
            )

    model = GraphEmbedding.from_layout(
        mapper.reference_,
        mapper.embedding_,
        source_knn=mapper.source_knn_,
        query_index=ForbiddenIndex(),
    )

    def forbidden(*a, **kw):
        raise AssertionError("explicit query_knn must bypass live exact search")

    monkeypatch.setattr(graph_api, "_exact_knn", forbidden)
    actual = model.transform(Q, query_knn=(ids[:, ::-1], d[:, ::-1]))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(Q, original)
    np.testing.assert_array_equal(ids, before[0])
    np.testing.assert_array_equal(d, before[1])
    np.testing.assert_array_equal(actual[1], actual[2])
    np.testing.assert_array_equal(
        actual[::-1], model.transform(Q[::-1], query_knn=(ids[::-1], d[::-1]))
    )


@pytest.mark.parametrize(
    "kind",
    [
        "few_rows",
        "extra_rows",
        "few_columns",
        "distance_shape",
        "float_ids",
        "bool_ids",
        "negative_id",
        "large_id",
        "duplicate",
        "negative_distance",
        "nan",
        "inf",
        "wrong_distance",
        "complex_distance",
        "masked_ids",
        "masked_distance",
        "unmasked_ids",
        "unmasked_distance",
        "missing_pair",
        "extra_pair",
        "not_pair",
    ],
)
def test_query_knn_invalid(mapper, kind, monkeypatch):
    Q = features()[:2]
    ids, d = query_neighbors(Q, mapper.reference_)
    if kind == "few_rows":
        ids, d = ids[:1], d[:1]
    elif kind == "extra_rows":
        ids, d = np.tile(ids, (2, 1)), np.tile(d, (2, 1))
    elif kind == "few_columns":
        ids, d = ids[:, :14], d[:, :14]
    elif kind == "distance_shape":
        d = d[:, :14]
    elif kind == "float_ids":
        ids = ids.astype(float)
    elif kind == "bool_ids":
        ids = ids.astype(bool)
    elif kind == "negative_id":
        ids[0, 0] = -1
    elif kind == "large_id":
        ids[0, 0] = len(mapper.reference_)
    elif kind == "duplicate":
        ids[0, 0] = ids[0, 1]
    elif kind == "negative_distance":
        d[0, 0] = -1
    elif kind == "nan":
        d[0, 0] = np.nan
    elif kind == "inf":
        d[0, 0] = np.inf
    elif kind == "wrong_distance":
        d[0, 0] += 0.1
    elif kind == "complex_distance":
        d = d.astype(complex)
    elif kind == "masked_ids":
        ids = np.ma.array(ids, mask=True)
    elif kind == "masked_distance":
        d = np.ma.array(d, mask=True)
    elif kind == "unmasked_ids":
        ids = np.ma.array(ids, mask=False)
    elif kind == "unmasked_distance":
        d = np.ma.array(d, mask=False)
    pair = (ids, d)
    if kind == "missing_pair":
        pair = (ids,)
    elif kind == "extra_pair":
        pair = (ids, d, d)
    elif kind == "not_pair":
        pair = {"ids": ids, "distances": d}

    def forbidden(*a, **kw):
        raise AssertionError("all supplied candidates must be validated before solving")

    monkeypatch.setattr(graph_api, "minimize", forbidden)
    with pytest.raises(ValueError):
        mapper.transform(Q, query_knn=pair)
    assert mapper.transform_diagnostics_["completed_queries"] == 0


def test_query_knn_empty(mapper):
    empty = np.empty((0, 4))
    pair = np.empty((0, 15), dtype=np.int64), np.empty((0, 15))
    assert mapper.transform(empty, query_knn=pair).shape == (0, 2)
    with pytest.raises(ValueError):
        mapper.transform(
            empty, query_knn=(np.empty((0, 14), dtype=int), np.empty((0, 14)))
        )


def test_query_knn_uses_valid_supplied_subset(mapper, monkeypatch):
    Q = features()[:1]
    ids = np.arange(17, 32, dtype=np.int64)[None, :]
    distances = cdist(Q, mapper.reference_[ids[0]])
    order = np.argsort(distances[0], kind="stable")
    retained = ids[0, order]
    p = graph_api._weights(distances[:, order])[0]
    p /= p.sum()
    anchors = (mapper.embedding_ - mapper.center_) / mapper.distance_unit_
    expected = np.einsum("i,ij->j", p, anchors[retained])
    seen = []
    original = graph_api.minimize

    def inspect(fg, x0, **kw):
        np.testing.assert_allclose(x0, expected, atol=1e-14, rtol=0)
        seen.append(True)
        return original(fg, x0, **kw)

    monkeypatch.setattr(graph_api, "minimize", inspect)
    mapper.transform(Q, query_knn=(ids, distances))
    assert seen == [True]


def test_factory_readonly_inputs_are_accepted_and_detached():
    X, Z = features(), layout()
    ids, d = knn(X)
    for a in (X, Z, ids, d):
        a.setflags(write=False)
    model = GraphEmbedding.from_layout(X, Z, source_knn=(ids, d))
    for original, actual in zip(
        (X, Z, ids, d), (model.reference_, model.embedding_, *model.source_knn_)
    ):
        assert not np.shares_memory(original, actual)
        np.testing.assert_array_equal(original, actual)
    assert np.isfinite(model.transform(X[:1])).all()
