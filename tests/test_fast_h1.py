"""Serialization parity (not a different PH algorithm) and reproducible timing.

Run benchmark: PYTHONPATH=/tmp/open-deep-tda-oracle:python python tests/test_fast_h1.py
"""

import copy

import numpy as np
import pytest
import torch

from open_deep_tda import losses, topology


@pytest.fixture
def native():
    return pytest.importorskip("open_deep_tda._core")


def square():
    return np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


def filtered(full):
    return dict(full, pairs=[p for p in full["pairs"] if p["dimension"] == 1
                            and (p["death"] > p["birth"] or p["essential"] or p["censored"])])


@pytest.mark.parametrize("n", [0, 1, 3, 4, 8, 32, 64, 128])
@pytest.mark.parametrize("radius", [float("inf"), 1.1])
def test_exact_result_schema_stats_order_and_critical_ids(native, n, radius):
    X = np.random.default_rng(17).normal(size=(n, 8))
    D = topology.distance_matrix(X)
    full = topology.persistence(D, max_radius=radius)
    fast = topology.h1_persistence(D, max_radius=radius)
    assert fast == filtered(full)
    assert fast == native.persistence_h1(D, max_radius=radius)
    # Full API still includes H0 (including essential/censored) and zero H1.
    assert len(full["pairs"]) >= n
    if not full["truncated"]:
        np.testing.assert_array_equal(topology.diagram(full), topology.diagram(fast))
        assert full["n_simplices"] == n + n * (n - 1) // 2 + n * (n - 1) * (n - 2) // 6


@pytest.mark.parametrize("X", [square(), np.zeros((8, 2)),
                              np.array([[0.], [1.], [2.], [3.]])])
def test_ties_duplicates_zero_bars_and_positive_endpoints(native, X):
    D = topology.distance_matrix(X)
    full = topology.persistence(D)
    fast = topology.h1_persistence(D)
    assert fast == filtered(full) == topology.h1_persistence(D)
    assert any(p["dimension"] == 1 and p["birth"] == p["death"] for p in full["pairs"])
    assert all(p["death"] > p["birth"] for p in fast["pairs"])
    for p in fast["pairs"]:
        assert D[tuple(p["birth_edge"])] == p["birth"]
        assert D[tuple(p["death_edge"])] == p["death"]


@pytest.mark.parametrize("radius", [0., 1., 1.1, 2.])
def test_censored_h1_retained_and_empty_truncation_rejected(native, radius):
    D = topology.distance_matrix(square())
    full = topology.persistence(D, max_radius=radius)
    fast = topology.h1_persistence(D, max_radius=radius)
    assert fast == filtered(full)
    if radius < 2:
        assert fast["truncated"]
        if radius >= 1:
            assert len(fast["pairs"]) == 1 and fast["pairs"][0]["censored"]
            assert np.isinf(fast["pairs"][0]["death"])
            with pytest.raises(ValueError, match="Censored"):
                topology.diagram(fast)
        else:
            assert fast["pairs"] == []  # dropped censored H0 cannot hide truncation
        S = torch.tensor(D)
        with pytest.raises(ValueError, match="truncated/censored"):
            losses.h1_loss(S, S, source_result=fast)


@pytest.mark.parametrize("key", ["max_simplices", "max_reduction_entries", "max_reduction_operations"])
def test_budget_exhaustion_identical_even_when_no_positive_h1(native, key):
    D = np.ones((6, 6)) - np.eye(6)
    for function in [native.persistence, native.persistence_h1,
                     topology.persistence, topology.h1_persistence]:
        with pytest.raises(RuntimeError):
            function(D, **{key: 0})
    full = topology.persistence(D)
    measured = {"max_simplices": "n_simplices", "max_reduction_entries": "peak_reduction_entries",
                "max_reduction_operations": "reduction_operations"}[key]
    limit = full[measured]
    assert topology.h1_persistence(D, **{key: limit}) == filtered(full)
    for function in [topology.persistence, topology.h1_persistence]:
        with pytest.raises(RuntimeError):
            function(D, **{key: limit - 1})


@pytest.mark.parametrize("options", [{"max_simplices": -1}, {"max_simplices": True},
    {"max_simplices": 2**100}, {"max_reduction_entries": 1.5},
    {"max_reduction_operations": np.inf}, {"max_radius": -1}, {"max_radius": np.nan}])
def test_invalid_budgets(native, options):
    for function in [native.persistence, native.persistence_h1,
                     topology.persistence, topology.h1_persistence]:
        with pytest.raises(ValueError):
            function(np.zeros((1, 1)), **options)


def test_wrapper_validation_and_lazy_backend(native, monkeypatch):
    for D in [np.ones((2, 3)), [[1]], [[0, -1], [-1, 0]], [[0, 1], [2, 0]],
              [[np.nan]], [[np.inf]], [[0j]]]:
        with pytest.raises(ValueError):
            topology.h1_persistence(D)
    with pytest.raises(ValueError, match="Unknown"):
        topology.h1_persistence([[0]], unknown=1)
    with pytest.raises(ValueError):
        native.persistence_h1(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        native.persistence_h1(np.zeros((4, 4))[::2, ::2])
    original = topology.importlib.import_module
    def unavailable(name, *args, **kwargs):
        if name == "open_deep_tda._core":
            raise ImportError("not built")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(topology.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="compiled.*_core"):
        topology.h1_persistence([[0]])


@pytest.mark.parametrize("n,scale", [(0, 1.), (3, 1.), (4, 1.1), (4, 4.), (32, 1.1), (64, 1.1), (128, 1.1)])
def test_losses_and_coordinate_gradients_exact_full_vs_fast(native, monkeypatch, n, scale):
    rng = np.random.default_rng(17)
    X = square() if n == 4 else rng.normal(size=(n, 3))
    S = losses.pairwise_distances(torch.tensor(X, requires_grad=True))
    S.retain_grad()
    Y = scale * X + rng.normal(scale=.03, size=X.shape)
    full_cache = topology.persistence(S.detach().numpy())
    fast_cache = topology.h1_persistence(S.detach().numpy())
    fast_api = topology.h1_persistence
    def evaluate(api, cache):
        monkeypatch.setattr(topology, "h1_persistence", api)
        target = torch.tensor(Y, requires_grad=True)
        result = losses.h1_loss(S, losses.pairwise_distances(target), source_result=cache)
        grads = tuple(torch.autograd.grad(result[key], target, retain_graph=True)[0]
                      for key in ("pd1", "crit1"))
        return result, grads
    expected, expected_grads = evaluate(topology.persistence, full_cache)
    for cache in [None, full_cache, fast_cache]:
        actual, grads = evaluate(fast_api, cache)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert actual[key] == expected[key]
        for a, b in zip(grads, expected_grads):
            assert torch.equal(a, b)
    assert S.grad is None


def test_h1_loss_dispatch_cache_and_failclosed(native, monkeypatch):
    S = torch.tensor(topology.distance_matrix(square()))
    original = topology.h1_persistence
    calls = []
    def counted(D, **options):
        calls.append(options)
        return original(D, **options)
    def forbidden(*args, **kwargs):
        pytest.fail("h1_loss must not serialize full persistence")
    monkeypatch.setattr(topology, "h1_persistence", counted)
    monkeypatch.setattr(topology, "persistence", forbidden)
    losses.h1_loss(S, S)
    assert len(calls) == 2 and calls[0] == topology._BUDGET_DEFAULTS
    cache = original(S.numpy())
    losses.h1_loss(S, S, source_result=cache)
    assert len(calls) == 3
    for mutation in [dict(essential=True), dict(censored=True), dict(death=np.inf)]:
        bad = copy.deepcopy(cache)
        bad["pairs"][0].update(mutation)
        with pytest.raises(ValueError):
            losses.h1_loss(S, S, source_result=bad)
        # Also reject bad target metadata. Full VR has no essential H1 naturally.
        monkeypatch.setattr(topology, "h1_persistence", lambda *a, **kw: bad)
        with pytest.raises(ValueError):
            losses.h1_loss(S, S, source_result=cache)
    monkeypatch.setattr(topology, "h1_persistence", original)
    for key in ["max_simplices", "max_reduction_entries", "max_reduction_operations"]:
        with pytest.raises(RuntimeError):
            losses.h1_loss(S, S, source_result=cache, budgets={key: 0})
    with pytest.raises(ValueError, match="full filtration"):
        losses.h1_loss(S, S, budgets={"max_radius": 2})
    with pytest.raises(RuntimeError, match="matching"):
        losses.h1_loss(S, S, source_result=cache, max_matching_size=1)


def benchmark():
    """Paired warm medians; no CI speed thresholds or native-memory claims."""
    import json
    import platform
    import statistics
    import sys
    import time
    from unittest.mock import patch

    def payload_bytes(value):
        # Deduplicate identities, including shared keys/small ints; not RSS.
        seen = set()
        def walk(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            total = sys.getsizeof(obj)
            if isinstance(obj, dict):
                total += sum(walk(k) + walk(v) for k, v in obj.items())
            elif isinstance(obj, (list, tuple)):
                total += sum(walk(v) for v in obj)
            return total
        return walk(value)

    torch.set_num_threads(1)
    rows = []
    repeats, warmups = 7, 2
    for n in [32, 64, 128]:
        rng = np.random.default_rng(17)
        X = rng.normal(size=(n, 8))
        Y = X + rng.normal(scale=.03, size=X.shape)
        D = topology.distance_matrix(X)
        S = torch.tensor(D)
        apis = {"full": topology.persistence, "fast": topology.h1_persistence}
        caches = {name: api(D) for name, api in apis.items()}
        assert caches["fast"] == filtered(caches["full"])
        row = dict(n=n, pairs={name: len(cache["pairs"]) for name, cache in caches.items()},
                   payload_bytes={name: payload_bytes(cache) for name, cache in caches.items()},
                   stats={key: value for key, value in caches["full"].items() if key != "pairs"})
        for mode in ["persistence", "cached_loss_backward", "uncached_loss_backward"]:
            times = {name: [] for name in apis}
            values = {}
            for iteration in range(-warmups, repeats):
                # Alternate execution order to reduce order/thermal bias.
                order = list(apis) if iteration % 2 == 0 else list(reversed(apis))
                for name in order:
                    api = apis[name]
                    # Patch is outside timing; selects serializer, not a fake PH.
                    with patch.object(topology, "h1_persistence", api):
                        start = time.perf_counter()
                        if mode == "persistence":
                            result = api(D)
                        else:
                            target = torch.tensor(Y, requires_grad=True)
                            result = losses.h1_loss(S, losses.pairwise_distances(target),
                                source_result=caches[name] if mode == "cached_loss_backward" else None)
                            (result["pd1"] + result["crit1"]).backward()
                        elapsed = time.perf_counter() - start
                    if mode != "persistence":
                        current = ([result[k].item() for k in ["pd1", "crit1"]], target.grad.clone())
                        if name in values:
                            assert current[0] == values[name][0] and torch.equal(current[1], values[name][1])
                        values[name] = current
                    if iteration >= 0:
                        times[name].append(elapsed)
                    del result
            if values:
                assert values["full"][0] == values["fast"][0]
                assert torch.equal(values["full"][1], values["fast"][1])
            row[mode] = {name: dict(median_seconds=statistics.median(samples), seconds=samples)
                         for name, samples in times.items()}
        rows.append(row)
    print(json.dumps(dict(platform=platform.platform(), python=platform.python_version(),
                         numpy=np.__version__, torch=torch.__version__, seed=17, dimensions=8,
                         warmups=warmups, repeats=repeats, torch_threads=1, rows=rows), indent=2))


if __name__ == "__main__":
    benchmark()
