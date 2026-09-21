"""Standalone binding smoke/regression tests; run by CTest when Python is enabled."""
import importlib
import math
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
core = importlib.import_module("_core")
x = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
d = np.ascontiguousarray(np.linalg.norm(x[:, None] - x[None, :], axis=-1))
r = core.persistence(d)
assert set(r) == {"pairs", "n_vertices", "n_simplices", "reduction_operations", "peak_reduction_entries", "truncated"}
assert r["n_simplices"] == 14 and not r["truncated"]
positive = [p for p in r["pairs"] if p["dimension"] == 1 and p["death"] > p["birth"]]
assert len(positive) == 1
assert positive[0]["birth"] == 1 and positive[0]["death"] == math.sqrt(2)
assert set(positive[0]) == {"dimension", "birth", "death", "birth_simplex", "death_simplex", "birth_edge", "death_edge", "essential", "censored"}
for p in r["pairs"]:
    for name in ("birth_simplex", "death_simplex", "birth_edge", "death_edge"):
        assert isinstance(p[name], list), (name, type(p[name]))
assert core.mst(d).shape == (3, 2)
assert core.mst(d).dtype.kind == "i"
assert core.mst(np.empty((0, 0), dtype=np.float64)).shape == (0, 2)
assert core.persistence(np.zeros((1, 1)))["pairs"][0]["essential"]
cut = core.persistence(d, max_radius=1)
assert cut["truncated"]
assert any(p["dimension"] == 1 and p["censored"] and math.isinf(p["death"]) for p in cut["pairs"])

def raises(kind, f):
    try:
        f()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")

for bad in (np.zeros((2, 3)), np.array([[0., -1.], [-1., 0.]]),
            np.array([[0., np.nan], [np.nan, 0.]]), np.ones((2, 2)),
            np.array([[0., 1.], [2., 0.]]), d.astype(np.float32), d[:, ::-1]):
    raises(ValueError, lambda: core.persistence(bad))
    raises(ValueError, lambda: core.mst(bad))
for key in ("max_simplices", "max_reduction_entries", "max_reduction_operations"):
    raises(RuntimeError, lambda: core.persistence(d, **{key: 1}))
    for invalid in (-1, 1.5, True, 2**100):
        raises(ValueError, lambda: core.persistence(d, **{key: invalid}))
raises(ValueError, lambda: core.persistence(d, max_radius=float("nan")))
assert core.persistence(d, max_simplices=np.int64(14))["n_simplices"] == 14
assert core.persistence(np.array([[1e-12, 1.], [1.+1e-12, -1e-12]]))["n_vertices"] == 2
# Barcode invariance away from ties; representative simplex IDs may change.
rng = np.random.default_rng(2026)
y = rng.normal(size=(12, 3))
dy = np.ascontiguousarray(np.linalg.norm(y[:, None] - y[None, :], axis=-1))
def barcode(distance, dimension):
    return sorted((p["birth"], p["death"]) for p in core.persistence(distance)["pairs"]
                  if p["dimension"] == dimension and math.isfinite(p["death"]))
permutation = rng.permutation(len(y))
rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
moved = y @ rotation + np.array([1., 3., -2.])
dm = np.ascontiguousarray(np.linalg.norm(moved[:, None] - moved[None, :], axis=-1))
for dimension in (0, 1):
    base = np.array(barcode(dy, dimension))
    np.testing.assert_allclose(barcode(np.ascontiguousarray(dy[permutation][:, permutation]), dimension), base)
    np.testing.assert_allclose(barcode(dm, dimension), base, atol=1e-14)
    np.testing.assert_allclose(barcode(dy * 2.5, dimension), base * 2.5)
print("Binding API, square PH, truncation, types, validation, budgets, and invariance tests passed.")
