"""Synthetic exact-rational interval, boundary, and preflight regressions.

Verified pitfalls: birth equality needs exact fallback, whereas ambiguous
strict disk clearance must fail closed. Underflow/overflow refusal does not
prove dependence. Reserving raw edges before cancellation makes budget
failure predictable and precedes even finite-coordinate scans. Every target
row matters, including a final external row in a million-row synthetic array.
No dataset files or network access are used.

Run: python3.9 -B -m pytest -q -p no:cacheprovider tests/test_structural_planar_numeric.py
"""
from dataclasses import replace
from fractions import Fraction as F
import math
import numpy as np
import pytest
import open_deep_tda.structural_planar as pc


def diamond():
    return np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]), [[(0, 1), (1, 2), (2, 3), (3, 0)]], np.zeros((1, 2))


def exact_squared(a, b):
    return sum((F(float(x))-F(float(y)))**2 for x, y in zip(a, b))


def assert_enclosed(a, b):
    lo, hi = pc.squared_distance_bounds(a, b)
    exact = exact_squared(a, b)
    assert not math.isnan(float(lo)) and not math.isnan(float(hi))
    assert F(float(lo)) <= exact
    assert math.isinf(float(hi)) or exact <= F(float(hi))


def test_interval_extremes_and_random_binary64():
    tiny = np.nextafter(0., 1.)
    huge = np.finfo(float).max
    values = [0., -0., tiny, -tiny, np.finfo(float).tiny, 1.,
              np.nextafter(1., 0.), np.nextafter(1., np.inf), huge, -huge]
    for x in values:
        for y in values:
            assert_enclosed(np.array([x, y]), np.array([y, x]))
    rng = np.random.default_rng(712)
    tested = 0
    while tested < 1000:
        values = rng.bytes(32)
        p = np.frombuffer(values, dtype=np.float64).reshape(2, 2)
        if np.isfinite(p).all():
            assert_enclosed(p[0], p[1])
            tested += 1


def test_exact_birth_equality_and_adjacent_floats():
    # Unit square edge equality uses exact integer fallback, not a tolerance.
    p = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    _, c, _ = diamond()
    h = np.array([[.5, .5]])
    result = pc.certify(p, c, 1., 1., h)
    assert result.certified and result.work['exact_edge_tests'] == 4
    assert not pc.certify(p, c, np.nextafter(1., 0.), 1., h).certified
    assert pc.certify(p, c, np.nextafter(1., np.inf), np.nextafter(1., np.inf), h).certified


def test_strict_disk_boundary_float_variants():
    p, c, h = diamond()
    # An external (not chain) point controls the disk. Check exact 3*d²>b².
    b = 1.5
    r = b / np.sqrt(3.)
    trials = [r]
    for toward in (0., np.inf):
        value = r
        for _ in range(16):
            value = np.nextafter(value, toward)
            trials.append(value)
    accepted = 0
    rejected = 0
    for x in trials:
        points = np.vstack([p, [x, 0.]])
        result = pc.certify(points, c, b, b, h)
        safe = 3 * F(float(x))**2 > F(b)**2
        if result.certified:
            assert safe
            accepted += 1
        else:
            rejected += 1
    assert accepted and rejected
    # An exactly representable zero-radius equality must also fail strictly.
    assert not pc.certify(np.zeros((4, 2)), c, 0., 0., h).certified


def test_unoriented_half_open_crossings_and_translation():
    p, c, h = diamond()
    expected = pc.certify(p, c, 1.5, 1.5, h)
    assert expected.winding == ((1,),)
    reversed_edges = [[(v, u) for u, v in reversed(c[0])]]
    assert pc.certify(p, reversed_edges, 1.5, 1.5, h).winding == expected.winding
    # Ray goes through a chain vertex; move hole by one subnormal either way.
    for delta in [0., np.nextafter(0., 1.), np.nextafter(0., -1.)]:
        result = pc.certify(p, c, 1.5, 1.5, np.array([[0., delta]]))
        assert result.certified and result.winding == ((1,),)
    for shift in [2.**40, -2.**40]:
        assert pc.certify(p+shift, c, 1.5, 1.5, h+shift).certified
    # Duplicates cancel modulo two, orientation has no effect.
    redundant = [c[0] + [(0, 1), (1, 0)]]
    assert pc.certify(p, redundant, 1.5, 1.5, h).certified


def test_extreme_scales_conservative_no_nan_accept():
    p, c, h = diamond()
    for scale in [2.**-500, 2.**500]:
        assert pc.certify(p*scale, c, 1.5*scale, 1.5*scale, h).certified
    for scale in [np.nextafter(0., 1.), 2.**1022]:
        result = pc.certify(p*scale, c, 1.5*scale, 1.5*scale, h)
        assert not result.certified  # under/overflow may only refuse proof


@pytest.mark.parametrize('field,cap', [('max_vertices', 3), ('max_holes', 0),
    ('max_cycles', 0), ('max_chain_edges', 3), ('max_distance_pairs', 7),
    ('max_orientation_tests', 3), ('max_exact_edge_tests', 3)])
def test_budget_first_no_partial_pass(monkeypatch, field, cap):
    p, c, h = diamond()
    def forbidden(*args, **kwargs):
        raise AssertionError('geometry ran before resource preflight')
    monkeypatch.setattr(pc.np, 'isfinite', forbidden)
    monkeypatch.setattr(pc, 'squared_distance_bounds', forbidden)
    with pytest.raises(pc.ResourceLimitError):
        pc.certify(p, c, 1.5, 1.5, h, limits=replace(pc.Limits(), **{field: cap}))


def test_budget_exact_boundary_and_chunking():
    p, c, h = diamond()
    lim = pc.Limits(max_vertices=4, max_holes=1, max_cycles=1,
                    max_chain_edges=4, max_distance_pairs=8,
                    max_orientation_tests=4, max_exact_edge_tests=4, chunk_size=1)
    result = pc.certify(p, c, 1.5, 1.5, h, limits=lim)
    assert result.certified and result.work['distance_pairs'] == 8
    for name, value in [('max_vertices', 1_000_001), ('chunk_size', 0),
                        ('max_holes', True), ('max_chain_edges', 1.5)]:
        with pytest.raises(ValueError):
            pc.certify(p, c, 1.5, 1.5, h, limits=replace(lim, **{name: value}))


def test_late_external_row_and_hard_vertex_bound():
    p, c, h = diamond()
    # Full upper-bound N; no pair matrix, and the final complement row matters.
    points = np.full((1_000_000, 2), 10., dtype=np.float64)
    points[:4] = p
    assert pc.certify(points, c, 1.5, 1.5, h).certified
    points[-1] = h[0]
    result = pc.certify(points, c, 1.5, 1.5, h)
    assert not result.certified
    assert result.work['distance_pairs'] == len(points)+4
    oversized = np.broadcast_to(np.zeros((1, 2)), (1_000_001, 2))
    with pytest.raises(pc.ResourceLimitError):
        pc.certify(oversized, c, 1.5, 1.5, h)


@pytest.mark.parametrize('birth,survival', [(2., 1.), (-1., 1.), (np.nan, 1.),
    (1., np.inf), (True, 1.), ('1', 1.)])
def test_invalid_radii(birth, survival):
    p, c, h = diamond()
    with pytest.raises(ValueError):
        pc.certify(p, c, birth, survival, h)


def test_input_validation():
    p, c, h = diamond()
    for bad in [p.astype(np.float32), p[:, 0], [[1., 2.]], np.full((4, 2), np.nan)]:
        with pytest.raises(ValueError):
            pc.certify(bad, c, 1.5, 1.5, h)
    for bad in [[[(0., 1.)]], [[(True, 1)]], [[(-1, 0)]], [[(0, 4)]],
                [[(0, 0)]], [[(0, 1)]], [[(0, 1, 2)]], [[np.array(1)]]]:
        with pytest.raises(ValueError):
            pc.certify(p, bad, 1.5, 1.5, h)
    assert not pc.certify(p, [], 1.5, 1.5, h).certified
    assert not pc.certify(p, [np.empty((0, 2), dtype=int)], 1.5, 1.5, h).certified


def test_public_api_exports_and_result_scope():
    assert set(pc.__all__) == {
        "Limits", "Certificate", "ResourceLimitError", "certify",
        "squared_distance_bounds",
    }
    assert all(hasattr(pc, name) for name in pc.__all__)
    assert issubclass(pc.ResourceLimitError, ValueError)
    p, c, h = diamond()
    result = pc.certify(p, c, 1.5, 1.5, h)
    assert isinstance(result, pc.Certificate) and result.certified
    assert result.scope == (
        "selected target H1 only; real Euclidean represented float64 coordinates"
    )
