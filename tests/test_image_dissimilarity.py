"""Synthetic experimental-image API contracts; real Digits checks are opt-in."""
import itertools
import os
import subprocess
import sys
import types
import numpy as np
import pytest
from open_deep_tda.image_dissimilarity import (
    TranslationTangentDissimilarity as API, _box_residual_norm as solve,
)


def _stable_norm(x):
    x = np.asarray(x)
    scale = float(np.max(abs(x)))
    return 0. if scale == 0 else float(scale*np.sqrt(np.sum((x/scale)**2)))


def _independent(J, delta):
    # Nine independent active sets, direct least squares and direct residuals.
    candidates = []
    for status in itertools.product((-1., 0., 1.), repeat=2):
        t = np.array(status)
        fixed = np.flatnonzero(t)
        free = np.flatnonzero(t == 0)
        rhs = delta-J[:, fixed]@t[fixed]
        if len(free):
            t[free] = np.linalg.lstsq(J[:, free], rhs, rcond=None)[0]
        if np.all(abs(t) <= 1):
            candidates.append((_stable_norm(J@t-delta), t))
    return min(candidates, key=lambda item: item[0])


def _fixtures():
    # Preserve the exact synthetic failure stream without accessing a dataset.
    rng = np.random.default_rng(20260918)
    rng.integers(0, 1797, size=(768, 2))
    rng.choice(1797, 48, replace=False)
    rng.integers(0, 1797, size=(250000, 3))
    u = rng.normal(size=64)
    v = rng.normal(size=64)
    v -= u*(u@v)/(u@u)
    return [
        ('rank0', np.zeros((64, 2)), rng.normal(size=64)),
        ('rank1_collinear', np.column_stack((u, 2*u)), 3*u),
        ('rank1_zero_column', np.column_stack((u, np.zeros(64))), 2*u),
        ('tiny_rank2', 1e-10*np.column_stack((u, v)), 1e-10*(.4*u-.3*v)),
        ('illconditioned_1e-8', np.column_stack((u, u+1e-8*v)), .7e-8*v),
        ('illconditioned_1e-6', np.column_stack((u, u+1e-6*v)), .7e-6*v),
        ('illconditioned_1e-8_scaled1e8', 1e8*np.column_stack((u, u+1e-8*v)), .7*v),
        ('cancellation_scale1e8', 1e8*np.column_stack((u, v)), 1e8*(.4*u-.3*v)),
    ]


oracle = types.SimpleNamespace(stable_norm=_stable_norm, independent=_independent,
                              original_fixtures=lambda: (None, None, _fixtures()))


def images():
    return np.random.default_rng(101).uniform(0, 3, (5, 3, 4))


def test_shapes_flat_explicit_and_orientation():
    x = images()
    a = API().fit_transform(x)
    b = API(image_shape=(3, 4)).fit_transform(x.reshape(5, 12))
    np.testing.assert_allclose(a, b, atol=1e-14, rtol=1e-14)
    cross = API().fit(x[:3]).transform(x[3:])
    assert cross.shape == (2, 3)
    np.testing.assert_allclose(cross, a[3:, :3], atol=1e-14, rtol=1e-14)
    with pytest.raises((ValueError, TypeError)):
        API().fit(x.reshape(5, 12))


@pytest.mark.parametrize('bad', [
    np.ones((2, 3, 4), dtype=bool), np.ones((2, 3, 4), dtype=complex),
    np.ones((2, 3, 4), dtype=object), np.full((2, 3, 4), '1'),
    np.ma.array(np.ones((2, 3, 4)), mask=False),
    np.full((2, 3, 4), -1.), np.full((2, 3, 4), np.inf),
    np.full((2, 3, 4), np.nan), np.ones((2, 3, 4, 1)),
    np.ones(12), np.ones((0, 3, 4)), np.ones((2, 0, 4)),
])
def test_reject_bad_fit(bad):
    with pytest.raises((ValueError, TypeError)):
        API().fit(bad)


@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float32, np.float64])
def test_real_numeric_dtypes(dtype):
    x = np.arange(36).reshape(3, 3, 4).astype(dtype)
    result = API().fit_transform(x)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result)) and np.all(result >= 0)


def test_query_validation_and_unfitted():
    model = API()
    with pytest.raises((RuntimeError, ValueError)):
        model.transform(images())
    model.fit(images())
    for bad in (np.ones((2, 4, 3)), np.ones((2, 3, 4), complex),
                np.full((2, 3, 4), np.nan), np.full((2, 3, 4), -1.),
                np.ma.array(images(), mask=False)):
        with pytest.raises((ValueError, TypeError)):
            model.transform(bad)


def test_ownership_query_independence_and_permutation():
    x = images()
    original = x.copy()
    model = API().fit(x)
    assert model is not None
    baseline = model.transform(original)
    x[:] = 100
    np.testing.assert_array_equal(model.transform(original), baseline)
    for i in range(len(original)):
        np.testing.assert_allclose(model.transform(original[i:i+1]), baseline[i:i+1], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(model.transform(original[::-1]), baseline[::-1], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(API().fit(original[::-1]).transform(original), baseline[:, ::-1], atol=1e-14, rtol=1e-14)
    baseline[:] = -1
    assert np.all(model.transform(original) >= 0)
    np.testing.assert_array_equal(original, images())


def test_noncontiguous_readonly_input():
    x = images()[:, :, ::-1]
    x.flags.writeable = False
    np.testing.assert_allclose(API().fit_transform(x), API().fit_transform(x.copy()), atol=1e-14, rtol=1e-14)


def test_zero_duplicates_ties_symmetry_and_raw_domination():
    x = np.stack([np.zeros((3, 4)), np.ones((3, 4)), np.ones((3, 4)), np.zeros((3, 4))])
    result = API().fit_transform(x)
    assert np.array_equal(result, result.T)
    assert np.all(np.diag(result) == 0)
    assert result[0, 3] == result[1, 2] == 0
    np.testing.assert_array_equal(result[:, 1], result[:, 2])
    raw = np.linalg.norm(x[:, None]-x[None, :], axis=(2, 3))
    assert np.all(result <= raw+1e-14)
    assert np.all(API().fit_transform(np.zeros((3, 3, 4))) == 0)


@pytest.mark.parametrize('shape', [(3,), (3, 4, 1), (0, 4), (3.5, 4), (True, 4)])
def test_invalid_image_shape(shape):
    with pytest.raises((ValueError, TypeError)):
        API(image_shape=shape).fit(images())


@pytest.mark.parametrize('parameter', ['max_reference', 'max_query', 'max_pixels', 'max_matrix_bytes', 'max_work'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_budget_values(parameter, value):
    with pytest.raises((ValueError, TypeError)):
        API(**{parameter: value}).fit(images())


def test_dense_budgets():
    x = images()
    with pytest.raises((ValueError, MemoryError)):
        API(max_reference=4).fit(x)
    with pytest.raises((ValueError, MemoryError)):
        API(max_pixels=11).fit(x)
    model = API(max_reference=5, max_query=2).fit(x)
    assert model.transform(x[:2]).shape == (2, 5)
    with pytest.raises((ValueError, MemoryError)):
        model.transform(x[:3])
    assert API(max_matrix_bytes=3*5*5*8).fit_transform(x).shape == (5, 5)
    with pytest.raises((ValueError, MemoryError)):
        API(max_matrix_bytes=3*5*5*8-1).fit_transform(x)
    with pytest.raises((ValueError, MemoryError)):
        API(max_work=1).fit_transform(x)


@pytest.mark.parametrize('name,J,delta', oracle.original_fixtures()[2], ids=lambda x: x if isinstance(x, str) else '')
def test_exact_historical_synthetic_regressions(name, J, delta):
    value, t = solve(J, delta)
    explicit = oracle.stable_norm(J @ t-delta)
    expected, _ = oracle.independent(J, delta)
    tol = 64*np.finfo(float).eps*(oracle.stable_norm(J.ravel())+oracle.stable_norm(delta))
    assert np.all(abs(t) <= 1)
    assert np.isfinite(value) and value >= 0
    assert value <= expected+tol
    assert abs(value-explicit) <= tol
    assert not (value == 0 and explicit > 0), 'must not conceal cancellation with zero clamp'
    columns_explicit = oracle.stable_norm(J[:, 0]*t[0]+J[:, 1]*t[1]-delta)
    assert abs(value-columns_explicit) <= 4*np.finfo(float).eps*columns_explicit
    if name.startswith('illconditioned'):
        np.testing.assert_allclose(t, [-.7, .7], atol=1e-6, rtol=0)


@pytest.mark.parametrize('scale', [1e-140, 1e140])
def test_solver_overflow_underflow_stable_residual(scale):
    J = scale*np.array([[1., 0.], [0., 1.], [0., 0.]])
    delta = scale*np.array([.25, -.5, .75])
    value, t = solve(J, delta)
    assert np.isfinite(value) and value > 0
    np.testing.assert_allclose(value/scale, .75, rtol=1e-14, atol=0)
    np.testing.assert_allclose(t, [.25, -.5], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('scale', [1e-140, 1e140])
def test_image_scale_overflow_underflow(scale):
    x = images()
    base = API().fit_transform(x)
    result = API(max_intensity=1e150).fit_transform(x*scale)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result/scale, base, rtol=1e-12, atol=1e-12)


def test_nonmetric_without_digits_or_labels():
    # Tiny deterministic nonnegative 2x3 grid; no cached dataset dependency.
    import itertools
    grid = np.array(list(itertools.product(range(3), repeat=3)), float).reshape(-1, 1, 3)
    grid = np.repeat(grid, 2, axis=1)
    distance = API().fit_transform(grid)
    gap = max(float(np.max(distance-distance[:, b, None]-distance[b, None, :])) for b in range(len(grid)))
    assert gap > 1e-8


def test_no_torch_import():
    # Whole-suite collection may legitimately have imported Torch elsewhere.
    code = ("import sys; import open_deep_tda.image_dissimilarity; "
            "assert 'torch' not in sys.modules; assert 'numba' not in sys.modules")
    subprocess.run([sys.executable, '-B', '-c', code], check=True, timeout=30)


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_explicit_unsupported_scale_guards(scale):
    # Guarded domain is documented; no silent infinity/underflow or auto clipping.
    J = scale*np.array([[1., 0.], [0., 1.], [0., 0.]])
    delta = scale*np.array([.25, -.5, .75])
    with pytest.raises((ValueError, FloatingPointError)):
        solve(J, delta)
    with pytest.raises((ValueError, FloatingPointError)):
        API(max_intensity=1e150).fit_transform(images()*scale)


def test_reference_properties_are_copies_and_fit_is_transactional():
    x = images()
    model = API().fit(x)
    assert model.n_reference_ == 5
    assert model.n_features_in_ == 12
    assert model.image_shape_ == (3, 4)
    assert model.diagnostics_['metric'] is False
    saved = model.transform(x)
    retrieved = model.reference_images_
    retrieved[:] = 0
    np.testing.assert_array_equal(model.reference_images_, x)
    with pytest.raises(ValueError):
        model.fit(np.full_like(x, np.nan))
    np.testing.assert_array_equal(model.transform(x), saved)
    with pytest.raises(ValueError):
        model.fit_transform(np.full_like(x, -1))
    np.testing.assert_array_equal(model.transform(x), saved)
    assert model.transform(np.empty((0, 3, 4))).shape == (0, 5)


def test_exact_work_budget_boundaries_and_batch_size_independence():
    x = images()
    # Five candidates per directed square pair; ten for a rectangular pair.
    square_work = 5*5*5*12
    assert API(max_work=square_work).fit_transform(x).shape == (5, 5)
    with pytest.raises(ValueError):
        API(max_work=square_work-1).fit(x)
    model = API(max_work=square_work).fit(x)
    assert model.transform(x[:2]).shape == (2, 5)
    with pytest.raises(ValueError):
        model.transform(x[:3])
    a = API(block_size=1).fit_transform(x)
    b = API(block_size=4).fit_transform(x)
    np.testing.assert_array_equal(a, b)
    q = np.random.default_rng(22).uniform(0, 4, (4, 3, 4))
    np.testing.assert_array_equal(API(block_size=1).fit(x).transform(q),
                                  API(block_size=3).fit(x).transform(q))


def test_maximum_intensity_and_workspace_budget_guards():
    x = images()
    with pytest.raises(ValueError):
        API(max_image_bytes=1).fit(x)
    with pytest.raises(ValueError):
        API(max_intensity=1).fit(x)
    for value in (0, -1, np.inf, np.nan, True, 1e151):
        with pytest.raises((ValueError, TypeError)):
            API(max_intensity=value)


@pytest.mark.parametrize('bad_J,bad_delta', [
    (np.zeros((4, 3)), np.zeros(4)),
    (np.zeros((4, 2)), np.zeros(3)),
    (np.zeros((4, 2), complex), np.zeros(4)),
    (np.zeros((4, 2)), np.full(4, np.nan)),
    (np.zeros((4, 2), bool), np.zeros(4)),
    (np.ma.array(np.zeros((4, 2))), np.zeros(4)),
])
def test_internal_solver_validation(bad_J, bad_delta):
    with pytest.raises((ValueError, TypeError)):
        solve(bad_J, bad_delta)


@pytest.mark.parametrize('parameter', ['max_reference', 'max_query'])
def test_experimental_dense_cap_2000(parameter):
    with pytest.raises(ValueError):
        API(**{parameter: 2001})


def test_raw_array_state_readonly_and_no_labels_argument():
    model = API().fit(images())
    assert not model._state[0].flags.writeable
    for factor in model._state[2]:
        for a in factor:
            if isinstance(a, np.ndarray):
                assert not a.flags.writeable
    with pytest.raises(TypeError):
        API().fit(images(), y=np.arange(5))
    assert model.diagnostics_['experimental'] is True


def test_known_huge_cancellation_regression_not_false_zero():
    name, J, delta = next(f for f in _fixtures() if f[0] == 'illconditioned_1e-8_scaled1e8')
    norm, t = solve(J, delta)
    # The old normal-equation/polynomial calculation could return zero at
    # (-1,1) with a large actual residual. Check the returned coefficients too.
    assert _stable_norm(J@np.array([-1., 1.])-delta) > 1
    np.testing.assert_allclose(t, [-.7, .7], atol=1e-6, rtol=0)
    assert norm > 0
    assert _stable_norm(J@t-delta) < 1e-5


@pytest.mark.skipif(os.environ.get('OPEN_DEEP_TDA_IMAGE_ORACLE') != '1',
                    reason='explicit opt-in bundled Digits numerical oracle; no quality fits')
def test_opt_in_digits_1536_qp_and_ordinary_reference():
    from sklearn.datasets import load_digits
    from scipy.optimize import lsq_linear
    from scipy.spatial.distance import cdist
    data = load_digits().images  # labels never read
    rng = np.random.default_rng(20260918)
    pairs = rng.integers(0, len(data), size=(768, 2))

    def jac(image):
        p = np.pad(image, 1)
        return np.column_stack(((p[1:-1, 2:]-p[1:-1, :-2]).ravel()/2,
                                (p[2:, 1:-1]-p[:-2, 1:-1]).ravel()/2))

    for i, j in pairs:
        for a, b in ((i, j), (j, i)):
            J, delta = jac(data[a]), (data[b]-data[a]).ravel()
            norm, t = solve(J, delta)
            ref = lsq_linear(J, delta, bounds=(-1, 1), tol=1e-13,
                             lsq_solver='exact', max_iter=1000)
            np.testing.assert_allclose(norm**2, 2*ref.cost, atol=1e-8, rtol=1e-12)
            assert abs(norm-_stable_norm(J@t-delta)) < 1e-12

    # Ordinary, bounded Digits-scale parity with the original algebraic
    # definition. This comparator is NOT used to judge ill-conditioned cases.
    block = data[rng.choice(len(data), 48, replace=False)]
    X = block.reshape(48, -1)
    base = cdist(X, X, metric='sqeuclidean')
    directional = np.empty_like(base)
    for i, image in enumerate(block):
        J = jac(image); G = J.T@J; B = (X-X[i])@J
        t = B@np.linalg.pinv(G)
        cost = base[i]-2*np.sum(B*t, axis=1)+np.einsum('ij,jk,ik->i', t, G, t)
        cost[np.any(abs(t)>1, axis=1)] = np.inf
        for axis in (0, 1):
            other = 1-axis
            for sign in (-1., 1.):
                t = np.empty((48, 2)); t[:, axis] = sign
                t[:, other] = np.clip((B[:, other]-G[other, axis]*sign)/G[other, other], -1, 1) if G[other, other] > 0 else 0
                cost = np.minimum(cost, base[i]-2*np.sum(B*t, axis=1)+np.einsum('ij,jk,ik->i', t, G, t))
        directional[i] = np.maximum(cost, 0)
    expected = np.sqrt(np.maximum((directional+directional.T)/2, 0))
    np.fill_diagonal(expected, 0)
    actual = API().fit_transform(block)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-14)
