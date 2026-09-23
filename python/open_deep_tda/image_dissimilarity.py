"""Experimental opt-in grayscale translation-tangent dissimilarity.

A local linear image prior, not a metric, exact image registration, pixel warp,
clustering fix, or topology/generalization guarantee. Generic numeric defaults
are unaffected. Images and derivative factors retained by ``fit`` are sensitive.

The general tangent-distance idea is classical (Simard, LeCun and Denker,
"Efficient Pattern Recognition Using a New Transformation Distance", 1993).
This independently implemented bounded, symmetric first-order construction is
not a reproduction of that algorithm. Each directional residual uses direct-J
SVD and five box active patterns; see ``TranslationTangentDissimilarity``.
"""
import numbers
import numpy as np

__all__ = ['TranslationTangentDissimilarity']

_MAX_VALUE = 1e150
_MIN_POSITIVE = 1e-150


def _integer(x, name, minimum=1, maximum=None):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < minimum or (maximum is not None and x > maximum):
        raise ValueError(name + ' outside supported integer domain')
    return int(x)


def _real_array(x, name):
    if np.ma.isMaskedArray(x):
        raise ValueError(name + ': masked arrays unsupported')
    a = np.asarray(x)
    if a.dtype.kind not in 'iuf':
        raise ValueError(name + ': real non-boolean numeric values required')
    return a


def _stable_norm_rows(a):
    """Scaled Euclidean norm: never form large/small raw squared values."""
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise FloatingPointError('nonfinite residual')
    scale = np.max(np.abs(a), axis=1)
    unit = np.divide(a, scale[:, None], out=np.zeros_like(a), where=scale[:, None] != 0)
    length = np.sqrt(np.sum(unit * unit, axis=1))
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        out = scale * length
    if not np.isfinite(out).all() or np.any((scale > 0) & (out == 0)):
        raise FloatingPointError('residual norm overflow/underflow')
    return out


def _factor(J):
    """SVD of the DIRECT two-column Jacobian, not J.T@J."""
    scale = float(np.max(np.abs(J)))
    if scale == 0:
        return (J, scale, np.zeros_like(J), np.zeros(2), np.eye(2))
    A = J / scale
    if np.any((J != 0) & (A == 0)):
        raise FloatingPointError('Jacobian scaling underflow')
    U, singular, Vt = np.linalg.svd(A, full_matrices=False)
    # Same explicit numerical-rank convention as direct lstsq(rcond=None).
    cutoff = np.finfo(float).eps * max(J.shape) * singular[0]
    inv = np.divide(1., singular, out=np.zeros_like(singular), where=singular > cutoff)
    return (J, scale, U, inv, Vt)


def _bounded_edge_coefficient(column, rhs):
    """Clipped 1D least squares without raw column squared-norm overflow."""
    cscale = float(np.max(np.abs(column)))
    if cscale == 0:
        return np.zeros(len(rhs))
    c = column / cscale
    rscale = np.max(np.abs(rhs), axis=1)
    r = np.divide(rhs, rscale[:, None], out=np.zeros_like(rhs), where=rscale[:, None] != 0)
    numerator = np.sum(r * c[None, :], axis=1)
    denominator = float(np.sum(c * c))
    # Compare in logarithmic magnitude before division: outside-box solutions
    # need only their sign, not a potentially overflowing coefficient.
    result = np.zeros(len(rhs))
    nz = (numerator != 0) & (rscale != 0)
    logs = np.full(len(rhs), -np.inf)
    logs[nz] = np.log(np.abs(numerator[nz])) + np.log(rscale[nz]) - np.log(cscale) - np.log(denominator)
    outside = logs >= 0
    result[outside] = np.sign(numerator[outside])
    inside = nz & ~outside
    # Log evaluation avoids overflow of rscale/cscale; exp underflow to zero
    # here is a negligible bounded coefficient, never a fabricated zero norm.
    result[inside] = np.sign(numerator[inside]) * np.exp(logs[inside])
    return np.clip(result, -1., 1.)


def _solve_many(factor, deltas):
    """Five active patterns: feasible interior, then each of four box edges.

    Every candidate is scored by its actual residual vector, stably normed.
    Returns norm and coefficients. No polynomial cost, square clamp, or JᵀJ.
    """
    J, jscale, U, inv, Vt = factor
    d = np.asarray(deltas, dtype=float)
    ds = np.max(np.abs(d), axis=1)
    dn = np.divide(d, ds[:, None], out=np.zeros_like(d), where=ds[:, None] != 0)
    if jscale == 0:
        return _stable_norm_rows(d), np.zeros((len(d), 2))
    # Fixed per-row contraction order (no batch-size-dependent GEMM/GEMV).
    t = np.einsum('ij,jk->ik', dn, U, optimize=False) * inv[None, :]
    t = np.einsum('ij,jk->ik', t, Vt, optimize=False)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        t *= (ds / jscale)[:, None]
    valid = np.isfinite(t).all(axis=1) & (np.abs(t) <= 1).all(axis=1)
    best = np.full(len(d), np.inf)
    choice = np.zeros((len(d), 2))

    def consider(candidates, mask=None):
        nonlocal best, choice
        if mask is None:
            mask = np.ones(len(d), dtype=bool)
        ids = np.flatnonzero(mask)
        if not len(ids):
            return
        v = candidates[ids]
        # Explicit residual in original units; two bounded coefficients ensure
        # finite products within the guarded domain. No base-2Bt+tGt.
        residual = J[None, :, 0] * v[:, None, 0] + J[None, :, 1] * v[:, None, 1] - d[ids]
        cost = _stable_norm_rows(residual)
        take = cost < best[ids]
        at = ids[take]
        best[at] = cost[take]
        choice[at] = v[take]

    consider(t, valid)
    for axis in (0, 1):
        other = 1 - axis
        for sign in (-1., 1.):
            candidate = np.zeros((len(d), 2))
            candidate[:, axis] = sign
            rhs = d - sign * J[:, axis][None, :]
            candidate[:, other] = _bounded_edge_coefficient(J[:, other], rhs)
            consider(candidate)
    if not np.isfinite(best).all():
        raise FloatingPointError('no finite box candidate')
    return best, choice


def _box_residual_norm(J, delta):
    """Auditable direct-J box solver (internal numeric helper), norm and t.

    Floating SVD rank truncation is explicit; not an exact-arithmetic oracle.
    Helper permits signed Jacobians/residuals up to1e150 and >=1e-150 positive
    magnitudes. All-zero columns and rank deficiency are allowed.
    """
    J = _real_array(J, 'J'); delta = _real_array(delta, 'delta')
    if J.ndim != 2 or J.shape[1] != 2 or J.shape[0] < 2 or delta.shape != (J.shape[0],):
        raise ValueError('expected J=(pixels>=2,2), delta=(pixels,)')
    for a in (J, delta):
        if not np.isfinite(a).all() or np.max(np.abs(a)) > _MAX_VALUE:
            raise ValueError('solver finite magnitude limit1e150 exceeded')
        positive = np.abs(a[a != 0])
        if positive.size and positive.min() < _MIN_POSITIVE:
            raise ValueError('solver nonzero magnitude below1e-150')
    norm, t = _solve_many(_factor(np.array(J, dtype=float, copy=True)), np.asarray(delta, dtype=float)[None, :])
    return float(norm[0]), t[0].copy()


class TranslationTangentDissimilarity:
    """Experimental dense grayscale reference, fixed ±1 pixel per tangent axis.

    Accept N,H,W, or N,(H*W) only with explicit image_shape=(H,W).
    Pixel spacing1, centered finite differences, one zero-pixel padding.
    Domain is nonnegative finite float64-representable images, each nonzero
    intensity >=1e-150, default maximum65535 (configurable up to1e150).
    fit retains copies of TRAIN images and derivative factors, but not a dense
    distance matrix. transform requires images; returned arrays are detached.
    Dense TRAIN/query row limits are at most 2000. ``max_matrix_bytes``
    conservatively accounts for three coexisting float64 distance matrices;
    ``max_image_bytes`` counts copied images, factors and blocked workspace,
    including an existing reference during queries/transactional refits.
    ``max_work`` counts directed-candidate-pixel evaluations (5*N*N*P for
    square, 10*Q*N*P for cross distances), not seconds or total process RSS.
    Changing one budget does not waive another. No save/load or checkpoint
    format is provided. Arbitrary large integers are canonicalized to float64,
    not promised lossless integer arithmetic. No generic numeric-data default
    or source-topology/held-out clustering guarantee is implied.
    """
    def __init__(self, *, image_shape=None, max_reference=2000, max_query=2000,
                 max_pixels=4096, max_matrix_bytes=128*1024**2,
                 max_image_bytes=256*1024**2, max_work=4_000_000_000,
                 max_intensity=65535., block_size=128):
        if image_shape is not None:
            if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2:
                raise ValueError('image_shape must be explicit (H,W)')
            image_shape = tuple(_integer(v, 'image dimension', 2) for v in image_shape)
        self.image_shape = image_shape
        self.max_reference = _integer(max_reference, 'max_reference', maximum=2000)
        self.max_query = _integer(max_query, 'max_query', maximum=2000)
        self.max_pixels = _integer(max_pixels, 'max_pixels', minimum=4, maximum=65536)
        self.max_matrix_bytes = _integer(max_matrix_bytes, 'max_matrix_bytes')
        self.max_image_bytes = _integer(max_image_bytes, 'max_image_bytes')
        self.max_work = _integer(max_work, 'max_work')
        self.block_size = _integer(block_size, 'block_size', maximum=4096)
        if isinstance(max_intensity, (bool, np.bool_)) or not isinstance(max_intensity, numbers.Real) or not np.isfinite(max_intensity) or not 0 < max_intensity <= _MAX_VALUE:
            raise ValueError('max_intensity must be finite in (0,1e150]')
        self.max_intensity = float(max_intensity)
        self._state = None

    def _images(self, images, *, query=False, expected=None):
        a = _real_array(images, 'images')
        if a.ndim == 3:
            shape = a.shape[1:]
            if self.image_shape is not None and tuple(shape) != self.image_shape:
                raise ValueError('images disagree with explicit image_shape')
        elif a.ndim == 2 and self.image_shape is not None:
            shape = self.image_shape
            if a.shape[1] != shape[0] * shape[1]:
                raise ValueError('flattened image width mismatch')
        else:
            raise ValueError('require N,H,W or flattened N,P with explicit image_shape')
        if min(shape) < 2 or np.prod(shape, dtype=object) > self.max_pixels:
            raise ValueError('image shape/pixel budget exceeded')
        if expected is not None and tuple(shape) != expected:
            raise ValueError('query shape differs from TRAIN')
        limit = self.max_query if query else self.max_reference
        if len(a) > limit or (not query and len(a) == 0):
            raise ValueError('reference/query row budget or empty TRAIN')
        pixels = int(shape[0] * shape[1])
        # TRAIN x + J + U + factors and a bounded residual/candidate block;
        # conservative array accounting, not a process-memory guarantee.
        estimate = len(a) * pixels * 8 * 6 + min(self.block_size, limit) * pixels * 8 * 8
        # Existing TRAIN state remains live during query or transactional refit.
        if self._state is not None:
            estimate += self._state[0].size * 8 * 6
        if estimate > self.max_image_bytes:
            raise ValueError('image/factor/workspace byte budget exceeded')
        if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > self.max_intensity):
            raise ValueError('images must be finite nonnegative and within max_intensity')
        nonzero = a[a > 0]
        if nonzero.size and np.min(nonzero) < _MIN_POSITIVE:
            raise ValueError('nonzero image intensities below1e-150 unsupported')
        x = np.array(a, dtype=np.float64, order='C', copy=True).reshape(len(a), pixels)
        if not np.isfinite(x).all():
            raise ValueError('float64 conversion overflow')
        return x, tuple(shape)

    @staticmethod
    def _factors(x, shape):
        factors = []
        for image in x.reshape((-1,) + shape):
            p = np.pad(image, 1)
            # Independent explicit centered stencil, exactly np.gradient crop.
            J = np.column_stack(((p[1:-1, 2:] - p[1:-1, :-2]).ravel()/2,
                                 (p[2:, 1:-1] - p[:-2, 1:-1]).ravel()/2))
            factors.append(_factor(J))
        return factors

    def _budget(self, q, n, pixels, *, symmetric=False):
        # Output and directional norm matrices coexist; cap all three matrices.
        if 3 * q * n * 8 > self.max_matrix_bytes:
            raise ValueError('dense matrix byte budget exceeded')
        work = (5 if symmetric else 10) * q * n * pixels
        if work > self.max_work:
            raise ValueError('directed candidate-pixel work budget exceeded')
        return work

    def fit(self, images):
        """Copy and retain TRAIN images/derivatives; return self.

        Accept grayscale (N,H,W), or flattened (N,H*W) with explicit
        ``image_shape``. No labels or learned normalization. A failed fit
        preserves the previous reference. Square feasibility is preflighted,
        although no full dissimilarity matrix is retained or computed here.
        """
        x, shape = self._images(images)
        # fit itself does not allocate a dense matrix, but references must admit
        # the declared square operation within the same published budgets.
        self._budget(len(x), len(x), x.shape[1], symmetric=True)
        factors = self._factors(x, shape)
        x.flags.writeable = False
        for factor in factors:
            for array in factor:
                if isinstance(array, np.ndarray):
                    array.flags.writeable = False
        self._state = (x, shape, tuple(factors))  # transactional commit
        return self

    def _require(self):
        if self._state is None:
            raise RuntimeError('fit must precede transform')
        return self._state

    def _directed(self, a, factors, b):
        result = np.empty((len(a), len(b)))
        for i, factor in enumerate(factors):
            for start in range(0, len(b), self.block_size):
                delta = b[start:start+self.block_size] - a[i]
                result[i, start:start+len(delta)] = _solve_many(factor, delta)[0]
        return result

    def transform(self, query_images):
        """Return independent query-to-TRAIN dissimilarities, shape (Q,N).

        TRAIN columns remain in fit order. Each pair uses only its two images;
        batching/permutation does not estimate query statistics or update fit
        state. Empty queries return (0,N). The output is caller-owned; this is
        not an embedding transform, a pixel warp or an exact translation match.
        """
        train, shape, factors = self._require()
        q, qshape = self._images(query_images, query=True, expected=shape)
        self._budget(len(q), len(train), train.shape[1])
        if not len(q):
            return np.empty((0, len(train)))
        qf = self._factors(q, qshape)
        forward = self._directed(q, qf, train)
        reverse = self._directed(train, factors, q).T
        out = np.hypot(forward, reverse)
        out /= np.sqrt(2.)
        if not np.isfinite(out).all() or np.any(((forward > 0) | (reverse > 0)) & (out == 0)):
            raise FloatingPointError('symmetric norm overflow/underflow')
        return out

    def fit_transform(self, images):
        """Fit and return a detached symmetric TRAIN dissimilarity matrix.

        The diagonal is exactly zero. Off-diagonal zeros/duplicates are allowed;
        no metric or triangle-inequality certificate is returned. Failure leaves
        any previous fitted reference intact.
        """
        # Entire operation transactional: a failed matrix computation must not
        # replace a previous usable reference.
        old = self._state
        try:
            self.fit(images)
            x, shape, factors = self._require()
            directed = self._directed(x, factors, x)
            result = np.hypot(directed, directed.T)
            result /= np.sqrt(2.)
            if not np.isfinite(result).all() or np.any((directed > 0) & (result == 0)):
                raise FloatingPointError('symmetric norm overflow/underflow')
            np.fill_diagonal(result, 0.)
            return result
        except BaseException:
            self._state = old
            raise

    @property
    def reference_images_(self):
        x, shape, _ = self._require()
        return x.reshape((-1,) + shape).copy()

    @property
    def n_features_in_(self):
        return self._require()[0].shape[1]

    @property
    def n_reference_(self):
        return len(self._require()[0])

    @property
    def image_shape_(self):
        return self._require()[1]

    @property
    def diagnostics_(self):
        x, shape, _ = self._require()
        return dict(reference_rows=len(x), image_shape=list(shape), bound_pixels=1.,
                    derivative='centered unit-grid with zero exterior padding',
                    solver='direct-J scaled SVD, five box active patterns, actual scaled residual norms',
                    rank_cutoff='eps * max(pixels,2) * largest singular value',
                    metric=False, triangle_inequality=False, exact_translation_invariance=False,
                    retained_data='copied TRAIN images, derivatives and SVD factors; sensitive',
                    dense_reference_matrix_retained=False, persistence='not provided', experimental=True)
