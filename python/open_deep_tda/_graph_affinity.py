"""Shared local graph affinities, with a scale-safe bandwidth repair.

Ordinary historical results retain their exact computation/order. A log-domain
root solve is used only when the finite bandwidth search misses its row-mass
contract. This repairs numerical scale behavior, not semantic clustering.
"""
import numpy as np


def weights(distances):
    raw = np.asarray(distances)
    if (raw.ndim != 2 or raw.shape[1] < 1 or raw.dtype.kind not in 'iuf'
            or np.ma.isMaskedArray(distances)):
        raise ValueError('distances must be a nonempty-column real matrix')
    d = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError('neighbor distances must be finite and nonnegative')
    if len(d) == 0:
        return np.empty_like(d)
    target = np.log2(d.shape[1])
    rho = np.min(np.where(d > 0, d, np.inf), axis=1)
    rho[~np.isfinite(rho)] = 0.
    gap = np.maximum(d-rho[:, None], 0)
    lo = np.zeros(len(d))
    hi = np.maximum(d.max(axis=1), 1e-12)
    saturated = (gap == 0).sum(axis=1) >= target
    # Keep this historical ordinary-scale path byte-compatible. Intermediate
    # overflow/underflow cannot escape: failed mass equations are repaired.
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        for _ in range(64):
            mid = (lo+hi)/2
            mass = np.exp(-gap/mid[:, None]).sum(axis=1)
            lo = np.where(mass < target, mid, lo)
            hi = np.where(mass >= target, mid, hi)
        w = np.exp(-gap/hi[:, None])
    w[saturated] = gap[saturated] == 0
    tolerance = 64*np.finfo(float).eps*d.shape[1]
    repair = (~saturated) & ((~np.isfinite(w).all(axis=1)) |
                             (np.abs(w.sum(axis=1)-target) > tolerance))
    for row in np.flatnonzero(repair):
        positive = gap[row] > 0
        logs = np.log(gap[row, positive])
        zeros = int((~positive).sum())
        required = target-zeros
        count = len(logs)
        # Lower mass <= zeros + required/2; upper mass >= target.
        left = float(logs.min()-np.log(-np.log(required/(2*count))))
        right = float(logs.max()-np.log(-np.log(required/count)))

        def affinities(log_sigma):
            with np.errstate(over='ignore', under='ignore'):
                return np.exp(-np.exp(logs-log_sigma))

        for _ in range(80):
            middle = (left+right)/2
            if affinities(middle).sum()+zeros < target:
                left = middle
            else:
                right = middle
        w[row] = 1.
        w[row, positive] = affinities((left+right)/2)
        if not np.isfinite(w[row]).all() or abs(w[row].sum()-target) > 1e-11:
            raise RuntimeError('local bandwidth mass could not be resolved')
    return w
