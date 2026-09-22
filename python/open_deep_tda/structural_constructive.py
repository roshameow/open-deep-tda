"""Bounded all-row structural construction for a selected H1 family.

The complete source distance matrix defines a closed-threshold flag complex;
row i must denote the SAME vertex in source, guide, cycles and target. No rows
are sampled or silently restricted to the witness. The sole supported family
is one simple cycle. This is a sufficient constructive procedure, not a general
feasibility solver, all-H1 preservation theorem, barcode equality or chain map.

The full minimax hierarchy U is computed before any seed restriction. Contract
it to W=max(0,U-delta), with explicit delta=.98*h0_tolerance. Positive cycle
chords propose a minor-arc circle, checked for hierarchy compatibility. Analytic
contact-circle intersections and guide-directed angles propose all remaining
rows while avoiding existing exclusion disks and a protected hole. No angular
grid, jitter, data-specific branch, optimizer or random state is used.

All candidate-search tolerances are provisional only. Acceptance recomputes
ALL same-ID source/target minimax distances, uses the UNCHANGED H0 tolerance,
and requires fresh full-domain source AND target sparse GF2 H1 certification.
The target matrix is explicitly NumPy hypot of float64 coordinate differences.
The additional planar check is about REAL Euclidean geometry of represented
float64 coordinates, not exact equivalence to rounded pdist/cdist filtration.
Its geometric proof is in structural_planar; it does not replace matrix checks.

Invalid inputs raise ValueError; budget exhaustion raises ResourceLimitError.
Unsupported, numerical or failed-final-check results expose embedding=None and
hole=None, never a partial embedding. Failure is not mathematical infeasibility.
Logical work budgets are not total RSS or wall-time guarantees. BLAS/LAPACK
may not honor NumPy errstate: explicit finite checks on derived Procrustes
arrays turn silent overflow/NaNs into unsupported outcomes, not malformed-input
errors or accidental certificates. Input arrays are not modified; callers must
not mutate them during verification. Successful
arrays are read-only and owned by the result. Standard mathematical algorithms
are implemented independently; no novelty or external-author equivalence claim.
"""
from dataclasses import dataclass, field
from numbers import Integral, Real
import math
import numpy as np
from scipy.optimize import brentq

from .structural_sparse_h1 import H1Limits, ResourceLimitError, analyze_sparse_h1
from .structural_planar import certify as planar_certify, squared_distance_bounds


__all__ = ['LayoutLimits', 'LayoutResult', 'construct_global_layout',
           'full_hierarchy', 'pairwise_planar']


@dataclass(frozen=True)
class LayoutLimits:
    """Hard constructor caps, lowerable only; zero denies the corresponding work.

    Source/target H1 calls have their own H1Limits. Matrix shape is checked before
    scans; contact budgets are charged before candidate/constraint work. These
    limits bound logical work, not total Python/library RSS or elapsed time.
    """
    max_vertices: int = 1024
    max_matrix_entries: int = 1_048_576
    max_parent_tests: int = 100_000
    max_candidate_points: int = 2_000_000
    max_contact_pairs: int = 200_000_000
    candidate_chunk: int = 256


@dataclass(frozen=True)
class LayoutResult:
    """Outcome for the selected same-ID family, not all source/target H1.

    A failure has no embedding or hole, even if construction reached a partial
    or complete candidate. Diagnostics and completed checks remain available.
    Resource exhaustion raises instead of returning this object.
    """
    certified: bool
    embedding: object = None
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)
    source: object = None
    target: object = None
    h0_error: object = None
    planar: object = None
    hole: object = None


class _Unsupported(RuntimeError):
    pass


def _limits(limits):
    if not isinstance(limits, LayoutLimits):
        raise ValueError("limits must be LayoutLimits")
    for key, hard in vars(LayoutLimits()).items():
        v = getattr(limits, key)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or not 0 <= v <= hard:
            raise ValueError(f"{key} must be integer in [0, {hard}]")
    if not limits.candidate_chunk:
        raise ValueError("candidate_chunk must be positive")


def _matrix(D, max_vertices=1024):
    if not isinstance(D, np.ndarray) or D.dtype != np.dtype(np.float64) or D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square float64 ndarray")
    if len(D) > max_vertices:
        raise ResourceLimitError("vertex budget exceeded before matrix scan")
    if not np.isfinite(D).all() or np.any(D < 0) or np.any(np.diag(D) != 0) or not np.array_equal(D, D.T):
        raise ValueError("D must be finite, symmetric, nonnegative with zero diagonal")
    return D


def _radius(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    original = value
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        raise ValueError(f"{name} cannot have precision wider than float64")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}")
    if (isinstance(original, Integral) and int(value) != int(original)) or (
            not isinstance(original, Integral) and original != value):
        raise ValueError(f"{name} must be exactly representable as float64")
    return value


def _mst(D):
    """Deterministic dense Prim with canonical-edge ties; zero edges retained."""
    n = len(D)
    if n < 2:
        return []
    chosen = np.zeros(n, dtype=bool)
    chosen[0] = True
    best = D[0].copy()
    parent = np.zeros(n, dtype=np.int64)
    edges = []
    for _ in range(n-1):
        remaining = np.flatnonzero(~chosen)
        v = min(remaining, key=lambda v: (best[v], min(int(parent[v]), int(v)), max(int(parent[v]), int(v))))
        u = int(parent[v]); v = int(v)
        edges.append((min(u, v), max(u, v)))
        chosen[v] = True
        for j in np.flatnonzero(~chosen):
            old = (min(int(parent[j]), int(j)), max(int(parent[j]), int(j)))
            new = (min(v, int(j)), max(v, int(j)))
            if D[v, j] < best[j] or (D[v, j] == best[j] and new < old):
                best[j], parent[j] = D[v, j], v
    return edges


def full_hierarchy(D):
    """All-pair minimax merge distances; no sampling, metric assumption or eps."""
    D = _matrix(D)
    n = len(D)
    U = np.zeros_like(D)
    groups = {i: [i] for i in range(n)}
    owner = list(range(n))
    for u, v in sorted(_mst(D), key=lambda e: (D[e], e)):
        i, j = owner[u], owner[v]
        left, right = groups[i], groups[j]
        U[np.ix_(left, right)] = U[np.ix_(right, left)] = D[u, v]
        for vertex in right:
            owner[vertex] = i
        left.extend(right)
        del groups[j]
    return U


def pairwise_planar(points):
    """Rounded float64 target edge weights: explicit hypot(subtractions).

    This matrix, not unspecified scipy/pdist rounding, defines the target checked
    by sparse H1 and H0. The additional planar certificate uses real coordinates.
    """
    if not isinstance(points, np.ndarray) or points.dtype != np.dtype(np.float64) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be (N,2) float64")
    if len(points) > 1024:
        raise ResourceLimitError("pairwise target vertex budget")
    if not np.isfinite(points).all():
        raise ValueError("nonfinite target points")
    with np.errstate(over='raise', invalid='raise'):
        dx = points[:, 0, None] - points[None, :, 0]
        dy = points[:, 1, None] - points[None, :, 1]
        D = np.hypot(dx, dy)
    if not np.isfinite(D).all():
        raise FloatingPointError("unrepresentable target distances")
    return D


def _simple_order(cycles, n):
    if not isinstance(cycles, (list, tuple, np.ndarray)):
        raise ValueError("cycles must be a sized sequence")
    if len(cycles) != 1:
        raise _Unsupported("unsupported family: exactly one simple cycle required")
    chain = cycles[0]
    if not isinstance(chain, (list, tuple, np.ndarray)):
        raise ValueError("cycle must be a sized edge sequence")
    if len(chain) > 100_000:
        raise ResourceLimitError("chain edge budget")
    adjacency = {}
    seen = set()
    for edge in chain:
        if not isinstance(edge, (list, tuple, np.ndarray)) or (isinstance(edge, np.ndarray) and edge.shape != (2,)) or len(edge) != 2:
            raise ValueError("edge must contain two IDs")
        u, v = edge
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in (u, v)):
            raise ValueError("edge IDs must be integers")
        u, v = int(u), int(v)
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError("edge IDs out of range or self-edge")
        key = tuple(sorted((u, v)))
        if key in seen:
            raise _Unsupported("unsupported family: repeated edge, not a simple cycle")
        seen.add(key)
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    if len(adjacency) < 3 or any(len(a) != 2 for a in adjacency.values()):
        raise _Unsupported("unsupported family: single simple cycle required")
    start = min(adjacency)
    seq, previous, current = [start], None, start
    while True:
        nxt = min(v for v in adjacency[current] if v != previous)
        if nxt == start:
            break
        if nxt in seq:
            raise _Unsupported("unsupported family: disconnected cycle")
        seq.append(nxt)
        previous, current = current, nxt
    if len(seq) != len(adjacency):
        raise _Unsupported("unsupported family: disconnected cycle union")
    return seq


class _ContactBudget:
    def __init__(self, limits, counters):
        self.limits, self.counters = limits, counters

    def charge(self, key, amount, cap):
        if self.counters[key] + amount > getattr(self.limits, cap):
            raise ResourceLimitError(f"{key} work budget exhausted; no embedding returned")
        self.counters[key] += amount


def _closest_contact(center, radius, centers, radii, guide, slack, budget):
    """Analytic candidates, not an angular grid. Slack only proposes candidates."""
    budget.charge('parent_tests', 1, 'max_parent_tests')
    delta = centers-center
    d = np.hypot(delta[:, 0], delta[:, 1])
    if np.any(d + radius < radii - slack):
        return None
    if radius == 0:
        # Zero insertion radii are supported exactly; zero seed chords are not.
        budget.charge('candidate_points', 1, 'max_candidate_points')
        budget.charge('contact_pairs', len(centers), 'max_contact_pairs')
        candidates = center.reshape(1, 2).copy()
    else:
        relevant = (d > 0) & (d <= radius+radii+slack) & (d+np.minimum(radius, radii) >= np.maximum(radius, radii)-slack)
        dd, rr, vv = d[relevant], radii[relevant], delta[relevant]
        # Normalize each contact triangle before squaring to avoid overflow.
        scale = np.maximum(np.maximum(dd, rr), radius)
        ds, rs, ps = dd/scale, rr/scale, radius/scale
        cosine = (ps*ps + ds*ds - rs*rs)/(2*ps*ds)
        keep = np.abs(cosine) <= 1 + 128*np.finfo(float).eps
        phi = np.arctan2(vv[keep, 1], vv[keep, 0])
        theta = np.arccos(np.clip(cosine[keep], -1., 1.))
        angles = np.r_[np.arctan2(guide[1]-center[1], guide[0]-center[0]), phi+theta, phi-theta]
        budget.charge('candidate_points', len(angles), 'max_candidate_points')
        budget.charge('contact_pairs', len(angles)*len(centers), 'max_contact_pairs')
        candidates = center + radius*np.column_stack([np.cos(angles), np.sin(angles)])
    best, best_cost = None, math.inf
    for first in range(0, len(candidates), budget.limits.candidate_chunk):
        z = candidates[first:first+budget.limits.candidate_chunk]
        diff = z[:, None, :] - centers[None, :, :]
        distances = np.hypot(diff[:, :, 0], diff[:, :, 1])
        valid = np.all(distances >= radii[None, :] - slack, axis=1)
        for point in z[valid]:
            cost = float(np.hypot(*(point-guide)))  # distance, not overflow-prone square
            if cost < best_cost:
                best, best_cost = point.copy(), cost
    return None if best is None else (best_cost, best)


def _circle_lower_bound(guide, center, radius):
    """Conservative real distance-to-circle lower bound for parent pruning."""
    lo, hi = squared_distance_bounds(guide, center)
    with np.errstate(over='ignore', under='ignore'):
        dlo = max(0., float(np.nextafter(np.sqrt(lo), -np.inf)))
        dhi = float(np.nextafter(np.sqrt(hi), np.inf))
        return max(0., float(np.nextafter(dlo-radius, -np.inf)),
                   float(np.nextafter(radius-dhi, -np.inf)))


def _cost_upper(point, guide):
    _, hi = squared_distance_bounds(point, guide)
    with np.errstate(over='ignore', under='ignore'):
        return float(np.nextafter(np.sqrt(hi), np.inf))


def construct_global_layout(D, guide, cycles, a, b, h0_tolerance=.05, *,
                            limits=LayoutLimits(), h1_limits=H1Limits()):
    """Construct or explicitly refuse one selected same-ID global H1 family.

    Success requires fresh full source/target sparse H1, all-pair H0 within the
    UNCHANGED requested tolerance, and additional real-coordinate planar proof.
    Source D may be a complete dissimilarity; triangle inequality is unnecessary.
    Guide is supplied, unnormalized and untrained by this API. Failure returns no
    partial embedding or hole. Unsupported is NOT an infeasibility assertion.
    """
    _limits(limits)
    if not isinstance(D, np.ndarray) or D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be square ndarray")
    n = len(D)
    if n > limits.max_vertices or n*n > limits.max_matrix_entries:
        raise ResourceLimitError("layout dimension budget exceeded before scans")
    if not isinstance(guide, np.ndarray) or guide.dtype != np.dtype(np.float64) or guide.shape != (n, 2):
        raise ValueError("guide must be a matching (N,2) float64 ndarray")
    D = _matrix(D, limits.max_vertices)
    if not np.isfinite(guide).all():
        raise ValueError("guide must be finite")
    a, b, tolerance = _radius(a, 'birth'), _radius(b, 'survival'), _radius(h0_tolerance, 'h0_tolerance')
    if a > b:
        raise ValueError("birth must not exceed survival")
    diagnostics = dict(n_vertices=n, delta=.98*tolerance, h0_tolerance=tolerance,
                       parent_tests=0, candidate_points=0, contact_pairs=0,
                       inserted_vertices=0, scope='all supplied rows; selected family only',
                       target_distance_rule='numpy.hypot of float64 coordinate subtractions')
    source = target = planar = h0_error = None
    def fail(reason):
        return LayoutResult(False, None, reason, dict(diagnostics), source, target, h0_error, planar, None)
    # Validate the complete source contract even for a well-formed family this
    # constructor cannot realize. Malformed IDs/chains are never hidden behind
    # an unsupported-family response.
    source = analyze_sparse_h1(D, cycles, a, b, limits=h1_limits)
    if not source.certified:
        return fail('source selected family not certified: '+source.reason)
    try:
        seq = _simple_order(cycles, n)
    except _Unsupported as exc:
        return fail(str(exc))
    budget = _ContactBudget(limits, diagnostics)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            U = full_hierarchy(D)
            delta = diagnostics['delta']
            W = np.maximum(0., U-delta)
            chords = np.array([W[seq[i], seq[(i+1) % len(seq)]] for i in range(len(seq))])
            if np.any(chords <= 0):
                raise _Unsupported('unsupported zero contracted seed chord; no jitter applied')
            # Center the guide for arithmetic, restore its coordinate frame only
            # after construction; no RNG, normalization, learning or resampling.
            origin = guide[0].copy()
            G = guide-origin
            scale = max(float(D.max()), float(np.max(np.abs(G))), b, float(chords.max()))
            slack = 128*np.finfo(float).eps*scale
            diagnostics['proposal_slack'] = slack
            unit = float(chords.max())
            q = chords/unit
            angle = lambda t: float(np.sum(2*np.arcsin(np.clip(q/(2*t), -1., 1.))) - 2*np.pi)
            if angle(.5) < 0:
                raise _Unsupported('unsupported minor-arc circle chord family')
            root = .5 if angle(.5) == 0 else brentq(angle, .5, float(len(seq)), xtol=4*np.finfo(float).eps, rtol=4*np.finfo(float).eps, maxiter=128)
            radius = unit*root
            if not math.isfinite(radius) or radius <= 0:
                raise _Unsupported('unrepresentable seed circle radius')
            phi = np.r_[0., np.cumsum(2*np.arcsin(np.clip(chords[:-1]/(2*radius), -1., 1.)))]
            z = radius*np.column_stack([np.cos(phi), np.sin(phi)])
            zm, gm = z.mean(axis=0), G[seq].mean(axis=0)
            # BLAS/LAPACK need not honor NumPy's errstate. Validate derived
            # arrays explicitly: overflow may otherwise return NaNs silently
            # on one NumPy/backend version but raise on another.
            covariance = (z-zm).T @ (G[seq]-gm)
            if not np.isfinite(covariance).all():
                raise _Unsupported('unrepresentable Procrustes covariance')
            left, singular, right = np.linalg.svd(covariance)
            if not all(np.isfinite(v).all() for v in (left, singular, right)):
                raise _Unsupported('nonfinite Procrustes factors')
            rotation = left @ right
            hole = -zm @ rotation + gm
            z = (z-zm) @ rotation + gm
            if not np.isfinite(z).all() or not np.isfinite(hole).all():
                raise _Unsupported('unrepresentable aligned seed coordinates')
            seed_hierarchy = full_hierarchy(pairwise_planar(z))
            if float(np.max(np.abs(seed_hierarchy-W[np.ix_(seq, seq)]))) > slack:
                raise _Unsupported('unsupported circle-incompatible seed hierarchy')
            # Strict guard is a candidate-generation margin, never a relaxation
            # of H0 or H1. The final interval certificate is authoritative.
            hole_radius = float(np.nextafter(b/np.sqrt(3.), np.inf)) + max(1e-6*b, 32*slack)
            if np.any(np.hypot(*(z-hole).T) <= hole_radius):
                raise _Unsupported('seed cannot satisfy protected-hole guard')
            Z = np.full((n, 2), np.nan)
            Z[seq] = z
            present = list(seq)
            included = np.zeros(n, dtype=bool)
            included[seq] = True
            nearest = U[:, seq].min(axis=1)
            diagnostics.update(inserted_vertices=len(present), circle_radius=radius, hole_radius=hole_radius)
            while len(present) < n:
                remaining = np.flatnonzero(~included)
                new = int(remaining[np.argmin(nearest[remaining])])
                level = float(nearest[new])
                r = max(0., level-delta)
                parents = [p for p in present if U[new, p] == level]
                # Sorting is heuristic only; pruning below uses an outward-
                # rounded real distance-to-circle lower bound.
                parents.sort(key=lambda p: (abs(float(np.hypot(*(G[new]-Z[p])))-r), p))
                centers = np.vstack([Z[present], hole])
                radii = np.r_[W[new, present], hole_radius]
                best = None
                for parent in parents:
                    lower = _circle_lower_bound(G[new], Z[parent], r)
                    if best is not None and lower > _cost_upper(best[1], G[new]):
                        # Do not break: rounded sorting keys need not preserve
                        # the order of outward lower bounds near a tie.
                        continue
                    choice = _closest_contact(Z[parent], r, centers, radii, G[new], slack, budget)
                    if choice is not None and (best is None or choice[0] < best[0]):
                        best = choice
                if best is None:
                    diagnostics['failed_vertex'] = new
                    raise _Unsupported('contact search failed; no general infeasibility claim')
                Z[new] = best[1]
                present.append(new)
                included[new] = True
                nearest = np.minimum(nearest, U[:, new])
                diagnostics['inserted_vertices'] = len(present)
            Z = Z+origin
            hole = hole+origin
            if not np.isfinite(Z).all() or not np.isfinite(hole).all():
                raise _Unsupported('unrepresentable final coordinates')
            T = pairwise_planar(Z)
            target_U = full_hierarchy(T)
            h0_error = float(np.max(np.abs(U-target_U)))
            # Outward bound rules out a rounded subtraction falsely passing an
            # exact tolerance boundary. Exact equality of U needs no expansion.
            h0_upper = 0. if h0_error == 0 else float(np.nextafter(h0_error, np.inf))
            diagnostics['h0_error_upper'] = h0_upper
            if h0_upper > tolerance:
                raise _Unsupported('full all-pair H0 tolerance not proved (no slack added)')
            target = analyze_sparse_h1(T, cycles, a, b, limits=h1_limits)
            if not target.certified:
                raise _Unsupported('full target selected H1 family not certified: '+target.reason)
            planar = planar_certify(Z, cycles, a, b, hole.reshape(1, 2))
            if not planar.certified:
                raise _Unsupported('additional real-coordinate planar certificate failed: '+planar.reason)
            Z.setflags(write=False)
            hole.setflags(write=False)
            return LayoutResult(True, Z, 'certified full-domain H0 and selected source/target H1',
                                dict(diagnostics), source, target, h0_error, planar, hole)
    except _Unsupported as exc:
        return fail(str(exc))
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return fail('numerical construction unsupported: '+str(exc))
