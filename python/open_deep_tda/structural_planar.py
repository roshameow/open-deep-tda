"""Sufficient certificate for selected target planar Vietoris-Rips H1 classes.

Original implementation of standard empty-disk geometry, winding modulo two,
and finite-field linear algebra; no mathematical novelty is claimed.
The public API is Limits, Certificate, ResourceLimitError, certify, and
squared_distance_bounds. Import them from open_deep_tda.structural_planar;
no package-root re-export or alternate certificate aliases are required.

`open_deep_tda.structural_planar.certify(points, cycles, birth, survival, holes, limits=...)`
accepts finite NumPy **float64** arrays `points: (N,2)`, `holes: (H,2)` and a sized
sequence of F2 edge chains. An edge is a pair of distinct integer **global row
indices into points**. Nonnegative finite radii must be exactly representable
as float64 (wider floating dtypes are rejected) and satisfy `birth <= survival`.
Every supplied edge, including duplicate edges
that cancel, is required to have real Euclidean length at most `birth`.
Unoriented duplicate edges cancel over F2 for closure and winding; each resulting
chain must have zero vertex boundary. Disconnected and self-crossing chains are
allowed. Empty/zero selected families are deliberately not certified.

A positive result certifies precisely:

* the submitted cycles exist in the **target** Vietoris–Rips complex at birth;
* their classes are linearly independent over F2 at survival, hence at every
  radius in the closed interval `[birth, survival]`.

It checks **every target row against every hole**, not just chain vertices or a
sample. The caller must supply the entire intended target domain and the correct
row identity mapping. No API can infer omitted rows or identify a plausible but
wrong in-range ID as an identity error. Such IDs define a different submitted
chain; only that chain can be certified.

There is **no source validity, source-to-target map, all-H1, barcode equality,
isomorphism, H0, out-of-sample or embedding/injectivity claim**. These sufficient
conditions can fail for genuinely independent cycles. `certified=False` means
not proved, not dependent. Malformed inputs raise `ValueError`; budget refusal
raises `ResourceLimitError` (a subclass). There is never a resource-partial pass.
Do not mutate input arrays/sequences concurrently with verification.

## Mathematical proof (real coordinates, closed VR thresholds)

Write `b = survival`. For a triangle with vertices `v1,v2,v3` of pairwise
distance at most b and any point x in its closed convex hull, choose barycentric
weights `lambda_i >= 0`, `sum lambda_i = 1`, `x = sum lambda_i v_i`. Then

```
sum_i lambda_i ||v_i-x||²
  = sum_{i<j} lambda_i lambda_j ||v_i-v_j||²
  <= b² (1 - sum_i lambda_i²)/2
  <= b²/3.
```

At least one positive-weight vertex therefore has distance at most `b/sqrt(3)`
from x. This includes degenerate triangles and their boundary. For a segment of
length at most b the same two-weight argument gives a vertex within `b/2` of any
point on it. This is a nearest-vertex statement, not a claim about where the
circumcenter lies.

For each hole h, require **every point p to satisfy strictly**
`||p-h|| > b/sqrt(3)`. Thus no VR edge or filled VR triangle at b contains h.
The affine realization of the VR 2-skeleton consequently lies in the plane
punctured at the holes. Winding about each hole modulo two is an F2 linear
functional on closed edge chains. It is zero on each triangle boundary: the
filled affine triangle avoids the hole and supplies its null-homology. It
therefore annihilates every sum of triangle boundaries. Degenerate triangles
are covered as well; higher simplices do not change H1.

Form `W[hole, cycle]` from winding parities. If W has full **column** rank k,
no nonzero linear combination of the k cycles can be a VR boundary at b.
Their classes are independent there. All required edges occur by birth; a
boundary relation at an earlier radius would also be one at b. This proves the
entire selected-family interval claim. Holes need not be distinct, but repeated
rows cannot manufacture rank. The geometric realization need not be injective.

## Numerical guarantee and deliberate limitations

The theorem is about **real Euclidean geometry of the represented binary64
coordinates and represented radii**. It is not about hypothetical pre-rounding
coordinates or a library's rounded `pdist` distance matrix.

1. Subtraction, squaring, addition and the radius-square/division operations
   use outward float64 intervals. Each elementary result is expanded with
   `nextafter` toward the appropriate infinity. Squares use interval endpoint
   minima/maxima, with zero included when the difference interval crosses zero.
   Known exact zero differences are preserved. Nonnegative lower bounds are
   clamped at zero. No norm/sqrt/BLAS reduction or numerical epsilon is used for
   certification.
2. Required edges pass if their squared-distance upper bound is at most the
   birth-square lower bound. A definite violation fails. An ambiguous comparison
   falls back to exact dyadic integer arithmetic; equality at birth is allowed.
3. Disk protection requires `lower(distance²) > upper(b²/3)`, with **strict**
   inequality. `sqrt(3)` is never rounded in this decision. An inconclusive
   interval refuses certification, even if the exact condition is true. There
   is no unbounded exact fallback over the all-vertex scan. Underflow, overflow
   and extremely close boundaries can therefore yield conservative refusal.
4. Chain vertices and holes are converted by `float.as_integer_ratio()` to
   integers at a common power-of-two scale. A half-open horizontal ray uses
   `(ay > hy) != (by > hy)` and the **exact integer determinant**
   `(bx-ax)*(hy-ay) - (by-ay)*(hx-ax)`. Its sign, together with the sign of
   `by-ay`, decides a strict right-ray crossing. Reversing an edge changes both
   signs and leaves parity unchanged. Horizontal edges and vertex crossings
   follow the half-open convention. Determinant zero on a straddling edge fails
   defensively (it cannot occur after successful disk protection).
5. Matrix rank is exact F2 integer-XOR elimination. No floating-point rank test.

The interval proof assumes IEEE-754 binary64 elementary operations with gradual
underflow, correctly rounded basic arithmetic and functioning `nextafter`, as
provided by the tested NumPy environment. It is not hardware/floating-point-mode
attestation; unsupported flush-to-zero or nonconforming arithmetic is outside
scope. Tests compare bounds to independent exact `Fraction` values, including
subnormal, overflow and random-bit inputs. Exact integer coordinates have at
most 2098 magnitude bits at the common scale; determinant arithmetic is bounded
by roughly 4200 bits, not arbitrary-size user integers.

### Rounded distance complexes are a separate contract

A rounded library distance can classify a threshold-edge differently. **This
API does not certify exact equivalence to a rounded `pdist` complex** and does
not assume an undocumented relative-error constant. To use the argument for
such a complex, independently establish that every selected edge is included
by its actual birth rule, and obtain a validated global real-length upper bound
B for **every edge included by its survival rule**. Protect disks using B and
ensure the selected real edges lie within the real birth threshold used here
(no larger than B). Then all triangles of that rounded complex also avoid the
holes, so the same winding proof applies. Establishing these guards is the
caller's separate obligation; it is not implemented or asserted by this API.
An arbitrary tolerance or rounded minimum-clearance diagnostic is not a guard.

## Bounded work, budget first

`Limits` may lower, never raise, the hard ceilings:

| Quantity | Hard ceiling / default |
|---|---:|
| N | 1,000,000 |
| holes H | 32 |
| cycle columns k | 32 |
| total **supplied** chain edges E, before cancellation | 100,000 |
| reserved distance pairs | N*H + E <= 32,100,000 |
| reserved orientation opportunities | H*E <= 3,200,000 |
| reserved exact required-edge fallbacks | E <= 100,000 |
| point/edge chunk size | 65,536 |

Sizes and all reservations are checked before finite-coordinate scans, chain
allocation or geometry. Inputs must already be sized arrays/sequences; unlimited
generators are rejected. Even if many checks would cancel, or an early failure
would occur, an over-budget request is refused first. Limits bound units of
algorithmic work, **not total process RSS or wall-clock time**. A negative result
may stop early; a positive result has checked all required pairs and all winding
columns. The report distinguishes reserved work from performed work.

Runtime is `O(NH + E + HE + E log E)` (sorting chain supports), plus bounded
exact-integer costs and tiny rank elimination. Finite validation scans N points.
Extra memory is `O(E + Hk + chunk_size)`, besides caller-owned input; there is no
`N*N` matrix and **no edge-graph or triangle enumeration**. Only chain vertices
need exact-coordinate conversion, never the million-row complement.
"""
from dataclasses import dataclass, field
from numbers import Integral, Real
import math
import numpy as np


__all__ = [
    "Limits", "Certificate", "ResourceLimitError", "certify",
    "squared_distance_bounds",
]


class ResourceLimitError(ValueError):
    """Whole request refused: never a partial certificate."""


@dataclass(frozen=True)
class Limits:
    """Work ceilings; callers may lower defaults but never raise them.

    Reservations count supplied edges before F2 cancellation. These limits
    bound algorithmic work, not process memory or elapsed time.
    """

    max_vertices: int = 1_000_000
    max_holes: int = 32
    max_cycles: int = 32
    max_chain_edges: int = 100_000
    max_distance_pairs: int = 32_100_000
    max_orientation_tests: int = 3_200_000
    max_exact_edge_tests: int = 100_000
    chunk_size: int = 65_536


@dataclass(frozen=True)
class Certificate:
    """Selected-family result, not a full persistence diagram.

    A false result means not proved, not dependent. Winding rows index holes
    and columns index supplied chains; work reports reservations and actual
    operations, with possible early exit on failure.
    """

    certified: bool
    reason: str
    winding: tuple = ()                 # rows = holes, columns = supplied chains
    rank: int = 0
    work: dict = field(default_factory=dict)
    scope: str = "selected target H1 only; real Euclidean represented float64 coordinates"


_HARD = Limits()


def _array_shape(a, name):
    # Do not silently copy/coerce unbounded iterables before budget checks.
    if not isinstance(a, np.ndarray) or a.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be a float64 ndarray")
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")


def _preflight(points, cycles, holes, limits):
    if not isinstance(limits, Limits):
        raise ValueError("limits must be Limits")
    for name, hard in vars(_HARD).items():
        value = getattr(limits, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= hard:
            raise ValueError(f"{name} must be an integer in [0, {hard}]")
    if limits.chunk_size == 0:
        raise ValueError("chunk_size must be positive")
    _array_shape(points, "points")
    _array_shape(holes, "holes")
    if not isinstance(cycles, (list, tuple, np.ndarray)):
        raise ValueError("cycles must be a sized list/tuple/ndarray of edge chains")
    if isinstance(cycles, np.ndarray) and cycles.ndim == 0:
        raise ValueError("cycles must be a sequence")
    n, h, k = len(points), len(holes), len(cycles)
    for name, count, cap in (("vertices", n, limits.max_vertices),
                             ("holes", h, limits.max_holes),
                             ("cycles", k, limits.max_cycles)):
        if count > cap:
            raise ResourceLimitError(f"{name}: {count} > {cap}")
    e = 0
    for chain in cycles:
        if not isinstance(chain, (list, tuple, np.ndarray)):
            raise ValueError("each chain must be a sized edge sequence")
        if isinstance(chain, np.ndarray) and (chain.ndim != 2 or chain.shape[1] != 2):
            raise ValueError("chain arrays must have shape (E, 2)")
        e += len(chain)
        if e > limits.max_chain_edges:
            raise ResourceLimitError("total supplied chain edges exceed budget")
    for name, count, cap in (("distance pairs", n*h + e, limits.max_distance_pairs),
                             ("orientation tests", e*h, limits.max_orientation_tests),
                             ("exact edge tests", e, limits.max_exact_edge_tests)):
        if count > cap:
            raise ResourceLimitError(f"{name}: reserved {count} > {cap}")
    return n, h, k, e


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def squared_distance_bounds(a, b):
    """Outward binary64 enclosure of exact squared 2D distances.

    Each elementary subtract/multiply/add is expanded by one nextafter; no
    BLAS, norm, sqrt, fused reduction, or tolerance. IEEE-754 gradual underflow
    and correctly rounded binary64 elementary operations are prerequisites.
    Overflow widens to infinity and can cause conservative non-certification.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        d = a - b
        lo, hi = _down(d), _up(d)
        # Equal represented coordinates have exactly zero difference.
        equal = (a == b)
        lo = np.where(equal, 0.0, lo)
        hi = np.where(equal, 0.0, hi)
        sl = np.maximum(0.0, _down(np.minimum(lo*lo, hi*hi)))
        sl = np.where((lo <= 0) & (hi >= 0), 0.0, sl)
        su = _up(np.maximum(lo*lo, hi*hi))
        su = np.where(equal, 0.0, su)
        lower = np.maximum(0.0, _down(sl[..., 0] + sl[..., 1]))
        upper = _up(su[..., 0] + su[..., 1])
        zero = np.all(equal, axis=-1)
        return np.where(zero, 0.0, lower), np.where(zero, 0.0, upper)


def _square_bounds(x):
    if x == 0.0:
        return 0.0, 0.0
    with np.errstate(over="ignore", under="ignore"):
        square = np.float64(x) * np.float64(x)
        return float(max(0.0, _down(square))), float(_up(square))


def _dyadic(x):
    numerator, denominator = float(x).as_integer_ratio()
    return numerator, denominator.bit_length() - 1


def _exact_edge_leq(a, b, radius):
    """Exact integer fallback only for interval-ambiguous required edges."""
    parts = [_dyadic(x) for x in (*a, *b, radius)]
    exponent = max(e for _, e in parts)
    ax, ay, bx, by, r = [m << (exponent-e) for m, e in parts]
    return (ax-bx)**2 + (ay-by)**2 <= r*r


def _integer_geometry(points, holes, vertices):
    # Only chain vertices, never the million-point complement, need exact
    # conversion. One common power-of-two scaling preserves determinant sign.
    vp = {v: (_dyadic(points[v, 0]), _dyadic(points[v, 1])) for v in vertices}
    hp = [(_dyadic(x), _dyadic(y)) for x, y in holes]
    exponent = max((e for pair in list(vp.values()) + hp for _, e in pair), default=0)
    def convert(pair):
        return tuple(m << (exponent-e) for m, e in pair)
    return {v: convert(pair) for v, pair in vp.items()}, [convert(pair) for pair in hp]


def _f2_rank(rows):
    pivots = {}
    for row in rows:
        word = sum(int(bit) << j for j, bit in enumerate(row))
        while word:
            pivot = word.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = word
                break
            word ^= pivots[pivot]
    return len(pivots)


def certify(points, cycles, birth, survival, holes, *, limits=Limits()):
    """Certify independence of supplied F2 chains in target H1 through survival.

    IDs are positional GLOBAL rows of points, never cycle-local IDs. Every
    supplied edge (even a duplicate cancelling over F2) must exist at birth.
    Chains are unoriented; duplicate undirected edges cancel for closure and
    winding. Self-edges and malformed/open chains raise ValueError. No rows may
    be omitted from points. Caller owns the identity/domain assertion, and must
    not mutate inputs concurrently. An in-range wrong ID cannot be recognized
    as an identity error; the actual submitted chain is what is certified.

    Budget preflight precedes coordinate scans, chain allocation, or geometry.
    Invalid input/resource refusal raises; certified=False is NOT a proof of
    dependence. Never returns a resource-partial success.
    """
    n, h, k, e = _preflight(points, cycles, holes, limits)
    radii = []
    for value in (birth, survival):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or value < 0:
            raise ValueError("radii must be finite nonnegative real scalars")
        if isinstance(value, np.floating) and value.dtype.itemsize > 8:
            raise ValueError("radii cannot have precision wider than float64")
        try:
            r = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError("radius cannot be represented as finite float64") from exc
        if not math.isfinite(r) or r < 0:
            raise ValueError("radii must be finite nonnegative float64")
        if (isinstance(value, Integral) and int(r) != int(value)) or (
                not isinstance(value, Integral) and value != r):
            raise ValueError("radii must be exactly representable as float64")
        radii.append(r)
    birth, survival = radii
    if birth > survival:
        raise ValueError("birth must not exceed survival")
    chunk = limits.chunk_size
    for start in range(0, n, chunk):
        if not np.isfinite(points[start:start+chunk]).all():
            raise ValueError("points must all be finite")
    if not np.isfinite(holes).all():
        raise ValueError("holes must all be finite")

    raw = np.empty((e, 2), dtype=np.int64)
    chains, vertices, cursor = [], set(), 0
    for chain in cycles:
        support, odd = set(), set()
        for edge in chain:
            if (not isinstance(edge, (list, tuple, np.ndarray))
                    or (isinstance(edge, np.ndarray) and edge.shape != (2,))
                    or len(edge) != 2):
                raise ValueError("each edge must contain two integer global row IDs")
            u, v = edge
            if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in (u, v)):
                raise ValueError("edge IDs must be integers, not floats/bools")
            u, v = int(u), int(v)
            if not (0 <= u < n and 0 <= v < n) or u == v:
                raise ValueError("edge IDs out of range or self-edge")
            edge = (min(u, v), max(u, v))
            raw[cursor] = edge
            cursor += 1
            support.symmetric_difference_update((edge,))
            odd.symmetric_difference_update((u, v))
        if odd:
            raise ValueError("chain is not an F2 cycle (odd boundary vertices)")
        chains.append(sorted(support))
        for u, v in support:
            vertices.update((u, v))
    work = dict(vertices=n, holes=h, cycles=k, supplied_chain_edges=e,
                reserved_distance_pairs=n*h+e, reserved_orientation_tests=e*h,
                reserved_exact_edge_tests=e, distance_pairs=0,
                orientation_tests=0, exact_edge_tests=0, chunk_size=int(chunk),
                triangles_enumerated=0)
    def fail(reason, winding=(), rank=0):
        return Certificate(False, reason, winding, rank, dict(work))
    if not n or not h or not k or any(not chain for chain in chains):
        return fail("nonempty domain, holes and nonzero cycles required")
    if k > h:
        return fail("fewer holes than cycle columns; full column rank impossible")

    birth_lo, birth_hi = _square_bounds(birth)
    for start in range(0, e, chunk):
        edges = raw[start:start+chunk]
        lo, hi = squared_distance_bounds(points[edges[:, 0]], points[edges[:, 1]])
        work["distance_pairs"] += len(edges)
        for index in np.flatnonzero(~(hi <= birth_lo)):
            if lo[index] > birth_hi:
                return fail("required edge exceeds birth")
            u, v = edges[index]
            work["exact_edge_tests"] += 1
            if not _exact_edge_leq(points[u], points[v], birth):
                return fail("required edge exceeds birth (exact fallback)")

    # Compare d^2 > b^2/3 without rounding sqrt(3). A strict lower-vs-upper
    # comparison suffices. No exact fallback for the N*H complement scan.
    _, survival_hi = _square_bounds(survival)
    with np.errstate(over="ignore", under="ignore"):
        protected_squared_upper = float(_up(np.float64(survival_hi) / np.float64(3)))
    if survival == 0:
        protected_squared_upper = 0.0
    work["protected_squared_upper"] = protected_squared_upper
    minimum_lower = math.inf
    for hole in holes:
        for start in range(0, n, chunk):
            lo, _ = squared_distance_bounds(points[start:start+chunk], hole)
            work["distance_pairs"] += len(lo)
            minimum_lower = min(minimum_lower, float(lo.min()))
            if not np.all(lo > protected_squared_upper):
                work["minimum_checked_squared_clearance_lower"] = minimum_lower
                return fail("strict protected disk clearance not proved for every target row")
    work["minimum_squared_clearance_lower"] = minimum_lower

    integer_points, integer_holes = _integer_geometry(points, holes, vertices)
    matrix = []
    for hx, hy in integer_holes:
        row = []
        for chain in chains:
            parity = 0
            for u, v in chain:
                ax, ay = integer_points[u]
                bx, by = integer_points[v]
                # Half-open horizontal ray convention handles vertex crossings.
                if (ay > hy) != (by > hy):
                    work["orientation_tests"] += 1
                    det = (bx-ax)*(hy-ay) - (by-ay)*(hx-ax)
                    if det == 0:
                        return fail("hole lies on a chain edge")
                    if (det > 0) == (by > ay):
                        parity ^= 1
            row.append(parity)
        matrix.append(tuple(row))
    matrix = tuple(matrix)
    rank = _f2_rank(matrix)
    if rank != k:
        return fail("hole-versus-cycle winding matrix lacks full column rank", matrix, rank)
    return Certificate(True, "certified selected target family through survival", matrix, rank, dict(work))
