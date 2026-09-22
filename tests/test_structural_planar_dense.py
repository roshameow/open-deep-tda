"""Independent, bounded real-Euclidean VR certificate checks (original code).

The oracle uses Fraction.from_float on each represented coordinate and radius,
then compares squared distances exactly. It does NOT build a rounded distance
matrix or import certificate geometry/rank helpers. Dense uint8 elimination of
triangle boundary columns computes dim(span(B_1, z_i)) - dim(B_1) over GF(2).
This is the rank of the supplied classes, not merely the ambient Betti number.
Only the 2-skeleton is needed for H_1. The VR convention is distance <= radius.

Reproducible numerical pitfall covered below: math.sqrt(2) is above the exact
square diagonal, while its float64 predecessor is below it. Rounded norms can
therefore silently implement a different filtration. Additional fixtures cover
external vertices filling a cycle and distinct chains representing one class.
All random tests have fixed seeds and tiny synthetic, network-free inputs.

Run: python3.9 -B -m pytest -q -p no:cacheprovider tests/test_structural_planar_dense.py
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math

import numpy as np
import pytest


@dataclass(frozen=True)
class DenseResult:
    birth_valid: bool
    closed: bool
    boundary_rank: int
    class_rank: int
    betti_one: int
    edge_count: int
    triangle_count: int
    independent: bool


def _gf2_rank(matrix):
    """Plain dense row elimination; independent of production bitset code."""
    a = np.array(matrix, dtype=np.uint8, copy=True) & 1
    assert a.ndim == 2
    pivot = 0
    for column in range(a.shape[1]):
        candidates = np.flatnonzero(a[pivot:, column])
        if not candidates.size:
            continue
        selected = pivot + int(candidates[0])
        a[[pivot, selected]] = a[[selected, pivot]]
        for row in range(pivot + 1, a.shape[0]):
            if a[row, column]:
                a[row] ^= a[pivot]
        pivot += 1
        if pivot == a.shape[0]:
            break
    return pivot


def _exact_edges(points, radius):
    coordinates = [tuple(Fraction.from_float(float(x)) for x in p) for p in points]
    threshold = Fraction.from_float(float(radius)) ** 2
    return [
        (i, j)
        for i, j in combinations(range(len(points)), 2)
        if sum((a - b) ** 2 for a, b in zip(coordinates[i], coordinates[j]))
        <= threshold
    ]


def dense_oracle(points, cycles, birth, survival):
    """Exact small-case oracle; cycles are unoriented GF(2) edge chains.

    Bounds are intentional: this is a test oracle, not a scalable backend.
    Invalid IDs are rejected here rather than silently indexing another row.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) > 32:
        raise ValueError("oracle needs at most 32 planar points")
    if not np.isfinite(points).all() or not (0 <= birth <= survival < math.inf):
        raise ValueError("oracle needs finite geometry and ordered radii")
    normalized = []
    for chain in cycles:
        edges = []
        for edge in chain:
            if len(edge) != 2:
                raise ValueError("edge must have two IDs")
            i, j = edge
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or not 0 <= int(v) < len(points) for v in (i, j)) or i == j:
                raise ValueError("invalid global vertex ID")
            edges.append(tuple(sorted((int(i), int(j)))))
        normalized.append(edges)
    early = set(_exact_edges(points, birth))
    edges = _exact_edges(points, survival)
    edge_ids = {edge: i for i, edge in enumerate(edges)}
    triangles = [
        (i, j, k)
        for i, j, k in combinations(range(len(points)), 3)
        if all(edge in edge_ids for edge in ((i, j), (i, k), (j, k)))
    ]
    d1 = np.zeros((len(points), len(edges)), dtype=np.uint8)
    for column, (i, j) in enumerate(edges):
        d1[[i, j], column] = 1
    d2 = np.zeros((len(edges), len(triangles)), dtype=np.uint8)
    for column, (i, j, k) in enumerate(triangles):
        for edge in ((i, j), (i, k), (j, k)):
            d2[edge_ids[edge], column] = 1
    z = np.zeros((len(edges), len(normalized)), dtype=np.uint8)
    closed = True
    birth_valid = True
    for column, chain in enumerate(normalized):
        boundary = np.zeros(len(points), dtype=np.uint8)
        for i, j in chain:
            boundary[i] ^= 1
            boundary[j] ^= 1
            birth_valid &= (i, j) in early
            if (i, j) in edge_ids:
                z[edge_ids[i, j], column] ^= 1
        closed &= not bool(boundary.any())
    boundary_rank = _gf2_rank(d2)
    class_rank = _gf2_rank(np.concatenate((d2, z), axis=1)) - boundary_rank
    betti_one = len(edges) - _gf2_rank(d1) - boundary_rank
    return DenseResult(
        bool(birth_valid), bool(closed), boundary_rank, class_rank, betti_one,
        len(edges), len(triangles),
        bool(birth_valid and closed and class_rank == len(normalized)),
    )


def _ring_edges(count, offset=0):
    return np.array([(offset + i, offset + (i + 1) % count) for i in range(count)],
                    dtype=np.int64)


def _square(center=(0.0, 0.0), half_width=1.0):
    return (np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
            * half_width + np.asarray(center, dtype=np.float64))


def _fundamental_cycles(vertex_count, edges):
    """A spanning-forest graph cycle basis, without any PH dependency."""
    adjacency = [[] for _ in range(vertex_count)]
    result = []
    for start, end in edges:
        parent = {start: None}
        queue = [start]
        for vertex in queue:
            for neighbor in adjacency[vertex]:
                if neighbor not in parent:
                    parent[neighbor] = vertex
                    queue.append(neighbor)
        if end not in parent:
            adjacency[start].append(end)
            adjacency[end].append(start)
            continue
        chain = [(start, end)]
        vertex = end
        while parent[vertex] is not None:
            chain.append((vertex, parent[vertex]))
            vertex = parent[vertex]
        result.append(np.array(chain, dtype=np.int64))
    return result


@pytest.fixture(scope="module")
def certificate_api():
    # Delayed import allows the independent oracle's own tests to run first.
    import open_deep_tda.structural_planar as planar
    return planar


def _check_soundness(api, points, cycles, birth, survival, holes):
    truth = dense_oracle(points, cycles, birth, survival)
    answer = api.certify(points, cycles, birth, survival, holes)
    assert isinstance(answer.certified, (bool, np.bool_))
    assert isinstance(answer.reason, str)
    assert isinstance(answer.winding, tuple)
    assert all(isinstance(row, tuple) for row in answer.winding)
    assert isinstance(answer.rank, (int, np.integer))
    assert isinstance(answer.work, dict)
    if answer.certified:
        assert truth.independent, (answer, truth, points, cycles, birth, survival, holes)
        assert answer.rank == len(cycles)
    return answer, truth


def test_oracle_known_square_and_triangle_ranks():
    cycle = [_ring_edges(4)]
    square = _square()
    live = dense_oracle(square, cycle, 2.0, 2.5)
    assert live.independent and live.class_rank == live.betti_one == 1
    assert live.boundary_rank == live.triangle_count == 0
    dead = dense_oracle(square, cycle, 2.0, 3.0)
    assert not dead.independent and dead.class_rank == dead.betti_one == 0
    assert dead.boundary_rank == 3 and dead.triangle_count == 4
    triangle = dense_oracle(square[:3], [_ring_edges(3)], 3.0, 3.0)
    assert triangle.closed and triangle.class_rank == 0 and triangle.boundary_rank == 1


def test_oracle_distinguishes_exact_real_threshold_from_rounded_norm():
    points = (_square() + 1) / 2
    upper = math.sqrt(2)
    lower = np.nextafter(upper, 0.0)
    assert Fraction.from_float(upper) ** 2 > 2
    assert Fraction.from_float(float(lower)) ** 2 < 2
    assert dense_oracle(points, [_ring_edges(4)], 1.0, lower).independent
    assert not dense_oracle(points, [_ring_edges(4)], 1.0, upper).independent


@pytest.mark.parametrize("scale", [2.0 ** -540, 2.0 ** 520])
def test_oracle_squared_distances_do_not_underflow_or_overflow(scale):
    points = _square() * scale
    assert dense_oracle(points, [_ring_edges(4)], 2 * scale, 2.5 * scale).independent
    assert not dense_oracle(points, [_ring_edges(4)], 2 * scale, 3 * scale).independent


def test_oracle_rejects_open_zero_late_and_dependent_chains():
    points = _square()
    ring = _ring_edges(4)
    assert not dense_oracle(points, [ring[:-1]], 2.0, 2.5).closed
    assert not dense_oracle(points, [np.concatenate([ring, ring])], 2.0, 2.5).independent
    assert not dense_oracle(points, [ring], 1.9, 2.5).birth_valid
    repeated = dense_oracle(points, [ring, ring[::-1, ::-1]], 2.0, 2.5)
    assert repeated.class_rank == 1 and not repeated.independent


def test_oracle_merged_annular_classes():
    points = np.concatenate([_square(), _square(half_width=0.8)])
    cycles = [_ring_edges(4), _ring_edges(4, 4)]
    truth = dense_oracle(points, cycles, 2.01, 2.05)
    assert truth.closed and truth.birth_valid
    assert truth.betti_one == truth.class_rank == 1 and not truth.independent
    assert all(dense_oracle(points, [cycle], 2.01, 2.05).independent for cycle in cycles)


def test_clear_square_is_certified_not_just_never_accepted(certificate_api):
    answer, truth = _check_soundness(certificate_api, _square(), [_ring_edges(4)],
                                     2.01, 2.1, np.array([[0., 0.]]))
    assert truth.independent
    assert answer.certified, answer.reason
    assert answer.rank == 1


def test_all_global_vertices_participate_external_center_fills(certificate_api):
    cycles = [_ring_edges(4)]
    assert dense_oracle(_square(), cycles, 2.01, 2.1).independent
    points = np.concatenate([_square(), [[0., 0.]]])
    answer, truth = _check_soundness(certificate_api, points, cycles, 2.01, 2.1,
                                     np.array([[0.05, 0.05]]))
    assert truth.class_rank == 0 and not answer.certified


def test_other_global_component_fills_without_vertex_at_witness(certificate_api):
    # A small triangle around the witness fills the square via additional edges.
    points = np.concatenate([_square(), [[-.15, -.1], [.15, -.1], [0., .15]]])
    answer, truth = _check_soundness(certificate_api, points, [_ring_edges(4)],
                                     2.01, 2.1, np.array([[0., 0.]]))
    assert truth.class_rank == 0 and not answer.certified


def test_wrong_valid_global_ids_do_not_certify_the_intended_ring(certificate_api):
    points = np.concatenate([_square((10., 0.), 0.1), _square()])
    hole = np.array([[0., 0.]])
    good, _ = _check_soundness(certificate_api, points, [_ring_edges(4, 4)],
                               2.01, 2.1, hole)
    assert good.certified, good.reason
    bad, truth = _check_soundness(certificate_api, points, [_ring_edges(4)],
                                  2.01, 2.1, hole)
    # The unrelated ring leaves ambient H1 nonzero: Betti-only checks are wrong.
    assert truth.betti_one == 1 and truth.class_rank == 0 and not bad.certified


@pytest.mark.parametrize("bad_id", [-1, 4, 2 ** 40])
def test_out_of_range_global_ids_raise_value_error(certificate_api, bad_id):
    cycle = _ring_edges(4)
    cycle[0, 0] = bad_id
    with pytest.raises(ValueError):
        certificate_api.certify(_square(), [cycle], 2.01, 2.1, np.array([[0., 0.]]))


def test_two_geometrically_distinct_cycles_can_be_one_class(certificate_api):
    points = np.concatenate([_square(), _square(half_width=0.8)])
    answer, truth = _check_soundness(
        certificate_api, points, [_ring_edges(4), _ring_edges(4, 4)],
        2.01, 2.05, np.array([[0., 0.], [.05, 0.]]),
    )
    assert truth.class_rank == 1 and not answer.certified


def test_multiple_separate_holes_certify_full_gf2_rank(certificate_api):
    points = np.concatenate([_square((-4., 0.)), _square((4., 0.))])
    cycles = [_ring_edges(4), _ring_edges(4, 4)]
    holes = np.array([[-4., 0.], [4., 0.]])
    answer, truth = _check_soundness(certificate_api, points, cycles, 2.01, 2.1, holes)
    assert truth.class_rank == 2 and answer.certified, answer.reason
    assert answer.rank == 2
    winding = np.asarray(answer.winding, dtype=np.uint8) % 2
    assert winding.shape == (2, 2) and _gf2_rank(winding) == 2


def test_repeated_hole_witness_does_not_create_independence(certificate_api):
    ring = _ring_edges(4)
    answer, truth = _check_soundness(certificate_api, _square(), [ring, ring[::-1]],
                                     2.01, 2.1, np.array([[0., 0.], [0., 0.]]))
    assert truth.class_rank == 1 and not answer.certified


def test_two_holes_and_chain_basis_change(certificate_api):
    points = np.concatenate([_square((-4., 0.)), _square((4., 0.))])
    first, second = _ring_edges(4), _ring_edges(4, 4)
    cycles = [np.concatenate([first, second]), second]
    answer, truth = _check_soundness(certificate_api, points, cycles, 2.01, 2.1,
                                     np.array([[-4., 0.], [4., 0.]]))
    assert truth.class_rank == 2
    assert answer.certified, answer.reason


@pytest.mark.parametrize("scale", [2.0 ** -540, 1.0, 2.0 ** 520])
def test_extreme_scales_allow_conservative_refusal_but_no_false_accept(certificate_api, scale):
    points = _square() * scale
    holes = np.array([[0., 0.]])
    # The exact oracle stays meaningful even when binary64 squared norms do not.
    _check_soundness(certificate_api, points, [_ring_edges(4)], 2 * scale,
                     2.5 * scale, holes)
    dead, truth = _check_soundness(certificate_api, points, [_ring_edges(4)],
                                   2 * scale, 3 * scale, holes)
    assert truth.class_rank == 0 and not dead.certified


def test_one_ulp_short_birth_cannot_pass_with_a_tolerance(certificate_api):
    answer, truth = _check_soundness(certificate_api, _square(), [_ring_edges(4)],
                                     float(np.nextafter(2., 0.)), 2.1,
                                     np.array([[0., 0.]]))
    assert not truth.birth_valid and not answer.certified


@pytest.mark.parametrize("radius", [float(np.nextafter(math.sqrt(2), 0.)), math.sqrt(2)])
def test_real_square_diagonal_threshold_no_false_accept(certificate_api, radius):
    points = (_square() + 1) / 2
    _check_soundness(certificate_api, points, [_ring_edges(4)], 1., radius,
                     np.array([[.5, .5]]))


def test_even_traversal_cancels_in_gf2_not_nonzero_integer_winding(certificate_api):
    cycle = _ring_edges(4)
    answer, truth = _check_soundness(certificate_api, _square(),
                                     [np.concatenate([cycle, cycle])], 2.01, 2.1,
                                     np.array([[0., 0.]]))
    assert truth.closed and truth.class_rank == 0 and not answer.certified


def test_crossing_bow_tie_is_not_a_nontrivial_vr_cycle(certificate_api):
    cycle = np.array([[0, 2], [2, 1], [1, 3], [3, 0]], dtype=np.int64)
    answer, truth = _check_soundness(certificate_api, _square(), [cycle],
                                     2.9, 3., np.array([[0., .25]]))
    assert truth.class_rank == 0 and not answer.certified


@pytest.mark.parametrize("seed", range(16))
def test_randomized_rings_global_permutations_and_external_fill(certificate_api, seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(5, 10))
    angles = np.arange(count) * (2 * math.pi / count) + rng.uniform(-.2, .2)
    radii = rng.uniform(.97, 1.03, count)
    ring_points = np.column_stack([np.cos(angles), np.sin(angles)]) * radii[:, None]
    scale = 2.0 ** int(rng.integers(-3, 4))
    translation = rng.integers(-8, 9, size=2).astype(np.float64)
    ring_points = ring_points * scale + translation
    cycles = [_ring_edges(count)]
    birth = max(math.dist(ring_points[i], ring_points[j]) for i, j in cycles[0]) * 1.03
    survival = birth * 1.05
    holes = np.asarray([translation], dtype=np.float64)
    answer, truth = _check_soundness(certificate_api, ring_points, cycles, birth,
                                     survival, holes)
    assert truth.independent
    assert answer.certified, (seed, answer.reason)
    # Remapping is global old-ID -> new-ID, not a locally sorted ring index.
    permutation = rng.permutation(count)
    inverse = np.argsort(permutation)
    permuted = [inverse[cycles[0]][rng.permutation(count), ::-1]]
    renamed, _ = _check_soundness(certificate_api, ring_points[permutation], permuted,
                                   birth, survival, holes)
    assert renamed.certified, (seed, renamed.reason)
    # Guarantee all spokes exist, even for a finely sampled ring.
    filled_points = np.concatenate([ring_points, holes])
    filled_survival = max(survival, 1.1 * scale)
    filled, filled_truth = _check_soundness(certificate_api, filled_points, cycles,
                                            birth, filled_survival, holes)
    assert filled_truth.class_rank == 0 and not filled.certified


@pytest.mark.parametrize("seed", range(32))
def test_randomized_small_dense_vr_no_false_accept(certificate_api, seed):
    rng = np.random.default_rng(1000 + seed)
    count = int(rng.integers(5, 13))
    # Dyadic grid gives ties, collinearity and occasional duplicate coordinates.
    points = rng.integers(-6, 7, size=(count, 2)).astype(np.float64) / 4
    birth = float(rng.choice([.5, .75, 1., 1.25, 1.5, 2.]))
    survival = birth + float(rng.choice([0., .125, .5, 1.]))
    basis = _fundamental_cycles(count, _exact_edges(points, birth))
    if not basis:
        # Still exercise a valid closed chain whose edges miss the birth scale.
        cycles = [_ring_edges(count)]
    else:
        indices = rng.choice(len(basis), size=min(3, len(basis)), replace=False)
        cycles = [basis[int(i)] for i in indices]
        if seed % 4 == 0:
            cycles.append(cycles[0][::-1, ::-1])
    holes = rng.integers(-5, 6, size=(max(1, len(cycles)), 2)).astype(np.float64) / 4
    _check_soundness(certificate_api, points, cycles, birth, survival, holes)


@pytest.mark.parametrize("seed", range(8))
def test_randomized_survival_beyond_complete_graph_cannot_certify(certificate_api, seed):
    rng = np.random.default_rng(2000 + seed)
    count = int(rng.integers(4, 10))
    points = rng.uniform(-1., 1., size=(count, 2)).astype(np.float64)
    answer, truth = _check_soundness(certificate_api, points, [_ring_edges(count)],
                                     3., 3., np.array([[0., 0.]]))
    assert truth.betti_one == truth.class_rank == 0 and not answer.certified
