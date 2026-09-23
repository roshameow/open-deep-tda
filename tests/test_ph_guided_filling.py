"""Independent dense GF2 checks of fixed-family filling certificates."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from open_deep_tda._ph_guided_filling import (
    FillingCutPool, FillingLimits, FillingResourceError, find_filling_obstructions,
)


def matrix(n, edges):
    D = np.full((n, n), 9.0)
    np.fill_diagonal(D, 0)
    for (u, v), length in edges.items():
        D[u, v] = D[v, u] = length
    return D


def rank(columns):
    pivots = {}
    for value in columns:
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    return len(pivots)


def dense(D, cycles, b):
    edges = [(i, j) for i in range(len(D)) for j in range(i + 1, len(D)) if D[i, j] <= b]
    ids = {edge: i for i, edge in enumerate(edges)}
    triangles = [(i, j, k) for i in range(len(D)) for j in range(i + 1, len(D))
                 for k in range(j + 1, len(D)) if all(e in ids for e in ((i, j), (i, k), (j, k)))]
    def vector(chain):
        out = 0
        for e in chain:
            out ^= 1 << ids[tuple(sorted(e))]
        return out
    boundaries = [vector(((i, j), (i, k), (j, k))) for i, j, k in triangles]
    base = rank(boundaries)
    present = [cycle for cycle in cycles if all(D[tuple(sorted(e))] <= b for e in cycle)]
    return rank(boundaries + [vector(chain) for chain in present]) - base


def square_external():
    cycle = [[0, 1], [1, 2], [2, 3], [3, 0]]
    sides = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (0, 3): 1}
    S = matrix(5, sides)
    D = matrix(5, sides)
    for i in range(4):
        D[i, 4] = D[4, i] = 1.5
    return S, D, [cycle]


def annulus():
    outer = [(i, (i + 1) % 4) for i in range(4)]
    inner = [(i + 4, (i + 1) % 4 + 4) for i in range(4)]
    sides = {tuple(sorted(e)): 1 for e in outer + inner}
    S = matrix(8, sides)
    D = matrix(8, sides)
    for i in range(4):
        for u, v in ((i, i + 4), (i, ((i + 1) % 4) + 4)):
            D[u, v] = D[v, u] = 1.5
    return S, D, [outer, inner]


def assert_certificates(result, D, cycles, b):
    ids = {(i, j): q for q, (i, j) in enumerate(
        (i, j) for i in range(len(D)) for j in range(i + 1, len(D)) if D[i, j] <= b)}
    def vec(edges):
        x = 0
        for e in edges:
            x ^= 1 << ids[tuple(sorted(e))]
        return x
    masks = []
    for rel in result.relations:
        left = 0
        mask = 0
        for q in rel.cycle_indices:
            mask ^= 1 << q
            left ^= vec(cycles[q])
        right = 0
        for i, j, k in rel.filling_triangles:
            assert max(D[i, j], D[i, k], D[j, k]) <= b
            right ^= vec(((i, j), (i, k), (j, k)))
        assert left == right and mask
        masks.append(mask)
    assert rank(masks) == len(masks)
    assert result.surviving_rank == dense(D, cycles, b)


def test_external_filler_ties_and_missing_birth():
    S, D, cycles = square_external()
    result = find_filling_obstructions(S, D, cycles, 1, 2)
    assert not result.accepted and result.surviving_rank == 0
    assert len(result.relations) == 1
    assert result.relations[0].cycle_indices == (0,)
    assert result.relations[0].cheapest_edge_to_cut == (0, 4)
    assert_certificates(result, D, cycles, 2)
    # Missing at birth but present at survival: still an active filling.
    D[0, 1] = D[1, 0] = 1.8
    r = find_filling_obstructions(S, D, cycles, 1, 2)
    assert r.missing_birth_edges == (((0, 1),),)
    assert r.edges_present_at_survival == (True,) and len(r.relations) == 1
    assert_certificates(r, D, cycles, 2)
    # Missing at survival: no invented filling; explicit failed status.
    D[0, 1] = D[1, 0] = 3
    r = find_filling_obstructions(S, D, cycles, 1, 2)
    assert r.missing_survival_edges == (((0, 1),),)
    assert r.relations == () and not r.accepted


def test_joint_relation_and_source_validation():
    S, D, cycles = annulus()
    r = find_filling_obstructions(S, D, cycles, 1, 2)
    assert r.surviving_rank == 1 and not r.accepted
    assert len(r.relations) == 1 and r.relations[0].cycle_indices == (0, 1)
    assert S[r.relations[0].cheapest_edge_to_cut] > 2
    assert_certificates(r, D, cycles, 2)
    no_fill = find_filling_obstructions(S, S, cycles, 1, 2)
    assert no_fill.accepted and no_fill.surviving_rank == 2 and not no_fill.relations
    with pytest.raises(ValueError, match="invalid_source_contract"):
        find_filling_obstructions(D, D, cycles, 2, 2)
    with pytest.raises(ValueError, match="invalid_source_contract"):
        find_filling_obstructions(S, S, cycles * 2, 1, 2)


def test_raw_cancelled_source_edge_still_requires_birth_membership():
    S, _, cycles = square_external()
    # The extra diagonal cancels in F2, but it was supplied as a source-birth
    # edge twice and is absent at radius 1. It must not be smuggled through.
    malformed = [cycles[0] + [[0, 2], [2, 0]]]
    with pytest.raises(ValueError, match='raw supplied edge missing at birth'):
        find_filling_obstructions(S, S, malformed, 1, 2)


def test_cut_threshold_float64_and_float32_overflow_refused():
    S, _, cycles = square_external()
    with pytest.raises(ValueError, match=r'survival \+ margin'):
        FillingCutPool(cycles, 1e308, 1e308, margin=1e308)
    import torch
    pool = FillingCutPool(cycles, 1e40, 1e40)
    with pytest.raises(ValueError, match='unrepresentable'):
        pool.loss(torch.zeros((5, 2), dtype=torch.float32))


def test_dense_random_full_rows():
    rng = np.random.default_rng(309)
    S, _, cycles = square_external()
    for _ in range(25):
        D = matrix(5, {(i, j): float(rng.choice([1, 1.5, 3]))
                       for i in range(5) for j in range(i + 1, 5)})
        r = find_filling_obstructions(S, D, cycles, 1, 2)
        assert_certificates(r, D, cycles, 2)


def test_explicit_limits_no_partial_and_guards():
    S, D, cycles = square_external()
    for limits, message in [
        (FillingLimits(max_edges=3), "max_edges"),
        (FillingLimits(max_triangles=0), "max_triangles"),
        (FillingLimits(max_word_ops=0), "max_word_ops"),
        (FillingLimits(max_peak_words=0), "max_peak_words"),
        (FillingLimits(deadline_seconds=1e-15), "deadline_seconds"),
    ]:
        with pytest.raises(FillingResourceError, match=message):
            find_filling_obstructions(S, D, cycles, 1, 2, limits=limits)
    with pytest.raises(ValueError):
        find_filling_obstructions(S.astype('float32'), D, cycles, 1, 2)
    with pytest.raises(ValueError):
        find_filling_obstructions(S, D, [[(0, 1)]], 1, 2)


def test_cut_pool_monotone_live_grad_and_zero_norm():
    torch = pytest.importorskip("torch")
    S, D, cycles = square_external()
    pool = FillingCutPool(cycles, 1, 2, margin=.1, max_cuts=1)
    r = find_filling_obstructions(S, D, cycles, 1, 2)
    assert pool.add(r) == ((0, 4),)
    assert pool.add(r) == ((0, 4),)
    D[1, 4] = D[4, 1] = 1.9
    r2 = find_filling_obstructions(S, D, cycles, 1, 2)
    assert r2.relations and r2.relations[0].cheapest_edge_to_cut != (0, 4)
    with pytest.raises(FillingResourceError, match="max_cuts"):
        pool.add(r2)
    assert pool.cuts == {(0, 4)}
    Z = torch.zeros((5, 2), dtype=torch.float64, requires_grad=True)
    birth, upper, lower = pool.loss(Z)
    assert birth.detach().item() == upper.detach().item() == 0.0 and lower.detach().item() > 0
    (birth + upper + lower).backward()
    assert torch.isfinite(Z.grad).all()
    Z2 = torch.tensor([[0., 0.], [2., 0.], [2., 2.], [0., 2.], [0., 0.]],
                      requires_grad=True)
    b, u, c = pool.loss(Z2)
    assert b.detach().item() > 0 and c.detach().item() > 0
    (b + u + c).backward()
    assert torch.isfinite(Z2.grad).all()


@pytest.mark.skipif(os.getenv("TDA_PRIVATE_K4_SMOKE") != "1", reason="private opt-in")
def test_private_k4_full_domain_smoke():
    from scipy.spatial.distance import pdist, squareform
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "docs/research/k4_same_cycle_audit/source_contract.json").read_text())
    source = np.load(contract['paths']['source'], allow_pickle=False)
    # Private source NPZ may use multiple columns: explicitly require coordinates.
    if isinstance(source, np.lib.npyio.NpzFile):
        with source:
            assert 'X' in source.files
            source = source['X']
    src = squareform(pdist(source)).astype(np.float64)
    result = find_filling_obstructions(src, src, contract['cycles'],
                                       contract['a'], contract['b'])
    assert result.accepted and result.target_edges <= 12000
    assert result.target_triangles <= 250000
    assert_certificates(result, src, contract['cycles'], contract['b'])
    # The denser private target is explicitly outside this oracle's edge cap.
    target = np.load(contract['paths']['source_only'], allow_pickle=False)
    dst = squareform(pdist(target)).astype(np.float64)
    with pytest.raises(FillingResourceError, match="max_edges"):
        find_filling_obstructions(src, dst, contract['cycles'], contract['a'], contract['b'])
