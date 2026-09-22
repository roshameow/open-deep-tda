"""Synthetic analytic and regression checks for isolated neighbor NCE."""
import math

import numpy as np
import pytest
import torch

from open_deep_tda.contrastive import (
    AnchoredNeighborSampler, neighbor_nce_loss, sample_anchor_pairs,
)


def inputs():
    return dict(pos_lengths=torch.tensor([.5, 1.], dtype=torch.double, requires_grad=True),
                neg_lengths=torch.tensor([[.2, 2., .7], [.4, 1.5, 3.]],
                                         dtype=torch.double, requires_grad=True),
                weights=torch.tensor([.2, .8], dtype=torch.double, requires_grad=True),
                valid=torch.tensor([[True, False, True], [True, True, True]]),
                scale=.7, temperature=1.3)


def test_analytic_weighted_formula_and_finite_difference():
    args = inputs()
    loss = neighbor_nce_loss(**args)
    # Independent probability-domain oracle at moderate distances.
    p = (1 + (args['pos_lengths'] / args['scale']).square()).pow(-1 / args['temperature'])
    q = (1 + (args['neg_lengths'] / args['scale']).square()).pow(-1 / args['temperature'])
    rows = ((p + (q * args['valid']).sum(dim=1)) / p).log()
    expected = (rows * args['weights']).sum() / args['weights'].sum()
    torch.testing.assert_close(loss, expected)
    assert torch.autograd.gradcheck(
        lambda p, n, w: neighbor_nce_loss(p, n, w, args['valid'], .7, 1.3),
        (args['pos_lengths'], args['neg_lengths'], args['weights']))
    args['weights'] = args['weights'] * 37
    torch.testing.assert_close(neighbor_nce_loss(**args), expected)


def test_known_value_singleton_weight_cancels():
    p, n = torch.tensor([1.]), torch.tensor([[1., 3.]])
    # Positive kernel=1/2, negatives=1/2 and 1/10; -log((1/2)/(11/10)).
    loss = neighbor_nce_loss(p, n, torch.tensor([7.]), torch.ones_like(n, dtype=torch.bool), 1.)
    assert loss.item() == pytest.approx(math.log(2.2))


def test_backward_direction_mask_and_zero_weight_row():
    args = inputs()
    args['weights'] = torch.tensor([1., 0.], dtype=torch.double)
    neighbor_nce_loss(**args).backward()
    gp, gn = args['pos_lengths'].grad, args['neg_lengths'].grad
    assert gp[0] > 0 and gp[1] == 0  # descent contracts positives
    assert (gn[0, [0, 2]] < 0).all()  # descent separates valid negatives
    assert gn[0, 1] == 0 and (gn[1] == 0).all()
    assert torch.isfinite(gp).all() and torch.isfinite(gn).all()


@pytest.mark.parametrize('k', [0, 1, 2, 3, 20, None])
def test_hard_selection_mask_safe_and_live(k):
    p = torch.tensor([.8, .9, 1.], dtype=torch.double, requires_grad=True)
    # Keep gradcheck perturbations inside the nonnegative domain, including
    # masked slots. Zero-length behavior is tested separately below.
    n = torch.tensor([[.01, .3, 4., .6], [.01, .01, 2., .01], [.2, .3, .4, .5]],
                     dtype=torch.double, requires_grad=True)
    valid = torch.tensor([[False, True, True, True], [False, False, True, False],
                          [False, False, False, False]])
    weights = torch.ones_like(p)
    chosen = torch.zeros_like(valid)
    for row in range(3):
        ids = torch.where(valid[row])[0]
        ids = ids[torch.argsort(n[row, ids].detach())]
        chosen[row, ids if k is None else ids[:k]] = True
    loss = neighbor_nce_loss(p, n, weights, valid, 1., hard_negatives=k)
    expected = neighbor_nce_loss(p, n, weights, chosen, 1.)
    torch.testing.assert_close(loss, expected)
    assert torch.autograd.gradcheck(
        lambda a, b: neighbor_nce_loss(a, b, weights, valid, 1., hard_negatives=k), (p, n))
    loss.backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(n.grad).all()
    assert (n.grad[~chosen] == 0).all()
    assert (n.grad[chosen] < 0).all()
    assert p.grad[2] == 0


@pytest.mark.parametrize('mode', ['empty', 'no_candidates', 'all_masked', 'zero_weights', 'hard_zero'])
def test_connected_zeros(mode):
    b, c = (0, 3) if mode == 'empty' else ((2, 0) if mode == 'no_candidates' else (2, 3))
    # Empty-slice zeros must not sum large lengths and then multiply infinity by zero.
    p = torch.full((b,), torch.finfo(torch.double).max, dtype=torch.double, requires_grad=True)
    n = torch.full((b, c), torch.finfo(torch.double).max, dtype=torch.double, requires_grad=True)
    w = torch.full((b,), 0. if mode == 'zero_weights' else 1., dtype=torch.double, requires_grad=True)
    valid = torch.full((b, c), mode != 'all_masked')
    loss = neighbor_nce_loss(p, n, w, valid, 1., hard_negatives=0 if mode == 'hard_zero' else None)
    assert loss.ndim == 0 and loss.item() == 0 and loss.requires_grad
    loss.backward()
    for t in (p, n, w):
        assert t.grad is not None and (t.grad == 0).all()


def test_rows_without_negatives_retain_normalization_weight():
    args = inputs()
    args['valid'][1] = False
    loss = neighbor_nce_loss(**args)
    first = neighbor_nce_loss(args['pos_lengths'][:1], args['neg_lengths'][:1],
                              args['weights'][:1], args['valid'][:1], .7, 1.3)
    torch.testing.assert_close(loss, .2 * first)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('scale', [1e-200, 1., 1e200])
def test_zero_duplicate_and_huge_lengths_are_finite(dtype, scale):
    p = torch.tensor([0., 1., torch.finfo(dtype).max], dtype=dtype, requires_grad=True)
    n = torch.tensor([[0., 0., 0.], [0., 2., 3.], [1., 0., torch.finfo(dtype).max]],
                     dtype=dtype, requires_grad=True)
    w = torch.ones_like(p)
    valid = torch.ones_like(n, dtype=torch.bool)
    loss = neighbor_nce_loss(p, n, w, valid, scale)
    assert torch.isfinite(loss)
    gp, gn = torch.autograd.grad(loss, (p, n))
    assert torch.isfinite(gp).all() and torch.isfinite(gn).all()
    assert gp[0] == 0 and (gn[n == 0] == 0).all()
    duplicate_loss = neighbor_nce_loss(p[:1], n[:1], w[:1], valid[:1], scale)
    assert duplicate_loss.item() == pytest.approx(math.log(4), rel=1e-6)


def test_masked_points_zero_gradient_and_live_coordinate_duplicates():
    z = torch.tensor([[0., 0.], [0., 0.], [.5, .2], [2., 1.]], requires_grad=True)
    p = torch.linalg.vector_norm(z[[0]] - z[[1]], dim=1)
    n = torch.linalg.vector_norm(z[[0]] - z[[1, 2, 3]], dim=1)[None, :]
    loss = neighbor_nce_loss(p, n, torch.ones(1), torch.tensor([[True, True, False]]), 1.)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad[2].abs().sum() > 0 and (z.grad[3] == 0).all()


def test_large_weight_mass_does_not_overflow():
    args = inputs()
    args['weights'] = torch.full((2,), torch.finfo(torch.double).max, dtype=torch.double)
    loss = neighbor_nce_loss(**args)
    args['weights'] = torch.ones(2, dtype=torch.double)
    torch.testing.assert_close(loss, neighbor_nce_loss(**args))


@pytest.mark.parametrize('name', ['scale', 'temperature'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, np.bool_(True), '1', 1j, None])
def test_invalid_scalar(name, value):
    args = inputs()
    args[name] = value
    with pytest.raises(ValueError, match=name):
        neighbor_nce_loss(**args)


@pytest.mark.parametrize('value', [-1, 1.5, True, np.bool_(False), '2', float('inf')])
def test_invalid_hard_count(value):
    with pytest.raises(ValueError, match='hard_negatives'):
        neighbor_nce_loss(**inputs(), hard_negatives=value)


@pytest.mark.parametrize('name', ['pos_lengths', 'neg_lengths', 'weights'])
@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_invalid_tensor_values_including_masked(name, value):
    args = inputs()
    args[name] = args[name].detach().clone()
    args[name].view(-1)[1] = value  # neg[0,1] is masked but still validated
    with pytest.raises(ValueError, match=name):
        neighbor_nce_loss(**args)


@pytest.mark.parametrize('name,value', [
    ('pos_lengths', [1., 2.]), ('pos_lengths', torch.ones(2, 1)),
    ('pos_lengths', torch.ones(2, dtype=torch.long)),
    ('neg_lengths', torch.ones(2)), ('neg_lengths', torch.ones(3, 3)),
    ('neg_lengths', torch.ones(2, 3, dtype=torch.complex64)),
    ('neg_lengths', torch.ones(2, 3, dtype=torch.float32)),
    ('weights', torch.ones(1, dtype=torch.double)), ('weights', torch.ones(2, 1)),
    ('weights', torch.ones(2, dtype=torch.float32)),
    ('valid', [[True] * 3] * 2), ('valid', torch.ones(2, 3)),
    ('valid', torch.ones(2, 2, dtype=torch.bool)),
    ('valid', torch.ones(2, 3, dtype=torch.bool, device='meta')),
])
def test_invalid_shapes_types_devices(name, value):
    args = inputs()
    args[name] = value
    with pytest.raises(ValueError):
        neighbor_nce_loss(**args)


def test_sampler_canonical_union_exclusion_orientation_and_uniformity():
    edges = np.array([[2, 0], [0, 2], [1, 0], [3, 1], [0, 1], [4, 3]])
    original = edges.copy()
    sampler = AnchoredNeighborSampler(edges, 6)
    np.testing.assert_array_equal(edges, original)
    expected_edges = {(0, 2), (0, 1), (1, 3), (3, 4)}
    assert set(map(tuple, sampler.edges)) == expected_edges
    pos, neg, valid = sampler.sample(24000, 6, np.random.default_rng(9))
    assert pos.shape == (24000, 2) and neg.shape == valid.shape == (24000, 6)
    assert pos.dtype == neg.dtype == np.int64 and valid.dtype == np.bool_
    assert ((0 <= neg) & (neg < 6)).all()
    # Independent tuple-set oracle checks *all* edges, not just the selected positive.
    expected_valid = np.array([[a != v and tuple(sorted((a, v))) not in expected_edges
                                for v in row] for a, row in zip(pos[:, 0], neg)])
    np.testing.assert_array_equal(valid, expected_valid)
    oriented, counts = np.unique(pos, axis=0, return_counts=True)
    assert set(map(tuple, oriented)) == expected_edges | {(b, a) for a, b in expected_edges}
    assert np.all(np.abs(counts - 3000) < 250)  # fixed-seed uniform-edge/orientation check
    # Candidates are uniform proposals, not rejection-filled, and duplicates are allowed.
    assert np.all(np.abs(np.bincount(neg.ravel(), minlength=6) - 24000) < 700)
    assert any(len(set(row[mask])) < mask.sum() for row, mask in zip(neg, valid))


def test_seed_determinism_wrapper_and_canonicalization_invariance():
    edges = np.array([[0, 1], [1, 2], [0, 3]])
    sampler = AnchoredNeighborSampler(edges, 7)
    expected = sampler.sample(30, 8, np.random.default_rng(16))
    variants = (edges[::-1, ::-1], np.concatenate((edges, edges[:, ::-1], edges)))
    for variant in variants:
        actual = AnchoredNeighborSampler(variant, 7).sample(30, 8, np.random.default_rng(16))
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
    a, p, neg, mask = sample_anchor_pairs(edges, 7, 30, 8, np.random.default_rng(16))
    for actual, wanted in zip((np.column_stack((a, p)), neg, mask), expected):
        np.testing.assert_array_equal(actual, wanted)
    # Reusing a generator advances it, rather than secretly reseeding each call.
    rng = np.random.default_rng(16)
    sampler.sample(30, 8, rng)
    assert not np.array_equal(sampler.sample(30, 8, rng)[1], expected[1])


@pytest.mark.parametrize('n', [2, 8])
def test_complete_graph_masks_every_negative(n):
    edges = np.array([(a, b) for a in range(n) for b in range(a + 1, n)])
    pos, neg, valid = AnchoredNeighborSampler(edges, n).sample(100, 20, np.random.default_rng(1))
    assert pos.shape == (100, 2) and neg.shape == valid.shape == (100, 20)
    assert not valid.any()
    # In particular the selected positive cannot be reused as a negative.
    assert not valid[neg == pos[:, 1:2]].any()


@pytest.mark.parametrize('empty,b,c', [(True, 5, 4), (False, 0, 4), (False, 5, 0), (True, 0, 0)])
def test_sampler_empty_and_zero_counts(empty, b, c):
    edges = np.empty((0, 2), dtype=np.int64) if empty else np.array([[0, 1]])
    pos, neg, valid = AnchoredNeighborSampler(edges, 3).sample(b, c, np.random.default_rng(1))
    rows = 0 if empty else b
    assert pos.shape == (rows, 2) and neg.shape == valid.shape == (rows, c)
    assert pos.dtype == neg.dtype == np.int64 and valid.dtype == np.bool_


def test_sparse_large_population_and_safe_edge_key_boundary():
    n = math.isqrt(np.iinfo(np.int64).max)
    edges = np.array([[n - 1, n - 2], [0, n - 1]], dtype=np.int64)
    sampler = AnchoredNeighborSampler(edges, n)
    assert sampler.keys[-1] == (n - 2) * n + n - 1
    # This call is feasible only without population-sized storage/matrices.
    pos, neg, valid = sampler.sample(10, 4, np.random.default_rng(1))
    assert pos.shape == (10, 2) and neg.shape == valid.shape == (10, 4)
    assert (sampler.keys >= 0).all() and ((neg >= 0) & (neg < n)).all()


@pytest.mark.parametrize('n', [0, 1, -1, 2., True, np.bool_(True), '3',
                                math.isqrt(np.iinfo(np.int64).max) + 1])
def test_invalid_population(n):
    with pytest.raises(ValueError):
        AnchoredNeighborSampler(np.array([[0, 1]]), n)


@pytest.mark.parametrize('edges', [
    [], [0, 1], [[0, 1, 2]], [[0., 1.]], [[False, True]], [['0', '1']],
    [[0j, 1j]], [[0, 0]], [[-1, 1]], [[0, 4]], np.array([[0, 1], [2]], dtype=object),
    np.empty((0, 2), dtype=float), np.array([[0, 2**64 - 1]], dtype=np.uint64),
])
def test_invalid_edges(edges):
    with pytest.raises(ValueError):
        AnchoredNeighborSampler(edges, 4)


@pytest.mark.parametrize('field', ['batch_size', 'candidates'])
@pytest.mark.parametrize('value', [-1, 1.5, True, np.bool_(False), '2'])
def test_invalid_sample_counts(field, value):
    args = dict(batch_size=2, candidates=3, rng=np.random.default_rng(1))
    args[field] = value
    with pytest.raises(ValueError, match=field):
        AnchoredNeighborSampler(np.array([[0, 1]]), 3).sample(**args)


def test_integer_scalars_unsigned_edges_and_rng_validation():
    sampler = AnchoredNeighborSampler(np.array([[1, 0]], dtype=np.uint64), np.int64(3))
    pos, neg, _ = sampler.sample(np.int32(2), np.int64(3), np.random.default_rng(1))
    assert pos.shape == (2, 2) and neg.shape == (2, 3)
    for rng in (None, 1, np.random.RandomState(1)):
        with pytest.raises(ValueError, match='rng'):
            sampler.sample(2, 3, rng)
    args = inputs()
    torch.testing.assert_close(neighbor_nce_loss(**args, hard_negatives=np.int64(1)),
                               neighbor_nce_loss(**args, hard_negatives=1))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('temperature', [1e-200, 1e-320, 1e200])
def test_equal_scores_and_no_negatives_at_extreme_temperature(dtype, temperature):
    p = torch.ones(2, dtype=dtype, requires_grad=True)
    n = torch.ones((2, 2), dtype=dtype, requires_grad=True)
    w = torch.ones_like(p)
    valid = torch.tensor([[True, True], [False, False]])
    # Center BEFORE dividing by temperature: separately scaled scores can
    # overflow to -inf even though this equal-score loss is representable.
    loss = neighbor_nce_loss(p, n, w, valid, 1., temperature)
    assert loss.item() == pytest.approx(math.log(3) / 2)
    # Zero terms must stay connected without evaluating unrepresentable scores.
    zero = neighbor_nce_loss(p, n, w, torch.zeros_like(valid), 1., temperature)
    assert zero.item() == 0
    zero.backward()
    assert (p.grad == 0).all() and (n.grad == 0).all()


@pytest.mark.parametrize('k', [1, 2, 5])
def test_hard_zero_length_ties_are_finite_and_mask_safe(k):
    p = torch.zeros(1, dtype=torch.double, requires_grad=True)
    n = torch.zeros((1, 4), dtype=torch.double, requires_grad=True)
    valid = torch.tensor([[False, True, False, True]])
    loss = neighbor_nce_loss(p, n, torch.ones_like(p), valid, 1., hard_negatives=k)
    assert loss.item() == pytest.approx(math.log1p(min(k, 2)))
    loss.backward()
    assert (p.grad == 0).all() and (n.grad == 0).all()


def test_weighted_mean_survives_individual_float32_logit_overflow():
    p = torch.tensor([1e30, 0.], requires_grad=True)
    n = torch.zeros((2, 1), requires_grad=True)
    valid = torch.ones_like(n, dtype=torch.bool)
    for first_weight in (0., 1e-10):
        w = torch.tensor([first_weight, 1.])
        loss = neighbor_nce_loss(p, n, w, valid, 1., 1e-37)
        expected = neighbor_nce_loss(p.double(), n.double(), w.double(), valid, 1., 1e-37)
        assert torch.isfinite(loss)
        torch.testing.assert_close(loss, expected)
        gp, gn = torch.autograd.grad(loss, (p, n))
        assert torch.isfinite(gp).all() and torch.isfinite(gn).all()
    assert loss.item() == pytest.approx(1.381551e29, rel=1e-6)


def test_small_kernel_derivative_does_not_underflow_before_temperature_factor():
    p = torch.tensor([1e-20], requires_grad=True)
    n = torch.tensor([[1e-20]], requires_grad=True)
    loss = neighbor_nce_loss(p, n, torch.ones_like(p), torch.ones_like(n, dtype=torch.bool),
                             1., 1e-20)
    gp, gn = torch.autograd.grad(loss, (p, n))
    assert gp.item() == pytest.approx(1., rel=1e-5)
    assert gn.item() == pytest.approx(-1., rel=1e-5)


def test_extreme_temperature_backward_combines_representable_factors():
    p = torch.tensor([1e300], dtype=torch.double, requires_grad=True)
    n = torch.tensor([[1e300]], dtype=torch.double, requires_grad=True)
    temperature = 1e-320
    loss = neighbor_nce_loss(p, n, torch.ones_like(p), torch.ones_like(n, dtype=torch.bool),
                             1., temperature)
    assert loss.item() == pytest.approx(math.log(2))
    gp, gn = torch.autograd.grad(loss, (p, n))
    expected = 1 / (1e300 * temperature)
    assert gp.item() == pytest.approx(expected, rel=1e-12)
    assert gn.item() == pytest.approx(-expected, rel=1e-12)


def test_subnormal_lengths_are_not_clamped_when_scale_is_small():
    scale = torch.finfo(torch.float32).tiny
    p = torch.tensor([scale / 2], requires_grad=True)
    n = torch.tensor([[scale]], requires_grad=True)
    loss = neighbor_nce_loss(p, n, torch.ones_like(p), torch.ones_like(n, dtype=torch.bool), scale)
    assert loss.item() == pytest.approx(math.log(1 + 1.25 / 2), rel=1e-5)
    expected = neighbor_nce_loss(p.double(), n.double(), torch.ones(1, dtype=torch.double),
                                 torch.ones_like(n, dtype=torch.bool), scale)
    gp, gn = torch.autograd.grad(loss, (p, n))
    ep, en = torch.autograd.grad(expected, (p, n))
    torch.testing.assert_close(gp, ep, rtol=2e-5, atol=0)
    torch.testing.assert_close(gn, en, rtol=2e-5, atol=0)
