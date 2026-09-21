"""Document objective limitations with counterexamples, not quality guarantees."""
import numpy as np
import torch

from open_deep_tda.fuzzy import fuzzy_losses
from open_deep_tda.losses import near_loss, separation_loss


def test_zero_stress_and_hinge_do_not_imply_neighbor_identity():
    source = torch.tensor([0., .1, 10., 11.], dtype=torch.double)
    target = torch.tensor([0., .1, .9, 1.9], dtype=torch.double)
    edges = torch.tensor([[0, 1], [2, 3]])
    nonedges = torch.tensor([[0, 2], [0, 3], [1, 2], [1, 3]])

    def lengths(x, pairs):
        return (x[pairs[:, 0]] - x[pairs[:, 1]]).abs()

    assert near_loss(lengths(source, edges), lengths(target, edges), delta=.55) < 1e-30
    assert separation_loss(lengths(source, nonedges), lengths(target, nonedges), margin=.775) == 0
    # Yet point 2's closest neighbor changes from 3 to 1.
    def neighbor(x):
        distances = (x - x[2]).abs()
        distances[2] = float('inf')
        return int(distances.argmin())
    assert neighbor(source) == 3
    assert neighbor(target) == 1


def test_fuzzy_graph_is_explicitly_batch_normalized_not_global_unbiased():
    weights = torch.tensor([.1, .9], dtype=torch.double)
    # Construct attraction losses exactly [1,3] at scale=1.
    lengths = torch.sqrt(torch.expm1(torch.tensor([1., 3.], dtype=torch.double)))
    full, _ = fuzzy_losses(lengths, lengths[:0], weights, positive_mode='attraction')
    singletons = [fuzzy_losses(lengths[i:i+1], lengths[:0], weights[i:i+1],
                              positive_mode='attraction')[0].item() for i in range(2)]
    np.testing.assert_allclose(full.item(), 2.8)
    np.testing.assert_allclose(np.mean(singletons), 2.)
