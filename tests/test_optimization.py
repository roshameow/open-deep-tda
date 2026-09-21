import torch
from open_deep_tda.losses import pairwise_distances, h1_loss


def test_actual_h1_optimization_reduces_distorted_ring_loss():
    """A controlled optimizer test, not a claim about difficult global topology recovery."""
    X = torch.tensor([[1., 0.], [.4, .95], [-.6, .8], [-1.1, -.2], [-.2, -1.], [.8, -.65]], dtype=torch.float64)
    Z = torch.nn.Parameter(X @ torch.tensor([[1.2, .15], [0., 1.3]], dtype=torch.float64))
    reference = pairwise_distances(X)
    optimizer = torch.optim.Adam([Z], lr=.01)
    initial = None
    for step in range(80):
        terms = h1_loss(reference, pairwise_distances(Z))
        loss = terms['pd1'] + .1 * terms['crit1']
        if initial is None:
            initial = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert terms['source_bars'] == terms['target_bars'] == 1
    assert float(loss.detach()) < initial * .01
