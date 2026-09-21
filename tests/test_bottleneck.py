import numpy as np
import pytest
from open_deep_tda.topology import bottleneck_distance


def test_empty_diagonal_and_matched():
    assert bottleneck_distance([], []) == 0
    assert bottleneck_distance([[0, 2]], []) == 1
    assert bottleneck_distance([], [[1, 5]]) == 2
    assert bottleneck_distance([[0, 2]], [[.1, 2.3]]) == pytest.approx(.3)
    assert bottleneck_distance([[0, 2]], [[10, 12]]) == 1
    assert bottleneck_distance([[0, 0]], []) == 0


def test_limits():
    with pytest.raises(RuntimeError):
        bottleneck_distance([[0, 1]], [[0, 2]], max_matching_size=1)
    with pytest.raises(ValueError):
        bottleneck_distance([[0, np.inf]], [])


def test_persim_oracle():
    persim = pytest.importorskip("persim")
    rng = np.random.default_rng(83)
    for _ in range(25):
        P = rng.uniform(size=(rng.integers(1, 8), 2))
        Q = rng.uniform(size=(rng.integers(1, 8), 2))
        P[:, 1] += P[:, 0]
        Q[:, 1] += Q[:, 0]
        assert bottleneck_distance(P, Q) == pytest.approx(persim.bottleneck(P, Q), abs=1e-12)
