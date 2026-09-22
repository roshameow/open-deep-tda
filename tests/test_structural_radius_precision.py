"""Do not certify a different threshold after coercing a real scalar."""
from fractions import Fraction

import numpy as np
import pytest

from open_deep_tda.structural_constructive import _radius
from open_deep_tda.structural_planar import certify


@pytest.mark.parametrize('radius', [Fraction(-1, 10**500), Fraction(1, 10),
                                     2**53+1, np.uint64(2**64-1)])
def test_constructor_rejects_nonrepresentable_or_negative_radii(radius):
    with pytest.raises(ValueError):
        _radius(radius, 'radius')


@pytest.mark.parametrize('radius', [Fraction(-1, 10**500), Fraction(1, 10),
                                     2**53+1, np.uint64(2**64-1)])
def test_planar_certificate_does_not_silently_round_thresholds(radius):
    p = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    cycles = [[(0,1),(1,2),(2,3),(3,0)]]
    with pytest.raises(ValueError):
        certify(p, cycles, radius, radius, np.array([[.5,.5]]))


def test_exactly_representable_scalars_remain_supported():
    assert _radius(Fraction(1,2), 'radius') == .5
    assert _radius(np.uint64(2**63), 'radius') == float(2**63)
    assert _radius(np.float32(.1), 'radius') == float(np.float32(.1))
