"""Post-measurement numerical-range and cooperative deadline hardening."""
import numpy as np
import pytest

from open_deep_tda._graph_mapping import CompactMap


def test_finite_extreme_anchors_do_not_create_an_infinite_membership_tolerance():
    with pytest.raises(ValueError, match='norm overflow'):
        CompactMap(np.array([[0.,0.],[1e200,0.],[0.,1e200]]))


def test_nonfinite_lapack_output_does_not_silently_become_dimension_zero(monkeypatch):
    def invalid(*args, **kwargs):
        return np.eye(3,2), np.array([np.inf, np.nan]), np.eye(2)
    monkeypatch.setattr(np.linalg, 'svd', invalid)
    with pytest.raises(ValueError, match='factorization'):
        CompactMap(np.array([[0.,0.],[1.,0.],[0.,1.]]))


def test_phase_two_deadline_propagates_instead_of_selecting_barycenter():
    model=CompactMap(np.array([[0.,0.],[1.,0.],[0.,1.]]))
    calls=0
    def deadline():
        nonlocal calls
        calls+=1
        if calls==3:
            raise TimeoutError('caller budget expired inside phase II')
    with pytest.raises(TimeoutError, match='phase II'):
        model.solve(np.array([4.,4.]),np.array([0,1,2]),np.ones(3)/3,
                    np.array([1/3,1/3]),check=deadline)
    assert calls==3
