import numpy as np
import pytest
from open_deep_tda.cycle_metrics import angle_neighbor_metrics, strict_polygon_crossings, evaluate_rotation_orbits


def circle(n=12):
    angles=np.arange(n)*360/n
    return np.column_stack([np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles))]),angles


def test_true_circle_neighbors_and_reflection():
    X,a=circle()
    for Y in [X,X*np.array([1,-1]),X*7+np.array([100,200])]:
        r=angle_neighbor_metrics(Y,a)
        assert r['adjacent_view_recall']==1
        assert r['mean_neighbor_angle_degrees']==30
        assert not r['collapsed']
        assert strict_polygon_crossings(Y)==0


def test_crossings_and_collapse_are_distinct():
    assert strict_polygon_crossings([[0,0],[1,1],[0,1],[1,0]])==1
    X,a=circle()
    Z=np.zeros_like(X)
    assert strict_polygon_crossings(Z)==0
    assert angle_neighbor_metrics(Z,a)['collapsed']
    assert angle_neighbor_metrics(Z,a)['adjacent_view_recall']<1


def test_invalid_annotations_and_points():
    X,a=circle()
    with pytest.raises(ValueError):angle_neighbor_metrics(X,np.zeros(12))
    with pytest.raises(ValueError):angle_neighbor_metrics(X,a,query_mask=np.zeros(12,dtype=bool))
    with pytest.raises(ValueError):angle_neighbor_metrics(X,a,query_mask=np.ones(12,dtype=int))
    with pytest.raises(ValueError):strict_polygon_crossings(np.ones((5,3)))


def test_identical_diagrams_do_not_imply_correct_rotation_order():
    pytest.importorskip('ripser')
    X,a=circle()
    Z=X[np.random.default_rng(9).permutation(len(X))]
    result=evaluate_rotation_orbits(X,Z,np.ones(len(X),dtype=int),a)
    obj=result['objects'][0]
    assert obj['global_squared_transport']==pytest.approx(0,abs=1e-12)
    assert obj['reference_angle']['adjacent_view_recall']==1
    assert obj['embedding_angle']['adjacent_view_recall']<.5
    assert obj['source_strong_h1'] and obj['target_strong_h1']


def test_all_objects_and_heldout_views_are_evaluated():
    pytest.importorskip('ripser')
    X,a=circle(12);X=np.concatenate([X,X+5])
    angles=np.tile(a,2);objects=np.repeat([1,2],12)
    query=np.tile(np.arange(12)%3==0,2)
    result=evaluate_rotation_orbits(X,X,objects,angles,query_mask=query)
    assert result['summary']['n_objects']==2
    assert result['summary']['embedding_adjacent_recall']==1
    assert all(o['embedding_angle']['queries']==4 for o in result['objects'])
