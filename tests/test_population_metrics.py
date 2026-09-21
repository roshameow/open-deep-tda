import numpy as np
import pytest
from sklearn.manifold import trustworthiness
from open_deep_tda.population_metrics import population_geometry
from open_deep_tda.real_datasets import load_uci_har, _validate_har


def test_population_all_queries_matches_sklearn():
    rng=np.random.default_rng(21)
    X=rng.normal(size=(31,5));Z=rng.normal(size=(31,2))
    result=population_geometry(X,Z,max_queries=31,k=4,block_size=3)
    assert result['trustworthiness']==pytest.approx(trustworthiness(X,Z,n_neighbors=4))
    assert result['continuity']==pytest.approx(trustworthiness(Z,X,n_neighbors=4))
    assert result['population_size']==result['query_count']==31


def test_one_query_still_ranks_against_full_population():
    X=np.array([[0.],[1.],[2.],[3.]])
    Z=np.array([[0.],[3.],[1.],[2.]])
    result=population_geometry(X,Z,query_indices=[0],max_queries=1,k=1)
    assert result['population_size']==4 and result['query_count']==1
    assert result['trustworthiness']==.5
    assert result['continuity']==0
    assert result['knn_overlap']==0
    assert result['rank_metric_status']=='defined'


def test_population_identical_and_query_validation():
    X=np.random.default_rng(6).normal(size=(50,4))
    a=population_geometry(X,X,max_queries=12,seed=3)
    assert a==population_geometry(X,X,max_queries=12,seed=3)
    assert a['trustworthiness']==a['continuity']==a['knn_overlap']==1
    for indices in [[0,0],[-1],[50],[.5],[]]:
        with pytest.raises(ValueError):population_geometry(X,X,query_indices=indices)
    with pytest.raises(ValueError):population_geometry(X,X,query_indices=[0,1],max_queries=1)


def test_har_download_is_opt_in(tmp_path,monkeypatch):
    def fail(*a,**k):raise AssertionError('unexpected network access')
    monkeypatch.setattr('urllib.request.urlopen',fail)
    with pytest.raises(FileNotFoundError):load_uci_har(tmp_path)
    archive=tmp_path/'human_activity_recognition_using_smartphones.zip'
    archive.write_bytes(b'not the pinned official data')
    with pytest.raises(ValueError,match='checksum'):load_uci_har(tmp_path)


def test_har_split_validator():
    data={'X_train':np.zeros((7352,561),dtype=np.float32),'X_test':np.zeros((2947,561),dtype=np.float32),
          'y_train':np.arange(7352)%6+1,'y_test':np.arange(2947)%6+1,
          'subjects_train':np.arange(7352)%21+1,'subjects_test':np.arange(2947)%9+22}
    _validate_har(data)
    data['subjects_test'][0]=1
    with pytest.raises(ValueError,match='subject-disjoint'):_validate_har(data)
