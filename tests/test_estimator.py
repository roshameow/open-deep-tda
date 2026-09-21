import json
import numpy as np
import pytest
import torch

from open_deep_tda import DeepTDA, TDAConfig
from open_deep_tda.datasets import make_dataset
from open_deep_tda.preprocessing import NumericPreprocessor
from open_deep_tda.sampling import knn_graph, GeometrySampler, TopologySampler


def small(**kwargs):
    options = dict(steps=8, warmup_steps=1, h0_size=16, h1_size=12,
                   subset_bank_size=3, topology_interval=2, evaluation_size=12,
                   validation_interval=7, log_interval=2, n_neighbors=4)
    options.update(kwargs)
    return DeepTDA(**options)


@pytest.fixture
def cloud():
    return make_dataset("circle", 40, 6, .02, 9)[0]


def test_training_roundtrip_and_heldout(cloud, tmp_path):
    model = small().fit(cloud[:30], validation_data=cloud[30:])
    assert model.embedding_.shape == (30, 2)
    assert len(model.history_) >= 3
    assert any(row["h1_evaluated"] for row in model.history_)
    assert model.validation_history_
    assert "held_out" in model.report_
    assert np.isfinite(model.embedding_).all()
    np.testing.assert_allclose(model.transform(cloud[:30]), model.embedding_, atol=1e-6)
    checkpoint = model.save(tmp_path / "model.pt")
    state = torch.load(checkpoint, weights_only=True)
    assert state["format_version"] == 1
    loaded = DeepTDA.load(checkpoint)
    np.testing.assert_array_equal(model.transform(cloud), loaded.transform(cloud))
    np.testing.assert_array_equal(model.reference_transform(cloud), loaded.reference_transform(cloud))
    assert loaded.ood_scores(cloud).shape == (40,)
    assert loaded.transform(np.empty((0, 6))).shape == (0, 2)
    json.dumps(model.report_, allow_nan=False)


def test_determinism_and_rng_restoration(cloud):
    torch.manual_seed(99)
    before = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    first = small().fit_transform(cloud)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.get_num_threads() == threads
    second = small().fit_transform(cloud)
    np.testing.assert_array_equal(first, second)


def test_semantic_frozen_and_missing_values(cloud, tmp_path):
    data = cloud.copy()
    data[::3, 0] = np.nan
    data[:, 2] = np.nan
    model = small(mode="semantic", semantic_dim=5, semantic_steps=5).fit(data)
    assert model.reference_.shape == (40, 5)
    assert model.semantic_history_
    assert all(not p.requires_grad for p in model.reference_encoder_.parameters())
    assert np.isfinite(model.transform(data)).all()
    loaded = DeepTDA.load(model.save(tmp_path / "semantic.pt"))
    np.testing.assert_array_equal(model.transform(data), loaded.transform(data))


@pytest.mark.parametrize("X", [np.zeros((1, 3)), np.full((12, 4), 7.), np.full((4, 2), np.nan)])
def test_constant_explicit_path(X, tmp_path):
    model = small(mode="semantic").fit(X)
    assert model.constant_
    assert model.report_["status"] == "constant_reference"
    assert np.all(model.embedding_ == 0)
    assert np.all(model.transform(X) == 0)
    loaded = DeepTDA.load(model.save(tmp_path / "constant.pt"))
    np.testing.assert_array_equal(loaded.transform(X), model.embedding_)


def test_coordinate_mode_is_not_parametric(cloud, tmp_path):
    model = small(optimizer_mode="coordinates", lambda_reconstruction=.1).fit(cloud)
    assert model.embedding_.shape == (40, 2)
    with pytest.raises(NotImplementedError):
        model.transform(cloud)
    loaded = DeepTDA.load(model.save(tmp_path / "coords.pt"))
    np.testing.assert_array_equal(loaded.embedding_, model.embedding_)
    with pytest.raises(NotImplementedError):
        loaded.transform(cloud)


def test_three_dimensions(cloud):
    model = small(n_components=3, steps=3).fit(cloud)
    assert model.transform(cloud).shape == (40, 3)


@pytest.mark.parametrize("X", [[], np.array([1, 2]), np.empty((0, 3)), np.ones((4, 0)), [[np.inf, 1]]])
def test_bad_data(X):
    with pytest.raises(ValueError):
        small().fit(X)


@pytest.mark.parametrize("options", [{"h1_size": 0}, {"steps": -1}, {"seed": True},
    {"n_components": 4}, {"lambda_h1": -1}, {"mode": "unknown"}, {"learning_rate": 0},
    {"mask_probability": 1}, {"lambda_h0": float("nan")}, {"steps": 1.2}])
def test_bad_config(options):
    with pytest.raises(ValueError):
        DeepTDA(**options)


def test_not_fitted_and_schema_checks(cloud):
    model = small()
    with pytest.raises(RuntimeError):
        model.transform(cloud)
    model.fit(cloud)
    with pytest.raises(ValueError):
        model.transform(np.ones((2, 5)))
    with pytest.raises(ValueError):
        model.fit(cloud, validation_data=np.ones((2, 7)))
    with pytest.raises(RuntimeError):
        model.transform(cloud)


def test_budget_failure_never_partial_success(cloud):
    model = small(max_simplices=10)
    with pytest.raises(RuntimeError, match="[Bb]udget|max_simplices"):
        model.fit(cloud)
    assert not model._fitted


def test_preprocessor_train_only_and_fixed_schema():
    train = np.array([[1., np.nan, 2.], [3., np.nan, 4.]])
    p = NumericPreprocessor().fit(train)
    before = p.state()
    result = p.transform([[1000., 20., np.nan]])
    assert result.shape == (1, 6)
    assert p.state() == before
    assert result[0, -1] == 1
    assert np.isfinite(result).all()


def test_samplers_no_self_edges_or_false_negatives(cloud):
    edges, neighbors = knn_graph(cloud, 4)
    assert all(i not in row for i, row in enumerate(neighbors))
    sampler = GeometrySampler(edges, len(cloud))
    p, q = sampler.sample(200, np.random.default_rng(4))
    edge_set = set(map(tuple, edges))
    assert all(tuple(pair) in edge_set for pair in p)
    assert all(tuple(pair) not in edge_set and pair[0] != pair[1] for pair in q)
    topo = TopologySampler(cloud, neighbors, 15, 2)
    for strategy in ("local", "random", "cover"):
        ids = topo.sample(12, strategy)
        assert len(ids) == len(set(ids)) == 12
        assert min(ids) >= 0 and max(ids) < len(cloud)


def test_two_points_duplicate_neighbors():
    X = np.array([[0., 1.], [0., 1.]])
    e, n = knn_graph(X, 10)
    np.testing.assert_array_equal(e, [[0, 1]])
    assert 0 not in n[0] and 1 not in n[1]
    _, negative = GeometrySampler(e, 2).sample(10, np.random.default_rng(0))
    assert negative.shape == (0, 2)
