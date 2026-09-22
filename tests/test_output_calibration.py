"""TRAIN-only output-scale contract and reusable regression checks.

Reproduce with: python -m pytest -q tests/test_output_calibration.py
Pair IDs are regenerated with seed+211, never stored in diagnostics. Float32
pair differences must be promoted before the norm; dot products use a common
scale. Calibration must not enter the reference encoder or optimizer history.
"""
import copy
import json

import numpy as np
import pytest
import torch
from scipy.spatial.distance import pdist

from open_deep_tda import DeepTDA, TDAConfig


@pytest.fixture
def cloud():
    return np.random.default_rng(63).normal(size=(40, 6)).astype(np.float32)


def small(**kwargs):
    options = dict(seed=17, steps=2, warmup_steps=0, batch_size=12,
                   n_neighbors=4, h0_size=8, h1_size=8, subset_bank_size=1,
                   evaluation_size=8, validation_interval=1, log_interval=1,
                   lambda_h0=0, lambda_h1=0)
    options.update(kwargs)
    return DeepTDA(**options)


def pair_distances(model, embedding):
    count = min(max(1024, 2 * len(model.reference_)), 20000)
    ids = np.random.default_rng(model.config.seed + 211).integers(
        len(model.reference_), size=(count, 2))
    source = np.linalg.norm(model.reference_[ids[:, 0]].astype(float)
                            - model.reference_[ids[:, 1]].astype(float), axis=1)
    target = np.linalg.norm(embedding[ids[:, 0]].astype(float)
                            - embedding[ids[:, 1]].astype(float), axis=1)
    return source, target


@pytest.mark.parametrize("objective", ["stress", "neighbor_nce"])
def test_train_only_ls_and_validation_independent(cloud, objective):
    options = dict(output_calibration="train_pairs", geometry_objective=objective)
    plain = small(geometry_objective=objective).fit(cloud[:30])
    calibrated = small(**options).fit(cloud[:30], validation_data=cloud[30:])
    other = small(**options).fit(cloud[:30], validation_data=cloud[30:] * 9 + 20)
    no_validation = small(**options).fit(cloud[:30])
    for model in (other, no_validation):
        assert model.calibration_diagnostics_ == calibrated.calibration_diagnostics_
        np.testing.assert_array_equal(model.embedding_, calibrated.embedding_)
    source, target = pair_distances(plain, plain.embedding_)
    alpha = float(source @ target / (target @ target))
    assert calibrated.output_scale_ == pytest.approx(alpha, rel=1e-14)
    assert abs(alpha - 1) > 1e-3
    diag = calibrated.calibration_diagnostics_
    assert diag["method"] == "train_pairs"
    assert diag["source"] == "TRAIN"
    assert diag["pairs"] == 1024
    assert diag["status"] == "fitted"
    assert diag["fit_stress_before"] == pytest.approx(np.linalg.norm(source-target) / np.linalg.norm(source))
    assert diag["fit_stress_after"] == pytest.approx(np.linalg.norm(source-alpha*target) / np.linalg.norm(source))
    assert diag["fit_stress_after"] <= diag["fit_stress_before"]
    _, actual = pair_distances(calibrated, calibrated.embedding_)
    assert np.linalg.norm(source-actual) <= np.linalg.norm(source-target) + 1e-7
    assert calibrated.history_ == plain.history_
    assert all(row["output_scale_scope"] == "pre-calibration" for row in calibrated.history_)
    for a, b in zip(calibrated.validation_history_, plain.validation_history_):
        assert "pre-calibration" in a["scope"]
        assert a["metrics"]["geometry"] == b["metrics"]["geometry"]
    assert calibrated.report_["output_calibration"] == diag
    assert "pre-calibration" in calibrated.report_["history_scope"]
    assert "held_out" in calibrated.report_
    # Bounded aggregates only, also safe for compact checkpoints.
    assert all(not isinstance(value, (list, dict, np.ndarray)) for value in diag.values())
    json.dumps(calibrated.report_, allow_nan=False)


@pytest.mark.parametrize("mode", ["geometry", "semantic"])
def test_distances_scale_reference_and_raw_forward_unchanged(cloud, mode):
    options = dict(mode=mode, semantic_steps=2, semantic_dim=4)
    plain = small(**options).fit(cloud[:30])
    calibrated = small(**options, output_calibration="train_pairs").fit(cloud[:30])
    np.testing.assert_array_equal(plain.reference_, calibrated.reference_)
    np.testing.assert_array_equal(plain.reference_transform(cloud), calibrated.reference_transform(cloud))
    for key, weight in plain.model_.state_dict().items():
        assert torch.equal(weight, calibrated.model_.state_dict()[key])
    raw = calibrated._forward_numpy(calibrated.model_, calibrated.reference_transform(cloud))
    np.testing.assert_array_equal(raw, plain.transform(cloud))
    np.testing.assert_array_equal(calibrated.transform(cloud), raw * calibrated.output_scale_)
    np.testing.assert_array_equal(calibrated.transform(cloud[:30]), calibrated.embedding_)
    np.testing.assert_allclose(pdist(calibrated.transform(cloud)),
                               pdist(plain.transform(cloud)) * calibrated.output_scale_, rtol=2e-6)
    np.testing.assert_array_equal(calibrated.ood_scores(cloud), plain.ood_scores(cloud))
    assert calibrated.transform(np.empty((0, 6))).shape == (0, 2)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("mode", ["geometry", "semantic"])
def test_checkpoint_roundtrip(cloud, tmp_path, full, mode):
    model = small(mode=mode, semantic_steps=2, semantic_dim=4,
                  output_calibration="train_pairs").fit(cloud[:30])
    path = model.save(tmp_path / "model.pt", include_training_data=full)
    state = torch.load(path, weights_only=True)
    assert state["output_scale"] == model.output_scale_
    assert state["output_calibration"] == model.calibration_diagnostics_
    assert state["report"]["output_calibration"] == model.calibration_diagnostics_
    loaded = DeepTDA.load(path)
    assert loaded.output_scale_ == model.output_scale_
    assert loaded.calibration_diagnostics_ == model.calibration_diagnostics_
    np.testing.assert_array_equal(loaded.transform(cloud), model.transform(cloud))
    np.testing.assert_array_equal(loaded.reference_transform(cloud), model.reference_transform(cloud))
    if full:
        np.testing.assert_array_equal(loaded.embedding_, model.embedding_)
    else:
        assert loaded.embedding_ is None and loaded.reference_ is None
        assert loaded.history_ == []
        assert state["embedding"] is None and state["reference"] is None
    again = DeepTDA.load(loaded.save(tmp_path / "again.pt", include_training_data=full))
    np.testing.assert_array_equal(again.transform(cloud), model.transform(cloud))


@pytest.mark.parametrize("full", [False, True])
def test_legacy_state_and_attribute_fallback(cloud, tmp_path, full):
    model = small().fit(cloud)
    expected = model.transform(cloud)
    path = model.save(tmp_path / "legacy.pt", include_training_data=full)
    state = torch.load(path, weights_only=True)
    del state["config"]["output_calibration"]
    del state["output_scale"]
    del state["output_calibration"]
    state["report"].pop("output_calibration")
    torch.save(state, path)
    loaded = DeepTDA.load(path)
    assert loaded.config.output_calibration == "none"
    assert loaded.output_scale_ == 1.0
    assert loaded.calibration_diagnostics_["status"] == "skipped"
    assert loaded.report_["output_calibration"] == loaded.calibration_diagnostics_
    np.testing.assert_array_equal(loaded.transform(cloud), expected)
    del loaded.output_scale_
    del loaded.calibration_diagnostics_
    np.testing.assert_array_equal(loaded.transform(cloud), expected)
    restored = DeepTDA.load(loaded.save(tmp_path / "fallback.pt", include_training_data=full))
    np.testing.assert_array_equal(restored.transform(cloud), expected)


@pytest.mark.parametrize("steps", [0, 2])
def test_default_none_exact_predictions(cloud, steps):
    assert TDAConfig().output_calibration == "none"
    implicit = small(steps=steps).fit(cloud)
    explicit = small(steps=steps, output_calibration="none").fit(cloud)
    np.testing.assert_array_equal(implicit.embedding_, explicit.embedding_)
    np.testing.assert_array_equal(implicit.transform(cloud), explicit.transform(cloud))
    np.testing.assert_array_equal(implicit.transform(cloud),
                                  implicit._forward_numpy(implicit.model_, implicit.reference_))
    assert implicit.output_scale_ == 1.0
    assert implicit.calibration_diagnostics_["status"] == "skipped"
    assert implicit.calibration_diagnostics_["pairs"] == 0


@pytest.mark.parametrize("steps", [0, 2])
@pytest.mark.parametrize("coordinates", [False, True])
def test_zero_step_and_direct_coordinates(cloud, tmp_path, steps, coordinates):
    options = dict(steps=steps, optimizer_mode="coordinates" if coordinates else "parametric")
    plain = small(**options).fit(cloud)
    model = small(**options, output_calibration="train_pairs")
    result = model.fit_transform(cloud)
    source, target = pair_distances(plain, plain.embedding_)
    assert model.output_scale_ == pytest.approx(source @ target / (target @ target))
    np.testing.assert_array_equal(result, model.embedding_)
    np.testing.assert_allclose(model.embedding_, plain.embedding_ * model.output_scale_)
    restored = DeepTDA.load(model.save(tmp_path / "coords.pt"))
    np.testing.assert_array_equal(restored.embedding_, model.embedding_)
    if coordinates:
        for fitted in (model, restored):
            with pytest.raises(NotImplementedError):
                fitted.transform(cloud)
        with pytest.raises(ValueError, match="direct-coordinate"):
            model.save(tmp_path / "compact.pt", include_training_data=False)
    else:
        np.testing.assert_array_equal(model.transform(cloud), model.embedding_)
    if steps == 0:
        assert model.history_ == [] and model.validation_history_ == []


@pytest.mark.parametrize("X", [np.zeros((1, 3)), np.full((10, 3), 7.), np.full((4, 3), np.nan)])
def test_constant_no_calibration_necessary(X, tmp_path):
    model = small(output_calibration="train_pairs", mode="semantic").fit(X)
    assert model.output_scale_ == 1.0
    assert model.calibration_diagnostics_["status"] == "skipped"
    assert model.calibration_diagnostics_["pairs"] == 0
    assert "constant_reference" in model.calibration_diagnostics_["reason"]
    assert not model.embedding_.any()
    restored = DeepTDA.load(model.save(tmp_path / "constant.pt", include_training_data=False))
    np.testing.assert_array_equal(restored.transform(X), model.embedding_)


def test_refit_resets_scale(cloud):
    model = small(output_calibration="train_pairs").fit(cloud)
    assert model.output_scale_ != 1.0
    model.config.output_calibration = "none"
    model.fit(cloud)
    assert model.output_scale_ == 1.0
    np.testing.assert_array_equal(model.transform(cloud), small().fit(cloud).transform(cloud))
    model.config.output_calibration = "train_pairs"
    model.fit(np.ones_like(cloud))
    assert model.output_scale_ == 1.0
    assert model.calibration_diagnostics_["status"] == "skipped"


def test_zero_target_degenerate(cloud, monkeypatch):
    def zero_optimize(self, *args):
        with torch.no_grad():
            for p in self.model_.parameters():
                p.zero_()
        self.embedding_ = self._forward_numpy(self.model_, self.reference_)
    monkeypatch.setattr(DeepTDA, "_optimize", zero_optimize)
    model = small(output_calibration="train_pairs").fit(cloud)
    assert model.output_scale_ == 1.0
    assert model.calibration_diagnostics_["status"] == "degenerate"
    assert model.calibration_diagnostics_["fit_stress_before"] == 1.0
    assert model.calibration_diagnostics_["fit_stress_after"] == 1.0
    assert not model.transform(cloud).any()
    json.dumps(model.report_, allow_nan=False)


@pytest.mark.parametrize("n, count", [(3, 1024), (600, 1200), (12000, 20000)])
def test_pair_budget_and_independent_rng(n, count):
    model = small(output_calibration="train_pairs", seed=2**32-1)
    model.reference_ = np.random.default_rng(7).normal(size=(n, 2)).astype(np.float32)
    model.embedding_ = model.reference_.copy() / 2
    before = copy.deepcopy(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    model._calibrate_output()
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert model.calibration_diagnostics_["pairs"] == count
    assert model.output_scale_ == pytest.approx(2.0)
    assert model.calibration_diagnostics_["fit_stress_after"] == 0.0


@pytest.mark.parametrize("source_scale, target_scale", [(1e30, 1e20), (1e-30, 1e-35)])
def test_float64_distance_and_scaled_dot_safety(source_scale, target_scale):
    model = small(output_calibration="train_pairs")
    base = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    model.reference_ = base * source_scale
    model.embedding_ = base * target_scale
    model._calibrate_output()
    assert model.output_scale_ == pytest.approx(source_scale / target_scale, rel=1e-6)
    assert np.isfinite(model.embedding_).all()
    assert model.calibration_diagnostics_["fit_stress_after"] < 1e-6
    json.dumps(model.calibration_diagnostics_, allow_nan=False)


@pytest.mark.parametrize("value", ["train", "validation_pairs", "TRAIN_PAIRS", "", None, True, 0])
def test_invalid_config(value):
    with pytest.raises(ValueError, match="output_calibration"):
        TDAConfig(output_calibration=value).validate()


def test_calibrated_ood_transform_rejects_post_scaling_overflow(monkeypatch):
    X = np.random.default_rng(4).normal(size=(12, 3)).astype(np.float32)
    model = DeepTDA(steps=0, evaluation_size=6).fit(X)
    # NumPy 1.x may promote a huge Python scalar to float64; exceed that
    # product range too rather than assuming NumPy 2.x scalar promotion.
    model.output_scale_ = float(np.finfo(np.float64).max)
    monkeypatch.setattr(model, '_forward_numpy',
                        lambda module, U: np.full((len(U), 2), 2., dtype=np.float32))
    with pytest.raises(ValueError, match='nonfinite transformed'):
        model.transform(X)
