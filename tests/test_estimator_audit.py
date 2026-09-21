"""Independent estimator audit (production files intentionally untouched).

Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
    tests/test_estimator_audit.py

Verified initial failures: nonintegral output dimension validation; accepted NumPy
config scalars break JSON / weights-only loading; nonfinite inference after scale
overflow; complex inputs lose imaginary components; diagnostic matching budget
is ignored; constant refit retains old timings; malformed CLI config escapes its
error boundary; t-SNE PCA initialization rejects otherwise valid 1D features.
Failing assertions are intentional regression specifications, not xfails.
Audit checkpoints: first pytest run after concurrent main-agent fixes yielded
20 passed / 5 failed (reference_transform overflow, matching budget, stale
constant timings, CLI unknown key, and t-SNE one-feature initialization).
Final rerun after further concurrent production fixes and two added controls:
27 passed in 3.25s. This audit itself changed no production files.

Passing controls cover CPU determinism/global RNG preservation, held-out isolation,
missing-value train/reference consistency, constant/all-NaN handling, coordinate
mode boundaries, and data-only checkpoint round trips. Checkpoints use BytesIO
and mocked filesystem operations: no model/data artifacts are written. All fits
use <=12 samples, h1_size <=8 and <=2 optimizer/semantic steps. t-SNE's iterative
optimizer is bypassed; its real initialization is tested. No CUDA/MPS claims.

Privacy observation (not a failure without a privacy promise): save() includes
all per-sample reference and embedding tensors. Geometry references plus saved
preprocessing statistics can recover observed training values approximately;
weights_only=True prevents arbitrary pickle execution, not data disclosure.
"""

from contextlib import contextmanager
from types import SimpleNamespace
import io
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
import torch

from open_deep_tda import DeepTDA, TDAConfig
from open_deep_tda import cli
from open_deep_tda.benchmark import run_benchmark
from open_deep_tda.preprocessing import numeric_matrix
from open_deep_tda.sampling import GeometrySampler, TopologySampler, knn_graph


def small_config(**overrides):
    values = dict(steps=2, semantic_steps=2, warmup_steps=0, h0_size=8,
                  h1_size=8, evaluation_size=8, subset_bank_size=2,
                  semantic_dim=4, batch_size=8, seed=3)
    values.update(overrides)
    return TDAConfig(**values)


def cloud():
    return np.random.default_rng(31).normal(size=(12, 3))


def checkpoint_bytes(model, monkeypatch):
    """Exercise actual save payload, without creating directories or model files."""
    buffer = io.BytesIO()
    real_save = torch.save
    @contextmanager
    def fake_temporary_file(**kwargs):
        yield SimpleNamespace(name="estimator-audit-in-memory.pt.tmp")
    with monkeypatch.context() as patch:
        patch.setattr("open_deep_tda.estimator.tempfile.NamedTemporaryFile", fake_temporary_file)
        patch.setattr(torch, "save", lambda state, path: real_save(state, buffer))
        patch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
        patch.setattr(Path, "replace", lambda *args, **kwargs: None)
        patch.setattr(Path, "exists", lambda *args, **kwargs: False)
        model.save("estimator-audit-in-memory.pt")
    return buffer.getvalue()


def load_checkpoint(payload, monkeypatch):
    real_load = torch.load

    def load_without_disk(path, **kwargs):
        assert kwargs["weights_only"] is True
        return real_load(io.BytesIO(payload), **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", load_without_disk)
        return DeepTDA.load("estimator-audit-in-memory.pt")


def test_nonintegral_output_dimension_rejected_at_config_boundary():
    with pytest.raises(ValueError, match="n_components"):
        small_config(n_components=2.0).validate()


def test_accepted_numpy_integer_config_is_json_serializable():
    config = small_config(seed=np.int64(3)).validate()
    # Validation explicitly accepts Integral, including NumPy integer scalars.
    json.dumps(config.to_dict(), allow_nan=False)


def test_accepted_numpy_float_config_weights_only_round_trip(monkeypatch):
    model = DeepTDA(small_config(learning_rate=np.float64(0.001))).fit(cloud())
    loaded = load_checkpoint(checkpoint_bytes(model, monkeypatch), monkeypatch)
    np.testing.assert_array_equal(loaded.transform(cloud()), model.embedding_)


@pytest.mark.parametrize("operation", ["reference_transform", "transform"])
def test_inference_rejects_overflow_instead_of_returning_nonfinite(operation):
    X = np.arange(12, dtype=float)[:, None] * 1e-30
    model = DeepTDA(small_config(standardize=False, lambda_h1=0)).fit(X)
    # Preprocessing remains finite float32; only division by learned scale fails.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises((ValueError, RuntimeError), match="finite|overflow|rescale|range"):
            getattr(model, operation)(np.array([[1e20]]))


def test_complex_inputs_rejected_instead_of_silently_projected_to_real():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            numeric_matrix(np.array([[1 + 2j], [1 + 9j]]))


def test_evaluation_honors_configured_matching_budget():
    config = small_config(lambda_h1=0, max_matching_size=1)
    # Eight distinct evaluation samples need more than one augmented matching
    # row even for H0; requesting a one-row budget must not silently use 512.
    with pytest.raises((ValueError, RuntimeError), match="budget|matching|size"):
        DeepTDA(config).fit(cloud())


def test_constant_refit_does_not_reuse_previous_run_timings():
    model = DeepTDA(small_config()).fit(cloud())
    assert model.timings_["optimization_seconds"] > 0
    model.fit(np.ones((4, 3)))
    assert model.report_["status"] == "constant_reference"
    assert model.report_["timings"].get("optimization_seconds", 0) == 0


def test_cli_unknown_json_config_key_has_clean_error_boundary(monkeypatch, capsys):
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: '{"stepz": 2}')
    # Config parsing happens before reading input or creating the output dir.
    result = cli.main(["fit", "--input", "unused.npy", "--config", "unused.json",
                       "--output", "unused-output"])
    assert result == 2
    assert "stepz" in capsys.readouterr().err


def test_benchmark_tsne_initialization_accepts_valid_one_feature_input(monkeypatch):
    from sklearn.manifold import TSNE

    # Avoid t-SNE's >=250 iteration minimum. Real validation and PCA initialization
    # still run; only the subsequent iterative optimization is bypassed.
    monkeypatch.setattr(TSNE, "_tsne",
                        lambda self, P, degrees_of_freedom, n_samples, X_embedded,
                        neighbors=None, skip_num_points=0: X_embedded)
    result = run_benchmark(np.arange(8.)[:, None],
                           small_config(steps=0, missing_indicators=False),
                           seeds=[3], methods=["tsne"])
    assert result["results"][0]["status"] in ("ok", "skipped")


@pytest.mark.parametrize("mode", ["geometry", "semantic"])
def test_train_reference_consistency_and_heldout_isolation(mode):
    X = cloud()
    X[1, 0] = np.nan
    X[:, 2] = np.nan
    heldout = np.full((3, 3), 1000.)
    baseline = DeepTDA(small_config(mode=mode)).fit(X)
    validated = DeepTDA(small_config(mode=mode)).fit(X, validation_data=heldout)
    np.testing.assert_array_equal(baseline.reference_, validated.reference_)
    np.testing.assert_array_equal(baseline.embedding_, validated.embedding_)
    np.testing.assert_array_equal(validated.reference_transform(X), validated.reference_)
    np.testing.assert_array_equal(validated.transform(X), validated.embedding_)
    assert "held_out" in validated.report_
    assert not validated.reference_encoder_ or all(
        not p.requires_grad for p in validated.reference_encoder_.parameters())
    assert validated.transform(np.empty((0, 3))).shape == (0, 2)
    assert validated.ood_scores(np.empty((0, 3))).shape == (0,)


@pytest.mark.parametrize("mode", ["geometry", "semantic"])
def test_cpu_determinism_and_no_global_rng_or_thread_side_effect(mode):
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    threads = torch.get_num_threads()
    first = DeepTDA(small_config(mode=mode)).fit(cloud())
    second = DeepTDA(small_config(mode=mode)).fit(cloud())
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.get_num_threads() == threads
    np.testing.assert_array_equal(first.embedding_, second.embedding_)
    assert first.history_ == second.history_


@pytest.mark.parametrize("mode", ["geometry", "semantic"])
@pytest.mark.parametrize("X", [np.ones((1, 3)), np.ones((4, 3)), np.full((4, 3), np.nan)])
def test_constant_and_nan_inputs_are_explicit_and_finite(mode, X):
    model = DeepTDA(small_config(mode=mode)).fit(X)
    assert model.report_["status"] == "constant_reference"
    np.testing.assert_array_equal(model.embedding_, np.zeros((len(X), 2)))
    np.testing.assert_array_equal(model.transform(X), model.embedding_)
    assert np.isfinite(model.reference_transform(X)).all()
    assert not model.history_
    json.dumps(model.report_, allow_nan=False)


@pytest.mark.parametrize("mode", ["geometry", "semantic"])
def test_builtin_config_checkpoint_round_trip_uses_safe_loader(mode, monkeypatch):
    model = DeepTDA(small_config(mode=mode, lambda_reconstruction=0.1)).fit(cloud())
    payload = checkpoint_bytes(model, monkeypatch)
    # Explicitly verify the data-only serialized state independently of load().
    state = torch.load(io.BytesIO(payload), weights_only=True)
    assert state["reference"].shape == model.reference_.shape
    rng_before = torch.random.get_rng_state().clone()
    loaded = load_checkpoint(payload, monkeypatch)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    np.testing.assert_array_equal(loaded.reference_transform(cloud()), model.reference_)
    np.testing.assert_array_equal(loaded.transform(cloud()), model.embedding_)
    np.testing.assert_allclose(loaded.ood_scores(cloud()), model.ood_scores(cloud()))


@pytest.mark.parametrize("constant", [False, True])
def test_coordinate_mode_rejects_transform_even_after_round_trip(constant, monkeypatch):
    X = np.ones((4, 3)) if constant else cloud()
    model = DeepTDA(small_config(optimizer_mode="coordinates")).fit(X)
    loaded = load_checkpoint(checkpoint_bytes(model, monkeypatch), monkeypatch)
    np.testing.assert_array_equal(loaded.embedding_, model.embedding_)
    for estimator in (model, loaded):
        with pytest.raises(NotImplementedError, match="coordinate"):
            estimator.transform(X)
    with pytest.raises(ValueError, match="parametric"):
        DeepTDA(small_config(optimizer_mode="coordinates")).fit(X, validation_data=X)


def test_sampling_with_duplicate_points_is_bounded_and_reproducible():
    X = np.repeat(np.arange(4.)[:, None], 3, axis=0)
    edges, neighbors = knn_graph(X, 2)
    assert all(i not in row for i, row in enumerate(neighbors))
    assert np.all(edges[:, 0] < edges[:, 1])
    assert len(np.unique(edges, axis=0)) == len(edges)
    first = TopologySampler(X, neighbors, landmark_size=6, seed=3).bank(3, 8)
    second = TopologySampler(X, neighbors, landmark_size=6, seed=3).bank(3, 8)
    assert [strategy for strategy, _ in first] == ["local", "cover", "random"]
    for (_, a), (_, b) in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert len(a) == len(np.unique(a)) == 8
        assert np.all((0 <= a) & (a < len(X)))
    positives, negatives = GeometrySampler(edges, len(X)).sample(8, np.random.default_rng(3))
    edge_set = set(map(tuple, edges))
    assert len(positives) <= 8 and len(negatives) <= 8
    assert all(tuple(pair) in edge_set for pair in positives)
    assert all(i < j and (i, j) not in edge_set for i, j in negatives)


def test_three_dimensional_transform_and_reference_contract():
    model = DeepTDA(small_config(n_components=3)).fit(cloud())
    assert model.embedding_.shape == (12, 3)
    np.testing.assert_array_equal(model.transform(cloud()), model.embedding_)
    np.testing.assert_array_equal(model.reference_transform(cloud()), model.reference_)


def test_failed_refit_does_not_allow_stale_transform():
    model = DeepTDA(small_config()).fit(cloud())
    with pytest.raises(ValueError):
        model.fit(np.array([[np.inf, 0., 0.]]))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.transform(cloud())
