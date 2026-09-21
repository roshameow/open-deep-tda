"""Residual-only scaling contracts; these synthetic tests do not establish benefit."""
import numpy as np
import pytest
import torch

from open_deep_tda import DeepTDA, TDAConfig
from open_deep_tda.models import ParametricEmbedding


def small(**overrides):
    options = dict(steps=2, warmup_steps=0, batch_size=16, n_neighbors=3,
                   lambda_h0=0, lambda_h1=0, evaluation_size=8, seed=7)
    options.update(overrides)
    return DeepTDA(**options)


@pytest.fixture
def cloud():
    return np.random.default_rng(8).normal(size=(24, 6)).astype(np.float32)


@pytest.mark.parametrize("output_dim", [2, 3])
def test_analytical_pca_initialization_at_both_scales(output_dim):
    # Orthogonal columns with distinct variances: signed PCA axes are exactly
    # e0, e1, e2, so centering and projection have an independent analytic oracle.
    centered = np.array([[3, 2, 1], [3, -2, -1], [-3, 2, -1], [-3, -2, 1]],
                        dtype=np.float32)
    X = centered + np.array([5, -7, 2], dtype=np.float32)
    outputs = []
    for scale in (1.0, 20.0):
        model = ParametricEmbedding(3, output_dim, scale)
        assert model.initialize_pca(X) == 3
        x = torch.from_numpy(X)
        with torch.no_grad():
            output = model(x)
            assert torch.equal(output, model.linear(x))
            assert torch.count_nonzero(model.residual(x * scale)) == 0
        np.testing.assert_allclose(output.numpy(), centered[:, :output_dim], atol=1e-6)
        outputs.append(output)
    assert torch.equal(*outputs)


def test_default_is_exact_legacy_forward_and_state():
    default = ParametricEmbedding(6, 2)
    # The zero-initialized last layer would hide accidental residual scaling.
    with torch.no_grad():
        default.residual[-1].weight.fill_(0.125)
        default.residual[-1].bias.fill_(0.25)
    explicit = ParametricEmbedding(6, 2, 1.0)
    explicit.load_state_dict(default.state_dict(), strict=True)
    x = torch.linspace(-2, 2, 30).reshape(5, 6)
    legacy = default.linear(x) + default.residual(x)
    assert torch.equal(default(x), legacy)
    assert torch.equal(explicit(x), legacy)
    assert default.residual_input_scale == 1.0
    assert isinstance(default.residual_input_scale, float)
    assert not dict(default.named_buffers())
    expected_keys = {f"{layer}.{parameter}" for layer in
                     ("linear", "residual.0", "residual.2", "residual.4")
                     for parameter in ("weight", "bias")}
    assert set(default.state_dict()) == expected_keys


def test_scale_only_changes_residual_inputs():
    model = ParametricEmbedding(3, 2, 20.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.05)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
    actual = model(x)
    assert torch.equal(actual, model.linear(x) + model.residual(x * 20))
    assert not torch.equal(actual, model.linear(x * 20) + model.residual(x * 20))
    assert not torch.equal(actual, model.linear(x) + model.residual(x))


@pytest.mark.parametrize("scale", [1.0, 20.0])
@pytest.mark.parametrize("active_residual", [False, True])
def test_backward_is_finite(scale, active_residual):
    model = ParametricEmbedding(6, 2, scale)
    if active_residual:
        with torch.no_grad():
            model.residual[-1].weight.fill_(0.125)
    x = torch.linspace(-1, 1, 30).reshape(5, 6).requires_grad_()
    loss = (model(x) - 0.5).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(model.residual[-1].weight.grad) > 0
    if active_residual:
        assert torch.count_nonzero(model.residual[0].weight.grad) > 0


@pytest.mark.parametrize("mode", ["geometry", "semantic"])
def test_reference_and_pca_initialization_unchanged(cloud, mode):
    options = dict(steps=0, mode=mode, semantic_dim=4, semantic_steps=2)
    default = small(**options).fit(cloud)
    scaled = small(**options, residual_input_scale=20).fit(cloud)
    assert scaled.model_.residual_input_scale == 20.0
    assert default.reference_scale_ == scaled.reference_scale_
    np.testing.assert_array_equal(default.reference_, scaled.reference_)
    np.testing.assert_array_equal(default.reference_transform(cloud), scaled.reference_transform(cloud))
    np.testing.assert_array_equal(default.embedding_, scaled.embedding_)
    np.testing.assert_array_equal(default.transform(cloud), scaled.transform(cloud))


@pytest.mark.parametrize("include_training_data", [True, False])
@pytest.mark.parametrize("mode", ["geometry", "semantic"])
def test_scaled_checkpoint_roundtrip(cloud, tmp_path, include_training_data, mode):
    model = small(residual_input_scale=20, mode=mode,
                  semantic_dim=4, semantic_steps=2).fit(cloud)
    assert model.model_.residual_input_scale == 20.0
    assert torch.count_nonzero(model.model_.residual[-1].weight) > 0
    probe = cloud[:5] + 0.1
    checkpoint = model.save(tmp_path / "scaled.pt", include_training_data=include_training_data)
    state = torch.load(checkpoint, weights_only=True)
    assert state["config"]["residual_input_scale"] == 20.0
    assert "residual_input_scale" not in state["model"]
    loaded = DeepTDA.load(checkpoint)
    assert loaded.config.residual_input_scale == loaded.model_.residual_input_scale == 20.0
    np.testing.assert_array_equal(model.transform(probe), loaded.transform(probe))
    np.testing.assert_array_equal(model.reference_transform(probe), loaded.reference_transform(probe))
    if include_training_data:
        np.testing.assert_array_equal(model.embedding_, loaded.embedding_)
        np.testing.assert_array_equal(model.reference_, loaded.reference_)
    else:
        assert loaded.embedding_ is None and loaded.reference_ is None
    # Ensure roundtrip equality is sensitive to the scale, not just a zero residual.
    loaded.model_.residual_input_scale = 1.0
    assert not np.array_equal(model.transform(probe), loaded.transform(probe))


@pytest.mark.parametrize("include_training_data", [True, False])
def test_legacy_checkpoint_without_scale(cloud, tmp_path, include_training_data):
    model = small().fit(cloud)
    checkpoint = model.save(tmp_path / "legacy.pt", include_training_data=include_training_data)
    state = torch.load(checkpoint, weights_only=True)
    del state["config"]["residual_input_scale"]
    del state["report"]["config"]["residual_input_scale"]
    torch.save(state, checkpoint)
    loaded = DeepTDA.load(checkpoint)
    assert loaded.config.residual_input_scale == loaded.model_.residual_input_scale == 1.0
    np.testing.assert_array_equal(model.transform(cloud), loaded.transform(cloud))
    np.testing.assert_array_equal(model.reference_transform(cloud), loaded.reference_transform(cloud))


@pytest.mark.parametrize("value", [0, -1, -20.0, float("nan"), float("inf"), -float("inf"),
                                   True, False, np.bool_(True), "20", None, 1 + 0j, [], {}])
def test_invalid_config_scale(value):
    with pytest.raises(ValueError, match="residual_input_scale"):
        TDAConfig(residual_input_scale=value).validate()
    with pytest.raises(ValueError, match="residual_input_scale"):
        DeepTDA(residual_input_scale=value)


@pytest.mark.parametrize("value", [1, 20, 0.25, np.float32(20), np.float64(0.5), np.int64(3)])
def test_positive_real_config_scale(value):
    config = TDAConfig(residual_input_scale=value).validate()
    assert type(config.residual_input_scale) is float
    assert config.to_dict()["residual_input_scale"] == float(value)


def test_legacy_config_default():
    assert TDAConfig().validate().residual_input_scale == 1.0
    assert DeepTDA({"steps": 0}).config.residual_input_scale == 1.0
