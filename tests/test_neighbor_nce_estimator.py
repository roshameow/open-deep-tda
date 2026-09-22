"""Production wiring for opt-in anchored conditional-neighbor optimization."""
import json

import numpy as np
import pytest

from open_deep_tda import DeepTDA, TDAConfig


@pytest.mark.parametrize('hard', [0, 2])
@pytest.mark.parametrize('topology', [False, True])
def test_fit_transform_compact_roundtrip(tmp_path, hard, topology):
    X = np.random.default_rng(5).normal(size=(40, 6)).astype('float32')
    model = DeepTDA(geometry_objective='neighbor_nce', contrastive_candidates=4,
                    contrastive_hard_negatives=hard, steps=3, batch_size=12,
                    warmup_steps=0, n_neighbors=4, h0_size=8, h1_size=8,
                    subset_bank_size=1, evaluation_size=8,
                    lambda_h0=.1 if topology else 0, lambda_h1=.01 if topology else 0).fit(X[:30])
    Z = model.transform(X[30:])
    assert np.isfinite(Z).all()
    diag = model.geometry_diagnostics_
    assert diag['positive_mode'] == 'conditional_neighbor_nce'
    assert diag['candidates_per_anchor'] == 4
    assert all(row['separation'] == 0 for row in model.history_)
    json.dumps(model.report_, allow_nan=False)
    restored = DeepTDA.load(model.save(tmp_path / 'nce.pt', include_training_data=False))
    np.testing.assert_array_equal(Z, restored.transform(X[30:]))
    assert restored.config.contrastive_candidates == 4


@pytest.mark.parametrize('options', [
    {'contrastive_candidates': 0}, {'contrastive_candidates': True},
    {'contrastive_candidates': 1.5}, {'contrastive_hard_negatives': -1},
    {'contrastive_candidates': 2, 'contrastive_hard_negatives': 3},
    {'contrastive_temperature': 0}, {'contrastive_temperature': float('nan')},
    {'contrastive_temperature': True},
])
def test_invalid_config(options):
    with pytest.raises(ValueError):
        TDAConfig(**options).validate()


def test_duplicate_and_complete_graph_safe():
    X = np.tile([[0., 1.], [1., 0.]], (6, 1)).astype('float32')
    model = DeepTDA(geometry_objective='neighbor_nce', steps=2, n_neighbors=len(X)-1,
                    evaluation_size=6, h0_size=6, h1_size=6, lambda_h0=0, lambda_h1=0).fit(X)
    assert np.isfinite(model.embedding_).all()
    assert all(row['near'] == 0 for row in model.history_)
