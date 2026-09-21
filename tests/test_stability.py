import json
import numpy as np
import pytest
from open_deep_tda.evaluation import evaluate_stability
from open_deep_tda.cli import load_array


def test_paired_resampling_and_bootstrap():
    t = np.linspace(0, 2*np.pi, 16, endpoint=False)
    X = np.column_stack([np.cos(t), np.sin(t)])
    for bootstrap in [False, True]:
        report = evaluate_stability(X, X, sample_sizes=[8, 16], repeats=2, seed=7, bootstrap=bootstrap)
        assert len(report['results']) == 4
        for row in report['results']:
            assert row['h0_raw_cost'] == 0
            assert row['h1_raw_cost'] == 0
            assert row['h1_bottleneck'] == 0
            if not bootstrap:
                assert row['distinct_samples'] == row['size']
        assert report == evaluate_stability(X, X, sample_sizes=[8, 16], repeats=2, seed=7, bootstrap=bootstrap)
        json.dumps(report, allow_nan=False)


def test_stability_budgets():
    X = np.ones((4, 2))
    with pytest.raises(ValueError):
        evaluate_stability(X, X, sample_sizes=[256])
    with pytest.raises(ValueError):
        evaluate_stability(X, X, repeats=100)


def test_csv_single_row_and_single_column(tmp_path):
    for shape in [(1, 4), (4, 1), (1, 1)]:
        X = np.arange(np.prod(shape)).reshape(shape)
        file = tmp_path / 'array.csv'
        np.savetxt(file, X, delimiter=',')
        np.testing.assert_array_equal(load_array(file), X)
