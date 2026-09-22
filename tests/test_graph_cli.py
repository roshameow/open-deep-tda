"""Public graph entrypoints; no benchmark data or optional parent JIT import."""
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest


def test_graph_import_and_cli_help_do_not_import_torch_or_numba():
    code = '''
import sys
import open_deep_tda
from open_deep_tda import GraphEmbedding, TDAConfig
from open_deep_tda import cli
assert "torch" not in sys.modules
assert "numba" not in sys.modules
assert "pynndescent" not in sys.modules
assert GraphEmbedding.__module__ == "open_deep_tda.graph_embedding"
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, timeout=30)
    result = subprocess.run([sys.executable, '-m', 'open_deep_tda', '--help'],
                            check=True, capture_output=True, text=True, timeout=30)
    assert 'graph-fit' in result.stdout and 'graph-transform' in result.stdout


def test_graph_cli_fit_and_safe_checkpoint_transform(tmp_path):
    if importlib.util.find_spec('numba') is None:
        pytest.skip('optional isolated graph worker requires numba')
    X = np.random.default_rng(912).normal(size=(32, 4))
    Q = X[:3] + .03
    np.save(tmp_path / 'train.npy', X)
    np.save(tmp_path / 'query.npy', Q)
    run = subprocess.run([sys.executable, '-m', 'open_deep_tda', 'graph-fit',
        '--input', str(tmp_path / 'train.npy'), '--validation', str(tmp_path / 'query.npy'),
        '--epochs', '2', '--output', str(tmp_path / 'fit')], capture_output=True,
        text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)['reference_data_in_checkpoint'] is True
    model = tmp_path / 'fit' / 'model.npz'
    assert model.exists()
    run = subprocess.run([sys.executable, '-m', 'open_deep_tda', 'graph-transform',
        '--model', str(model), '--input', str(tmp_path / 'query.npy'),
        '--output', str(tmp_path / 'pred.npy'), '--diagnostics', str(tmp_path / 'diag.json')],
        capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    np.testing.assert_array_equal(np.load(tmp_path / 'pred.npy'),
                                  np.load(tmp_path / 'fit' / 'validation_embedding.npy'))
    assert json.loads((tmp_path / 'diag.json').read_text())['query_count'] == 3
