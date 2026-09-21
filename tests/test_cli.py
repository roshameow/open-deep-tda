import json
import numpy as np
from open_deep_tda.cli import main
from open_deep_tda.benchmark import run_benchmark
from open_deep_tda import TDAConfig
from open_deep_tda.datasets import make_dataset


def test_cli_full_pipeline(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(steps=4, warmup_steps=1, h0_size=12, h1_size=12,
                                     evaluation_size=12, subset_bank_size=3, topology_interval=1)))
    out = tmp_path / 'demo'
    assert main(['demo', '--samples', '32', '--features', '5', '--config', str(config),
                 '--output', str(out)]) == 0
    for filename in ('model.pt', 'embedding.npy', 'metrics.json', 'report.html', 'mapper.json', 'history.json'):
        assert (out / filename).is_file()
    text = (out / 'report.html').read_text()
    assert '<svg' in text and 'cdn' not in text.lower()
    prediction = tmp_path / 'new.npy'
    ood = tmp_path / 'ood.npy'
    assert main(['transform', '--input', str(out / 'validation.npy'), '--model', str(out / 'model.pt'),
                 '--output', str(prediction), '--ood-output', str(ood)]) == 0
    assert np.load(prediction).shape == (6, 2)
    assert np.load(ood).shape == (6,)
    result = tmp_path / 'eval.json'
    assert main(['evaluate', '--reference', str(out / 'reference.npy'), '--embedding', str(out / 'embedding.npy'),
                 '--topology-size', '12', '--output', str(result)]) == 0
    assert json.loads(result.read_text())['sampling']['same_ids_source_target']
    assert main(['demo', '--samples', '32', '--config', str(config), '--output', str(out)]) == 2


def test_csv_fit(tmp_path):
    X = make_dataset('circle', 16, 4, seed=3)[0]
    data = tmp_path / 'data.csv'
    np.savetxt(data, X, delimiter=',', header='a,b,c,d', comments='')
    assert main(['fit', '--input', str(data), '--skip-header', '1', '--steps', '0',
                 '--output', str(tmp_path / 'fit')]) == 0


def test_small_benchmark():
    X = make_dataset('circle', 16, 4, seed=3)[0]
    cfg = TDAConfig(steps=2, warmup_steps=0, h1_size=8, h0_size=8,
                    evaluation_size=8, subset_bank_size=1)
    result = run_benchmark(X, cfg, seeds=[0], methods=['pca', 'ae', 'topoae_h0', 'deep_tda', 'no_h0', 'no_h1', 'no_critical', 'local_only'])
    assert len(result['results']) == 8
    assert all(r['status'] == 'ok' for r in result['results'])
    assert all(r['metrics']['sampling']['same_ids_source_target'] for r in result['results'])
    json.dumps(result, allow_nan=False)
