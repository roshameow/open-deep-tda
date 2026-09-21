"""Selection is recomputed from complete TRAIN-only records, not trusted labels."""
import importlib
import json
from pathlib import Path

import pytest

from open_deep_tda import TDAConfig


@pytest.fixture
def confirmation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    return importlib.import_module('confirm_geometry_optimization')


def records(path, candidates, protocol='production-geometry-ablation-v1'):
    runs = []
    for i, candidate in enumerate(candidates):
        overrides = {k: v for k, v in candidate.items() if k != 'name'}
        runs.append(dict(candidate=candidate['name'], seed=0,
                         config=TDAConfig(**overrides).validate().to_dict(),
                         population_geometry=dict(knn_overlap=.1 + i * .01, trustworthiness=.9)))
    result = dict(plan=dict(protocol=protocol, seeds=[0], common={}, candidates=candidates),
                  runs=runs, selected_candidate=candidates[-1]['name'])
    path.write_text(json.dumps(result))
    return result


def base(path):
    return records(path, [dict(name='stress_600', steps=600),
                          dict(name='graph_weak_2400', steps=2400,
                               geometry_objective='fuzzy_graph', lambda_h0=.1, lambda_h1=.01)])


def test_selection_complete_stage_and_legacy_default(confirmation, tmp_path):
    path = tmp_path / 'pilot.json'
    result = base(path)
    for row in result['runs']:
        row['config'].pop('residual_input_scale')
    path.write_text(json.dumps(result))
    selection, arms = confirmation.load_selection(path)
    assert selection['selected_candidate'] == 'graph_weak_2400'
    assert len(arms) == 2
    assert all(arm['config']['residual_input_scale'] == 1 for arm in arms)


@pytest.mark.parametrize('corruption', ['winner', 'missing', 'config', 'nan', 'seed'])
def test_selection_rejects_inconsistent_pilot(confirmation, tmp_path, corruption):
    path = tmp_path / 'pilot.json'
    result = base(path)
    if corruption == 'winner':
        result['selected_candidate'] = 'stress_600'
    elif corruption == 'missing':
        result['runs'].pop()
    elif corruption == 'config':
        result['runs'][0]['config']['steps'] = 999
    elif corruption == 'nan':
        result['runs'][0]['population_geometry']['knn_overlap'] = float('nan')
    else:
        result['runs'][0]['seed'] = 10
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        confirmation.load_selection(path)


def test_conditioning_stage_does_not_discard_better_first_stage(confirmation, tmp_path):
    first, second = tmp_path / 'one.json', tmp_path / 'two.json'
    original = base(first)
    original['runs'][-1]['population_geometry']['knn_overlap'] = .8
    first.write_text(json.dumps(original))
    records(second, [dict(name='graph_r1_conditioned', steps=2400, geometry_objective='fuzzy_graph'),
                     dict(name='graph_weak_conditioned', steps=2400, geometry_objective='fuzzy_graph')],
            protocol='residual-conditioning-ablation-v1')
    selection, arms = confirmation.load_selection(first, second)
    assert selection['selected_candidate'] == 'graph_weak_2400'
    assert len(arms) == 4
    assert len(selection['stages']) == 2
