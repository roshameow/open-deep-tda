"""The confirmation must honor complete, preregistered label-free selection."""
import importlib
import json
from pathlib import Path

import pytest
from open_deep_tda import TDAConfig


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    return importlib.import_module('confirm_neighbor_nce')


def fixture_result():
    candidates=[dict(name='graph_2400',geometry_objective='fuzzy_graph',steps=2400),
                dict(name='nce16_2400',geometry_objective='neighbor_nce',steps=2400)]
    rows=[]
    for i,c in enumerate(candidates):
        cfg=TDAConfig(**{k:v for k,v in c.items() if k!='name'}).validate().to_dict()
        rows.append(dict(candidate=c['name'],seed=0,config=cfg,
                         population_geometry=dict(knn_overlap=.1+i*.01,trustworthiness=.9)))
    return dict(plan=dict(protocol='fashion-train-neighbor-nce-v1',seeds=[0],common={},candidates=candidates),
                runs=rows,selected_candidate='nce16_2400')


def test_selection_recomputes_winner(protocol,tmp_path):
    p=tmp_path/'pilot.json';p.write_text(json.dumps(fixture_result()))
    selected,previous=protocol.selection(p)
    assert selected['candidate']=='nce16_2400'
    assert previous['candidate']=='graph_2400'


@pytest.mark.parametrize('change',['missing','selection','config','metric','seed'])
def test_rejects_corrupt_or_incomplete_results(protocol,tmp_path,change):
    result=fixture_result()
    if change=='missing':result['runs'].pop()
    elif change=='selection':result['selected_candidate']='graph_2400'
    elif change=='config':result['runs'][1]['config']['steps']=123
    elif change=='metric':result['runs'][1]['population_geometry']['knn_overlap']=float('nan')
    else:result['runs'][1]['seed']=1
    p=tmp_path/'pilot.json';p.write_text(json.dumps(result))
    with pytest.raises(ValueError):protocol.selection(p)
