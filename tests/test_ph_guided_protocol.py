"""Publication boundary: guided PH training is not a graph-only image or OOS claim."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_reviewed_scalar_figures_have_all_rows_and_explicit_scope():
    data=json.loads((ROOT/'benchmarks/results/ph_guided_selected_cycles.json').read_text())
    assert data['status']=='completed_conditional_opt_in'
    assert 'PH-trained DeepTDA' in data['method']
    assert {r['dataset'] for r in data['results']}=={'K4','COIL20-1'}
    for row in data['results']:
        assert row['source_rows'] in (72,300)
        assert row['source_cycles']==row['trained_selected_rank']
        assert row['h0_tolerance']==.05
        assert row['trained_h0_max_merge_error']<=.05
        fig=row['figure']
        assert fig['all_rows'] and fig['full_extent']
        assert fig['method_order']==['PH-Regularized Embedding (DeepTDA)','UMAP genuine',
                                     'TopoAE++ adapted author core']
        asset=ROOT/fig['file']
        assert asset.exists() and asset.suffix=='.png'
        assert hashlib.sha256(asset.read_bytes()).hexdigest()==fig['sha256']
        assert row['methods']['PHRE source-guided (same MLP)']['neighbor_overlap15'] < 1
    assert any('unsupported' in text for text in data['known_limits'])
    assert all('no omitted-query' in text or 'no unseen' in text or 'OOS' in text
               for text in [data['results'][0]['caveats'][0],data['known_limits'][1]])


def test_renderer_rejects_missing_or_nonfinite_points():
    spec=importlib.util.spec_from_file_location('ph_renderer',ROOT/'benchmarks/render_ph_guided.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with pytest.raises(ValueError,match='all'):
        m.matrix(np.zeros((15,2)),16)
    with pytest.raises(ValueError,match='finite'):
        m.matrix(np.array([[float('nan'),1.]]),1)


def test_readmes_use_project_ph_fits_only_no_research_history_leak():
    for filename in ('README.md','README.zh-CN.md'):
        text=(ROOT/filename).read_text()
        assert all('assets/ph-guided-'+key+'.png' in text for key in ('k4','coil72'))
        assert 'fit_with_topology_guidance' in text
        assert 'TopoAE++' in text and 'UMAP' in text
        assert not any(name in text for name in ('GraphEmbedding','docs/research','v0.3.0'))
        assert 'no' in text.lower() if filename=='README.md' else '不' in text
