"""Predictor schema compatibility and origin declarations; no optional JIT required."""
import io
import json
import zipfile

import numpy as np
import pytest

from open_deep_tda.graph_embedding import GraphEmbedding


def _model():
    X = np.random.default_rng(518).normal(size=(16, 3))
    angles = np.arange(16) * (2 * np.pi / 16)
    Z = np.column_stack((np.cos(angles), np.sin(angles)))
    return GraphEmbedding.from_layout(X, Z)


def _rewrite_metadata(source, target, change):
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    raw = np.load(io.BytesIO(members['metadata.npy']), allow_pickle=False)
    metadata = json.loads(raw.tobytes().decode('utf8'))
    change(metadata)
    stream = io.BytesIO()
    np.save(stream, np.frombuffer(json.dumps(metadata).encode('utf8'), dtype=np.uint8),
            allow_pickle=False)
    members['metadata.npy'] = stream.getvalue()
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_current_version_without_origin_remains_loadable(tmp_path):
    model = _model()
    current, legacy = tmp_path / 'current.npz', tmp_path / 'legacy.npz'
    with pytest.warns(UserWarning, match='training features'):
        model.save(current)
    _rewrite_metadata(current, legacy, lambda meta: meta.pop('layout_origin'))
    loaded = GraphEmbedding.load(legacy)
    assert loaded.diagnostics_['layout_origin'] == 'unknown'
    np.testing.assert_array_equal(loaded.embedding_, model.embedding_)
    np.testing.assert_array_equal(loaded.transform(model.reference_[:2]),
                                  model.transform(model.reference_[:2]))
    roundtrip = tmp_path / 'roundtrip.npz'
    with pytest.warns(UserWarning):
        loaded.save(roundtrip)
    assert GraphEmbedding.load(roundtrip).diagnostics_['layout_origin'] == 'unknown'


@pytest.mark.parametrize('bad_origin', [None, '', 'certified', True, 2, [], {}, ['external']])
def test_invalid_origin_declaration_rejected(tmp_path, bad_origin, monkeypatch):
    model = _model()
    good, bad = tmp_path / 'good.npz', tmp_path / 'bad.npz'
    with pytest.warns(UserWarning):
        model.save(good)
    _rewrite_metadata(good, bad, lambda meta: meta.update(layout_origin=bad_origin))

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid metadata must be rejected before allocating model arrays')

    monkeypatch.setattr(np.lib.format, 'read_array', forbidden)
    with pytest.raises(ValueError, match='layout_origin'):
        GraphEmbedding.load(bad)


def test_version2_absent_domain_loads_historical_unbounded(tmp_path):
    rng = np.random.default_rng(2018)
    X, Z = rng.normal(size=(24, 3)), rng.normal(size=(24, 2))
    model = GraphEmbedding.from_layout(X, Z, mapping_domain='unbounded')
    current, legacy = tmp_path / 'current.npz', tmp_path / 'v2.npz'
    with pytest.warns(UserWarning):
        model.save(current)

    def downgrade(meta):
        meta['version'] = 2
        meta['config'].pop('mapping_domain')
        meta.pop('layout_origin', None)

    _rewrite_metadata(current, legacy, downgrade)
    loaded = GraphEmbedding.load(legacy)
    Q = np.random.default_rng(19).normal(size=(30, 3))
    expected = model.transform(Q)
    np.testing.assert_array_equal(loaded.transform(Q), expected)
    bounded = GraphEmbedding.from_layout(X, Z, mapping_domain='train_hull')
    assert not np.array_equal(bounded.transform(Q), expected)
    upgraded = tmp_path / 'upgraded.npz'
    with pytest.warns(UserWarning):
        loaded.save(upgraded)
    with np.load(upgraded, allow_pickle=False) as archive:
        meta = json.loads(archive['metadata'].tobytes().decode('utf8'))
    assert meta['version'] == 3
    assert meta['config']['mapping_domain'] == 'unbounded'
    assert meta['layout_origin'] == 'unknown'


@pytest.mark.parametrize('bad_domain', [None, '', 'bounded', True, 2, [], {}])
def test_invalid_saved_mapping_domain_rejected_before_arrays(tmp_path, bad_domain, monkeypatch):
    model = _model()
    good, bad = tmp_path / 'good.npz', tmp_path / 'bad.npz'
    with pytest.warns(UserWarning):
        model.save(good)
    _rewrite_metadata(good, bad,
                      lambda meta: meta['config'].update(mapping_domain=bad_domain))

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid policy must fail before allocating model arrays')

    monkeypatch.setattr(np.lib.format, 'read_array', forbidden)
    with pytest.raises(ValueError, match='mapping_domain'):
        GraphEmbedding.load(bad)


def test_version3_missing_policy_rejected(tmp_path):
    model = _model()
    good, bad = tmp_path / 'good.npz', tmp_path / 'bad.npz'
    with pytest.warns(UserWarning):
        model.save(good)
    _rewrite_metadata(good, bad, lambda meta: meta['config'].pop('mapping_domain'))
    with pytest.raises(ValueError):
        GraphEmbedding.load(bad)
