"""Offline synthetic teacher contract; optional 300-row stress via PH_GUIDED_STRESS=1."""
from itertools import combinations
import os

import numpy as np
import pytest

from open_deep_tda import _ph_guided_source as teacher
from open_deep_tda._ph_guided_source import (TeacherLimits, construct_source_teacher)
from open_deep_tda.structural_constructive import full_hierarchy, pairwise_planar
from open_deep_tda.structural_sparse_h1 import (ResourceLimitError, analyze_sparse_h1)


def distances(X):
    X = np.asarray(X, dtype=np.float64)
    return np.linalg.norm(X[:, None]-X[None, :], axis=-1)


def tetrahedron():
    # Six subdivided edges; no labels, blocks or private dataset.
    branch = np.array([[0., 0., 0.], [1., 0., 0.],
                       [.5, .8660254, 0.], [.5, .288675, .8164966]])
    points = list(branch)
    paths = {}
    for u, v in combinations(range(4), 2):
        path = [u]
        for fraction in (1/3, 2/3):
            path.append(len(points))
            points.append((1-fraction)*branch[u]+fraction*branch[v])
        path.append(v)
        paths[u, v] = path
    cycles = []
    for triangle in ((0, 1, 2), (0, 1, 3), (0, 2, 3)):
        cycles.append([edge for u, v in combinations(triangle, 2)
                       for edge in zip(paths[u, v], paths[u, v][1:])])
    X = np.array(points, dtype=np.float64)
    return distances(X), X, cycles


def test_single_square_source_only_and_owned_result():
    X = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
    D = distances(X)
    cycles = [[(0, 1), (1, 2), (2, 3), (3, 0)]]
    original = (D.copy(), X.copy())
    result = construct_source_teacher(D, X, cycles, 2.01, 2.1, .05)
    assert result.certified, result.reason
    assert result.embedding.shape == (4, 2) and not result.embedding.flags.writeable
    assert result.embedding.flags.owndata
    assert result.source.certified and result.target.certified
    assert np.max(np.abs(full_hierarchy(D)-full_hierarchy(pairwise_planar(result.embedding)))) <= .05
    np.testing.assert_array_equal(D, original[0])
    np.testing.assert_array_equal(X, original[1])
    assert result.diagnostics['strategy'] == 'single'
    with pytest.raises(ResourceLimitError, match='max_seconds'):
        construct_source_teacher(D, X, cycles, 2.01, 2.1, .05,
                                 limits=TeacherLimits(max_seconds=0))


def test_k4_union_four_faces_and_fail_closed():
    D, X, cycles = tetrahedron()
    assert analyze_sparse_h1(D, cycles, .35, .4).rank == 3
    old = X.copy(), D.copy()
    result = construct_source_teacher(D, X, cycles, .35, .4, 1.,
                                      strategy='subdivided_k4',
                                      limits=TeacherLimits(max_iterations=3))
    assert result.diagnostics['branch_ids'] == [0, 1, 2, 3]
    assert [c['interior_branch'] for c in result.diagnostics['candidates']] == [0, 1, 2, 3]
    assert all('initial' in c and 'repaired' in c for c in result.diagnostics['candidates'])
    assert result.diagnostics['side_rule'] == '2.5 * median positive source pair distance'
    assert not result.certified and result.embedding is None
    np.testing.assert_array_equal(X, old[0]); np.testing.assert_array_equal(D, old[1])
    # A closed pair of source classes cannot masquerade as a K4 witness union.
    unsupported = construct_source_teacher(D, X, cycles[:2], .35, .4, 1.,
                                           strategy='subdivided_k4')
    assert not unsupported.certified and unsupported.embedding is None
    assert 'unsupported witness union' in unsupported.reason


def test_k4_acceptance_requires_fresh_full_domain_checks(monkeypatch):
    # Source itself is a planar subdivided K4. Inject a source-only planar
    # candidate for one branch to isolate the independent acceptance gate.
    branch = np.array([[0., 0.], [3., 0.], [1.5, 2.598], [1.5, .866]])
    points = list(branch)
    paths = {}
    for u, v in combinations(range(4), 2):
        path = [u]
        for k in range(1, 5):
            path.append(len(points))
            points.append((1-k/5)*branch[u]+k/5*branch[v])
        path.append(v)
        paths[u, v] = path
    cycles = [[edge for u, v in combinations(triangle, 2)
               for edge in zip(paths[u, v], paths[u, v][1:])]
              for triangle in ((0, 1, 2), (0, 1, 3), (0, 2, 3))]
    X = np.array(points)
    D = distances(X)
    assert analyze_sparse_h1(D, cycles, .65, .7).certified
    original = teacher._scaffold
    monkeypatch.setattr(teacher, '_scaffold', lambda features, graph, branch_ids, paths, center, side:
                        X.copy() if center == 2 else original(features, graph, branch_ids, paths, center, side))
    result = construct_source_teacher(D, X, cycles, .65, .7, .05, strategy='subdivided_k4',
                                      limits=TeacherLimits(max_iterations=0))
    assert result.certified and result.source.certified and result.target.certified
    assert result.diagnostics['selected_branch'] == 2
    assert len(result.diagnostics['candidates']) == 4
    assert result.h0_error <= .05
    assert result.embedding.flags.owndata and not result.embedding.flags.writeable
    np.testing.assert_array_equal(result.embedding, X)
    # A later deterministic branch still consumes the declared audit budget;
    # its exhaustion/timeout must not be hidden by an earlier valid winner.
    for failure, expected in ((ResourceLimitError('late budget'), 'late budget'),
                              (teacher._TimedOut('late clock'), 'max_seconds')):
        def late(features, graph, branch_ids, paths, center, side):
            if center == 3:
                raise failure
            return X.copy() if center == 2 else original(features, graph, branch_ids,
                                                         paths, center, side)
        with monkeypatch.context() as patch:
            patch.setattr(teacher, '_scaffold', late)
            with pytest.raises(ResourceLimitError, match=expected):
                construct_source_teacher(D, X, cycles, .65, .7, .05,
                    strategy='subdivided_k4', limits=TeacherLimits(max_iterations=0))


def test_rejects_bad_input_budgets_and_unsupported_union():
    D, X, cycles = tetrahedron()
    with pytest.raises(ValueError, match='zero-diagonal'):
        construct_source_teacher(D+np.eye(len(D))*.01, X, cycles, .35, .4, .05)
    changed = D.copy(); changed[0, 1] += .001; changed[1, 0] += .001
    with pytest.raises(ValueError, match='Euclidean'):
        construct_source_teacher(changed, X, cycles, .35, .4, .05)
    with pytest.raises(ResourceLimitError):
        construct_source_teacher(D, X, cycles, .35, .4, .05,
                                 limits=TeacherLimits(max_vertices=15))
    with pytest.raises(ResourceLimitError, match='max_seconds'):
        construct_source_teacher(D, X, cycles, .35, .4, .05,
                                 strategy='subdivided_k4',
                                 limits=TeacherLimits(max_seconds=0))
    with pytest.raises(ValueError, match='max_iterations'):
        construct_source_teacher(D, X, cycles, .35, .4, .05,
                                 limits=TeacherLimits(max_iterations=401))
    with pytest.raises(ValueError, match='unknown strategy'):
        construct_source_teacher(D, X, cycles, .35, .4, .05, strategy='ripser')
    square = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
    unsupported = construct_source_teacher(distances(square), square,
                                            [[(0, 1), (1, 2), (2, 3), (3, 0)]],
                                            2.01, 2.1, .05, strategy='subdivided_k4')
    assert not unsupported.certified and unsupported.embedding is None
    assert 'unsupported witness union' in unsupported.reason


def test_high_feature_one_cycle_and_preflight_workspace_cap():
    theta = 2 * np.pi * np.arange(16) / 16
    X = np.zeros((16, 1024), dtype=np.float64)
    X[:, 0], X[:, 1] = np.cos(theta), np.sin(theta)
    D = distances(X)
    gamma = [[(i, (i+1) % 16) for i in range(16)]]
    with pytest.raises(ResourceLimitError, match='workspace'):
        construct_source_teacher(D, X, gamma, .5, .9, .05,
                                 limits=TeacherLimits(max_vertices=16, max_pairs=120,
                                                       max_distance_workspace_bytes=1024))
    result = construct_source_teacher(D, X, gamma, .5, .9, .05,
                                      limits=TeacherLimits(max_vertices=16, max_pairs=120))
    assert result.certified and result.embedding.shape == (16, 2)


@pytest.mark.skipif(os.getenv('PH_GUIDED_STRESS') != '1', reason='explicit 300-row synthetic opt-in')
def test_300_row_fixture_budget_preflight():
    # Entirely generated locally; no external file or dataset access.
    X = np.column_stack([np.arange(300, dtype=np.float64), np.zeros(300)])
    D = distances(X)
    with pytest.raises(ResourceLimitError, match='pair limit'):
        construct_source_teacher(D, X, [], 1., 2., .05,
                                 limits=TeacherLimits(max_pairs=44_849))
    result = construct_source_teacher(D, X, [], 1., 2., .05,
                                      limits=TeacherLimits(max_iterations=0))
    assert not result.certified and result.embedding is None
