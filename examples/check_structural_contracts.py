"""Executable structural acceptance cases, NOT a dimensionality reducer.

Run after installation: python examples/check_structural_contracts.py
All data are analytic fixtures. No datasets, training, source-class selection,
layout optimization, or claims about unseen/omitted population rows.
"""
from dataclasses import asdict
import json

import numpy as np

from open_deep_tda.topology import distance_matrix
from open_deep_tda.structural_h0 import compare_h0
from open_deep_tda.structural_h1 import check_h1_witnesses


SQUARE = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
CYCLE = [(0, 1), (1, 2), (2, 3), (3, 0)]


def h1_case(source, target, cycles):
    result = check_h1_witnesses(distance_matrix(source), distance_matrix(target),
                                cycles, birth_radius=1., survival_radius=1.2)
    return dict(asdict(result), accepted=result.accepted,
                n_vertices=len(source), birth_radius=1., survival_radius=1.2,
                scope='Only the specified same-ID source classes on all supplied rows')


def run_cases():
    cases = {}
    cases['intact_square'] = h1_case(SQUARE, SQUARE.copy(), [CYCLE])
    # Old H1 critical edges (2,3) and (0,2) retain lengths 1 and sqrt(2),
    # but the source perimeter no longer represents a surviving target class.
    broken = np.array([[0., 0.], [3., .1], [1.4, .2],
                       [1.4 + np.sqrt(1. - .35**2), -.15]])
    cases['missing_cycle'] = h1_case(SQUARE, broken, [CYCLE])
    # The target point set/barcode is unchanged, but the same-ID witness is not.
    cases['same_barcode_wrong_ids'] = h1_case(SQUARE, SQUARE[[0, 2, 1, 3]], [CYCLE])
    # A vertex outside the specified perimeter supplies filling triangles.
    cases['filled_by_extra_vertex'] = h1_case(
        np.vstack([SQUARE, [10., 10.]]), np.vstack([SQUARE, [.5, .5]]), [CYCLE])
    # Both image cycles can survive individually, yet represent one class.
    cases['two_classes_merge'] = h1_case(
        np.vstack([SQUARE, SQUARE + [4., 0.]]), np.vstack([SQUARE, SQUARE]),
        [CYCLE, [(i + 4, j + 4) for i, j in CYCLE]])
    bridge = compare_h0(distance_matrix(np.array([[0.], [1.], [10.], [11.]])),
                        distance_matrix(np.array([[0.], [1.], [100.], [101.]])))
    cases['wrong_global_bridge'] = bridge

    assert cases['intact_square']['accepted']
    assert all(not cases[k]['accepted'] for k in
               ['missing_cycle', 'same_barcode_wrong_ids', 'filled_by_extra_vertex', 'two_classes_merge'])
    assert all(w['survives'] for w in cases['two_classes_merge']['witnesses'])
    assert cases['two_classes_merge']['surviving_rank'] == 1
    assert not bridge['certified_within_tolerance']
    assert bridge['max_merge_error'] == 90.
    return cases


if __name__ == '__main__':
    print(json.dumps(run_cases(), indent=2, allow_nan=False))
