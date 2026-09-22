"""End-to-end acceptance gates for the independent structural-checking core."""
import importlib.util
from pathlib import Path


def test_analytic_acceptance_cases():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'check_structural_contracts.py'
    spec = importlib.util.spec_from_file_location('structural_acceptance_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = module.run_cases()
    assert cases['intact_square']['accepted']
    assert not cases['same_barcode_wrong_ids']['accepted']
    assert cases['filled_by_extra_vertex']['witnesses'][0]['edges_present']
    assert not cases['filled_by_extra_vertex']['witnesses'][0]['survives']
    assert all(w['survives'] for w in cases['two_classes_merge']['witnesses'])
    assert not cases['two_classes_merge']['all_classes_independent']
    assert cases['wrong_global_bridge']['max_merge_error_witness']['source_merge_distance'] == 9.
    assert cases['wrong_global_bridge']['max_merge_error_witness']['target_merge_distance'] == 99.
