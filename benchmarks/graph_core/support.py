"""Standalone production mapping helper; no package/Torch import or local paths."""
import importlib.util
import sys
from pathlib import Path


def production_graph_mapping():
    """Return the exact production CompactMap API without package __init__."""
    path=Path(__file__).resolve().parents[2]/'python/open_deep_tda/_graph_mapping.py'
    spec=importlib.util.spec_from_file_location('benchmark_production_graph_mapping',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module
