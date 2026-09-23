"""Optional SOURCE-only three-H1 witness proposal on all original row IDs.

Ripser (Bauer; Ripser.py by Tralie, Saul and Bar-On) is imported only with
explicit consent. This module does not import Torch or the guided trainer:
run the external solver in an isolated process when training uses Torch+Numba.
The external solver is NOT subject to the local memory/time/work ceilings;
OS/process isolation is required for hard RSS/wall-clock limits. No target,
labels, subsampling, random seeds or alternate bar/interval attempts are used.
A proposal is never a certificate until fresh exact sparse source reduction.
"""
from dataclasses import dataclass, field
import math
from numbers import Integral
import time

import numpy as np

from .structural_constructive import _mst
from .structural_sparse_h1 import (H1Limits, ResourceLimitError,
                                   validate_distance_matrix, analyze_sparse_h1,
                                   _survival_graph, _triangles)

__all__ = ['FamilyLimits', 'FamilyResult', 'auto_source_h1_family']


@dataclass(frozen=True)
class FamilyLimits:
    """Lowerable local ceilings; external Ripser allocations cannot be capped here."""
    max_vertices: int = 300
    max_edges: int = 12_000
    max_triangles: int = 250_000
    max_operations: int = 3_000_000
    max_seconds: float = 300.
    max_word_ops: int = 500_000_000
    max_storage_words: int = 2_000_000


@dataclass(frozen=True)
class FamilyResult:
    certified: bool
    cycles: tuple = ()
    birth: object = None
    survival: object = None
    diagnostics: dict = field(default_factory=dict)
    source: object = None


def _validate_limits(limits):
    if not isinstance(limits, FamilyLimits):
        raise ValueError('limits must be FamilyLimits')
    for name, cap in vars(FamilyLimits()).items():
        value = getattr(limits, name)
        if name == 'max_seconds':
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= cap:
                raise ValueError('max_seconds must be finite and within the hard ceiling')
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value <= cap:
            raise ValueError(name + ' must be a nonnegative integer within the hard ceiling')


class _Budget:
    def __init__(self, limits):
        self.limits = limits
        self.start = time.monotonic()
        self.operations = 0

    def charge(self, count=1):
        if count > self.limits.max_operations - self.operations:
            raise ResourceLimitError('max_operations exceeded in family selection')
        self.operations += count
        self.check_time()

    def check_time(self):
        if time.monotonic() - self.start > self.limits.max_seconds:
            raise ResourceLimitError('max_seconds exceeded in family selection')


def _fundamental(edge, parent, depth, max_edges):
    u, v = edge
    chain = [edge]
    while u != v:
        if len(chain) >= max_edges:
            raise ResourceLimitError('fundamental cycle exceeds chain-edge cap')
        if depth[u] < depth[v]:
            u, v = v, u
        p = parent[u]
        chain.append((min(u, p), max(u, p)))
        u = p
    return tuple(sorted(chain))


def auto_source_h1_family(D, *, allow_external=False, limits=FamilyLimits()):
    """Select exactly three longest finite positive Ripser H1 bars (ties: birth, index).

    Use the intersection of their fixed 20%-80% interiors; refuse without
    retry if empty. Check all survival triangles against all three cochains,
    then greedily choose three source-filtration-ordered global-MST fundamental cycles
    born at the common birth radius with independent GF(2) cochain pairings.
    Independently certify those same-ID cycles in the complete SOURCE flag
    complex with analyze_sparse_h1. Failure has cycles=(); budget exhaustion
    raises ResourceLimitError, not a partial family. Solver errors propagate;
    missing/malformed optional output refuses. No full-barcode or target claim.
    Inputs must not be concurrently mutated. External calls are unbounded by
    these logical limits: execute in an OS-isolated process for hard limits.
    """
    if allow_external is not True:
        raise ValueError('external Ripser requires explicit allow_external=True')
    _validate_limits(limits)
    budget = _Budget(limits)
    h1_limits = H1Limits(max_vertices=limits.max_vertices, max_edges=limits.max_edges,
                         max_triangles=limits.max_triangles,
                         max_word_ops=limits.max_word_ops,
                         max_storage_words=limits.max_storage_words,
                         max_cycles=3, max_chain_edges=3 * limits.max_vertices)
    D = validate_distance_matrix(D, limits=h1_limits)
    n = len(D)
    # These limits are unconditionally insufficient for the three selected
    # positive source H1 classes; refuse before launching optional Ripser.
    # Radius-dependent edge/triangle ceilings can only be checked AFTER the
    # external bar proposal supplies a fixed survival radius.
    if limits.max_edges < 6:
        raise ResourceLimitError('max_edges insufficient for three source H1 classes')
    if limits.max_operations < n*n:
        raise ResourceLimitError('max_operations insufficient for source graph preflight')
    if limits.max_seconds == 0:
        raise ResourceLimitError('max_seconds exhausted before external solver')
    a = b = source = None
    diagnostics = {'vertices': n, 'scope': 'all source rows; selected family only',
                   'external_solver': 'Ripser / Ripser.py',
                   'external_resources': 'not bounded; use isolated process',
                   'interval_fractions': (0.2, 0.8)}

    def fail(reason):
        return FamilyResult(False, (), a, b, dict(diagnostics, reason=reason), source)

    if n < 4:
        return fail('fewer_than_three_finite_positive_h1_bars')
    budget.check_time()
    try:
        from ripser import ripser
    except ImportError:
        return fail('optional_ripser_unavailable')
    output = ripser(D.copy(), distance_matrix=True, maxdim=1, coeff=2, do_cocycles=True)
    budget.check_time()
    if not isinstance(output, dict):
        return fail('malformed_external_output')
    dgms, cocycles = output.get('dgms'), output.get('cocycles')
    if not isinstance(dgms, (list, tuple)) or len(dgms) < 2:
        return fail('malformed_h1_diagram')
    bars = dgms[1]
    max_pairs = n * (n-1) // 2
    if (not isinstance(bars, np.ndarray) or bars.ndim != 2 or bars.shape[1] != 2
            or bars.dtype.kind not in 'fiu' or len(bars) > max_pairs):
        return fail('malformed_h1_diagram')
    options = []
    for index, row in enumerate(bars):
        budget.charge()
        birth, death = map(float, row)
        if math.isfinite(birth) and math.isfinite(death) and 0 <= birth < death:
            options.append((-(death-birth), birth, index, death))
    if len(options) < 3:
        return fail('fewer_than_three_finite_positive_h1_bars')
    selected = sorted(options)[:3]
    indices = tuple(item[2] for item in selected)
    diagnostics['bar_indices'] = indices
    diagnostics['bars'] = tuple((item[1], item[3]) for item in selected)
    interior = [(birth + .2*(-neg), death - .2*(-neg))
                for neg, birth, _, death in selected]
    if any(not birth < lo < hi < death for (neg, birth, _, death), (lo, hi) in zip(selected, interior)):
        return fail('unrepresentable_fixed_interior')
    a, b = max(lo for lo, _ in interior), min(hi for _, hi in interior)
    if not a < b:
        return fail('no_common_fixed_interior')
    budget.charge(n*n)  # graph pair scan; logical budget, not Python RSS
    forward, edge_ids = _survival_graph(D, b, h1_limits)
    budget.check_time()
    # Complete bounded triangle count precedes EVERY cochain refusal; a bad
    # first cochain cannot hide a triangle resource exhaustion later in stream.
    triangles=0
    for _ in _triangles(forward):
        triangles+=1
        if triangles > limits.max_triangles:
            raise ResourceLimitError('max_triangles exceeded in family preflight')
        budget.charge()
    if (not isinstance(cocycles, (list, tuple)) or len(cocycles) < 2
            or not isinstance(cocycles[1], (list, tuple)) or len(cocycles[1]) != len(bars)):
        return fail('malformed_h1_cocycles')
    supports = []
    for index in indices:
        cochain = cocycles[1][index]
        if (not isinstance(cochain, np.ndarray) or cochain.ndim != 2
                or cochain.shape[1] != 3 or cochain.dtype.kind not in 'iu'
                or len(cochain) > max_pairs):
            return fail('malformed_selected_cochain')
        support = set()
        for u, v, coefficient in cochain:
            budget.charge()
            u, v, coefficient = int(u), int(v), int(coefficient)
            if not (0 <= u < n and 0 <= v < n) or u == v:
                return fail('malformed_selected_cochain')
            edge = (min(u, v), max(u, v))
            if coefficient % 2 and edge in edge_ids:
                support.symmetric_difference_update((edge,))
        supports.append(support)
    for i, j, k in _triangles(forward):
        budget.charge(3)
        if any(((i, j) in s) ^ ((i, k) in s) ^ ((j, k) in s) for s in supports):
            return fail('cochain_not_closed_on_exact_survival_complex')
    diagnostics.update(survival_edges=len(edge_ids), survival_triangles=triangles)
    budget.charge(n*n)  # dense Prim comparison work, conservatively charged
    tree = set(_mst(D))
    budget.check_time()
    adjacency = [[] for _ in range(n)]
    for u, v in tree:
        adjacency[u].append(v)
        adjacency[v].append(u)
    parent, depth, parity = [-1]*n, [0]*n, [0]*n
    parent[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            budget.charge()
            if parent[v] != -1:
                continue
            parent[v], depth[v] = u, depth[u]+1
            parity[v] = parity[u] ^ sum(1 << t for t, s in enumerate(supports)
                                         if (min(u, v), max(u, v)) in s)
            stack.append(v)
    basis = {}
    cycles = []
    edges = []
    # Deterministic source VR filtration order: shortest birth chord first,
    # then original vertex IDs. This is source-only and precedes target access.
    candidates = sorted((float(D[u, v]), u, v) for u, v in edge_ids
                        if (u, v) not in tree and D[u, v] <= a)
    budget.charge(len(edge_ids) * max(1, (len(edge_ids)-1).bit_length()))
    for _, u, v in candidates:
        budget.charge()
        edge = (u, v)
        vector = parity[u] ^ parity[v] ^ sum(1 << t for t, s in enumerate(supports) if edge in s)
        reduced = vector
        for pivot in sorted(basis, reverse=True):
            budget.charge()
            if (reduced >> pivot) & 1:
                reduced ^= basis[pivot]
        if not reduced:
            continue
        cycle = _fundamental(edge, parent, depth, 3*n)
        budget.charge(len(cycle))
        if any(D[x, y] > a for x, y in cycle):
            continue
        basis[reduced.bit_length()-1] = reduced
        cycles.append(cycle)
        edges.append(edge)
        if len(cycles) == 3:
            break
    if len(cycles) != 3:
        return fail('no_three_independent_birth_fundamental_cycles')
    diagnostics['fundamental_edges'] = tuple(edges)
    budget.check_time()
    source = analyze_sparse_h1(D, tuple(cycles), a, b, limits=h1_limits)
    budget.check_time()
    if not source.certified or source.rank != 3:
        return fail('exact_source_rejected: ' + source.reason)
    return FamilyResult(True, tuple(cycles), a, b,
                        dict(diagnostics, reason='certified_selected_source_family'), source)
