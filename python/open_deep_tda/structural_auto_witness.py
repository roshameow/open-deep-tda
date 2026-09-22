"""Optional all-row H1 witness proposal and exact selected-family certification.

Provenance: the optional external solver is Ulrich Bauer's Ripser
(https://github.com/Ripser/ripser), accessed through Ripser.py by Christopher
Tralie, Nathaniel Saul and Rann Bar-On (https://github.com/scikit-tda/ripser.py;
https://doi.org/10.21105/joss.00925). No upstream implementation is copied here.
This original adapter imports Ripser lazily, only with explicit consent. It is
outside the bounded core structural construction and sparse-analysis APIs.

Explicit allow_external=True consent is mandatory. The in-process external
solver's work, allocations and output are NOT bounded by H1Limits beyond the
input vertex cap. Use process/OS isolation for hard memory/time limits. Local
survival graph/triangle checks and fresh exact reduction obey H1Limits; these
are logical budgets, not total RSS bounds or a shared external-solver budget.

Fixed policy: longest finite positive-length H1 bar, ties by increasing birth
then original diagram index; use its 20%-80% interior. Never retry another bar
or interval. Scan non-tree birth edges lexicographically in the global MST
fundamental-cycle basis, returning the first cycle with cochain pairing one.
Failure returns NO family and makes NO infeasibility or full-H1 claim.

Verified pitfalls (offline tests in test_structural_auto_witness.py): a cocycle
must annihilate triangles involving rows outside its support; float32 solver
endpoints cannot replace checks on the supplied float64 matrix. A rejected
fixed candidate must not trigger adaptive bar selection. Inputs must not be
mutated concurrently. External output is only a proposal, never a certificate.
"""
from dataclasses import dataclass
import math

import numpy as np

from .structural_constructive import _mst
from .structural_sparse_h1 import (H1Limits, H1Result, ResourceLimitError,
                                  validate_distance_matrix, analyze_sparse_h1,
                                  _survival_graph, _triangles)

__all__ = ["WitnessResult", "auto_global_witnesses"]


@dataclass(frozen=True)
class WitnessResult:
    """One fixed proposal's lifecycle, not a full persistence certificate.

    On success, ``cycles`` is an immutable selected family on original row IDs,
    and ``source`` is its fresh exact :class:`H1Result`. On refusal, ``cycles``
    is empty; thresholds and source may remain available for diagnostics if
    that stage was reached. The frozen record does not deep-freeze diagnostics.
    No embedding is constructed here: pass a certified family to
    ``structural_constructive.construct_global_layout`` separately. That API
    returns ``embedding=None`` on construction failure, not a partial layout.
    Neither selection nor construction failure proves general infeasibility.
    """

    certified: bool
    cycles: tuple
    birth: object
    survival: object
    diagnostics: dict
    source: object  # fresh H1Result, or None if no candidate reached analysis


def auto_global_witnesses(D, *, limits=H1Limits(), allow_external=False):
    """Propose one immutable cycle on original IDs and certify it exactly.

    D obeys structural_sparse_h1.validate_distance_matrix: a finite,
    nonnegative, exactly symmetric float64 ndarray with zero diagonal, without
    coercion or a triangle-inequality requirement. All supplied rows enter the
    closed-threshold flag complex (edge iff D[i,j] <= radius), including rows
    outside the cycle. Caller owns row identity and completeness. Certification
    is only of the selected class over GF(2), not all H1, a full barcode, or a
    source-to-target chain map. No rows are sampled and no RNG is used locally.
    birth/survival are the fixed interior thresholds, not diagram endpoints.
    Invalid input/consent raises ValueError; local budget exhaustion raises
    ResourceLimitError with no partial result. Missing optional dependency or
    unusable solver output returns an uncertified empty family. Other solver
    exceptions propagate; the external call has no timeout or budget sandbox.
    diagnostics['reason'] records refusals; source retains fresh exact analysis
    even when that analysis rejects a candidate. No backend callback is accepted.
    """
    if allow_external is not True:
        raise ValueError("external solver requires explicit allow_external=True")
    D = validate_distance_matrix(D, limits=limits)
    n = len(D)
    a = b = source = None
    diagnostics = dict(vertices=n, scope="all supplied rows; selected family only",
                       external_solver="Ripser / Ripser.py",
                       external_resources="unbounded beyond input vertex cap",
                       interval_fractions=(0.2, 0.8))

    def fail(reason):
        return WitnessResult(False, (), a, b, dict(diagnostics, reason=reason), source)

    if n < 4:
        return fail("no_finite_positive_h1_bar")
    try:
        from ripser import ripser
    except ImportError:
        return fail("optional_ripser_unavailable")
    # No row restriction, subsampling, metric conversion or float32 cast here.
    # Keep caller-owned matrix immutable even if an external backend writes to
    # its input. All final checks still use the original supplied matrix.
    output = ripser(D.copy(), distance_matrix=True, maxdim=1, coeff=2, do_cocycles=True)
    if not isinstance(output, dict):
        return fail("malformed_external_output")
    diagrams, cocycles = output.get("dgms"), output.get("cocycles")
    if not isinstance(diagrams, (list, tuple)) or len(diagrams) < 2:
        return fail("malformed_h1_diagram")
    bars = diagrams[1]
    if (not isinstance(bars, np.ndarray) or bars.ndim != 2 or bars.shape[1] != 2
            or bars.dtype.kind not in "fiu"):
        return fail("malformed_h1_diagram")
    max_pairs = n * (n - 1) // 2
    if len(bars) > max_pairs:
        return fail("malformed_h1_diagram_size")
    candidates = []
    for index, row in enumerate(bars):
        birth, death = map(float, row)
        if math.isfinite(birth) and math.isfinite(death) and 0 <= birth < death:
            candidates.append((-(death - birth), birth, index, death))
    if not candidates:
        return fail("no_finite_positive_h1_bar")
    neg_length, birth, index, death = min(candidates)
    length = -neg_length
    a, b = birth + 0.2 * length, death - 0.2 * length
    diagnostics.update(bar_index=index, bar=(birth, death))
    if not birth < a < b < death:
        return fail("unrepresentable_fixed_interior")
    forward, edge_ids = _survival_graph(D, b, limits)
    triangle_count = 0
    for _ in _triangles(forward):
        triangle_count += 1
        if triangle_count > limits.max_triangles:
            raise ResourceLimitError("max_triangles exceeded in witness preflight")
    diagnostics.update(survival_edges=len(edge_ids), survival_triangles=triangle_count)
    if (not isinstance(cocycles, (list, tuple)) or len(cocycles) < 2
            or not isinstance(cocycles[1], (list, tuple))
            or len(cocycles[1]) != len(bars)):
        return fail("malformed_h1_cocycles")
    cochain = cocycles[1][index]
    if (not isinstance(cochain, np.ndarray) or cochain.ndim != 2
            or cochain.shape[1] != 3 or cochain.dtype.kind not in "iu"
            or len(cochain) > max_pairs):
        return fail("malformed_selected_cochain")
    support = set()
    for u, v, coefficient in cochain:
        u, v, coefficient = int(u), int(v), int(coefficient)
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return fail("malformed_selected_cochain")
        edge = min(u, v), max(u, v)
        # Restrict the external cochain to the exact survival complex.
        if coefficient % 2 and edge in edge_ids:
            support.symmetric_difference_update((edge,))
    for i, j, k in _triangles(forward):
        if ((i, j) in support) ^ ((i, k) in support) ^ ((j, k) in support):
            return fail("cochain_not_closed_on_exact_survival_complex")

    tree = set(_mst(D))
    adjacency = [[] for _ in range(n)]
    for u, v in tree:
        adjacency[u].append(v)
        adjacency[v].append(u)
    parent, depth, parity = [-1] * n, [0] * n, [0] * n
    parent[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if parent[v] != -1:
                continue
            parent[v], depth[v] = u, depth[u] + 1
            parity[v] = parity[u] ^ ((min(u, v), max(u, v)) in support)
            stack.append(v)
    # edge_ids insertion order is lexicographic. Prefix parity avoids building
    # O(E) paths: only the first pairing-one candidate is materialized (<=N edges).
    for edge in edge_ids:
        u, v = edge
        if edge in tree or D[u, v] > a:
            continue
        if not (parity[u] ^ parity[v] ^ (edge in support)):
            continue
        if limits.max_cycles < 1:
            raise ResourceLimitError("max_cycles exceeded")
        chain = [edge]
        while u != v:
            if len(chain) >= limits.max_chain_edges:
                raise ResourceLimitError("max_chain_edges exceeded")
            if depth[u] < depth[v]:
                u, v = v, u
            p = parent[u]
            chain.append((min(u, p), max(u, p)))
            u = p
        cycles = (tuple(sorted(chain)),)
        diagnostics['fundamental_edge'] = edge
        source = analyze_sparse_h1(D, cycles, a, b, limits=limits)
        if not source.certified:
            return fail("exact_source_rejected: " + source.reason)
        return WitnessResult(True, cycles, a, b,
                             dict(diagnostics, reason="certified_selected_family"), source)
    return fail("no_pairing_one_birth_fundamental_cycle")
