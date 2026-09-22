"""Bounded, oracle-guided planar layout REPAIR, not a trained reducer.

Required source cycle edges are upper-distance constraints. A certified target
filling/relation produces a new source-consistent edge-separation constraint.
Optional H0 uses a conservative global-tree/merge envelope. Only the independent
structural checkers can accept a result; SLSQP status or loss cannot do so.
No convergence, infeasibility, semantic, whole-H1, or out-of-sample guarantee.
"""
from dataclasses import asdict
from itertools import combinations
import numbers

import numpy as np
from scipy.optimize import minimize

from . import topology
from .structural_h0 import compare_h0, _path_maxima, _preflight_size
from .structural_h1 import check_h1_witnesses, _chains, _integer_budget
from .structural_obstructions import find_h1_obstructions


def _nonnegative(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative real scalar')
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and representable') from exc
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'invalid {name}')
    return value


def _envelope(D, tolerance):
    """A sufficient (not necessary) same-ID H0 envelope, no subsampling."""
    n = len(D)
    edges = topology.mst(D)
    adjacency = [[] for _ in range(n)]
    for i, j in edges:
        adjacency[i].append((int(j), float(D[i, j])))
        adjacency[j].append((int(i), float(D[i, j])))
    # Use a small interior tolerance so floating constraint residuals do not
    # silently relax the separately checked user tolerance.
    epsilon = .9 * tolerance
    lower = {}
    for i in range(n):
        merge = _path_maxima(adjacency, i)
        for j in range(i + 1, n):
            lower[(i, j)] = max(0., merge[j] - epsilon)
    upper = {tuple(sorted((int(i), int(j)))): float(D[i, j]) + epsilon for i, j in edges}
    return lower, upper


def _optimize(anchor, start, lower, upper, unit, iterations):
    """Minimize displacement, with analytical quadratic pair constraints."""
    n = len(start)
    items = [(edge, bound / unit, 1.) for edge, bound in sorted(lower.items()) if bound > 0]
    items += [(edge, bound / unit, -1.) for edge, bound in sorted(upper.items())]
    pairs = np.asarray([item[0] for item in items], dtype=np.int64).reshape(-1, 2)
    bounds = np.asarray([item[1] for item in items], dtype=float)
    signs = np.asarray([item[2] for item in items], dtype=float)
    if not np.isfinite(bounds).all() or not np.isfinite(bounds**2).all():
        raise ValueError('constraint scale overflow; rescale inputs')

    def fun(flat):
        error = flat.reshape(n, 2) - anchor
        return float(np.mean(error**2))

    def jac(flat):
        return (2 * (flat.reshape(n, 2) - anchor) / (2*n)).ravel()

    def constraints(flat):
        Z = flat.reshape(n, 2)
        delta = Z[pairs[:, 0]] - Z[pairs[:, 1]]
        return signs * (np.einsum('ij,ij->i', delta, delta) - bounds**2)

    def constraint_jac(flat):
        Z = flat.reshape(n, 2)
        delta = 2 * signs[:, None] * (Z[pairs[:, 0]] - Z[pairs[:, 1]])
        result = np.zeros((len(pairs), n, 2))
        rows = np.arange(len(pairs))
        result[rows, pairs[:, 0]] = delta
        result[rows, pairs[:, 1]] = -delta
        return result.reshape(len(pairs), 2*n)

    cons = [{'type': 'ineq', 'fun': constraints, 'jac': constraint_jac}] if len(pairs) else []
    result = minimize(fun, start.ravel(), jac=jac, method='SLSQP', constraints=cons,
                      options=dict(maxiter=iterations, ftol=1e-10, disp=False))
    return result.x.reshape(n, 2), dict(optimizer_success=bool(result.success),
        optimizer_status=int(result.status), iterations=int(result.nit),
        displacement_objective=float(result.fun), constraint_count=len(pairs))


def repair_layout(D_source, initial, cycles=(), birth_radius=None, survival_radius=None, *,
                  h0_tolerance=None, max_vertices=64, max_rounds=16, max_iterations=300,
                  restarts=2, seed=0, jitter=.01, separation_margin=.01,
                  max_simplices=200000, max_reduction_operations=20000000,
                  max_reduction_entries=2000000):
    """Seek a checked 2D repair on the complete supplied domain (hard cap64).

    ``D_source`` is converted to float64; initial is finite (N,2). Cycles are
    explicit source F2 witnesses; an empty family means no H1 obligation. At
    least H1 or h0_tolerance must be requested. Radii share the source units.
    Invalid source contracts/inputs and resource exhaustion raise, never return
    success. No labels, feature training, sampling or normalization of outputs.

    Each target filling certificate yields one lower-distance cut on an edge
    absent in the SOURCE at survival. The chosen branch is heuristic (longest
    currently present candidate, deterministic tie break). Alternative fillings
    are discovered in subsequent rounds. All required birth edges have upper
    bounds. For optional H0, global source MST edges have upper bounds and all
    pairs have lower bounds from the source merge hierarchy. This envelope is
    sufficient, not necessary: failure is NOT proof of structural infeasibility.

    Only check_h1_witnesses / compare_h0 decide acceptance, in original units.
    SLSQP and small interior margins are numerical search tools, not certificates.
    Restarts use explicit seeded jitter around the initial layout to escape
    zero-distance stationary points; no silent retries or best-seed search.
    Budget limits are PER oracle call; candidate-check counts and cumulative
    charged H1 word-work are reported. At most 1+(restarts+1)*max_rounds candidate
    checks each invoke at most one obstruction finder, one independent H1 check
    and one H0 check, plus one initial source validation. This is not scalable DR.

    Returns dict: status='certified' with embedding, or status='unresolved' with
    embedding=None and last_candidate for diagnostics. Every returned embedding
    marked certified satisfies ALL requested checks; omitted classes/populations
    and a whole-complex chain map are not certified. No transform is provided.
    """
    max_vertices = min(64, _integer_budget('max_vertices', max_vertices))
    n = _preflight_size(D_source, 'D_source', max_vertices)
    D = topology._distance_array(D_source)
    raw = np.asarray(initial)
    if raw.dtype.kind not in 'iuf' or raw.shape != (n, 2):
        raise ValueError('initial must be a real (N,2) matrix')
    Z0 = np.asarray(raw, dtype=np.float64).copy()
    if not np.isfinite(Z0).all():
        raise ValueError('initial must be finite')
    for name, value in [('max_rounds',max_rounds),('max_iterations',max_iterations),
                        ('restarts',restarts),('seed',seed)]:
        checked = _integer_budget(name, value)
        if name=='max_rounds': max_rounds=checked
        elif name=='max_iterations': max_iterations=checked
        elif name=='restarts': restarts=checked
        else: seed=checked
    jitter = _nonnegative(jitter, 'jitter')
    separation_margin = _nonnegative(separation_margin, 'separation_margin', positive=True)
    if h0_tolerance is not None:
        h0_tolerance = _nonnegative(h0_tolerance, 'h0_tolerance')
    cycles = list(cycles)
    chains = _chains(cycles, n) if cycles else []
    if not chains and h0_tolerance is None:
        raise ValueError('request at least one H0 or H1 obligation')
    options = dict(max_vertices=max_vertices, max_simplices=max_simplices,
                   max_reduction_operations=max_reduction_operations,
                   max_reduction_entries=max_reduction_entries)
    # Validate budgets even when only H0 is requested.
    for key, value in options.items():
        options[key] = _integer_budget(key, value)
    source_validation = None
    if chains:
        birth_radius = _nonnegative(birth_radius, 'birth_radius')
        survival_radius = _nonnegative(survival_radius, 'survival_radius')
        source_validation = find_h1_obstructions(D, D, chains, birth_radius, survival_radius, **options)
    elif birth_radius is not None or survival_radius is not None:
        raise ValueError('radii must be omitted when no H1 witnesses are supplied')

    calls = 0
    operations = 0 if source_validation is None else source_validation['reduction_operations']
    history = []

    def verify(Z):
        nonlocal calls, operations
        DT = topology.distance_matrix(Z)
        h0 = compare_h0(D, DT, tolerance=h0_tolerance, max_vertices=max_vertices) if h0_tolerance is not None else None
        h1 = find_h1_obstructions(D, DT, chains, birth_radius, survival_radius, **options) if chains else None
        calls += 1
        if h1 is not None:
            operations += h1['reduction_operations']
            checked = check_h1_witnesses(D, DT, chains, birth_radius, survival_radius, **options)
            operations += checked.reduction_operations
            if checked.accepted != h1['all_classes_independent'] or checked.surviving_rank != h1['surviving_rank']:
                raise RuntimeError('obstruction finder disagrees with independent H1 checker')
            h1['independent_check'] = asdict(checked)
            h1['accepted'] = checked.accepted
        accepted = ((h0 is None or h0['certified_within_tolerance']) and
                    (h1 is None or h1['accepted']))
        return accepted, dict(h0=h0, h1=h1), DT

    def finish(status, Z, certificate):
        return dict(status=status, embedding=Z.copy() if status=='certified' else None,
            last_candidate=Z.copy(), certificate=certificate, history=history,
            verification_calls=calls, charged_oracle_operations=operations,
            scope='Complete supplied domain only; specified witness representatives; no optimizer convergence or out-of-sample claim',
            search='Source-consistent filling cuts plus optional conservative H0 envelope; not an infeasibility certificate')

    accepted, certificate, DT = verify(Z0)
    if accepted: return finish('certified', Z0, certificate)
    if max_rounds == 0 or max_iterations == 0: return finish('unresolved', Z0, certificate)
    positive = D[D > 0]
    unit = max(float(np.median(positive)) if positive.size else 1.,
               survival_radius if chains else 0., np.finfo(float).tiny)
    center = Z0.mean(axis=0)
    anchor = (Z0-center)/unit
    if not np.isfinite(anchor).all():
        raise ValueError('layout scaling overflow; rescale inputs')
    base_lower, base_upper = _envelope(D, h0_tolerance) if h0_tolerance is not None else ({},{})
    if chains:
        safety = min(birth_radius*1e-5, separation_margin*.01)
        if h0_tolerance is not None: safety=min(safety,h0_tolerance*.1)
        for chain in chains:
            for edge in chain:
                base_upper[edge]=min(base_upper.get(edge,float('inf')),max(0.,birth_radius-safety))
    rng=np.random.default_rng(seed)
    last=Z0.copy()
    for restart in range(restarts+1):
        lower, upper = dict(base_lower), dict(base_upper)
        working=anchor.copy()
        if restart:
            noise=rng.normal(size=working.shape)*jitter
            working+=noise-noise.mean(axis=0)
        # Obstructions from the previous failed candidate remain valid cuts for
        # this restart too; each cut is explicitly checked against the source.
        for round_id in range(max_rounds):
            cuts=[]
            if chains:
                for relation in certificate['h1']['relations']:
                    edges={tuple(sorted(edge)) for triangle in relation['filling_triangles']
                           for edge in combinations(triangle,2)}
                    choices=[e for e in edges if D[e] > survival_radius]
                    if not choices:
                        raise RuntimeError('filling certificate contradicts validated source contract')
                    edge=min(choices,key=lambda e:(-float(DT[e]),e))
                    gap=float(D[edge])-survival_radius
                    bound=max(float(np.nextafter(survival_radius,np.inf)),
                              survival_radius+min(separation_margin,gap/2))
                    lower[edge]=max(lower.get(edge,0.),bound)
                    cuts.append(dict(edge=list(edge),minimum_distance=bound,
                                     relation_cycles=relation['cycle_indices']))
            working, record=_optimize(anchor,working,lower,upper,unit,max_iterations)
            candidate=working*unit+center
            if not np.isfinite(candidate).all():
                history.append(dict(restart=restart,round=round_id,status='nonfinite_candidate'))
                break
            last=candidate
            accepted, certificate, DT=verify(candidate)
            history.append(dict(restart=restart,round=round_id,cuts=cuts,
                                accepted=accepted,**record))
            if accepted: return finish('certified',candidate,certificate)
    return finish('unresolved',last,certificate)
