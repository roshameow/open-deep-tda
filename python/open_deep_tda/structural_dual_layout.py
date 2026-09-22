"""Oracle-checked small planar repair using source-dual batch triangle cuts.

This is not a reducer, learned mapping or performance claim. The source dual
cochains are explicit sufficient witnesses; constraining them can be stronger
than the required classes. Independent H0/H1 verification alone accepts output.
"""
from dataclasses import asdict
import heapq
import numpy as np

from . import topology
from .structural_h0 import compare_h0, _preflight_size
from .structural_h1 import check_h1_witnesses, _chains, _integer_budget
from .structural_dual import dual_h1_certificate
from .structural_layout import _nonnegative, _envelope, _optimize


def _merge_upper(D, target, lower, epsilon):
    """Choose a reference-hierarchy-compatible tree, not original MST edges.

    Global source merges determine children/levels. Among their cross pairs,
    choose the currently shortest pair compatible with existing lower cuts.
    This tree realizes the SAME minimax hierarchy when its edges are assigned
    their merge levels. Coupled with pairwise merge lower bounds, satisfying
    these upper bounds suffices for H0 error <=epsilon. It is not necessary.
    """
    n=len(D)
    clusters={i:[i] for i in range(n)}
    owner=list(range(n))
    upper={}
    tree=sorted(topology.mst(D).tolist(),key=lambda e:(D[tuple(e)],tuple(sorted(e))))
    for i,j in tree:
        a,b=owner[i],owner[j]
        bound=float(D[i,j])+epsilon
        candidates=[tuple(sorted((u,v))) for u in clusters[a] for v in clusters[b]]
        candidates=[e for e in candidates if lower.get(e,0.)<=bound]
        if not candidates:
            raise RuntimeError('no source-compatible H0 merge witness; inconsistent cuts')
        edge=min(candidates,key=lambda e:(float(target[e]),e))
        upper[edge]=bound
        clusters[a]+=clusters.pop(b)
        for v in clusters[a]:owner[v]=a
    return upper


def _batch_cuts(dual,D,target,survival,margin,unit,lower):
    """Greedy weighted cover of ALL currently forbidden triangles.

    Only source-absent edges may be cut. Selecting a cover removes each current
    odd-parity triangle, not merely one particular filling. Newly created
    triangles need a new oracle round. Set-cover is heuristic, not optimal.
    """
    covers={}
    candidates=[]
    for index,triangle in enumerate(dual['forbidden_triangles']):
        edges=[tuple(edge) for edge in triangle['eligible_source_absent_edges']]
        if not edges: raise RuntimeError('forbidden triangle lacks a source-absent edge')
        candidates.append(edges)
        for edge in edges:
            if D[edge]<=survival: raise RuntimeError('invalid source-dual edge certificate')
            covers.setdefault(edge,set()).add(index)
    bounds={e:max(float(np.nextafter(survival,np.inf)),
                  survival+min(margin,(float(D[e])-survival)/2)) for e in covers}
    costs={e:max(((bounds[e]-float(target[e]))/unit)**2,1e-12) for e in covers}
    counts={e:len(ids) for e,ids in covers.items()}
    heap=[(-counts[e]/costs[e],e,counts[e]) for e in covers]
    heapq.heapify(heap)
    pending=set(range(len(candidates)))
    result=[]
    while pending:
        if not heap: raise RuntimeError('incomplete forbidden-triangle edge cover')
        _,edge,count=heapq.heappop(heap)
        if count==0 or count!=counts[edge]: continue
        hit=covers[edge]&pending
        lower[edge]=max(lower.get(edge,0.),bounds[edge])
        result.append(dict(edge=list(edge),minimum_distance=bounds[edge],triangles_covered=len(hit)))
        for index in sorted(hit):
            pending.remove(index)
            for other in candidates[index]:
                counts[other]-=1
                if counts[other]:heapq.heappush(heap,(-counts[other]/costs[other],other,counts[other]))
    return result


def solve_structural_layout(D_source, initial, cycles=(), birth_radius=None, survival_radius=None, *,
                  h0_tolerance=None, max_vertices=64, max_rounds=16, max_iterations=300,
                  restarts=2, seed=0, jitter=.01, separation_margin=.01,
                  max_simplices=200000, max_reduction_operations=20000000,
                  max_reduction_entries=2000000):
    """Seek a checked 2D repair on all supplied rows (hard cap64).

    D_source is converted to float64, initial is finite(N,2). Required cycles
    must be valid source witnesses on [birth_radius,survival_radius]. Empty
    cycles mean H0-only; at least one H0/H1 obligation is required.

    Instead of removing one filling per round, source dual cocycles identify
    ALL currently forbidden target triangles. A greedy source-consistent edge
    cover supplies separation cuts. H0 uses all-pair minimax lower bounds and
    dynamically selected edges realizing the reference merge hierarchy, rather
    than requiring the original source MST's geometric edges. These sufficient
    constraints can still be stronger than necessary and nonconvex/infeasible.

    SLSQP minimizes displacement with explicit constraints. Seeded jitter is
    used only in the declared restart budget. Independent structural_h0/h1
    checks, never the optimizer flag or a small loss, determine acceptance.
    Unresolved search is NOT proof of infeasibility. This is not scalable DR,
    semantic validation, full-H1 isomorphism, or out-of-sample inference.

    Budgets apply PER oracle call. At most 1+restarts+(restarts+1)*max_rounds
    candidate checks and 1+(restarts+1)*max_rounds dual calls are made. Reported
    word-work sums actual charged H1/dual operations; it excludes SLSQP, tree
    traversal and bounded-domain edge-cover bookkeeping, not a total RSS/time
    cap. Initial/source validation and resource errors propagate, not a partial
    success. Output is in original units, with no post-hoc unchecked scaling.

    status='certified' returns embedding. status='unresolved' returns embedding
    None and only last_candidate for diagnostics. The certificate covers only
    requested obligations on supplied IDs, not omitted population/classes.
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
        source_validation = dual_h1_certificate(D, D, chains, birth_radius, survival_radius, **options)
    elif birth_radius is not None or survival_radius is not None:
        raise ValueError('radii must be omitted when no H1 witnesses are supplied')

    calls = 0
    operations = 0 if source_validation is None else source_validation['reduction_operations']
    history = []

    def verify(Z):
        nonlocal calls, operations
        DT = topology.distance_matrix(Z)
        h0 = compare_h0(D, DT, tolerance=h0_tolerance, max_vertices=max_vertices) if h0_tolerance is not None else None
        h1 = None
        calls += 1
        if chains:
            checked = check_h1_witnesses(D, DT, chains, birth_radius, survival_radius, **options)
            operations += checked.reduction_operations
            h1 = dict(independent_check=asdict(checked), accepted=checked.accepted,
                      surviving_rank=checked.surviving_rank,
                      all_classes_independent=checked.all_classes_independent)
        accepted = ((h0 is None or h0['certified_within_tolerance']) and
                    (h1 is None or h1['accepted']))
        return accepted, dict(h0=h0, h1=h1), DT

    def finish(status, Z, certificate):
        return dict(status=status, embedding=Z.copy() if status=='certified' else None,
            last_candidate=Z.copy(), certificate=certificate, history=history,
            verification_calls=calls, charged_oracle_operations=operations,
            scope='Complete supplied domain only; specified witness representatives; no optimizer convergence or out-of-sample claim',
            search='Batch source-dual triangle cuts plus adaptive reference-merge tree envelope; not an infeasibility certificate')

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
    base_lower, _ = _envelope(D, h0_tolerance) if h0_tolerance is not None else ({},{})
    base_upper = {}
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
        if restart:
            last = working*unit+center
            accepted, certificate, DT = verify(last)
            if accepted: return finish('certified', last, certificate)
        for round_id in range(max_rounds):
            cuts=[]
            forbidden_count=None
            if chains and not certificate['h1']['accepted']:
                dual=dual_h1_certificate(D, DT, chains, birth_radius, survival_radius, **options)
                operations += dual['reduction_operations']
                if dual['certificate_satisfied'] and not certificate['h1']['accepted']:
                    raise RuntimeError('dual certificate disagrees with independent H1 checker')
                forbidden_count=len(dual['forbidden_triangles'])
                cuts=_batch_cuts(dual,D,DT,survival_radius,separation_margin,unit,lower)
            upper=(_merge_upper(D,DT,lower,.9*h0_tolerance) if h0_tolerance is not None else {})
            for edge,bound in base_upper.items():
                upper[edge]=min(upper.get(edge,float('inf')),bound)
            working, record=_optimize(anchor,working,lower,upper,unit,max_iterations)
            candidate=working*unit+center
            if not np.isfinite(candidate).all():
                history.append(dict(restart=restart,round=round_id,status='nonfinite_candidate'))
                break
            last=candidate
            accepted, certificate, DT=verify(candidate)
            history.append(dict(restart=restart,round=round_id,cuts=cuts,
                                forbidden_triangles=forbidden_count,accepted=accepted,**record))
            if accepted: return finish('certified',candidate,certificate)
    return finish('unresolved',last,certificate)
