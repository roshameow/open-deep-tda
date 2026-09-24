"""Opt-in bounded source-only single-cycle beam. No target-guided branch ranking.

Only independently checked complete all-row states can be returned. Work limits
count logical operations, not process RSS or an interrupt for native BLAS calls.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.optimize import brentq
from .structural_constructive import (
    _simple_order, _Unsupported, full_hierarchy, pairwise_planar)
from .structural_sparse_h1 import analyze_sparse_h1, ResourceLimitError
from .structural_planar import (certify as planar_certify,
                                ResourceLimitError as PlanarResourceLimitError)
from .structural_h0 import compare_h0

WIDTH, FAN, CHUNK = 16, 4, 256

@dataclass
class Result:
    certified: bool
    embedding: object
    hole: object
    target: object
    h0_error: object
    reason: str
    events: list
    diagnostics: dict


def _seed_compatible(seed_hierarchy, source_hierarchy, seq, delta, slack):
    """A restricted minimax metric may LOSE shortcuts, never require equality."""
    return bool(np.all(seed_hierarchy >=
                       np.maximum(0., source_hierarchy[np.ix_(seq, seq)]-delta)-slack))


def seed(D, guide, cycles, a, b, tolerance=.05):
    """Mirror public structural_constructive seed; deliberately no alternate orbit."""
    n = len(D)
    seq = _simple_order(cycles, n)
    U = full_hierarchy(D)
    delta = .98*tolerance
    W = np.maximum(0., U-delta)
    chords = np.array([W[seq[i],seq[(i+1)%len(seq)]] for i in range(len(seq))])
    if np.any(chords <= 0):
        raise ValueError('unsupported zero contracted seed chord')
    origin = guide[0].copy()
    G = guide-origin
    scale = max(float(D.max()),float(np.max(np.abs(G))), b,float(chords.max()))
    slack = 128*np.finfo(float).eps*scale
    unit = float(chords.max())
    q = chords/unit
    angle = lambda t: float(np.sum(2*np.arcsin(np.clip(q/(2*t),-1.,1.)))-2*np.pi)
    if angle(.5)<0:
        raise ValueError('unsupported minor-arc circle chord family')
    root = .5 if angle(.5)==0 else brentq(angle,.5,float(len(seq)),xtol=4*np.finfo(float).eps,rtol=4*np.finfo(float).eps,maxiter=128)
    radius=unit*root
    if not math.isfinite(radius) or radius<=0:
        raise ValueError('unrepresentable seed circle radius')
    phi=np.r_[0.,np.cumsum(2*np.arcsin(np.clip(chords[:-1]/(2*radius),-1.,1.)))]
    z=radius*np.column_stack([np.cos(phi),np.sin(phi)])
    zm,gm=z.mean(axis=0),G[seq].mean(axis=0)
    covariance=(z-zm).T @ (G[seq]-gm)
    if not np.isfinite(covariance).all():
        raise ValueError('unrepresentable Procrustes covariance')
    left,singular,right=np.linalg.svd(covariance)
    if not all(np.isfinite(x).all() for x in (left,singular,right)):
        raise ValueError('nonfinite Procrustes factors')
    rotation=left @ right
    hole=-zm @ rotation+gm
    z=(z-zm) @ rotation+gm
    if not np.isfinite(z).all() or not np.isfinite(hole).all():
        raise ValueError('unrepresentable aligned seed')
    seed_hierarchy=full_hierarchy(pairwise_planar(z))
    # Restricting a minimax hierarchy to the cycle can remove source-only
    # shortcuts through other IDs. That can INCREASE seed pair levels; only a
    # decrease beyond the contracted source lower bound is incompatible.
    if not _seed_compatible(seed_hierarchy, U, seq, delta, slack):
        raise ValueError('seed hierarchy collapses source lower bound')
    hole_radius=float(np.nextafter(b/np.sqrt(3.),np.inf))+max(1e-6*b,32*slack)
    if np.any(np.hypot(*(z-hole).T)<=hole_radius):
        raise ValueError('seed cannot satisfy protected-hole guard')
    Z=np.full((n,2),np.nan,dtype=np.float64)
    Z[seq]=z
    present=list(seq)
    nearest=U[:,seq].min(axis=1)
    return dict(Z=Z, present=present, nearest=nearest, U=U, W=W, G=G,
                hole=hole, hole_radius=hole_radius, origin=origin, slack=slack,
                seq=seq, radius=radius, delta=delta)


def contacts(center,r,centers,radii,guide,slack,counts,deadline,limits):
    """Every public analytic angle, instead of only its closest admissible point."""
    if counts['parent_tests'] >= limits.max_beam_parent_tests: raise ResourceLimitError('beam parent tests exhausted')
    counts['parent_tests']+=1
    if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
    delta=centers-center
    d=np.hypot(delta[:,0],delta[:,1])
    if np.any(d+r<radii-slack): return []
    if r==0:
        angles=np.empty(1,dtype=np.float64)
    else:
        relevant=(d>0)&(d<=r+radii+slack)&(d+np.minimum(r,radii)>=np.maximum(r,radii)-slack)
        dd,rr,vv=d[relevant],radii[relevant],delta[relevant]
        scale=np.maximum(np.maximum(dd,rr),r)
        ds,rs,ps=dd/scale,rr/scale,r/scale
        cosine=(ps*ps+ds*ds-rs*rs)/(2*ps*ds)
        keep=np.abs(cosine)<=1+128*np.finfo(float).eps
        phi=np.arctan2(vv[keep,1],vv[keep,0])
        theta=np.arccos(np.clip(cosine[keep],-1.,1.))
        angles=np.r_[np.arctan2(guide[1]-center[1],guide[0]-center[0]),phi+theta,phi-theta]
    count=len(angles)
    if (counts['candidate_points']+count>limits.max_beam_candidate_points or
            counts['contact_pairs']+count*len(centers)>limits.max_beam_contact_pairs):
        raise ResourceLimitError('beam contact work exhausted')
    counts['candidate_points']+=count
    counts['contact_pairs']+=count*len(centers)
    candidates=(center.reshape(1,2).copy() if r==0 else
                center+r*np.column_stack([np.cos(angles),np.sin(angles)]))
    result=[]
    for first in range(0,len(candidates),CHUNK):
        if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
        z=candidates[first:first+CHUNK]
        diff=z[:,None,:]-centers[None,:,:]
        distances=np.hypot(diff[:,:,0],diff[:,:,1])
        valid=np.all(distances>=radii[None,:]-slack,axis=1)
        for index in np.flatnonzero(valid):
            point=z[index].copy()
            result.append((float(np.hypot(*(point-guide))),first+int(index),point))
    return result


def search(D,guide,cycles,a,b,tolerance,*,limits,h1_limits,deadline):
    """No target-guided branch selection; final checks are accept/reject only."""
    events=[]
    counts=dict(expanded_states=0,parent_tests=0,candidate_points=0,contact_pairs=0,complete_checked=0)
    start=time.monotonic()
    def finish(ok,Z,hole,target,error,reason):
        # Even refusal/certification after an expensive native check cannot
        # bypass the caller's finite beam clock.
        if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
        counts['elapsed_seconds']=time.monotonic()-start
        return Result(ok,Z,hole,target,error,reason,events,dict(counts))
    try:
        if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
        if guide.dtype!=np.float64 or guide.shape!=(len(D),2) or not np.isfinite(guide).all():
            raise ValueError('invalid guide')
        # The caller has already validated the source metric, cycle and independent source H1.
        with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
            s=seed(D,guide,cycles,a,b,tolerance)
            if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
            Z0=s['Z'];present=s['present'];nearest=s['nearest'];U=s['U'];W=s['W'];G=s['G']
            hole=s['hole'];hr=s['hole_radius'];origin=s['origin'];slack=s['slack'];delta=s['delta']
            # Insertion sequence depends only on U and the existing set, not placement.
            ordering=[]
            while len(present)<len(D):
                if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
                remaining=[v for v in range(len(D)) if v not in present]
                new=min(remaining,key=lambda v:(nearest[v],v))
                ordering.append((new,float(nearest[new]),tuple(present)))
                present.append(new)
                nearest=np.minimum(nearest,U[:,new])
            beam=[(0.,(),Z0)]
            for new,level,present in ordering:
                next_states=[]
                for score,path,Z in beam:
                    if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
                    if counts['expanded_states']>=limits.max_beam_expansions: raise ResourceLimitError('state expansion budget exhausted')
                    counts['expanded_states']+=1
                    state_id=counts['expanded_states']
                    r=max(0.,level-delta)
                    parents=[p for p in present if U[new,p]==level]
                    parents.sort(key=lambda p:(abs(float(np.hypot(*(G[new]-Z[p])))-r),p))
                    centers=np.vstack([Z[list(present)],hole])
                    radii=np.r_[W[new,list(present)],hr]
                    choices=[]
                    for parent in parents:
                        if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
                        for distance,index,point in contacts(Z[parent],r,centers,radii,G[new],slack,counts,deadline,limits):
                            choices.append((distance,parent,index,point))
                    choices.sort(key=lambda x:(x[0],x[1],x[2]))
                    unique=[];seen=set()
                    for distance,parent,index,point in choices:
                        key=point.tobytes()
                        if key in seen: continue
                        seen.add(key)
                        unique.append((distance,parent,index,point))
                        if len(unique)==FAN:break
                    events.append(dict(state=state_id,vertex=new,path=path,parents=parents,
                                       valid_candidates=len(choices),kept=len(unique),
                                       status='expanded' if unique else 'contact_failed'))
                    for distance,parent,index,point in unique:
                        child=Z.copy();child[new]=point
                        next_states.append((score+distance,path+((parent,index),),child))
                next_states.sort(key=lambda x:(x[0],x[1]))
                if not next_states:
                    return finish(False,None,None,None,None,'all contact branches refused; no infeasibility claim')
                beam=next_states[:WIDTH]
            # Also handle all-seed cycle, with zero expansions.
            for score,path,Z in beam:
                if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
                counts['complete_checked']+=1
                entry=dict(complete=counts['complete_checked'],path=path,score=score)
                Z=Z+origin;H=hole+origin
                if not np.isfinite(Z).all() or not np.isfinite(H).all():
                    entry['status']='nonfinite';events.append(entry);continue
                T=pairwise_planar(Z)
                h0=compare_h0(D,T,tolerance=tolerance,max_vertices=limits.max_vertices)
                err=float(h0['max_merge_error'])
                entry['h0_error']=err
                # Match the established constructive teacher's outward-rounded
                # boundary policy, rather than accepting a rounded equality.
                h0_upper=0. if err==0 else float(np.nextafter(err,np.inf))
                entry['h0_error_upper']=h0_upper
                if not h0['certified_within_tolerance'] or h0_upper>tolerance:
                    entry['status']='h0_failed';events.append(entry);continue
                target=analyze_sparse_h1(T,cycles,a,b,limits=h1_limits)
                entry['h1_rank']=target.rank
                if not target.certified:
                    entry['status']='h1_failed: '+target.reason;events.append(entry);continue
                planar=planar_certify(Z,cycles,a,b,H.reshape(1,2))
                entry['status']='certified' if planar.certified else 'planar_failed: '+planar.reason
                events.append(entry)
                if time.monotonic()>=deadline: raise TimeoutError('beam deadline exhausted')
                if planar.certified:
                    Z.setflags(write=False);H.setflags(write=False)
                    return finish(True,Z,H,target,err,'first certified complete beam state')
            return finish(False,None,None,None,None,'all complete beam states refused; no infeasibility claim')
    except (ResourceLimitError, PlanarResourceLimitError, TimeoutError):
        # Both resource exceptions subclass ValueError: NEVER convert them
        # into an ordinary unsupported result, even after other branches fail.
        raise
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError, ValueError, _Unsupported) as exc:
        events.append(dict(status='unsupported', reason=str(exc)))
        return finish(False,None,None,None,None,str(exc)+'; no infeasibility claim')
