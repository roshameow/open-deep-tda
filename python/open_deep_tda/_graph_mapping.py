"""Compact TRAIN-convex-hull phase II for the conditional Cauchy query mapper.

A continuous objective attains a minimum on this compact domain, but the local
SLSQP search need not find its global minimum. Feasible visited incumbents,
including the barycenter, are retained; projection is a stationarity diagnostic,
never post-hoc clipping. No topology or semantic preservation is implied.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, LinearConstraint

MAXITER=100
MAXCALLS=2000

class CompactMap:
    def __init__(self,Y,max_vertices=60000):
        if isinstance(max_vertices,bool) or not isinstance(max_vertices,(int,np.integer)) or max_vertices<1:raise ValueError('invalid vertex budget')
        if np.ma.isMaskedArray(Y):raise ValueError('masked anchors unsupported')
        raw=np.asarray(Y)
        if raw.ndim!=2 or raw.shape[1]!=2 or len(raw)<1 or len(raw)>max_vertices or raw.dtype.kind not in 'iuf':raise ValueError('invalid anchors or vertex budget')
        self.Y=np.array(raw,dtype=float,copy=True)
        if not np.isfinite(self.Y).all():raise ValueError('nonfinite anchors')
        self.origin=self.Y[0].copy();diff=self.Y-self.origin
        if not np.isfinite(diff).all():raise ValueError('anchor subtraction overflow')
        with np.errstate(over='ignore', invalid='ignore'):
            norms=np.linalg.norm(self.Y,axis=1)
        if not np.isfinite(norms).all():
            raise ValueError('anchor norm overflow; rescale the layout')
        self.tolerance=64*np.finfo(float).eps*(1+float(np.max(norms)))
        _,s,v=np.linalg.svd(diff,full_matrices=False)
        if not np.isfinite(s).all() or not np.isfinite(v).all():
            raise ValueError('nonfinite affine-support factorization')
        threshold=(s[0]*max(diff.shape)*np.finfo(float).eps) if len(s) else 0.
        if not np.isfinite(threshold):
            raise ValueError('affine-rank threshold overflow; rescale the layout')
        self.dimension=int(np.count_nonzero(s>threshold));self.A=np.empty((0,2));self.b=np.empty(0)
        if self.dimension==2:
            hull=ConvexHull(self.Y) # deliberately NO QJ/joggle/sampling
            self.vertices=self.Y[hull.vertices].copy();self.A=hull.equations[:,:2].copy();self.b=-hull.equations[:,2].copy()
        elif self.dimension==1:
            self.direction=v[0].copy()
            if self.direction[np.argmax(abs(self.direction))]<0:self.direction*=-1
            t=diff@self.direction;self.lo=float(t.min());self.hi=float(t.max())
            # Numerical rank classification must not silently discard geometric thickness.
            if np.max(np.linalg.norm(diff-t[:,None]*self.direction,axis=1))>self.tolerance:
                raise ValueError('near-collinear anchors outside numerical affine-line tolerance')
            self.vertices=self.origin+np.array([self.lo,self.hi])[:,None]*self.direction
        else:self.vertices=self.origin[None].copy()

    def _point(self,y):
        if np.ma.isMaskedArray(y):raise ValueError('masked point')
        a=np.asarray(y)
        if a.shape!=(2,) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():raise ValueError('invalid finite2D point')
        return a.astype(float,copy=True)

    def violation(self,y):
        y=self._point(y)
        if self.dimension==2:return float(np.max(self.A@y-self.b))
        if self.dimension==1:
            d=y-self.origin;t=float(d@self.direction)
            return max(self.lo-t,t-self.hi,float(np.linalg.norm(d-t*self.direction)))
        return float(np.linalg.norm(y-self.origin))

    def contains(self,y):return bool(self.violation(y)<=self.tolerance)

    def project(self,y):
        y=self._point(y)
        if self.dimension==0:return self.origin.copy()
        if self.dimension==1:return self.origin+np.clip((y-self.origin)@self.direction,self.lo,self.hi)*self.direction
        if np.all(self.A@y<=self.b):return y.copy()
        starts=self.vertices;ends=np.roll(starts,-1,axis=0);edges=ends-starts
        t=np.clip(np.einsum('ij,ij->i',y-starts,edges)/np.einsum('ij,ij->i',edges,edges),0,1)
        candidate=starts+t[:,None]*edges
        return candidate[np.argmin(np.sum((candidate-y)**2,axis=1))].copy()

    def _positive(self,ids,p):
        ii=np.asarray(ids);pp=np.asarray(p)
        if (np.ma.isMaskedArray(ids) or np.ma.isMaskedArray(p) or ii.ndim!=1 or len(ii)<1 or ii.dtype.kind not in 'iu'
            or np.any(ii<0) or np.any(ii>=len(self.Y)) or len(np.unique(ii))!=len(ii)
            or pp.shape!=ii.shape or pp.dtype.kind not in 'iuf' or not np.isfinite(pp).all() or np.any(pp<0)
            or abs(float(pp.sum())-1)>64*np.finfo(float).eps*len(pp)):
            raise ValueError('invalid positive IDs/probabilities')
        return ii.astype(int,copy=True),pp.astype(float,copy=True)

    def cauchy(self,y,ids,p):
        r=y-self.Y;s=1+np.einsum('ij,ij->i',r,r);w=1/s;q=w/w.sum()
        value=float(np.dot(p,np.log(s[ids]))+np.log(w.sum())+np.dot(p[p>0],np.log(p[p>0])))
        gradient=2*((p*w[ids])@r[ids]-(q*w)@r)
        if not np.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('nonfinite exact KL/gradient')
        return value,gradient

    def solve(self,old,ids,p,bary,*,check=None):
        old=self._point(old);bary=self._point(bary);ids,p=self._positive(ids,p)
        if not self.contains(bary):raise ValueError('barycenter not feasible')
        expected=np.einsum('i,ij->j',p,self.Y[ids])
        if np.linalg.norm(expected-bary)>self.tolerance*max(1,len(ids)):raise ValueError('barycenter mismatches positive weights')
        evaluations=0
        def objective(y):
            nonlocal evaluations
            if check is not None:check()
            evaluations+=1
            return self.cauchy(y,ids,p)
        f_old,g_old=objective(old)
        record=dict(old_inside=self.contains(old),old_violation=self.violation(old),old_escape_distance=float(np.linalg.norm(old-self.project(old))),old_objective=f_old,dimension=self.dimension,tolerance=self.tolerance)
        if record['old_inside']:
            y=old.copy();f,g=f_old,g_old
            record.update(phase='retained_unbounded_inside',solver_success=None,solver_status=None,iterations=0,objective_calls=1,feasible_visits=1,selected='old_inside')
        else:
            f_bary,_=objective(bary);best=[f_bary,bary.copy(),'bary'];calls=0;feasible=1
            def consider(y,f,label):
                nonlocal feasible
                if np.isfinite(f) and self.contains(y):
                    feasible+=1
                    if f<best[0]:best[:]=[f,y.copy(),label]
            def fg(v):
                nonlocal calls
                calls+=1
                if calls>MAXCALLS:raise RuntimeError('objective-call budget exceeded')
                y=(self.origin+self.direction*float(v[0])) if self.dimension==1 else np.asarray(v)
                f,g=objective(y);consider(y,f,'visited')
                return f,np.array([g@self.direction]) if self.dimension==1 else g
            record['bary_objective']=f_bary
            try:
                if self.dimension==0:
                    result=None;f=objective(self.origin)[0];consider(self.origin,f,'singleton')
                    record.update(solver_success=True,solver_status=0,message='singleton compact domain',iterations=0)
                else:
                    initial=np.array([(bary-self.origin)@self.direction]) if self.dimension==1 else bary.copy()
                    options=dict(maxiter=MAXITER,ftol=1e-12)
                    kwargs=dict(bounds=[(self.lo,self.hi)]) if self.dimension==1 else dict(constraints=[LinearConstraint(self.A,-np.inf,self.b)])
                    result=minimize(fg,initial,jac=True,method='SLSQP',options=options,**kwargs)
                    final=(self.origin+self.direction*float(result.x[0])) if self.dimension==1 else np.asarray(result.x)
                    if np.isfinite(final).all():consider(final,objective(final)[0],'final')
                    record.update(solver_success=bool(result.success),solver_status=int(result.status),message=str(result.message),iterations=int(result.nit))
            except TimeoutError:
                # A caller's global prediction deadline is not a local
                # optimizer failure and must not return a partial prediction.
                raise
            except Exception as exc:
                record.update(solver_success=False,solver_status='exception',message=repr(exc),iterations=None)
            y=best[1];f,g=objective(y)
            record.update(phase='constrained_phase_II',objective_calls=calls+2,feasible_visits=feasible,selected=best[2],objective_minus_bary=f-f_bary)
        assert self.contains(y) and np.isfinite(y).all()
        pg=y-self.project(y-g)
        record.update(objective_calls=evaluations,final_objective=f,gradient=g.tolist(),gradient_linf=float(np.max(abs(g))),projected_gradient_l2=float(np.linalg.norm(pg)),final_violation=self.violation(y),objective_minus_unbounded=f-f_old,final_norm=float(np.linalg.norm(y)))
        return y,record
