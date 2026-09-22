"""Frozen benchmark arithmetic: exact ranks and fuzzy source15 weights.
Original implementation of standard neighborhood/Cauchy ideas; not author UMAP.
No Torch/Numba import here. Source inputs keep their original dtype.
"""

import numpy as np

from scipy.spatial.distance import cdist

from scipy.optimize import minimize

def knn(Q,X,k,self_ids=None):
    ids=np.empty((len(Q),k),dtype=np.int64); dist=np.empty((len(Q),k))
    for start in range(0,len(Q),64):
        d=cdist(Q[start:start+64],X)
        if self_ids is not None: d[np.arange(len(d)),self_ids[start:start+len(d)]]=np.inf
        # Full stable sort is inexpensive at5k, deterministic ID boundary ties.
        order=np.argsort(d,axis=1,kind='stable')[:,:k]
        ids[start:start+len(d)]=order
        dist[start:start+len(d)]=np.take_along_axis(d,order,axis=1)
    return ids,dist

def weights(d):
    target=np.log2(d.shape[1]); rho=np.min(np.where(d>0,d,np.inf),axis=1);rho[~np.isfinite(rho)]=0
    gap=np.maximum(d-rho[:,None],0); lo=np.zeros(len(d)); hi=np.maximum(d.max(axis=1),1e-12)
    saturated=(gap==0).sum(axis=1)>=target
    for _ in range(64):
        mid=(lo+hi)/2; mass=np.exp(-gap/mid[:,None]).sum(axis=1)
        lo=np.where(mass<target,mid,lo);hi=np.where(mass>=target,mid,hi)
    w=np.exp(-gap/hi[:,None]);w[saturated]=(gap[saturated]==0)
    return w

def metrics(X,Q,Z,Y,self_ids=None):
    values={str(k):[] for k in (5,15,50)};M=len(X)-(self_ids is not None)
    for start in range(0,len(Q),64):
        ds=cdist(Q[start:start+64],X);dt=cdist(Y[start:start+64],Z)
        if self_ids is not None:
            ii=np.arange(len(ds));jj=self_ids[start:start+len(ds)];ds[ii,jj]=np.inf;dt[ii,jj]=np.inf
        a=np.argsort(ds,axis=1,kind='stable');b=np.argsort(dt,axis=1,kind='stable')
        ra=np.empty_like(a);rb=np.empty_like(b);ii=np.arange(len(a))[:,None]
        ra[ii,a]=np.arange(1,len(X)+1);rb[ii,b]=np.arange(1,len(X)+1)
        for k in (5,15,50):
            overlap=np.sum(ra[ii,b[:,:k]]<=k,axis=1)/k
            trust=1-2*np.maximum(ra[ii,b[:,:k]]-k,0).sum(axis=1)/(k*(2*M-3*k+1))
            cont=1-2*np.maximum(rb[ii,a[:,:k]]-k,0).sum(axis=1)/(k*(2*M-3*k+1))
            values[str(k)].extend(np.column_stack((overlap,trust,cont)).tolist())
    return {k:dict(zip(('overlap','trustworthiness','continuity'),np.mean(v,axis=0).tolist())) for k,v in values.items()},values

def rerank(X,Q,candidates,k=15,self_ids=None):
    ids=np.empty((len(Q),k),dtype=np.int64);dist=np.empty((len(Q),k))
    for i,row in enumerate(candidates):
        row=np.unique(row)
        if self_ids is not None:row=row[row!=self_ids[i]]
        assert len(row)>=k and np.all((row>=0)&(row<len(X)))
        d=cdist(Q[i:i+1],X[row])[0];order=np.lexsort((row,d))[:k]
        ids[i]=row[order];dist[i]=d[order]
    return ids,dist

def audit(X,Q,ids,self_ids=None):
    exact,d=knn(Q,X,ids.shape[1],self_ids)
    strict=[];tie=[]
    for i,row in enumerate(ids):
        strict.append(len(set(row)&set(exact[i]))/len(row))
        distances=cdist(Q[i:i+1],X[row])[0];radius=d[i,-1]
        required=int(np.sum(d[i]<radius));near=int(np.sum(distances<radius));boundary=int(np.sum(distances==radius))
        tie.append((near+min(len(row)-required,boundary))/len(row))
    return dict(queries=len(Q),population=len(X),k=ids.shape[1],strict_recall=float(np.mean(strict)),tie_aware_recall=float(np.mean(tie)),exact_query_ids=exact.tolist(),per_query_strict=strict)

OPTIONS=dict(maxiter=100,maxls=30,ftol=1e-12,gtol=1e-8)

class FrozenMap:
    """Exact ALL-anchor conditional Cauchy KL; unchanged solver arithmetic."""
    def __init__(self,coordinates):
        self.Y=np.asarray(coordinates,dtype=float).copy()

    def cauchy(self,y,ids,p):
        r=y-self.Y;s=1+np.einsum('ij,ij->i',r,r);w=1/s;q=w/w.sum()
        # q normalization is over every frozen point, including nonpositives.
        value=float(np.dot(p,np.log(s[ids]))+np.log(w.sum())+np.dot(p[p>0],np.log(p[p>0])))
        gradient=2*((p*w[ids])@r[ids]-(q*w)@r)
        return value,gradient

    def solve(self,which,ids,p,bary):
        if which!='A': raise ValueError('This fixed benchmark has only mapper A')
        calls=0
        def fg(y):
            nonlocal calls
            calls+=1
            return self.cauchy(y,ids,p)
        initial,_=fg(bary)
        r=minimize(fg,bary.copy(),jac=True,method='L-BFGS-B',options=OPTIONS)
        f,g=fg(r.x)
        rec=dict(success=bool(r.success),status=int(r.status),message=str(r.message),iterations=int(r.nit),calls=calls,initial_objective=initial,final_objective=f,objective_delta=f-initial,gradient_linf=float(np.max(abs(g))),finite=bool(np.isfinite(r.x).all()))
        return r.x,rec

