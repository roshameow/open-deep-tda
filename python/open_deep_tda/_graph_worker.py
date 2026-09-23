#!/usr/bin/env python3
"""Standalone Torch-free source15 fuzzy-graph fitting worker.

Execute this file in a child process, never through the package initializer.
Component charts use normalized graph spectra and source-centroid PCA centers;
coincident centers have an explicit ID-ordered-circle rule. Numerical operation
order and RNG draw order are part of the reproducibility contract. No topology
preservation or global optimization guarantee is provided.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import sys
if 'torch' in sys.modules:
    raise RuntimeError('Torch preloaded in graph worker')
import json
import time
import traceback
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import importlib.util

# This module is also loaded by file in Torch-free benchmark processes. Load
# its pure-NumPy sibling explicitly rather than importing a package initializer.
_affinity_spec = importlib.util.spec_from_file_location(
    '_open_deep_tda_graph_affinity', Path(__file__).with_name('_graph_affinity.py'))
_affinity_module = importlib.util.module_from_spec(_affinity_spec)
_affinity_spec.loader.exec_module(_affinity_module)

def knn(Q,X,k,self_ids=None):
    ids=np.empty((len(Q),k),dtype=np.int64); dist=np.empty((len(Q),k))
    for start in range(0,len(Q),64):
        d=cdist(Q[start:start+64],X)
        if self_ids is not None: d[np.arange(len(d)),self_ids[start:start+len(d)]]=np.inf
        # Full stable sorting resolves distance-boundary ties by fitting row ID.
        order=np.argsort(d,axis=1,kind='stable')[:,:k]
        ids[start:start+len(d)]=order
        dist[start:start+len(d)]=np.take_along_axis(d,order,axis=1)
    return ids,dist

def weights(d):
    return _affinity_module.weights(d)

def graph(X):
    ids,d=knn(X,X,15,np.arange(len(X)));w=weights(d)
    A=sparse.csr_matrix((w.ravel(),(np.repeat(np.arange(len(X)),15),ids.ravel())),shape=(len(X),len(X)))
    G=(A+A.T-A.multiply(A.T)).tocsr();G.eliminate_zeros()
    return G,ids,d

def sgd_impl(Z,head,tail,w,epochs,rate,seed,max_sgd_events):
    # Sequential per-edge coordinate SGD. Independent derivation from Cauchy
    # attraction log(1+r2), repulsion log(1+1/r2); .001 stabilizes repulsion.
    np.random.seed(seed)
    counts=np.zeros(len(w),dtype=np.int64);neg_count=0; events=0
    for epoch in range(epochs):
        alpha=1.-epoch/epochs
        for e in np.random.permutation(len(w)):
            events+=1
            if events>max_sgd_events: raise RuntimeError("max_sgd_events exceeded")
            if np.random.random()>=w[e]: continue
            events+=1
            if events>max_sgd_events: raise RuntimeError("max_sgd_events exceeded")
            counts[e]+=1;i=head[e];j=tail[e]
            dx=Z[i,0]-Z[j,0];dy=Z[i,1]-Z[j,1];r2=dx*dx+dy*dy
            for c in range(2):
                delta=dx if c==0 else dy
                step=alpha*max(-4.,min(4.,-2.*delta/(1.+r2)))
                Z[i,c]+=step;Z[j,c]-=step
            for _ in range(rate):
                events+=1
                if events>max_sgd_events: raise RuntimeError("max_sgd_events exceeded")
                j=np.random.randint(len(Z))
                if i==j:continue
                dx=Z[i,0]-Z[j,0];dy=Z[i,1]-Z[j,1];r2=dx*dx+dy*dy
                coeff=2./((.001+r2)*(1.+r2))
                Z[i,0]+=alpha*max(-4.,min(4.,coeff*dx))
                Z[i,1]+=alpha*max(-4.,min(4.,coeff*dy));neg_count+=1
    return Z,counts,neg_count,events


def initialize(X,G,seed=0):
 count,labels=connected_components(G);centroids=np.array([X[labels==c].mean(0) for c in range(count)])
 if count==1:centers=np.zeros((1,2));scale=10.;rule='connected original'
 else:
  C=centroids-centroids.mean(0);_,_,v=np.linalg.svd(C,full_matrices=False);centers=C@v[:2].T
  if centers.shape[1]<2:centers=np.pad(centers,((0,0),(0,2-centers.shape[1])))
  centers*=10/max(np.max(abs(centers)),1e-12);dist=cdist(centers,centers);np.fill_diagonal(dist,np.inf);rule='sourcecentroidPCA'
  if dist.min()<1e-12:
   angle=np.arange(count)*2*np.pi/count;centers=10*np.c_[np.cos(angle),np.sin(angle)];dist=cdist(centers,centers);np.fill_diagonal(dist,np.inf);rule='explicit coincidentcentroid IDcircle'
  scale=dist.min()/4
 Z=np.zeros((len(X),2))
 for c in range(count):
  ids=np.flatnonzero(labels==c);sub=G[ids][:,ids];degree=np.asarray(sub.sum(1)).ravel();rng=np.random.default_rng(seed+c)
  if len(ids)==1:local=np.zeros((1,2))
  elif len(ids)==2:local=np.array([[-1.,0],[1,0]])
  else:
   inv=sparse.diags(1/np.sqrt(degree));L=sparse.eye(len(ids))-inv@sub@inv
   if len(ids)==3:val,vec=np.linalg.eigh(L.toarray())
   else:val,vec=eigsh(L,k=3,which='SM',v0=rng.normal(size=len(ids)),tol=1e-6)
   local=vec[:,np.argsort(val)[1:3]]
   for axis in range(2):
    if local[np.argmax(abs(local[:,axis])),axis]<0:local[:,axis]*=-1
   local/=np.max(abs(local))
  Z[ids]=centers[c]+scale*local
 Z+=np.random.default_rng(seed).normal(0,1e-4,Z.shape)
 return np.ascontiguousarray(Z),dict(components=int(count),center_rule=rule,component_sizes=np.bincount(labels).tolist())

def main(directory):
    out = Path(directory)
    cfg = json.loads((out/'config.json').read_text())
    start = time.perf_counter()
    X = np.load(out/'features.npy', allow_pickle=False)
    if (X.ndim != 2 or X.dtype.kind not in 'iuf' or X.shape[1] < 1 or
            len(X) < 16 or len(X) > cfg['max_vertices'] or
            X.size > cfg['max_feature_entries'] or not np.isfinite(X).all()):
        raise ValueError('invalid features / vertex or feature-entry budget')
    n = len(X)
    if 30*n > cfg['max_edges']:
        raise RuntimeError('max_edges conservative preflight exceeded')
    with np.load(out/'source.npz', allow_pickle=False) as a:
        ids, d = a['ids'], a['distances']
    if (ids.shape != (n,15) or ids.dtype.kind not in 'iu' or d.shape != (n,15)
            or d.dtype.kind not in 'iuf' or not np.isfinite(d).all()
            or np.any(d < 0) or np.any(ids < 0) or np.any(ids >= n)):
        raise ValueError('invalid source15 IDs or distances')
    for i,row in enumerate(ids):
        if i in row or len(np.unique(row)) != 15:
            raise ValueError('source15 contains self or duplicate IDs')
    w = weights(d)
    A = sparse.csr_matrix((w.ravel(),(np.repeat(np.arange(n),15),ids.ravel())),shape=(n,n))
    G = (A+A.T-A.multiply(A.T)).tocsr(); G.eliminate_zeros()
    coo = G.tocoo(); E = G.nnz
    bound = cfg['epochs']*E*(2+cfg['negative_rate'])
    if E > cfg['max_edges'] or bound > cfg['max_sgd_events']:
        raise RuntimeError('max_edges or max_sgd_events preflight exceeded')
    if not np.isfinite(G.data).all(): raise ValueError('nonfinite graph')
    Z, initialization = initialize(X,G,cfg['seed'])
    if not np.isfinite(Z).all(): raise RuntimeError('nonfinite component initialization')
    np.save(out/'spectral_initial.npy',Z,allow_pickle=False)
    from numba import njit
    if 'torch' in sys.modules:
        raise RuntimeError('Torch imported in graph worker')
    Z,counts,negative,events = njit(sgd_impl)(Z,coo.row,coo.col,coo.data,
        cfg['epochs'],cfg['negative_rate'],cfg['seed'],cfg['max_sgd_events'])
    if not np.isfinite(Z).all(): raise RuntimeError('nonfinite SGD output')
    np.save(out/'embedding.npy',Z,allow_pickle=False)
    np.savez_compressed(out/'graph.npz',neighbors=ids,distances=d,head=coo.row,
        tail=coo.col,weights=coo.data,counts=counts)
    result = dict(status='completed',directed_edges=int(E),
        positive_updates=int(counts.sum()),negative_updates=int(negative),sgd_events=int(events),
        sgd_event_upper_bound=int(bound),expected_positive_updates=float(cfg['epochs']*coo.data.sum()),
        unvisited_edges=int((counts==0).sum()),torch_absent='torch' not in sys.modules,
        threads=1,worker_seconds=time.perf_counter()-start,**initialization)
    (out/'worker_status.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except BaseException as exc:
        if len(sys.argv)>1:
            (Path(sys.argv[1])/'worker_status.json').write_text(
                json.dumps(dict(status='failed',error=repr(exc)),indent=2)+'\n')
        traceback.print_exc()
        sys.exit(1)
