"""Known-viewpoint diagnostics for real object rotations.

Angle/object annotations are evaluation-only. Appearance symmetry may make
physical angles unidentifiable from images, so reference-space scores MUST be
reported alongside latent scores. A strong H1 bar alone is not identified-cycle
preservation, and absence of crossings alone is not proof of a valid embedding.
"""
import numpy as np
from .evaluation import _matrix, _paired
from .topology import distance_matrix, diagram_matching


def angle_neighbor_metrics(X, angles_degrees, query_mask=None, k=2):
    X=_matrix(X,'X')
    angles=np.asarray(angles_degrees,dtype=float)
    n=len(X)
    if angles.shape!=(n,) or not np.isfinite(angles).all() or n<3:
        raise ValueError('need >=3 rows and one finite angle per row')
    angles=np.mod(angles,360.)
    if len(np.unique(angles))!=n:
        raise ValueError('each physical view angle must be unique within an object')
    if not isinstance(k,(int,np.integer)) or isinstance(k,bool) or k<1:
        raise ValueError('k must be a positive integer')
    k=min(k,n-1)
    query=np.ones(n,dtype=bool) if query_mask is None else np.asarray(query_mask)
    if query.shape!=(n,) or query.dtype.kind!='b' or not query.any():
        raise ValueError('query_mask must select at least one row')
    order=np.argsort(angles,kind='stable')
    adjacent={int(order[i]):{int(order[(i-1)%n]),int(order[(i+1)%n])} for i in range(n)}
    D=distance_matrix(X);np.fill_diagonal(D,np.inf)
    nearest=np.argsort(D,axis=1,kind='stable')[:,:k]
    recall=[];separations=[]
    for i in np.flatnonzero(query):
        recall.append(len(set(nearest[i]) & adjacent[int(i)])/2.)
        diff=np.abs(angles[nearest[i]]-angles[i]);separations.extend(np.minimum(diff,360-diff).tolist())
    return {'adjacent_view_recall':float(np.mean(recall)),
            'mean_neighbor_angle_degrees':float(np.mean(separations)),
            'fraction_neighbor_angle_gt30':float(np.mean(np.asarray(separations)>30)),
            'queries':int(query.sum()),'n_views':n,'k':k,
            'collapsed':bool(np.max(np.linalg.norm(X-X[0],axis=1))==0),
            'scope':'neighbors conditioned on object identity; true adjacent views defined by cyclic angle order'}


def strict_polygon_crossings(points):
    """Count proper crossings in an angle-ordered 2D closed polygon.

    Adjacent edges, collinear overlap and endpoint touch are excluded. Returned
    count is only a diagnostic; all-coincident points also have zero crossings.
    """
    P=_matrix(points,'points')
    if P.shape[1]!=2 or len(P)<3:
        raise ValueError('need >=3 2D points ordered along the physical rotation')
    unit=max(float(np.max(np.abs(P))),np.finfo(float).tiny)
    P=P/unit
    P=(P-P.mean(axis=0))/max(float(np.ptp(P,axis=0).max()),np.finfo(float).tiny)
    A=P[:,None,:];B=np.roll(P,-1,axis=0)[:,None,:]
    C=P[None,:,:];D=np.roll(P,-1,axis=0)[None,:,:]
    def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    s1=cross(B-A,C-A);s2=cross(B-A,D-A)
    s3=cross(D-C,A-C);s4=cross(D-C,B-C)
    eps=1e-12
    opposite=lambda a,b:((a>eps)&(b<-eps))|((a<-eps)&(b>eps))
    crossing=opposite(s1,s2)&opposite(s3,s4)
    i,j=np.indices((len(P),len(P)))
    valid=(j>i+1)&~((i==0)&(j==len(P)-1))
    return int(np.count_nonzero(crossing&valid))


def _h1(D):
    from ripser import ripser
    P=ripser(D,distance_matrix=True,maxdim=1)['dgms'][1]
    return P[np.isfinite(P).all(axis=1)&(P[:,1]>P[:,0])].astype(float)


def evaluate_rotation_orbits(reference,embedding,object_ids,angles_degrees,query_mask=None,
                              global_scale=1.,strength_threshold=.1):
    """All views of EVERY supplied object, not tiny random subcloud PH.

    For COIL-20: full 72-view H1 per object, 20 objects; angle metrics may be
    restricted to held-out views. One global scale and separate per-orbit scale
    results are both explicit. No automatic universal/semantic topology claim.
    """
    X,Z=_paired(reference,embedding)
    objects=np.asarray(object_ids);angles=np.asarray(angles_degrees,dtype=float)
    if objects.shape!=(len(X),) or angles.shape!=(len(X),):raise ValueError('annotation length mismatch')
    query=np.ones(len(X),dtype=bool) if query_mask is None else np.asarray(query_mask)
    if query.shape!=(len(X),) or query.dtype.kind!='b':raise ValueError('query_mask must be boolean with one value per row')
    if not np.isfinite(global_scale) or global_scale<0 or not np.isfinite(strength_threshold) or strength_threshold<0:
        raise ValueError('scales/threshold must be finite nonnegative')
    rows=[]
    for object_id in np.unique(objects):
        ids=np.flatnonzero(objects==object_id)
        ids=ids[np.argsort(angles[ids],kind='stable')]
        if len(ids)<3 or not query[ids].any():raise ValueError('each evaluated object needs >=3 views and a query')
        DX,DZ=distance_matrix(X[ids]),distance_matrix(Z[ids])
        P,Q=_h1(DX),_h1(DZ)
        numerator=float(np.sum(DX*DZ));denominator=float(np.sum(DZ*DZ))
        local_scale=numerator/denominator if denominator else 1.
        src_life=P[:,1]-P[:,0];dst_life=Q[:,1]-Q[:,0]
        source_median=float(np.median(DX[DX>0])) if np.any(DX>0) else 0.
        target_median=float(np.median(DZ[DZ>0])) if np.any(DZ>0) else 0.
        source_strength=float(src_life.max()/source_median) if len(P) and source_median else 0.
        target_strength=float(dst_life.max()/target_median) if len(Q) and target_median else 0.
        global_match=diagram_matching(P,Q*global_scale,max_matching_size=4096)
        local_match=diagram_matching(P,Q*local_scale,max_matching_size=4096)
        longest=int(np.argmax(src_life)) if len(P) else None
        row={'object_id':object_id.item() if isinstance(object_id,np.generic) else object_id,
             'sample_ids':ids.tolist(),'source_bars':len(P),'target_bars':len(Q),
             'source_diagram':P.tolist(),'target_diagram':Q.tolist(),
             'reference_angle':angle_neighbor_metrics(X[ids],angles[ids],query[ids]),
             'embedding_angle':angle_neighbor_metrics(Z[ids],angles[ids],query[ids]),
             'source_relative_longest_persistence':source_strength,
             'target_relative_longest_persistence':target_strength,
             'source_strong_h1':bool(source_strength>0 and source_strength>=strength_threshold),
             'target_strong_h1':bool(target_strength>0 and target_strength>=strength_threshold),
             'global_scale':float(global_scale),'per_object_scale':local_scale,
             'global_squared_transport':global_match['cost'], 'per_object_aligned_squared_transport':local_match['cost'],
             'longest_source_matched_global':None if longest is None else longest not in set(global_match['unmatched_source']),
             'longest_source_matched_per_object':None if longest is None else longest not in set(local_match['unmatched_source'])}
        if Z.shape[1]==2:row['strict_angle_order_polygon_crossings']=strict_polygon_crossings(Z[ids])
        rows.append(row)
    if not rows:raise ValueError('no objects')
    def mean(key,subkey):return float(np.mean([r[key][subkey] for r in rows]))
    return {'objects':rows,'summary':{
        'n_objects':len(rows),'source_strong_h1_objects':sum(r['source_strong_h1'] for r in rows),
        'target_strong_h1_objects':sum(r['target_strong_h1'] for r in rows),
        'reference_adjacent_recall':mean('reference_angle','adjacent_view_recall'),
        'embedding_adjacent_recall':mean('embedding_angle','adjacent_view_recall'),
        'reference_neighbor_angle_degrees':mean('reference_angle','mean_neighbor_angle_degrees'),
        'embedding_neighbor_angle_degrees':mean('embedding_angle','mean_neighbor_angle_degrees'),
        'mean_strict_crossings':float(np.mean([r['strict_angle_order_polygon_crossings'] for r in rows])) if Z.shape[1]==2 else None,
        'mean_per_object_aligned_transport':float(np.mean([r['per_object_aligned_squared_transport'] for r in rows]))},
        'strength_threshold':strength_threshold,
        'limitations':'Physical angles can be ambiguous for symmetric images. Strength threshold is a preset diagnostic, not a significance test. Diagram equality alone does not identify the same loop. Object IDs/angles used only for evaluation.'}
