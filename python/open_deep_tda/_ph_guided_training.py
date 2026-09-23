"""Opt-in source-witness-guided TRAIN fit of the SAME PH-trained DeepTDA MLP.

A certified source-only planar teacher supplies a target for the already
PH-trained estimator. Full all-pair H0 contacts and permanent exact F2 filling
cuts then enforce prescribed same-ID TRAIN obligations. This is conditional on
a supplied label-blind source witness family and supported one-loop or planar
subdivided-K4 structure. It is not generic automatic PH preservation or an
out-of-sample topology certificate. No graph-only estimator is used.
"""
from dataclasses import dataclass
import json
import math
import time

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from .structural_constructive import full_hierarchy, pairwise_planar, _mst
from .structural_h0 import compare_h0
from .structural_sparse_h1 import H1Limits, analyze_sparse_h1, ResourceLimitError
from ._ph_guided_source import construct_source_teacher, TeacherLimits
from ._ph_guided_filling import (FillingCutPool, FillingLimits,
                                 find_filling_obstructions, FillingResourceError)


class GuidedTrainingUnresolved(RuntimeError):
    """No accepted full-TRAIN embedding; no partial embedding exposed."""
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class GuidedLimits:
    max_vertices: int = 300
    max_feature_workspace_bytes: int = 268_435_456
    teacher_steps: int = 5000
    contact_steps: int = 2000
    cut_steps: int = 300
    max_seconds: float = 900.


def _limits(limits):
    if not isinstance(limits, GuidedLimits):
        raise ValueError('limits must be GuidedLimits')
    ceilings = dict(max_vertices=300,max_feature_workspace_bytes=268_435_456,
                    teacher_steps=5000,contact_steps=2000,cut_steps=300)
    for name,cap in ceilings.items():
        value=getattr(limits,name)
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not 0 <= value <= cap:
            raise ValueError(name+' must be an integer within the declared ceiling')
    t=limits.max_seconds
    if isinstance(t,(bool,np.bool_)) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<t<=3600:
        raise ValueError('max_seconds must be finite and in (0,3600]')


def _scale(reference,source_scale):
    if isinstance(source_scale, str) and source_scale == 'median_all_pairs':
        distances=pdist(reference)
        positive=distances[distances>0]
        value=float(np.median(positive)) if len(positive) else 0.0
    elif isinstance(source_scale,(int,float,np.integer,np.floating)) and not isinstance(source_scale,(bool,np.bool_)):
        value=float(source_scale)
    else:raise ValueError('source_scale must be positive or median_all_pairs')
    if not math.isfinite(value) or value<=0:
        raise ValueError('source_scale must be finite positive')
    return value


def _primitive(value):
    """Keep checkpoint reports weights-only-loadable (no NumPy scalars/pickle objects)."""
    if isinstance(value, np.generic):
        return _primitive(value.item())
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('nonfinite guided report scalar')
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('guided report keys must be strings')
        return {key: _primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    raise ValueError('guided report contains unsupported object type')


def _matrix(source):
    D=squareform(pdist(source))
    np.fill_diagonal(D,0.)
    if not np.isfinite(D).all() or not np.array_equal(D,D.T):
        raise ValueError('source Euclidean distance overflow/asymmetry')
    return D


def guide_existing_model(model, *, cycles, birth_radius, survival_radius,
                         source_scale=1., h0_tolerance=.05, strategy='single',
                         limits=GuidedLimits(), teacher_limits=TeacherLimits(),
                         filling_limits=FillingLimits(), h1_limits=H1Limits()):
    """Continue a PH-trained ``DeepTDA`` candidate; never return on failure.

    Caller must keep this candidate transactional until successful independent
    certification. Input features, source interval and cycles must be frozen
    before any target updates; no labels or validation data are consumed.
    An external Ripser proposal, if used to choose cycles, must be executed
    *separately* with explicit consent, not Torch+Numba co-imported here.
    """
    _limits(limits)
    cfg=model.config
    if (not model._fitted or cfg.mode!='geometry' or cfg.optimizer_mode!='parametric'
            or cfg.n_components!=2 or cfg.device!='cpu' or cfg.lambda_h0<=0 or cfg.lambda_h1<=0
            or cfg.steps<=cfg.warmup_steps or cfg.output_calibration!='none'):
        raise ValueError('requires fitted, uncalibrated CPU parametric 2D DeepTDA with at least one active PH H0/H1 update')
    original=np.asarray(model.reference_,dtype=np.float64)
    n,d=original.shape
    if not 4<=n<=limits.max_vertices or 3*n*n*d*8>limits.max_feature_workspace_bytes:
        raise ResourceLimitError('guided source vertex/feature-workspace budget exceeded')
    scale=_scale(original*model.reference_scale_,source_scale)
    # Source units are defined by TRAIN reference only. float32 in the native
    # estimator is promoted to float64 BEFORE its distance construction.
    X=np.ascontiguousarray(original*(model.reference_scale_/scale),dtype=np.float64)
    D=_matrix(X)
    if isinstance(h0_tolerance,(bool,np.bool_)) or not isinstance(h0_tolerance,(int,float,np.integer,np.floating)) or not math.isfinite(float(h0_tolerance)) or not 0<float(h0_tolerance)<=1:
        raise ValueError('h0_tolerance must be finite in (0,1]')
    started=time.monotonic()
    teacher=construct_source_teacher(D,X,cycles,birth_radius,survival_radius,h0_tolerance,
                                     strategy=strategy,limits=teacher_limits,h1_limits=h1_limits)
    if not teacher.certified or teacher.embedding is None:
        raise GuidedTrainingUnresolved('source structural teacher unsupported: '+teacher.reason,
             {'stage':'source_teacher','reason':teacher.reason,'strategy':strategy,
              'source_diagnostics':teacher.diagnostics})
    if time.monotonic()-started>limits.max_seconds:
        raise TimeoutError('guided source teacher exceeded wall limit')
    guide=np.asarray(teacher.embedding,dtype=np.float64)
    if guide.shape!=(n,2) or not np.isfinite(guide).all():
        raise RuntimeError('source teacher returned invalid embedding')
    source_hierarchy=full_hierarchy(D)
    pairs=np.triu_indices(n,1)
    lower=torch.tensor(np.maximum(0.,source_hierarchy[pairs]-.9*h0_tolerance),dtype=torch.float32)
    tree=np.array(_mst(D),dtype=np.int64).reshape(-1,2)
    upper=torch.tensor(D[tree[:,0],tree[:,1]]+.9*h0_tolerance,dtype=torch.float32)
    source=torch.tensor(model.reference_.copy(),dtype=torch.float32)
    target=torch.tensor(guide,dtype=torch.float32)
    net=model.model_;net.train()
    factor=model.reference_scale_/scale
    trace=[]
    def candidate():
        # Certify the SAME represented float32 coordinates that are stored in
        # embedding_ and returned by transform, then convert units in float64.
        # Multiplication inside torch would round factor to float32 first and
        # could pass a threshold that the reloaded model fails by one ULP.
        with torch.no_grad():
            encoded=net(source).detach().cpu().numpy().astype(np.float64)
        return encoded*float(factor)
    def certificate(Z):
        T=pairwise_planar(Z)
        h0=compare_h0(D,T,tolerance=h0_tolerance,max_vertices=limits.max_vertices)
        h1=analyze_sparse_h1(T,cycles,birth_radius,survival_radius,limits=h1_limits)
        return {'h0_error':float(h0['max_merge_error']), 'selected_rank':int(h1.rank),
                'h1_certified':bool(h1.certified),
                'accepted':bool(h0['max_merge_error']<=h0_tolerance and h1.certified)}
    def deadline():
        if time.monotonic()-started>limits.max_seconds:
            raise TimeoutError('guided PH training wall limit exceeded')
    # All intermediate status entries, including rank regressions, are retained
    # in the successful report; never pick a best-looking iterate or seed.
    optim=torch.optim.Adam(net.parameters(),lr=cfg.learning_rate)
    for step in range(limits.teacher_steps):
        deadline()
        Z=net(source)*factor
        loss=(Z-target).square().mean()
        optim.zero_grad(set_to_none=True);loss.backward();optim.step()
    trace.append(dict(stage='teacher',steps=limits.teacher_steps,**certificate(candidate())))
    if not trace[-1]['accepted']:
        optim=torch.optim.Adam(net.parameters(),lr=cfg.learning_rate)
        for step in range(limits.contact_steps+1):
            deadline()
            Z=net(source)*factor
            distance=torch.linalg.vector_norm(Z[pairs[0]]-Z[pairs[1]],dim=1)
            below=torch.relu(lower-distance).square().sum()
            e=torch.linalg.vector_norm(Z[tree[:,0]]-Z[tree[:,1]],dim=1)
            above=torch.relu(e-upper).square().sum()
            drift=(Z-target).square().sum()
            loss=below+above+.0001*drift
            if step%50==0 or step==limits.contact_steps:
                checked=certificate(candidate())
                if step%500==0 or checked['accepted']:
                    trace.append(dict(stage='h0_contacts',step=step,**checked))
                if checked['accepted']:
                    break
            if step==limits.contact_steps:break
            optim.zero_grad(set_to_none=True);loss.backward();optim.step()
        else:raise AssertionError('contact loop protocol error')
    current=certificate(candidate())
    if not current['accepted'] and limits.cut_steps:
        pool=FillingCutPool(cycles,birth_radius,survival_radius,
                    margin=.001*(survival_radius-birth_radius),max_cuts=500)
        optim=torch.optim.Adam(net.parameters(),lr=cfg.learning_rate)
        for step in range(0,limits.cut_steps+1,5):
            deadline()
            current=certificate(candidate())
            if current['accepted']:
                trace.append(dict(stage='persistent_f2_cuts',step=step,pool_size=len(pool.cuts),**current))
                break
            Zdet=candidate();T=pairwise_planar(Zdet)
            # A missing target survival edge is explicitly handled by the
            # dedicated live birth/survival hinges before another oracle scan.
            family_present=all(T[tuple(sorted(e))]<=survival_radius for cyc in cycles for e in cyc)
            if family_present:
                result=find_filling_obstructions(D,T,cycles,birth_radius,survival_radius,limits=filling_limits)
                pool.add(result)
            trace.append(dict(stage='persistent_f2_cuts',step=step,pool_size=len(pool.cuts),
                              oracle_status='complete' if family_present else 'missing_survival_edges',**current))
            if step==limits.cut_steps:break
            for _ in range(min(5,limits.cut_steps-step)):
                deadline()
                Z=net(source)*factor
                d=torch.linalg.vector_norm(Z[pairs[0]]-Z[pairs[1]],dim=1)
                below=torch.relu(lower-d).square().sum()
                e=torch.linalg.vector_norm(Z[tree[:,0]]-Z[tree[:,1]],dim=1)
                above=torch.relu(e-upper).square().sum()
                born,survive,cut=pool.loss(Z)
                objective=below+above+born+survive+cut
                optim.zero_grad(set_to_none=True);objective.backward();optim.step()
        current=certificate(candidate())
    if not current['accepted']:
        raise GuidedTrainingUnresolved('complete TRAIN H0/H1 constraints unresolved',
            {'stage':'joint_confirmation','strategy':strategy,'source_rows':n,
             'final':current,'history':trace,'source_diagnostics':teacher.diagnostics})
    model.model_.eval()
    with torch.no_grad():model.embedding_=model.model_(source).detach().cpu().numpy().copy()
    represented=model.embedding_.astype(np.float64)*float(factor)
    stored_certificate=certificate(represented)
    if not stored_certificate['accepted']:
        raise GuidedTrainingUnresolved('represented checkpoint coordinates fail full TRAIN certificate',
            {'stage':'stored_coordinate_confirmation','final':stored_certificate,'history':trace})
    current=stored_certificate
    model.report_['guided_training']=_primitive({
        'status':'certified_selected_TRAIN_family', 'scope':'all supplied TRAIN IDs only; no OOS/full-barcode guarantee',
        'strategy':strategy, 'source_scale':scale,'source_units':'TRAIN-only fixed reference distance unit',
        'a':float(birth_radius),'b':float(survival_radius),'h0_tolerance':float(h0_tolerance),
        'source_witness_lengths':[len(c) for c in cycles],
        'requested_native_ph_subcloud_size':cfg.h1_size,
        'effective_native_ph_subcloud_size':min(cfg.h1_size,n),
        'certificate':current,'source_teacher':teacher.diagnostics,
        'optimization_history':trace,
        'phase_scopes':'original native PH fit; source-certified teacher; full-source H0 contacts; adaptive source-ID F2 cuts',
        'timings':{'guided_seconds':time.monotonic()-started},
        'query_claim':'transform remains inductive, but adding even one query changes PH/H0; no inherited certificate'})
    json.dumps(model.report_, allow_nan=False)
    return model
