"""Opt-in source-only teacher interpolation of a PH-trained MLP's existing head.

This is a TRAIN-ID certificate, not evidence that sampled native H1 caused the
layout and not a certificate for new rows. The caller owns the temporary model
and publishes it only after transform parity and report serialization succeed.
"""
import math
import time

import numpy as np
import torch

from ._ph_guided_training import (GuidedTrainingUnresolved, _matrix, _primitive,
                                  _scale)
from ._ph_guided_source import construct_source_teacher
from .preprocessing import NumericPreprocessor, numeric_matrix
from .structural_constructive import pairwise_planar
from .structural_h0 import compare_h0
from .structural_sparse_h1 import ResourceLimitError, analyze_sparse_h1


RCOND = 1e-12
MAX_CONDITION = 1e12


def preflight_affine_reference(X, cfg, source_scale, max_seconds):
    """Check the exact TRAIN reference rank before the costly native PH fit.

    Geometry mode's preprocessing, float32 cast, and seeded pair-sample scale
    match DeepTDA._fit. No model parameters, labels or target coordinates enter.
    This is a checked wall budget, not an interrupt for a native LAPACK call.
    """
    started = time.monotonic()
    original = numeric_matrix(X)
    H = NumericPreprocessor(cfg.standardize, cfg.missing_indicators).fit(original).transform(original)
    n = len(H)
    rng = np.random.default_rng(cfg.seed)
    pair_count = min(max(1024, 2*n), 20000)
    ids = rng.integers(n, size=(pair_count, 2))
    lengths = np.linalg.norm(H[ids[:, 0]].astype(float)-H[ids[:, 1]], axis=1)
    positive = lengths[lengths > 0]
    reference_scale = float(np.median(positive)) if len(positive) else 1.0
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        reference = np.ascontiguousarray(H/reference_scale, dtype=np.float32)
    if not np.isfinite(reference).all() or not math.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError('affine preflight reference scale is not finite and positive')
    scale = _scale(reference.astype(np.float64)*reference_scale, source_scale)
    factor = reference_scale/scale
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('unrepresentable affine source unit conversion before native fit')
    with np.errstate(over='ignore', invalid='ignore'):
        scaled = reference.astype(np.float64)*factor
    if not np.isfinite(scaled).all():
        raise ValueError('affine scaled source coordinates overflow before native fit')
    # A finite factor can still overflow the squared differences in Euclidean
    # distances; this is the very source metric the teacher later certifies.
    from scipy.spatial.distance import pdist
    if not np.isfinite(pdist(scaled)).all():
        raise ValueError('affine scaled source distances overflow before native fit')
    if time.monotonic()-started > max_seconds:
        raise TimeoutError('guided affine preflight wall limit exceeded before native fit')
    A = np.column_stack((reference.astype(np.float64), np.ones(n, dtype=np.float64)))
    singular = np.linalg.svd(A, compute_uv=False)
    if time.monotonic()-started > max_seconds:
        raise TimeoutError('guided affine preflight wall limit exceeded before native fit')
    if not np.isfinite(singular).all() or not len(singular) or singular[0] <= 0:
        raise ValueError('affine preflight source reference is rank deficient')
    rank = int(np.count_nonzero(singular > singular[0]*RCOND))
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else math.inf
    if rank != n or not math.isfinite(condition) or condition > MAX_CONDITION:
        raise ValueError('affine preflight source reference rank/condition refused')
    return rank, condition


def guide_affine_min_norm(model, *, cycles, birth_radius, survival_radius,
                          source_scale, h0_tolerance, limits, teacher_limits,
                          h1_limits):
    """Modify only a candidate's linear head after independently certified teacher."""
    cfg = model.config
    coverage = model.training_coverage_
    if (not model._fitted or cfg.mode != 'geometry' or cfg.optimizer_mode != 'parametric'
            or cfg.n_components != 2 or cfg.device != 'cpu'
            or cfg.output_calibration != 'none' or cfg.lambda_h0 <= 0 or cfg.lambda_h1 <= 0
            or coverage.get('h0_updates', 0) <= 0 or coverage.get('h1_updates', 0) <= 0):
        raise ValueError('affine teacher requires actual PH-active CPU parametric 2D native fit')
    if len(cycles) != 1:
        raise ValueError('affine teacher requires exactly one source cycle')
    n, d = model.reference_.shape
    # A, thin Vt, several working matrices and SVD temporaries; logical
    # allocation ceiling, not a bound on BLAS library internals or process RSS.
    if (n > limits.max_vertices or n > teacher_limits.max_vertices or
            3*n*n*d*8 > min(limits.max_feature_workspace_bytes,
                            teacher_limits.max_distance_workspace_bytes) or
            4*n*(d+1)*8 + 4*n*n*8 + 4*(d+1)*2*8 >
            limits.max_feature_workspace_bytes):
        raise ResourceLimitError('affine head workspace budget exceeded')
    if d+1 < n:
        raise ValueError('affine head cannot interpolate: reference dimension + 1 < TRAIN rows')
    started = time.monotonic()
    def deadline():
        if time.monotonic()-started > limits.max_seconds:
            raise TimeoutError('guided affine solve/certificate/report wall limit exceeded')
    original = model.reference_.astype(np.float64)
    scale = _scale(original*model.reference_scale_, source_scale)
    factor = float(model.reference_scale_/scale)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('unrepresentable source unit conversion')
    X = np.ascontiguousarray(original*factor)
    D = _matrix(X)
    if (isinstance(h0_tolerance, (bool, np.bool_)) or
            not isinstance(h0_tolerance, (int, float, np.integer, np.floating)) or
            not math.isfinite(float(h0_tolerance)) or not 0 < h0_tolerance <= 1):
        raise ValueError('h0_tolerance must be finite in (0,1]')
    deadline()
    # Do not rely on a constructor's boolean alone: the original-ID source
    # witness is independently valid in these very source units, even if a
    # buggy/fault-injected teacher reports itself certified.
    source = analyze_sparse_h1(D, cycles, birth_radius, survival_radius,
                               limits=h1_limits)
    deadline()
    if not source.certified:
        raise GuidedTrainingUnresolved('affine source selected H1 family refused',
             {'stage':'source_contract', 'reason':source.reason})
    teacher = construct_source_teacher(D, X, cycles, birth_radius, survival_radius,
                                        h0_tolerance, strategy='single_beam',
                                        limits=teacher_limits, h1_limits=h1_limits)
    deadline()
    if not teacher.certified or teacher.embedding is None:
        raise GuidedTrainingUnresolved('source structural teacher unsupported: '+teacher.reason,
              {'stage':'source_teacher', 'reason':teacher.reason,
               'source_diagnostics':teacher.diagnostics})
    guide = np.asarray(teacher.embedding, dtype=np.float64)
    if guide.shape != (n, 2) or not np.isfinite(guide).all():
        raise ValueError('invalid certified source teacher coordinates')
    A = np.column_stack((original, np.ones(n, dtype=np.float64)))
    # The LIVE native float32 forward, not a reimplementation of SiLU in
    # float64. The head correction is additive even with a nonzero residual.
    before = model._forward_numpy(model.model_, model.reference_).astype(np.float64)
    target = guide/factor
    if not np.isfinite(target).all() or not np.isfinite(before).all():
        raise ValueError('unrepresentable teacher/native coordinates')
    deadline()
    U, singular, Vt = np.linalg.svd(A, full_matrices=False)
    deadline()  # an individual LAPACK call is not preemptible
    rank = int(np.count_nonzero(singular > singular[0]*RCOND))
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else math.inf
    if (not np.isfinite(singular).all() or rank != n or
            not math.isfinite(condition) or condition > MAX_CONDITION):
        raise GuidedTrainingUnresolved('affine head rank/condition refused',
             {'stage':'head_solve', 'rank':rank,
              'condition':condition if math.isfinite(condition) else None})
    delta = Vt.T @ ((U.T @ (target-before))/singular[:, None])
    error = A @ delta - (target-before)
    deadline()
    if (not np.isfinite(delta).all() or not np.isfinite(error).all() or
            np.max(np.abs(error)) > 1e-8*max(1., float(np.max(np.abs(target-before))))):
        raise GuidedTrainingUnresolved('affine head cannot represent exact float64 solve',
                                       {'stage':'head_solve'})
    linear = model.model_.linear
    weights = linear.weight.detach().cpu().numpy().astype(np.float64) + delta[:-1].T
    bias = linear.bias.detach().cpu().numpy().astype(np.float64) + delta[-1]
    with np.errstate(over='ignore', invalid='ignore'):
        w32, b32 = weights.astype(np.float32), bias.astype(np.float32)
    if not all(np.isfinite(x).all() for x in (weights, bias, w32, b32)):
        raise GuidedTrainingUnresolved('affine head update unrepresentable in float32',
                                       {'stage':'head_solve'})
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(w32))
        linear.bias.copy_(torch.from_numpy(b32))
    model.model_.eval()
    embedding = model._forward_numpy(model.model_, model.reference_)
    represented = embedding.astype(np.float64)*factor
    if not np.isfinite(represented).all():
        raise GuidedTrainingUnresolved('affine output nonfinite', {'stage':'final_certificate'})
    teacher_error = float(np.max(np.abs(represented-guide)))
    deadline()
    T = pairwise_planar(represented)
    h0 = compare_h0(D, T, tolerance=h0_tolerance, max_vertices=limits.max_vertices)
    h1 = analyze_sparse_h1(T, cycles, birth_radius, survival_radius, limits=h1_limits)
    deadline()
    certificate = {'h0_error':float(h0['max_merge_error']),
                   'selected_rank':int(h1.rank), 'h1_certified':bool(h1.certified),
                   'accepted':bool(h0['certified_within_tolerance'] and h1.certified)}
    if not certificate['accepted']:
        raise GuidedTrainingUnresolved('represented float32 affine TRAIN certificate refused',
             {'stage':'final_certificate', 'final':certificate})
    model.embedding_ = embedding.copy()
    model.report_['guided_training'] = _primitive({
        'status':'certified_selected_TRAIN_family',
        'teacher_realization':'affine_min_norm', 'strategy':'single_beam',
        'scope':'all supplied TRAIN IDs only; no OOS/full-barcode guarantee',
        'phase_scopes':'original native PH fit; first source-certified complete beam state; source-derived affine head; no native H1 causal or OOS guarantee',
        'source_scale':scale, 'source_units':'TRAIN-only fixed reference distance unit',
        'a':float(birth_radius), 'b':float(survival_radius),
        'h0_tolerance':float(h0_tolerance), 'source_witness_lengths':[len(c) for c in cycles],
        'svd_rcond':RCOND, 'rank':rank, 'condition':condition,
        'max_fitted_teacher_error':teacher_error,
        'native_sampled_ph_activation':{
            'optimizer_steps':coverage['optimizer_steps'],
            'h0_updates':coverage['h0_updates'], 'h1_updates':coverage['h1_updates']},
        'certificate':certificate, 'source_teacher':teacher.diagnostics,
        'timings':{'guided_seconds':time.monotonic()-started},
        'query_claim':'transform remains inductive, but adding even one query changes PH/H0; no inherited certificate'})
    deadline()
