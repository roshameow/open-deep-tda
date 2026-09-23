"""End-to-end frozen-reference topology-regularized dimensionality reduction."""
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
import copy
import json
import time
import tempfile

import numpy as np
import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors

from .config import TDAConfig
from .models import ParametricEmbedding, NumericAutoencoder, reconstruction_decoder
from .preprocessing import NumericPreprocessor, numeric_matrix
from .sampling import TopologySampler, GeometrySampler


@contextmanager
def _torch_context(seed, num_threads, device):
    previous_threads = torch.get_num_threads()
    cuda_devices = list(range(torch.cuda.device_count())) if device == "cuda" else []
    mps_state = torch.mps.get_rng_state() if device == "mps" else None
    try:
        torch.set_num_threads(num_threads)
        with torch.random.fork_rng(devices=cuda_devices):
            # Seed only the generators whose states this context restores.
            torch.random.default_generator.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)
            elif device == "mps":
                torch.mps.manual_seed(seed)
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        torch.set_num_threads(previous_threads)


def _tensor_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _skipped_output_calibration(method="none", reason="disabled"):
    return {"method": method, "status": "skipped", "reason": reason,
            "source": "TRAIN", "pairs": 0, "alpha": 1.0,
            "fit_stress_before": None, "fit_stress_after": None}


class DeepTDA:
    """Numeric geometry or masked-autoencoder semantic reduction.

    Labels never enter fit. Supply validation_data explicitly for held-out evaluation;
    otherwise report_ describes training samples/subsets, not held-out generalization.
    CPU is the reproducible default. Direct coordinate mode intentionally has no transform.
    geometry_objective='fuzzy_graph' opts into weighted graph attraction instead
    of fuzzy edge cross-entropy, with the same sampled nonedge repulsion. This
    is not exact UMAP or a claim of improvement over stress/fuzzy.
    output_calibration='train_pairs' fits one positive output-distance scalar
    after optimization, using only frozen TRAIN references and embeddings. It
    changes scale, not topology shape or the training loss. Training histories
    and subset diagnostics remain pre-calibration; final reports use calibrated
    outputs. Constant references need no calibration and retain scale one.
    """
    def __init__(self, config=None, **config_overrides):
        if config is None:
            values = {}
        elif isinstance(config, TDAConfig):
            values = config.to_dict()
        elif isinstance(config, dict):
            values = dict(config)
        else:
            raise TypeError("config must be TDAConfig, dict or None")
        values.update(config_overrides)
        self.config = TDAConfig(**values).validate()
        self._fitted = False

    def _device(self):
        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available")
        return torch.device(device)

    def _forward_numpy(self, module, X):
        if len(X) == 0:
            raise ValueError("internal forward requires nonempty array")
        was_training = module.training
        module.eval()
        results = []
        try:
            with torch.inference_mode():
                for start in range(0, len(X), self.config.inference_batch_size):
                    batch = torch.from_numpy(np.ascontiguousarray(X[start:start + self.config.inference_batch_size])).to(self.device_)
                    results.append(module(batch).cpu().numpy())
        finally:
            module.train(was_training)
        result = np.concatenate(results)
        if not np.isfinite(result).all():
            raise RuntimeError("neural inference produced nonfinite values; rescale inputs")
        return result

    def _train_reference(self, processed, original):
        cfg = self.config
        if cfg.semantic_steps < 1:
            raise ValueError("semantic mode requires semantic_steps >= 1")
        model = NumericAutoencoder(processed.shape[1], cfg.semantic_dim).to(self.device_)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        data = torch.from_numpy(processed)
        d = original.shape[1]
        observed = torch.from_numpy(~np.isnan(original))
        rng = np.random.default_rng(cfg.seed + 11)
        self.semantic_history_ = []
        for step in range(cfg.semantic_steps):
            ids = rng.integers(len(data), size=min(cfg.batch_size, len(data)))
            clean = data[ids].to(self.device_)
            valid = observed[ids].to(self.device_)
            mask = (torch.rand(clean.shape[0], d, device=self.device_) < cfg.mask_probability) & valid
            if not bool(mask.any()):
                mask = valid
            if not bool(mask.any()):
                continue
            corrupted = clean.clone()
            corrupted[:, :d] = torch.where(mask, torch.zeros_like(clean[:, :d]), clean[:, :d])
            if cfg.missing_indicators:
                corrupted[:, d:] = torch.maximum(corrupted[:, d:], mask.to(clean.dtype))
            pred = model(corrupted)
            loss = ((pred[:, :d] - clean[:, :d])[mask] ** 2).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite self-supervised loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            if step % cfg.log_interval == 0 or step == cfg.semantic_steps - 1:
                self.semantic_history_.append({"step": step, "masked_mse": float(loss.detach())})
        self.reference_encoder_ = model.encoder.eval()
        for parameter in self.reference_encoder_.parameters():
            parameter.requires_grad_(False)
        return self._forward_numpy(self.reference_encoder_, processed)

    def fit(self, X, validation_data=None):
        self.config.validate()
        self._fitted = False
        self.output_scale_ = 1.0
        self.calibration_diagnostics_ = _skipped_output_calibration(self.config.output_calibration)
        self.device_ = self._device()
        original = numeric_matrix(X)
        validation = None if validation_data is None else numeric_matrix(validation_data)
        if validation is not None and validation.shape[1] != original.shape[1]:
            raise ValueError("validation_data has different feature count")
        start = time.perf_counter()
        if validation is not None and self.config.optimizer_mode == "coordinates":
            raise ValueError("validation_data requires parametric mode")
        try:
            with _torch_context(self.config.seed, self.config.num_threads, self.config.device):
                self._fit(original, validation)
        except Exception:
            self._fitted = False
            raise
        self.fit_seconds_ = time.perf_counter() - start
        self.report_["fit_seconds"] = self.fit_seconds_
        self.report_["timings"] = getattr(self, "timings_", {})
        return self

    def fit_with_topology_guidance(self, X, *, cycles, birth_radius, survival_radius,
                                   source_scale=1.0, h0_tolerance=0.05,
                                   strategy='single', limits=None,
                                   teacher_limits=None, filling_limits=None,
                                   h1_limits=None):
        """Opt-in, fail-closed source-witness-guided fit of THIS PH-trained MLP.

        ``cycles`` and radii must be selected from the source without labels or
        target access. Source PH proposal with external Ripser, if required,
        must be performed separately with explicit user consent. Supported
        topology is one simple source cycle or a subdivided-K4 family; this is
        not generic topology preservation. An accepted TRAIN certificate
        NEVER applies automatically to transform/new vertices.

        On failure this estimator remains unchanged (including if it was
        previously fitted), and no partial embedding is returned. Resource
        exhaustion is an error, not a partial certificate. The original
        ordinary ``fit`` and legacy checkpoints retain their behavior.
        """
        from ._ph_guided_training import (GuidedLimits, GuidedTrainingUnresolved,
                                          guide_existing_model, _limits, _primitive)
        from ._ph_guided_source import TeacherLimits, _check_limits as _check_teacher_limits
        from ._ph_guided_filling import FillingLimits, _limits as _check_filling_limits
        from .structural_sparse_h1 import (H1Limits, ResourceLimitError,
                                           _check_limits as _check_h1_limits)
        from .evaluation import evaluate_embedding
        if strategy not in ('single', 'single_beam', 'subdivided_k4'):
            raise ValueError('unsupported source topology strategy')
        plan = GuidedLimits() if limits is None else limits
        source_limits = TeacherLimits() if teacher_limits is None else teacher_limits
        obstruction_limits = FillingLimits() if filling_limits is None else filling_limits
        topological_limits = H1Limits() if h1_limits is None else h1_limits
        _limits(plan)
        _check_teacher_limits(source_limits)
        _check_filling_limits(obstruction_limits)
        _check_h1_limits(topological_limits)
        # Fail BEFORE native PH fit or materializing an unbounded generator.
        # This opt-in mode requires ordinary ndarray metadata for hard shape
        # and temporary distance-workspace preflight; ordinary fit remains
        # permissive and unchanged.
        if type(X) is not np.ndarray or X.ndim != 2 or X.dtype.kind not in 'iuf':
            raise ValueError('guided X must be an ordinary real numeric ndarray')
        n,d = X.shape
        if n < 4 or not 2 <= d <= 4096:
            raise ValueError('guided source requires at least four rows and 2..4096 features')
        if (n > plan.max_vertices or n > source_limits.max_vertices or
                n > obstruction_limits.max_vertices or n > topological_limits.max_vertices or
                n*(n-1)//2 > source_limits.max_pairs or
                3*n*n*d*8 > min(plan.max_feature_workspace_bytes,
                                 source_limits.max_distance_workspace_bytes)):
            raise ResourceLimitError('guided source vertex/pair/feature-workspace budget exceeded before fit')
        expected = 3 if strategy == 'subdivided_k4' else 1
        family = []
        try:
            for index, chain in enumerate(cycles):
                if index >= expected:
                    raise ResourceLimitError('guided source witness family exceeds supported size')
                edges = []
                for j, edge in enumerate(chain):
                    if j >= n:
                        raise ResourceLimitError('guided source cycle exceeds vertex/edge budget')
                    # An individual edge may itself be an unbounded iterator.
                    # Read at most three items before rejecting non-pair input.
                    pair = tuple(islice(edge, 3))
                    if len(pair) != 2:
                        raise ValueError('guided source edge must contain exactly two vertex IDs')
                    edges.append(pair)
                family.append(tuple(edges))
        except ResourceLimitError:
            raise
        except (TypeError, ValueError) as exc:
            raise ValueError('cycles must be a finite family of edge lists') from exc
        if len(family) != expected:
            raise ValueError('guided strategy requires exactly %d source cycle(s)' % expected)
        family = tuple(family)
        # A candidate owns every temporary parameter/tensor and report.
        # Publishing its state is the final atomic act, AFTER verification.
        start = time.perf_counter()
        candidate = type(self)(self.config.to_dict())
        candidate.fit(X)
        guided_started = time.monotonic()
        with _torch_context(candidate.config.seed, candidate.config.num_threads, candidate.config.device):
            guide_existing_model(candidate, cycles=family, birth_radius=birth_radius,
                                 survival_radius=survival_radius, source_scale=source_scale,
                                 h0_tolerance=h0_tolerance, strategy=strategy, limits=plan,
                                 teacher_limits=source_limits, filling_limits=obstruction_limits,
                                 h1_limits=topological_limits)
        # The stored float32 output must be EXACTLY what public transform
        # returns for the same training rows. It is the representation whose
        # float64 source-unit conversion received the independent certificate.
        query = candidate.transform(X)
        if not np.array_equal(query, candidate.embedding_):
            error = float(np.max(np.abs(query.astype(np.float64)-candidate.embedding_)))
            raise GuidedTrainingUnresolved('stored TRAIN layout differs from transform',
                                            {'stage':'transform_parity','maximum_absolute_error':error})
        # The initial fit's sampled geometry/PH report describes OLD coordinates.
        # Recompute its ordinary diagnostics after guided training; the separate
        # all-TRAIN selected-family certificate is in guided_training.
        candidate.report_.update(evaluate_embedding(candidate.reference_, candidate.embedding_,
            topology_size=min(candidate.config.evaluation_size, len(candidate.reference_)),
            seed=candidate.config.seed + 991, k=candidate.config.n_neighbors,
            budgets=candidate.config.evaluation_budgets()))
        candidate.report_['status'] = 'guided_certified_train'
        candidate.report_['claims'] = ('Independent selected same-ID H0/H1 certificate on all supplied TRAIN rows; '
                                      'not all H1, generic clustering, or new-query topology.')
        candidate.report_['history_scope'] = ('native PH training history precedes source-only teacher, '
                                              'full-domain contact and F2 guidance phases')
        candidate.fit_seconds_ = time.perf_counter() - start
        candidate.report_['fit_seconds'] = candidate.fit_seconds_
        candidate.report_['guided_training'] = _primitive(candidate.report_['guided_training'])
        json.dumps(candidate.report_, allow_nan=False)
        if time.monotonic() - guided_started > plan.max_seconds:
            raise TimeoutError('guided fit/report wall limit exceeded before publication')
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)
        return self

    def _fit(self, original, validation):
        from . import topology, losses
        from .evaluation import evaluate_embedding
        cfg = self.config
        self.history_, self.validation_history_, self.semantic_history_ = [], [], []
        self.timings_ = {}
        self.training_coverage_ = {}
        self.neighbor_diagnostics_ = {}
        self.geometry_diagnostics_ = {}
        preprocess_start = time.perf_counter()
        self.preprocessor_ = NumericPreprocessor(cfg.standardize, cfg.missing_indicators).fit(original)
        processed = self.preprocessor_.transform(original)
        self.timings_["preprocessing_seconds"] = time.perf_counter() - preprocess_start
        self.processed_dim_ = processed.shape[1]
        self.reference_encoder_ = None
        # Constant input does not acquire invented structure from a randomly initialized encoder.
        input_constant = bool(np.all(processed == processed[0]))
        reference_start = time.perf_counter()
        H = self._train_reference(processed, original) if cfg.mode == "semantic" and not input_constant else processed
        self.timings_["reference_learning_seconds"] = time.perf_counter() - reference_start
        rng = np.random.default_rng(cfg.seed)
        pair_count = min(max(1024, len(H) * 2), 20000)
        ids = rng.integers(len(H), size=(pair_count, 2))
        lengths = np.linalg.norm(H[ids[:, 0]].astype(float) - H[ids[:, 1]], axis=1)
        positive = lengths[lengths > 0]
        self.reference_scale_ = float(np.median(positive)) if len(positive) else 1.0
        self.reference_ = np.ascontiguousarray(H / self.reference_scale_, dtype=np.float32)
        if not np.isfinite(self.reference_).all():
            raise ValueError("nonfinite reference space; rescale data")
        self.reference_dim_ = self.reference_.shape[1]
        self.constant_ = bool(np.all(self.reference_ == self.reference_[0]))
        self.model_ = ParametricEmbedding(self.reference_dim_, cfg.n_components,
                                          residual_input_scale=cfg.residual_input_scale)
        self.input_rank_ = self.model_.initialize_pca(self.reference_)
        self.model_.to(self.device_)
        self.decoder_ = None
        self._coordinates = None
        if self.constant_:
            self.calibration_diagnostics_ = _skipped_output_calibration(
                cfg.output_calibration, "constant_reference: no calibration necessary")
            if cfg.geometry_objective == "fuzzy_graph":
                self.geometry_diagnostics_ = {"objective": "fuzzy_graph",
                    "positive_mode": "attraction", "trained": False}
            with torch.no_grad():
                for p in self.model_.parameters():
                    p.zero_()
            self.embedding_ = np.zeros((len(original), cfg.n_components), dtype=np.float32)
            self._fitted = True
            self.report_ = {"status": "constant_reference", "n_samples": len(original),
                            "reference_rank": 0, "warning": "No nontrivial structure is inferred."}
        else:
            self._optimize(topology, losses, rng)
            if cfg.output_calibration == "train_pairs":
                self._calibrate_output()
            self._fitted = True
            self.report_ = evaluate_embedding(self.reference_, self.embedding_,
                topology_size=min(cfg.evaluation_size, len(self.reference_)), seed=cfg.seed + 991,
                k=cfg.n_neighbors, budgets=cfg.evaluation_budgets())
            self.report_.update({"status": "fitted", "reference_rank": self.input_rank_})
        self.model_.eval()
        if self.decoder_ is not None:
            self.decoder_.eval()
        self.report_["reference_mode"] = cfg.mode
        self.report_["evaluation_scope"] = "training data, independent fixed evaluation subset"
        self.report_["config"] = cfg.to_dict()
        self.report_["reference_scale"] = self.reference_scale_
        self.report_["output_calibration"] = copy.deepcopy(self.calibration_diagnostics_)
        self.report_["history_scope"] = "training history and training-subset diagnostics are pre-calibration"
        self.report_["training_coverage"] = self.training_coverage_
        self.report_["neighbor_graph"] = self.neighbor_diagnostics_
        self.report_["geometry_objective"] = self.geometry_diagnostics_
        self.report_["claims"] = "Sampled H0/H1 regularization; no full-data or out-of-distribution guarantee."
        if validation is not None:
            if cfg.optimizer_mode == "coordinates":
                raise ValueError("validation_data requires parametric mode (coordinates cannot transform)")
            ref = self.reference_transform(validation)
            z = self.transform(validation)
            self.report_["held_out"] = evaluate_embedding(ref, z, topology_size=cfg.evaluation_size,
                seed=cfg.seed + 997, k=cfg.n_neighbors, budgets=cfg.evaluation_budgets())
        # Test JSON compatibility now, rather than failing after a lengthy CLI run.
        json.dumps(self.report_, allow_nan=False)

    def _calibrate_output(self):
        """Fit bounded TRAIN-pair LS scale; never consume optimizer/evaluation RNG.

        Sampling is with replacement (including self-pairs). Diagnostics contain
        only aggregates, not pairs or rows. Stress follows evaluation's normalized
        root squared-error convention, on these calibration pairs only.
        """
        count = min(max(1024, 2 * len(self.reference_)), 20000)
        seed = self.config.seed + 211
        ids = np.random.default_rng(seed).integers(len(self.reference_), size=(count, 2))

        def distances(X):
            # Promote BEFORE subtraction/norm to avoid float32 overflow.
            difference = X[ids[:, 0]].astype(np.float64) - X[ids[:, 1]].astype(np.float64)
            return np.linalg.norm(difference, axis=1)

        source, target = distances(self.reference_), distances(self.embedding_)
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("output calibration requires finite pair distances")
        unit = max(float(source.max()), float(target.max()), np.finfo(np.float64).tiny)
        a, b = source / unit, target / unit
        denominator = float(b @ b)
        alpha = float((a @ b) / denominator) if denominator > 0 else 1.0
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("output calibration requires a finite positive scale")
        source_energy = float(a @ a)

        def stress(scale):
            residual = a - scale * b
            energy = float(residual @ residual)
            result = float(np.sqrt(energy / source_energy)) if source_energy > 0 else (0.0 if energy == 0 else None)
            if result is not None and not np.isfinite(result):
                raise ValueError("output calibration stress overflowed; rescale inputs")
            return result

        diagnostics = {"method": "train_pairs", "source": "TRAIN", "pairs": count,
            "seed": seed, "sampling": "independent uniform pairs with replacement, including self-pairs",
            "status": "fitted" if denominator > 0 else "degenerate", "alpha": alpha,
            "fit_stress_before": stress(1.0), "fit_stress_after": stress(alpha),
            "stress_convention": "sqrt(sum squared distance errors / sum squared source distances)",
            "scope": "post-optimization scale only; histories/subset diagnostics are pre-calibration; no topology-shape improvement claimed"}
        if denominator == 0:
            diagnostics["reason"] = "all sampled target distances are zero; scale one retained"
        with np.errstate(over="ignore", invalid="ignore"):
            self.embedding_ *= alpha
        if not np.isfinite(self.embedding_).all():
            raise ValueError("output calibration produced nonfinite coordinates; rescale inputs")
        self.output_scale_ = alpha
        self.calibration_diagnostics_ = diagnostics

    def _optimize(self, topology, losses, rng):
        cfg, U = self.config, self.reference_
        is_fuzzy = cfg.geometry_objective in ("fuzzy", "fuzzy_graph")
        is_nce = cfg.geometry_objective == "neighbor_nce"
        positive_mode = "attraction" if cfg.geometry_objective == "fuzzy_graph" else "cross_entropy"
        if cfg.steps == 0:
            # A PCA-initialized model is useful as a reference/initialization
            # baseline. No graph, landmark bank, PH cache or optimizer is needed.
            self.embedding_ = self._forward_numpy(self.model_, U)
            self.training_coverage_ = {"n_training_samples":len(U), "optimizer_steps":0,
                "geometry_unique_samples":0, "h0_unique_samples":0, "h1_unique_samples":0,
                "h0_updates":0, "h1_updates":0, "h1_initial_subsets":0,
                "h0_online_refreshes":0, "h1_online_refreshes":0, "h0_fraction":0., "h1_fraction":0.,
                "definition":"initialization-only run; no training objective evaluated"}
            self.neighbor_diagnostics_ = {"skipped":True,"reason":"steps=0"}
            self.geometry_diagnostics_ = {"objective":cfg.geometry_objective,"trained":False}
            if is_fuzzy:
                self.geometry_diagnostics_["positive_mode"] = positive_mode
            self.timings_.update({"knn_and_landmarks_seconds":0., "source_cache_seconds":0.,
                "online_source_ph_seconds":0., "optimization_seconds":0.})
            return
        graph_start = time.perf_counter()
        from .neighbors import build_neighbor_graph
        graph = build_neighbor_graph(U, cfg.n_neighbors, backend=cfg.neighbor_backend,
            working_memory_mb=cfg.neighbor_working_memory_mb, seed=cfg.seed,
            audit_queries=cfg.neighbor_audit_queries, min_recall=cfg.neighbor_min_recall,
            timeout_seconds=cfg.neighbor_timeout_seconds)
        edges, neighbors = graph["edges"], graph["neighbors"]
        self.neighbor_diagnostics_ = graph["diagnostics"]
        if is_nce:
            from .contrastive import AnchoredNeighborSampler, neighbor_nce_loss
            sampler = AnchoredNeighborSampler(edges, len(U))
        else:
            sampler = GeometrySampler(edges, len(U))
        topo_sampler = TopologySampler(U, neighbors, cfg.landmark_size, cfg.seed + 17)
        h0_sampler = h1_sampler = topo_sampler
        if cfg.h0_refresh_every or cfg.h1_refresh_every:
            # Separate random streams make H0 subsets identical in a paired
            # H1/no-H1 ablation. Share immutable data and landmark arrays only.
            h0_sampler, h1_sampler = copy.copy(topo_sampler), copy.copy(topo_sampler)
            h0_sampler.rng = np.random.default_rng(cfg.seed + 31)
            h1_sampler.rng = np.random.default_rng(cfg.seed + 37)
        sample_edges = edges[rng.choice(len(edges), min(20000, len(edges)), replace=False)]
        edge_lengths = np.linalg.norm(U[sample_edges[:, 0]].astype(float) - U[sample_edges[:, 1]], axis=1)
        positives = edge_lengths[edge_lengths > 0]
        self.local_scale_ = float(np.median(positives)) if len(positives) else 0.01
        margin = float(np.quantile(positives, .75)) if len(positives) else self.local_scale_
        tau = max(self.local_scale_ * .1, 1e-8)
        fuzzy_weights = fuzzy_keys = None
        fuzzy_scale = cfg.fuzzy_scale if cfg.fuzzy_scale is not None else self.local_scale_
        if is_fuzzy or is_nce:
            from .fuzzy import build_fuzzy_weights, fuzzy_losses
            weights, diagnostics = build_fuzzy_weights(U, neighbors, edges)
            edge_keys = edges[:, 0] * len(U) + edges[:, 1]
            order = np.argsort(edge_keys)
            fuzzy_keys, fuzzy_weights = edge_keys[order], weights[order]
            positive_description = ("Weight-normalized graph attraction" if positive_mode == "attraction"
                                    else "Fuzzy edge cross-entropy")
            self.geometry_diagnostics_ = dict(diagnostics, objective=cfg.geometry_objective,
                positive_mode=positive_mode, kernel_scale=float(fuzzy_scale),
                negative_repulsion=cfg.fuzzy_repulsion,
                note=f"{positive_description} with sampled nonedge repulsion; not an exact ParametricUMAP reproduction. PH metric is unchanged.")
            if is_nce:
                self.geometry_diagnostics_.update(positive_mode="conditional_neighbor_nce",
                    candidates_per_anchor=cfg.contrastive_candidates,
                    hard_negatives=cfg.contrastive_hard_negatives,
                    temperature=cfg.contrastive_temperature,
                    negative_repulsion=None,
                    note="Conditional Cauchy neighbor loss with anchored sampled nonedges; optional detached hard mining changes the objective. Batch-normalized affinities. Not exact t-SNE/UMAP. fuzzy_repulsion and lambda_sep are unused; lambda_near scales the complete NCE term. PH metric unchanged.")
        else:
            self.geometry_diagnostics_ = {"objective":"stress", "huber_delta":self.local_scale_, "separation_margin":margin}
        graph_seconds = time.perf_counter() - graph_start
        cache_start = time.perf_counter()

        def make_item(strategy, subset, with_h1):
            D = topology.distance_matrix(U[subset])
            return {"ids": subset, "strategy": strategy,
                    "D": torch.tensor(D, dtype=torch.float32, device=self.device_),
                    "source": topology.h1_persistence(D, **cfg.budgets()) if with_h1 else topology.mst(D)}

        def build_bank(size, with_h1):
            # Online mode needs only one resident H1 target when refreshed every
            # active update. Do not precompute targets that will never be used.
            refresh = cfg.h1_refresh_every if with_h1 else cfg.h0_refresh_every
            count = 1 if refresh == 1 else cfg.subset_bank_size
            source_sampler = h1_sampler if with_h1 else h0_sampler
            return [make_item(strategy, subset, with_h1)
                    for strategy, subset in source_sampler.bank(count, size, cfg.topology_sampling)]

        h0_bank = build_bank(cfg.h0_size, False) if cfg.lambda_h0 else []
        h1_bank = build_bank(cfg.h1_size, True) if cfg.lambda_h1 else []
        cache_seconds = time.perf_counter() - cache_start
        self.model_.train()
        if cfg.optimizer_mode == "coordinates":
            initial = self._forward_numpy(self.model_, U)
            self._coordinates = nn.Parameter(torch.from_numpy(initial).to(self.device_))
            parameters = [self._coordinates]
        else:
            parameters = list(self.model_.parameters())
        if cfg.lambda_reconstruction:
            self.decoder_ = reconstruction_decoder(cfg.n_components, self.reference_dim_).to(self.device_)
            parameters += list(self.decoder_.parameters())
        optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate)
        U_cpu = torch.from_numpy(U)
        train_start = time.perf_counter()
        seen_geometry = np.zeros(len(U), dtype=bool)
        seen_positive_vertices = np.zeros(len(U), dtype=bool)
        geometry_edge_keys = np.sort(edges[:, 0] * len(U) + edges[:, 1])
        seen_positive_edges = np.zeros(len(edges), dtype=bool)
        positive_draws = 0
        seen_h0 = np.zeros(len(U), dtype=bool)
        seen_h1 = np.zeros(len(U), dtype=bool)
        h0_updates = h1_updates = online_refreshes = h0_online_refreshes = 0
        online_source_seconds = 0.0
        initial_h1_subsets = len(h1_bank)
        for step in range(cfg.steps):
            # Refresh one half of the finite H1 bank, without changing the frozen target metric.
            if not cfg.h1_refresh_every and step == (3 * cfg.steps) // 4 and cfg.steps >= 20 and h1_bank:
                renewed = build_bank(cfg.h1_size, True)
                h1_bank[len(h1_bank) // 2:] = renewed[len(h1_bank) // 2:]
            if is_nce:
                positive_pairs, negative_ids, negative_valid = sampler.sample(
                    cfg.batch_size, cfg.contrastive_candidates, rng)
                negative_pairs = np.column_stack((
                    np.repeat(positive_pairs[:, 0], cfg.contrastive_candidates), negative_ids.ravel()))
            else:
                positive_pairs, negative_pairs = sampler.sample(cfg.batch_size, rng)
            item0 = h0_bank[step % len(h0_bank)] if h0_bank else None
            active = step >= cfg.warmup_steps
            if h0_bank and active and cfg.h0_refresh_every:
                update0 = step - cfg.warmup_steps
                slot0 = update0 % len(h0_bank)
                if update0 > 0 and update0 % cfg.h0_refresh_every == 0:
                    strategy0 = ("local", "cover", "random")[update0 % 3] if cfg.topology_sampling == "mixed" else cfg.topology_sampling
                    refresh_start = time.perf_counter()
                    h0_bank[slot0] = make_item(strategy0, h0_sampler.sample(cfg.h0_size, strategy0), False)
                    online_source_seconds += time.perf_counter() - refresh_start
                    h0_online_refreshes += 1
                item0 = h0_bank[slot0]
            h1_step = active and ((step - cfg.warmup_steps) % cfg.topology_interval == 0)
            item1 = None
            if h1_bank and h1_step:
                update_index = (step - cfg.warmup_steps) // cfg.topology_interval
                slot = update_index % len(h1_bank)
                if cfg.h1_refresh_every and update_index > 0 and update_index % cfg.h1_refresh_every == 0:
                    strategy = ("local", "cover", "random")[update_index % 3] if cfg.topology_sampling == "mixed" else cfg.topology_sampling
                    refresh_start = time.perf_counter()
                    h1_bank[slot] = make_item(strategy, h1_sampler.sample(cfg.h1_size, strategy), True)
                    online_source_seconds += time.perf_counter() - refresh_start
                    online_refreshes += 1
                item1 = h1_bank[slot]
            seen_geometry[positive_pairs.ravel()] = True
            seen_positive_vertices[positive_pairs.ravel()] = True
            canonical_positive = np.sort(positive_pairs, axis=1)
            sampled_keys = canonical_positive[:, 0] * len(U) + canonical_positive[:, 1]
            seen_positive_edges[np.searchsorted(geometry_edge_keys, sampled_keys)] = True
            positive_draws += len(positive_pairs)
            seen_geometry[(negative_pairs[negative_valid.ravel()] if is_nce else negative_pairs).ravel()] = True
            if item0 is not None and active:
                seen_h0[item0["ids"]] = True
                h0_updates += 1
            if item1 is not None:
                seen_h1[item1["ids"]] = True
                h1_updates += 1
            required = [positive_pairs.ravel(), negative_pairs.ravel()]
            required += [item["ids"] for item in (item0, item1) if item is not None]
            union = np.unique(np.concatenate(required))
            if len(union) == 0:
                continue
            x = U_cpu[union].to(self.device_)
            z = self._coordinates[torch.as_tensor(union, device=self.device_)] if self._coordinates is not None else self.model_(x)
            zero = z.sum() * 0

            def lengths(pairs):
                local = torch.as_tensor(np.searchsorted(union, pairs), device=self.device_)
                return (torch.linalg.vector_norm(x[local[:, 0]] - x[local[:, 1]], dim=1),
                        torch.linalg.vector_norm(z[local[:, 0]] - z[local[:, 1]], dim=1))

            positive_source, positive_target = lengths(positive_pairs)
            negative_source, negative_target = lengths(negative_pairs)
            if is_fuzzy or is_nce:
                canonical = np.sort(positive_pairs, axis=1) if is_nce else positive_pairs
                keys = canonical[:, 0] * len(U) + canonical[:, 1]
                weights = torch.as_tensor(fuzzy_weights[np.searchsorted(fuzzy_keys, keys)], device=self.device_, dtype=z.dtype)
                # A missing graph edge between duplicate inputs is not a true
                # negative: a parametric encoder must map identical inputs alike.
                if is_nce:
                    mask = torch.as_tensor(negative_valid, device=self.device_) & (negative_source.reshape(negative_ids.shape) > 0)
                    near = neighbor_nce_loss(positive_target, negative_target.reshape(negative_ids.shape),
                        weights, mask, scale=fuzzy_scale, temperature=cfg.contrastive_temperature,
                        hard_negatives=cfg.contrastive_hard_negatives or None)
                    separation = zero  # NCE includes its conditional repulsion.
                else:
                    near, separation = fuzzy_losses(positive_target, negative_target[negative_source > 0],
                        weights, scale=fuzzy_scale, positive_mode=positive_mode)
            else:
                near = losses.near_loss(positive_source, positive_target, delta=self.local_scale_) if len(positive_source) else zero
                separation = losses.separation_loss(negative_source, negative_target, margin=margin) if len(negative_source) else zero
            h0, pd1, crit1, rec = zero, zero, zero, zero
            if item0 is not None and active:
                local = torch.as_tensor(np.searchsorted(union, item0["ids"]), device=self.device_)
                h0 = losses.h0_loss(item0["D"], losses.pairwise_distances(z[local]), source_edges=item0["source"])
            source_bars = target_bars = 0
            if item1 is not None:
                local = torch.as_tensor(np.searchsorted(union, item1["ids"]), device=self.device_)
                result = losses.h1_loss(item1["D"], losses.pairwise_distances(z[local]),
                    source_result=item1["source"], tau=tau, budgets=cfg.budgets(),
                    max_matching_size=cfg.max_matching_size)
                pd1, crit1 = result["pd1"], result["crit1"]
                source_bars, target_bars = result["source_bars"], result["target_bars"]
            if self.decoder_ is not None:
                rec = torch.mean((self.decoder_(z) - x) ** 2)
            ramp = min(1.0, max(0.0, (step - cfg.warmup_steps + 1) / max(1, cfg.warmup_steps)))
            separation_weight = cfg.fuzzy_repulsion if is_fuzzy else cfg.lambda_sep
            parts = {"near": cfg.lambda_near * near, "separation": separation_weight * separation,
                     "h0": ramp * cfg.lambda_h0 * h0,
                     "h1": ramp * cfg.lambda_h1 * cfg.topology_interval * (pd1 + cfg.lambda_critical * crit1),
                     "reconstruction": cfg.lambda_reconstruction * rec}
            total = sum(parts.values())
            if not bool(torch.isfinite(total)):
                raise RuntimeError(f"nonfinite loss at step {step}")
            log = step % cfg.log_interval == 0 or step == cfg.steps - 1 or item1 is not None
            component_gradients = {}
            if log:
                for name, term in parts.items():
                    grad = torch.autograd.grad(term, z, retain_graph=True, allow_unused=True)[0]
                    component_gradients[name] = float(torch.linalg.vector_norm(grad).detach()) if grad is not None else 0.0
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = nn.utils.clip_grad_norm_(parameters, cfg.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            if log:
                self.history_.append({"step": step, "output_scale_scope": "pre-calibration", "loss": float(total.detach()),
                    "near": float(near.detach()), "separation": float(separation.detach()),
                    "h0": float(h0.detach()), "pd1": float(pd1.detach()), "crit1": float(crit1.detach()),
                    "reconstruction": float(rec.detach()), "gradient_norm": float(grad_norm),
                    "component_gradient_norms": component_gradients, "topology_ramp": ramp,
                    "h1_evaluated": item1 is not None, "source_bars": int(source_bars),
                    "target_bars": int(target_bars),
                    "h1_strategy": item1["strategy"] if item1 else None})
            if (step + 1) % cfg.validation_interval == 0:
                # Held-out subset of training IDs (not held-out samples), fixed across checkpoints.
                self.validation_history_.append(self._subset_diagnostic(step, topology))
        self.embedding_ = (self._coordinates.detach().cpu().numpy().copy() if self._coordinates is not None
                           else self._forward_numpy(self.model_, U))
        if not np.isfinite(self.embedding_).all():
            raise RuntimeError("training generated nonfinite coordinates")
        self.training_coverage_ = {
            "n_training_samples": len(U), "optimizer_steps": cfg.steps,
            "geometry_unique_samples": int(seen_geometry.sum()),
            "positive_unique_samples": int(seen_positive_vertices.sum()),
            "positive_pair_draws": int(positive_draws),
            "positive_unique_edges": int(seen_positive_edges.sum()),
            "graph_edge_count": len(edges),
            "positive_edge_fraction": float(seen_positive_edges.mean()) if len(edges) else 0.,
            "h0_unique_samples": int(seen_h0.sum()), "h1_unique_samples": int(seen_h1.sum()),
            "h0_updates": h0_updates, "h1_updates": h1_updates,
            "h1_initial_subsets": initial_h1_subsets, "h1_online_refreshes": online_refreshes,
            "h0_online_refreshes": h0_online_refreshes, "h0_fraction": float(seen_h0.mean()),
            "h1_fraction": float(seen_h1.mean()),
            "definition": "unique sample IDs present in active loss subsets; not proof of nonzero per-sample gradients; steps are not epochs"}
        self.timings_.update({"knn_and_landmarks_seconds": graph_seconds, "source_cache_seconds": cache_seconds,
                              "online_source_ph_seconds": online_source_seconds,
                              "optimization_seconds": time.perf_counter() - train_start})

    def _subset_diagnostic(self, step, topology):
        from .evaluation import evaluate_embedding
        cfg = self.config
        rng = np.random.default_rng(cfg.seed + 1009)
        ids = np.sort(rng.choice(len(self.reference_), min(cfg.evaluation_size, len(self.reference_)), replace=False))
        U = self.reference_[ids]
        Z = (self._coordinates.detach().cpu().numpy()[ids] if self._coordinates is not None
             else self._forward_numpy(self.model_, U))
        metrics = evaluate_embedding(U, Z, topology_size=len(ids), seed=cfg.seed + 1013,
                                     k=cfg.n_neighbors, budgets=cfg.evaluation_budgets())
        return {"step": step, "scope": "fixed training subset, not out-of-sample; pre-calibration", "metrics": metrics}

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("DeepTDA is not fitted")

    def reference_transform(self, X):
        self._require_fitted()
        processed = self.preprocessor_.transform(X)
        if len(processed) == 0:
            return np.empty((0, self.reference_dim_), dtype=np.float32)
        H = (self._forward_numpy(self.reference_encoder_, processed)
             if self.reference_encoder_ is not None else processed)
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(H / self.reference_scale_, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("reference scaling produced nonfinite values; rescale inputs")
        return result

    def transform(self, X):
        self._require_fitted()
        if self.config.optimizer_mode == "coordinates":
            raise NotImplementedError("direct coordinate mode has no out-of-sample transform; use embedding_")
        U = self.reference_transform(X)
        if len(U) == 0:
            return np.empty((0, self.config.n_components), dtype=np.float32)
        result = self._forward_numpy(self.model_, U)
        scale = getattr(self, "output_scale_", 1.0)
        if scale == 1.0:
            return result
        with np.errstate(over="ignore", invalid="ignore"):
            result = result * scale
        if not np.isfinite(result).all():
            raise ValueError("output calibration produced nonfinite transformed coordinates; rescale inputs")
        return result

    def fit_transform(self, X, validation_data=None):
        return self.fit(X, validation_data=validation_data).embedding_.copy()

    def ood_scores(self, X):
        """Nearest training reference distance, not a calibrated anomaly probability."""
        if self.reference_ is None:
            raise RuntimeError("training references were omitted from this inference-only checkpoint")
        U = self.reference_transform(X)
        if len(U) == 0:
            return np.empty(0)
        return NearestNeighbors(n_neighbors=1, n_jobs=1).fit(self.reference_).kneighbors(U)[0][:, 0]

    def save(self, path, include_training_data=True):
        """Save a trusted tensor checkpoint.

        include_training_data=False omits training rows, coordinates and detailed
        diagnostics/history. It preserves transform, NOT OOD or training plots.
        Learned weights/preprocessing statistics remain data-derived: this is
        smaller storage and less exposure, not differential privacy/anonymity.
        """
        self._require_fitted()
        if not isinstance(include_training_data, bool):
            raise ValueError("include_training_data must be boolean")
        if not include_training_data and self.config.optimizer_mode == "coordinates":
            raise ValueError("direct-coordinate models require their training coordinates")
        if include_training_data and (self.reference_ is None or self.embedding_ is None):
            raise ValueError("training data unavailable; save with include_training_data=False")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"format_version": 1, "config": self.config.to_dict(),
            "preprocessor": self.preprocessor_.state(), "reference_scale": self.reference_scale_,
            "output_scale": getattr(self, "output_scale_", 1.0),
            "output_calibration": copy.deepcopy(getattr(self, "calibration_diagnostics_",
                _skipped_output_calibration(reason="legacy checkpoint"))),
            "processed_dim": self.processed_dim_, "reference_dim": self.reference_dim_,
            "constant": self.constant_, "input_rank": self.input_rank_,
            "model": _tensor_state(self.model_),
            "reference_encoder": _tensor_state(self.reference_encoder_) if self.reference_encoder_ is not None else None,
            "decoder": _tensor_state(self.decoder_) if include_training_data and self.decoder_ is not None else None,
            "training_data_included": include_training_data,
            "reference": torch.from_numpy(self.reference_.copy()) if include_training_data else None,
            "embedding": torch.from_numpy(self.embedding_.copy()) if include_training_data else None,
            "history": copy.deepcopy(self.history_) if include_training_data else [],
            "validation_history": copy.deepcopy(self.validation_history_) if include_training_data else [],
            "semantic_history": copy.deepcopy(self.semantic_history_) if include_training_data else [],
            "report": copy.deepcopy(self.report_) if include_training_data else {
                "status":"inference_only", "config":self.config.to_dict(),
                "output_calibration": copy.deepcopy(getattr(self, "calibration_diagnostics_",
                    _skipped_output_calibration(reason="legacy checkpoint"))),
                "scope":"training rows, coordinates, histories and detailed reports omitted; weights/statistics retained"},
            "timings": getattr(self, "timings_", {}), "fit_seconds": getattr(self, "fit_seconds_", 0.0)}
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(state, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        obj = cls(state["config"], device=device)
        obj.device_ = obj._device()
        obj.preprocessor_ = NumericPreprocessor.from_state(state["preprocessor"])
        obj.reference_scale_ = float(state["reference_scale"])
        obj.output_scale_ = float(state.get("output_scale", 1.0))
        if not np.isfinite(obj.output_scale_) or obj.output_scale_ <= 0:
            raise ValueError("checkpoint output_scale must be finite and positive")
        obj.calibration_diagnostics_ = copy.deepcopy(state.get("output_calibration",
            _skipped_output_calibration(reason="legacy checkpoint")))
        obj.processed_dim_, obj.reference_dim_ = state["processed_dim"], state["reference_dim"]
        obj.constant_, obj.input_rank_ = state["constant"], state["input_rank"]
        # Construction must not alter the caller's global random stream.
        with _torch_context(obj.config.seed, obj.config.num_threads, device):
            obj.model_ = ParametricEmbedding(obj.reference_dim_, obj.config.n_components,
                                             residual_input_scale=obj.config.residual_input_scale).to(obj.device_)
            obj.model_.load_state_dict(state["model"])
            obj.model_.eval()
            obj.reference_encoder_ = None
            if state["reference_encoder"] is not None:
                obj.reference_encoder_ = NumericAutoencoder(obj.processed_dim_, obj.config.semantic_dim).encoder.to(obj.device_)
                obj.reference_encoder_.load_state_dict(state["reference_encoder"])
                obj.reference_encoder_.eval()
                for p in obj.reference_encoder_.parameters():
                    p.requires_grad_(False)
            obj.decoder_ = None
            if state["decoder"] is not None:
                obj.decoder_ = reconstruction_decoder(obj.config.n_components, obj.reference_dim_).to(obj.device_)
                obj.decoder_.load_state_dict(state["decoder"])
        obj.reference_ = state["reference"].numpy().copy() if state["reference"] is not None else None
        obj.embedding_ = state["embedding"].numpy().copy() if state["embedding"] is not None else None
        obj.history_, obj.validation_history_ = state["history"], state["validation_history"]
        obj.semantic_history_, obj.report_ = state["semantic_history"], state["report"]
        obj.report_["output_calibration"] = copy.deepcopy(obj.calibration_diagnostics_)
        obj.timings_, obj.fit_seconds_ = state["timings"], state["fit_seconds"]
        obj.training_coverage_ = obj.report_.get("training_coverage", {})
        obj.neighbor_diagnostics_ = obj.report_.get("neighbor_graph", {})
        obj.geometry_diagnostics_ = obj.report_.get("geometry_objective", {})
        obj._coordinates = None
        obj._fitted = True
        return obj
