"""Serializable, validated training configuration."""
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Optional


@dataclass
class TDAConfig:
    n_components: int = 2
    mode: str = "geometry"
    optimizer_mode: str = "parametric"
    seed: int = 0
    device: str = "cpu"
    standardize: bool = True
    missing_indicators: bool = True
    steps: int = 200
    warmup_steps: int = 20
    learning_rate: float = 0.001
    batch_size: int = 256
    inference_batch_size: int = 4096
    n_neighbors: int = 15
    neighbor_backend: str = "exact"
    neighbor_working_memory_mb: int = 64
    neighbor_audit_queries: int = 64
    neighbor_min_recall: float = 0.9
    neighbor_timeout_seconds: int = 300
    h0_size: int = 128
    h1_size: int = 64
    topology_interval: int = 5
    h0_refresh_every: int = 0
    h1_refresh_every: int = 0
    subset_bank_size: int = 12
    landmark_size: int = 256
    topology_sampling: str = "mixed"
    geometry_objective: str = "stress"
    fuzzy_repulsion: float = 0.1
    fuzzy_scale: Optional[float] = None
    lambda_near: float = 1.0
    lambda_sep: float = 0.1
    lambda_h0: float = 1.0
    lambda_h1: float = 0.1
    lambda_critical: float = 0.1
    lambda_reconstruction: float = 0.0
    gradient_clip: float = 10.0
    log_interval: int = 10
    validation_interval: int = 50
    evaluation_size: int = 64
    max_simplices: int = 1000000
    max_reduction_entries: int = 10000000
    max_reduction_operations: int = 100000000
    max_matching_size: int = 512
    semantic_dim: int = 32
    semantic_steps: int = 100
    mask_probability: float = 0.2
    num_threads: int = 1

    def validate(self):
        if self.n_components not in (2, 3):
            raise ValueError("n_components must be 2 or 3")
        if self.mode not in ("geometry", "semantic"):
            raise ValueError("mode must be geometry or semantic")
        if self.optimizer_mode not in ("parametric", "coordinates"):
            raise ValueError("optimizer_mode must be parametric or coordinates")
        if self.topology_sampling not in ("mixed", "local", "cover", "random"):
            raise ValueError("invalid topology_sampling")
        if self.geometry_objective not in ("stress", "fuzzy"):
            raise ValueError("geometry_objective must be stress or fuzzy")
        if self.neighbor_backend not in ("exact", "pynndescent"):
            raise ValueError("neighbor_backend must be exact or pynndescent")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda or mps")
        nonnegative_ints = ("steps", "warmup_steps", "seed", "semantic_steps", "h0_refresh_every", "h1_refresh_every")
        positive_ints = ("n_components", "batch_size", "inference_batch_size", "n_neighbors", "h0_size", "h1_size",
                        "topology_interval", "subset_bank_size", "landmark_size",
                        "log_interval", "validation_interval", "evaluation_size",
                        "max_simplices", "max_reduction_entries", "max_reduction_operations",
                        "max_matching_size", "semantic_dim", "num_threads",
                        "neighbor_working_memory_mb", "neighbor_audit_queries", "neighbor_timeout_seconds")
        for name in nonnegative_ints + positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name in nonnegative_ints else 1):
                raise ValueError(f"invalid {name}: {value}")
            setattr(self, name, int(value))
        for name in ("learning_rate", "gradient_clip", "lambda_near", "lambda_sep", "lambda_h0",
                     "lambda_h1", "lambda_critical", "lambda_reconstruction", "fuzzy_repulsion"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, float(value))
        if isinstance(self.neighbor_min_recall, bool) or not isinstance(self.neighbor_min_recall, Real) or not math.isfinite(self.neighbor_min_recall) or not 0 <= self.neighbor_min_recall <= 1:
            raise ValueError("neighbor_min_recall must be in [0,1]")
        self.neighbor_min_recall = float(self.neighbor_min_recall)
        if self.fuzzy_scale is not None:
            if isinstance(self.fuzzy_scale, bool) or not isinstance(self.fuzzy_scale, Real) or not math.isfinite(self.fuzzy_scale) or self.fuzzy_scale <= 0:
                raise ValueError("fuzzy_scale must be None or finite positive")
            self.fuzzy_scale = float(self.fuzzy_scale)
        if self.seed >= 2**32:
            raise ValueError("seed must be in [0, 2**32-1] for all supported baselines")
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if isinstance(self.mask_probability, bool) or not isinstance(self.mask_probability, Real) or not 0 < self.mask_probability < 1:
            raise ValueError("mask_probability must be in (0,1)")
        self.mask_probability = float(self.mask_probability)
        for name in ("standardize", "missing_indicators"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        return self

    def to_dict(self):
        return asdict(self)

    def evaluation_budgets(self):
        return dict(self.budgets(), max_matching_size=self.max_matching_size)

    def budgets(self):
        return {name: getattr(self, name) for name in
                ("max_simplices", "max_reduction_entries", "max_reduction_operations")}
