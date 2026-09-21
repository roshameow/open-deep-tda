"""Train-only numeric preprocessing with stable missing-value schema."""
import numpy as np


def numeric_matrix(X, allow_empty=False):
    try:
        if np.iscomplexobj(X):
            raise ValueError("complex-valued inputs are not supported")
        a = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("expected a numeric 2D array") from exc
    if a.ndim != 2 or a.shape[1] == 0 or (not allow_empty and a.shape[0] == 0):
        raise ValueError("expected nonempty numeric matrix of shape (samples, features)")
    if np.isinf(a).any():
        raise ValueError("infinite inputs are not supported; NaN may represent missing values")
    return a


class NumericPreprocessor:
    def __init__(self, standardize=True, missing_indicators=True):
        self.standardize = standardize
        self.missing_indicators = missing_indicators

    def fit(self, X):
        a = numeric_matrix(X)
        self.n_features = a.shape[1]
        self.medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any()
                                 else 0.0 for c in a.T])
        filled = np.where(np.isnan(a), self.medians, a)
        self.mean = filled.mean(axis=0) if self.standardize else np.zeros(self.n_features)
        self.scale = filled.std(axis=0) if self.standardize else np.ones(self.n_features)
        self.scale[self.scale < 1e-12] = 1.0
        self.all_missing = np.all(np.isnan(a), axis=0)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("input magnitude overflows preprocessing; rescale your data")
        return self

    def transform(self, X):
        if not hasattr(self, "n_features"):
            raise RuntimeError("preprocessor is not fitted")
        a = numeric_matrix(X, allow_empty=True)
        if a.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, received {a.shape[1]}")
        missing = np.isnan(a)
        result = (np.where(missing, self.medians, a) - self.mean) / self.scale
        if self.missing_indicators:
            result = np.concatenate([result, missing.astype(float)], axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            result = result.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("preprocessed data exceeds finite float32 range")
        return result

    def state(self):
        return {"standardize": self.standardize, "missing_indicators": self.missing_indicators,
                "n_features": self.n_features, "medians": self.medians.tolist(),
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "all_missing": self.all_missing.tolist()}

    @classmethod
    def from_state(cls, state):
        obj = cls(state["standardize"], state["missing_indicators"])
        obj.n_features = int(state["n_features"])
        for key in ("medians", "mean", "scale", "all_missing"):
            setattr(obj, key, np.asarray(state[key]))
        return obj
