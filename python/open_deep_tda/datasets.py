"""Local synthetic point clouds; labels are diagnostics, not training inputs.

The canonical coordinate dimension is 2 (circle, figure_eight, two_circles,
blobs) or 3 (swiss_roll, sphere, torus). A random orthonormal linear map
preserves all canonical distances before optional ambient Gaussian noise.
"""

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def make_dataset(name='circle', n_samples=256, n_features=8, noise=0.01, seed=0):
    """Return ``(X, integer_labels)`` without downloads or dimensional flattening.

    ``n_features`` must accommodate the canonical coordinates, even when the
    manifold's topological dimension is lower (a circle requires two axes).
    Labels partition angles/heights, or identify disconnected circles/blobs.
    """
    n_samples = _integer(n_samples, 'n_samples')
    n_features = _integer(n_features, 'n_features', 1)
    dimensions = dict(circle=2, figure_eight=2, two_circles=2,
                      swiss_roll=3, sphere=3, torus=3, blobs=2)
    if name not in dimensions:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(dimensions)}")
    dim = dimensions[name]
    if n_features < dim:
        raise ValueError(f"{name} requires n_features >= {dim}; cannot flatten canonical coordinates")
    if not np.isscalar(noise) or not np.isfinite(noise) or noise < 0:
        raise ValueError('noise must be finite and nonnegative')
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n_samples)
    labels = np.floor(t / (np.pi / 2)).astype(np.int64)
    if name == 'circle':
        points = np.column_stack((np.cos(t), np.sin(t)))
    elif name == 'figure_eight':
        points = np.column_stack((np.sin(t), np.sin(2 * t)))
        labels = (np.sin(t) >= 0).astype(np.int64)
    elif name == 'two_circles':
        labels = rng.integers(0, 2, n_samples)
        points = np.column_stack((np.cos(t) + 4 * labels - 2, np.sin(t)))
    elif name == 'swiss_roll':
        angle = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n_samples)
        points = np.column_stack((angle * np.cos(angle), rng.uniform(0, 10, n_samples),
                                  angle * np.sin(angle)))
        labels = np.minimum(3, ((angle - 1.5 * np.pi) / (3 * np.pi) * 4).astype(np.int64))
    elif name == 'sphere':
        height = rng.uniform(-1, 1, n_samples)
        radius = np.sqrt(np.maximum(0, 1 - height ** 2))
        points = np.column_stack((radius * np.cos(t), radius * np.sin(t), height))
        labels = (height >= 0).astype(np.int64)
    elif name == 'torus':
        v = rng.uniform(0, 2 * np.pi, n_samples)
        radius = 2 + 0.6 * np.cos(v)
        points = np.column_stack((radius * np.cos(t), radius * np.sin(t), 0.6 * np.sin(v)))
    else:
        labels = rng.integers(0, 3, n_samples)
        centers = np.array([[-2., -1.], [2., -1.], [0., 2.]])
        points = centers[labels] + rng.normal(0, 0.25, (n_samples, 2))
    basis, _ = np.linalg.qr(rng.normal(size=(n_features, dim)), mode='reduced')
    X = points @ basis.T
    if noise:
        X += rng.normal(0, noise, X.shape)
    return np.asarray(X, dtype=np.float64), labels
