"""Small neural models. PCA skip initialization is exact, not a fitted approximation."""
import numpy as np
import torch
from torch import nn


class ParametricEmbedding(nn.Module):
    """PCA skip plus a residual with opt-in input scaling.

    High-dimensional, distance-normalized references may leave SiLU near-linear.
    Scaling residual inputs is a hypothesis pending a TRAIN-only experiment,
    not a demonstrated improvement; it never rescales the PCA skip or metric.
    """
    def __init__(self, input_dim, output_dim, residual_input_scale=1.0):
        super().__init__()
        # Configuration scalar, not a state buffer: legacy tensor states still load.
        self.residual_input_scale = float(residual_input_scale)
        self.linear = nn.Linear(input_dim, output_dim)
        self.residual = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(),
                                      nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, output_dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        return self.linear(x) + self.residual(x * self.residual_input_scale)

    def initialize_pca(self, X):
        mean = X.astype(np.float64).mean(axis=0)
        centered = X.astype(np.float64) - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        components = np.zeros((self.linear.out_features, X.shape[1]), dtype=np.float32)
        count = min(len(vt), len(components))
        components[:count] = vt[:count]
        # Fix SVD signs for deterministic coordinates, including on different LAPACK builds.
        for row in components:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(components))
            self.linear.bias.copy_(torch.from_numpy((-components @ mean).astype(np.float32)))
        tol = max(centered.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0)
        return int(np.count_nonzero(singular > tol))


class NumericAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(),
                                     nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(),
                                     nn.Linear(128, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_decoder(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.SiLU(), nn.Linear(64, output_dim))
