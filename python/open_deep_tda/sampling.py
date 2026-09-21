"""Bounded subset sampling; these subsets do not prove global topology preservation."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_graph(X, k):
    n = len(X)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty((n, 0), dtype=np.int64)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    neighbors = np.array([row[row != i][:k] for i, row in enumerate(indices)], dtype=np.int64)
    edges = np.column_stack((np.repeat(np.arange(n), k), neighbors.ravel()))
    edges.sort(axis=1)
    return np.unique(edges, axis=0), neighbors


def farthest_points(X, count, rng):
    n = len(X)
    count = min(count, n)
    if count == n:
        return np.arange(n, dtype=np.int64)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = rng.integers(n)
    minimum = np.full(n, np.inf)
    for i in range(count):
        delta = X - X[chosen[i]]
        minimum = np.minimum(minimum, np.einsum("ij,ij->i", delta, delta))
        minimum[chosen[:i + 1]] = -1.0
        if i + 1 < count:
            chosen[i + 1] = int(np.argmax(minimum))
    return chosen


class TopologySampler:
    def __init__(self, reference, neighbors, landmark_size=256, seed=0):
        self.reference = reference
        self.neighbors = neighbors
        self.rng = np.random.default_rng(seed)
        self.landmarks = farthest_points(reference, landmark_size, self.rng)

    def sample(self, size, strategy="random"):
        n = len(self.reference)
        size = min(size, n)
        if size == n:
            return np.arange(n, dtype=np.int64)
        if strategy == "random":
            selected = self.rng.choice(n, size, replace=False)
        elif strategy == "cover":
            if size > len(self.landmarks):
                selected = farthest_points(self.reference, size, self.rng)
            else:
                ids = farthest_points(self.reference[self.landmarks], size, self.rng)
                selected = self.landmarks[ids]
        elif strategy == "local":
            seed = int(self.rng.integers(n))
            queue, seen = [seed], {seed}
            position = 0
            while position < len(queue) and len(queue) < size:
                for v in self.neighbors[queue[position]]:
                    v = int(v)
                    if v not in seen:
                        queue.append(v)
                        seen.add(v)
                        if len(queue) == size:
                            break
                position += 1
            if len(queue) < size:
                remaining = np.setdiff1d(np.arange(n), np.array(queue), assume_unique=False)
                queue.extend(self.rng.choice(remaining, size - len(queue), replace=False).tolist())
            selected = np.array(queue)
        else:
            raise ValueError("unknown subset strategy")
        return np.sort(selected).astype(np.int64)

    def bank(self, count, size, strategy="mixed"):
        strategies = ("local", "cover", "random") if strategy == "mixed" else (strategy,)
        return [(strategies[i % len(strategies)], self.sample(size, strategies[i % len(strategies)]))
                for i in range(count)]


class GeometrySampler:
    def __init__(self, edges, n):
        self.edges = edges
        self.n = n
        self.keys = np.sort(edges[:, 0] * n + edges[:, 1])

    def sample(self, size, rng):
        if len(self.edges):
            positive = self.edges[rng.integers(len(self.edges), size=min(size, len(self.edges)))]
        else:
            positive = np.empty((0, 2), dtype=np.int64)
        if len(self.edges) >= self.n * (self.n - 1) // 2:
            return positive, np.empty((0, 2), dtype=np.int64)
        chunks, found = [], 0
        for _ in range(10):
            pairs = rng.integers(self.n, size=(max(16, size * 2), 2))
            pairs.sort(axis=1)
            keys = pairs[:, 0] * self.n + pairs[:, 1]
            pos = np.searchsorted(self.keys, keys)
            is_edge = np.zeros(len(keys), dtype=bool)
            valid = pos < len(self.keys)
            is_edge[valid] = self.keys[pos[valid]] == keys[valid]
            accepted = pairs[(pairs[:, 0] != pairs[:, 1]) & ~is_edge]
            chunks.append(accepted)
            found += len(accepted)
            if found >= size:
                break
        negative = np.concatenate(chunks, axis=0)[:size] if chunks else np.empty((0, 2), dtype=np.int64)
        return positive, negative
