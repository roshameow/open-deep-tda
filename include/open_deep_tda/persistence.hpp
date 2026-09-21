#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace open_deep_tda {
using Edge = std::array<int, 2>;

struct PersistenceOptions {
    std::size_t max_simplices = 1000000;
    std::size_t max_reduction_entries = 10000000;
    std::size_t max_reduction_operations = 100000000;
    double max_radius = std::numeric_limits<double>::infinity();
};

struct PersistencePair {
    int dimension = 0;
    double birth = 0;
    double death = std::numeric_limits<double>::infinity();
    std::vector<int> birth_simplex;
    std::vector<int> death_simplex;
    Edge birth_edge{{-1, -1}};
    Edge death_edge{{-1, -1}};
    bool essential = false;
    bool censored = false;
};

struct PersistenceResult {
    std::vector<PersistencePair> pairs;
    std::size_t n_vertices = 0;
    std::size_t n_simplices = 0;
    std::size_t reduction_operations = 0;
    std::size_t peak_reduction_entries = 0;
    bool truncated = false;
};

// Row-major n*n distances; duplicate points and non-metric dissimilarities are
// accepted. Finite, symmetric, nonnegative, zero-diagonal input is required.
// Roundoff policy: absolute tolerance 1e-10 for zero/nonnegativity and
// 1e-10 + 1e-8*max(|a|,|b|) for symmetry. Accepted entries are averaged,
// tiny negatives clamped, and the diagonal set to zero, in BOTH algorithms.
// Invalid input throws std::invalid_argument; exhausted budgets throw
// std::runtime_error. All simplex ordering comparisons are exact (not fuzzy).
std::vector<Edge> mst(const std::vector<double>& distances, std::size_t n);
PersistenceResult persistence(const std::vector<double>& distances, std::size_t n,
                              const PersistenceOptions& options = {});

// Resource semantics: simplex count is checked before allocating simplices.
// Reduction work counts boundary entries, pivot probes, merge inputs, and
// merge output writes. Live entries include stored reduced columns, the
// current column, AND the simultaneously constructed XOR output; vector
// capacity, matrix storage, simplex metadata, and sorting are not entries.
// Truncated means at least one edge is excluded by max_radius. Unpaired bars
// in that case are censored, not asserted essential. A fully included complex
// has its single essential H0 bar (unless empty) and no essential H1 bars.
}  // namespace open_deep_tda
