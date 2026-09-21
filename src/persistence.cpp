#include "open_deep_tda/persistence.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace open_deep_tda {
namespace {
constexpr std::size_t absent = std::numeric_limits<std::size_t>::max();
constexpr double absolute_tolerance = 1e-10;

std::vector<double> validate(const std::vector<double>& input, std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        (n && n > std::numeric_limits<std::size_t>::max() / n) || input.size() != n * n)
        throw std::invalid_argument("distances must be a square matrix with representable vertex IDs");
    for (double value : input)
        if (!std::isfinite(value)) throw std::invalid_argument("distances must be finite");
    std::vector<double> d(input.size(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(input[i * n + i]) > absolute_tolerance)
            throw std::invalid_argument("distances must have a zero diagonal");
        for (std::size_t j = i + 1; j < n; ++j) {
            const double a = input[i * n + j], b = input[j * n + i];
            if (a < -absolute_tolerance || b < -absolute_tolerance)
                throw std::invalid_argument("distances must be nonnegative");
            if (std::abs(a - b) > absolute_tolerance + 1e-8 * std::max(std::abs(a), std::abs(b)))
                throw std::invalid_argument("distances must be symmetric");
            // Avoid overflow for finite distances near DBL_MAX.
            const double v = std::max(0.0, a * 0.5 + b * 0.5);
            d[i * n + j] = d[j * n + i] = v;
        }
    }
    return d;
}

Edge edge(int a, int b) { return {{std::min(a, b), std::max(a, b)}}; }
std::uint64_t edge_key(int a, int b) {
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint32_t>(b);
}

struct Simplex {
    double value;
    int dimension;
    std::array<int, 3> vertices;
    Edge critical;
};

// Add a binomial coefficient without overflowing even if the budget is SIZE_MAX.
void add_choose(std::size_t n, unsigned k, std::size_t limit, std::size_t& count) {
    if (n < k) return;
    std::array<std::size_t, 3> factors{{n, n - 1, n >= 2 ? n - 2 : 0}};
    std::size_t divisor = k == 2 ? 2 : 6;
    for (unsigned i = 0; i < k; ++i) {
        const auto common = std::gcd(factors[i], divisor);
        factors[i] /= common;
        divisor /= common;
    }
    std::size_t product = 1;
    for (unsigned i = 0; i < k; ++i) {
        if (factors[i] && product > (limit - count) / factors[i])
            throw std::runtime_error("max_simplices budget exceeded during preflight");
        product *= factors[i];
    }
    count += product;
}

struct Accounting {
    const PersistenceOptions& options;
    std::size_t work = 0, live = 0, peak = 0;
    void operations(std::size_t n) {
        if (n > options.max_reduction_operations - work)
            throw std::runtime_error("max_reduction_operations budget exceeded");
        work += n;
    }
    void acquire(std::size_t n) {
        if (n > options.max_reduction_entries - live)
            throw std::runtime_error("max_reduction_entries budget exceeded");
        live += n;
        peak = std::max(peak, live);
    }
    void release(std::size_t n) { live -= n; }
};

std::vector<int> vertices(const Simplex& s) {
    return {s.vertices.begin(), s.vertices.begin() + s.dimension + 1};
}
}  // namespace

std::vector<Edge> mst(const std::vector<double>& distances, std::size_t n) {
    const auto d = validate(distances, n);
    std::vector<Edge> result;
    if (n < 2) return result;
    result.reserve(n - 1);
    std::vector<bool> included(n, false);
    std::vector<double> best(n, std::numeric_limits<double>::infinity());
    std::vector<int> parent(n, -1);
    included[0] = true;
    for (std::size_t j = 1; j < n; ++j) { best[j] = d[j]; parent[j] = 0; }
    for (std::size_t step = 1; step < n; ++step) {
        std::size_t selected = absent;
        for (std::size_t j = 0; j < n; ++j) {
            if (included[j]) continue;
            if (selected == absent || best[j] < best[selected] ||
                (best[j] == best[selected] && edge(parent[j], static_cast<int>(j)) <
                 edge(parent[selected], static_cast<int>(selected)))) selected = j;
        }
        result.push_back(edge(parent[selected], static_cast<int>(selected)));
        included[selected] = true;
        for (std::size_t j = 0; j < n; ++j) {
            if (included[j]) continue;
            const double candidate = d[selected * n + j];
            if (candidate < best[j] || (candidate == best[j] &&
                edge(static_cast<int>(selected), static_cast<int>(j)) < edge(parent[j], static_cast<int>(j)))) {
                best[j] = candidate;
                parent[j] = static_cast<int>(selected);
            }
        }
    }
    return result;
}

PersistenceResult persistence(const std::vector<double>& distances, std::size_t n,
                              const PersistenceOptions& options) {
    if (std::isnan(options.max_radius) || options.max_radius < 0)
        throw std::invalid_argument("max_radius must be nonnegative (positive infinity is allowed)");
    const auto d = validate(distances, n);
    PersistenceResult result;
    result.n_vertices = n;
    if (n > options.max_simplices)
        throw std::runtime_error("max_simplices budget exceeded during vertex preflight");
    std::size_t count = n;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (d[i * n + j] > options.max_radius) result.truncated = true;
    if (!result.truncated) {
        add_choose(n, 2, options.max_simplices, count);
        add_choose(n, 3, options.max_simplices, count);
    } else {
        auto increment = [&]() {
            if (count == options.max_simplices)
                throw std::runtime_error("max_simplices budget exceeded during radius preflight");
            ++count;
        };
        // Count exactly the retained clique complex BEFORE allocating simplices.
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) {
                if (d[i * n + j] > options.max_radius) continue;
                increment();
                for (std::size_t k = j + 1; k < n; ++k)
                    if (d[i * n + k] <= options.max_radius && d[j * n + k] <= options.max_radius) increment();
            }
    }
    result.n_simplices = count;
    std::vector<Simplex> simplices;
    simplices.reserve(count);
    for (std::size_t i = 0; i < n; ++i)
        simplices.push_back({0, 0, {{static_cast<int>(i), -1, -1}}, {{-1, -1}}});
    for (int i = 0; i < static_cast<int>(n); ++i)
        for (int j = i + 1; j < static_cast<int>(n); ++j) {
            const double ij = d[static_cast<std::size_t>(i) * n + j];
            if (ij > options.max_radius) continue;
            simplices.push_back({ij, 1, {{i, j, -1}}, {{i, j}}});
            for (int k = j + 1; k < static_cast<int>(n); ++k) {
                const double ik = d[static_cast<std::size_t>(i) * n + k];
                const double jk = d[static_cast<std::size_t>(j) * n + k];
                const double value = std::max({ij, ik, jk});
                if (value > options.max_radius) continue;
                // Edges are considered in lexicographic order; strict > keeps
                // the lexicographically FIRST longest edge on exact ties.
                Edge critical{{i, j}};
                double longest = ij;
                if (ik > longest) { longest = ik; critical = {{i, k}}; }
                if (jk > longest) critical = {{j, k}};
                simplices.push_back({value, 2, {{i, j, k}}, critical});
            }
        }
    std::sort(simplices.begin(), simplices.end(), [](const Simplex& a, const Simplex& b) {
        if (a.value != b.value) return a.value < b.value;
        if (a.dimension != b.dimension) return a.dimension < b.dimension;
        return a.vertices < b.vertices;
    });

    std::vector<std::size_t> vertex_indices(n);
    std::unordered_map<std::uint64_t, std::size_t> edge_indices;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& s = simplices[i];
        if (s.dimension == 0) vertex_indices[s.vertices[0]] = i;
        if (s.dimension == 1) edge_indices.emplace(edge_key(s.vertices[0], s.vertices[1]), i);
    }
    // Only pivot columns are retained. Zero columns (including all H2 births)
    // are discarded; there is no transformation matrix or dense boundary.
    std::unordered_map<std::size_t, std::vector<std::size_t>> pivots;
    std::vector<bool> positive(count, false);
    std::vector<std::size_t> deaths(count, absent);
    Accounting accounting{options};
    std::size_t alive_h1 = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& s = simplices[i];
        // Exact H0/H1-only shortcut: if current H1=0, a new triangle's
        // boundary already lies in the span of earlier triangle boundaries.
        // Its reduced column is therefore zero. It can only create H2,
        // which this API deliberately does not report. Future edge births
        // re-enable reduction; no pivot column or H1 pair is discarded.
        if (s.dimension == 2 && alive_h1 == 0) continue;
        std::vector<std::size_t> column;
        const auto boundary_size = s.dimension == 0 ? 0u : static_cast<unsigned>(s.dimension + 1);
        accounting.operations(boundary_size);
        accounting.acquire(boundary_size);
        if (s.dimension == 1) {
            column = {vertex_indices[s.vertices[0]], vertex_indices[s.vertices[1]]};
        } else if (s.dimension == 2) {
            column = {edge_indices.at(edge_key(s.vertices[0], s.vertices[1])),
                      edge_indices.at(edge_key(s.vertices[0], s.vertices[2])),
                      edge_indices.at(edge_key(s.vertices[1], s.vertices[2]))};
        }
        std::sort(column.begin(), column.end());
        while (!column.empty()) {
            accounting.operations(1);
            const auto found = pivots.find(column.back());
            if (found == pivots.end()) break;
            const auto& other = found->second;
            std::vector<std::size_t> merged;
            std::size_t a = 0, b = 0;
            while (a < column.size() || b < other.size()) {
                if (a < column.size() && b < other.size() && column[a] == other[b]) {
                    accounting.operations(2);
                    ++a; ++b;
                } else {
                    accounting.operations(2); // One input consumed and one output written.
                    accounting.acquire(1); // Includes both old column and new output.
                    if (b == other.size() || (a < column.size() && column[a] < other[b]))
                        merged.push_back(column[a++]);
                    else merged.push_back(other[b++]);
                }
            }
            accounting.release(column.size());
            column = std::move(merged);
        }
        if (column.empty()) {
            positive[i] = true;
            if (s.dimension == 1) ++alive_h1;
        } else {
            if (s.dimension == 2) --alive_h1;
            const auto pivot = column.back();
            deaths[pivot] = i;
            pivots.emplace(pivot, std::move(column)); // Ownership changes, live count does not.
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto& birth = simplices[i];
        if (!positive[i] || birth.dimension > 1) continue;
        PersistencePair pair;
        pair.dimension = birth.dimension;
        pair.birth = birth.value;
        pair.birth_simplex = vertices(birth);
        pair.birth_edge = birth.critical;
        if (deaths[i] != absent) {
            const auto& death = simplices[deaths[i]];
            pair.death = death.value;
            pair.death_simplex = vertices(death);
            pair.death_edge = death.critical;
        } else {
            pair.censored = result.truncated;
            pair.essential = !result.truncated;
        }
        result.pairs.push_back(std::move(pair));
    }
    result.reduction_operations = accounting.work;
    result.peak_reduction_entries = accounting.peak;
    return result;
}
}  // namespace open_deep_tda
