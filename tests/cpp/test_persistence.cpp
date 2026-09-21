#include "open_deep_tda/persistence.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace open_deep_tda;
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class E, class F> void throws(F f, const char* message) {
    try { f(); } catch (const E&) { return; }
    throw std::runtime_error(message);
}
std::vector<double> points(const std::vector<std::array<double, 2>>& x) {
    std::vector<double> d(x.size() * x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = 0; j < x.size(); ++j)
            d[i * x.size() + j] = std::hypot(x[i][0] - x[j][0], x[i][1] - x[j][1]);
    return d;
}

// Independent tiny oracle: dense bitset columns, all boundaries materialized.
// Deliberately different data structures from the production sparse reducer.
void oracle(const std::vector<double>& d, int n, double radius) {
    struct S { double value; std::vector<int> v; };
    std::vector<S> s;
    for (int i = 0; i < n; ++i) s.push_back({0, {i}});
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
        if (d[i*n+j] <= radius) s.push_back({d[i*n+j], {i,j}});
        for (int k = j + 1; k < n; ++k) {
            const auto value = std::max({d[i*n+j], d[i*n+k], d[j*n+k]});
            if (value <= radius) s.push_back({value, {i,j,k}});
        }
    }
    std::sort(s.begin(), s.end(), [](const S& a, const S& b) {
        return std::make_tuple(a.value, a.v.size(), a.v) < std::make_tuple(b.value, b.v.size(), b.v);
    });
    require(s.size() <= 64, "oracle exceeds word size");
    std::vector<std::uint64_t> boundary(s.size()), reduced(s.size());
    std::vector<int> pivot(s.size(), -1), death(s.size(), -1);
    for (std::size_t j = 0; j < s.size(); ++j) {
        if (s[j].v.size() == 1) continue;
        for (std::size_t a = 0; a < s[j].v.size(); ++a) {
            auto face = s[j].v; face.erase(face.begin() + static_cast<std::ptrdiff_t>(a));
            std::size_t row = 0;
            while (row < s.size() && s[row].v != face) ++row;
            require(row < j, "face does not precede coface");
            boundary[j] ^= std::uint64_t(1) << row;
        }
        std::uint64_t dd = 0;
        for (std::size_t i = 0; i < j; ++i)
            if ((boundary[j] >> i) & 1) dd ^= boundary[i];
        require(dd == 0, "boundary squared is nonzero");
    }
    for (std::size_t j = 0; j < s.size(); ++j) {
        auto c = boundary[j];
        while (c) {
            int row = 63;
            while (!((c >> row) & 1)) --row;
            if (pivot[row] == -1) { pivot[row] = static_cast<int>(j); death[row] = static_cast<int>(j); break; }
            c ^= reduced[pivot[row]];
        }
        reduced[j] = c;
    }
    PersistenceOptions options; options.max_radius = radius;
    const auto actual = persistence(d, n, options);
    require(actual.n_simplices == s.size(), "oracle simplex count mismatch");
    std::size_t k = 0;
    for (std::size_t j = 0; j < s.size(); ++j) {
        if (reduced[j] || s[j].v.size() > 2) continue;
        require(k < actual.pairs.size(), "oracle missing pair");
        const auto& p = actual.pairs[k++];
        require(p.birth_simplex == s[j].v && p.birth == s[j].value, "oracle birth mismatch");
        if (death[j] >= 0) {
            require(p.death_simplex == s[death[j]].v && p.death == s[death[j]].value, "oracle death mismatch");
            double longest = -1;
            Edge critical{{-1,-1}};
            for (std::size_t a = 0; a < p.death_simplex.size(); ++a)
                for (std::size_t b = a + 1; b < p.death_simplex.size(); ++b) {
                    const int u = p.death_simplex[a], v = p.death_simplex[b];
                    if (d[u*n+v] > longest) { longest = d[u*n+v]; critical = {{u,v}}; }
                }
            require(p.death_edge == critical, "critical edge mismatch");
        } else require(std::isinf(p.death) && p.death_simplex.empty(), "oracle unpaired mismatch");
    }
    require(k == actual.pairs.size(), "oracle extra pair");
    std::vector<double> h0, tree;
    for (const auto& p : actual.pairs) if (p.dimension == 0 && std::isfinite(p.death)) h0.push_back(p.death);
    for (const auto& e : mst(d, n)) if (d[e[0]*n+e[1]] <= radius) tree.push_back(d[e[0]*n+e[1]]);
    std::sort(h0.begin(), h0.end()); std::sort(tree.begin(), tree.end());
    require(h0 == tree, "H0 deaths differ from MST lengths");
}

void fixtures() {
    require(persistence({}, 0).pairs.empty(), "empty persistence");
    require(mst({}, 0).empty(), "empty MST");
    const auto singleton = persistence({0}, 1);
    require(singleton.pairs.size() == 1 && singleton.pairs[0].essential, "singleton essential H0");
    const std::vector<double> equilateral{0,1,1,1,0,1,1,1,0};
    const auto tri = persistence(equilateral, 3);
    int zero_h1 = 0;
    for (const auto& p : tri.pairs) if (p.dimension == 1) {
        require(p.birth == 1 && p.death == 1, "equilateral H1 not zero");
        require(p.birth_edge == Edge{{1,2}} && p.death_edge == Edge{{0,1}}, "tie critical edge");
        ++zero_h1;
    }
    require(zero_h1 == 1, "equilateral zero bar omitted");
    require(mst(equilateral,3) == std::vector<Edge>{{{0,1}},{{0,2}}}, "Prim ties");
    const auto square = points({{{0,0}},{{1,0}},{{1,1}},{{0,1}}});
    const auto full = persistence(square, 4);
    int positive_h1 = 0;
    for (const auto& p : full.pairs) if (p.dimension == 1 && p.death > p.birth) {
        require(p.birth == 1 && p.death == std::sqrt(2.), "square barcode"); ++positive_h1;
    }
    require(positive_h1 == 1 && full.n_simplices == 14, "square counts");
    PersistenceOptions o; o.max_radius = 1;
    const auto cut = persistence(square, 4, o);
    require(cut.truncated && cut.n_simplices == 8, "radius counts");
    int censored_h1 = 0;
    for (const auto& p : cut.pairs) if (p.dimension == 1) {
        require(p.censored && !p.essential && std::isinf(p.death) && p.death_edge == Edge{{-1,-1}}, "censored H1");
        ++censored_h1;
    }
    require(censored_h1 == 1, "missing censored square");
    o.max_radius = 2;
    require(!persistence(square, 4, o).truncated, "finite complete radius is not truncated");
    const auto duplicates = persistence(std::vector<double>(16,0),4);
    require(duplicates.pairs.size() == 7, "duplicates zero bar count");
    for (const auto& p : duplicates.pairs) require(p.birth == 0 && (p.essential || p.death == 0), "duplicates values");
    // Exact budgets succeed, one less fails: includes transient XOR output.
    o = {}; o.max_simplices = full.n_simplices;
    o.max_reduction_entries = full.peak_reduction_entries;
    o.max_reduction_operations = full.reduction_operations;
    const auto exact = persistence(square,4,o);
    require(exact.reduction_operations == full.reduction_operations, "nondeterministic accounting");
    --o.max_simplices;
    throws<std::runtime_error>([&]{persistence(square,4,o);}, "simplex budget not enforced");
    ++o.max_simplices; --o.max_reduction_entries;
    throws<std::runtime_error>([&]{persistence(square,4,o);}, "entry budget not enforced");
    ++o.max_reduction_entries; --o.max_reduction_operations;
    throws<std::runtime_error>([&]{persistence(square,4,o);}, "work budget not enforced");
    o = {}; o.max_radius = 1; o.max_simplices = 7;
    throws<std::runtime_error>([&]{persistence(square,4,o);}, "truncated preflight not enforced");
    o = {}; o.max_simplices = 0; o.max_reduction_entries = 0; o.max_reduction_operations = 0;
    require(persistence({},0,o).n_simplices == 0, "zero budget empty input");
}

void validation() {
    auto invalid = [](std::vector<double> d, std::size_t n) {
        throws<std::invalid_argument>([&]{persistence(d,n);}, "PH malformed input accepted");
        throws<std::invalid_argument>([&]{mst(d,n);}, "MST malformed input accepted");
    };
    invalid({0,1,1},2); invalid({0,-1,-1,0},2); invalid({1,1,1,0},2);
    invalid({0,1,2,0},2); invalid({0,std::nan(""),1,0},2);
    invalid({0,INFINITY,INFINITY,0},2);
    const std::vector<double> roundoff{1e-12, 1, 1+1e-12, -1e-12};
    require(persistence(roundoff,2).pairs.size() == 2 && mst(roundoff,2).size() == 1, "legitimate roundoff rejected");
    require(persistence({0,-1e-12,-1e-12,0},2).pairs.size() == 2, "tiny negative roundoff rejected");
    const double large = std::numeric_limits<double>::max();
    require(std::isfinite(persistence({0,large,large,0},2).pairs.back().death), "averaging overflow");
    PersistenceOptions o; o.max_radius = -1;
    throws<std::invalid_argument>([&]{persistence({0},1,o);}, "negative radius");
    o.max_radius = std::nan("");
    throws<std::invalid_argument>([&]{persistence({0},1,o);}, "NaN radius");
}
}  // namespace
int main() {
    try {
        fixtures(); validation();
        std::mt19937 rng(12345);
        for (int n = 0; n <= 7; ++n) for (int trial = 0; trial < 40; ++trial) {
            std::vector<double> d(n*n);
            for (int i = 0; i < n; ++i) for (int j = i+1; j < n; ++j)
                d[i*n+j] = d[j*n+i] = static_cast<double>(rng()%9) / 4;
            oracle(d,n,INFINITY); oracle(d,n,0.75);
        }
        std::cout << "Native PH/MST: fixtures, validation, budgets, and 640 dense-oracle comparisons passed.\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
