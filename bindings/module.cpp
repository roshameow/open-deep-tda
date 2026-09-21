#include "open_deep_tda/persistence.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace {
std::vector<double> matrix(const py::array& input, std::size_t& n) {
    if (input.ndim() != 2 || input.shape(0) != input.shape(1))
        throw std::invalid_argument("distances must be a square two-dimensional ndarray");
    if (!input.dtype().is(py::dtype::of<double>()) || !(input.flags() & py::array::c_style))
        throw std::invalid_argument("distances must be a C-contiguous float64 ndarray");
    n = static_cast<std::size_t>(input.shape(0));
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        (n && n > std::numeric_limits<std::size_t>::max() / n))
        throw std::invalid_argument("distance matrix is too large");
    if (n == 0) return {};
    const auto* data = static_cast<const double*>(input.data());
    return {data, data + n * n}; // Own the input while the GIL is released.
}
std::size_t budget(const py::object& value, const char* name) {
    if (py::isinstance<py::bool_>(value) || !PyIndex_Check(value.ptr()))
        throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
    py::object integer = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if (!integer) throw py::error_already_set();
    const auto result = PyLong_AsSize_t(integer.ptr());
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw std::invalid_argument(std::string(name) + " must be a representable nonnegative integer");
    }
    return result;
}
py::dict persistence_result(const py::array& distances, const py::object& max_simplices,
                            const py::object& max_reduction_entries,
                            const py::object& max_reduction_operations, double max_radius,
                            bool h1_only) {
    std::size_t n;
    const auto input = matrix(distances, n);
    open_deep_tda::PersistenceOptions options;
    options.max_simplices = budget(max_simplices, "max_simplices");
    options.max_reduction_entries = budget(max_reduction_entries, "max_reduction_entries");
    options.max_reduction_operations = budget(max_reduction_operations, "max_reduction_operations");
    options.max_radius = max_radius;
    open_deep_tda::PersistenceResult result;
    { py::gil_scoped_release release; result = open_deep_tda::persistence(input, n, options); }
    py::list pairs;
    for (const auto& pair : result.pairs) {
        // Filtering here avoids Python objects, not native PH work or storage.
        // Keep unpaired H1 explicitly so consumers can reject censoring/essentials.
        if (h1_only && (pair.dimension != 1 ||
            !(pair.death > pair.birth || pair.censored || pair.essential))) continue;
        py::dict p;
        p["dimension"] = pair.dimension;
        p["birth"] = pair.birth;
        p["death"] = pair.death;
        p["birth_simplex"] = py::cast(pair.birth_simplex);
        p["death_simplex"] = py::cast(pair.death_simplex);
        p["birth_edge"] = py::cast(pair.birth_edge);
        p["death_edge"] = py::cast(pair.death_edge);
        p["essential"] = pair.essential;
        p["censored"] = pair.censored;
        pairs.append(std::move(p));
    }
    py::dict output;
    output["pairs"] = pairs;
    output["n_vertices"] = result.n_vertices;
    output["n_simplices"] = result.n_simplices;
    output["reduction_operations"] = result.reduction_operations;
    output["peak_reduction_entries"] = result.peak_reduction_entries;
    output["truncated"] = result.truncated;
    return output;
}
}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Original C++17 exact VR H0/H1 persistence and deterministic dense Prim MST.";
    m.def("mst", [](const py::array& distances) {
        std::size_t n;
        const auto input = matrix(distances, n);
        std::vector<open_deep_tda::Edge> edges;
        { py::gil_scoped_release release; edges = open_deep_tda::mst(input, n); }
        py::array_t<int> result({static_cast<py::ssize_t>(edges.size()), py::ssize_t(2)});
        auto view = result.mutable_unchecked<2>();
        for (std::size_t i = 0; i < edges.size(); ++i) {
            view(i, 0) = edges[i][0]; view(i, 1) = edges[i][1];
        }
        return result;
    }, py::arg("distances"), "Deterministic Prim MST; ties use lexicographic undirected edges.");
    m.def("persistence", [](const py::array& distances, const py::object& max_simplices,
                            const py::object& max_reduction_entries,
                            const py::object& max_reduction_operations, double max_radius) {
        return persistence_result(distances, max_simplices, max_reduction_entries,
                                  max_reduction_operations, max_radius, false);
    }, py::arg("distances"), py::arg("max_simplices") = py::int_(1000000),
       py::arg("max_reduction_entries") = py::int_(10000000),
       py::arg("max_reduction_operations") = py::int_(100000000),
       py::arg("max_radius") = std::numeric_limits<double>::infinity(),
       "Exact VR 2-skeleton over F2. Zero bars retained; radius-unpaired bars are censored.\n"
       "Budget exhaustion raises RuntimeError, never a partial result. See persistence.hpp for accounting.");
    m.def("persistence_h1", [](const py::array& distances, const py::object& max_simplices,
                               const py::object& max_reduction_entries,
                               const py::object& max_reduction_operations, double max_radius) {
        return persistence_result(distances, max_simplices, max_reduction_entries,
                                  max_reduction_operations, max_radius, true);
    }, py::arg("distances"), py::arg("max_simplices") = py::int_(1000000),
       py::arg("max_reduction_entries") = py::int_(10000000),
       py::arg("max_reduction_operations") = py::int_(100000000),
       py::arg("max_radius") = std::numeric_limits<double>::infinity(),
       "Same exact VR computation and budgets as persistence; serialize only positive H1 and unpaired H1.\n"
       "Censored/essential H1 and full computation statistics are retained; H0 and zero H1 are omitted.");

}
