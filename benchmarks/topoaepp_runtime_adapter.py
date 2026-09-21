#!/usr/bin/env python3
"""Build a small genuine external TopoAE++ C++ loss/PH subset, never copy it.
Creates only temporary CMake/adapter/driver/build files; persists result JSON.
Requires CMake, C++17, Boost headers, and CPU LibTorch (or installed torch).
No ParaView, CGAL, OpenMP, downloads or installs. Each build is bounded.
"""
import argparse
import hashlib
import json
import os
import signal
import platform
import subprocess
import tempfile
import time
from pathlib import Path

PIN = "03f941d97305dbc045af7f595b6c5a3388808af2"
# Own compilation adapter: missing standalone standard includes and the stale
# no-CGAL spelling. All official translation units remain external and unchanged.
COMPAT = """#pragma once
// Boost 1.74 feature detection predates libc++ removing std::unary_function.
#ifndef BOOST_NO_CXX98_FUNCTION_BASE
#define BOOST_NO_CXX98_FUNCTION_BASE
#endif
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <RipsPersistenceDiagramUtils.h>
#include <PairCells.h>
namespace ttk::rpd { using edge_set_set_t = EdgeSetSet; }
"""
DRIVER = r'''
#include <TopologicalLoss.h>
#include <DimensionReductionModel.h>
#include <iostream>
#include <iomanip>
#include <chrono>
int main() {
  torch::set_num_threads(1);
  torch::manual_seed(2026);
  std::cout << std::setprecision(12);
  auto started = std::chrono::steady_clock::now();
  auto x = torch::tensor({{0.f,0.f,0.f},{1.1f,-.1f,.1f},{2.f,.4f,0.f},
                         {1.7f,1.5f,.1f},{.6f,1.8f,0.f},{-.3f,.9f,.1f}});
  std::vector<std::vector<double>> points;
  for(int i=0;i<x.size(0);++i) {
    std::vector<double> row;
    for(int j=0;j<x.size(1);++j) row.push_back(x[i][j].item<double>());
    points.push_back(row);
  }
  ttk::rpd::MultidimensionalDiagram pd;
  ttk::rpd::PairCellsWithOracle::callOracle(points,pd);
  ttk::rpd::PairCellsWithOracle pc(points,pd,false,false);
  pc.run();
  ttk::rpd::EdgeSet cascades;
  ttk::rpd::EdgeSetSet critical;
  pc.getCascades(cascades,critical);
  int positive=0;
  for(auto &p:pd.at(1)) if(p.second.second>p.first.second && std::isfinite(p.second.second)) ++positive;
  if(!positive || cascades.empty()) return 2;
  for(auto reg:{ttk::TopologicalLoss::REGUL::TOPOAE_DIM1,
                ttk::TopologicalLoss::REGUL::ASYMMETRIC_CASCADE}) {
    auto z=(x.slice(1,0,2)*torch::tensor({.7f,1.2f})).contiguous().detach().set_requires_grad(true);
    ttk::TopologicalLoss loss(x,points,reg);
    auto value=loss.computeLoss(z);
    value.backward();
    double norm=z.grad().norm().item<double>();
    if(!torch::isfinite(value).all().item<bool>() || !torch::isfinite(z.grad()).all().item<bool>() || norm<=0) return 3;
    double identity = loss.computeLoss(x).item<double>();
    if(std::abs(identity)>1e-8) return 4;
    std::cout << "RUNTIME_JSON {\"kind\":\"loss\",\"identity_loss\":" << identity << ",\"regul\":" << int(reg)
              << ",\"loss\":" << value.item<double>() << ",\"gradient_norm\":" << norm
              << ",\"source_positive_h1\":" << positive
              << ",\"source_strict_cascade_edges\":" << cascades.size()
              << ",\"source_mst_edges\":" << critical.at(0).size()
              << ",\"source_h1_birth_edges\":" << critical.at(1).size()
              << ",\"source_h1_death_edges\":" << critical.at(2).size() << "}" << std::endl;
  }
  // Own tiny driver; actual upstream AutoEncoder, PH, and loss are unchanged.
  ttk::AutoEncoder model(3, 2, "128 32", "ReLU", true);
  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(.01));
  ttk::TopologicalLoss regularizer(x, points, ttk::TopologicalLoss::REGUL::ASYMMETRIC_CASCADE);
  for(int step=0;step<2;++step) {
    optimizer.zero_grad();
    auto latent=model.encode(x).contiguous();
    auto reconstructed=model.decode(latent);
    auto topo=regularizer.computeLoss(latent);
    auto rec=torch::mse_loss(reconstructed,x);
    topo.backward({}, true);
    double topoGrad2=0;
    for(auto &p:model.parameters()) if(p.grad().defined()) topoGrad2+=p.grad().square().sum().item<double>();
    optimizer.zero_grad();
    auto total=rec+topo;
    total.backward();
    double grad2=0;
    for(auto &p:model.parameters()) if(p.grad().defined()) {
      if(!torch::isfinite(p.grad()).all().item<bool>()) return 5;
      grad2+=p.grad().square().sum().item<double>();
    }
    if(!torch::isfinite(total).all().item<bool>() || topoGrad2<=0 || !std::isfinite(topoGrad2)) return 6;
    optimizer.step();
    std::cout << "RUNTIME_JSON {\"kind\":\"model_update\",\"step\":" << step
              << ",\"reconstruction\":" << rec.item<double>() << ",\"topology\":" << topo.item<double>()
              << ",\"total\":" << total.item<double>() << ",\"topology_parameter_gradient_norm\":" << sqrt(topoGrad2)
              << ",\"total_parameter_gradient_norm\":" << sqrt(grad2) << "}" << std::endl;
  }
  model.eval();
  torch::NoGradGuard noGrad;
  auto heldout=torch::tensor({{.25f,.35f,.05f},{1.25f,1.15f,.07f}});
  auto embedded=model.encode(heldout);
  torch::serialize::OutputArchive outputArchive;
  model.save(outputArchive);
  std::stringstream bytes;
  outputArchive.save_to(bytes);
  ttk::AutoEncoder reloaded(3,2,"128 32","ReLU",true);
  torch::serialize::InputArchive inputArchive;
  bytes.seekg(0);
  inputArchive.load_from(bytes);
  reloaded.load(inputArchive);
  reloaded.eval();
  auto embeddedAgain=reloaded.encode(heldout);
  double error=(embedded-embeddedAgain).abs().max().item<double>();
  if(!torch::isfinite(embeddedAgain).all().item<bool>() || error>1e-7) return 7;
  double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  std::cout << "RUNTIME_JSON {\"kind\":\"encoder_roundtrip\",\"heldout_rows\":2,\"latent_dim\":2,\"max_error\":"
            << error << ",\"archive_bytes\":" << bytes.str().size() << ",\"native_elapsed_seconds\":" << elapsed << "}" << std::endl;
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("upstream", type=Path)
    ap.add_argument("result", type=Path)
    ap.add_argument("--torch-prefix", help="LibTorch CMake prefix; otherwise use Python torch")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--boost-root", type=Path, help="Isolated portable header prefix containing include/boost")
    ap.add_argument("--work-dir", type=Path, help="Reuse an isolated build directory for bounded incremental retries")
    args = ap.parse_args()
    root = args.upstream.resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        raise SystemExit(f"Expected {PIN}, got {head}")
    before = subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True)
    if before:
        raise SystemExit("Refusing a dirty upstream checkout: " + before)
    b = root / "ttk-tcdr/core/base"
    if args.torch_prefix:
        prefix = args.torch_prefix
    else:
        import torch
        prefix = torch.utils.cmake_prefix_path
    work = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="topoaepp-runtime-", dir="/tmp"))
    if not work.resolve().is_relative_to(Path("/tmp").resolve()):
        raise SystemExit("Build work directory must be isolated under /tmp")
    work.mkdir(parents=True, exist_ok=True)
    def write_if_changed(path, text):
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
    write_if_changed(work / "compat.h", COMPAT)
    write_if_changed(work / "driver.cpp", DRIVER)
    files = ["common/BaseClass.cpp", "common/Debug.cpp", "common/Os.cpp",
             "ripsPersistenceDiagram/RipsPersistenceDiagramUtils.cpp",
             "ripsPersistenceDiagram/ripserpy.cpp", "ripsPersistenceDiagram/PairCells.cpp",
             "ripsPersistenceDiagram/PairCellsWithOracle.cpp",
             "persistenceDiagramAuction/PersistenceDiagramAuction.cpp",
             "topologicallyConstrainedDimensionReduction/TopologicalLoss.cpp",
             "topologicallyConstrainedDimensionReduction/DimensionReductionModel.cpp"]
    sources = "\n".join('"' + str(b / f) + '"' for f in files)
    includes = "\n".join('"' + str(p) + '"' for p in sorted(b.iterdir()) if p.is_dir())
    cmake = '''cmake_minimum_required(VERSION 3.18)
project(upstream_runtime LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED)
add_executable(runtime driver.cpp SOURCES)
target_include_directories(runtime PRIVATE INCLUDES)
target_include_directories(runtime SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
target_compile_definitions(runtime PRIVATE TTK_ENABLE_TORCH)
target_compile_options(runtime PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/compat.h")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(runtime PRIVATE "${TORCH_LIBRARIES}")
'''.replace("SOURCES", sources).replace("INCLUDES", includes)
    write_if_changed(work / "CMakeLists.txt", cmake)
    report = {"commit": head, "platform": platform.platform(), "workdir": str(work),
              "scope": "external unchanged official PH/loss/model; standalone driver, portable Boost, no-CGAL compile adapters; tiny AE updates and encoder archive roundtrip",
              "adapters": COMPAT, "commands": [], "success": False, "upstream_status_before": before,
              "source_sha256": {f: hashlib.sha256((b/f).read_bytes()).hexdigest() for f in files}}
    start = time.perf_counter()
    commands = [["cmake", "-S", str(work), "-B", str(work/"build"),
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_PREFIX_PATH=" + prefix],
                ["cmake", "--build", str(work/"build"), "--parallel", "2"],
                [str(work/"build/runtime")], [str(work/"build/runtime")]]
    if args.boost_root:
        boost = args.boost_root.resolve()
        if not (boost / "include/boost/version.hpp").is_file():
            raise SystemExit("--boost-root must contain include/boost/version.hpp")
        commands[0] += ["-DBOOST_ROOT=" + str(boost), "-DBOOST_INCLUDEDIR=" + str(boost / "include"),
                        "-DBoost_NO_SYSTEM_PATHS=ON", "-DBoost_NO_BOOST_CMAKE=ON"]
        report["boost_root"] = str(boost)
        report["boost_version_header_sha256"] = hashlib.sha256((boost / "include/boost/version.hpp").read_bytes()).hexdigest()
    if args.result.exists():
        previous = json.loads(args.result.read_text())
        report["prior_attempts"] = previous.pop("prior_attempts", []) + [previous]
    try:
        for command in commands:
            try:
                p = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
                output, _ = p.communicate(timeout=args.timeout)
                report["commands"].append({"argv": command, "returncode": p.returncode, "output": output})
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                output, _ = p.communicate()
                report["commands"].append({"argv": command, "error": "timeout", "output": output})
                break
            if p.returncode:
                break
        else:
            report["values"] = [json.loads(line.split("RUNTIME_JSON ",1)[1]) for line in output.splitlines() if line.startswith("RUNTIME_JSON ")]
            report["success"] = ([v["kind"] for v in report["values"]] ==
                                 ["loss", "loss", "model_update", "model_update", "encoder_roundtrip"])
            report["executable"] = str(work / "build/runtime")
            report["executable_sha256"] = hashlib.sha256((work / "build/runtime").read_bytes()).hexdigest()
            first = [json.loads(line.split("RUNTIME_JSON ",1)[1]) for line in report["commands"][-2]["output"].splitlines() if line.startswith("RUNTIME_JSON ")]
            def stable(values):
                return [{k: v for k, v in item.items() if k != "native_elapsed_seconds"} for item in values]
            report["repeat_values_match"] = stable(first) == stable(report["values"])
            report["success"] = report["success"] and report["repeat_values_match"]
    finally:
        report["elapsed_seconds"] = time.perf_counter()-start
        report["upstream_status_after"] = subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True)
        args.result.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
