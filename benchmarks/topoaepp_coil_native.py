#!/usr/bin/env python3
"""Bounded shared-COIL official TopoAE++ model/loss adapter; no labels/angles.
Only reads train/test arrays; no preprocessing or quality-based selection.
External upstream sources remain unchanged. CPU, max two compile jobs.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

PIN = "03f941d97305dbc045af7f595b6c5a3388808af2"
MAGIC = b"TDAF32LE"
ACK = "This document includes materials generated with TTK (the Topology ToolKit) which is developed by the CNRS & Sorbonne Universite and its contributors."
DRIVER = r'''
#include <TopologicalLoss.h>
#include <DimensionReductionModel.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <sys/resource.h>
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point a) {return std::chrono::duration<double>(Clock::now()-a).count();}
long peakRSS() {struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
return r.ru_maxrss;
#else
return r.ru_maxrss*1024L;
#endif
}
torch::Tensor readMatrix(const std::string &path, uint64_t expectedCols) {
  uint16_t endian=1;
  if(*reinterpret_cast<char*>(&endian)!=1) throw std::runtime_error("requires little endian host");
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f) throw std::runtime_error("cannot open input");
  auto size=f.tellg(); f.seekg(0);
  char magic[8]; uint64_t n=0,d=0;
  f.read(magic,8); f.read(reinterpret_cast<char*>(&n),8); f.read(reinterpret_cast<char*>(&d),8);
  if(!f || std::memcmp(magic,"TDAF32LE",8) || !n || n>960 || d!=expectedCols || size!=std::streamoff(24+4*n*d))
    throw std::runtime_error("invalid matrix header/shape/exact payload size");
  auto x=torch::empty({int64_t(n),int64_t(d)},torch::kFloat32);
  f.read(reinterpret_cast<char*>(x.data_ptr<float>()),4*n*d);
  if(!f || !torch::isfinite(x).all().item<bool>()) throw std::runtime_error("nonfinite/truncated matrix");
  return x;
}
void writeMatrix(const std::string &path,torch::Tensor x) {
  x=x.cpu().contiguous();
  if(x.scalar_type()!=torch::kFloat32 || x.dim()!=2 || !torch::isfinite(x).all().item<bool>()) throw std::runtime_error("invalid output tensor");
  uint64_t n=x.size(0),d=x.size(1);
  std::ofstream f(path,std::ios::binary); f.write("TDAF32LE",8);
  f.write(reinterpret_cast<char*>(&n),8);f.write(reinterpret_cast<char*>(&d),8);
  f.write(reinterpret_cast<char*>(x.data_ptr<float>()),4*n*d);
  if(!f) throw std::runtime_error("output write failed");
}
int main(int argc,char**argv) { try {
  if(argc!=6) throw std::runtime_error("usage: runtime train.f32 test.f32 outdir updates seed");
  std::cout<<std::setprecision(12);
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  int updates=std::stoi(argv[4]), seed=std::stoi(argv[5]);
  if(updates<1 || updates>1000 || seed!=0) throw std::runtime_error("invalid fixed protocol");
  torch::manual_seed(seed);
  auto begin=Clock::now();
  auto train=readMatrix(argv[1],64), test=readMatrix(argv[2],64);
  if(test.size(0)!=480 || (train.size(0)!=72 && train.size(0)!=960)) throw std::runtime_error("incorrect COIL protocol row counts");
  std::string out=argv[3];
  std::vector<std::vector<double>> points(train.size(0),std::vector<double>(64));
  auto data=train.data_ptr<float>();
  for(int64_t i=0;i<train.size(0);++i)for(int j=0;j<64;++j)points[i][j]=double(data[i*64+j]);
  ttk::AutoEncoder model(64,2,"128 32","ReLU",true);
  torch::optim::Adam opt(model.parameters(),torch::optim::AdamOptions(.01));
  std::cout<<"EVENT {\"kind\":\"source_ph_start\",\"n_train\":"<<train.size(0)<<",\"n_test\":"<<test.size(0)<<"}"<<std::endl;
  auto phStart=Clock::now();
  ttk::TopologicalLoss loss(train,points,ttk::TopologicalLoss::REGUL::ASYMMETRIC_CASCADE);
  double sourceSeconds=seconds(phStart);
  std::cout<<"EVENT {\"kind\":\"source_ph_ready\",\"seconds\":"<<sourceSeconds<<",\"peak_rss_bytes\":"<<peakRSS()<<"}"<<std::endl;
  auto fitStart=Clock::now();
  for(int step=0;step<updates;++step) {
    auto stepStart=Clock::now();opt.zero_grad();
    auto latent=model.encode(train).contiguous();
    auto rec=torch::mse_loss(model.decode(latent),train);
    auto topo=loss.computeLoss(latent);
    auto total=rec+.01*topo;
    if(!torch::isfinite(total).all().item<bool>()) throw std::runtime_error("nonfinite loss");
    total.backward();double g2=0;
    for(auto &p:model.parameters())if(p.grad().defined()) {
      if(!torch::isfinite(p.grad()).all().item<bool>()) throw std::runtime_error("nonfinite gradient");
      g2+=p.grad().square().sum().item<double>();
    }
    opt.step();
    std::cout<<"EVENT {\"kind\":\"step\",\"step\":"<<step<<",\"reconstruction\":"<<rec.item<double>()
             <<",\"topology_raw\":"<<topo.item<double>()<<",\"total\":"<<total.item<double>()
             <<",\"gradient_norm\":"<<sqrt(g2)<<",\"seconds\":"<<seconds(stepStart)<<",\"peak_rss_bytes\":"<<peakRSS()<<"}"<<std::endl;
  }
  double fitSeconds=seconds(fitStart);auto inferenceStart=Clock::now();
  model.eval();torch::NoGradGuard noGrad;
  auto ztrain=model.encode(train),ztest=model.encode(test);
  torch::serialize::OutputArchive saved;model.save(saved);saved.save_to(out+"/model.pt");
  ttk::AutoEncoder restored(64,2,"128 32","ReLU",true);
  torch::serialize::InputArchive loaded;loaded.load_from(out+"/model.pt");restored.load(loaded);restored.eval();
  auto testAgain=restored.encode(test);
  double error=(ztest-testAgain).abs().max().item<double>();
  if(!torch::isfinite(testAgain).all().item<bool>() || error>1e-7) throw std::runtime_error("encoder reload mismatch");
  writeMatrix(out+"/train_embedding.f32",ztrain);writeMatrix(out+"/test_embedding.f32",ztest);
  std::cout<<"EVENT {\"kind\":\"complete\",\"updates\":"<<updates<<",\"source_ph_seconds\":"<<sourceSeconds
           <<",\"fit_seconds\":"<<fitSeconds<<",\"inference_and_archive_seconds\":"<<seconds(inferenceStart)
           <<",\"total_native_seconds\":"<<seconds(begin)<<",\"reload_max_error\":"<<error
           <<",\"peak_rss_bytes\":"<<peakRSS()<<"}"<<std::endl;
  return 0;
} catch(const std::exception &e) {std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;} }
'''


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_matrix(path, matrix):
    if matrix.dtype != np.dtype("<f4") or matrix.ndim != 2 or not matrix.flags.c_contiguous:
        raise ValueError("expected exact C-contiguous little-endian float32 matrix")
    if not all(matrix.shape) or not np.isfinite(matrix).all():
        raise ValueError("empty or nonfinite matrix")
    with open(path, "wb") as f:
        f.write(struct.pack("<8sQQ", MAGIC, *matrix.shape))
        f.write(matrix.tobytes(order="C"))


def read_matrix(path, shape):
    raw = Path(path).read_bytes()
    if len(raw) < 24:
        raise ValueError("truncated header")
    magic, n, d = struct.unpack("<8sQQ", raw[:24])
    if magic != MAGIC or (n, d) != tuple(shape) or len(raw) != 24 + n * d * 4:
        raise ValueError("invalid header, shape, or exact payload size")
    x = np.frombuffer(raw, dtype="<f4", offset=24).reshape(n, d).copy()
    if not np.isfinite(x).all():
        raise ValueError("nonfinite matrix")
    return x


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def guarded(command, log, timeout, rss_limit=4 * 1024**3):
    """Sample this Python worker + child process tree every 50ms.
    Kill at 3.75 GiB total RSS, leaving 256MiB margin under the 4GiB budget.
    Native getrusage separately records the kernel-observed child peak.
    """
    started = time.monotonic()
    peak = 0
    reason = None
    with open(log, "w") as output:
        p = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
        me = psutil.Process()
        while p.poll() is None:
            try:
                rss = me.memory_info().rss + sum(c.memory_info().rss for c in psutil.Process(p.pid).children(recursive=True)) + psutil.Process(p.pid).memory_info().rss
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if peak >= rss_limit - 256 * 1024**2:
                reason = "rss_guard"
            elif time.monotonic() - started >= timeout:
                reason = "timeout"
            if reason:
                os.killpg(p.pid, signal.SIGKILL)
                break
            try:
                p.wait(timeout=.05)
            except subprocess.TimeoutExpired:
                pass
        p.wait()
    return {"argv": command, "returncode": p.returncode, "stop_reason": reason,
            "wall_seconds": time.monotonic() - started, "worker_plus_tree_peak_sampled_rss_bytes": peak,
            "rss_limit_bytes": rss_limit, "rss_kill_threshold_bytes": rss_limit - 256 * 1024**2,
            "rss_sample_seconds": .05, "timeout_seconds": timeout, "log": str(log)}


def build(upstream, boost, output):
    import torch
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("upstream commit mismatch")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--short"], text=True):
        raise ValueError("dirty upstream")
    helper = Path(__file__).resolve().parent / "topoaepp_runtime_adapter.py"
    spec = importlib.util.spec_from_file_location("native_compile_adapter", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = Path(tempfile.mkdtemp(prefix="topoaepp-coil-", dir="/tmp"))
    (work / "compat.h").write_text(module.COMPAT)
    (work / "driver.cpp").write_text(DRIVER)
    base = upstream / "ttk-tcdr/core/base"
    names = ["common/BaseClass.cpp", "common/Debug.cpp", "common/Os.cpp",
             "ripsPersistenceDiagram/RipsPersistenceDiagramUtils.cpp", "ripsPersistenceDiagram/ripserpy.cpp",
             "ripsPersistenceDiagram/PairCells.cpp", "ripsPersistenceDiagram/PairCellsWithOracle.cpp",
             "persistenceDiagramAuction/PersistenceDiagramAuction.cpp",
             "topologicallyConstrainedDimensionReduction/TopologicalLoss.cpp",
             "topologicallyConstrainedDimensionReduction/DimensionReductionModel.cpp"]
    sources = "\n".join('"' + str(base / n) + '"' for n in names)
    includes = "\n".join('"' + str(p) + '"' for p in base.iterdir() if p.is_dir())
    (work / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.18)
project(coil_runtime LANGUAGES CXX)
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
'''.replace("SOURCES", sources).replace("INCLUDES", includes))
    info = {"upstream_commit": PIN, "upstream": str(upstream), "workdir": str(work),
            "source_sha256": {n: sha(base / n) for n in names}, "compile_adapter": module.COMPAT,
            "compile_adapter_harness_sha256": sha(helper), "benchmark_sha256": sha(__file__),
            "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "boost_root": str(boost), "boost_version_header_sha256": sha(boost / "include/boost/version.hpp"), "commands": []}
    commands = [["cmake", "-S", str(work), "-B", str(work / "build"), "-DCMAKE_BUILD_TYPE=Release",
                 "-DCMAKE_PREFIX_PATH=" + torch.utils.cmake_prefix_path, "-DBOOST_ROOT=" + str(boost),
                 "-DBOOST_INCLUDEDIR=" + str(boost / "include"), "-DBoost_NO_SYSTEM_PATHS=ON", "-DBoost_NO_BOOST_CMAKE=ON"],
                ["cmake", "--build", str(work / "build"), "--parallel", "2"]]
    for i, command in enumerate(commands):
        result = guarded(command, output / f"build-{i}.log", 120)
        info["commands"].append(result)
        dump(output / "build.json", info)
        if result["returncode"]:
            raise RuntimeError("native build failed; see build.json/logs")
    exe = work / "build/runtime"
    info["executable_sha256"] = sha(exe)
    dump(output / "build.json", info)
    return exe, info


def stage(exe, name, train, test, rows, updates, timeout, output, common):
    folder = output / name
    folder.mkdir(exist_ok=False)
    selected = np.ascontiguousarray(train[rows])
    write_matrix(folder / "train.f32", selected)
    write_matrix(folder / "test.f32", test)
    np.save(folder / "train_row_indices.npy", rows, allow_pickle=False)
    info = {**common, "stage": name, "status": "running", "n_train": len(rows), "n_test": len(test), "input_dim": 64,
            "requested_updates": updates, "actual_updates": 0, "full_training_set": len(rows) == 960,
            "train_row_indices": rows.tolist(), "input_files_sha256": {n: sha(folder / n) for n in ["train.f32", "test.f32"]},
            "configuration": {"model": "official ttk::AutoEncoder", "architecture": "64-128-32-2", "activation": "ReLU", "batch_normalization": True,
                              "loss": "official ASYMMETRIC_CASCADE", "lambda": .01, "optimizer": "Adam", "lr": .01, "seed": 0, "torch_threads": 1,
                              "minibatching": False, "checkpoint_selection": "last update; no best-run selection", "output_mode": "eval; no-grad"},
            "scope": "official unchanged model/loss with standalone input/training driver and no-CGAL compile adapters; not paper/launcher reproduction",
            "acknowledgment": ACK}
    dump(folder / "metadata.json", info)
    process = guarded([str(exe), str(folder / "train.f32"), str(folder / "test.f32"), str(folder), str(updates), "0"], folder / "runtime.log", timeout)
    events = []
    for line in (folder / "runtime.log").read_text().splitlines():
        if line.startswith("EVENT "):
            events.append(json.loads(line[6:]))
    steps = [e for e in events if e["kind"] == "step"]
    completed = [e for e in events if e["kind"] == "complete"]
    info.update(process=process, events=events, actual_updates=len(steps), status="failed")
    if steps:
        info["step_seconds_mean"] = float(np.mean([s["seconds"] for s in steps]))
        info["step_seconds_max"] = max(s["seconds"] for s in steps)
    if process["returncode"] == 0 and len(steps) == updates and len(completed) == 1:
        a = read_matrix(folder / "train_embedding.f32", (len(rows), 2))
        b = read_matrix(folder / "test_embedding.f32", (480, 2))
        np.savez(folder / "embedding.npz", train=a, test=b)
        info.update(status="success", completion=completed[0],
                    embedding_sha256=sha(folder / "embedding.npz"), model_sha256=sha(folder / "model.pt"))
    info["native_peak_rss_bytes"] = max([e.get("peak_rss_bytes", 0) for e in events] or [0])
    dump(folder / "metadata.json", info)
    return info


def load_shared_features(features, prepared):
    # Only these arrays are materialized, even if the archive has other keys.
    with np.load(features, allow_pickle=False) as arrays:
        train, test = arrays["train"], arrays["test"]
    if train.shape != (960, 64) or test.shape != (480, 64):
        raise ValueError("wrong shared feature shape")
    for x in [train, test]:
        if x.dtype != np.float32 or not x.flags.c_contiguous or not np.isfinite(x).all():
            raise ValueError("shared features must be finite row-major float32")
    digest = hashlib.sha256(train.tobytes(order="C") + test.tobytes(order="C")).hexdigest()
    protocol = json.loads(Path(prepared).read_text())
    if digest != protocol["feature_sha256"]:
        raise ValueError("prepared feature content digest mismatch")
    return train, test, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("outputs/v02-coil20/features.npz"))
    ap.add_argument("--prepared", type=Path, default=Path("outputs/v02-coil20/prepared.json"))
    ap.add_argument("--output", type=Path, default=Path("outputs/topoaepp-coil-native"))
    ap.add_argument("--upstream", type=Path, default=Path("/tmp/tda-upstream-research/TopologicalAutoencodersPlusPlus"))
    ap.add_argument("--boost-root", type=Path, default=Path("/tmp/topoaepp-boost/prefix/usr"))
    ap.add_argument("--pilot-step-gate", type=float, default=0.3,
                    help="explicit runtime-only gate; changing this is recorded, never quality based")
    ap.add_argument("--final-timeout", type=float, default=600.)
    args = ap.parse_args()
    if not np.isfinite(args.pilot_step_gate) or args.pilot_step_gate <= 0 or not np.isfinite(args.final_timeout) or not 0 < args.final_timeout <= 3600:
        ap.error("runtime gate/timeout must be finite and positive; timeout <=3600s")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise SystemExit("refusing to overwrite an existing run")
    train, test, feature_digest = load_shared_features(args.features, args.prepared)
    common = {"features_path": str(args.features.resolve()), "features_archive_sha256": sha(args.features),
              "features_content_sha256": feature_digest,
              "prepared_path": str(args.prepared.resolve()), "prepared_sha256": sha(args.prepared),
              "preprocessing": "exact shared arrays, no further scaling/normalization; float32 values converted to double for source PH",
              "labels_or_angles_loaded": False, "budget_selection": "runtime only, not quality", "acknowledgment": ACK,
              "pilot_step_gate_seconds": args.pilot_step_gate, "final_timeout_seconds": args.final_timeout}
    result = {**common, "status": "running", "stages": {}, "final_requested_updates": 1000,
              "final_actual_updates": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dump(output / "metadata.json", result)
    begin = time.monotonic()
    try:
        exe, buildinfo = build(args.upstream.resolve(), args.boost_root.resolve(), output)
        common["build"] = buildinfo
        result["build"] = buildinfo
        rows = np.sort(np.random.RandomState(0).choice(960, 72, replace=False))
        probe = stage(exe, "probe72", train, test, rows, 5, 60, output, common)
        result["stages"]["probe72"] = str(output / "probe72/metadata.json")
        dump(output / "metadata.json", result)
        if probe["status"] != "success":
            raise RuntimeError("72-row runtime probe failed; no full run attempted")
        pilot = stage(exe, "pilot100", train, test, np.arange(960), 100, 180, output, common)
        result["stages"]["pilot100"] = str(output / "pilot100/metadata.json")
        dump(output / "metadata.json", result)
        if pilot["status"] != "success":
            raise RuntimeError("FULL960 pilot failed; no subset substituted")
        if pilot["step_seconds_mean"] >= args.pilot_step_gate:
            raise RuntimeError(f"FULL960 pilot step time >= {args.pilot_step_gate:g}s; final run blocked by declared runtime gate")
        final = stage(exe, "final1000", train, test, np.arange(960), 1000, args.final_timeout, output, common)
        result["stages"]["final1000"] = str(output / "final1000/metadata.json")
        result["final_actual_updates"] = final["actual_updates"]
        if final["status"] != "success":
            raise RuntimeError("FULL960 1000-update final failed; pilot not relabelled final")
        shutil.copyfile(output / "final1000/embedding.npz", output / "embedding.npz")
        result.update(final)
        result["final_stage_metadata"] = str(output / "final1000/metadata.json")
        result["status"] = "success"
        result["final_actual_updates"] = final["actual_updates"]
    except Exception as exc:
        result.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["overall_wall_seconds"] = time.monotonic()-begin
        result["upstream_status_after"] = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--short"], text=True)
        dump(output / "metadata.json", result)
    print(json.dumps({k: result.get(k) for k in ["status", "error", "actual_updates", "overall_wall_seconds", "stages"]}, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
