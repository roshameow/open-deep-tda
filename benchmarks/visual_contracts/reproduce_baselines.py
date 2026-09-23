#!/usr/bin/env python3
"""No-download, seed-0 baseline runner; all raw artifacts stay in external scratch.

Public interface (also printed by --help):
  --upstream DIR       Required external, clean author Git checkout at UPSTREAM_PIN.
  --data-dir DIR       Optional external CSV directory; defaults to upstream/data.
  --output DIR         Required NEW directory outside every Git checkout and this
                       repository. Existing paths (even empty directories) fail.
  --dataset NAME [...] Twist, K4, COIL20-1, 3Clusters, Digits; default all four CSVs.
  --protocol author-data (default) or digits-matched; see distinctions below.
  --methods NAME [...] PCA, UMAP, GraphEmbedding, AUTHOR. Defaults: all four for
                       author-data; first three for digits-matched.
  --cpu-adapter EXE --adapter-sha256 SHA --adapter-manifest JSON
                       All required when AUTHOR is selected, even --check-only.
  --check-only         Validate provenance, files, input and declared protocol;
                       freeze registration, but NEVER execute adapters or fits.

Requires already installed numpy; controls additionally need sklearn, genuine
umap-learn/pynndescent, and this source checkout's GraphEmbedding dependencies.
No installs, downloads, builds, external edits, retries or best-seed selection.
No author source is bundled. --upstream is inspected, never imported/executed.
The adapter is a TRUSTED external native program, NOT a sandboxed dependency.
Its user-supplied JSON must contain every field/value of ADAPTER_CONTRACT below,
plus executable_sha256 equal to --adapter-sha256. Additional metadata is allowed.
The full manifest template is printed at the bottom of --help. A declaration and
hash cannot prove an executable implements its claims: audit it before use.
A generic-dimensional adapter is mandatory; COIL64-specific binaries are refused.

Native CLI: EXE input.f64 output.f32 (no flags). Input is b'TDAF64LE', uint64
little-endian n,d, then exactly n*d row-major float64 little-endian values. Output
is exactly n*2 raw float32 little-endian values, finite, with no header. Source PH
uses original doubles; the network uses float32. Stdout emits 'EVENT ' followed by
one JSON object per line: kind='step', step=0 through 999 in order, then exactly
one kind='complete', updates=1000, output_mode='train'. Source-PH progress events
may precede steps. Exit zero, all 1000 steps, complete and valid output are ALL
required; timeout/partial output is failure, never an embedding. Each subprocess
has a 150-second wall cap; all fit children share a 900-second budget, no retries.

AUTHOR: original upstream model/loss, asymmetric cascade, d->128->32->2 and
reverse decoder, ReLU then BatchNorm hidden blocks, Adam lr=.01, lambda=.01,
1000 full-batch updates, CPU one thread, seed0, no input normalization. Final
encode stays in train mode (BN batch statistics), after the last update.

AUTHOR-DATA: all rows, raw TTK-selected CSV columns. 3Clusters selects x,y,z by
header (never ClusterId); headerless Twist/K4 select first three columns without
losing row0; COIL20-1 selects headers '1'..'1024' and excludes constant '1025'.
Digits is separately sklearn's raw 1797x64 data, NOT an author manuscript case.
PCA(n_components=2,random_state=0); UMAP(n_components=2,n_neighbors=15,
min_dist=.1,random_state=0), other library defaults; GraphEmbedding(seed=0),
current production defaults. Author-data Digits is NOT the matched public figure.

DIGITS-MATCHED: Digits only, no AUTHOR; separate split (default_rng(0) permutation,
first1400 fit/remaining397 query) and transductive all1797 runs. PCA full SVD;
GraphEmbedding(seed=0), requiring current exact-neighbor defaults; genuine UMAP
uses exact15 nonself neighbors plus self (n_neighbors=16), 300 epochs, 5 negatives,
spectral init, min_dist=.1, spread=1, lr=1, seeds0, n_jobs=1, forced approximation,
precomputed_knn with a separately prepared genuine NNDescent31 index. Queries
transform against fitted rows only. No labels enter fits. Current Graph defaults
and source hashes are recorded: a future core change is a NEW run, not replacement
of the frozen historical evidence. This runner emits coordinates, not metrics or
figures, and makes no bitwise historical equivalence promise across dependencies.

Each PCA/UMAP/Graph case runs in a fresh --internal-worker file subprocess. Torch
imports are forbidden; optional TensorFlow imports raise ImportError explicitly
(umap's optional parametric backend is disabled). Never Torch/Numba co-import.
Registration freezes configuration, source/dependency hashes, selected inputs and
row IDs BEFORE fit-child invocation. Runtime logs may contain local paths: NEVER
publish this scratch directory. Public script contains no private source paths.

Test helpers: parse_author_csv(text,dataset) -> float64 ndarray;
sha256(path_or_bytes) -> hex digest; validate_adapter_manifest(dict) -> dict;
validate_protocol(event_log_text) -> {'updates':1000,'output_mode':'train'}.
All validators raise ValueError on contract violations (I/O may raise OSError).
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True

UPSTREAM_PIN = "03f941d97305dbc045af7f595b6c5a3388808af2"
DIGITS_SOURCE_SHA256 = "09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22"
REPO = Path(__file__).resolve().parents[2]
DATASETS = ("Twist", "K4", "COIL20-1", "3Clusters", "Digits")
METHODS = ("PCA", "UMAP", "GraphEmbedding", "AUTHOR")
CASE_TIMEOUT = 150
TOTAL_TIMEOUT = 900
DATA_SHA256 = {
    "Twist": "7bdc03989481bc942496072bfaf06a867f42e5e2844e0346057c65e4c31c1801",
    "K4": "7ec67006e8077dfcf9ba0d3188aaf942693d1e623ecc96477f6f8d4b5ecbc27f",
    "COIL20-1": "b8232ff169ec39d6f0d816dc9ab04bda576f556fbd54ebc06a6765be40545234",
    "3Clusters": "0763e181ec10f1c0188156efecda7a14912e2906f19f17890ea2ab5c8828c4a5",
}
# Raw selected little-endian float64 bytes, not an archive/container hash.
X_SHA256 = {
    "Twist": "5e4a0bb37be8ab1284740073cbea3aa69ca8b16eb12ecd9f33a36068bc77060d",
    "K4": "622ca9abd94a2362ae1cc5f3b203449848aef57c97bcf8d47205a0090d5b7e81",
    "COIL20-1": "20e4ff9d34ef825893d6ed212fc213a31a0bc9f5930d09bd041cdc16142ad5e2",
    "3Clusters": "69157d6d748444bb21ffc3ee572cded325c5697fad7d685d2a75bb6c116d53b9",
    "Digits": "20def7f70a702f0af9732fbba4375e147a7d54fe70d8c45569b8e7c1c7010c10",
}
CORE_PREFIX = "ttk-tcdr/core/base/topologicallyConstrainedDimensionReduction/"
UPSTREAM_SOURCE_SHA256 = {
    "scripts/Compute.py": "333538490753ba310b6b43eb33f35df2b490f02b5ac8362baca8c14795677d8c",
    "ttk-tcdr/paraview/xmls/DimensionReduction.xml": "1cbeb7d28cb200cb1ec777ae6d16e4baf863b592ca292600c086cf2bf87a4002",
    CORE_PREFIX + "DimensionReductionModel.cpp": "fea4dcc176caa7f6b11cbec147b9664d1906928ff75dea8a28ab1076039179e6",
    CORE_PREFIX + "DimensionReductionModel.h": "4dcc9d6cfc1d769e027ac579b5d88c092ccf157b561c542af725f6c114afa7b8",
    CORE_PREFIX + "TopologicalLoss.cpp": "3713193bb142a38bad4e93687b141ecc5dcab661623ff7fe6b3a418d5f13c5d5",
    CORE_PREFIX + "TopologicalLoss.h": "5e94fbaaeeb99e3436c076645ce7f8fb6e997ee50c7098a5861ece3813cb0c0f",
    CORE_PREFIX + "TopologicallyConstrainedDimensionReduction.cpp": "7c30f98f84582b099f9c77988f3c47a21c71972e8688b6ac94f20b415977ba92",
    CORE_PREFIX + "TopologicallyConstrainedDimensionReduction.h": "634f3357e58e6a42f5b8eb59aac8a28a892a3be101309d2b9c72fef20bfb5547",
}
ADAPTER_CONTRACT = {
    "schema": "visual-contracts-cpu-adapter-v1", "upstream_commit": UPSTREAM_PIN,
    "generic_dimension": True, "standalone_native": True, "device": "cpu",
    "threads": 1, "interop_threads": 1, "seed": 0,
    "input_format": "TDAF64LE", "output_format": "raw-f32le-n-by-2",
    "source_ph_dtype": "float64", "network_dtype": "float32",
    "input_normalization": "none", "hidden_dimensions": [128, 32],
    "output_dimensions": 2, "activation": "ReLU",
    "batch_norm": True, "batch_norm_eps": 1e-5, "batch_norm_momentum": 0.1,
    "batch_norm_affine": True, "batch_norm_track_running_stats": True,
    "training_mode": "train", "output_mode": "train", "full_batch": True,
    "optimizer": "Adam", "learning_rate": 0.01, "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8, "weight_decay": 0, "amsgrad": False,
    "updates": 1000, "lambda": 0.01, "reconstruction": "mean_mse",
    "topology": "ASYMMETRIC_CASCADE", "topology_implementation": "upstream-unmodified",
    "model_implementation": "upstream-unmodified", "progress": "EVENT-step-0..999-complete",
}
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def sha256(path_or_bytes):
    """Hash bytes directly or stream a filesystem path; never interpret text as data."""
    digest = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        digest.update(path_or_bytes)
    else:
        with Path(path_or_bytes).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def parse_author_csv(text, dataset):
    """Return ALL selected feature rows in source order, excluding label columns."""
    import numpy as np
    if dataset not in DATA_SHA256:
        raise ValueError("CSV dataset must be Twist, K4, COIL20-1 or 3Clusters")
    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error as exc:
        raise ValueError("malformed CSV") from exc
    if not rows or any(not row for row in rows):
        raise ValueError("empty CSV or blank data row")
    if dataset in ("Twist", "K4"):
        width = len(rows[0])
        if width < 3:
            raise ValueError("headerless CSV needs at least three columns")
        selected = [0, 1, 2]
    else:
        header = [cell.strip() for cell in rows.pop(0)]
        width = len(header)
        expected = (["ClusterId", "x", "y", "z"] if dataset == "3Clusters"
                    else [str(i) for i in range(1, 1026)])
        if len(set(header)) != width or set(header) != set(expected):
            raise ValueError("unexpected or duplicate CSV header columns")
        names = ["x", "y", "z"] if dataset == "3Clusters" else [str(i) for i in range(1, 1025)]
        selected = [header.index(name) for name in names]
    if not rows or any(len(row) != width for row in rows):
        raise ValueError("empty data or inconsistent CSV row width")
    try:
        raw = np.asarray(rows, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("CSV contains nonnumeric values") from exc
    if not np.isfinite(raw).all():
        raise ValueError("CSV contains nonfinite values")
    if dataset == "COIL20-1":
        excluded = raw[:, header.index("1025")]
        if not np.all(excluded == 1):
            raise ValueError("COIL excluded column 1025 must be constant 1")
    return np.ascontiguousarray(raw[:, selected], dtype="<f8")


def _same_value(actual, expected):
    # Reject True for integer 1, including nested fields.
    if isinstance(expected, bool) or isinstance(expected, int):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_value(a, b) for a, b in zip(actual, expected))
    return not isinstance(actual, bool) and actual == expected


def validate_adapter_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("adapter manifest must be a JSON object")
    for key, expected in ADAPTER_CONTRACT.items():
        if key not in manifest or not _same_value(manifest[key], expected):
            raise ValueError("adapter contract mismatch: " + key)
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("executable_sha256", ""))):
        raise ValueError("manifest requires lowercase executable_sha256")
    return manifest


def _strict_json(text):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number
    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=pairs)


def validate_protocol(text):
    """Validate complete native stdout. Non-EVENT diagnostic lines are allowed."""
    steps = 0
    complete = False
    for line in text.splitlines():
        if line.lstrip().startswith("EVENT") and line != line.lstrip():
            raise ValueError("EVENT prefix must start at column zero")
        if not line.startswith("EVENT"):
            continue
        if not line.startswith("EVENT "):
            raise ValueError("malformed EVENT prefix")
        event = _strict_json(line[6:])
        if not isinstance(event, dict) or complete:
            raise ValueError("invalid EVENT or EVENT after complete")
        kind = event.get("kind")
        if kind == "step":
            if type(event.get("step")) is not int or event["step"] != steps or steps >= 1000:
                raise ValueError("expected consecutive steps 0..999 exactly once")
            for key in ("reconstruction", "topology_raw", "total", "gradient_norm", "elapsed_seconds"):
                if key in event and (isinstance(event[key], bool) or not isinstance(event[key], (int, float))
                                     or not math.isfinite(event[key])):
                    raise ValueError("nonfinite/non-numeric progress: " + key)
            steps += 1
        elif kind == "complete":
            if steps != 1000 or type(event.get("updates")) is not int or event["updates"] != 1000:
                raise ValueError("partial progress is not success")
            if event.get("output_mode") != "train":
                raise ValueError("final encoding must use train mode")
            complete = True
        elif kind not in ("source_ph_start", "source_ph_ready") or steps:
            raise ValueError("unexpected EVENT kind/order")
    if not complete:
        raise ValueError("missing complete EVENT")
    return {"updates": steps, "output_mode": "train"}


def require_external(path, label):
    resolved = Path(path).expanduser().resolve()
    if resolved == REPO or REPO in resolved.parents:
        raise ValueError(label + " must be outside this repository")
    return resolved


def validate_output_path(path):
    raw = Path(path).expanduser()
    if raw.exists() or raw.is_symlink():
        raise ValueError("output already exists; no overwrites or retries")
    resolved = require_external(raw, "output")
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValueError("output must be non-repository scratch, not inside a Git checkout")
    return resolved


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def git(upstream, *args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.check_output(["git", "-C", str(upstream), *args], env=env,
                                   stderr=subprocess.PIPE, timeout=20).decode().strip()


def verify_upstream(upstream):
    if Path(git(upstream, "rev-parse", "--show-toplevel")).resolve() != upstream:
        raise ValueError("--upstream must be the author checkout root")
    if git(upstream, "rev-parse", "HEAD") != UPSTREAM_PIN:
        raise ValueError("upstream commit does not match pinned author commit")
    if git(upstream, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("upstream must be clean, including untracked files/submodules")
    for relative, expected in UPSTREAM_SOURCE_SHA256.items():
        if sha256(upstream / relative) != expected:
            raise ValueError("upstream source hash mismatch: " + relative)


def validate_native_adapter(path):
    if "coil64" in re.sub(r"[^a-z0-9]", "", path.name.lower()):
        raise ValueError("COIL64-specific executable is forbidden")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("adapter must be an executable regular file")
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic not in (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",
                     b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xce", b"\xca\xfe\xba\xbe",
                     b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"):
        raise ValueError("adapter must be a native ELF/Mach-O executable, not a Python/shell driver")


def install_import_guard():
    if "torch" in sys.modules or "tensorflow" in sys.modules:
        raise RuntimeError("Torch/TensorFlow preloaded: use a clean Python process")
    class Isolate(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".")[0]
            if root == "torch":
                raise RuntimeError("Torch forbidden in baseline worker")
            if root == "tensorflow":
                raise ImportError("optional TensorFlow disabled for genuine nonparametric UMAP")
            return None
    sys.meta_path.insert(0, Isolate())


def freeze_sources():
    """Source checkout plus installed control distributions, without importing them."""
    files = {Path(__file__).resolve()}
    for root in ("python/open_deep_tda", "src", "include", "bindings"):
        files.update(p for p in (REPO / root).rglob("*") if p.is_file()
                     and p.suffix in (".py", ".cpp", ".h", ".hpp", ".so", ".dylib", ".pyd"))
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "umap-learn", "pynndescent", "numba", "llvmlite"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
            continue
        versions[name] = dist.version
        for entry in dist.files or ():
            if str(entry).endswith((".py", ".so", ".pyd", ".dylib")):
                p = Path(dist.locate_file(entry)).resolve()
                if p.is_file():
                    files.add(p)
    # Absolute paths are intentionally LOCAL runtime metadata, never public assets.
    return {str(p): sha256(p) for p in sorted(files)}, versions


def graph_defaults():
    """Read literal public API defaults without importing any numerical backend."""
    tree = ast.parse((REPO / "python/open_deep_tda/graph_embedding.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GraphEmbedding")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    return {arg.arg: ast.literal_eval(value) for arg, value in zip(init.args.kwonlyargs, init.args.kw_defaults)}


def assert_frozen(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError("registered file changed: " + Path(path).name)


def write_native_input(path, X):
    import numpy as np
    X = np.ascontiguousarray(X, dtype="<f8")
    if X.ndim != 2 or not (2 <= len(X) <= 2000 and 1 <= X.shape[1] <= 4096):
        raise ValueError("input outside native generic dimension bounds")
    if not np.isfinite(X).all() or np.max(np.abs(X)) > np.finfo(np.float32).max:
        raise ValueError("input must be finite also after float32 conversion")
    with Path(path).open("xb") as stream:
        stream.write(b"TDAF64LE" + struct.pack("<QQ", *X.shape))
        stream.write(X.tobytes())


def read_native_output(path, n):
    import numpy as np
    if Path(path).stat().st_size != n * 2 * 4:
        raise ValueError("native output must contain exactly n*2 float32 values")
    Z = np.fromfile(path, dtype="<f4").reshape(n, 2)
    if not np.isfinite(Z).all():
        raise ValueError("nonfinite native output")
    return Z


def run_bounded(command, cwd, timeout):
    """POSIX process group termination also bounds Graph's nested fit workers."""
    if os.name != "posix":
        raise ValueError("POSIX process groups required for bounded child execution")
    env = dict(os.environ, **{key: "1" for key in THREAD_KEYS},
               PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(cwd / "numba-cache"),
               MPLCONFIGDIR=str(cwd / "mpl-cache"), TMPDIR=str(cwd))
    started = time.monotonic()
    with (cwd / "stdout.log").open("xb") as stdout, (cwd / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                 start_new_session=True)
        timed_out = False
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
        finally:
            # A trusted executable should not leave descendants; clean them too.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return {"returncode": code, "timed_out": timed_out,
            "wall_seconds": time.monotonic() - started, "timeout_seconds": timeout}


def internal_worker(case_path):
    install_import_guard()
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    sys.dont_write_bytecode = True
    import numpy as np
    case_path = require_external(case_path, "internal case")
    case = _strict_json(case_path.read_text())
    output = case_path.parent
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("internal worker output must be non-repository scratch")
    require_external(case["registration"], "registration")
    require_external(case["input"], "input")
    registration = _strict_json(Path(case["registration"]).read_text())
    if sha256(case["registration"]) != case["registration_sha256"]:
        raise ValueError("registration hash changed")
    assert_frozen(registration["frozen_files"])
    if sha256(case["input"]) != case["input_sha256"]:
        raise ValueError("input hash changed")
    with np.load(case["input"], allow_pickle=False) as data:
        F, Q = data["fit"], data["query"]
    before = sha256(F.tobytes() + Q.tobytes())
    method = case["method"]
    matched = registration["protocol"] == "digits-matched"
    sys.path.insert(0, str(REPO / "python"))
    details = {}
    if method == "PCA":
        from sklearn.decomposition import PCA
        model = PCA(n_components=2, svd_solver="full") if matched else PCA(n_components=2, random_state=0)
        Z = model.fit_transform(F)
    elif method == "GraphEmbedding":
        from open_deep_tda import GraphEmbedding
        model = GraphEmbedding(seed=0)
        if matched and (model._config["neighbor_backend"] != "exact" or model._config["epochs"] != 300
                        or model._config["negative_rate"] != 5):
            raise ValueError("current Graph defaults no longer match historical matched protocol")
        expected = {k: v for k, v in registration["graph_defaults"].items() if k not in ("work_dir", "debug_dir")}
        if model._config != expected:
            raise ValueError("Graph defaults differ from preregistered configuration")
        details["graph_configuration"] = model._config
        Z = model.fit_transform(F)
    elif method == "UMAP":
        from umap import UMAP
        if matched:
            from pynndescent import NNDescent
            from open_deep_tda._graph_worker import knn
            ids, distances = knn(F, F, 15, np.arange(len(F)))
            indices = np.column_stack((np.arange(len(F)), ids))
            distances = np.column_stack((np.zeros(len(F)), distances))
            index = NNDescent(F.astype(np.float32), n_neighbors=31, metric="euclidean",
                              random_state=0, n_jobs=1, low_memory=True, compressed=False,
                              parallel_batch_queries=False)
            index.prepare()
            model = UMAP(n_neighbors=16, n_components=2, metric="euclidean", min_dist=.1,
                         spread=1., n_epochs=300, negative_sample_rate=5, learning_rate=1.,
                         init="spectral", random_state=0, transform_seed=0, n_jobs=1,
                         force_approximation_algorithm=True,
                         precomputed_knn=(indices, distances, index))
            model.fit(F)
            if not np.array_equal(model._knn_indices, indices):
                raise ValueError("UMAP did not retain supplied exact source neighbors")
            Z = model.embedding_
            details["exact15_plus_self"] = True
            details["prepared_NNDescent_neighbors"] = 31
        else:
            model = UMAP(n_components=2, n_neighbors=15, min_dist=.1, random_state=0)
            Z = model.fit_transform(F)
    else:
        raise ValueError("invalid internal method")
    Y = model.transform(Q) if len(Q) else np.empty((0, 2), dtype=Z.dtype)
    if Z.shape != (len(F), 2) or Y.shape != (len(Q), 2) or not np.isfinite(Z).all() or not np.isfinite(Y).all():
        raise ValueError("invalid control coordinates")
    if before != sha256(F.tobytes() + Q.tobytes()) or "torch" in sys.modules:
        raise ValueError("mutated input or forbidden Torch import")
    target = output / "coordinates.npz"
    with target.open("xb") as stream:
        np.savez(stream, fit=Z, query=Y)
    details.update(status="completed", output_sha256=sha256(target), torch_loaded=False,
                   numba_loaded="numba" in sys.modules, fit_shape=list(Z.shape), query_shape=list(Y.shape))
    dump_new(output / "worker-result.json", details)
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Adapter manifest template (replace hash):\n" + json.dumps(
                                    dict(ADAPTER_CONTRACT, executable_sha256="<64 lowercase hex characters>"), indent=2))
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS[:4]))
    p.add_argument("--protocol", choices=("author-data", "digits-matched"), default="author-data")
    p.add_argument("--methods", nargs="+", choices=METHODS)
    p.add_argument("--cpu-adapter", type=Path)
    p.add_argument("--adapter-sha256")
    p.add_argument("--adapter-manifest", type=Path)
    p.add_argument("--check-only", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if len(set(args.dataset)) != len(args.dataset):
        raise ValueError("duplicate datasets would constitute retries")
    methods = args.methods or list(METHODS[:3] if args.protocol == "digits-matched" else METHODS)
    if len(set(methods)) != len(methods):
        raise ValueError("duplicate methods would constitute retries")
    if args.protocol == "digits-matched" and (args.dataset != ["Digits"] or "AUTHOR" in methods):
        raise ValueError("digits-matched requires --dataset Digits and excludes AUTHOR")
    output = validate_output_path(args.output)
    upstream = require_external(args.upstream, "upstream")
    data_dir = require_external(args.data_dir or upstream / "data", "data-dir")
    if upstream == output or upstream in output.parents:
        raise ValueError("output must not modify upstream")
    verify_upstream(upstream)
    frozen, versions = freeze_sources()
    frozen.update({str(upstream / p): h for p, h in UPSTREAM_SOURCE_SHA256.items()})
    adapter = None
    manifest = None
    if "AUTHOR" in methods:
        if not all((args.cpu_adapter, args.adapter_sha256, args.adapter_manifest)):
            raise ValueError("AUTHOR requires --cpu-adapter, --adapter-sha256 and --adapter-manifest")
        adapter = require_external(args.cpu_adapter, "cpu-adapter")
        manifest_path = require_external(args.adapter_manifest, "adapter-manifest")
        validate_native_adapter(adapter)
        manifest = validate_adapter_manifest(_strict_json(manifest_path.read_text()))
        if manifest["executable_sha256"] != args.adapter_sha256 or sha256(adapter) != args.adapter_sha256:
            raise ValueError("adapter executable/manifest/CLI SHA256 mismatch")
        frozen[str(adapter)] = args.adapter_sha256
        frozen[str(manifest_path)] = sha256(manifest_path)
    elif any((args.cpu_adapter, args.adapter_sha256, args.adapter_manifest)):
        raise ValueError("adapter arguments supplied without selecting AUTHOR")
    install_import_guard()
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    import numpy as np
    prepared = []
    for dataset in args.dataset:
        if dataset == "Digits":
            bundle = Path(importlib.metadata.distribution("scikit-learn").locate_file("sklearn/datasets/data/digits.csv.gz")).resolve()
            if sha256(bundle) != DIGITS_SOURCE_SHA256:
                raise ValueError("recorded bundled Digits source hash mismatch")
            frozen[str(bundle)] = DIGITS_SOURCE_SHA256
            from sklearn.datasets import load_digits
            X = np.ascontiguousarray(load_digits().data, dtype="<f8")
            source = "sklearn.datasets.load_digits (bundled; no download)"
        else:
            source = require_external(data_dir / (dataset + ".csv"), "CSV")
            raw = source.read_bytes()
            if sha256(raw) != DATA_SHA256[dataset]:
                raise ValueError("recorded source CSV hash mismatch: " + dataset)
            X = parse_author_csv(raw.decode("utf-8"), dataset)
            frozen[str(source)] = DATA_SHA256[dataset]
            source = source.name
        if sha256(X.tobytes()) != X_SHA256[dataset]:
            raise ValueError("recorded raw feature hash mismatch: " + dataset)
        ids = np.arange(len(X), dtype=np.int64)
        cohorts = [("all", ids, ids[:0])]
        if args.protocol == "digits-matched":
            order = np.random.default_rng(0).permutation(len(X))
            cohorts = [("split", order[:1400], order[1400:]), ("transductive", ids, ids[:0])]
        for cohort, fit_ids, query_ids in cohorts:
            prepared.append((dataset, cohort, X, fit_ids, query_ids, source))
    # No raw output exists until ALL externally supplied inputs have passed checks.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    inputs = []
    for dataset, cohort, X, fit_ids, query_ids, source in prepared:
        case_dir = output / (dataset + "-" + cohort)
        case_dir.mkdir()
        path = case_dir / "input.npz"
        with path.open("xb") as stream:
            np.savez(stream, fit=X[fit_ids], query=X[query_ids], fit_ids=fit_ids, query_ids=query_ids)
        item = dict(dataset=dataset, cohort=cohort, source=source, input=str(path),
                    input_sha256=sha256(path), raw_X_sha256=sha256(X.tobytes()),
                    fit_shape=[len(fit_ids), X.shape[1]], query_shape=[len(query_ids), X.shape[1]],
                    fit_ids_sha256=sha256(fit_ids.astype("<i8").tobytes()),
                    query_ids_sha256=sha256(query_ids.astype("<i8").tobytes()))
        frozen[str(path)] = item["input_sha256"]
        if "AUTHOR" in methods:
            native = case_dir / "input.f64"
            write_native_input(native, X[fit_ids])
            item["native_input"] = str(native)
            item["native_sha256"] = sha256(native)
            frozen[str(native)] = item["native_sha256"]
        inputs.append(item)
    registration = dict(schema="visual-contracts-reproduction-v1", protocol=args.protocol,
                        seed=0, datasets=args.dataset, methods=methods, check_only=args.check_only,
                        upstream_commit=UPSTREAM_PIN, upstream_clean=True, inputs=inputs,
                        adapter_manifest=manifest, frozen_files=frozen, dependency_versions=versions,
                        python_version=sys.version, case_timeout_seconds=CASE_TIMEOUT,
                        total_child_budget_seconds=TOTAL_TIMEOUT, retries=0, preprocessing="none",
                        labels_used=False, graph_defaults=graph_defaults(), graph_policy="current GraphEmbedding(seed=0) defaults; source-bound new run",
                        historical_equivalence="not guaranteed; dependency/core changes do not replace historical evidence",
                        control_configuration={"PCA": "full SVD" if args.protocol == "digits-matched" else "n_components=2,random_state=0",
                                               "UMAP": "exact15+self,16 neighbors,300 epochs,NNDescent31,forced approximation,seeds0" if args.protocol == "digits-matched" else "n_components=2,n_neighbors=15,min_dist=.1,random_state=0; library defaults",
                                               "GraphEmbedding": "seed=0; current production defaults"})
    registration_path = output / "registration.json"
    dump_new(registration_path, registration)
    registration_hash = sha256(registration_path)
    if args.check_only:
        assert_frozen(frozen)
        verify_upstream(upstream)
        dump_new(output / "check-result.json", dict(status="validated-no-fits", registration_sha256=registration_hash,
                 adapter_execution_verified=False, protocol="declared contract only; no EVENT stream executed"))
        print("Validated files and declared protocol; no adapter or fit executed.")
        return 0
    results = []
    used = 0.0
    for item in inputs:
        for method in methods:
            folder = Path(item["input"]).parent / method
            folder.mkdir()
            result = dict(dataset=item["dataset"], cohort=item["cohort"], method=method,
                          attempts=0, status="failed", registration_sha256=registration_hash)
            try:
                assert_frozen(frozen)
                if sha256(registration_path) != registration_hash:
                    raise ValueError("registration changed")
                remaining = TOTAL_TIMEOUT - used
                if remaining <= 0:
                    result["status"] = "not-run-budget-exhausted"
                    raise ValueError("total child wall budget exhausted")
                case = dict(item, method=method, registration=str(registration_path),
                            registration_sha256=registration_hash)
                case_path = folder / "case.json"
                dump_new(case_path, case)
                command = ([str(adapter), item["native_input"], str(folder / "output.f32")] if method == "AUTHOR"
                           else [sys.executable, "-B", str(Path(__file__).resolve()), "--internal-worker", str(case_path)])
                result["attempts"] = 1
                process = run_bounded(command, folder, min(CASE_TIMEOUT, remaining))
                used += process["wall_seconds"]
                result.update(process)
                if process["timed_out"]:
                    result["status"] = "timeout"
                    raise ValueError("child timeout; partial output is not success")
                if process["returncode"] != 0:
                    raise ValueError("child failed; see local stderr.log")
                if method == "AUTHOR":
                    result["progress"] = validate_protocol((folder / "stdout.log").read_text())
                    Z = read_native_output(folder / "output.f32", item["fit_shape"][0])
                    with (folder / "coordinates.npz").open("xb") as stream:
                        np.savez(stream, fit=Z, query=np.empty((0, 2), dtype=np.float32))
                else:
                    details = _strict_json((folder / "worker-result.json").read_text())
                    if details["status"] != "completed" or details["output_sha256"] != sha256(folder / "coordinates.npz"):
                        raise ValueError("worker success/output hash mismatch")
                    with np.load(folder / "coordinates.npz", allow_pickle=False) as coordinates:
                        for key, shape in (("fit", item["fit_shape"]), ("query", item["query_shape"])):
                            Z = coordinates[key]
                            if Z.shape != (shape[0], 2) or not np.isfinite(Z).all():
                                raise ValueError("invalid worker output")
                assert_frozen(frozen)
                result.update(status="completed", coordinates="coordinates.npz",
                              output_sha256=sha256(folder / "coordinates.npz"))
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                result["error"] = str(exc)
            dump_new(folder / "result.json", result)
            results.append(result)
    final = dict(status="completed" if all(r["status"] == "completed" for r in results) else "incomplete",
                 results=results, total_child_wall_seconds=used, registration_sha256=registration_hash)
    try:
        verify_upstream(upstream)
        assert_frozen(frozen)
        final["provenance_unchanged"] = True
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        final.update(status="incomplete", provenance_unchanged=False, error=str(exc))
    dump_new(output / "results.json", final)
    print("Run " + final["status"] + "; raw local artifacts must not be published.")
    return 0 if final["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--internal-worker":
            raise SystemExit(internal_worker(sys.argv[2]))
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        raise SystemExit(2)
