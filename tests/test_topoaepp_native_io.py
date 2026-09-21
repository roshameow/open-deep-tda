"""I/O contract tests for the independent native COIL adapter, not production."""
import importlib.util
import hashlib
import json
import os
import subprocess
import struct
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "coil_native_io", Path(__file__).resolve().parents[1] / "benchmarks/topoaepp_coil_native.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_shared_features_digest_and_no_label_angle_materialization(tmp_path):
    train = np.zeros((960, 64), dtype=np.float32)
    test = np.zeros((480, 64), dtype=np.float32)
    archive = tmp_path / "features.npz"
    # Accessing these object arrays with allow_pickle=False would raise.
    np.savez(archive, train=train, test=test, labels=np.array([{}], dtype=object), angles=np.array([{}], dtype=object))
    digest = hashlib.sha256(train.tobytes() + test.tobytes()).hexdigest()
    protocol = tmp_path / "prepared.json"
    protocol.write_text(json.dumps({"feature_sha256": digest}))
    a, b, actual = MODULE.load_shared_features(archive, protocol)
    assert a.shape == train.shape and b.shape == test.shape and actual == digest
    protocol.write_text(json.dumps({"feature_sha256": "wrong"}))
    with pytest.raises(ValueError, match="digest mismatch"):
        MODULE.load_shared_features(archive, protocol)


def test_float32_row_major_roundtrip(tmp_path):
    x = np.arange(12, dtype="<f4").reshape(3, 4)
    path = tmp_path / "matrix.f32"
    MODULE.write_matrix(path, x)
    assert path.read_bytes()[:24] == struct.pack("<8sQQ", b"TDAF32LE", 3, 4)
    assert path.read_bytes()[24:] == x.tobytes(order="C")
    np.testing.assert_array_equal(MODULE.read_matrix(path, (3, 4)), x)


@pytest.mark.parametrize("x", [
    np.zeros((3, 4), dtype=np.float64),
    np.zeros((3, 4), dtype=">f4"),
    np.zeros((3, 4), dtype=np.float32, order="F"),
    np.array([[np.nan]], dtype=np.float32),
    np.array([[np.inf]], dtype=np.float32),
    np.zeros((0, 4), dtype=np.float32),
    np.zeros(4, dtype=np.float32),
])
def test_reject_invalid_input(tmp_path, x):
    with pytest.raises(ValueError):
        MODULE.write_matrix(tmp_path / "bad.f32", x)


@pytest.mark.parametrize("mutation", ["short_header", "short_payload", "extra_payload", "magic", "shape", "nonfinite"])
def test_reject_invalid_native_output(tmp_path, mutation):
    path = tmp_path / "matrix.f32"
    MODULE.write_matrix(path, np.zeros((3, 2), dtype=np.float32))
    raw = path.read_bytes()
    if mutation == "short_header":
        raw = raw[:12]
    elif mutation == "short_payload":
        raw = raw[:-1]
    elif mutation == "extra_payload":
        raw += b"x"
    elif mutation == "magic":
        raw = b"BROKEN!!" + raw[8:]
    elif mutation == "shape":
        raw = struct.pack("<8sQQ", b"TDAF32LE", 2, 3) + raw[24:]
    else:
        raw = raw[:24] + np.full((3, 2), np.nan, dtype="<f4").tobytes()
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        MODULE.read_matrix(path, (3, 2))


@pytest.mark.parametrize("mutation", ["truncated", "trailing", "wrong_dimension", "float64_payload", "nonfinite"])
def test_actual_cpp_rejects_malformed_input(tmp_path, mutation):
    executable = os.environ.get("TOPOAEPP_NATIVE_EXE")
    if not executable:
        pytest.skip("set TOPOAEPP_NATIVE_EXE to the built independent native driver")
    path = tmp_path / "bad.f32"
    x = np.zeros((72, 64), dtype="<f4")
    MODULE.write_matrix(path, x)
    raw = path.read_bytes()
    if mutation == "truncated":
        raw = raw[:-4]
    elif mutation == "trailing":
        raw += b"extra"
    elif mutation == "wrong_dimension":
        raw = struct.pack("<8sQQ", b"TDAF32LE", 72, 63) + raw[24:]
    elif mutation == "float64_payload":
        raw = raw[:24] + x.astype("<f8").tobytes()
    else:
        x[0, 0] = np.nan
        raw = raw[:24] + x.tobytes()
    path.write_bytes(raw)
    process = subprocess.run([executable, str(path), str(tmp_path / "unused.f32"),
                              str(tmp_path), "5", "0"], capture_output=True, text=True, timeout=10)
    assert process.returncode == 1
    assert "ERROR " in process.stderr
    assert "source_ph_start" not in process.stdout
