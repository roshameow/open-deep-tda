"""Synthetic, offline checks for the frozen all-row structural replay protocol.

Keep replay mocked at the protocol boundary except for the public-API wiring
checks below: ordinary CI must not fit a graph or invoke optional Ripser.
Object-dtype annotation members are intentional tripwires for accidental reads.

Reusable regression methods: inspect the exclusive started record from inside
mock replay; simulate source-pin changes without touching production files;
and replace public numerical collaborators to test the real all-row plumbing.
The source H1 recheck is a separate call from target H1 verification. Preserve
both in call-count assertions. Restore thread environment changes per fixture.
"""
import ast
import builtins
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "validate_global_structural.py"


@pytest.fixture
def protocol(monkeypatch):
    # Production pins numerical thread counts; restore the caller's environment
    # after each case so these protocol tests cannot change neighboring tests.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, "1"))
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module("validate_global_structural")


@pytest.fixture
def inputs(tmp_path):
    X = np.arange(48, dtype=np.float64).reshape(16, 3) / 7
    archive = tmp_path / "private-training-cache.npz"
    np.savez(archive, train=X, test=np.full((19, 3), np.nan),
             train_labels=np.array([{"private_annotation": "must not load"}], dtype=object),
             sample_ids=np.array([object()], dtype=object))
    return X, archive, tmp_path / "registration.json", tmp_path / "result.json"


def _read(path):
    return json.loads(path.read_text())


def _freeze(protocol, inputs):
    X, archive, registration, output = inputs
    protocol.preregister(archive, "train", registration, output)
    return _read(registration)


def _forbid_compute(monkeypatch, protocol):
    def forbidden(*args, **kwargs):
        pytest.fail("replay must not run before protocol validation")
    monkeypatch.setattr(protocol, "_compute_replay", forbidden)


def _assert_private_free(document, *private_values):
    text = json.dumps(document, allow_nan=False)
    for value in private_values:
        assert str(value) not in text
    forbidden_keys = {"coordinates", "embedding", "sample_ids", "query_ids",
                      "row_ids", "raw_logs", "traceback", "input_path", "root"}
    def visit(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(document)


def test_preregistration_never_computes_or_reads_annotations(protocol, inputs, monkeypatch):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    document = _freeze(protocol, inputs)
    assert not output.exists()
    _assert_private_free(document, archive, archive.parent, ROOT)
    with np.load(archive, allow_pickle=False) as data:
        with pytest.raises(ValueError, match="Object arrays"):
            data["train_labels"]
    assert registration.is_file()


@pytest.mark.parametrize("shape", [(15, 2), (1025, 2), (16,), (16, 2, 1), (16, 0)])
def test_invalid_shape_rejected_before_replay(protocol, tmp_path, monkeypatch, shape):
    _forbid_compute(monkeypatch, protocol)
    archive = tmp_path / "bad.npz"
    np.savez(archive, train=np.zeros(shape, dtype=np.float64))
    registration, output = tmp_path / "registration.json", tmp_path / "result.json"
    with pytest.raises(ValueError):
        protocol.preregister(archive, "train", registration, output)
    assert not registration.exists()
    assert not output.exists()


@pytest.mark.parametrize("dtype", [np.bool_, np.complex128, object, "U3"])
def test_unsafe_array_dtypes_rejected_before_replay(protocol, tmp_path, monkeypatch, dtype):
    _forbid_compute(monkeypatch, protocol)
    archive = tmp_path / "bad.npz"
    np.savez(archive, train=np.ones((16, 2), dtype=dtype))
    with pytest.raises(ValueError):
        protocol.preregister(archive, "train", tmp_path / "reg.json", tmp_path / "out.json")
    assert not (tmp_path / "reg.json").exists()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_features_rejected(protocol, tmp_path, monkeypatch, nonfinite):
    _forbid_compute(monkeypatch, protocol)
    X = np.zeros((16, 2), dtype=np.float64)
    X[-1, -1] = nonfinite
    archive = tmp_path / "bad.npz"
    np.savez(archive, train=X)
    with pytest.raises(ValueError):
        protocol.preregister(archive, "train", tmp_path / "reg.json", tmp_path / "out.json")
    assert not (tmp_path / "reg.json").exists()


@pytest.mark.parametrize("rows", [16, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_row_boundaries_and_float_types_preregister(protocol, tmp_path, monkeypatch, rows, dtype):
    _forbid_compute(monkeypatch, protocol)
    archive = tmp_path / "valid.npz"
    np.savez(archive, train=np.zeros((rows, 2), dtype=dtype))
    registration, output = tmp_path / "reg.json", tmp_path / "out.json"
    protocol.preregister(archive, "train", registration, output)
    assert registration.is_file()
    assert not output.exists()


@pytest.mark.parametrize("member", ["", None, "test", "train_labels", "sample_ids", "missing"])
def test_explicit_training_member_required(protocol, inputs, monkeypatch, member):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    with pytest.raises((ValueError, TypeError)):
        protocol.preregister(archive, member, registration, output)
    assert not registration.exists()
    assert not output.exists()


def test_cli_requires_explicit_member(protocol, inputs, monkeypatch, capsys):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    with pytest.raises(SystemExit) as error:
        protocol.main(["--preregister", "--input", str(archive),
                       "--registration", str(registration), "--output", str(output)])
    assert error.value.code == 2
    assert "member" in capsys.readouterr().err
    assert not registration.exists()


def test_registration_is_exclusive(protocol, inputs, monkeypatch):
    _forbid_compute(monkeypatch, protocol)
    _freeze(protocol, inputs)
    _, archive, registration, output = inputs
    before = registration.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        protocol.preregister(archive, "train", registration, output)
    assert registration.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("stage", ["preregister", "run_registered"])
def test_existing_output_never_overwritten(protocol, inputs, monkeypatch, stage):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    if stage == "run_registered":
        _freeze(protocol, inputs)
    sentinel = b"existing private result: do not truncate\n"
    output.write_bytes(sentinel)
    with pytest.raises((FileExistsError, ValueError)):
        if stage == "preregister":
            protocol.preregister(archive, "train", registration, output)
        else:
            protocol.run_registered(archive, registration, output)
    assert output.read_bytes() == sentinel


def test_run_requires_registration(protocol, inputs, monkeypatch):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    with pytest.raises((FileNotFoundError, ValueError)):
        protocol.run_registered(archive, registration, output)
    assert not output.exists()


def test_changed_input_archive_rejected_before_replay(protocol, inputs, monkeypatch):
    _forbid_compute(monkeypatch, protocol)
    _freeze(protocol, inputs)
    X, archive, registration, output = inputs
    np.savez(archive, train=X + 1)
    with pytest.raises(ValueError):
        protocol.run_registered(archive, registration, output)
    assert not output.exists()


def test_changed_unread_annotation_also_changes_archive_pin(protocol, inputs, monkeypatch):
    _forbid_compute(monkeypatch, protocol)
    _freeze(protocol, inputs)
    _, archive, registration, output = inputs
    with zipfile.ZipFile(archive, "a") as data:
        data.comment = b"archive changed without changing training rows"
    with pytest.raises(ValueError):
        protocol.run_registered(archive, registration, output)
    assert not output.exists()


def test_module_import_is_stdlib_only(monkeypatch):
    original = builtins.__import__
    forbidden = {"numpy", "scipy", "sklearn", "torch", "numba", "ripser", "open_deep_tda"}
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in forbidden, "numerical imports must remain lazy"
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    namespace = runpy.run_path(str(SCRIPT), run_name="_protocol_import_test")
    assert callable(namespace["preregister"])
    assert callable(namespace["run_registered"])
    assert namespace["ROOT"] == ROOT


def test_production_imports_are_public_and_offline():
    tree = ast.parse(SCRIPT.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "protocol must use explicit public imports"
            imported.add(node.module)
    assert not any(name.split(".")[0] in {
        "requests", "urllib", "http", "socket", "subprocess", "runpy",
        "docs", "outputs", "benchmarks",
    } for name in imported)
    numerical = {name for name in imported if name.startswith("open_deep_tda")}
    assert numerical
    for name in numerical:
        assert not any(part.startswith("_") for part in name.split("."))
        relative = Path("python", *name.split("."))
        assert (ROOT / relative.with_suffix(".py")).is_file() or (ROOT / relative / "__init__.py").is_file()
    # Loading code by a filename, exec or eval would bypass the public imports.
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"exec", "eval", "__import__"}
                   for node in ast.walk(tree))


def test_only_explicit_train_member_is_opened(protocol, inputs, monkeypatch):
    original = zipfile.ZipFile.open
    seen = []
    def guarded(archive, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        seen.append(name)
        assert name == "train.npy", "TEST, labels, and IDs must never be parsed"
        return original(archive, member, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", guarded)
    _forbid_compute(monkeypatch, protocol)
    _freeze(protocol, inputs)
    assert seen and set(seen) == {"train.npy"}
    seen.clear()
    monkeypatch.setattr(protocol, "_compute_replay", lambda X, reg: {"status": "certified"})
    _, archive, registration, output = inputs
    assert protocol.run_registered(archive, registration, output)["status"] == "certified"
    assert seen and set(seen) == {"train.npy"}


@pytest.mark.parametrize("collision", ["registration-input", "output-input", "output-registration"])
def test_input_registration_output_paths_must_be_distinct(protocol, inputs, monkeypatch, collision):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    original = archive.read_bytes()
    if collision == "registration-input":
        registration = archive
    elif collision == "output-input":
        output = archive
    else:
        output = registration
    with pytest.raises((ValueError, FileExistsError)):
        protocol.preregister(archive, "train", registration, output)
    assert archive.read_bytes() == original


def _string_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _string_values(child)
    elif isinstance(value, str):
        yield value


def test_registration_pins_archive_member_and_public_sources(protocol, inputs, monkeypatch):
    _forbid_compute(monkeypatch, protocol)
    registration = _freeze(protocol, inputs)
    _, archive, _, _ = inputs
    values = set(_string_values(registration))
    assert hashlib.sha256(archive.read_bytes()).hexdigest() in values
    with zipfile.ZipFile(archive) as data:
        assert hashlib.sha256(data.read("train.npy")).hexdigest() in values
    for source in [SCRIPT, *[ROOT / "python" / "open_deep_tda" / name for name in (
        "graph_embedding.py", "_graph_worker.py", "structural_auto_witness.py",
        "structural_constructive.py", "structural_sparse_h1.py", "structural_planar.py",
    )]]:
        assert hashlib.sha256(source.read_bytes()).hexdigest() in values, source.name


@pytest.mark.parametrize("member", ["../train", "/train", "nested/train", "train.npy"])
def test_member_is_an_explicit_npz_key_not_a_path(protocol, inputs, monkeypatch, member):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    with pytest.raises(ValueError):
        protocol.preregister(archive, member, registration, output)
    assert not registration.exists()


def test_corrupt_npz_rejected_without_frozen_artifacts(protocol, inputs, monkeypatch):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    archive.write_bytes(b"not a zip archive")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        protocol.preregister(archive, "train", registration, output)
    assert not registration.exists()
    assert not output.exists()


@pytest.mark.parametrize("arguments", [[], ["--preregister", "--run"], ["--run"],
                                       ["--preregister", "--member", "train"]])
def test_cli_rejects_missing_or_conflicting_protocol_arguments(protocol, arguments, capsys):
    with pytest.raises(SystemExit) as error:
        protocol.main(arguments)
    assert error.value.code == 2
    assert "error" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["registration", "output"])
def test_existing_symlink_destination_is_not_followed(protocol, inputs, monkeypatch, target):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    victim = archive.parent / "keep.json"
    original = b'{"keep": true}\n'
    victim.write_bytes(original)
    (registration if target == "registration" else output).symlink_to(victim)
    with pytest.raises((FileExistsError, ValueError)):
        protocol.preregister(archive, "train", registration, output)
    assert victim.read_bytes() == original


def test_missing_member_is_not_inferred_even_for_single_array(protocol, tmp_path, monkeypatch):
    _forbid_compute(monkeypatch, protocol)
    archive = tmp_path / "one.npz"
    np.savez(archive, train=np.zeros((16, 2), dtype=np.float64))
    with pytest.raises((TypeError, ValueError)):
        protocol.preregister(archive, None, tmp_path / "reg.json", tmp_path / "out.json")
    assert not (tmp_path / "reg.json").exists()


@pytest.mark.parametrize("pin", ["source", "member", "settings", "runtime"])
def test_mismatched_frozen_contract_rejected_before_compute(protocol, inputs, monkeypatch, pin):
    document = _freeze(protocol, inputs)
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    if pin == "source":
        # Simulate a changed installed source without editing production files.
        current = dict(document["source_sha256"])
        current["python/open_deep_tda/structural_constructive.py"] = "0" * 64
        monkeypatch.setattr(protocol, "_source_hashes", lambda: current)
    else:
        if pin == "member":
            document["input"]["member_sha256"] = "0" * 64
        elif pin == "settings":
            document["settings"]["h0_tolerance"] = .1
        else:
            document["runtime"]["python"] = "0.0.0"
        registration.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        protocol.run_registered(archive, registration, output)
    assert not output.exists()


def test_freeze_and_exclusive_started_record_precede_compute(protocol, inputs, monkeypatch):
    frozen = _freeze(protocol, inputs)
    X, archive, registration, output = inputs
    frozen_bytes = registration.read_bytes()
    calls = []
    def replay(rows, record):
        calls.append(1)
        np.testing.assert_array_equal(rows, X)
        assert record == frozen
        assert registration.read_bytes() == frozen_bytes
        started = _read(output)
        assert started["status"] == "started"
        assert started["registration_sha256"] == hashlib.sha256(frozen_bytes).hexdigest()
        return {"status": "certified", "vertices": len(rows), "attempts": 1}
    monkeypatch.setattr(protocol, "_compute_replay", replay)
    report = protocol.run_registered(archive, registration, output)
    assert calls == [1]
    assert report == _read(output)
    assert report["status"] == "certified"
    assert report["input"]["shape"][0] == report["result"]["vertices"] == len(X)
    assert registration.read_bytes() == frozen_bytes
    _assert_private_free(report, archive, archive.parent, ROOT)
    with pytest.raises(ValueError):
        protocol.run_registered(archive, registration, output)
    assert calls == [1]


@pytest.mark.parametrize("failure", ["returned", "raised", "nonfinite"])
def test_failure_is_persisted_without_retry_or_discard(protocol, inputs, monkeypatch, capsys, failure):
    _freeze(protocol, inputs)
    _, archive, registration, output = inputs
    calls = []
    private = "/home/protocol-private/cache.npz coordinates=[[123,456]] sample_ids=[987]"
    def replay(rows, record):
        calls.append(1)
        print(private)
        if failure == "raised":
            raise RuntimeError(private)
        if failure == "nonfinite":
            return {"status": "certified", "h0_error": float("nan")}
        return {"status": "unsupported", "stage": "construction", "attempts": 1,
                "constructor_certified": False, "h0_error": .17}
    monkeypatch.setattr(protocol, "_compute_replay", replay)
    report = protocol.run_registered(archive, registration, output)
    assert calls == [1]
    assert report == _read(output)
    assert report["status"] == ("unsupported" if failure == "returned" else "failed")
    assert report["result"]["attempts"] == 1
    if failure == "returned":
        assert report["result"]["h0_error"] == .17
    if failure == "raised":
        assert report["result"]["error_type"] == "RuntimeError"
    _assert_private_free(report, private, archive, archive.parent, ROOT)
    captured = capsys.readouterr()
    assert private not in captured.out + captured.err
    before = output.read_bytes()
    with pytest.raises(ValueError):
        protocol.run_registered(archive, registration, output)
    assert output.read_bytes() == before
    assert calls == [1]


def test_identical_archive_can_move_without_recording_its_path(protocol, inputs, monkeypatch):
    _freeze(protocol, inputs)
    _, archive, registration, output = inputs
    moved = archive.with_name("relocated-secret-cache.npz")
    archive.rename(moved)
    monkeypatch.setattr(protocol, "_compute_replay", lambda X, reg: {"status": "certified"})
    report = protocol.run_registered(moved, registration, output)
    assert report["status"] == "certified"
    _assert_private_free(report, moved, archive, archive.parent, ROOT)


def test_output_binding_prevents_retry_with_a_different_filename(protocol, inputs, monkeypatch):
    _freeze(protocol, inputs)
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    replacement = output.with_name("second-attempt.json")
    with pytest.raises(ValueError):
        protocol.run_registered(archive, registration, replacement)
    assert not replacement.exists()
    assert not output.exists()


def test_cli_reports_scrubbed_runtime_failure(protocol, inputs, monkeypatch, capsys):
    _, archive, registration, output = inputs
    common = ["--input", str(archive), "--registration", str(registration), "--output", str(output)]
    assert protocol.main(["--preregister", "--member", "train", *common]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "preregistered"
    def fail(*args):
        raise RuntimeError(str(archive) + " coordinates=[[99,88]] sample_ids=[765]")
    monkeypatch.setattr(protocol, "_compute_replay", fail)
    assert protocol.main(["--run", *common]) == 1
    message = json.loads(capsys.readouterr().out)
    assert message["status"] == "failed"
    _assert_private_free(message, archive, "coordinates", "sample_ids")
    assert _read(output)["status"] == "failed"


def test_cli_refusal_does_not_echo_private_exception_paths(protocol, inputs, capsys):
    _, archive, registration, output = inputs
    code = protocol.main(["--run", "--input", str(archive),
                          "--registration", str(registration), "--output", str(output)])
    assert code == 2
    captured = capsys.readouterr()
    message = json.loads(captured.out)
    assert message["status"] == "refused"
    _assert_private_free(message, registration, archive, archive.parent, ROOT)
    assert not captured.err
    assert not output.exists()


@pytest.mark.parametrize("outcome", ["selection_refused", "construction_failed", "certified"])
def test_public_pipeline_uses_all_rows_fixed_policy_and_one_attempt(protocol, inputs, monkeypatch, outcome):
    """Exercise real replay plumbing, replacing only expensive public operations."""
    from open_deep_tda import graph_embedding, structural_auto_witness
    from open_deep_tda import structural_constructive, structural_planar, structural_sparse_h1

    _freeze(protocol, inputs)
    X, archive, registration, output = inputs
    n = len(X)
    calls = []
    private = "/home/protocol-private/raw.npz coordinates=[[12,34]] sample_ids=[567]"
    cycles = (((0, 1), (1, 2), (0, 2)),)
    work = dict.fromkeys(("vertices", "edges", "triangles", "basis_width", "words_per_vector",
                          "word_ops", "peak_storage_words", "boundary_rank", "selected_rank"), 1)
    work["vertices"] = n
    h1 = SimpleNamespace(certified=True, rank=1, work=work,
                         statuses=[SimpleNamespace(edges_present=True, survives=True)], reason=private)
    expected_D = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    expected_D /= np.median(expected_D[expected_D > 0])
    guide = np.column_stack((np.arange(n, dtype=float), np.sin(np.arange(n))))
    neighbors = np.array([[j for j in range(n) if j != i] for i in range(n)])

    def propose(D, *, allow_external):
        calls.append("selection")
        np.testing.assert_allclose(D, expected_D)
        assert D.shape == (n, n) and allow_external is True
        return SimpleNamespace(certified=outcome != "selection_refused", cycles=cycles,
                               birth=.2, survival=.8, source=h1,
                               diagnostics={"reason": private, "sample_ids": [567]})

    class Graph:
        def __init__(self, **kwargs):
            calls.append("graph_init")
            assert kwargs["neighbor_backend"] == "exact"
            assert kwargs["seed"] == 0 and kwargs["epochs"] == 300
            assert kwargs["negative_rate"] == 5
            assert kwargs["max_vertices"] == 1024
            assert "debug_dir" not in kwargs

        def fit_transform(self, rows):
            calls.append("graph_fit")
            np.testing.assert_array_equal(rows, X)
            return guide.copy()

        @property
        def source_knn_(self):
            return neighbors, np.ones_like(neighbors, dtype=float)

    def construct(D, points, selected, a, b, *, h0_tolerance):
        calls.append("construction")
        np.testing.assert_allclose(D, expected_D)
        assert points.shape == (n, 2)
        assert selected == cycles and (a, b) == (.2, .8)
        assert h0_tolerance == .05
        return SimpleNamespace(certified=outcome == "certified", embedding=points.copy(),
                               hole=np.zeros(2), h0_error=.17, reason=private,
                               diagnostics={"coordinates": points.tolist(), "sample_ids": [567],
                                            "inserted_vertices": n, "parent_tests": 0,
                                            "candidate_points": 0, "contact_pairs": 0})

    def analyze(D, selected, a, b):
        calls.append("target_h1" if "construction" in calls else "source_h1")
        assert D.shape == (n, n) and selected == cycles and (a, b) == (.2, .8)
        return h1

    def hierarchy(D):
        calls.append("full_h0")
        assert D.shape == (n, n)
        return np.zeros_like(D)

    def planar(points, selected, a, b, holes):
        calls.append("planar")
        assert points.shape == (n, 2) and selected == cycles and (a, b) == (.2, .8)
        return SimpleNamespace(certified=True, rank=1, work={"distance_pairs": n*(n-1)//2})

    def overlap(D, points):
        calls.append("overlap")
        assert D.shape == (n, n) and points.shape == (n, 2)
        return .5

    monkeypatch.setattr(graph_embedding, "GraphEmbedding", Graph)
    monkeypatch.setattr(structural_auto_witness, "auto_global_witnesses", propose)
    monkeypatch.setattr(structural_constructive, "construct_global_layout", construct)
    monkeypatch.setattr(structural_constructive, "full_hierarchy", hierarchy)
    monkeypatch.setattr(structural_sparse_h1, "analyze_sparse_h1", analyze)
    monkeypatch.setattr(structural_planar, "certify", planar)
    monkeypatch.setattr(protocol, "_overlap", overlap)
    report = protocol.run_registered(archive, registration, output)
    assert report == _read(output)
    result = report["result"]
    assert result["vertices"] == report["input"]["shape"][0] == n
    assert result["attempts"] == 1 and calls.count("selection") == 1
    _assert_private_free(report, private, archive, archive.parent, ROOT)
    if outcome == "selection_refused":
        assert calls == ["selection"]
        assert result["stage"] == "automatic_selection"
        assert report["status"] == "unsupported"
    else:
        for name in ("graph_init", "graph_fit", "construction"):
            assert calls.count(name) == 1
        assert calls.count("source_h1") == 1
        assert result["source"]["work"]["vertices"] == n
        if outcome == "construction_failed":
            assert report["status"] == "unsupported"
            assert result["constructor_certified"] is False
            assert result["stage"] == "construction" and result["h0_error"] == .17
            assert "target_h1" not in calls and "planar" not in calls
            assert calls.count("overlap") == 2
        else:
            assert report["status"] == "certified"
            assert result["h0"]["all_rows"] == result["planar"]["all_rows"] == n
            assert result["target"]["work"]["vertices"] == n
            assert calls.count("full_h0") == 2
            assert calls.count("target_h1") == calls.count("planar") == 1
            assert calls.count("overlap") == 3


def test_interrupted_output_is_retained_and_cannot_be_retried(protocol, inputs, monkeypatch):
    _freeze(protocol, inputs)
    _, archive, registration, output = inputs
    calls = []
    def interrupted(*args):
        calls.append(1)
        raise KeyboardInterrupt()
    monkeypatch.setattr(protocol, "_compute_replay", interrupted)
    with pytest.raises(KeyboardInterrupt):
        protocol.run_registered(archive, registration, output)
    assert _read(output)["status"] == "started"
    before = output.read_bytes()
    with pytest.raises(ValueError):
        protocol.run_registered(archive, registration, output)
    assert calls == [1]
    assert output.read_bytes() == before


@pytest.mark.parametrize("changed", ["source", "input"])
def test_changes_during_compute_are_recorded_as_failure(protocol, inputs, monkeypatch, changed):
    frozen = _freeze(protocol, inputs)
    _, archive, registration, output = inputs
    def replay(rows, record):
        if changed == "source":
            modified = dict(frozen["source_sha256"])
            modified[next(iter(modified))] = "0" * 64
            monkeypatch.setattr(protocol, "_source_hashes", lambda: modified)
        else:
            with zipfile.ZipFile(archive, "a") as data:
                data.comment = b"changed during compute"
        return {"status": "certified"}
    monkeypatch.setattr(protocol, "_compute_replay", replay)
    report = protocol.run_registered(archive, registration, output)
    assert report == _read(output)
    assert report["status"] == "failed"
    assert report["result"]["attempts"] == 1


@pytest.mark.parametrize("shape", [(16, 100_000), (1025, 2)])
def test_oversized_header_rejected_before_array_materialization(protocol, inputs, monkeypatch, shape):
    _, archive, registration, output = inputs
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        "descr": "<f8", "fortran_order": False, "shape": shape,
    })
    with zipfile.ZipFile(archive, "w") as data:
        data.writestr("train.npy", stream.getvalue())
    def forbidden(*args, **kwargs):
        pytest.fail("oversized header must be rejected before array materialization")
    monkeypatch.setattr(np, "load", forbidden)
    _forbid_compute(monkeypatch, protocol)
    with pytest.raises(ValueError):
        protocol.preregister(archive, "train", registration, output)
    assert not registration.exists()
    assert not output.exists()


def test_duplicate_selected_member_is_rejected(protocol, inputs, monkeypatch):
    _forbid_compute(monkeypatch, protocol)
    _, archive, registration, output = inputs
    with zipfile.ZipFile(archive, "a") as data:
        original = data.read("train.npy")
        with pytest.warns(UserWarning, match="Duplicate"):
            data.writestr("train.npy", original)
    with pytest.raises(ValueError):
        protocol.preregister(archive, "train", registration, output)
    assert not registration.exists()
    assert not output.exists()
