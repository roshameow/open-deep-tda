"""Experimental image-prior example: raw versus translation-tangent Digits geometry.

All 1,797 bundled sklearn 8x8 images, original order, raw pixel intensities,
transductive fit, fixed seed 0. No downloads, normalization, tuning or held-out
claim. Both default arms use the SAME PrecomputedGraphEmbedding settings
(300 epochs, 5 negatives); only the input dissimilarity changes. --umap adds
both arms of the genuine optional author umap-learn implementation.

The tangent reference uses zero-padded centered pixel gradients and bounded
+/-1-pixel coefficients in a local first-order translation model, NOT a pixel
warp, exact translation invariance, or a metric (triangle inequality need not
hold). This experimental image prior has no generic clustering, out-of-sample
or persistent-homology (PH) preservation guarantee. Earlier independent-seed
checks (seeds 1 and 2) had mixed split/held-out ARI: each reducer won one seed
and lost one when tangent was compared with raw. Transductive benefits occurred
for BOTH reducers: evidence about the reference, not our optimizer's superiority.
Seed 0 is illustrative/development, not new independent confirmation.

Labels are accessed only AFTER ALL fits, for KMeans10 ARI/NMI, label silhouette,
full 10x10 contingencies and all-point, full-range plots with identical colors.
K=10 is a dataset-informed post-fit diagnostic, not a clustering guarantee.

Privacy: the fitted reference retains copied TRAIN images; graph/layout
neighborhoods and even plots are sensitive, not anonymized. No automatic
checkpoint, coordinates, raw data, distances, model or pickle is exported.
Only bounded summary.json and digits.png go to a fresh/empty subdirectory of
Path.cwd()/outputs (ignored in this checkout). The graph API uses temporary
sensitive worker scratch with cleanup, not secure erasure. Review before sharing.

Example (installed package plus graph/plot dependencies):
    python examples/digits_image_geometry.py
    python examples/digits_image_geometry.py --umap --output outputs/digits-with-umap
"""

import argparse
from contextlib import contextmanager
import importlib.abc
from importlib import metadata
import json
from pathlib import Path
import sys
from time import perf_counter


SEED = 0
EPOCHS = 300
NEGATIVES = 5
SUMMARY_LIMIT = 65536
GRAPH_OPTIONS = dict(seed=SEED, epochs=EPOCHS, negative_rate=NEGATIVES)
UMAP_OPTIONS = dict(
    metric="precomputed", n_components=2, n_neighbors=16, n_epochs=EPOCHS,
    negative_sample_rate=NEGATIVES, min_dist=0.1, spread=1.0,
    learning_rate=1.0, init="spectral", random_state=SEED,
    transform_seed=SEED, n_jobs=1,
)
INTERPRETATION = (
    "Experimental image prior; local first-order translation, not a pixel warp "
    "or exact translation invariance. Tangent dissimilarity is not a metric. "
    "No generic clustering, out-of-sample or PH/raw-topology guarantee. "
    "Prior independent-seed split/held-out ARI was mixed at seeds 1 and 2: "
    "each reducer won one and lost one against its raw control. "
    "Prior transductive benefits occurred for both reducers, supporting a "
    "reference effect, not our optimizer's superiority. Seed 0 is development."
)


def prepare_output(path):
    """Resolve symlinks before enforcing a proper descendant of cwd/outputs."""
    root = Path.cwd().resolve() / "outputs"
    output = Path(path).expanduser().resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("--output must resolve to a subdirectory of Path.cwd()/outputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--output must be a new or empty directory; no overwriting")
    output.mkdir(parents=True, exist_ok=True)
    return output


@contextmanager
def numerical_imports_only():
    """Disable optional parametric imports, not UMAP's actual numerical reducer.

    umap-learn may attempt TensorFlow in its package initializer. Raising an
    ordinary ImportError lets upstream mark only its parametric extra absent.
    No source patch, fake module, replacement estimator or hidden fallback.
    This guard blocks new imports; it does not unload already imported modules.
    """
    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in {"tensorflow", "torch"}:
                raise ImportError("optional parametric dependencies disabled in this example")
            return None

    guard = Guard()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)


def load_methods(include_umap):
    """Use only the public APIs; optional upstream failure is an explicit error."""
    from open_deep_tda.image_dissimilarity import TranslationTangentDissimilarity
    from open_deep_tda.precomputed_graph import PrecomputedGraphEmbedding

    methods = {"graph": PrecomputedGraphEmbedding}
    if include_umap:
        try:
            metadata.version("umap-learn")  # Reject an unrelated 'umap' package.
            with numerical_imports_only():
                from umap import UMAP
        except (ImportError, metadata.PackageNotFoundError) as exc:
            raise RuntimeError(
                "--umap requested but genuine umap-learn or its numerical dependencies "
                "are unavailable; install umap-learn in this environment. No fallback."
            ) from exc
        methods["umap"] = UMAP
    return TranslationTangentDissimilarity, methods


def build_references(raw, images, reference_class, report):
    """No labels: raw Euclidean pixels versus the default tangent reference."""
    from scipy.spatial.distance import cdist

    start = perf_counter()
    raw_d = cdist(raw, raw, metric="euclidean")
    report["reference_seconds"]["raw"] = perf_counter() - start
    start = perf_counter()
    reference = reference_class()
    tangent_d = reference.fit_transform(images)
    report["reference_seconds"]["tangent"] = perf_counter() - start
    # The reference retains copied TRAIN images in memory; never serialize it.
    return {"raw": raw_d, "tangent": tangent_d}


def fit_methods(references, methods, report):
    """Fit every declared arm before the caller can perform label diagnostics."""
    import numpy as np

    layouts = {}
    for geometry, distances in references.items():
        for reducer, constructor in methods.items():
            key = geometry + "_" + reducer
            result = report["arms"][key]
            start = perf_counter()
            try:
                options = GRAPH_OPTIONS if reducer == "graph" else UMAP_OPTIONS
                with numerical_imports_only():
                    model = constructor(**options)
                    # Equal numerical D for both reducers; independent copies
                    # prevent a reducer's in-place changes contaminating its peer.
                    layout = np.asarray(model.fit_transform(distances.copy()))
                if layout.shape != (len(distances), 2) or not np.isfinite(layout).all():
                    raise ValueError("reducer must return finite (N,2) coordinates")
                layouts[key] = layout
                result["status"] = "fitted"
            except Exception as exc:
                result.update(status="failed", error_type=type(exc).__name__)
                raise  # No retries, alternative settings, or substituted reducer.
            finally:
                result["fit_seconds"] = perf_counter() - start
    return layouts


def score_layout(layout, labels):
    """All-row Euclidean-layout diagnostics, strictly post-fit; never select an arm."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
    )

    clusters = KMeans(n_clusters=10, n_init=20, random_state=SEED).fit_predict(layout)
    contingency = np.zeros((10, 10), dtype=int)
    np.add.at(contingency, (labels, clusters), 1)
    return dict(
        kmeans10_ari=float(adjusted_rand_score(labels, clusters)),
        kmeans10_nmi=float(normalized_mutual_info_score(labels, clusters)),
        label_silhouette=float(silhouette_score(layout, labels, metric="euclidean")),
        digit_by_cluster=contingency.tolist(),
    )


def plot_layouts(layouts, labels, path):
    """No subset, crop, outlier removal, alignment or label-dependent rescaling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = len(layouts) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows), squeeze=False)
    colors = plt.get_cmap("tab10")
    try:
        for ax, (key, layout) in zip(axes.flat, layouts.items()):
            for digit in range(10):
                selected = labels == digit
                ax.scatter(layout[selected, 0], layout[selected, 1],
                           s=7, alpha=0.65, color=colors(digit), linewidths=0)
            # Full individual ranges (not identical coordinate scales), with
            # margins for every point. Equal aspect, no quantile/axis clipping.
            ax.margins(0.05)
            ax.set_aspect("equal", adjustable="box")
            ax.set(title=key.replace("_", " / "), xlabel="Embedding 1", ylabel="Embedding 2")
        legend = [Line2D([], [], marker="o", linestyle="", color=colors(i), label=str(i))
                  for i in range(10)]
        fig.legend(handles=legend, title="Post-fit digit label", loc="lower center", ncol=10)
        fig.suptitle("Experimental image prior | all 1,797 points per panel | seed 0\n"
                     "Transductive; full individual ranges; no clustering or PH guarantee")
        fig.tight_layout(rect=(0, 0.09 / rows, 1, 0.92))
        with Path(path).open("xb") as stream:
            fig.savefig(stream, format="png", dpi=150)
    finally:
        plt.close(fig)


def make_report(include_umap):
    reducers = ("graph", "umap") if include_umap else ("graph",)
    return dict(
        status="started", stage="dependencies",
        dataset=dict(source="sklearn.datasets.load_digits (bundled UCI Digits)",
                     rows=1797, image_shape=[8, 8], features=64,
                     cohort="all rows, original order, transductive; no held-out data",
                     preprocessing="raw 0..16 pixels; no normalization or learned features"),
        seed=SEED, interpretation=INTERPRETATION,
        reference=dict(raw="scipy.spatial.distance.cdist(rawX, rawX, metric='euclidean')",
                       tangent="TranslationTangentDissimilarity().fit_transform(images); defaults",
                       tangent_convention="spacing 1, zero-padded centered gradients, +/-1 per axis; "
                                          "sqrt(mean of the two directed squared residuals)"),
        graph_options=dict(GRAPH_OPTIONS),
        graph_other_options="unchanged public PrecomputedGraphEmbedding defaults",
        umap_options=dict(UMAP_OPTIONS) if include_umap else None,
        umap_status="requested" if include_umap else "not_requested",
        comparison=("Identical D values for both reducers within each geometry; graph settings "
                    "identical across geometries. UMAP n_neighbors=16 uses self+15 fit convention; "
                    "its native float conversion, tie handling, graph, initializer and objective "
                    "can differ. Equal epochs/negatives are not identical work or matched graphs."),
        evaluation=("Only after all fits: KMeans K=10 (dataset-informed), n_init=20, seed=0; "
                    "ARI/NMI against digit labels; Euclidean label silhouette on all rows; "
                    "10x10 contingency rows=digits 0..9, columns=arbitrary cluster IDs 0..9. "
                    "No subsampling, label-based selection, retries or tuning."),
        timing_scope=("perf_counter wall seconds, separate reference construction and each fit "
                      "(including its D copy). Import/load/evaluation/plot time excluded from fits. "
                      "Fixed raw-before-tangent order; first-use worker/startup/JIT/cache effects "
                      "are included when incurred, not equalized. No speed or RSS claim."),
        privacy=("Reference retains copied TRAIN images; graph/layout neighborhoods and plots "
                 "are sensitive. No automatic checkpoint or raw data/D/coordinates/model/pickle "
                 "export. Graph API temporary sensitive worker scratch is cleaned, not securely "
                 "erased. Only bounded summary.json and all-point digits.png; review sharing."),
        reference_seconds={},
        arms={geometry + "_" + reducer: {"status": "not_run"}
              for geometry in ("raw", "tangent") for reducer in reducers},
    )


def save_summary(output, report):
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > SUMMARY_LIMIT:
        raise ValueError("summary exceeds the fixed 64 KiB export budget")
    with (output / "summary.json").open("xb") as stream:
        stream.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("outputs/digits-image-geometry"),
                        help="fresh/empty subdirectory of Path.cwd()/outputs; symlinks resolved")
    parser.add_argument("--umap", action="store_true",
                        help="also fit genuine umap-learn on BOTH D matrices; error if unavailable")
    args = parser.parse_args(argv)
    output = prepare_output(args.output)
    report = make_report(args.umap)
    try:
        reference_class, methods = load_methods(args.umap)
        for package in ("open-deep-tda", "numpy", "scipy", "scikit-learn", "numba", "matplotlib") + (
                ("umap-learn",) if args.umap else ()):
            try:
                version = metadata.version(package)
            except metadata.PackageNotFoundError:
                version = "unavailable as installed distribution"
            report.setdefault("versions", {})[package] = version
        from sklearn.datasets import load_digits
        report["stage"] = "load"
        data = load_digits()
        if data.data.shape != (1797, 64) or data.images.shape != (1797, 8, 8):
            raise ValueError("expected all 1797 bundled raw 8x8 Digits images")
        report["stage"] = "references"
        references = build_references(data.data, data.images, reference_class, report)
        report["stage"] = "fits"
        layouts = fit_methods(references, methods, report)
        del references  # Dense distances are never exported.
        report["stage"] = "postfit_evaluation"
        labels = data.target  # First label access: every declared fit has finished.
        for key, layout in layouts.items():
            report["arms"][key]["postfit"] = score_layout(layout, labels)
        report["stage"] = "plot"
        plot_layouts(layouts, labels, output / "digits.png")
        report.update(status="complete", stage="complete")
        if args.umap:
            report["umap_status"] = "completed_genuine_author_implementation"
    except Exception as exc:
        # Exception messages can include machine paths; persist only the type
        # and stage. The console still receives the genuine exception traceback.
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        save_summary(output, report)
    print("Saved summary.json and digits.png under", output)


if __name__ == "__main__":
    main()
