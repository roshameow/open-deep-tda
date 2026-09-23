#!/usr/bin/env python3
"""Render frozen PH-trained DeepTDA vs external reducers without refitting.

Inputs are separately produced, sensitive TRAIN/TEST embeddings and annotations.
This script never trains a reducer, chooses a seed, clips points, or exports
coordinates. Pass explicit trusted paths; no local/private path is embedded here.
The optional third arm must be the AUTHOR TopoAE++ train/test archive on the
identical rows and representation, not a substituted graph-only result.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SPECS = {
    "fashion": (60000, 10000, list(range(10)), ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]),
    "har": (7352, 2947, list(range(1, 7)), ["Walking", "Walking upstairs", "Walking downstairs", "Sitting", "Standing", "Laying"]),
    "coil": (960, 480, list(range(1, 21)), [f"Object {i}" for i in range(1, 21)]),
}


def embedding(value, rows):
    a = np.asarray(value)
    if a.shape != (rows, 2) or a.dtype.kind not in "iuf" or not np.isfinite(a).all():
        raise ValueError("expected finite (rows,2) embedding; no point exclusion allowed")
    return a


def load_pair(directory, counts):
    path = Path(directory)
    return (embedding(np.load(path / "fit.npy", allow_pickle=False), counts[0]),
            embedding(np.load(path / "test.npy", allow_pickle=False), counts[1]))


def load_author(archive, counts):
    with np.load(archive, allow_pickle=False) as arrays:
        if not {"train", "test"}.issubset(arrays.files):
            raise ValueError("author archive requires train and test")
        return embedding(arrays["train"], counts[0]), embedding(arrays["test"], counts[1])


def render(dataset, labels, arms, destination):
    n, m, classes, names = SPECS[dataset]
    labels = np.asarray(labels)
    if labels.shape != (m,) or labels.dtype.kind not in "iu" or sorted(np.unique(labels).tolist()) != classes:
        raise ValueError("full TEST annotation/class contract failed")
    for name, (fit, test) in arms:
        embedding(fit, n)
        embedding(test, m)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    palette = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    fig, axes = plt.subplots(1, len(arms), figsize=(6.3 * len(arms), 5.2), squeeze=False)
    try:
        for ax, (title, (_, test)) in zip(axes[0], arms):
            for i, label in enumerate(classes):
                points = test[labels == label]
                ax.scatter(points[:, 0], points[:, 1], color=palette(i),
                           s=3 if dataset == "fashion" else 6 if dataset == "har" else 15,
                           alpha=0.7, linewidths=0, rasterized=True)
            ax.margins(0.05)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Embedding 1")
            ax.set_ylabel("Embedding 2")
            # Independent full ranges; no global or quantile cropping, no missing rows.
        handles = [Line2D([], [], linestyle="", marker="o", color=palette(i),
                          label=names[i], markersize=6) for i in range(len(classes))]
        fig.legend(handles=handles, loc="lower center", ncol=10 if dataset != "har" else 6,
                   fontsize=8, frameon=False)
        fig.suptitle(f"{dataset.upper()} | all {m:,} TEST points | fixed seed 0", fontsize=15)
        fig.text(.5, .13, "Same input rows; independent full axes; labels used only after fitting. "
                 "Different methods and compute budgets.", ha="center", fontsize=9)
        fig.subplots_adjust(top=.82, bottom=.26, wspace=.27)
        fig.savefig(destination, dpi=160, metadata={"Title": f"PH-trained DeepTDA vs external methods on {dataset}",
                    "Description": "All TEST points and classes; no clipping, downsampling, refitting or model selection"})
    finally:
        plt.close(fig)
    return {"dataset": dataset, "test_rows": m, "panels": [name for name, _ in arms],
            "seed": 0, "all_test_points": True, "clipping": False, "labels_postfit_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--ph", type=Path, required=True, help="frozen PH-trained DeepTDA fit.npy/test.npy directory")
    parser.add_argument("--umap", type=Path, required=True, help="frozen genuine UMAP fit.npy/test.npy directory")
    parser.add_argument("--topoaepp", type=Path, help="optional author-core archive containing train/test")
    parser.add_argument("--labels", type=Path, required=True, help="private annotation NPZ, not exported")
    parser.add_argument("--label-key", default="test_labels.npy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = SPECS[args.dataset][:2]
    with np.load(args.labels, allow_pickle=False) as source:
        labels = source[args.label_key]
    arms = [("PH-Regularized Embedding (DeepTDA)", load_pair(args.ph, counts)),
            ("UMAP (external)", load_pair(args.umap, counts))]
    if args.topoaepp:
        arms.append(("TopoAE++ (author core adapter)", load_author(args.topoaepp, counts)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(render(args.dataset, labels, arms, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
