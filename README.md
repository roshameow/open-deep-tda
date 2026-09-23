# Open Deep-TDA

[![Tests](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml/badge.svg)](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文](README.zh-CN.md) · [Install](#install) · [Examples](#examples) · [Results](#results)

An independent, inspectable dimensionality-reduction project inspired by publicly described Deep-TDA ideas. **Not DataRefiner's official implementation or an exact reproduction.**

## Which version is this?

**This page shows the current GitHub `main` branch. Its principal reducer is `GraphEmbedding` (Open Deep-TDA): that is the project result labeled “GraphEmbedding” or “Direct graph + A” in the figures.** The earlier `DeepTDA` neural estimator remains available as a separate, older algorithm. `PrecomputedGraphEmbedding` is an optional adapter for an explicitly supplied dissimilarity, not the default for numerical data.

The **`v0.3.0` Git tag/release is an older snapshot** and does **not** contain `GraphEmbedding` or the image-dissimilarity adapter. The current main branch still reports Python package version `0.3.0`; this unchanged version string **does not identify the old release's contents**. Install from `main` to run the figures here; use `git rev-parse --short HEAD` to identify a particular main checkout. No newer release has been tagged.

The current graph reducer does not guarantee cluster separation or preservation of source cycles.

## Results

**Which plot is ours?** In the K4 comparison below, the **middle panel, “GraphEmbedding,” is Open Deep-TDA**. The left panel is an external TopoAE++ author-core CPU adapter; the right panel is UMAP. All 300 input points are shown. Colors encode source row order, *not* classes. The methods have different compute budgets, and the adapter is one fixed run, not the author's best-of-ten paper figure. Visually similar bars alone do not certify preservation of the *same* source cycles.

![K4: left TopoAE++ adapter, middle Open Deep-TDA GraphEmbedding, right UMAP](assets/visual-contracts-K4.png)

The current graph layout retains fewer substantial K4 H₁ bars than this TopoAE++ adapter (one versus three). This is a limitation of the current result, not a topology-preservation success claim.

**Other author-data comparisons, same panel order (left: TopoAE++ adapter; middle: our current `GraphEmbedding`; right: UMAP):** Twist shows all 100 points; COIL20-1 shows 72 views of **one object**, so it measures view progression, not 20-class separation. Colors follow source row order, not class labels. Axes are independently scaled and all points are displayed.

![Twist: TopoAE++ adapter | Open Deep-TDA GraphEmbedding | UMAP](assets/visual-contracts-Twist.png)

![COIL20-1: TopoAE++ adapter | Open Deep-TDA GraphEmbedding | UMAP](assets/visual-contracts-COIL20-1.png)

### Current graph on Fashion-MNIST, HAR and COIL-20

**In each figure below, the *third* panel, “Direct graph + A (TRAIN hull),” is the current project's `GraphEmbedding` + conditional mapper.** From left to right: PCA control; older `DeepTDA` strong configuration; **current graph**; external UMAP. The older `DeepTDA` panel is also project code, but **not** the current graph algorithm. Each plot shows every TEST point with full individual axis ranges; class colors are for post-fit display. The numbers printed above panels are seed-0 diagnostics, not the three-seed means. These controls have different optimization budgets.

![Fashion-MNIST: PCA | older DeepTDA | current Open Deep-TDA graph | UMAP; all 10,000 TEST points](assets/graph-core-fashion.png)

![HAR: PCA | older DeepTDA | current Open Deep-TDA graph | UMAP; all 2,947 TEST points](assets/graph-core-har.png)

![COIL-20: PCA | older DeepTDA | current Open Deep-TDA graph | UMAP; all 480 TEST points](assets/graph-core-coil.png)

On the fixed Fashion-MNIST graph benchmark (three seeds, 60,000 TRAIN / 10,000 TEST), our current graph's post-fit KMeans ARI is **.421**, below UMAP's **.471**; post-fit 15-NN accuracy is **78.49%** versus **77.88%**. Accuracy does not establish a better cluster layout. The COIL-20 older-model control is the recorded 600-step stress setting, not an optimized competitor. [Fixed benchmark results](benchmarks/results/graph_core_compact_confirmation.json).

### Digits: raw graph comparison and an optional image reference

For the **raw-pixel** Digits comparison below, the left column is the current `GraphEmbedding` and the right column is UMAP. The top row fits 1,400 rows of a fixed 1,400/397 split; the bottom row fits all 1,797 separately (not held-out). Both layouts split some instances of the same digit; the labels were used only after fitting.

![Digits raw-pixel graph comparison: current Open Deep-TDA at left, matched UMAP at right; split above, transductive below](assets/visual-contracts-Digits.png)

### Digits: two input representations, two reducers

In this figure **both left panels are our project**: top-left uses raw pixels with the Open Deep-TDA precomputed-graph adapter; bottom-left uses the explicit translation-tangent image dissimilarity with the **same** graph adapter. Both right panels use external UMAP on the corresponding input. All 1,797 Digits points appear in each panel; colors are digit labels used only *after* fitting. This is a transductive illustration, not a held-out test.

![Digits: left Open Deep-TDA precomputed graph, right UMAP; top raw pixels, bottom experimental translation-tangent dissimilarity](assets/digits-image-geometry.png)

| Input, all 1,797 Digits, seed 0 | Open Deep-TDA precomputed graph: KMeans10 ARI | UMAP: KMeans10 ARI |
|---|---:|---:|
| Raw pixels | .822 | .822 |
| Translation-tangent dissimilarity | .848 | .904 |

The image dissimilarity improves this example for **both** reducers; it does not establish that our optimizer outperforms UMAP. It is not a metric or exact shift invariance, changes the input topology, and is **not** the default for generic numeric vectors. Results on fixed train/held-out Digits splits were mixed. Some digit classes remain fragmented.

## Install

Python 3.9+ and a C++17 compiler are required. The neural baseline also requires PyTorch; the graph and image examples need the optional dependencies below.

```bash
git clone https://github.com/roshameow/open-deep-tda.git
cd open-deep-tda
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[graph,plot,images]'
```

The command above installs the checkout of **main**. The old tagged `v0.3.0` artifact does not provide these graph/image APIs. Installation does not download benchmark datasets.

## Examples

```python
from open_deep_tda import GraphEmbedding

model = GraphEmbedding(seed=0)
Z_train = model.fit_transform(X_train)  # finite numeric array; >=16 rows
Z_new = model.transform(X_new)         # same feature representation
```

`GraphEmbedding` uses a source-neighbor graph, coordinate optimization and a conditional out-of-sample mapper. It has **no H₀/H₁ preservation guarantee**. Its predictor retains training features; do not treat saved models as anonymized.

The optional small-image route explicitly changes the input dissimilarity:

```python
from open_deep_tda import TranslationTangentDissimilarity, PrecomputedGraphEmbedding

reference = TranslationTangentDissimilarity()
D_train = reference.fit_transform(images)  # grayscale (N,H,W)
model = PrecomputedGraphEmbedding()
Z_train = model.fit_transform(D_train)
Z_new = model.transform(reference.transform(new_images))
```

The dense adapter defaults to at most 2,000 reference images. Query-distance columns must retain exactly the training-row order. No checkpoint persistence or out-of-sample quality guarantee is provided for this experimental adapter. The fitted reference retains copied training images.

```bash
python examples/run_graph_embedding.py --plot
python examples/digits_image_geometry.py --umap  # --umap requires umap-learn
```

Outputs go to local ignored `outputs/`; do not publish raw coordinates or models without reviewing data permissions. For the historical neural baseline, use `from open_deep_tda import DeepTDA`. Structural checkers and constructive layouts are separate optional APIs; graph outputs do not automatically inherit their certificates.

## Attribution and license

[TopoAE++](https://github.com/MClemot/TopologicalAutoencodersPlusPlus), [TopoAE](https://github.com/BorgwardtLab/topological-autoencoders), [UMAP](https://github.com/lmcinnes/umap) and [TopoMap](https://github.com/harishd10/TopoMap) are separate projects. The TopoAE++ panel uses an external author-core adapter, not code bundled with this repository. [Third-party notices](THIRD_PARTY_NOTICES.md) cover dependencies and datasets. Original project code is [MIT licensed](LICENSE).
