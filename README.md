# Open Deep-TDA

[![Tests](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml/badge.svg)](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文](README.zh-CN.md) · [Install](#install) · [Examples](#examples) · [Results](#results)

An independent, inspectable dimensionality-reduction project inspired by publicly described Deep-TDA ideas. **Not DataRefiner's official implementation or an exact reproduction.** The current graph reducer does not guarantee cluster separation or preservation of source cycles. The older `DeepTDA` neural estimator and bounded structural verification tools remain available.

## Results

**Which plot is ours?** In the K4 comparison below, the **middle panel, “GraphEmbedding,” is Open Deep-TDA**. The left panel is an external TopoAE++ author-core CPU adapter; the right panel is UMAP. All 300 input points are shown. Colors encode source row order, *not* classes. The methods have different compute budgets, and the adapter is one fixed run, not the author's best-of-ten paper figure. Visually similar bars alone do not certify preservation of the *same* source cycles.

![K4: left TopoAE++ adapter, middle Open Deep-TDA GraphEmbedding, right UMAP](assets/visual-contracts-K4.png)

The current graph layout retains fewer substantial K4 H₁ bars than this TopoAE++ adapter (one versus three). This is a limitation of the current result, not a topology-preservation success claim.

### Digits: two input representations, two reducers

In this figure **both left panels are our project**: top-left uses raw pixels with the Open Deep-TDA precomputed-graph adapter; bottom-left uses the explicit translation-tangent image dissimilarity with the **same** graph adapter. Both right panels use external UMAP on the corresponding input. All 1,797 Digits points appear in each panel; colors are digit labels used only *after* fitting. This is a transductive illustration, not a held-out test.

![Digits: left Open Deep-TDA precomputed graph, right UMAP; top raw pixels, bottom experimental translation-tangent dissimilarity](assets/digits-image-geometry.png)

| Input, all 1,797 Digits, seed 0 | Open Deep-TDA precomputed graph: KMeans10 ARI | UMAP: KMeans10 ARI |
|---|---:|---:|
| Raw pixels | .822 | .822 |
| Translation-tangent dissimilarity | .848 | .904 |

The image dissimilarity improves this example for **both** reducers; it does not establish that our optimizer outperforms UMAP. It is not a metric or exact shift invariance, changes the input topology, and is **not** the default for generic numeric vectors. Results on fixed train/held-out Digits splits were mixed. Some digit classes remain fragmented.

On the separate fixed Fashion-MNIST graph benchmark (three seeds, 60,000 TRAIN / 10,000 TEST), our post-fit KMeans ARI is **.421**, below UMAP's **.471**; post-fit 15-NN accuracy is **78.49%** versus **77.88%**. That accuracy is not evidence of better cluster layout. [Benchmark record](benchmarks/results/graph_core_compact_confirmation.json).

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

Graph/image features described here are on **unreleased main**, not the older v0.3.0 release artifact. Installation does not download benchmark datasets.

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
