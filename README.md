# PH-Regularized Embedding

[![Tests](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml/badge.svg)](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文](README.zh-CN.md) · [Results](#results) · [Install](#install) · [Use](#use)

**PH-Regularized Embedding** is the working name for this project's **one topology-aware dimensionality-reduction algorithm**, exposed as `DeepTDA` in the current source. The repository/package identifiers remain `open-deep-tda` / `open_deep_tda` for now. This is an independent implementation, **not** DataRefiner's official Deep TDA or an exact reproduction.

The `DeepTDA` training objective combines reference-space geometry with sampled Vietoris–Rips H₀ and H₁ losses and source critical-edge constraints. The exact bounded PH computation runs in C++; the fitted parametric reducer supports `transform` on new rows. **Using PH in training is not a guarantee that a particular source cycle, all supplied rows, or new queries retain their topology.** Cluster separation is not guaranteed either.

## Results

All comparisons below show the **PH-trained `DeepTDA`** output, not a graph-only replacement. Labels are used after each unsupervised fit for display and clustering evaluation. Plots retain **all TEST points** with full, independent axis ranges; no label-dependent point selection. The left panel is our algorithm, the other panels are **external implementations**.

### Fashion-MNIST: PH-Regularized Embedding versus UMAP

60,000 fit / 10,000 test; identical train-fitted PCA64 input. Fixed seed-0 visualization:

![All Fashion-MNIST test points: PH-regularized DeepTDA on left and external UMAP on right](assets/phre-vs-external-fashion.png)

### HAR: PH-Regularized Embedding versus UMAP

Official subject-disjoint 7,352 fit / 2,947 test; same training-defined numeric reference:

![All HAR test points: PH-regularized DeepTDA on left and external UMAP on right](assets/phre-vs-external-har.png)

### COIL-20: PH-Regularized Embedding versus UMAP and TopoAE++

960 fit / 480 held-out views of seen objects; all three methods use the same ordered PCA64 inputs. TopoAE++ is a fixed seed-0 **author model/loss with an independent CPU training adapter** (1,000 updates), not an untouched paper pipeline or a best-of-ten figure. Training budgets and architectures are not equal.

![All COIL-20 test views: PH-regularized DeepTDA on left, external UMAP center, external TopoAE++ author-core adapter right](assets/phre-vs-external-coil.png)

KMeans is fit on each method's TRAIN embedding and evaluated on **all TEST rows** (class count follows each dataset). Three predeclared seeds, same inputs within each dataset:

| Dataset | PH-Regularized Embedding (`DeepTDA`) TEST ARI | External UMAP TEST ARI |
|---|---:|---:|
| Fashion-MNIST | .397 | **.471** |
| HAR | **.653** | .522 |
| COIL-20 | .391 | **.734** |

On COIL-20 alone, the separate one-seed comparison at seed 0 gives **.422** (`DeepTDA`), **.735** (UMAP) and **.527** (TopoAE++ adapter) under TRAIN-fitted KMeans (**20 clusters**, `n_init=10`, fixed seed). These one-seed scores are **not** the three-seed means above. More importantly, ARI or similar-looking persistence bars do **not** certify preservation of the *same* source cycles. The current PH-trained method has **not** passed a full-domain, same-source-cycle benchmark on K4, and is not presented as a solved topology-preserving reducer. [Audited scalar records](benchmarks/results/phre_external_comparison.json).

## Install

Python 3.9+, a C++17 compiler and PyTorch are required:

```bash
git clone https://github.com/roshameow/open-deep-tda.git
cd open-deep-tda
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Installation does not download benchmark datasets.

## Use

```python
from open_deep_tda import DeepTDA

model = DeepTDA(steps=200, h1_size=64)
Z_train = model.fit_transform(X_train)
Z_new = model.transform(X_new)
```

Fit and transform must use the same numeric feature schema; input validation and missing-value handling are explicit. The default uses a training-fitted reference, geometry stress and nonzero H₀/H₁/critical-edge weights; PH is computed on bounded subclouds, **not** silently on an unlimited full Rips complex. Resource limits fail explicitly. Domain-specific settings underlying the plots above differ from these constructor defaults and are retained with the [comparison record](benchmarks/results/phre_external_comparison.json). A trained model may encode or retain sensitive data; saving it does not anonymize it.

## Attribution and license

[TopoAE++](https://github.com/MClemot/TopologicalAutoencodersPlusPlus) and [UMAP](https://github.com/lmcinnes/umap) are independent external methods, not this project's implementations. Standard PH and topology-autoencoder ideas are attributed in [Third-party notices](THIRD_PARTY_NOTICES.md). Original project code is [MIT licensed](LICENSE).
