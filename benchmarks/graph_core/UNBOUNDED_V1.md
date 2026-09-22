# Fixed graph-core benchmark

This directory reproduces the **fixed three-seed confirmation protocol**, not a
hyperparameter search. The published numbers come from the original frozen run;
this public harness is a reviewed code-path relocation. See
[`../results/graph_core_provenance.json`](../results/graph_core_provenance.json)
for original source hashes under logical roles and their public-port mapping.
Different paths/files have different hashes: **no false hash-equivalence claim**.
No real fits or transforms were rerun to create this public port.

## Results and limitations

[`../results/graph_core_confirmation.json`](../results/graph_core_confirmation.json)
contains **all 30 records**: three seeds (0, 1, 2) for strong, direct+A and genuine
UMAP on each dataset, plus one PCA2 control per dataset. It includes all
k=5/15/50 fit/TEST geometry, all-TEST classification/clustering, costs, optimizer
nonconvergence, graph audits, full resolved strong configs and adaptations.
Means use sample standard deviation (ddof=1); PCA has no invented replication.
Failures are records, not silently removed seeds.

| Dataset | Strong overlap@15 | Direct+A overlap@15 | UMAP overlap@15 |
|---|---:|---:|---:|
| Fashion | .0641 ± .0025 | .1728 ± .0022 | .1442 ± .0013 |
| HAR | .0547 ± .0012 | .1970 ± .0010 | .1550 ± .0023 |
| COIL | .4339 ± .0124 | .7899 ± .0035 | .7700 ± .0050 |

These neighborhood gains are **not an across-metric win**:

- Fashion direct+A ARI **.4211**, below UMAP **.4711**; NMI also lower.
- HAR direct+A ARI **.5356**, below the old strong graph control **.6526**;
  NMI also lower. Its classification accuracy is below UMAP here.
- Fashion direct+A transforms all 10,000 TEST rows in **141.98 s** on average,
  versus **8.13 s** for UMAP and **.012 s** for the strong MLP. Shared graph/index
  and preprocessing costs are additional. Different timing scopes and objectives
  preclude matched-compute speedup claims.
- COIL direct+A ARI/NMI remain slightly below UMAP.
- Exact conditional KL can produce extreme **finite** query positions, including
  a Fashion seed-0 output far outside the fitting layout. Optimizer success and
  a small gradient do not guarantee globally sensible placement. No point was
  clipped, replaced or removed from evaluation.

The official TEST sets were used historically: **not virgin external evidence**.
Three stochastic seeds are not independent datasets or generalization confidence.
No semantic, global topology, state-of-the-art or official DeepTDA equivalence
claim is made. COIL “strong” is the existing **stress600 documented control**,
**not the best possible COIL configuration**.

## Fixed data and evaluation

- Fashion: all 60,000 TRAIN / 10,000 TEST, cached unwhitened PCA64 fitted on full
  official TRAIN. Never reuse an internal-50k-split PCA for the official split.
- HAR: official subject-disjoint 7,352 / 2,947, 561 features. TRAIN-only float64
  mean/std then float32, identical to native preprocessing. Strong receives raw
  features and its original standardization; every method sees the same reference.
- COIL: 960 fitting / 480 held-out views of the same 20 objects, cached TRAIN-fitted
  PCA64. This is viewpoint interpolation, **not an unseen-object split**.

Geometry: fixed 512 TRAIN and 1,024 TEST queries (all 480 COIL TEST) ranked against
**all fitting references**, not a small subcloud; TRAIN self excluded. Exact
blocked overlap/trustworthiness/continuity@5/15/50. Original feature dtype is
retained; distance reranking is float64. Label/subject/pose payloads are read
**only after all 30 fit/transform jobs are terminal and outputs hash-frozen**.
Then all TEST rows receive uniform 15NN classification; KMeans fits TRAIN
coordinates with n_init=10 and run seed, cluster count from TRAIN labels, then
predicts TEST for ARI/NMI. COIL pose adjacency is object-conditioned k2 over all
72 views/object, TEST queries only, with a source-feature control.

## Methods and isolation

- `strong_configs.json` freezes Fashion NCE7200, HAR graph2400, COIL stress600.
  Only a process-local data provider supplies the shared graph; losses/training
  are not changed.
- Direct graph: 300 Cauchy coordinate-SGD sweeps, five negative draws per accepted
  directed edge, spectral initialization for connected graphs. V2 explicitly
  initializes disconnected component charts using source-centroid PCA placement
  and the documented coincident-centroid circle fallback.
- A mapper: exact **all-fitting-anchor** conditional Cauchy KL; TRAIN-derived
  layout edge-median unit, one barycentric start, L-BFGS-B
  maxiter=100/maxls=30/ftol=1e-12/gtol=1e-8. No approximate forces. All TEST outputs,
  statuses and full-gradient residuals retained, including finite optimizer
  failures; no retries. `kernels.py` preserves the original A arithmetic and
  omits only unused ranking-mapper/tree machinery.
- Author UMAP: actual installed library, common 15 nonself fitting neighbors plus
  self column (`n_neighbors=16`), 300 epochs, min_dist=.1, spread=1, learning_rate=1,
  negative_sample_rate=5, spectral, random_state=transform_seed=seed, n_jobs=1.
  `force_approximation_algorithm=True` preserves supplied neighbors on small COIL.
  This API adaptation is explicit, not an untouched-author-default claim.

Each dataset/seed shares one NNDescent31 candidate index, reranked to 15 fitting
neighbors; persistent index for queries. Exact 256-query audits against fullTRAIN
check fitting, A-query and author-query recall separately, threshold .9 tie-aware
mean. Failures are retained without fallback. Affinity formulas and objectives
still differ across methods.

**Torch and Numba/NNDescent/UMAP never share a fitting process.** The graph
numerical worker is loaded by file from `python/open_deep_tda/_graph_worker.py`,
not through package `__init__`. Its event counter is nonbinding at the registered
4-billion-event cap; synthetic checks verify unchanged floating updates/RNG.
The public port does not silently call the newer package predictor or change
source float32 into float64 before component-centroid initialization.

## Reproduction (explicit, no implicit downloads)

Use a source checkout and an environment with the package/native extension,
NumPy, SciPy, scikit-learn, Torch, Numba, pynndescent, umap-learn and matplotlib.
The recorded author version is umap-learn 0.5.3. Versions/source hashes are pinned
per registration; cross-version/platform bitwise reproduction is not promised.
Optional dependencies are not required for ordinary source-only protocol tests.

The default input caches must already exist:

```
outputs/v02-fashion_mnist/{features,annotations,reference_pca}.npz
outputs/v02-coil20/{features,annotations,reference_pca}.npz
data/uci_har/features.npz
```

For existing downloaded raw archives, the repository's cache preparation can be
invoked explicitly, separately from the benchmark (ingestion may read split and
annotation payloads, but reference fitting is TRAIN-only):

```sh
PYTHONPATH=python:benchmarks python - <<'PY'
from image_protocol import prepare_images
from open_deep_tda.real_datasets import load_uci_har
prepare_images('fashion_mnist', 'outputs/v02-fashion_mnist', download=False)
prepare_images('coil20', 'outputs/v02-coil20', download=False)
load_uci_har(download=False)
PY
```

Missing caches/raw archives fail; no benchmark command downloads anything.
Review dataset access/licensing separately. Do not publish these caches.

```sh
# Synthetic compatibility only; no real dataset reads or refitting:
python benchmarks/confirm_graph_core.py --preflight-only --output outputs/graph-preflight
python -m pytest -q tests/test_graph_core_protocol.py

# A NEW confirmation requires an empty artifact directory:
python benchmarks/confirm_graph_core.py --preregister --output outputs/graph-confirmation-new
python benchmarks/confirm_graph_core.py --run --output outputs/graph-confirmation-new
# Optional shell background execution, retaining the bounded supervisor:
# nohup python benchmarks/confirm_graph_core.py --run --output outputs/graph-confirmation-new \
#   > outputs/graph-confirmation-runner.log 2>&1 &
```

Registration runs synthetic preflight, resolves frozen strong configs and reads
only TRAIN feature payloads. It binds input ZIP metadata without hashing TEST or
labels. TEST payload hashes are recorded on their first permitted read. No code,
config, Python file set or input changes after registration. The exclusive marker
rejects in-place retries; use a new explicitly identified experiment for repairs.

Supervisor: maximum two isolated children, one native thread each, 45-minute
wall cap, sampled combined-process RSS limit 8 GiB. Per-job caps and every
failure/skip are recorded. Sampling is not a hard instantaneous OS memory quota.
`checkpoint.json`, `runner_status.json`, `results.json`, `verification.json` and
`runner_artifacts.json` are **local artifacts**, not automatically public data.
Models/indexes use trusted-local checksummed pickle where necessary; never load
untrusted pickle. Raw coordinates, predictions, labels, sample IDs and logs must
not be published.

## Publishing and figures

`publish.py` exports only allowlisted metrics/configs/audit summaries and rejects
raw fields or local paths. This is a code-reviewed publication boundary, not a
promise that `.gitignore` anonymizes arbitrary files.

The three `assets/graph-core-*.png` figures use **fixed seed 0, never best seed**,
PCA/strong/direct+A/UMAP in the same order, all TEST points and complete class
legends. Top panels show full TEST extent, including extreme A outputs. Bottom
panels use each method's TRAIN min/max plus 5% for labeled display-only detail,
with counts outside; no metrics change. Numbers on plots are seed0, not means.
The complete original three-seed grids remain retained locally. No model was
rerun to draw public figures.

Mathematical ideas are attributed to Laplacian Eigenmaps (Belkin & Niyogi, 2003),
UMAP fuzzy graphs/negative sampling (McInnes, Healy & Melville, 2018,
https://arxiv.org/abs/1802.03426), and standard normalized Cauchy KL/L-BFGS-B query
optimization. The graph/mapper implementation is independent, not copied author
source and not numerically identified with UMAP or proprietary DeepTDA.

To export or redraw from a **completed local run** without invoking any model:

```sh
export GRAPH_RUN=outputs/graph-confirmation-new
python - <<'PY'
import os
from benchmarks.graph_core.publish import write_aggregate
from benchmarks.graph_core.figures import render
run = os.environ['GRAPH_RUN']
# Inspect locally before promoting any new result over the published history:
write_aggregate(run, run + '/sanitized_confirmation.json')
for dataset in ('fashion', 'har', 'coil'):
    render(run, dataset, run + '/seed0-' + dataset + '.png')
PY
```

The published environment was macOS 15.4 arm64, NumPy 1.21.6, SciPy 1.10.1,
scikit-learn 1.0.2, Torch 2.7.1, Numba 0.55.0 / llvmlite 0.38.0, matplotlib
3.5.3, author umap-learn 0.5.3. Dependency locations are deliberately not published.
