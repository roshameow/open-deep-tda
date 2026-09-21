# Third-party dependencies and attribution

The repository's implementation is original code under MIT. No proprietary DataRefiner code, conformal_geometry source, TetGen, pretrained model weights, or third-party source trees are vendored.

Runtime/build dependencies are installed separately and retain their own licenses:

| Dependency | Role | Upstream license family |
|---|---|---|
| NumPy | Numeric arrays | BSD-3-Clause |
| SciPy | Assignment, distances, sparse graph matching | BSD-3-Clause |
| PyTorch | Neural training / autodiff | BSD-style (plus bundled dependency notices) |
| scikit-learn | Exact nearest neighbors, PCA/t-SNE baselines, bundled example data loader | BSD-3-Clause |
| pybind11 | C++/Python binding headers | BSD-3-Clause |
| setuptools / wheel | Packaging | MIT |
| pytest | Optional tests | MIT |
| Ripser.py | Optional independent persistence oracle | MIT |
| persim | Optional bottleneck oracle, installed with Ripser.py | MIT |
| umap-learn | Optional baseline only | BSD-3-Clause |
| pynndescent / numba | Optional UMAP dependencies | BSD-style; review exact distributions |

The table is a dependency inventory, not a replacement for the complete notices of each installed distribution or legal advice. Binary redistribution must preserve applicable upstream and transitive-library notices. Inspect licenses of the exact wheels/platform libraries being redistributed.

The mathematical ideas are attributed in the README and references below, especially Topological Autoencoders (Moor et al., 2020). DataRefiner and other third-party names identify external publications/software only; there is no claimed affiliation or endorsement.

The optional `sklearn.datasets.load_digits` validation example uses scikit-learn's bundled optical recognition digits data. `assets/digits-example.png` contains ten rendered 8×8 examples and a derived embedding visualization, but no raw dataset arrays. Consult scikit-learn's upstream dataset description and original UCI attribution when redistributing derived artifacts. Synthetic benchmark generators are original and included.

## Added real-data / upstream comparisons

- UCI HAR original 561-feature data: Reyes-Ortiz et al. (2013), DOI
  https://doi.org/10.24432/C54S4K. UCI declares CC BY 4.0. Download/cache is
  explicit opt-in under ignored `data/`; real feature files are not included in
  source archives. The loader records the official URL and observed checksum.
- Original Topological Autoencoders: BSD-3-Clause, external repository
  https://github.com/BorgwardtLab/topological-autoencoders at commit
  `203e94a69c5f9cda049b9c3985b7c2b1e39ca922`. The HAR harness imports unchanged
  official model/loss modules from a user-supplied checkout; those modules are
  not vendored here. Cite Moor et al., ICML 2020 when reporting that baseline.
- TopoAE++ and RTD-AE were source-reviewed, not incorporated into this codebase.
  In particular, the audited bundled TTK license has extra acknowledgment and
  citation conditions; a blanket MIT/BSD-3 label for the whole TopoAE++ tree
  would be incorrect. See the README related-projects section and the upstream repositories for scope and license caveats.

## v0.2 image data and optional native research adapter

- Fashion-MNIST: official Zalando SE dataset, MIT (2017); cite Xiao, Rasul and Vollgraf (2017), arXiv:1708.07747. Original gzip checksums and observed SHA256 are recorded by the loader. Data archives remain outside the source distribution. `assets/fashion-mnist-example.png` includes ten rendered examples and a derived, display-subsampled embedding. `assets/fashion-optimization.png` is a derived comparison of all official-test coordinates.
- UCI Human Activity Recognition: Reyes-Ortiz et al. (2013), DOI https://doi.org/10.24432/C54S4K, CC BY 4.0. `assets/har-example.png` is a derived, display-subsampled embedding of the official subject-disjoint split; `assets/har-optimization.png` compares all official-test coordinates. Raw sensor rows are not shipped.
- COIL-20: cite Nene, Nayar and Murase (1996), CUCS-005-96, Columbia CAVE. The inspected official pages do not state an explicit license grant. Do NOT assume MIT, commercial permission, or invented research-only terms; confirm rights before redistribution/use beyond the intended local research. No original photographs are bundled here.
- Optional NN-descent uses external `pynndescent`/Numba in an isolated process; their own licenses and binary dependency notices apply. Pillow is used to decode/downsample images. psutil is used by the bounded research native-run adapter.
- TopoAE++ is now also used in a real external-source native CPU experiment. Only independent adapters and result metadata are stored here; upstream source, Boost headers, compiled upstream libraries and trained model binaries are not shipped. The audited TTK license has five conditions, including acknowledgment/citation, and must be checked before redistributing any linked executable.
- Portable Boost 1.74 headers were unpacked in `/tmp` for this experiment, not installed globally or vendored. Boost BSL-1.0/general and per-header notices remain applicable. No foreign-platform binary was executed.

TTK acknowledgment: This document includes materials generated with TTK (the Topology ToolKit) which is developed by the CNRS & Sorbonne Universite and its contributors. Cite Julien Tierny et al., *The Topology ToolKit* ([publications](https://topology-tool-kit.github.io/publications.html)), and [Topological Autoencoders++](https://arxiv.org/abs/2502.20215).

## Additional directly used tools

- [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE): image decoding/resampling; Pillow/PIL license terms (HPND family), with separately licensed bundled codecs in binary distributions.
- [psutil](https://github.com/giampaolo/psutil/blob/master/LICENSE): optional bounded native-worker process/RSS monitoring; BSD-3-Clause.
- [threadpoolctl](https://github.com/joblib/threadpoolctl/blob/master/LICENSE): explicit benchmark thread-pool control; BSD-3-Clause. Declared directly in benchmark extras.
- [pynndescent](https://github.com/lmcinnes/pynndescent) and [Numba](https://github.com/numba/numba): optional approximate neighbor construction; inspect their shipped notices and llvmlite/LLVM transitive notices when redistributing binaries.

The v0.3 fuzzy objective uses existing local exponential affinity / fuzzy-union ideas (McInnes, Healy & Melville, UMAP, arXiv:1802.03426). It is an independently implemented sampled graph objective, not exact ParametricUMAP and not a novel affinity construction.
