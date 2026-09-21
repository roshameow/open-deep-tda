# Contributing

Open Deep-TDA is an independent research implementation, not DataRefiner's official software or a numerically equivalent reproduction. Contributions should improve correctness, usability and reproducibility without claiming new state-of-the-art results from limited experiments.

## Development and tests

Use an isolated environment and a C++17 compiler. The commands below assume a POSIX shell and a source checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q

cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DOPEN_DEEP_TDA_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
```

Dependency installation may require network access; ordinary tests must not download datasets or require network access, credentials, an external source checkout, or a GPU. Use small synthetic fixtures, temporary directories and mocked download responses. Do not make local dataset caches prerequisites for passing tests.

Five external TopoAE++ native-input tests currently skip unless `TOPOAEPP_NATIVE_EXE` points to a separately built, trusted adapter executable. Do not enable that optional research path in ordinary CI or count skipped tests as passed. Report the actual pass/skip counts, environment and commands. Optional ANN/baseline extras and their availability must also be stated.

## Correctness and code provenance

- Keep the native implementation original. Cite mathematical ideas; do not copy external implementations without explicit license/provenance review and retained notices. Publicly readable source is not automatically licensed for redistribution.
- For native changes, preserve deterministic filtration/tie ordering, face-before-coface ordering, F₂ reduction, simplex/critical-edge identities, and explicit budget failures. Do not silently drop bars or substitute a different filtration/backend.
- Test analytic fixtures, ties/duplicates, malformed inputs, complete and censored filtrations, exact budget boundaries, and the independent dense reduction oracle. Compare diagrams against an independent PH library where available.
- For loss changes, retain live tensor gradient paths and test finite differences away from pairing switches, empty/diagonal-only diagrams, and degenerate inputs. Output PH and assignments must reflect the current embedding.
- Run relevant Python tests and CTest; use AddressSanitizer/UndefinedBehaviorSanitizer for native memory or indexing changes where supported. Explain any changed numerical contract or approximation.

Public behavior is defined by the source, API documentation, tests, and README. Historical local research notes are intentionally not part of the repository.

## Benchmark integrity

- Declare data provenance, permissions, split, preprocessing, seeds, architecture, optimization budget, metrics and selection rule before comparing results.
- Fit preprocessing/reference features on training data only. Keep final test data out of parameter selection. For HAR selection, use subject-grouped validation within the official training split.
- Class labels must not enter unsupervised training or hyperparameter selection. Labels may be used afterward for a clearly labeled supervised probe. Subject IDs or object/angle annotations may define a documented split and final diagnostics, not a hidden training target.
- Use identical declared reference features, evaluation IDs and metric definitions across methods. Distinguish subset PH from global PH, raw from scale-aligned metrics, and sampling diagnostics from population guarantees.
- Report every declared method/seed and all failures, timeouts and retries. Do not select the best seed, relabel a pilot as final, or silently replace an unavailable upstream method with an independent implementation. Keep single-seed results separate from multi-seed summaries.
- Disclose unequal compute, model capacity, backend adaptations, startup/JIT time and the scope of memory measurements. A resource budget is not a total RSS guarantee.
- Keep historical results tied to their original code/configuration. New objectives, preprocessing or tuning require a new protocol/result, not an overwrite presented as the same experiment.

External TopoAE/TopoAE++/RTD-AE adapters are optional research tools, not DataRefiner code or proof of an untouched paper reproduction. Record the upstream commit/hash and every adaptation; keep external source, headers and binaries outside this repository.

## Submitting a change

Provide a focused description, motivation, tests run and limitations. Add a regression test for bug fixes and update affected API documentation. Disclose borrowed material and dependency/license changes. Contributions intended for inclusion in the project's MIT code must be compatible with that license; do not submit material you cannot authorize for that use.

Do not include datasets, model checkpoints, embeddings, raw logs, local-machine paths, credentials, caches or compiled binaries. Inspect JSON and report metadata, not just source code. Publish only reviewed, bounded result summaries; `.gitignore` alone is not a source-archive privacy policy. See [distribution checker](tools/check_dist.py).

Report suspected vulnerabilities using [SECURITY.md](SECURITY.md), not a public issue containing exploit details or private data.
