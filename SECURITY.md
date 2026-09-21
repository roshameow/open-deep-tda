# Security policy

Open Deep-TDA is experimental research software, not a hardened service or privacy-preserving system. There is no published security-support lifetime or guaranteed response time. Use current, patched dependencies and isolate research workloads from sensitive data and credentials.

## Reporting a vulnerability

Do not post exploit details, credentials, personal data or sensitive model/data files in public issues or pull requests.

Use [Security → Report a vulnerability](https://github.com/roshameow/open-deep-tda/security/advisories/new). GitHub private vulnerability reporting was enabled and verified for this repository during v0.3 publication preparation. If the option is unavailable, ask maintainers to establish a private channel using only a non-sensitive request, and withhold details until that channel is confirmed. No reporting email address or response-time guarantee is specified.

A private report should describe affected versions/commits, impact, environment and minimal reproduction steps using synthetic data. Redact secrets and identifying paths. Do not send a real dataset or checkpoint unless explicitly requested through the agreed private channel. If a credential has leaked, revoke or rotate it through its provider; deleting a public file alone is insufficient.

## Data and trust boundaries

- Normal local training does not upload data. Explicit dataset-download options contact external hosts; installing dependencies and fetching external research sources also require separate network access.
- CLI run directories can contain the original input, reference features, coordinates, sample IDs, labels used for reports, and diagnostics. Full checkpoints retain training reference features and embeddings as well as model state. `save(..., include_training_data=False)` omits training rows, coordinates and detailed reports, but retains learned weights and preprocessing statistics; this is not anonymization or differential privacy. HTML/SVG reports can disclose samples or annotations even though they work offline. None of these artifacts is anonymized automatically.
- `DeepTDA.load` uses PyTorch's restricted `weights_only=True` loading, and numeric input paths use `allow_pickle=False`. These are risk reductions, not a sandbox or a guarantee that hostile files are safe. Load only trusted inputs/checkpoints; malformed or oversized files can exhaust resources or expose dependency/native-library vulnerabilities. Do not disable restricted loading to accept an unknown file.
- External research adapters import or compile code from operator-supplied checkouts. Commit/hash checks identify expected content; they do not establish that code is safe. Subprocess isolation is not a security sandbox. Review upstream code and licenses, use an unprivileged isolated environment, and do not run unfamiliar install scripts with elevated privileges.
- Persistence, matching and neighbor-workspace budgets are algorithmic limits, not hard process-memory or wall-clock isolation. Enforce OS/container memory, CPU and time limits when processing untrusted or large inputs.
- Dataset checksums detect mismatches against recorded values, not permission to redistribute data. An observed download checksum is not a publisher signature. Keep dataset/model caches private and outside published source archives.

Before sharing any run or bug report, inspect all metadata and generated files for credentials, personal data, local paths and proprietary content. Ignored files may still enter manually created archives; review the actual publication payload.
