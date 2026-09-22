# Graph-core benchmarks: two separate protocol versions

**Current `GraphEmbedding` prediction defaults to `mapping_domain='train_hull'`.**
The original unbounded confirmation remains historical evidence; it has **not**
been rewritten as a result for the new default. Explicit
`mapping_domain='unbounded'` retains the historical package behavior.

| Protocol | Published artifact | What was executed |
|---|---|---|
| **unbounded-v1** | [`graph_core_confirmation.json`](../results/graph_core_confirmation.json) | Original30 records: three seeds of strong/direct+A/UMAP on Fashion/HAR/COIL, PCA once per dataset |
| **compact-v2 followup** | [`graph_core_compact_confirmation.json`](../results/graph_core_compact_confirmation.json) | Separate9 direct+A mapping-only arms, same frozen layouts/source15 neighborhoods; refine only outside-TRAIN-hull outputs |

The original 30-record JSON, source/provenance manifest, unbounded numerical
kernels, strong configs and private completed run are unchanged. The former CLI
source is retained in `history/confirm_graph_core_v1.py`; its detailed protocol
README is preserved as [`UNBOUNDED_V1.md`](UNBOUNDED_V1.md). The archived entrypoint
is a byte-preserved source snapshot, **not** an executable at its relocated path;
use the current CLI's `--protocol unbounded-v1` route instead.

The old source manifest records original filenames as historical snapshots.
Exact old figure bytes now also live at
`assets/graph-core-unbounded-{fashion,har,coil}.png`. The compact aggregate links
these retained files and their hashes; the old manifest is not silently edited
to describe different pictures or scores.

## Why a separate well-posedness followup?

The unconstrained conditional Cauchy objective can yield extreme finite query
positions outside the fitting layout—even with an optimizer success flag or a
small gradient. The historical Fashion/HAR failure figures retain that evidence.
A separate **TRAIN-internal validation study** motivated the authorized fixed
TRAIN-convex-hull domain. Official TEST had already been seen; neither protocol
is virgin external validation, and no post-TEST parameter search is claimed.

The compact method is **optimization on a TRAIN-derived compact domain**, not
coordinate clipping, visual cropping, dropping outliers or refitting the layout:

1. Keep the frozen layout, source15 query neighbors/probabilities, TRAIN mean and
   layout edge-median unit, and cached phase-I outputs.
2. Validate every cached phase-I exact all-anchor objective/gradient against the
   original diagnostic record.
3. If a normalized output is inside the TRAIN hull, copy its **physical bytes**
   unchanged. Otherwise use the exact production `CompactMap.solve(old,ids,p,bary)`
   API, constrained SLSQP with the feasible barycenter included among incumbents.
4. Retain every finite feasible incumbent, including on solver failure; never
   retry, delete a query or choose a favorable seed. Source/domain tolerance and
   solver are hash-frozen. A local solve is **not** a global-optimum certificate.

The production helper is loaded by file through
`support.py::production_graph_mapping()`; no copied alternative convex optimizer
or package-default ambiguity is introduced. TRAIN-hull compactness provides an
attained minimum for the continuous objective, not universal neighborhood,
semantic or topology preservation.

## Completed compact results—all nine arms retained

All **40,281** TEST outputs across datasets/seeds were scored. Exactly **430**
outside outputs changed; **39,851** inside physical rows remain bitwise identical.
All430 SLSQP phase-II solves reported success, none excluded; original phase-I
failures remain recorded. All resulting outputs are finite and hull-feasible
within the documented numerical tolerance.

| Dataset | Compact accuracy mean ± sample std | Compact overlap@15 mean ± sample std |
|---|---:|---:|
| Fashion | .784867 ± .000764 | .172743 ± .002185 |
| HAR | .830223 ± .002383 | .197331 ± .000959 |
| COIL | .870833 ± .002083 | .789907 ± .003625 |

This is **not universal metric improvement**. Fashion accuracy changes by
-.0002/-.0001/0 across seeds0/1/2; HAR gains are 0/+.001697/+.000339; COIL accuracy
is unchanged. The new JSON retains all k5/15/50 geometry changes, improved/
worsened/unchanged query counts, correct classifications gained/lost, clustering
deltas, and original strong/UMAP/unbounded-A comparisons. In particular:

- Fashion compact ARI/NMI remain below UMAP.
- HAR compact ARI/NMI remain below the old strong graph control; mean accuracy
  remains below UMAP.
- COIL compact ARI/NMI remain below UMAP despite higher accuracy/neighbor overlap.
- “Strong COIL” means the existing documented **stress600 control**, not the best
  possible COIL setting.

### Honest cost accounting

The historical unbounded transforms totaled **450.743s**; they were not rerun.
Outside-only refinement added **2.897s**. The completed followup separately spent
40.984s validating all cached objectives/gradients,13.375s on fresh API parity,
62.407s scoring, and122.273s total wall time including other overhead.

**Refinement-only time is not full inference time.** The current default still
includes phase I. “Old measured unbounded time + separately measured refinement”
is a component-cost account, **not a fresh end-to-end timing**. Shared preprocessing,
index construction and layout fitting remain additional. Fashion exact-A
inference remains substantially slower than UMAP and the strong MLP. Timing
scopes/objectives differ; do not advertise a matched-compute speedup.

## Reproducibility and label barriers

The original protocol still uses full official Fashion60000/10000 and
subject-disjoint HAR7352/2947, plus COIL960/480 held-out views of seen objects.
Fashion/COIL PCA64 was fitted on full TRAIN, not a50k internal-pilot preprocessor;
HAR standardization is TRAIN-only. Shared audited ANN15 / author16-neighbor API
adaptation, exact fullTRAIN-reference geometry, three seeds and original failures
are unchanged. See [the preserved original protocol](UNBOUNDED_V1.md) for all
cache preparation, optional dependency, native isolation and numerical details.
No command implicitly downloads data.

The compact followup requires a completed **local unbounded benchmark output**.
It hashes old arrays, normalization, neighbor files, diagnostics, query metrics
and source metadata before execution. It never fits a layout, rebuilds a graph
or reruns phase I. A **new all-nine complete-finite output/hash barrier** precedes
labels, pose annotations or saved cluster predictions. It reuses the original
TRAIN KMeans centers and verifies every old TEST cluster assignment, rather than
refitting centers using TEST or the new predictions. TRAIN geometry is reused;
the original fixed TEST geometry queries are rescored against fullTRAIN.

```sh
# Existing, explicit historical route (legacy CLI default remains unbounded-v1):
python benchmarks/confirm_graph_core.py --protocol unbounded-v1 \
  --preregister --output outputs/unbounded-new
python benchmarks/confirm_graph_core.py --protocol unbounded-v1 \
  --run --output outputs/unbounded-new

# Synthetic compact API/protocol check only; no dataset or mapping rerun:
python benchmarks/confirm_graph_core.py --protocol compact-v2 \
  --preflight-only --output outputs/compact-preflight

# NEW separately registered fixed9 mapping-only followup of a completed run:
python benchmarks/confirm_graph_core.py --protocol compact-v2 --preregister \
  --prior-output outputs/unbounded-new --output outputs/compact-new
python benchmarks/confirm_graph_core.py --protocol compact-v2 --run \
  --output outputs/compact-new

python -m pytest -q tests/test_graph_core_protocol.py tests/test_graph_core_compact_protocol.py
```

The CLI protocol names are deliberately separate from the package predictor's
current default: old benchmark commands cannot silently acquire new predictions.
Registration/source/environment mismatch fails; existing started directories
cannot be retried. The compact supervisor has a600s total cap, one numerical
thread, retains partial/failure records and blocks labels for incomplete output
arms. It does not claim an RSS cap or guaranteed optimizer convergence.

## Publication and figures

Only exports/plots and synthetic protocol checks were executed for this public
update—**no new real fit or map**. The completed official compact run and all old
inputs are retained read-only. `compact_publish.py` exports a nine-record
allowlisted schema with hashes under logical roles, scalar metrics/diagnostics,
changes and costs; no sample IDs, coordinates, predictions, model paths or logs.
Original30 and new9 records are never merged into a fictitious single run.

Current `assets/graph-core-{fashion,har,coil}.png` use **fixed seed0, ALL TEST points
at full extents**, complete class legends and seed0 numbers. Only direct+A is
replaced by its saved TRAIN-hull followup; PCA/strong/UMAP remain original controls.
There is no coordinate clipping, axis cropping, asinh warp, point exclusion or
best-seed selection. The old two-row failure/full-range/detail figures are retained
under `assets/graph-core-unbounded-*`; old three-seed grids also remain local.

To export/redraw saved completed outputs without invoking a model:

```sh
export GRAPH_PRIOR=outputs/unbounded-new GRAPH_COMPACT=outputs/compact-new
python - <<'PY'
import os
from benchmarks.graph_core.compact_publish import write_compact
from benchmarks.graph_core.compact_figures import render_compact
old, new = os.environ['GRAPH_PRIOR'], os.environ['GRAPH_COMPACT']
write_compact(new, old, new + '/sanitized_compact_confirmation.json')
for dataset in ('fashion', 'har', 'coil'):
    render_compact(new, old, dataset, new + '/seed0-' + dataset + '.png')
PY
```

Review local results before publishing. Do not overwrite historical aggregate or
source records. Raw/model artifacts and trusted-local pickle remain nonpublic.
These fixed benchmark protocols are independent implementations using established
graph/UMAP/Cauchy ideas, not an official DeepTDA reproduction or a topology guarantee.

### Source history versus current package hardening

The compact JSON records the authorized **as-run** helper/predictor hashes and a
separate current-source comparison. Subsequent production validation hardening
changed those file hashes after the completed followup; this export does not
pretend the new files have the old hashes or that the benchmark was rerun.
Original frozen data and followup source/closeout records remain unchanged.
The current exact helper API is checked with synthetic fixtures; a future
`--preregister --protocol compact-v2` pins the then-current source and environment.
