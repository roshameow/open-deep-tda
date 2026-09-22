"""Deterministic, all-row synthetic structural construction (12 rows).

Run after installation: python examples/construct_global_layout.py
Optional external proposal: python examples/construct_global_layout.py --auto

The default fixes a regular octagon's perimeter and thresholds in advance;
there is no solver dependency, dataset, file I/O, fitting, sampling or RNG use.
Four distant rows are included in every source/target check, not discarded.
The source is a finite nonnegative symmetric float64 distance matrix with zero
diagonal; sparse H1 uses closed thresholds on its represented entries, without
assuming a triangle inequality. Vertex IDs identify the same rows throughout.

Lifecycle: certify the fixed source family with structural_sparse_h1, then
construct_global_layout checks full same-ID H0 and the selected source/target
H1 family, plus its separate real-coordinate planar certificate. Success gives
a read-only (N,2) embedding; unsupported/failure gives embedding=None, never a
partial layout or proof of general infeasibility. This is not a full-H1/barcode
or chain-map certificate, an optimal layout, or a guarantee for omitted rows.
Only summary metadata is printed; no arrays or artifacts are written.

--auto explicitly consents to lazy Ripser/Ripser.py invocation, outside the
bounded core's resource guarantees. It fixes one bar and interval; any refusal
is reported without retry, switching bars or falling back to the known cycle.
All failures produce JSON and a nonzero exit status, including default failure.

Verified in the owned Python 3.9 regressions: this fixed 12-row default passes
source/target rank-one, all-pair H0 and planar checks, with H0 error about .049
within .05. Repeated runs agree and preserve Python/NumPy RNG state. A refused
optional proposal never falls back to the known perimeter. This synthetic
success is not a construction-feasibility or external-solver performance claim.
"""
import argparse
import json

import numpy as np

from open_deep_tda.structural_constructive import construct_global_layout, pairwise_planar
from open_deep_tda.structural_sparse_h1 import analyze_sparse_h1


def synthetic_case():
    """A fixed octagon plus four far points; no random state or external data."""
    angles = np.arange(8, dtype=np.float64) * (np.pi / 4.)
    octagon = 3. * np.column_stack((np.cos(angles), np.sin(angles)))
    guide = np.vstack((octagon, [[12., 0.], [0., 12.], [-12., 0.], [0., -12.]]))
    D = pairwise_planar(guide)
    cycles = (tuple((i, (i + 1) % 8) for i in range(8)),)
    return D, guide, cycles, 2.4, 3.


def run_example(*, auto=False):
    """Run exactly one declared construction; return JSON-compatible metadata."""
    D, guide, cycles, birth, survival = synthetic_case()
    report = dict(certified=False, embedding=None, vertices=len(D),
                  mode="auto" if auto else "known_cycle", synthetic=True,
                  scope="all supplied rows; selected family only", h0_tolerance=.05)
    if auto:
        # This import and the external invocation are opt-in, never a fallback.
        from open_deep_tda.structural_auto_witness import auto_global_witnesses
        proposal = auto_global_witnesses(D, allow_external=True)
        report['proposal'] = proposal.diagnostics
        if not proposal.certified:
            return dict(report, stage="proposal", reason=proposal.diagnostics['reason'])
        cycles, birth, survival = proposal.cycles, proposal.birth, proposal.survival
    report.update(birth=birth, survival=survival)
    source = analyze_sparse_h1(D, cycles, birth, survival)
    report.update(source_certified=source.certified, source_rank=source.rank)
    if not source.certified:
        return dict(report, stage="source", reason=source.reason)
    result = construct_global_layout(D, guide, cycles, birth, survival, h0_tolerance=.05)
    report.update(stage="construction", reason=result.reason, diagnostics=result.diagnostics)
    if not result.certified:
        return report
    # The constructor's certificate, not appearance of its candidate, is decisive.
    report.pop('embedding')
    report.update(certified=True, embedding_shape=list(result.embedding.shape),
                  h0_error=result.h0_error, target_rank=result.target.rank,
                  target_certified=result.target.certified, planar_certified=result.planar.certified)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--auto', action='store_true',
                        help='explicitly allow optional external Ripser; no retry on refusal')
    args = parser.parse_args(argv)
    try:
        report = run_example(auto=args.auto)
    except Exception as exc:
        # CLI boundary only: expose failure, never substitute a different run.
        report = dict(certified=False, embedding=None, stage="exception",
                      reason=f"{type(exc).__name__}: {exc}",
                      mode="auto" if args.auto else "known_cycle", synthetic=True)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report['certified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
