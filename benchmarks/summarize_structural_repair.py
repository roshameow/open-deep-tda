"""Publishable scalar-only summary of bounded structural-repair diagnostics.

Never exports point IDs, chains, triangles, trees, coordinates or machine paths.
All failed/error rows remain present. This is not a DR performance benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(paths):
    output={'scope':'Small TRAIN-only supplied-domain structural diagnostics; not full-data quality or out-of-sample evidence',
            'inputs':[], 'runs':[]}
    for path in paths:
        raw=path.read_bytes(); report=json.loads(raw)
        solver=report['plan'].get('solver','sequential')
        output['inputs'].append(dict(solver=solver,result_sha256=hashlib.sha256(raw).hexdigest(),
                                     plan=report['plan'],source_hashes=report['source_hashes']))
        for row in report['runs']:
            cert=row.get('certificate',{})
            h1=cert.get('h1') or {}
            h0=cert.get('h0') or {}
            history=row.get('history',[])
            output['runs'].append(dict(dataset=row['dataset'],solver=solver,
                mode=row.get('mode'),status=row['status'],error_type=row.get('error_type'),
                n_vertices=row.get('n_vertices'),selected_count=row.get('selected_count'),
                source_interval_rank=row.get('image_rank'),
                birth_radius=row.get('birth_radius'),survival_radius=row.get('survival_radius'),
                before_h1_accepted=row.get('before_accepted'),
                before_h1_rank=row.get('before_h1',{}).get('surviving_rank'),
                after_h1_accepted=h1.get('accepted',h1.get('all_classes_independent')),
                after_h1_rank=h1.get('surviving_rank'),
                h0_required=row.get('mode')=='joint_h0_h1',
                before_h0_error=row.get('before_h0',{}).get('max_merge_error'),
                after_required_h0_error=h0.get('max_merge_error'),
                rounds=len(history),cuts=sum(len(r.get('cuts',[])) for r in history),
                unsuccessful_optimizer_calls=sum(r.get('optimizer_success') is False for r in history),
                seconds=row.get('seconds')))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.inputs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
