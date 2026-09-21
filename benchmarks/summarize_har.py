"""Compact, auditable summaries of completed benchmark_har.py runs."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(data):
    records=data['records']
    if not records or any(r['status']!='ok' for r in records):
        raise ValueError('refusing a success-only summary that hides failed or empty runs')
    extract={
        'trustworthiness':lambda r:r['heldout_population_geometry']['trustworthiness'],
        'continuity':lambda r:r['heldout_population_geometry']['continuity'],
        'knn_overlap':lambda r:r['heldout_population_geometry']['knn_overlap'],
        'test_accuracy':lambda r:r['heldout_label_probe']['accuracy'],
        'test_macro_f1':lambda r:r['heldout_label_probe']['macro_f1'],
        'h1_aligned_cost_1024':lambda r:r['ph_audit']['1024']['h1']['aligned_normalized_squared_transport'],
        'h1_top20_unmatched_1024':lambda r:r['ph_audit']['1024']['h1']['aligned_top20_source_unmatched_fraction'],
        'fit_seconds':lambda r:r['fit_seconds'],
        'process_peak_rss_mib':lambda r:r['process_peak_rss_mib']}
    per_run=[]
    for row in records:
        compact={'method':row['method'],'seed':row['seed'],'status':row['status']}
        compact.update({key:float(function(row)) for key,function in extract.items()})
        if 'coverage' in row:compact['training_coverage']=row['coverage']
        if 'upstream_commit' in row:compact['upstream_commit']=row['upstream_commit']
        per_run.append(compact)
    summary=[]
    for method in dict.fromkeys(r['method'] for r in records):
        group=[r for r in per_run if r['method']==method]
        entry={'method':method,'n_runs':len(group)}
        for key in extract:
            values=[r[key] for r in group]
            entry[key]={'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else None}
        summary.append(entry)
    return {'protocol':data['protocol'],'std_definition':'sample standard deviation across optimizer seeds (ddof=1), not a generalization confidence interval',
            'scope':'fixed official HAR split and fixed evaluation IDs; no test-based tuning; unequal architectures/compute',
            'summary':summary,'per_run':per_run}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='outputs/har-real/results.json')
    p.add_argument('--output',default='benchmarks/results/har_summary.json');args=p.parse_args()
    result=summarize(json.loads(Path(args.input).read_text()))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False));print(output)


if __name__=='__main__':main()
