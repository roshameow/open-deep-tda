"""Compact summaries; never discard failed runs or select best seeds."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    root=Path(root);data=json.loads((root/'results.json').read_text())
    if not data['records'] or any(r['status']!='ok' for r in data['records']):
        raise ValueError('refusing to hide failures or summarize empty results')
    getters={
        'trustworthiness':lambda r:r['heldout_population_geometry']['trustworthiness'],
        'continuity':lambda r:r['heldout_population_geometry']['continuity'],
        'knn_overlap':lambda r:r['heldout_population_geometry']['knn_overlap'],
        'test_accuracy':lambda r:r['heldout_label_probe']['accuracy'],
        'macro_f1':lambda r:r['heldout_label_probe']['macro_f1'],
        'fit_seconds':lambda r:r['fit_seconds'],
        'parent_peak_rss_mib':lambda r:r['process_peak_rss_mib'],
        'h1_aligned_cost':lambda r:r['ph_audit']['h1']['aligned_normalized_squared_transport']}
    if data['protocol']['prepared']['dataset']=='coil20':
        getters.update({
            'angle_adjacency_recall':lambda r:r['rotation_orbits']['summary']['embedding_adjacent_recall'],
            'mean_neighbor_angle_degrees':lambda r:r['rotation_orbits']['summary']['embedding_neighbor_angle_degrees'],
            'strong_h1_objects':lambda r:r['rotation_orbits']['summary']['target_strong_h1_objects'],
            'mean_strict_crossings':lambda r:r['rotation_orbits']['summary']['mean_strict_crossings']})
    rows=[]
    for record in data['records']:
        row={'method':record['method'],'seed':record['seed'],'status':record['status']}
        row.update({key:float(get(record)) for key,get in getters.items()})
        if 'coverage' in record:row['coverage']=record['coverage']
        if 'neighbor_graph' in record:
            row['neighbor_audit']=record['neighbor_graph']['audit']
            row['ann_worker_memory']=record['neighbor_graph'].get('ann_worker',{})
        rows.append(row)
    summary=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        group=[r for r in rows if r['method']==method];item={'method':method,'runs':len(group)}
        for key in getters:
            values=[r[key] for r in group]
            item[key]={'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else None}
        summary.append(item)
    return {'protocol':data['protocol'],'reference':json.loads((root/'reference_evaluation.json').read_text()),
        'summary':summary,'per_run':rows,'std_definition':'sample standard deviation across fixed optimizer seeds; not a confidence interval',
        'scope':'all declared train/test images, common train-only PCA64; not raw-image SSL; unequal architecture/compute; no best-run selection'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    result=summarize(args.root);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False));print(path)


if __name__=='__main__':main()
