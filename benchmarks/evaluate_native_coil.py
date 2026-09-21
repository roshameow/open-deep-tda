"""Evaluate a COMPLETED full common-split native TopoAE++ run, without tuning it."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from benchmark_images import label_probe
from benchmark_har import finite_diagrams, ph_metrics, write_json
from open_deep_tda.population_metrics import population_geometry
from open_deep_tda.topology import distance_matrix
from open_deep_tda.cycle_metrics import evaluate_rotation_orbits


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-root',default='outputs/v02-coil20')
    p.add_argument('--native-root',default='outputs/topoaepp-coil-native-final');args=p.parse_args()
    reference=Path(args.reference_root);native=Path(args.native_root)
    meta=json.loads((native/'metadata.json').read_text());prepared=json.loads((reference/'prepared.json').read_text())
    if meta.get('status')!='success' or meta.get('actual_updates')!=1000 or not meta.get('full_training_set'):
        raise ValueError('not a completed FULL960 1000-update run; do not relabel a pilot/timeout')
    if meta['features_content_sha256']!=prepared['feature_sha256'] or meta['train_row_indices']!=list(range(960)):
        raise ValueError('native features/row identities differ from the shared protocol')
    with np.load(reference/'features.npz',allow_pickle=False) as a:train,test=a['train'],a['test']
    with np.load(reference/'annotations.npz',allow_pickle=False) as a:ann={key:a[key] for key in a.files}
    with np.load(native/'embedding.npz',allow_pickle=False) as a:ztrain,ztest=a['train'],a['test']
    if ztrain.shape!=(960,2) or ztest.shape!=(480,2):raise ValueError('wrong native embedding dimensions')
    X,Z=np.concatenate([train,test]),np.concatenate([ztrain,ztest])
    ids=json.loads((reference/'evaluation_ids.json').read_text())
    q=np.asarray(ids['query_ids']);ph_ids=np.asarray(ids['ph_ids'])
    with threadpool_limits(limits=1):
        DX,DZ=distance_matrix(X[ph_ids]),distance_matrix(Z[ph_ids])
        denominator=float(np.sum(DZ*DZ));scale=float(np.sum(DX*DZ)/denominator) if denominator else 1.
        result={'method':'TopoAE++ official core, native CPU/no-CGAL compile adapter','status':'ok','seed':0,'n_runs':1,
            'scope':'shared train960/test480 PCA64, official model and asymmetric-cascade loss; independent driver/eval-mode inference; not untouched ParaView protocol',
            'training_metadata':str((native/'metadata.json').resolve()),'actual_updates':1000,
            'fit_plus_source_ph_seconds':meta['completion']['fit_seconds']+meta['completion']['source_ph_seconds'],
            'native_runtime_peak_rss_mib':meta['completion']['peak_rss_bytes']/1024**2,
            'reload_max_error':meta['completion']['reload_max_error'],
            'heldout_population_geometry':population_geometry(X,Z,query_indices=q,max_queries=len(q),k=15),
            'heldout_label_probe':label_probe(ztrain,ztest,ann['train_labels'],ann['test_labels']),
            'diagnostic_scale_alignment':scale,
            'ph_audit':ph_metrics(finite_diagrams(DX),finite_diagrams(DZ),scale),
            'rotation_orbits':evaluate_rotation_orbits(X,Z,ann['object_ids'],ann['angles_degrees'],
                query_mask=np.arange(len(X))>=len(train),global_scale=scale),
            'acknowledgment':meta['acknowledgment'],
            'citation':['Topological Autoencoders++: Fast and Accurate Cycle-Aware Dimensionality Reduction (TVCG 2025)',
                        'Julien Tierny et al., The Topology ToolKit']}
    write_json(native/'quality.json',result)
    print(json.dumps({'trust':result['heldout_population_geometry']['trustworthiness'],
        'test_accuracy':result['heldout_label_probe']['accuracy'],'rotation':result['rotation_orbits']['summary']},indent=2))


if __name__=='__main__':main()
