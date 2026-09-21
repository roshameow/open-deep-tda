"""Same-checkpoint inference/storage profile; never changes learned parameters."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from open_deep_tda import DeepTDA


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--features',required=True)
    p.add_argument('--output',default='outputs/estimator-profile');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with np.load(args.features,allow_pickle=False) as a:X=a['test']
    model=DeepTDA.load(args.model);torch.set_num_threads(1)
    measurements=[];predictions=[]
    with threadpool_limits(limits=1):
        for batch in [256,4096]:
            model.config.inference_batch_size=batch
            model.transform(X)
            durations=[]
            for _ in range(7):
                started=time.perf_counter();prediction=model.transform(X)
                durations.append(time.perf_counter()-started)
            predictions.append(prediction)
            measurements.append({'batch_size':batch,'median_seconds':float(np.median(durations)),'seconds':durations})
    full=model.save(out/'full.pt')
    compact=model.save(out/'inference.pt',include_training_data=False)
    restored=DeepTDA.load(compact)
    diff=float(np.max(np.abs(model.transform(X)-restored.transform(X))))
    report={'scope':'same learned model, CPU one thread; warm public transform including preprocessing; not training speed or accuracy',
        'queries':len(X),'features':X.shape[1],'measurements':measurements,
        'max_abs_difference_between_batch_partitions':float(np.max(np.abs(predictions[0]-predictions[1]))),
        'full_checkpoint_bytes':full.stat().st_size,'inference_checkpoint_bytes':compact.stat().st_size,
        'compact_reload_max_abs_error':diff,
        'privacy_note':'inference checkpoint omits training rows/coordinates/histories; learned weights/statistics are still data-derived'}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
