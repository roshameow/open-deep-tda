"""Isolated, same-input graph profiling; retains the old helper as a comparator."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',default='outputs/har-real/features.npz')
    p.add_argument('--output',default='outputs/neighbors-same-input');p.add_argument('--worker',choices=['legacy','exact','pynndescent'])
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not args.worker:
        rows=[]
        for method in ['legacy','exact','pynndescent']:
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
            run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--features',str(Path(args.features).resolve()),
                '--output',str(out.resolve()),'--worker',method],env=env,capture_output=True,text=True,timeout=180)
            (out/(method+'.log')).write_text(run.stdout+'\n'+run.stderr)
            if run.returncode:rows.append({'method':method,'status':'failed','reason':run.stderr[-2000:]})
            else:rows.append(json.loads((out/(method+'.json')).read_text()))
        (out/'summary.json').write_text(json.dumps({'measurements':rows,
            'scope':'same stored HAR train reference input; independent process per backend; RSS includes imports; ANN child RSS separately reported'},indent=2))
        print(out/'summary.json');return
    from threadpoolctl import threadpool_limits
    from open_deep_tda.neighbors import build_neighbor_graph, _audit
    from open_deep_tda.sampling import knn_graph
    with np.load(args.features,allow_pickle=False) as a:X=a['train']
    with threadpool_limits(limits=1):
        started=time.perf_counter()
        if args.worker=='legacy':
            edges,neighbors=knn_graph(X,15);diagnostics={}
        else:
            result=build_neighbor_graph(X,15,backend=args.worker,working_memory_mb=64,seed=0,audit_queries=64)
            edges,neighbors,diagnostics=result['edges'],result['neighbors'],result['diagnostics']
        seconds=time.perf_counter()-started
        # Report the same independent global audit even for the legacy routine.
        audit=_audit(np.asarray(X,dtype=float),neighbors,64,0,0)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    record={'method':args.worker,'status':'ok','samples':len(X),'features':X.shape[1],
        'graph_seconds':seconds,'process_peak_rss_mib':rss,'audit':audit,'backend_diagnostics':diagnostics}
    (out/(args.worker+'.json')).write_text(json.dumps(record,indent=2));print(args.worker,seconds,rss)


if __name__=='__main__':main()
