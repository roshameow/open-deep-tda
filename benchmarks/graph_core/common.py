"""Public fixed-confirmation contracts; no input payload is read on import."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import sys
import json
import hashlib
import zipfile
import io
import datetime
import pickle
import importlib.util
from pathlib import Path
import numpy as np
if __package__:
    from . import kernels as numeric
    from .kernels import rerank,audit
else:
    import kernels as numeric
    from kernels import rerank,audit
CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
OUT=Path(os.environ.get('GRAPH_CORE_OUTPUT',str(ROOT/'outputs/graph-core-confirmation'))).resolve()
DATASETS={
 'fashion':dict(path='outputs/v02-fashion_mnist/features.npz',train='train.npy',test='test.npy',annotations='outputs/v02-fashion_mnist/annotations.npz',train_labels='train_labels.npy',test_labels='test_labels.npy',n=60000,m=10000,d=64),
 'har':dict(path='data/uci_har/features.npz',train='X_train.npy',test='X_test.npy',annotations='data/uci_har/features.npz',train_labels='y_train.npy',test_labels='y_test.npy',n=7352,m=2947,d=561),
 'coil':dict(path='outputs/v02-coil20/features.npz',train='train.npy',test='test.npy',annotations='outputs/v02-coil20/annotations.npz',train_labels='train_labels.npy',test_labels='test_labels.npy',n=960,m=480,d=64)}
PLAN=dict(protocol='graph-core-official-confirmation-public-port-v1',datasets=DATASETS,
 seeds=[0,1,2],methods=['strong','directA','umap'],
 preprocessing='Fashion/COIL: existing full-TRAIN-fitted unwhitened PCA64 cache and TRAIN distance scalar. HAR: official subject split, float64 TRAIN mean/std then float32; original strong standardizes raw input identically. Preserve source feature dtype; no 50k pilot PCA.',
 strong='Frozen strong_configs.json: Fashion NCE7200; HAR graph2400; COIL documented stress600, NOT best possible. Only process-local shared graph provider, no loss changes.',
 direct='300 Cauchy SGD sweeps,5 negatives,lr1 linear decay,clip4. Component-aware V2 initializer. Standalone public graph-worker numerical functions loaded by file, not package; 4e9 event cap only, no RNG or floating update change.',
 A='Exact all-TRAIN conditional Cauchy KL; TRAIN layout edge-median unit; L-BFGS-B maxiter100,maxls30,ftol1e-12,gtol1e-8. All TEST queries independent, finite failed candidates kept, no retry.',
 umap='Author UMAP16 including self+15 nonself,300epochs,min_dist.1,spread1,lr1,negative5,spectral,seed=transform_seed,n_jobs1,force_approximation_algorithm=True to retain supplied COIL graph.',
 graph='Shared per dataset/seed NNDescent31, float64 rerank15, persistent index, exact256-fit/256-TEST audit vsfullTRAIN for fit/A/author, mean tie-aware recall>=.9; no fallback.',
 evaluation='512 fixed TRAIN queries and min(1024,NTEST) TEST queries vsFULLTRAIN, self excluded for TRAIN; COILall480TEST. k5/15/50 overlap/trust/continuity. AllTEST15NN; TRAIN KMeans n_init10,seed,cluster count from TRAIN labels after fit, TEST ARI/NMI. COIL annotated pose adjacency@2 after fit.',
 labels='No labels/pose/subjects until all30 fitting/control jobs terminal and every returned coordinate array hash-frozen. No labels passed to fitting.',
 input_freeze='TRAIN-member hashes and cache ZIP member CRC/size metadata pinned before TEST; TEST and labels read only in permitted phases, never implicit downloads.',
 controls='Full-SVD PCA2 once per dataset; not three replicated controls.',
 statistics='All3 seeds, mean and sample std(ddof1); failed seeds retained, incomplete summaries labeled; PCA std=null.',
 resources=dict(wall_seconds=2700,parallel_children=2,thread_per_child=1,sampled_combined_rss_gib=8,graph_timeout=240,fit_timeouts=dict(strong=480,directA=900,umap=240,pca=90),evaluation_timeout=180,prepare_timeout=90),
 caveats='Historically seen official TEST, not virgin external validation; no tuning/restarts, no topology or official DeepTDA equivalence; objectives/compute differ. Published historical data belong to original frozen run, not this relocated path.')

def environment_snapshot():
    import platform
    from importlib.metadata import version,PackageNotFoundError
    packages={}
    for name in ('numpy','scipy','scikit-learn','torch','numba','llvmlite','matplotlib','umap-learn','pynndescent'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    return dict(platform=platform.platform(),packages=packages)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.new');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(p)
def member(path,key):
    with zipfile.ZipFile(path) as archive:payload=archive.read(key)
    return np.load(io.BytesIO(payload),allow_pickle=False),hashlib.sha256(payload).hexdigest()
def descriptor(path):
    p=Path(path);stat=p.stat()
    with zipfile.ZipFile(p) as archive:
        members={i.filename:dict(crc=i.CRC,size=i.file_size,compressed=i.compress_size) for i in archive.infolist()}
    return dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,members=members)
def source_files():
    """Logical roles in registrations, never author-install absolute path keys."""
    paths={f'benchmark/{p.name}':p for p in CODE.glob('*.py')}
    paths['benchmark/strong_configs.json']=CODE/'strong_configs.json'
    paths['benchmark/cli']=ROOT/'benchmarks/confirm_graph_core.py'
    for pattern in ('*.py','*.so'):
        paths.update({f'production/{p.name}':p for p in (ROOT/'python/open_deep_tda').glob(pattern)})
    for directory,pattern in [('src','*.cpp'),('include','*.hpp')]:
        paths.update({f'native/{p.relative_to(ROOT)}':p for p in (ROOT/directory).rglob(pattern)})
    for package in ('umap','pynndescent'):
        spec=importlib.util.find_spec(package)
        if spec is None: raise RuntimeError(f'Install optional {package} before registration; no downloads are performed')
        directory=Path(spec.origin).parent
        paths.update({f'author/{package}/{p.relative_to(directory)}':p for p in directory.rglob('*.py')})
    return paths
def sources(): return {role:sha(path) for role,path in sorted(source_files().items())}
def registration():
    r=json.loads((OUT/'preregistration.json').read_text())
    if r['plan']!=PLAN or r['sources']!=sources(): raise RuntimeError('Frozen source/plan mismatch; use a new registration, never repair this run')
    if 'environment' in r and r['environment']!=environment_snapshot(): raise RuntimeError('Frozen numerical environment changed')
    return r
def validate_output(path):
    path=Path(path).resolve()
    for forbidden in (ROOT/'benchmarks',ROOT/'assets',ROOT/'python',ROOT/'tests'):
        if path==forbidden or forbidden in path.parents:raise ValueError('Output must be outside source/public artifact directories')
    return path
def require_label_barrier(directory=None):
    directory=OUT if directory is None else Path(directory)
    barrier=json.loads((directory/'embeddings_frozen.json').read_text())
    if barrier.get('terminal_fit_jobs')!=30 or len(barrier.get('fit_statuses',{}))!=30:
        raise RuntimeError('All30 fit/control jobs must be terminal before labels')
    if any(v.get('status') not in ('completed','failed','skipped','timeout') for v in barrier['fit_statuses'].values()):
        raise RuntimeError('A fit job is still active')
    for path,h in barrier['embeddings'].items():
        if sha(path)!=h:raise RuntimeError('Frozen embedding changed before label access')
    return barrier

def load_graph_worker():
    """Do not import the package: its __init__ may load Torch."""
    if 'torch' in sys.modules:raise RuntimeError('Graph numerical worker must be isolated from Torch')
    path=ROOT/'python/open_deep_tda/_graph_worker.py'
    spec=importlib.util.spec_from_file_location('benchmark_standalone_graph_worker',path)
    module=importlib.util.module_from_spec(spec)
    # Numba resolves the function's defining module during compilation.
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def base(d):return OUT/d
def graphdir(d,s):return base(d)/('graph'+str(s))
def armdir(d,m,s):return base(d)/(m+'-seed'+str(s))
def load_data(d):return np.load(base(d)/'train.npy',allow_pickle=False),np.load(base(d)/'test.npy',allow_pickle=False)
def save_pickle(obj,path):
    with Path(path).open('wb') as f:pickle.dump(obj,f,protocol=4)
    dump(str(path)+'.sha256.json',dict(sha256=sha(path),warning='trusted local pickle only'))
def load_pickle(path):
    if sha(path)!=json.loads(Path(str(path)+'.sha256.json').read_text())['sha256']:raise RuntimeError('Local pickle checksum mismatch')
    with Path(path).open('rb') as f:return pickle.load(f)
def umap_options(seed):return dict(n_neighbors=16,n_components=2,metric='euclidean',min_dist=.1,spread=1.,n_epochs=300,negative_sample_rate=5,learning_rate=1.,init='spectral',random_state=seed,transform_seed=seed,n_jobs=1,force_approximation_algorithm=True)
