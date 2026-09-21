"""One isolated image reducer on common train/test features; no labels are loaded.

Invoked by benchmark_images.py. v0.2 engineering settings are explicit. Upstream TopoAE uses unchanged official model/loss
source; only module import plumbing, input width and the training loop adapt.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['pca', 'deep_tda', 'no_h1', 'upstream_topoae', 'umap'])
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--upstream-epochs', type=int, default=20)
    parser.add_argument('--upstream-root')
    parser.add_argument('--config', required=True, help='explicit common TDAConfig JSON')
    args = parser.parse_args()
    import numpy as np
    from threadpoolctl import threadpool_limits
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    with np.load(args.features, allow_pickle=False) as data:
        train, test = data['train'], data['test']
    metadata = {'method': args.method, 'seed': args.seed, 'n_train': len(train), 'n_test': len(test),
                'features': train.shape[1], 'input_feature_file': str(Path(args.features).resolve())}
    with threadpool_limits(limits=1):
        if args.method == 'pca':
            from sklearn.decomposition import PCA
            model = PCA(n_components=2, svd_solver='full')
            started = time.perf_counter(); z_train = model.fit_transform(train)
            metadata['fit_seconds'] = time.perf_counter()-started
            started = time.perf_counter(); z_test = model.transform(test)
            metadata['test_transform_seconds'] = time.perf_counter()-started
            metadata['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
        elif args.method in ('deep_tda', 'no_h1'):
            from open_deep_tda import DeepTDA
            configuration = json.loads(Path(args.config).read_text())
            configuration.update(standardize=False, missing_indicators=False, seed=args.seed, steps=args.steps)
            if args.method == 'no_h1': configuration['lambda_h1'] = 0.
            model = DeepTDA(configuration)
            started = time.perf_counter(); model.fit(train)
            metadata['fit_seconds'] = time.perf_counter()-started
            # Undo the estimator's extra scalar normalization to restore common units.
            z_train = model.embedding_ * model.reference_scale_
            started = time.perf_counter(); z_test = model.transform(test) * model.reference_scale_
            metadata['test_transform_seconds'] = time.perf_counter()-started
            metadata['config'] = model.config.to_dict()
            metadata['coverage'] = model.training_coverage_
            metadata['neighbor_graph'] = model.neighbor_diagnostics_
            metadata['internal_reference_scale'] = model.reference_scale_
            model.save(out/'model.pt')
            (out/'history.json').write_text(json.dumps(model.history_, indent=2))
        elif args.method == 'umap':
            # Standard UMAP, not its unrelated optional TensorFlow plugin.
            sys.modules['tensorflow'] = None
            import umap
            model = umap.UMAP(n_neighbors=15, min_dist=.1, n_components=2,
                             random_state=args.seed, transform_seed=args.seed, n_jobs=1)
            started = time.perf_counter(); z_train = model.fit_transform(train)
            metadata['fit_seconds'] = time.perf_counter()-started
            started = time.perf_counter(); z_test = model.transform(test)
            metadata['test_transform_seconds'] = time.perf_counter()-started
            metadata['version'] = umap.__version__
            metadata['timing_note'] = 'fit includes any JIT first-use overhead; input/import excluded'
        else:
            if not args.upstream_root:
                raise ValueError('upstream TopoAE requires --upstream-root, no substitute baseline is used')
            root = Path(args.upstream_root).resolve()
            commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip()
            if commit != '203e94a69c5f9cda049b9c3985b7c2b1e39ca922':
                raise ValueError('upstream checkout differs from audited/pinned TopoAE commit')
            for name, path in [('src',root/'src'),('src.models',root/'src/models')]:
                package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
            import torch
            torch.set_num_threads(1);torch.manual_seed(args.seed)
            official=importlib.import_module('src.models.approx_based')
            model=official.TopologicallyRegularizedAutoencoder(lam=1., autoencoder_model='DeepAE',
                ae_kwargs={'input_dims':(train.shape[1],)}, toposig_kwargs={'match_edges':'symmetric'})
            optimizer=torch.optim.Adam(model.parameters(), lr=.001)
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(train),torch.arange(len(train))),
                batch_size=128,shuffle=True,drop_last=True)
            seen=np.zeros(len(train),dtype=bool);history=[];updates=0
            started=time.perf_counter()
            for epoch in range(args.upstream_epochs):
                model.train(); epoch_losses=[]
                for x, ids in loader:
                    seen[ids.numpy()]=True
                    optimizer.zero_grad(set_to_none=True)
                    loss, parts=model(x)
                    if not torch.isfinite(loss):raise RuntimeError('official TopoAE returned nonfinite loss')
                    loss.backward();optimizer.step();updates+=1;epoch_losses.append(float(loss.detach()))
                history.append({'epoch':epoch,'mean_loss':float(np.mean(epoch_losses))})
            metadata['fit_seconds']=time.perf_counter()-started
            def encode(X):
                model.eval()
                with torch.no_grad():return np.concatenate([model.encode(torch.from_numpy(X[i:i+512])).numpy() for i in range(0,len(X),512)])
            z_train=encode(train);started=time.perf_counter();z_test=encode(test)
            metadata['test_transform_seconds']=time.perf_counter()-started
            metadata.update(upstream_commit=commit,epochs=args.upstream_epochs,updates=updates,
                batch_size=128,unique_training_samples=int(seen.sum()),lambda_topology=1.,learning_rate=.001,
                implementation='unchanged official model/loss; input width and surrounding import/training harness adapted; not original paper hyperparameter protocol')
            torch.save({'model':model.state_dict(),'input_dims':train.shape[1],'upstream_commit':commit},out/'model.pt')
            (out/'history.json').write_text(json.dumps(history,indent=2))
    if z_train.shape!=(len(train),2) or z_test.shape!=(len(test),2) or not np.isfinite(z_train).all() or not np.isfinite(z_test).all():
        raise RuntimeError('invalid reducer outputs')
    np.savez_compressed(out/'embedding.npz',train=z_train,test=z_test)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata['process_peak_rss_mib']=float(rss/(1024**2 if sys.platform=='darwin' else 1024))
    metadata['peak_memory_scope']='isolated worker process high-water RSS, including imports/data/model, not only PH'
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    print(json.dumps({'method':args.method,'seed':args.seed,'fit_seconds':metadata['fit_seconds']}),flush=True)


if __name__=='__main__':
    main()
