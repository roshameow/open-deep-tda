"""Frozen post-fit aggregation, independent audits and all-seed effect figures."""
if __package__:
    from .common import *
else:
    from common import *
from collections import defaultdict

def main():
    reg=registration();barrier=require_label_barrier()
    for p,h in barrier['embeddings'].items():assert sha(p)==h
    records=[];audits=[]
    for d in DATASETS:
        for m in ('pca','strong','directA','umap'):
            for s in ([0] if m=='pca' else [0,1,2]):
                root=armdir(d,m,s);path=root/'evaluation.json';fit=root/'result.json'
                if not path.exists() or json.loads(path.read_text()).get('status')!='completed':records.append(dict(dataset=d,method=m,seed=s,status='failed_or_missing',fit_result=json.loads(fit.read_text()) if fit.exists() else None,evaluation=json.loads(path.read_text()) if path.exists() else None));continue
                ev=json.loads(path.read_text());fr=json.loads(fit.read_text());q=json.loads((root/'query_metrics.json').read_text());pred=np.load(root/'evaluation_predictions.npz');spec=DATASETS[d]
                yt,_=member(ROOT/spec['annotations'],spec['train_labels']);yv,_=member(ROOT/spec['annotations'],spec['test_labels'])
                for split in ('train','test'):
                    assert q[split+'_ids']==reg['query_ids'][d][split]
                    for k in ('5','15','50'):
                        a=np.asarray(q[split][k]);assert np.max(abs(a.mean(0)-list(ev[split+'_metrics'][k].values())))<1e-12
                # Independent simple accuracy and contingency-table ARI/NMI.
                assert abs(float(np.mean(pred['classification']==yv))-ev['classification']['accuracy'])<1e-12
                labels=pred['cluster'];_,a=np.unique(yv,return_inverse=True);_,b=np.unique(labels,return_inverse=True);C=np.zeros((a.max()+1,b.max()+1),np.int64);np.add.at(C,(a,b),1);N=C.sum();choose=lambda v:np.sum(v*(v-1)/2);both=choose(C);row=choose(C.sum(1));col=choose(C.sum(0));expected=row*col/(N*(N-1)/2);den=(row+col)/2-expected;ari=(both-expected)/den if den else 1.
                prob=C/N;r=prob.sum(1);c=prob.sum(0);nonzero=prob>0;MI=float(np.sum(prob[nonzero]*np.log((prob/(r[:,None]*c[None,:]))[nonzero])));entropy=lambda v:float(-np.sum(v[v>0]*np.log(v[v>0])));hs=(entropy(r)+entropy(c))/2;nmi=MI/hs if hs else 1.
                assert abs(ari-ev['kmeans']['ari'])<1e-10 and abs(nmi-ev['kmeans']['nmi'])<1e-10
                audits.append(dict(dataset=d,method=m,seed=s,all_saved_metric_means=True,accuracy_checked=True,independent_ARI=ari,independent_NMI=nmi,source_hashes_current=True))
                row=dict(dataset=d,method=m,seed=s,status='completed',fit_seconds=fr['fit_seconds'],transform_seconds=fr['transform_seconds'],accuracy=ev['classification']['accuracy'],balanced_accuracy=ev['classification']['balanced_accuracy'],ari=ev['kmeans']['ari'],nmi=ev['kmeans']['nmi'])
                for split in ('train','test'):
                    for k in ('5','15','50'):
                        for metric,value in ev[split+'_metrics'][k].items():row[f'{split}_{metric}_{k}']=value
                if 'pose' in ev:row['pose_adjacency']=ev['pose']['embedding']['adjacent_view_recall'];row['reference_pose_adjacency']=ev['pose']['reference']['adjacent_view_recall']
                if m=='directA':row['A_audit']=ev['A_audit'];row['initializer']=fr['initializer'];row['optimizer_nonconvergences']=fr['optimizer_nonconvergences']
                records.append(row)
    summaries=[]
    for d in DATASETS:
        for m in ('pca','strong','directA','umap'):
            allrows=[r for r in records if r['dataset']==d and r['method']==m];ok=[r for r in allrows if r['status']=='completed'];expected=1 if m=='pca' else 3;keys=sorted(k for r in ok for k,v in r.items() if isinstance(v,(float,int)) and k!='seed');aggregate={}
            for k in set(keys):
                vals=[r[k] for r in ok if k in r];aggregate[k]=dict(mean=float(np.mean(vals)),sample_std=float(np.std(vals,ddof=1)) if len(vals)>1 else None,values=vals)
            summaries.append(dict(dataset=d,method=m,status='complete' if len(ok)==expected else 'incomplete',completed=len(ok),expected=expected,statistics=aggregate))
    dump(OUT/'results.json',dict(records=records,summary=summaries,all_outcomes_retained=True,official_TEST_historically_seen=True,source_hashes_unchanged=reg['sources']==sources()))
    actual={p:dict(expected=h,actual=sha(source_files()[p])) for p,h in reg['sources'].items()};assert all(v['actual']==v['expected'] for v in actual.values())
    dump(OUT/'verification.json',dict(status='passed',source_hashes=actual,embeddings_frozen_hashes_verified=True,metrics=audits,scope='Every completed arm: saved-query aggregate checks, independent16-query exact rank audit in evaluation worker, independent accuracy and contingency-table ARI/NMI here; every A query exact full-gradient oracle in evaluation worker. Failed/missing arms remain failures.'))
    lines=['# Fixed official-split benchmark reproduction','', 'All methods/configurations, input TRAIN hashes and source hashes frozen before first officialTEST feature access. Labels/pose accessed only after global30-fit-job terminal barrier and output hashing. Historical TEST use disclosed: this is **not virgin external validation**. No tuning, retries or best-seed choice.','', '| Dataset | Method | Seeds complete | TEST overlap15 mean ± sample std | AllTEST15NN accuracy | ARI | NMI | Fit s | Transform s |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        def f(k):
            v=row['statistics'].get(k)
            return '—' if v is None else f"{v['mean']:.6f}"+(f" ± {v['sample_std']:.6f}" if v['sample_std'] is not None else ' (one run)')
        lines.append(f"| {row['dataset']} | {row['method']} | {row['completed']}/{row['expected']} | {f('test_overlap_15')} | {f('accuracy')} | {f('ari')} | {f('nmi')} | {f('fit_seconds')} | {f('transform_seconds')} |")
    lines+=['','Machine-readable `results.json` retains every k5/15/50 fit/TEST overlap, trustworthiness, continuity, seed, regression, cost and failed arm. No partial-seed mean is labeled complete. PCA once per dataset has no invented variance. All classification rows cover every TEST output; geometry uses fixed512 TRAIN and1024 TEST queries against fullTRAIN (COILall480TEST). No full-population PH or topology claim.','', 'Models/checkpoints are under dataset/method-seed directories. Direct+A deployment needs frozen fit coordinates, source features/index, TRAIN unit/center and unchanged source; model.npz plus shared index and prepared inputs retain that state. Trusted local pickle only, hashes verified; do not load untrusted model files. Author UMAP is genuine library code with disclosed precomputed graph API adaptation.','', 'Runtime excludes shared graph/index and preprocessing in per-method fit/transform columns; those stage costs and sampled combined-process RSS are in checkpoint/runner_status. Native startup/JIT/serialization accounting differs, so no matched-compute speedup claim. Two isolated children maximum, one numerical thread each; no Torch/Numba co-import. Exact A normalizers and forces include all fitting anchors.','', 'COIL strong control is the existing documented600-step stress configuration, fixed before TEST; not an after-the-fact strongest setting. The disconnected initializer is component-aware V2 function, componentwise spectral with source-centroidPCA placement; no hidden structural-COIL arm.','', '## Verified experience / integrity','', f"{len(audits)} completed arm/control reports independently checked. All{len(actual)} registered source/native/author hashes matched actual bytes at final audit. Every A query has recorded optimizer status and full-gradient residual; finite unsuccessful candidates were retained, never retried. Inspect all failures and residuals before quality claims.",'', 'Raw inputs, coordinates, labels, logs and models are local-only. Use an allowlisted aggregate for publication. Runner retains checkpoints and final inventory; never overwrite a confirmation with a repaired run.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Figures only AFTER all embedding/evaluation results; all seeds, no best-case selection.
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    colors=['#777777','#d95f02','#1b9e77','#7570b3']
    for d in DATASETS:
        spec=DATASETS[d];labels,_=member(ROOT/spec['annotations'],spec['test_labels']);fig,axes=plt.subplots(3,4,figsize=(15,10),squeeze=False)
        for r,seed in enumerate((0,1,2)):
            for c,m in enumerate(('pca','strong','directA','umap')):
                ax=axes[r,c];path=armdir(d,m,0 if m=='pca' else seed)/'test.npy';ev=armdir(d,m,0 if m=='pca' else seed)/'evaluation.json'
                if path.exists() and ev.exists() and json.loads(ev.read_text()).get('status')=='completed':
                    Y=np.load(path);ax.scatter(Y[:,0],Y[:,1],c=labels,s=2 if len(Y)>3000 else 8,cmap='tab20',alpha=.65,rasterized=True)
                else:ax.text(.5,.5,'FAILED / unavailable',ha='center')
                ax.set_title(f'{m}, '+('one PCA control repeated' if m=='pca' else f'seed{seed}'));ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(d+' — ALL TEST outputs, independent display scales, not topology evidence');fig.tight_layout();fig.savefig(figdir/(d+'_all_seed_test_layouts.png'),dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,k in zip(axes,(5,15,50)):
            subset=[r for r in summaries if r['dataset']==d];means=[];std=[]
            for row in subset:
                x=row['statistics'].get('test_overlap_'+str(k));means.append(x['mean'] if x else np.nan);std.append((x['sample_std'] or 0) if x else 0)
            ax.bar(['PCA','strong','direct+A','UMAP'],means,yerr=std,color=colors,capsize=4);ax.set_title(f'FullTRAIN-reference TEST overlap@{k}');ax.set_ylim(bottom=0)
        fig.suptitle(d+' — mean ± sample std; PCA once; incomplete arms in results');fig.tight_layout();fig.savefig(figdir/(d+'_neighbor_effects.png'),dpi=140);plt.close(fig)
    dump(OUT/'figure_manifest.json',dict(scope='post-results figures; every seed/method, allTESTpoints; no selected showcase',files=[dict(path=str(p),sha256=sha(p)) for p in sorted(figdir.glob('*.png'))]))
if __name__=='__main__':main()
