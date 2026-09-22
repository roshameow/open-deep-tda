"""Fixed-seed0 compact outputs at full extent; no crop, warp, fitting or mapping."""
import json
from pathlib import Path
import numpy as np
if __package__:
    from .common import ROOT,DATASETS,member,require_label_barrier
    from .compact import require_barrier
    from .figures import LABEL_NAMES,TITLES
else:
    from common import ROOT,DATASETS,member,require_label_barrier
    from compact import require_barrier
    from figures import LABEL_NAMES,TITLES


def render_compact(directory,prior_directory,dataset,destination):
    directory=Path(directory);prior=Path(prior_directory)
    require_barrier(directory);require_label_barrier(prior)
    spec=DATASETS[dataset];labels,_=member(ROOT/spec['annotations'],spec['test_labels'])
    classes=list(range(10)) if dataset=='fashion' else list(range(1,7)) if dataset=='har' else list(range(1,21))
    assert sorted(np.unique(labels).tolist())==classes
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(1,4,figsize=(19,5.7));palette=plt.get_cmap('tab10' if len(classes)<=10 else 'tab20')
    for ax,method in zip(axes,('pca','strong','directA','umap')):
        root=directory/f'{dataset}-seed0' if method=='directA' else prior/dataset/f'{method}-seed0'
        report=json.loads((root/'evaluation.json').read_text());assert report['status']=='completed'
        Y=np.load(root/'test.npy',allow_pickle=False);assert Y.shape==(spec['m'],2) and np.isfinite(Y).all()
        for i,label in enumerate(classes):
            mask=labels==label;ax.scatter(Y[mask,0],Y[mask,1],c=[palette(i)],s=3 if dataset=='fashion' else 6 if dataset=='har' else 15,alpha=.7,linewidths=0,rasterized=True)
        title='Direct graph + A (TRAIN hull)' if method=='directA' else TITLES[method]
        if dataset=='coil' and method=='strong':title+=' (stress600)'
        ax.set_title(f"{title}\nseed0 overlap@15 {report['test_metrics']['15']['overlap']:.4f} | 15NN {100*report['classification']['accuracy']:.2f}%",fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
    handles=[Line2D([0],[0],marker='o',linestyle='',markersize=6,color=palette(i),label=LABEL_NAMES[dataset][i]) for i in range(len(classes))]
    fig.legend(handles=handles,loc='lower center',ncol=10 if dataset!='har' else 6,fontsize=8,frameon=False)
    fig.suptitle(f"{dataset.upper()} — fixed seed0 compact followup, all {spec['m']:,} TEST points",fontsize=15)
    fig.text(.5,.115,'Full extents; no clipping, cropped views or excluded outputs. Independent axes. Original controls reused; direct+A only is the TRAIN-hull followup.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.77,bottom=.23,wspace=.13)
    fig.savefig(destination,dpi=160,metadata={'Title':f'{dataset} fixed seed0 TRAIN-hull followup','Description':'All TEST points at full extents, complete class legend; original PCA/strong/UMAP controls unchanged; no clipping'});plt.close(fig)
    return dict(dataset=dataset,seed=0,points=spec['m'],classes=len(classes),directA_protocol='train_hull compact followup',controls='original unbounded-v1 benchmark controls reused',full_extent=True,clipping=False,view_cropping=False,excluded_points=0,numbers='seed0, not three-seed means')
