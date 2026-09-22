"""Render fixed seed0 all-TEST effect figures; never fit/transform a model."""
import json
from pathlib import Path
import numpy as np

LABEL_NAMES={
 'fashion':['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'],
 'har':['Walking','Walking upstairs','Walking downstairs','Sitting','Standing','Laying'],
 'coil':[f'Object {i}' for i in range(1,21)]}
TITLES={'pca':'PCA2 control','strong':'Current strong','directA':'Direct graph + exact A','umap':'Author UMAP'}


def render(directory,dataset,destination):
    """Only saved coordinates/metrics and evaluation labels; no model invocation."""
    if __package__:from .common import DATASETS,ROOT,member,require_label_barrier
    else:from common import DATASETS,ROOT,member,require_label_barrier
    directory=Path(directory);require_label_barrier(directory)
    spec=DATASETS[dataset];labels,_=member(ROOT/spec['annotations'],spec['test_labels'])
    expected_classes=list(range(10)) if dataset=='fashion' else list(range(1,7)) if dataset=='har' else list(range(1,21))
    if sorted(np.unique(labels).tolist())!=expected_classes:raise ValueError('Unexpected class labels; do not silently mislabel the legend')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(2,4,figsize=(19,9));palette=plt.get_cmap('tab10' if len(expected_classes)<=10 else 'tab20')
    colors={label:palette(i) for i,label in enumerate(expected_classes)}
    outside_counts={}
    for column,method in enumerate(('pca','strong','directA','umap')):
        ax=axes[0,column];detail=axes[1,column]
        root=directory/dataset/f'{method}-seed0';report=json.loads((root/'evaluation.json').read_text())
        if report['status']!='completed':raise ValueError('Seed0 method failed; do not substitute a better seed')
        Z=np.load(root/'test.npy',allow_pickle=False)
        if Z.shape!=(spec['m'],2):raise ValueError('Expected ALL TEST points')
        train=np.load(root/'fit.npy',allow_pickle=False);lo=train.min(0);hi=train.max(0);pad=np.maximum(hi-lo,1e-12)*.05;lo-=pad;hi+=pad
        outside=int(np.sum(np.any((Z<lo)|(Z>hi),axis=1)));outside_counts[method]=outside
        for label in expected_classes:
            mask=labels==label
            for panel in (ax,detail):
                panel.scatter(Z[mask,0],Z[mask,1],s=3 if dataset=='fashion' else 6 if dataset=='har' else 15,c=[colors[label]],alpha=.7,linewidths=0,rasterized=True)
        detail.set_xlim(lo[0],hi[0]);detail.set_ylim(lo[1],hi[1]);detail.set_title(f'TRAIN-range detail (+5%); {outside:,} TEST points outside',fontsize=9)
        name=TITLES[method]+(' (stress600)' if method=='strong' and dataset=='coil' else '')
        ax.set_title(f"{name}\nseed0 overlap@15 {report['test_metrics']['15']['overlap']:.3f} | 15NN accuracy {100*report['classification']['accuracy']:.1f}%",fontsize=10)
        ax.set_xlabel('Full TEST extent; all points retained',fontsize=8)
        for panel in (ax,detail):panel.set_xticks([]);panel.set_yticks([])
    handles=[Line2D([0],[0],marker='o',linestyle='',markersize=6,color=colors[v],label=LABEL_NAMES[dataset][i]) for i,v in enumerate(expected_classes)]
    fig.legend(handles=handles,loc='lower center',ncol=10 if dataset!='har' else 6,fontsize=8,frameon=False)
    fig.suptitle(f"{dataset.upper()} — fixed seed0 (not best seed), all {spec['m']:,} TEST points",fontsize=15)
    fig.text(.5,.10,'Top: ALL TEST full extent, including extreme outputs. Bottom: fixed TRAIN min/max +5% detail, display only. Metrics unchanged; aggregate keeps all seeds.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.86,bottom=.18,wspace=.13,hspace=.3);fig.savefig(destination,dpi=160,metadata={'Title':f'{dataset} fixed seed0 all-TEST benchmark','Description':'PCA / strong / direct+A / genuine UMAP; not best seed; full class legend'});plt.close(fig)
    return dict(dataset=dataset,seed=0,points=spec['m'],classes=len(expected_classes),methods=['pca','strong','directA','umap'],display_subset=False,detail_rule='TRAIN min/max plus5% range; top full extent retains every point',outside_detail_counts=outside_counts,numbers='seed0',selection='fixed seed0, never best seed')
