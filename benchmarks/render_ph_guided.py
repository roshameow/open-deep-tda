#!/usr/bin/env python3
"""Render externally supplied full-row PH-guided DeepTDA/UMAP/TopoAE++ layouts.

This never fits/selects a reducer or downloads data. Provide trusted frozen
TRAIN coordinates in exactly the same original row order as --input, an
independently selected source-only witness, and a PH-trained guided checkpoint.
Every point and original-ID cycle edge is drawn, never smoothed or cropped.
Raw coordinates/checkpoints stay under caller-controlled ignored paths; review
rendered figures and their source rights before publishing.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from open_deep_tda import DeepTDA
from open_deep_tda.structural_h0 import compare_h0
from open_deep_tda.structural_sparse_h1 import analyze_sparse_h1


def matrix(value, n):
    a=np.asarray(value)
    if a.shape!=(n,2) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError('every external output must contain all (N,2) finite rows')
    return np.asarray(a,dtype=np.float64)


def render(source_path, model_path, witness_path, umap_path, author_path,
           output_path, *, name):
    with np.load(source_path,allow_pickle=False) as loaded:
        if 'X' not in loaded.files:
            raise ValueError('source NPZ needs original X rows')
        X=np.array(loaded['X'],dtype=np.float64,copy=True)
    if X.ndim!=2 or len(X)>300 or X.shape[1]<2 or not np.isfinite(X).all():
        raise ValueError('bounded finite source required')
    witness=json.loads(Path(witness_path).read_text())
    cycles=witness['cycles']
    a,b=float(witness['a']),float(witness['b'])
    if not cycles or not 0<a<b:
        raise ValueError('source witness malformed')
    model=DeepTDA.load(model_path)
    report=model.report_.get('guided_training',{})
    if report.get('status')!='certified_selected_TRAIN_family' or not report.get('certificate',{}).get('accepted'):
        raise ValueError('checkpoint lacks a completed guided TRAIN certificate')
    if report['a']!=a or report['b']!=b or report['source_witness_lengths']!=[len(c) for c in cycles]:
        raise ValueError('source witness/fit contract mismatch')
    scale=float(report['source_scale'])
    D=squareform(pdist(X))/scale
    np.fill_diagonal(D,0.)
    source_check=analyze_sparse_h1(D,cycles,a,b)
    if not source_check.certified:
        raise ValueError('original source input rejects supplied witness')
    Z=matrix(model.transform(X),len(X))*(model.reference_scale_/scale)
    U=matrix(np.load(umap_path,allow_pickle=False),len(X))
    A=matrix(np.load(author_path,allow_pickle=False),len(X))
    T=squareform(pdist(Z))
    h0=compare_h0(D,T,tolerance=float(report['h0_tolerance']),max_vertices=300)
    target_check=analyze_sparse_h1(T,cycles,a,b)
    if not h0['certified_within_tolerance'] or not target_check.certified:
        raise ValueError('fresh TRAIN coordinates did not retain accepted H0/H1')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=[('PH-Regularized Embedding (DeepTDA)\nTRAIN selected H₁ rank %d, H₀ error %.4f'
               %(target_check.rank,h0['max_merge_error']),Z),
             ('UMAP (external)',U),('TopoAE++ (author-core CPU adapter)',A)]
    fig,axes=plt.subplots(1,3,figsize=(17,5.5))
    try:
        for ax,(title,coordinates) in zip(axes,methods):
            ax.scatter(coordinates[:,0],coordinates[:,1],c=np.arange(len(X)),
                       cmap='hsv',s=18 if len(X)>100 else 28,alpha=.85,linewidths=0)
            for cycle in cycles:
                for u,v in cycle:
                    ax.plot([coordinates[u,0],coordinates[v,0]],
                            [coordinates[u,1],coordinates[v,1]],c='#444444',
                            lw=.55,alpha=.17)
            ax.set_title(title,fontsize=11)
            ax.set_xlabel('Embedding 1');ax.set_ylabel('Embedding 2')
            ax.margins(.05)
            ax.set_aspect('equal',adjustable='datalim')
        fig.suptitle('%s | all %d original TRAIN rows | same source-ID cycle overlays'
                     %(name,len(X)),fontsize=15)
        fig.text(.5,.04,'Color = source row order, not class/verified angle. Full individual axes; '
                 'no smoothing, missing points or best-run selection. Different budgets.',
                 ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.94))
        fig.savefig(output_path,dpi=160,metadata={'Title':'PH-guided full-TRAIN visual versus external author methods',
                    'Description':'All original row IDs and source cycles; no source arrays/models/author code bundled'})
    finally:
        plt.close(fig)
    return dict(schema='ph-guided-visual-v1',dataset=name,source_rows=len(X),
                source_selected_rank=source_check.rank,trained_selected_rank=target_check.rank,
                trained_h0_max_error=h0['max_merge_error'],
                source_cycle_lengths=[len(c) for c in cycles],
                full_extent=True,all_rows=True,geometry_scale='different axes; no posthoc target rescaling',
                output_methods=[v[0].split('\n')[0] for v in methods],
                limitations=['TRAIN-only source cycle/H0 certificate for project method, no omitted-query guarantee',
                   'External output coordinate units differ; equal bar counts are not class correspondence',
                   'TopoAE++ uses independently adapted original author model/loss, not best-of-ten paper image'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('input','model','witness','umap','author','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps(render(args.input,args.model,args.witness,args.umap,args.author,args.output,
                            name=args.name),sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
