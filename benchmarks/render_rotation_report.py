"""Offline angle-coloured COIL orbit plots, fixed seed (never best-object picking)."""
import argparse
import colorsys
import html
import json
from pathlib import Path
import numpy as np


def plot(points,angles,test,title,caption):
    points=np.asarray(points,dtype=float)
    centered=points-points.mean(axis=0)
    span=max(float(np.ptp(centered,axis=0).max()),1e-12)
    P=centered/span*150+np.array([120,110]);P[:,1]=220-P[:,1]
    order=np.argsort(angles)
    line=P[np.r_[order,order[0]]]
    output=[f'<svg viewBox="0 0 240 260" role="img" aria-label="{html.escape(title,quote=True)}">',
            f'<title>{html.escape(title)}</title><rect width="240" height="260" fill="white"/>',
            f'<text x="10" y="18" font-size="12">{html.escape(title)}</text>',
            '<polyline fill="none" stroke="#cbd5e1" stroke-width="1" points="'+ ' '.join(f'{x:.2f},{y:.2f}' for x,y in line)+'"/>']
    for (x,y),angle,heldout in zip(P,angles,test):
        rgb=colorsys.hsv_to_rgb(float(angle)/360,.8,.85)
        color='#'+''.join(f'{int(v*255):02x}' for v in rgb)
        stroke='#0f172a' if heldout else 'none'
        output.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{3.1 if heldout else 2.1}" fill="{color}" stroke="{stroke}" stroke-width=".8"><title>{angle:g} degrees; {"held out" if heldout else "train"}</title></circle>')
    output.append(f'<text x="8" y="246" font-size="9">{html.escape(caption)}</text></svg>')
    return ''.join(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/v02-coil20');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--native-embedding',help='optional full common train960/test480 NPZ from native TopoAE++ adapter')
    p.add_argument('--output');args=p.parse_args()
    root=Path(args.root)
    with np.load(root/'annotations.npz',allow_pickle=False) as a:objects=a['object_ids'];angles=a['angles_degrees'];ntrain=len(a['train_labels'])
    holdout=np.arange(len(objects))>=ntrain
    columns=[]
    for method,title in [('pca','PCA'),('deep_tda','Open Deep-TDA'),('no_h1','Without H1'),('upstream_topoae','Official TopoAE model'),('umap','UMAP')]:
        directory=root/f'{method}-seed{args.seed}'
        if not (directory/'result.json').exists():continue
        record=json.loads((directory/'result.json').read_text())
        if record['status']!='ok':continue
        with np.load(directory/'embedding.npz',allow_pickle=False) as a:Z=np.concatenate([a['train'],a['test']])
        columns.append((title,Z,{r['object_id']:r for r in record['rotation_orbits']['objects']}))
    if args.native_embedding:
        with np.load(args.native_embedding,allow_pickle=False) as a:
            if a['train'].shape!=(960,2) or a['test'].shape!=(480,2):raise ValueError('native data is not the full common split')
            columns.append(('TopoAE++ CPU adapter',np.concatenate([a['train'],a['test']]),{}))
    output=['<!doctype html><html><head><meta charset="utf-8"><title>COIL rotation diagnostics</title>',
        '<style>body{font:15px system-ui;max-width:1700px;margin:2em auto;padding:0 1em;background:#f8fafc;color:#0f172a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}svg{width:100%;border:1px solid #cbd5e1}summary{font-weight:bold;cursor:pointer;margin:1em 0}</style></head><body>',
        '<h1>Real COIL-20 rotation diagnostics</h1>',
        f'<p>All 20 objects, all 72 views per object; fixed seed {args.seed}. Every third view is held out. Hue encodes physical angle; black outlines mark test views. Lines connect the true cyclic view order, not learned graph edges.</p>',
        '<p>The ideal-angle circle is annotation, NOT a learned reference embedding. Symmetric appearances can make angles ambiguous. Each cell has independent display scale. Use numeric reports for distortion, and do not infer cycle identity merely from a strong H1 bar.</p>']
    for obj in np.unique(objects):
        ids=np.flatnonzero(objects==obj);a=angles[ids];test=holdout[ids]
        output.append(f'<details {"open" if obj==1 else ""}><summary>Object {int(obj)} — 48 train / 24 held-out views</summary><div class="grid">')
        ideal=np.column_stack([np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a))])
        output.append(plot(ideal,a,test,'Physical angle annotation','This is not an embedding result'))
        for title,Z,metrics in columns:
            m=metrics.get(int(obj),{})
            caption=(f"angle adjacency {m['embedding_angle']['adjacent_view_recall']:.1%}; crossings {m['strict_angle_order_polygon_crossings']}" if m else 'See separate native-adapter metrics and caveats')
            output.append(plot(Z[ids],a,test,title,caption))
        output.append('</div></details>')
    output+=['<p>Data: Nene, Nayar and Murase, Columbia COIL-20, CUCS-005-96 (1996). Original photos are not embedded in this report. Raw pixels are downsampled to32x32, then shared train-only PCA64 is fitted before reduction. Dataset rights are separate from software licensing.</p>']
    if args.native_embedding:
        output+=['<p>This document includes materials generated with TTK (the Topology ToolKit) which is developed by the CNRS &amp; Sorbonne Universite and its contributors. Cite Julien Tierny et al., The Topology ToolKit, and Topological Autoencoders++ (TVCG 2025). The native column uses a disclosed standalone CPU/no-CGAL adapter, not the original ParaView launcher.</p>']
    output.append('</body></html>')
    target=Path(args.output) if args.output else root/'rotation_comparison.html'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(''.join(output));print(target)


if __name__=='__main__':main()
