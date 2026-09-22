"""Render all official TEST coordinates from a fixed real-data confirmation seed.

No fit, selection, axis clipping, or display subsampling. Labels only color
already fitted coordinates. Raw arrays remain in ignored data/outputs folders.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['har', 'fashion_mnist'], default='har')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    fashion = args.dataset == 'fashion_mnist'
    args.run = args.run or Path('outputs/fashion-geometry-confirmation' if fashion else 'outputs/geometry-optimization-confirmation')
    args.cache = args.cache or Path('outputs/v02-fashion_mnist/annotations.npz' if fashion else 'data/uci_har/features.npz')
    args.output = args.output or Path('assets/fashion-optimization.png' if fashion else 'assets/har-optimization.png')
    result = json.loads((args.run / 'results.json').read_text())
    nce_round = result['plan'].get('protocol') == 'neighbor-nce-confirmation-v1'
    selected = ('selected' if nce_round else 'transferred_graph_weak_2400' if fashion else result['plan']['selection']['selected_candidate'])
    methods = ['previous_graph' if nce_round else 'original_stress_1200' if fashion else 'stress_600', selected]
    with np.load(args.cache, allow_pickle=False) as data:
        labels = data['test_labels'] if fashion else data['y_test'] - 1
    names = (['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] if fashion else
             ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying'])
    colors = plt.get_cmap('tab10' if fashion else 'Dark2').colors[:len(names)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor='#f8fafc')
    fig.subplots_adjust(left=.06, right=.98, bottom=.25, top=.79, wspace=.18)
    title = (('Fashion-MNIST: conditional neighbor objective vs. the previous graph' if fashion else
              'UCI HAR: direct transfer of the Fashion-selected neighbor objective') if nce_round else
             ('Fashion-MNIST: direct transfer of the HAR-selected graph objective' if fashion else
              'UCI HAR: distance stress vs. the TRAIN-selected graph objective'))
    fig.suptitle(title,
                 fontsize=17, fontweight='bold', color='#0f172a', y=.96)
    for ax, method in zip(axes, methods):
        record = next(r for r in result['records'] if r['method'] == method and r['seed'] == args.seed)
        if record['status'] != 'ok':
            raise ValueError('cannot render failed run')
        with np.load(args.run / 'local' / method / ('seed' + str(args.seed)) / 'embeddings.npz', allow_pickle=False) as data:
            Z = data['Z_test']
        if Z.shape != (len(labels), 2) or not np.isfinite(Z).all():
            raise ValueError('invalid complete TEST embedding')
        # Every row in one fixed label-independent order, rather than class-wise
        # overpainting. Limits autoscale to include all rows; panels scale separately.
        ids = np.random.default_rng(2026).permutation(len(Z))
        ax.scatter(Z[ids, 0], Z[ids, 1], c=[colors[k] for k in labels[ids]], s=8, alpha=.6, linewidths=0)
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(color='#e2e8f0', alpha=.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color('#cbd5e1')
        g, p, k = record['population_geometry'], record['probe'], record['kmeans']
        title = ('Stress baseline (1,200 steps)' if fashion else 'Stress baseline (600 steps)') if method == methods[0] else 'Graph + weak topology (2,400 steps)'
        if nce_round:
            objective = 'Previous graph' if method == methods[0] else 'Conditional neighbor NCE'
            title = f'{objective} ({record["config"]["steps"]:,} steps)'
        ax.set_title(f'{title}\n15-NN {p["test_accuracy"]:.1%} · overlap {g["knn_overlap"]:.3f} · KMeans ARI {k["test_ari"]:.3f}', fontsize=10.5, pad=12)
        ax.set_xlabel('embedding dimension 1')
        ax.set_ylabel('embedding dimension 2')
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=c, markeredgecolor='none', label=n, markersize=7)
               for c, n in zip(colors, names)]
    fig.legend(handles=handles, ncol=5 if fashion else 6, loc='lower center', bbox_to_anchor=(.5, .10),
               title='Class labels: post-fit diagnostics only', frameon=False)
    scope = ('All 10,000 official TEST rows; both fit all 60,000 TRAIN PCA64 rows.' if fashion else
             'All 2,947 official TEST rows; both fit all 7,352 TRAIN rows with identical train-only standardization.')
    if nce_round:
        scope += ' Both use TRAIN-only output-scale calibration.'
    fig.text(.5, .035, f'{scope} Fixed seed {args.seed}; out-of-sample transform.\n'
             'Axes autoscale separately; no clipped points. Unequal training budgets. This view does not prove topology or baseline superiority.',
             ha='center', fontsize=9, color='#475569', linespacing=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()
