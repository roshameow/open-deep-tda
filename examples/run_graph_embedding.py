"""Real bundled Digits demo for the unreleased graph core (no downloads).

Labels are used only for post-fit scoring/plotting. No topology guarantee is
implied. Output directories can contain training-data-bearing checkpoints.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

from open_deep_tda import GraphEmbedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/graph-digits'))
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save-model', action='store_true',
                        help='opt in: predictor contains all training features')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('use a new output directory; existing results are not overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_digits()
    order = np.random.default_rng(0).permutation(len(data.data))
    train, test = order[:1400], order[1400:]
    model = GraphEmbedding(seed=0)
    Z = model.fit_transform(data.data[train])
    Y = model.transform(data.data[test])
    # Post-fit labels only; never passed to the reducer or query objective.
    predicted = KNeighborsClassifier(n_neighbors=15).fit(Z, data.target[train]).predict(Y)
    report = dict(dataset='real sklearn Digits',train_rows=len(train),test_rows=len(test),
        accuracy=float(np.mean(predicted == data.target[test])),
        nonconverged_queries=model.transform_diagnostics_['nonconverged'],
        scope='one fixed illustrative split/seed; no topology or superiority claim')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.save_model:
        model.save(args.output / 'predictor.npz')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.get_cmap('tab10')
        for label in range(10):
            fit_mask = data.target[train] == label
            test_mask = data.target[test] == label
            ax.scatter(*Z[fit_mask].T, color=colors(label), s=8, alpha=.3)
            ax.scatter(*Y[test_mask].T, color=colors(label), s=18, marker='x', label=str(label))
        ax.set(title='Real Digits: 1,400 training dots / 397 held-out crosses',
               xlabel='Embedding 1', ylabel='Embedding 2', aspect='equal')
        ax.legend(title='Post-fit digit label', ncol=5)
        fig.tight_layout()
        fig.savefig(args.output / 'digits.png', dpi=150)
        plt.close(fig)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
