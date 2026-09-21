"""Local numeric SSL integration benchmark on sklearn's bundled digits data.

Labels are used only for plotting. This is not a tuned classification benchmark.
"""
from pathlib import Path
import argparse
import numpy as np
from sklearn.datasets import load_digits
from open_deep_tda import DeepTDA
from open_deep_tda.cli import save_run, output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/digits-semantic')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    out = output_dir(args.output, args.overwrite)
    X, labels = load_digits(return_X_y=True)
    ids = np.random.default_rng(0).permutation(len(X))
    train, test = ids[:1400], ids[1400:]
    model = DeepTDA(mode='semantic', semantic_dim=32, semantic_steps=80, steps=80,
                    warmup_steps=10, h1_size=32, h0_size=64, subset_bank_size=6,
                    evaluation_size=32).fit(X[train], validation_data=X[test])
    save_run(model, out, labels[train])
    np.save(out / 'heldout_embedding.npy', model.transform(X[test]))
    print(out / 'report.html')


if __name__ == '__main__':
    main()
