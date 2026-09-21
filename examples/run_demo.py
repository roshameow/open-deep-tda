"""Run after `pip install -e .`; creates a fully offline local report."""
from pathlib import Path
import numpy as np
from open_deep_tda import DeepTDA, TDAConfig
from open_deep_tda.datasets import make_dataset
from open_deep_tda.visualization import save_report

X, labels = make_dataset("circle", n_samples=256, n_features=8, seed=7)
ids = np.random.default_rng(7).permutation(len(X))
train, test = ids[:200], ids[200:]
model = DeepTDA(TDAConfig(steps=80, warmup_steps=10, h1_size=32, evaluation_size=32))
Z = model.fit_transform(X[train], validation_data=X[test])
out = Path("outputs/python-demo")
out.mkdir(parents=True, exist_ok=True)
model.save(out / "model.pt")
np.save(out / "test_embedding.npy", model.transform(X[test]))
save_report(model.reference_, Z, out / "report.html", labels=labels[train], metrics=model.report_)
restored = DeepTDA.load(out / "model.pt")
np.testing.assert_array_equal(restored.transform(X[test]), model.transform(X[test]))
print(out / "report.html")
