"""Local command-line interface; all reports work without a network connection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .config import TDAConfig


def load_array(path, delimiter=",", skip_header=0):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as bundle:
            if "X" not in bundle:
                raise ValueError("NPZ input must contain an X array")
            return bundle["X"]
    if skip_header < 0:
        raise ValueError("skip_header must be nonnegative")
    # genfromtxt acquired ndmin only in newer numpy. Preserve one-column versus
    # one-row shape on the supported numpy>=1.21 line instead of guessing.
    array = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)
    if array.size == 0:
        return np.empty((0, 0))
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        column = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header, usecols=(0,))
        return array.reshape(np.size(column), -1)
    return array


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def config_from_args(args):
    values = json.loads(Path(args.config).read_text()) if getattr(args, "config", None) else {}
    if not isinstance(values, dict):
        raise ValueError("configuration JSON must be an object")
    for name in ("steps", "seed", "mode", "n_components", "h1_size", "device"):
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    try:
        return TDAConfig(**values).validate()
    except TypeError as exc:
        raise ValueError(f"invalid configuration: {exc}") from exc


def output_dir(path, overwrite=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ValueError(f"output directory is not empty: {path}; choose a new directory or --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(model, out, labels=None):
    from .mapper import mapper_graph
    from .visualization import save_report
    model.save(out / "model.pt")
    np.save(out / "embedding.npy", model.embedding_)
    np.save(out / "reference.npy", model.reference_)
    write_json(out / "metrics.json", model.report_)
    write_json(out / "history.json", {"training": model.history_, "validation": model.validation_history_,
                                       "semantic": model.semantic_history_})
    # Mapper is an interpretation of this explicitly bounded subset, not all samples.
    rng = np.random.default_rng(model.config.seed)
    ids = np.sort(rng.choice(len(model.reference_), min(1000, len(model.reference_)), replace=False))
    graph = mapper_graph(model.reference_[ids])
    graph["source_sample_ids"] = ids.tolist()
    write_json(out / "mapper.json", graph)
    if labels is not None:
        np.save(out / "labels.npy", labels)
    save_report(model.reference_, model.embedding_, out / "report.html", labels=labels,
                metrics=model.report_, mapper=graph, seed=model.config.seed)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="deep-tda", description="Independent experimental topology-regularized reduction")
    parser.add_argument("--version", action="version", version="open-deep-tda 0.3.0")
    sub = parser.add_subparsers(dest="command", required=True)

    def training_options(p):
        p.add_argument("--config", help="JSON TDAConfig overrides")
        p.add_argument("--steps", type=int)
        p.add_argument("--seed", type=int)
        p.add_argument("--mode", choices=["geometry", "semantic"])
        p.add_argument("--n-components", type=int, choices=[2, 3])
        p.add_argument("--h1-size", type=int)
        p.add_argument("--device", choices=["cpu", "cuda", "mps"])

    def data_options(p):
        p.add_argument("--input", required=True, help="numeric NPY/NPZ(X)/CSV")
        p.add_argument("--delimiter", default=",")
        p.add_argument("--skip-header", type=int, default=0)

    def output_options(p):
        p.add_argument("--output", required=True, help="output directory")
        p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("demo", help="train on a synthetic point cloud and write an offline HTML report")
    training_options(p); output_options(p)
    p.add_argument("--dataset", default="circle")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--features", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--validation-fraction", type=float, default=0.2)

    p = sub.add_parser("fit", help="fit a numeric array")
    training_options(p); data_options(p); output_options(p)
    p.add_argument("--validation", help="optional separate held-out array (same columns)")

    p = sub.add_parser("transform", help="map new samples with a saved parametric model")
    data_options(p)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True, help="output NPY file")
    p.add_argument("--ood-output", help="optional output NPY for nearest-reference distances")

    p = sub.add_parser("graph-fit", help="fit the independent graph layout and conditional query mapper")
    data_options(p); output_options(p)
    p.add_argument("--validation", help="held-out array; never used by the layout optimizer")
    p.add_argument("--neighbor-backend", choices=["exact", "pynndescent"], default="exact")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("graph-transform", help="map queries using a safe NPZ graph predictor")
    data_options(p)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True, help="output NPY file")
    p.add_argument("--diagnostics", help="optional JSON optimizer diagnostics")

    p = sub.add_parser("evaluate", help="evaluate aligned sample IDs in two arrays")
    p.add_argument("--reference", required=True)
    p.add_argument("--embedding", required=True)
    p.add_argument("--output", required=True, help="output JSON file")
    p.add_argument("--topology-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stability-repeats", type=int, default=0, help="optional paired 16/32/64-point resampling audit")
    p.add_argument("--bootstrap", action="store_true", help="resample with replacement (default 5 repeats)")

    p = sub.add_parser("benchmark", help="fixed-reference baselines and ablations")
    training_options(p); output_options(p)
    p.add_argument("--input", help="NPY/NPZ(X)/CSV, or use generated dataset")
    p.add_argument("--dataset", default="circle")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--features", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--methods", default="pca,tsne,umap,ae,topoae_h0,deep_tda,no_h0,no_h1,no_critical,local_only")

    args = parser.parse_args(argv)
    try:
        if args.command in ("demo", "fit"):
            from .estimator import DeepTDA
            cfg = config_from_args(args)
            out = output_dir(args.output, args.overwrite)
            validation, labels = None, None
            if args.command == "demo":
                from .datasets import make_dataset
                X, labels = make_dataset(args.dataset, args.samples, args.features, args.noise, cfg.seed)
                if not 0 <= args.validation_fraction < 1:
                    raise ValueError("validation-fraction must be in [0,1)")
                count = int(len(X) * args.validation_fraction)
                if count:
                    ids = np.random.default_rng(cfg.seed).permutation(len(X))
                    validation = X[ids[:count]]
                    X, labels = X[ids[count:]], labels[ids[count:]]
                    np.save(out / "validation.npy", validation)
            else:
                X = load_array(args.input, args.delimiter, args.skip_header)
                if args.validation:
                    validation = load_array(args.validation, args.delimiter, args.skip_header)
            model = DeepTDA(cfg).fit(X, validation_data=validation)
            np.save(out / "input.npy", X)
            save_run(model, out, labels)
            print(json.dumps({"status": "ok", "output": str(out.resolve()), "samples": len(X),
                              "fit_seconds": model.fit_seconds_}, ensure_ascii=False))
        elif args.command == "transform":
            from .estimator import DeepTDA
            model = DeepTDA.load(args.model)
            X = load_array(args.input, args.delimiter, args.skip_header)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, model.transform(X))
            if args.ood_output:
                Path(args.ood_output).parent.mkdir(parents=True, exist_ok=True)
                np.save(args.ood_output, model.ood_scores(X))
            print(str(output))
        elif args.command == "graph-fit":
            from .graph_embedding import GraphEmbedding
            out = output_dir(args.output, args.overwrite)
            X = load_array(args.input, args.delimiter, args.skip_header)
            model = GraphEmbedding(neighbor_backend=args.neighbor_backend,
                                   epochs=args.epochs, seed=args.seed).fit(X)
            np.save(out / "embedding.npy", model.embedding_)
            write_json(out / "fit.json", model.diagnostics_)
            if args.validation:
                queries = load_array(args.validation, args.delimiter, args.skip_header)
                prediction, diagnostic = model.transform(queries, return_diagnostics=True)
                np.save(out / "validation_embedding.npy", prediction)
                write_json(out / "transform.json", diagnostic)
            # This predictor necessarily contains reference features. Unlike the
            # legacy neural compact checkpoint, it is NOT training-data-free.
            model.save(out / "model.npz", overwrite=args.overwrite)
            print(json.dumps({"status": "ok", "output": str(out.resolve()),
                              "samples": len(X), "reference_data_in_checkpoint": True}))
        elif args.command == "graph-transform":
            from .graph_embedding import GraphEmbedding
            model = GraphEmbedding.load(args.model)
            queries = load_array(args.input, args.delimiter, args.skip_header)
            prediction, diagnostic = model.transform(queries, return_diagnostics=True)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, prediction)
            if args.diagnostics:
                Path(args.diagnostics).parent.mkdir(parents=True, exist_ok=True)
                write_json(args.diagnostics, diagnostic)
            print(str(output))
        elif args.command == "evaluate":
            from .evaluation import evaluate_embedding, evaluate_stability
            reference, embedding = load_array(args.reference), load_array(args.embedding)
            metrics = evaluate_embedding(reference, embedding, topology_size=args.topology_size, seed=args.seed)
            if args.stability_repeats < 0:
                raise ValueError("stability-repeats must be nonnegative")
            repeats = args.stability_repeats or (5 if args.bootstrap else 0)
            if repeats:
                metrics["stability"] = evaluate_stability(reference, embedding, repeats=repeats,
                    seed=args.seed, bootstrap=args.bootstrap)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, metrics)
            print(str(args.output))
        else:
            from .datasets import make_dataset
            from .benchmark import run_benchmark
            from threadpoolctl import threadpool_limits
            cfg = config_from_args(args)
            out = output_dir(args.output, args.overwrite)
            X = load_array(args.input) if args.input else make_dataset(args.dataset, args.samples, args.features, args.noise, cfg.seed)[0]
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
            with threadpool_limits(limits=cfg.num_threads):
                result = run_benchmark(X, cfg, seeds=seeds, methods=args.methods.split(","))
            write_json(out / "benchmark.json", result)
            print(str(out / "benchmark.json"))
    except (ValueError, RuntimeError, OSError, ImportError, NotImplementedError) as exc:
        print(f"deep-tda: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
