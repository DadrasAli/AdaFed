#!/usr/bin/env python3
"""Export all 10,000-round adaptive-method comparisons for PGFPlots.

Stochastic curves are pointwise arithmetic means over the benchmark seeds. For every stochastic
table a band table per method (<group>-<dataset>-stochastic-<regime>-<method>-band.dat: the same
directional one-sided RMS band, thinned to the same log-spaced rounds, as the primary rate figure's)
is written beside it. Datasets are the two benchmark datasets, the IJCNN1 Dirichlet splits
(e.g. ijcnn1-32-dirichlet0.1; selected group only, which is what the Dirichlet figures use) and
the k-means splits of the two benchmark datasets (<dataset>-kmeans; every group).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from tools import export_rate_study_pgfplots as rate_exporter
except ImportError:                                  # run as a script from tools/
    import export_rate_study_pgfplots as rate_exporter


ROOT = Path(__file__).resolve().parents[1]
ADAPTATION_RESULTS = ROOT / "rate-study" / "comparison-by-adaptation-10000" / "results"
SELECTED_RESULTS = ROOT / "rate-study" / "comparison-selected-joint-10000" / "results"
DEFAULT_OUT = ROOT / "figures" / "benchmarks" / "experiments-data"
DATASETS = ("synthetic-kappa1", "ijcnn1-32")
REGIMES = ("smooth-nonconvex", "smooth-convex", "nonsmooth-convex")
GROUPS = {
    "server": ("adafed", "fedadagrad", "fedadam", "fedur"),
    "client": ("adafed", "local_adaalter", "fedsps", "fedlada"),
    "both": ("adafed", "fafed", "fedada2"),
    "selected": ("adafed", "fedadagrad", "fedlada", "fafed"),
}
SELECTED_METHODS = {"adafed", "fedadagrad", "fedlada", "fafed"}
SEEDS = {"full": (2027,), "stochastic": tuple(range(2027, 2047))}
BATCH = {"synthetic-kappa1": 16, "ijcnn1-32": 32}
SPLIT_PREFIXES = ("ijcnn1-32-dirichlet", "ijcnn1-32-kmeans")
KMEANS_SUFFIX = "-kmeans"
METRIC = {"smooth-nonconvex": "grad_norm_sq", "smooth-convex": "gap", "nonsmooth-convex": "gap"}
FLOOR = 1e-16


def batch_of(dataset):
    """The stochastic batch: a Dirichlet or k-means split of IJCNN1 uses IJCNN1's, the k-means split of synthetic-kappa1
    synthetic-kappa1's."""
    if dataset.startswith(SPLIT_PREFIXES):
        return BATCH["ijcnn1-32"]
    if dataset.endswith(KMEANS_SUFFIX) and dataset[:-len(KMEANS_SUFFIX)] in BATCH:
        return BATCH[dataset[:-len(KMEANS_SUFFIX)]]
    return BATCH[dataset]


def load_curve(dataset, regime, mode, method, seed):
    results = SELECTED_RESULTS if method in SELECTED_METHODS else ADAPTATION_RESULTS
    run = results / dataset / regime / mode / method / f"seed{seed}"
    manifest = json.loads((run / "run.json").read_text())
    config = manifest["config"]
    expected = {"dataset": dataset, "mode": mode, "algorithm": method,
                "seed": seed, "T": 10_000, "K": 10, "p": 1.0}
    if mode == "stochastic":
        expected["b"] = batch_of(dataset)
    if (manifest.get("status") != "done" or manifest.get("final_round") != 10_000
            or any(config.get(key) != value for key, value in expected.items())):
        raise ValueError(f"invalid run manifest: {run / 'run.json'}")
    with (run / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metric = METRIC[regime]
    rounds = np.asarray([int(row["round"]) for row in rows])
    random = np.asarray([float(row[f"mean_{metric}"]) for row in rows])
    last = np.asarray([float(row[metric]) for row in rows])
    if not len(rows) or rounds[0] != 1 or rounds[-1] != 10_000:
        raise ValueError(f"incomplete checkpoint range: {run / 'metrics.csv'}")
    if not np.isfinite(random).all() or not np.isfinite(last).all():
        raise ValueError(f"non-finite metric: {run / 'metrics.csv'}")
    return rounds, random, last


def export(output=DEFAULT_OUT, groups=None, datasets=None):
    """Write the tables of `groups` (default all) for `datasets` (default the two benchmark datasets)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for group, methods in GROUPS.items():
        if groups is not None and group not in groups:
            continue
        for dataset in (DATASETS if datasets is None else datasets):
            for mode in ("full", "stochastic"):
                for regime in REGIMES:
                    columns, samples = {}, {}
                    common_rounds = None
                    for method in methods:
                        curves = [load_curve(dataset, regime, mode, method, seed) for seed in SEEDS[mode]]
                        rounds = curves[0][0]
                        if any(not np.array_equal(rounds, curve[0]) for curve in curves[1:]):
                            raise ValueError(f"checkpoint mismatch: {dataset}/{regime}/{mode}/{method}")
                        if common_rounds is not None and not np.array_equal(common_rounds, rounds):
                            raise ValueError(f"method checkpoint mismatch: {dataset}/{regime}/{mode}")
                        common_rounds = rounds
                        samples[method] = (np.stack([curve[1] for curve in curves]),
                                           np.stack([curve[2] for curve in curves]))
                        columns[f"{method}_random"] = np.maximum(samples[method][0].mean(axis=0), FLOOR)
                        columns[f"{method}_last"] = np.maximum(samples[method][1].mean(axis=0), FLOOR)
                    names = [name for method in methods
                             for name in (f"{method}_random", f"{method}_last")]
                    path = output / f"{group}-{dataset}-{mode}-{regime}.dat"
                    lines = ["round " + " ".join(names)]
                    for index, t in enumerate(common_rounds):
                        lines.append(f"{int(t)} " + " ".join(
                            f"{columns[name][index]:.17g}" for name in names))
                    path.write_text("\n".join(lines) + "\n")
                    paths.append(path)
                    if mode == "stochastic":
                        for method in methods:
                            rate_exporter.write_band(
                                output / f"{group}-{dataset}-{mode}-{regime}-{method}-band.dat",
                                common_rounds, *samples[method])
    print(f"validated source runs and wrote {len(paths)} PGFPlots tables (and their band tables) to {output}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    export(args.out.resolve())


if __name__ == "__main__":
    main()
