#!/usr/bin/env python3
"""Export all AdaFed rate-study curves used by the standalone PGFPlots files.

Stochastic trajectories are aggregated with the pointwise arithmetic mean.
The primary IJCNN1 b=1 panel uses the dedicated 10,000-round runs and includes
directional one-sided standard-deviation bands. All other panels use their
paper horizons and retain every saved checkpoint. `export(datasets=...)` writes
the same tables for the k-means splits of the two datasets (<dataset>-kmeans),
with the same batches and the same eta and gamma values.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "rate-study" / "results"
DEFAULT_OUT = ROOT / "figures" / "rates" / "experiments-data"

DATASETS = ("synthetic-kappa10", "ijcnn1-32")
APPENDIX_BATCH = {"synthetic-kappa10": 8, "ijcnn1-32": 32}
KMEANS_SUFFIX = "-kmeans"
REGIMES = ("smooth-nonconvex", "smooth-convex", "nonsmooth-convex")
KS = (1, 10, 100)
SEEDS = tuple(range(2027, 2047))
FLOOR = 1e-16
ABLATIONS = {
    "eta": {"values": (0.01, 0.1, 1, 10, 100), "root": "eta-paper"},
    "gamma": {"values": (0.01, 0.1, 1, 10, 100), "root": "gamma-paper"},
    "partial": {"values": (0.25, 0.5, 1), "root": "partial-paper"},
}


def read_curve(path, regime):
    random_column = "mean_grad_norm_sq" if regime == "smooth-nonconvex" else "mean_gap"
    last_column = "grad_norm_sq" if regime == "smooth-nonconvex" else "gap"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    return (
        np.asarray([int(row["round"]) for row in rows]),
        np.asarray([float(row[random_column]) for row in rows]),
        np.asarray([float(row[last_column]) for row in rows]),
    )


def aggregate(paths, regime):
    curves = [read_curve(path, regime) for path in paths]
    rounds = curves[0][0]
    if any(not np.array_equal(curve[0], rounds) for curve in curves[1:]):
        raise ValueError("curves do not share checkpoint rounds")
    random_samples = np.stack([curve[1] for curve in curves])
    last_samples = np.stack([curve[2] for curve in curves])
    return rounds, random_samples.mean(axis=0), last_samples.mean(axis=0), random_samples, last_samples


def directional_band(values):
    """Mean and asymmetric RMS deviations, split below and above the mean."""
    mean = values.mean(axis=0)
    deviations = values - mean
    lower_mask = deviations < 0
    upper_mask = deviations > 0
    lower_count = lower_mask.sum(axis=0)
    upper_count = upper_mask.sum(axis=0)
    lower_radius = np.sqrt(np.divide(
        np.where(lower_mask, deviations**2, 0).sum(axis=0), lower_count,
        out=np.zeros_like(mean), where=lower_count > 0))
    upper_radius = np.sqrt(np.divide(
        np.where(upper_mask, deviations**2, 0).sum(axis=0), upper_count,
        out=np.zeros_like(mean), where=upper_count > 0))
    return mean, np.maximum(mean - lower_radius, 0), mean + upper_radius


def write_table(path, rounds, random_values, last_values):
    lines = ["round random last"]
    for t, random_value, last_value in zip(rounds, random_values, last_values):
        lines.append(f"{int(t)} {max(random_value, FLOOR):.17g} {max(last_value, FLOOR):.17g}")
    path.write_text("\n".join(lines) + "\n")


def write_band(path, rounds, random_samples, last_samples):
    """Write a compact directional-deviation band table on log-spaced rounds."""
    _, random_low, random_high = directional_band(random_samples)
    _, last_low, last_high = directional_band(last_samples)
    targets = np.geomspace(float(rounds[0]), float(rounds[-1]), 121)
    indices = np.unique(np.clip(np.searchsorted(rounds, targets), 0, len(rounds) - 1))
    indices = np.unique(np.concatenate(([0], indices, [len(rounds) - 1])))
    lines = ["round random_low random_high last_low last_high"]
    for i in indices:
        lines.append(f"{int(rounds[i])} {max(random_low[i], FLOOR):.17g} "
                     f"{max(random_high[i], FLOOR):.17g} {max(last_low[i], FLOOR):.17g} "
                     f"{max(last_high[i], FLOOR):.17g}")
    path.write_text("\n".join(lines) + "\n")


def eta_values(dataset):
    """The etas the eta-sensitivity figure of `dataset` draws (the same for its k-means split)."""
    return ABLATIONS["eta"]["values"] if dataset.startswith("synthetic") else (0.1, 1, 10, 100, 1000)


def base_of(dataset):
    """The dataset whose protocol a name follows: itself, or the dataset a k-means split splits."""
    return dataset[:-len(KMEANS_SUFFIX)] if dataset.endswith(KMEANS_SUFFIX) else dataset


def export_long_ijcnn(out, dataset="ijcnn1-32"):
    """The 10,000-round IJCNN1 b=1 panel (Figure 1) of `dataset`: ijcnn1-32 or an IJCNN1 Dirichlet or k-means split."""
    source = RESULTS / "ijcnn-t10000" / dataset
    for regime in REGIMES:
        for mode, tag in (("full", "exact"), ("stochastic", "b1")):
            seeds = (2027,) if mode == "full" else SEEDS
            for K in KS:
                paths = [source / regime / mode / f"K{K}" / f"seed{seed}" / "metrics.csv"
                         for seed in seeds]
                rounds, random, last, random_samples, last_samples = aggregate(paths, regime)
                write_table(out / f"{dataset}-{tag}-{regime}-K{K}.dat", rounds, random, last)
                if mode == "stochastic":
                    write_band(out / f"{dataset}-{tag}-{regime}-K{K}-band.dat",
                               rounds, random_samples, last_samples)


def export_ijcnn_panel(dataset="ijcnn1-32", out=DEFAULT_OUT):
    """Only the 10,000-round IJCNN1 b=1 panel (Figure 1) of `dataset`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    export_long_ijcnn(out, dataset)
    print(f"wrote the {dataset} 10,000-round panel tables to {out}")


def export(out=DEFAULT_OUT, datasets=DATASETS):
    """Every rate and ablation table of `datasets` (the two figure datasets, default split or their k-means splits)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    for dataset in datasets:
        for regime in REGIMES:
            for K in KS:
                exact = RESULTS / "main" / dataset / regime / "full" / f"K{K}" / "seed2027" / "metrics.csv"
                rounds, random, last, _, _ = aggregate([exact], regime)
                target = out / f"{dataset}-exact-{regime}-K{K}.dat"
                write_table(target, rounds, random, last)
                generated.append(target)

                for batch in (1, APPENDIX_BATCH[base_of(dataset)]):
                    paths = [RESULTS / f"batch{batch}" / dataset / regime / f"K{K}" /
                             f"seed{seed}" / "metrics.csv" for seed in SEEDS]
                    rounds, random, last, random_samples, last_samples = aggregate(paths, regime)
                    target = out / f"{dataset}-b{batch}-{regime}-K{K}.dat"
                    write_table(target, rounds, random, last)
                    write_band(target.with_name(target.stem + "-band.dat"), rounds,
                               random_samples, last_samples)
                    generated.append(target)

    for dataset in datasets:
        for study, spec in ABLATIONS.items():
            values = eta_values(dataset) if study == "eta" else spec["values"]
            folder_prefix = {"eta": "eta", "gamma": "gamma", "partial": "p"}[study]
            for mode in ("full", "stochastic"):
                for regime in REGIMES:
                    for value in values:
                        seeds = ((2027,) if mode == "full" and not (study == "partial" and value < 1)
                                 else SEEDS)
                        paths = [RESULTS / spec["root"] / dataset / regime / mode /
                                 f"{folder_prefix}{value:g}" / f"seed{seed}" / "metrics.csv"
                                 for seed in seeds]
                        rounds, random, last, random_samples, last_samples = aggregate(paths, regime)
                        target = out / f"{dataset}-{study}-{mode}-{regime}-{value:g}.dat"
                        write_table(target, rounds, random, last)
                        if len(seeds) > 1:
                            write_band(target.with_name(target.stem + "-band.dat"), rounds,
                                       random_samples, last_samples)
                        generated.append(target)

    export_long_ijcnn(out, next(d for d in datasets if base_of(d) == "ijcnn1-32"))
    print(f"wrote {len(list(out.glob('*.dat')))} PGFPlots tables to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    export(args.out.resolve())


if __name__ == "__main__":
    main()
