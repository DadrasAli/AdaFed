"""Reproducible convergence-rate study for the paper's AdaFed algorithm.

This module runs only Algorithm 1, uses its prescribed tau_t, distinguishes the last iterate from the
uniform-random output in the theorem, and records only checkpointed metrics.

Usage (from this repository):
    python -m adafed.rate_study all --jobs 4
    python -m adafed.rate_study prepare|tune|main|eta|batch1|batch32|batch8|gamma|partial|ijcnn-t10000|export-ijcnn-t10000|plot|report
"""

import argparse
import concurrent.futures as futures
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from . import data, fl, problem


ROOT = data.ROOT / "rate-study"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TUNING = ROOT / "tuning.json"
BATCH_TUNING = {b: ROOT / f"tuning-b{b}.json" for b in (1, 8, 32)}

DATASETS = {
    "synthetic-kappa1": {"kind": "synthetic", "T": 2000},
    "synthetic-kappa10": {"kind": "synthetic", "T": 2000},
    "synthetic-kappa100": {"kind": "synthetic", "T": 2000},
    "ijcnn1-32": {"kind": "real", "T": 1000},
}
TUNE_DATASETS = ("synthetic-kappa10", "ijcnn1-32")
REGIMES = {
    "smooth-convex": {"penalty": "l2sq", "lam": 0.01, "metric": "gap", "rate": -1.0},
    "nonsmooth-convex": {"penalty": "l1", "lam": 0.01, "metric": "gap", "rate": -0.5},
    "smooth-nonconvex": {"penalty": "nonconvex", "lam": 0.01,
                         "metric": "grad_norm_sq", "rate": -1.0},
}
MODES = ("full", "stochastic")
KS = (1, 10, 100)
BENCHMARK_STOCHASTIC_SEEDS = tuple(range(2027, 2047))
BENCHMARK_BATCH = {"synthetic-kappa1": 16, "ijcnn1-32": 32}
PAPER_STOCHASTIC_SEEDS = tuple(range(2027, 2047))
PAPER_ABLATION_SEEDS = PAPER_STOCHASTIC_SEEDS
PRIMARY_BATCH = 1
APPENDIX_BATCH_BY_KIND = {"synthetic": 8, "real": 32}
EXACT_SEED = 2027
GAMMAS = (0.01, 0.1, 1.0, 10.0, 100.0)
# The five etas each eta-sensitivity figure draws (the default split's tuning grids; tools/export_rate_study_pgfplots.py).
FIGURE_ETAS = {"synthetic": (0.01, 0.1, 1.0, 10.0, 100.0), "real": (0.1, 1.0, 10.0, 100.0, 1000.0)}
PARTICIPATION = (0.25, 0.5, 1.0)
PANEL_REGIMES = ("smooth-nonconvex", "smooth-convex", "nonsmooth-convex")
_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                 "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                 "eighteen", "nineteen", "twenty")


def count_word(n):
    """A count as it is written in the report texts: a word up to twenty, digits beyond."""
    return _NUMBER_WORDS[n] if 0 <= n < len(_NUMBER_WORDS) else str(n)


def seed_word(seeds=None):
    """The number of stochastic paper seeds in words (the seeds of `seeds`, default the paper's)."""
    return count_word(len(PAPER_STOCHASTIC_SEEDS if seeds is None else seeds))


def seed_range(seeds=None):
    """The seed span as written in the texts, for example 2027--2046."""
    seeds = PAPER_STOCHASTIC_SEEDS if seeds is None else seeds
    return f"{seeds[0]}--{seeds[-1]}"


def _oracle_seeds(mode, seeds, p=1.0):
    """Seeds one setting is run with: the exact seed once if deterministic, else all of `seeds`."""
    return (EXACT_SEED,) if mode == "full" and p == 1.0 else seeds


def gamma_trajectory_count():
    """Trajectories of the gamma ablation: datasets x regimes x oracles x gammas x seeds."""
    return len(TUNE_DATASETS) * len(REGIMES) * sum(
        len(GAMMAS) * len(_oracle_seeds(mode, PAPER_ABLATION_SEEDS)) for mode in MODES)


def eta_trajectory_count(table, batch_table):
    """Trajectories of the eta ablation under the tuning records `eta_runs` reads."""
    count = 0
    for dataset in TUNE_DATASETS:
        for regime in REGIMES:
            for mode in MODES:
                source = table if mode == "full" else batch_table
                etas = {float(c["eta"]) for c in source[f"{dataset}/{regime}/{mode}"]["candidates"]}
                count += len(etas) * len(_oracle_seeds(mode, PAPER_ABLATION_SEEDS))
    return count


def partial_trajectory_count():
    """Trajectories of the partial-participation study: only the p=1 exact run is deterministic."""
    return len(TUNE_DATASETS) * len(REGIMES) * sum(
        len(_oracle_seeds(mode, PAPER_ABLATION_SEEDS, p)) for mode in MODES for p in PARTICIPATION)
IJCNN_LONG_T = 10_000


def tau_for(K):
    """Theorem 1 choice: tau=1 at K=1 and 1/((K-1)t) otherwise."""
    return "1" if K == 1 else f"1/({K - 1}*t)"


def regime_cfg(name):
    return REGIMES[name]


def base_dataset(dataset):
    """The dataset a name's protocol settings come from: IJCNN1 for its Dirichlet and k-means splits, a synthetic dataset
    for its k-means split."""
    if data.generate.is_ijcnn1_split(dataset):
        return data.generate.DIRICHLET_BASE
    return data.generate.kmeans_base(dataset) or dataset


def split_datasets(datasets, split="default"):
    """`datasets` (paper datasets) under a client split (`default` or `kmeans`), in order."""
    return tuple(data.generate.split_dataset(dataset, split) for dataset in datasets)


def split_suffix(split="default"):
    """The suffix of a split's summary files: none for the default split, -<split> otherwise."""
    return "" if split == "default" else f"-{split}"


def tuning_source(dataset):
    """The dataset whose tuned eta `dataset` uses: synthetic-kappa10 of the same split for a synthetic dataset (one eta is
    shared across the condition numbers), else the dataset itself."""
    if not base_dataset(dataset).startswith("synthetic"):
        return dataset
    return "synthetic-kappa10" if data.generate.kmeans_base(dataset) is None else data.generate.kmeans_name("synthetic-kappa10")


def _batch_tuning_datasets(batch, split):
    """The datasets tune_batch tunes for the runs of `split`: None (its own default, as before) for the default split."""
    if split == "default":
        return None
    eligible = set(paper_batch_datasets(batch))
    return split_datasets(tuple(d for d in TUNE_DATASETS if d in eligible), split)


def paper_batches(dataset):
    """Primary and appendix stochastic batches appropriate to a dataset's local size."""
    return (PRIMARY_BATCH, APPENDIX_BATCH_BY_KIND[DATASETS[base_dataset(dataset)]["kind"]])


def paper_batch_datasets(batch):
    if batch == PRIMARY_BATCH:
        return tuple(DATASETS)
    return tuple(name for name, spec in DATASETS.items()
                 if APPENDIX_BATCH_BY_KIND[spec["kind"]] == batch)


def local_smoothness(dataset, regime):
    """Maximum client logistic smoothness plus the smooth penalty's bound."""
    D = data.load(dataset)
    X = torch.from_numpy(D["X"])
    if D.get("offsets") is not None:               # unequal clients: flat rows and client offsets
        X = [X[lo:hi] for lo, hi in zip(D.get("offsets")[:-1], D.get("offsets")[1:])]
    vals = []
    for Xi in X:
        vals.append(float(torch.linalg.eigvalsh(Xi.T @ Xi)[-1]) / (4.0 * Xi.shape[0]))
    r = regime_cfg(regime)
    return max(vals) + problem.penalty_smoothness_bound(r["penalty"], r["lam"])


def ensure_references():
    for dataset in DATASETS:
        for r in REGIMES.values():
            if problem.penalty(r["penalty"])[2]:
                data.fstar(dataset, r["penalty"], r["lam"])


def prepare_dataset(name):
    """Build a dataset outside DATASETS (an IJCNN1 Dirichlet split) and its convex references."""
    ROOT.mkdir(exist_ok=True)
    if not (data.DATA / f"{name}.npz").exists():
        data.prepare(name)
    for r in REGIMES.values():
        if problem.penalty(r["penalty"])[2]:
            data.fstar(name, r["penalty"], r["lam"])


def prepare(datasets=None):
    """Build the paper datasets and references; with `datasets`, only those (and no datasets.json)."""
    if datasets is not None:
        for name in datasets:
            if name in DATASETS:
                ROOT.mkdir(exist_ok=True)
                if not (data.DATA / f"{name}.npz").exists():
                    data.prepare(name)
                for r in REGIMES.values():
                    if problem.penalty(r["penalty"])[2]:
                        data.fstar(name, r["penalty"], r["lam"])
            else:
                prepare_dataset(name)
        return
    ROOT.mkdir(exist_ok=True)
    for name in DATASETS:
        if not (data.DATA / f"{name}.npz").exists():
            data.prepare(name)
    ensure_references()
    meta = {}
    for name in DATASETS:
        D = data.load(name)
        meta[name] = {"metadata": D["meta"],
                      "local_L": {r: local_smoothness(name, r) for r in REGIMES}}
    (ROOT / "datasets.json").write_text(json.dumps(meta, indent=2))


def base_cfg(dataset, regime, mode, K, T, seed, eta, gamma=1.0, out=None, p=1.0):
    r = regime_cfg(regime)
    return {
        "dataset": dataset, "penalty": r["penalty"], "lam": r["lam"],
        "algorithm": "adafed", "mode": mode, "b": PRIMARY_BATCH,
        "K": K, "p": p, "T": T, "seed": seed, "eta": eta, "gamma": gamma,
        "tau": tau_for(K), "eval_every": 10, "device": "cpu", "dtype": "float64",
        "out": None if out is None else str(out),
    }


def _target(summary, regime):
    key = "final_mean_gap" if regime_cfg(regime)["metric"] == "gap" else "final_mean_grad_norm_sq"
    value = summary.get(key)
    return math.inf if summary["status"] != "done" or value is None or not math.isfinite(value) else value


def _candidate_summary(summary, regime):
    return {"eta": summary["eta"], "status": summary["status"],
            "criterion": _target(summary, regime),
            "last": summary["final_gap"] if regime_cfg(regime)["metric"] == "gap"
                    else summary["final_grad_norm_sq"],
            "random_output_mean": summary["final_mean_gap"] if regime_cfg(regime)["metric"] == "gap"
                                  else summary["final_mean_grad_norm_sq"]}


def tune(force=False, datasets=TUNE_DATASETS):
    """Tune exact-gradient eta at gamma=1, K=10 on absolute powers of ten.

    The selected eta is shared over K to make the K comparison controlled, and
    the synthetic-kappa10 choice is shared over all synthetic condition numbers.
    Stochastic eta is batch-specific and is handled exclusively by ``tune_batch``.
    """
    prepare(None if datasets == TUNE_DATASETS else datasets)
    table = {} if not TUNING.exists() else json.loads(TUNING.read_text())
    if force:                                        # recompute these datasets' entries (all of them: the whole table)
        table = {k: v for k, v in table.items() if k.split("/")[0] not in datasets} if datasets != TUNE_DATASETS else {}
    for dataset in datasets:
        for regime in REGIMES:
            for mode in ("full",):
                key = f"{dataset}/{regime}/{mode}"
                if key in table and table[key].get("interior") and not force:
                    print(f"tune {key}: reuse eta={table[key]['selected_eta']:g}")
                    continue
                L = local_smoothness(dataset, regime)
                centre = int(round(math.log10(1.0 / L)))
                exponents = list(range(centre - 2, centre + 3))
                seen = {}
                tune_T = 300 if base_dataset(dataset).startswith("synthetic") else 150
                for extension in range(9):
                    new = [e for e in exponents if e not in seen]
                    if new:
                        cfgs = [base_cfg(dataset, regime, mode, 10, tune_T, 1901, 10.0 ** e) for e in new]
                        for e, s in zip(new, fl.run_batch(cfgs, write=False, threads=1)):
                            seen[e] = _candidate_summary(s, regime)
                    ordered = sorted(seen)
                    values = [_target_from_record(seen[e]) for e in ordered]
                    best_pos = int(np.argmin(values))
                    if 0 < best_pos < len(ordered) - 1:
                        break
                    exponents.append(ordered[0] - 1 if best_pos == 0 else ordered[-1] + 1)
                ordered = sorted(seen)
                values = [_target_from_record(seen[e]) for e in ordered]
                best_pos = int(np.argmin(values))
                table[key] = {
                    "dataset": dataset, "regime": regime, "mode": mode,
                    "diagnostic_local_L": L, "diagnostic_1_over_L": 1.0 / L,
                    "grid_rule": "absolute integer powers of ten around 1/L; exact 1/L not used",
                    "gamma": 1.0, "K_tuned": 10, "T_tuned": tune_T,
                    "selected_eta": 10.0 ** ordered[best_pos],
                    "interior": 0 < best_pos < len(ordered) - 1,
                    "candidates": [seen[e] for e in ordered],
                }
                ROOT.mkdir(exist_ok=True)
                TUNING.write_text(json.dumps(table, indent=2))
                print(f"tune {key}: eta={table[key]['selected_eta']:g}, "
                      f"interior={table[key]['interior']}")
    return table


def _target_from_record(record):
    x = record["criterion"]
    return math.inf if x is None or not math.isfinite(x) else x


def tuned_eta(table, dataset, regime, mode):
    return float(table[f"{tuning_source(dataset)}/{regime}/{mode}"]["selected_eta"])


def _run_one(cfg):
    out = Path(cfg["out"])
    done = out / "run.json"
    if done.exists():
        return json.loads(done.read_text())
    return fl.run(cfg, write=True)


def _run_parallel(cfgs, jobs):
    pending = [c for c in cfgs if not (Path(c["out"]) / "run.json").exists()]
    print(f"{len(cfgs) - len(pending)} complete, {len(pending)} runs pending")
    if not pending:
        return
    if jobs == 1:
        for c in pending:
            _run_one(c)
        return
    with futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, s in enumerate(pool.map(_run_one, pending), 1):
            print(f"completed {i}/{len(pending)}: {s.get('out', s['config']['out'])}")


def main_runs(jobs=4, split="default"):
    """Exact-gradient rate runs shared by the primary and appendix figures (of the datasets under `split`)."""
    table = tune(datasets=split_datasets(TUNE_DATASETS, split))
    cfgs = []
    for base, ds in DATASETS.items():
        dataset = data.generate.split_dataset(base, split)
        for regime in REGIMES:
            mode = "full"
            eta = tuned_eta(table, dataset, regime, mode)
            for K in KS:
                out = RESULTS / "main" / dataset / regime / mode / f"K{K}" / f"seed{EXACT_SEED}"
                cfgs.append(base_cfg(dataset, regime, mode, K, ds["T"], EXACT_SEED, eta, out=out))
    _run_parallel(cfgs, jobs)


def gamma_runs(jobs=4, split="default"):
    """Paper gamma ablation: exact plus b=1 stochastic trajectories over the ablation seeds."""
    table = tune(datasets=split_datasets(TUNE_DATASETS, split))
    batch_table = tune_batch(1, datasets=_batch_tuning_datasets(1, split))
    cfgs = []
    for dataset in split_datasets(TUNE_DATASETS, split):
        for regime in REGIMES:
            for mode in MODES:
                eta = (tuned_eta(table, dataset, regime, mode) if mode == "full" else
                       float(batch_table[f"{dataset}/{regime}/stochastic"]["selected_eta"]))
                seeds = (EXACT_SEED,) if mode == "full" else PAPER_ABLATION_SEEDS
                for gamma in GAMMAS:
                    for seed in seeds:
                        out = (RESULTS / "gamma-paper" / dataset / regime / mode /
                               f"gamma{gamma:g}" / f"seed{seed}")
                        cfg = base_cfg(dataset, regime, mode, 10, DATASETS[base_dataset(dataset)]["T"],
                                       seed, eta, gamma=gamma, out=out)
                        if mode == "stochastic":
                            cfg["b"] = PRIMARY_BATCH
                        cfgs.append(cfg)
    _run_parallel(cfgs, jobs)


def figure_etas(dataset):
    """The five etas the eta-sensitivity figure of `dataset` (or of the dataset it splits) draws."""
    return FIGURE_ETAS[DATASETS[base_dataset(dataset)]["kind"]]


def with_figure_etas(etas, dataset):
    """`etas` with any of the eta figure's values for `dataset` that they lack, sorted."""
    return sorted(list(etas) + [e for e in figure_etas(dataset) if not any(math.isclose(e, x, rel_tol=1e-12) for x in etas)])


def eta_runs(jobs=4, split="default"):
    """Paper eta ablation: exact plus b=1 stochastic trajectories over the ablation seeds, at every candidate of the
    tuning record; for the k-means split also at any of the eta figure's five values the record lacks (its tuning grid can
    be centred elsewhere than the default split's)."""
    table = tune(datasets=split_datasets(TUNE_DATASETS, split))
    batch_table = tune_batch(1, datasets=_batch_tuning_datasets(1, split))
    cfgs = []
    for dataset in split_datasets(TUNE_DATASETS, split):
        for regime in REGIMES:
            for mode in MODES:
                source = table if mode == "full" else batch_table
                key = f"{dataset}/{regime}/{mode}"
                etas = sorted({float(c["eta"]) for c in source[key]["candidates"]})
                if split != "default":
                    etas = with_figure_etas(etas, dataset)
                seeds = (EXACT_SEED,) if mode == "full" else PAPER_ABLATION_SEEDS
                for eta in etas:
                    for seed in seeds:
                        out = (RESULTS / "eta-paper" / dataset / regime / mode /
                               f"eta{eta:g}" / f"seed{seed}")
                        cfg = base_cfg(dataset, regime, mode, 10, DATASETS[base_dataset(dataset)]["T"],
                                       seed, eta, out=out)
                        if mode == "stochastic":
                            cfg["b"] = PRIMARY_BATCH
                        cfgs.append(cfg)
    _run_parallel(cfgs, jobs)


def tune_batch(batch, force=False, datasets=None):
    """Tune eta for fixed-minibatch stochastic studies at K=10.

    The synthetic-kappa10 selection is shared by all synthetic condition
    numbers, exactly as in the original tuning protocol.  The appendix uses
    b=8 for synthetic clients (n_i=64) and b=32 for IJCNN1 (n_i=1000).
    """
    if batch not in BATCH_TUNING:
        raise ValueError(f"unsupported pilot batch {batch}; choose one of {sorted(BATCH_TUNING)}")
    prepare(datasets)
    tuning_path = BATCH_TUNING[batch]
    mode = "stochastic"
    eligible = set(paper_batch_datasets(batch))
    tune_datasets = (tuple(dataset for dataset in TUNE_DATASETS if dataset in eligible) if datasets is None
                     else tuple(datasets))
    table = {} if not tuning_path.exists() else json.loads(tuning_path.read_text())
    if force:                                        # recompute these datasets' entries (default: the whole table)
        table = {k: v for k, v in table.items() if k.split("/")[0] not in tune_datasets} if datasets is not None else {}
    if not tune_datasets:
        raise ValueError(f"batch {batch} is not an active paper batch for any tuning dataset")
    for dataset in tune_datasets:
        tune_T = 300 if base_dataset(dataset).startswith("synthetic") else 150
        for regime in REGIMES:
            key = f"{dataset}/{regime}/{mode}"
            if key in table and table[key].get("interior") and not force:
                print(f"tune b={batch} {key}: reuse eta={table[key]['selected_eta']:g}")
                continue
            L = local_smoothness(dataset, regime)
            centre = int(round(math.log10(1.0 / L)))
            exponents = list(range(centre - 2, centre + 3))
            seen = {}
            for extension in range(9):
                new = [e for e in exponents if e not in seen]
                if new:
                    cfgs = []
                    for e in new:
                        cfg = base_cfg(dataset, regime, mode, 10, tune_T, 1901, 10.0 ** e)
                        cfg["b"] = batch
                        cfgs.append(cfg)
                    for e, summary in zip(new, fl.run_batch(cfgs, write=False, threads=1)):
                        seen[e] = _candidate_summary(summary, regime)
                ordered = sorted(seen)
                values = [_target_from_record(seen[e]) for e in ordered]
                best_pos = int(np.argmin(values))
                if 0 < best_pos < len(ordered) - 1:
                    break
                exponents.append(ordered[0] - 1 if best_pos == 0 else ordered[-1] + 1)
            ordered = sorted(seen)
            values = [_target_from_record(seen[e]) for e in ordered]
            best_pos = int(np.argmin(values))
            table[key] = {
                "dataset": dataset, "regime": regime, "mode": mode, "b": batch,
                "diagnostic_local_L": L, "diagnostic_1_over_L": 1.0 / L,
                "grid_rule": "absolute integer powers of ten around 1/L; exact 1/L not used",
                "gamma": 1.0, "K_tuned": 10, "T_tuned": tune_T,
                "selected_eta": 10.0 ** ordered[best_pos],
                "interior": 0 < best_pos < len(ordered) - 1,
                "candidates": [seen[e] for e in ordered],
            }
            tuning_path.write_text(json.dumps(table, indent=2))
            print(f"tune b={batch} {key}: eta={table[key]['selected_eta']:g}, "
                  f"interior={table[key]['interior']}")
    return table


def batch_runs(batch, jobs=4, split="default"):
    """Fixed-minibatch stochastic runs over K in {1,10,100}."""
    table = tune_batch(batch, datasets=_batch_tuning_datasets(batch, split))
    mode = "stochastic"
    datasets = paper_batch_datasets(batch)
    if not datasets:
        raise ValueError(f"batch {batch} is not an active paper batch")
    seeds = PAPER_STOCHASTIC_SEEDS
    cfgs = []
    for base in datasets:
        dataset = data.generate.split_dataset(base, split)
        for regime in REGIMES:
            eta = float(table[f"{tuning_source(dataset)}/{regime}/{mode}"]["selected_eta"])
            for K in KS:
                for seed in seeds:
                    out = RESULTS / f"batch{batch}" / dataset / regime / f"K{K}" / f"seed{seed}"
                    cfg = base_cfg(dataset, regime, mode, K, DATASETS[base]["T"], seed, eta, out=out)
                    cfg["b"] = batch
                    cfgs.append(cfg)
    _run_parallel(cfgs, jobs)


def batch8_runs(jobs=4, split="default"):
    batch_runs(8, jobs, split)


def batch1_runs(jobs=4, split="default"):
    batch_runs(1, jobs, split)


def batch32_runs(jobs=4, split="default"):
    batch_runs(32, jobs, split)


def ijcnn_t10000_runs(jobs=4, dataset="ijcnn1-32", only=False):
    """Long-horizon IJCNN1 rate panel (Figure 1): exact gradients and b=1 SGD over the paper seeds.

    `dataset` is ijcnn1-32 or one of its Dirichlet splits. For ijcnn1-32 the tuning covers every tuning
    dataset, as before, unless `only`; for a Dirichlet split (or with `only`) just `dataset` is tuned.
    """
    restrict = (dataset,) if (only or dataset != "ijcnn1-32") else None
    full_table = tune(datasets=restrict or TUNE_DATASETS)
    batch_table = tune_batch(PRIMARY_BATCH, datasets=restrict)
    cfgs = []
    for regime in REGIMES:
        for mode in MODES:
            if mode == "full":
                eta = tuned_eta(full_table, dataset, regime, mode)
                seeds = (EXACT_SEED,)
            else:
                eta = float(batch_table[f"{dataset}/{regime}/{mode}"]["selected_eta"])
                seeds = PAPER_STOCHASTIC_SEEDS
            for K in KS:
                for seed in seeds:
                    out = (RESULTS / "ijcnn-t10000" / dataset / regime / mode /
                           f"K{K}" / f"seed{seed}")
                    cfg = base_cfg(dataset, regime, mode, K, IJCNN_LONG_T,
                                   seed, eta, out=out)
                    if mode == "stochastic":
                        cfg["b"] = PRIMARY_BATCH
                    cfgs.append(cfg)
    _run_parallel(cfgs, jobs)


def _ijcnn_t10000_curves(regime, mode, K, output, dataset="ijcnn1-32"):
    metric = regime_cfg(regime)["metric"]
    column = ("mean_" if output == "random" else "") + metric
    seeds = (EXACT_SEED,) if mode == "full" else PAPER_STOCHASTIC_SEEDS
    curves = [read_metrics(RESULTS / "ijcnn-t10000" / dataset / regime /
                           mode / f"K{K}" / f"seed{s}" / "metrics.csv") for s in seeds]
    x = curves[0]["round"]
    if any(not np.array_equal(c["round"], x) for c in curves[1:]):
        raise ValueError("IJCNN1 T=10000 curves do not share checkpoint rounds")
    return x, np.stack([c[column] for c in curves])


def _directional_std_band(values):
    """Mean with asymmetric one-sided standard-deviation bounds.

    Each radius is the RMS deviation from the pointwise mean conditional on
    being below or above it. Since the convergence metrics are nonnegative,
    every downward deviation is at most the mean itself; hence the lower bound
    is nonnegative without clipping.
    """
    mean = np.mean(values, axis=0)
    deviations = values - mean
    lower_mask = deviations < 0.0
    upper_mask = deviations > 0.0
    lower_count = np.sum(lower_mask, axis=0)
    upper_count = np.sum(upper_mask, axis=0)
    lower_radius = np.sqrt(np.divide(
        np.sum(np.where(lower_mask, np.square(deviations), 0.0), axis=0),
        lower_count, out=np.zeros_like(mean), where=lower_count > 0))
    upper_radius = np.sqrt(np.divide(
        np.sum(np.where(upper_mask, np.square(deviations), 0.0), axis=0),
        upper_count, out=np.zeros_like(mean), where=upper_count > 0))
    lower = mean - lower_radius
    upper = mean + upper_radius
    if np.any(lower < -1e-15):
        raise ValueError("directional standard-deviation band became negative")
    return mean, np.maximum(lower, 0.0), upper


def export_ijcnn_t10000():
    """Export compact PGFPlots tables for the long-horizon IJCNN1 rate panel."""
    out_dir = ROOT / "exports" / "ijcnn-t10000"
    out_dir.mkdir(parents=True, exist_ok=True)
    for regime in PANEL_REGIMES:
        for mode, tag in (("full", "exact"), ("stochastic", "b1")):
            for K in KS:
                xr, yr = _ijcnn_t10000_curves(regime, mode, K, "random")
                xl, yl = _ijcnn_t10000_curves(regime, mode, K, "last")
                if not np.array_equal(xr, xl):
                    raise ValueError("random-output and last-iterate checkpoints differ")
                random_curve, random_low, random_high = _directional_std_band(yr)
                last_curve, last_low, last_high = _directional_std_band(yl)
                path = out_dir / f"ijcnn1-32-{tag}-{regime}-K{K}.dat"
                with open(path, "w") as f:
                    f.write("round random last random_low random_high last_low last_high\n")
                    rows = zip(xr, random_curve, last_curve, random_low, random_high,
                               last_low, last_high)
                    for t, random_value, last_value, rlow, rhigh, llow, lhigh in rows:
                        f.write(f"{int(t)} {random_value:.17g} {last_value:.17g} "
                                f"{rlow:.17g} {rhigh:.17g} {llow:.17g} {lhigh:.17g}\n")
                if mode == "stochastic":
                    targets = np.geomspace(float(xr[0]), float(xr[-1]), 121)
                    band_indices = np.unique(np.clip(
                        np.searchsorted(xr, targets), 0, len(xr) - 1))
                    band_indices = np.unique(np.concatenate(
                        ([0], band_indices, [len(xr) - 1])))
                    band_path = out_dir / f"ijcnn1-32-{tag}-{regime}-K{K}-band.dat"
                    with open(band_path, "w") as f:
                        f.write("round random_low random_high last_low last_high\n")
                        for i in band_indices:
                            f.write(f"{int(xr[i])} {random_low[i]:.17g} "
                                    f"{random_high[i]:.17g} {last_low[i]:.17g} "
                                    f"{last_high[i]:.17g}\n")
    print(out_dir)


def partial_runs(jobs=4, split="default"):
    """Appendix-only partial participation; all ablation seeds whenever participation is random."""
    table = tune(datasets=split_datasets(TUNE_DATASETS, split))
    batch_table = tune_batch(1, datasets=_batch_tuning_datasets(1, split))
    cfgs = []
    for dataset in split_datasets(TUNE_DATASETS, split):
        for regime in REGIMES:
            for mode in MODES:
                eta = (tuned_eta(table, dataset, regime, mode) if mode == "full" else
                       float(batch_table[f"{dataset}/{regime}/stochastic"]["selected_eta"]))
                for p in PARTICIPATION:
                    seeds = _oracle_seeds(mode, PAPER_ABLATION_SEEDS, p)
                    for seed in seeds:
                        out = (RESULTS / "partial-paper" / dataset / regime / mode /
                               f"p{p:g}" / f"seed{seed}")
                        cfg = base_cfg(dataset, regime, mode, 10, DATASETS[base_dataset(dataset)]["T"],
                                       seed, eta, out=out, p=p)
                        if mode == "stochastic":
                            cfg["b"] = PRIMARY_BATCH
                        cfgs.append(cfg)
    _run_parallel(cfgs, jobs)


def fixed_etas(dataset, batch):
    """The eta every curve of a K-comparison figure (exact row and stochastic b=`batch` row, columns PANEL_REGIMES) was run
    with, read from the runs: a tuple of six, exact first. Each panel's K and seeds must share one eta."""
    etas = []
    for mode in MODES:
        for regime in PANEL_REGIMES:
            if mode == "full":
                runs = [RESULTS / "main" / dataset / regime / mode / f"K{K}" / f"seed{EXACT_SEED}" for K in KS]
            else:
                runs = [RESULTS / f"batch{batch}" / dataset / regime / f"K{K}" / f"seed{seed}"
                        for K in KS for seed in PAPER_STOCHASTIC_SEEDS]
            found = {json.loads((run / "run.json").read_text())["config"]["eta"] for run in runs}
            if len(found) != 1:
                raise ValueError(f"{dataset} {mode} {regime}: the K-comparison runs use etas {sorted(found)}, not one")
            etas.append(found.pop())
    return tuple(etas)


def eta_ablation_values(dataset, regime, mode):
    """The etas the eta ablation runs for (dataset, regime, mode), as eta_runs plans them: the candidates of the exact
    tuning record (tuning.json) or of the b=1 stochastic one (tuning-b1.json), for a k-means split with the eta figure's
    values they lack."""
    source = TUNING if mode == "full" else BATCH_TUNING[PRIMARY_BATCH]
    record = json.loads(source.read_text())[f"{dataset}/{regime}/{mode}"]
    etas = sorted({float(c["eta"]) for c in record["candidates"]})
    return tuple(with_figure_etas(etas, dataset) if data.generate.kmeans_base(dataset) else etas)


def missing_eta_runs(dataset, regime, mode, etas):
    """Those of `etas` that the eta ablation of (dataset, regime, mode) did not plan or did not finish for every seed."""
    planned = eta_ablation_values(dataset, regime, mode)
    seeds = (EXACT_SEED,) if mode == "full" else PAPER_ABLATION_SEEDS
    return [eta for eta in etas
            if not any(math.isclose(eta, p, rel_tol=1e-12) for p in planned)
            or not all((RESULTS / "eta-paper" / dataset / regime / mode / f"eta{float(eta):g}" / f"seed{seed}" / "run.json").exists()
                       for seed in seeds)]


def read_metrics(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows if r[k] != ""], dtype=float) for k in rows[0]}


def _main_curves(dataset, regime, mode, K, output):
    if mode == "stochastic":
        return _batch_curves(PRIMARY_BATCH, dataset, regime, K, output)
    seeds = (EXACT_SEED,)
    metric = regime_cfg(regime)["metric"]
    column = ("mean_" if output == "random" else "") + metric
    curves = [read_metrics(RESULTS / "main" / dataset / regime / mode / f"K{K}" /
                           f"seed{s}" / "metrics.csv") for s in seeds]
    return curves[0]["round"], np.stack([c[column] for c in curves])


def _eta_curves(dataset, regime, mode, eta, output):
    seeds = (EXACT_SEED,) if mode == "full" else PAPER_ABLATION_SEEDS
    metric = regime_cfg(regime)["metric"]
    column = ("mean_" if output == "random" else "") + metric
    curves = [read_metrics(RESULTS / "eta-paper" / dataset / regime / mode /
                           f"eta{eta:g}" / f"seed{s}" / "metrics.csv") for s in seeds]
    length = min(len(c["round"]) for c in curves)
    x = curves[0]["round"][:length]
    if any(not np.array_equal(c["round"][:length], x) for c in curves[1:]):
        raise ValueError("eta curves do not share checkpoint rounds")
    return x, np.stack([c[column][:length] for c in curves])


def _batch_curves(batch, dataset, regime, K, output):
    metric = regime_cfg(regime)["metric"]
    column = ("mean_" if output == "random" else "") + metric
    if batch not in paper_batches(dataset):
        raise ValueError(f"batch {batch} is not part of the frozen paper protocol for {dataset}")
    seeds = PAPER_STOCHASTIC_SEEDS
    curves = [read_metrics(RESULTS / f"batch{batch}" / dataset / regime /
                           f"K{K}" / f"seed{s}" / "metrics.csv") for s in seeds]
    return curves[0]["round"], np.stack([c[column] for c in curves])


def _paper_curves(dataset, regime, mode, K, output):
    """Primary-paper curves: exact gradients plus the b=1 suite over the paper seeds."""
    if mode == "stochastic":
        return _batch_curves(1, dataset, regime, K, output)
    return _main_curves(dataset, regime, mode, K, output)


def slope(x, y):
    ok = np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    start = len(x) // 2
    if len(y) and np.mean(y[start:]) <= 1e-14:
        return math.nan  # numerical-precision plateau, not a meaningful empirical rate
    return float(np.polyfit(np.log10(x[start:]), np.log10(y[start:]), 1)[0]) if len(x) - start >= 3 else math.nan


def displayed_rate(regime, mode):
    """Asymptotic power in the theorem (logs suppressed)."""
    return -0.5 if mode == "stochastic" else regime_cfg(regime)["rate"]


def plot_main():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {1: "#0072B2", 10: "#D55E00", 100: "#009E73"}
    titles = {"smooth-convex": "Smooth convex", "nonsmooth-convex": "Nonsmooth convex",
              "smooth-nonconvex": "Smooth nonconvex"}
    for dataset in DATASETS:
        for mode in MODES:
            fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.1), sharex=True)
            for col, regime in enumerate(REGIMES):
                for row, output in enumerate(("last", "random")):
                    ax = axes[row, col]
                    for K in KS:
                        x, ys = _paper_curves(dataset, regime, mode, K, output)
                        mean = np.mean(ys, axis=0)
                        ax.loglog(x, np.maximum(mean, 1e-16), color=colors[K], lw=2, label=f"K={K}")
                        if len(ys) > 1:
                            _, low, high = _directional_std_band(ys)
                            ax.fill_between(x, np.maximum(low, 1e-16),
                                            np.maximum(high, 1e-16), color=colors[K], alpha=.13, lw=0)
                    rate = displayed_rate(regime, mode)
                    anchor_x = x[len(x) // 3]
                    anchor_y = np.mean(_paper_curves(dataset, regime, mode, 10, output)[1], axis=0)[len(x) // 3]
                    guide = anchor_y * (x / anchor_x) ** rate
                    ax.loglog(x, guide, color="0.35", ls="--", lw=1.15,
                              label=rf"$T^{{{rate:g}}}$ guide")
                    ax.grid(True, which="both", color="0.9", lw=.6)
                    if row == 0:
                        ax.set_title(titles[regime])
                    if col == 0:
                        ax.set_ylabel("Last iterate\n" + _metric_label(regime))
                    elif row == 1:
                        pass
                    if row == 1 and col == 0:
                        ax.set_ylabel("Uniform-random output\n" + _metric_label(regime))
                    if row == 1:
                        ax.set_xlabel("Communication rounds $T$")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.015))
            kind = ("exact local gradients" if mode == "full" else
                    f"stochastic local gradients (b=1; {len(PAPER_STOCHASTIC_SEEDS)} seeds; directional SD band)")
            fig.suptitle(f"AdaFed convergence on {_dataset_label(dataset)} — {kind}", y=1.055, fontsize=13)
            fig.tight_layout()
            stem = FIGURES / f"rates-{dataset}-{mode}"
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
            plt.close(fig)


def plot_conditioning():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"synthetic-kappa1": "#0072B2", "synthetic-kappa10": "#D55E00",
              "synthetic-kappa100": "#009E73"}
    FIGURES.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.1), sharex=True)
        for col, regime in enumerate(REGIMES):
            for row, output in enumerate(("last", "random")):
                ax = axes[row, col]
                for dataset, color in colors.items():
                    x, ys = _paper_curves(dataset, regime, mode, 10, output)
                    ax.loglog(x, np.maximum(np.mean(ys, axis=0), 1e-16), lw=2, color=color,
                              label=rf"$\kappa={dataset.rsplit('kappa', 1)[1]}$")
                ax.grid(True, which="both", color="0.9", lw=.6)
                if row == 0:
                    ax.set_title(regime.replace("-", " ").title())
                if col == 0:
                    prefix = "Last iterate" if row == 0 else "Uniform-random output"
                    ax.set_ylabel(prefix + "\n" + _metric_label(regime))
                if row == 1:
                    ax.set_xlabel("Communication rounds $T$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.015))
        fig.suptitle(f"Conditioning ablation at K=10 — {mode}", y=1.055, fontsize=13)
        fig.tight_layout()
        stem = FIGURES / f"conditioning-{mode}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_gamma():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    for dataset in TUNE_DATASETS:
        for mode in MODES:
            fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35))
            for ax, regime in zip(axes, REGIMES):
                last, random = [], []
                metric = regime_cfg(regime)["metric"]
                for gamma in GAMMAS:
                    seeds = (EXACT_SEED,) if mode == "full" else PAPER_ABLATION_SEEDS
                    curves = [read_metrics(RESULTS / "gamma-paper" / dataset / regime / mode /
                                            f"gamma{gamma:g}" / f"seed{s}" / "metrics.csv")
                              for s in seeds]
                    last.append(float(np.mean([m[metric][-1] for m in curves])))
                    random.append(float(np.mean([m["mean_" + metric][-1] for m in curves])))
                ax.loglog(GAMMAS, np.maximum(last, 1e-16), "o-", lw=1.8, label="last")
                ax.loglog(GAMMAS, np.maximum(random, 1e-16), "s--", lw=1.8,
                          label="uniform-random")
                ax.set_title(regime.replace("-", " ").title())
                ax.set_xlabel(r"Accumulator initialization $\gamma$")
                ax.set_ylabel(_metric_label(regime))
                ax.grid(True, which="both", color="0.9", lw=.6)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=2, loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.04))
            seed_note = "deterministic" if mode == "full" else f"b=1; mean of {len(PAPER_ABLATION_SEEDS)} seeds"
            fig.suptitle(f"Gamma ablation: {_dataset_label(dataset)}, {mode}, K=10 ({seed_note})",
                         y=1.13, fontsize=13)
            fig.tight_layout()
            stem = FIGURES / f"gamma-{dataset}-{mode}"
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_partial():
    """Appendix-only partial-participation plots outside the current theorem."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    colors = {0.25: "#CC79A7", 0.5: "#E69F00", 1.0: "#0072B2"}
    FIGURES.mkdir(parents=True, exist_ok=True)
    for dataset in TUNE_DATASETS:
        for mode in MODES:
            fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.1), sharex=True)
            for col, regime in enumerate(REGIMES):
                metric = regime_cfg(regime)["metric"]
                for row, output in enumerate(("last", "random")):
                    ax = axes[row, col]
                    column = ("mean_" if output == "random" else "") + metric
                    for p, color in colors.items():
                        seeds = ((EXACT_SEED,) if mode == "full" and p == 1.0
                                 else PAPER_ABLATION_SEEDS)
                        curves = [read_metrics(RESULTS / "partial-paper" / dataset / regime / mode /
                                                f"p{p:g}" / f"seed{s}" / "metrics.csv")
                                  for s in seeds]
                        values = np.mean(np.stack([m[column] for m in curves]), axis=0)
                        ax.loglog(curves[0]["round"], np.maximum(values, 1e-16), color=color, lw=2,
                                  label=rf"$p={p:g}$")
                    ax.grid(True, which="both", color="0.9", lw=.6)
                    if row == 0:
                        ax.set_title(regime.replace("-", " ").title())
                    if col == 0:
                        prefix = "Last iterate" if row == 0 else "Uniform-random output"
                        ax.set_ylabel(prefix + "\n" + _metric_label(regime))
                    if row == 1:
                        ax.set_xlabel("Communication rounds $T$")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False,
                       bbox_to_anchor=(.5, 1.015))
            fig.suptitle(f"Partial participation at K=10 — {_dataset_label(dataset)}, {mode}",
                         y=1.055, fontsize=13)
            fig.tight_layout()
            stem = FIGURES / f"partial-{dataset}-{mode}"
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
            plt.close(fig)


def _paired_panel(dataset, varying, stochastic_batch=None):
    """Two oracle rows by three regime columns; solid=random output, dashed=last iterate."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    table = json.loads(TUNING.read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.55), sharex=True)
    titles = {"smooth-nonconvex": "Smooth nonconvex", "smooth-convex": "Smooth convex",
              "nonsmooth-convex": "Nonsmooth convex"}
    if varying == "eta":
        reference_key = f"{dataset}/smooth-convex/full"
        values = sorted({float(c["eta"]) for c in table[reference_key]["candidates"]})
        colors = dict(zip(values, plt.get_cmap("viridis")(np.linspace(.08, .88, len(values)))))
        color_handles = [Line2D([0], [0], color=colors[v], lw=2.2, label=rf"$\eta={v:g}$")
                         for v in values]
    else:
        values = KS
        palette = ("#0072B2", "#D55E00", "#009E73")
        colors = dict(zip(values, palette))
        color_handles = [Line2D([0], [0], color=colors[v], lw=2.2, label=rf"$K={v}$")
                         for v in values]

    for row, mode in enumerate(MODES):
        for col, regime in enumerate(PANEL_REGIMES):
            ax = axes[row, col]
            metric = regime_cfg(regime)["metric"]
            for value in values:
                if varying == "eta":
                    xr, yr = _eta_curves(dataset, regime, mode, value, "random")
                    xl, yl = _eta_curves(dataset, regime, mode, value, "last")
                else:
                    if mode == "stochastic" and stochastic_batch is not None:
                        xr, yr = _batch_curves(stochastic_batch, dataset, regime, value, "random")
                        xl, yl = _batch_curves(stochastic_batch, dataset, regime, value, "last")
                    else:
                        xr, yr = _main_curves(dataset, regime, mode, value, "random")
                        xl, yl = _main_curves(dataset, regime, mode, value, "last")
                random_curve = np.mean(yr, axis=0)
                last_curve = np.mean(yl, axis=0)
                ax.loglog(xr, np.maximum(random_curve, 1e-16), color=colors[value], lw=2.05,
                          solid_capstyle="round")
                markevery = max(1, len(xl) // 13)
                ax.loglog(xl, np.maximum(last_curve, 1e-16), color=colors[value], lw=1.15,
                          ls=(0, (3, 2)), marker="o", ms=2.5, markevery=markevery, alpha=.82)
            ax.grid(True, which="major", color="#d7d7d7", lw=.65)
            ax.grid(False, which="minor")
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(titles[regime], fontsize=11.5, pad=8)
            if row == 1:
                ax.set_xlabel("Communication rounds $t$")
            ax.set_ylabel(r"$\|\nabla F(x)\|^2$" if metric == "grad_norm_sq"
                          else r"$F(x)-F^\star$")
            if varying == "K":
                if mode == "stochastic":
                    batch = PRIMARY_BATCH if stochastic_batch is None else stochastic_batch
                    batch_tuning = json.loads(BATCH_TUNING[batch].read_text())
                    eta = float(batch_tuning[f"{dataset}/{regime}/{mode}"]["selected_eta"])
                else:
                    eta = tuned_eta(table, dataset, regime, mode)
                ax.text(.035, .055, rf"fixed $\eta={eta:g}$", transform=ax.transAxes,
                        fontsize=8.2, color="0.35")

    style_handles = [
        Line2D([0], [0], color="0.18", lw=2.05, ls="-", label="uniform-random mean"),
        Line2D([0], [0], color="0.18", lw=1.15, ls=(0, (3, 2)), marker="o", ms=3,
               label="last iterate"),
    ]
    if varying == "eta":
        heading = rf"Step-size sensitivity at $K=10$, $\gamma=1$ — {_dataset_label(dataset)}"
    elif stochastic_batch is not None:
        heading = (rf"Local-step comparison at fixed $\eta$, $\gamma=1$ — {_dataset_label(dataset)}; "
                   rf"stochastic row $b={stochastic_batch}$")
    else:
        heading = rf"Local-step comparison at fixed $\eta$, $\gamma=1$ — {_dataset_label(dataset)}"
    fig.suptitle(heading, fontsize=13, y=.995)
    fig.legend(handles=color_handles + style_handles, ncol=len(color_handles) + 2, loc="upper center",
               bbox_to_anchor=(.5, .947), frameon=False, columnspacing=1.25, handlelength=2.5)
    fig.text(.012, .675, "Exact local gradients", rotation=90, va="center", ha="center",
             fontsize=10.5, weight="semibold")
    batch_seed_count = len(PAPER_STOCHASTIC_SEEDS)
    stochastic_label = (f"Stochastic local gradients, b={stochastic_batch}\n"
                        f"(mean of {batch_seed_count} seeds)"
                        if stochastic_batch is not None else
                        f"Stochastic local gradients, b=1\n(mean of {batch_seed_count} seeds)")
    fig.text(.012, .255, stochastic_label, rotation=90,
             va="center", ha="center", fontsize=10.5, weight="semibold")
    fig.tight_layout(rect=(.035, .015, 1, .885), w_pad=1.1, h_pad=1.15)
    suffix = f"-b{stochastic_batch}" if stochastic_batch is not None else ""
    stems = [FIGURES / f"{varying}-{dataset}{suffix}"]
    if varying == "K" and stochastic_batch == 1:
        stems.append(FIGURES / f"K-{dataset}")  # b=1 is the primary-paper default
    for stem in stems:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_requested_panels():
    """The four paired-output panels: eta/K sensitivity on synthetic and real data."""
    if (RESULTS / "eta-paper").exists():
        for dataset in TUNE_DATASETS:
            _paired_panel(dataset, "eta")
    for dataset in TUNE_DATASETS:
        for batch in paper_batches(dataset):
            if (RESULTS / f"batch{batch}" / dataset).exists():
                _paired_panel(dataset, "K", stochastic_batch=batch)


def _metric_label(regime):
    return r"Objective gap $F(x)-F^\star$" if regime_cfg(regime)["metric"] == "gap" \
        else r"Squared gradient norm $\|\nabla F(x)\|^2$"


def _dataset_label(dataset):
    return "IJCNN1 (32,000 examples)" if dataset == "ijcnn1-32" else \
        rf"synthetic ($\kappa={dataset.rsplit('kappa', 1)[1]}$)"


def plot():
    plot_main()
    plot_conditioning()
    plot_requested_panels()
    if (RESULTS / "gamma-paper").exists():
        plot_gamma()
    if (RESULTS / "partial-paper").exists():
        plot_partial()


def report(split="default"):
    """rates.csv, stochastic-rates.csv and REPORT.md; for the k-means split only the two summaries, as rates-kmeans.csv and
    stochastic-rates-kmeans.csv (the report's text describes the default split's results)."""
    table = json.loads(TUNING.read_text())
    batch_table_b1 = json.loads(BATCH_TUNING[PRIMARY_BATCH].read_text())
    rows = []
    for dataset in split_datasets(DATASETS, split):
        for regime in REGIMES:
            for mode in MODES:
                for K in KS:
                    for output in ("last", "random"):
                        x, ys = _paper_curves(dataset, regime, mode, K, output)
                        mean = np.mean(ys, axis=0)
                        rows.append({"dataset": dataset, "regime": regime, "mode": mode, "K": K,
                                     "output": output, "slope_second_half": slope(x, mean),
                                     "final_mean": float(mean[-1]), "seeds": len(ys)})
    ROOT.mkdir(exist_ok=True)
    with open(ROOT / f"rates{split_suffix(split)}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader(); w.writerows(rows)

    batch_rows = []
    for dataset in split_datasets(DATASETS, split):
        for batch in paper_batches(dataset):
            for regime in REGIMES:
                for K in KS:
                    for output in ("last", "random"):
                        x, ys = _batch_curves(batch, dataset, regime, K, output)
                        mean = np.mean(ys, axis=0)
                        batch_rows.append({"dataset": dataset, "batch": batch, "regime": regime,
                                           "K": K, "output": output,
                                           "slope_second_half": slope(x, mean),
                                           "final_mean": float(mean[-1]), "seeds": len(ys)})
    with open(ROOT / f"stochastic-rates{split_suffix(split)}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=batch_rows[0])
        w.writeheader(); w.writerows(batch_rows)
    if split != "default":
        print(ROOT / f"rates{split_suffix(split)}.csv", ROOT / f"stochastic-rates{split_suffix(split)}.csv")
        return

    lines = ["# AdaFed convergence-rate study", "",
             "This study runs only Algorithm 1 from the paper. Metrics are evaluated on the server iterate "
             "before each update, matching the paper's set `{x_1,...,x_T}`. The random-output curve is not "
             "one random draw: it is the exact conditional expectation of the theorem metric, accumulated "
             "online over all iterates. The last-iterate curve is reported separately and is not claimed by "
             "the theorem. The full data-generation, objective, oracle, tuning, and execution specification "
             "is in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).", "", "## Protocol", "",
             "- Full participation; 32 clients; float64; common zero initialization.",
             "- `K in {1,10,100}`. The prescribed scale is `tau_t=1` for K=1 and "
             "`tau_t=1/((K-1)t)` for K>1.",
             "- Exact mode uses each client's whole local dataset. Primary stochastic rate curves use "
             f"uniform-with-replacement minibatches of size b=1 and seeds {seed_range().replace('--', '–')}. The matched appendix "
             f"uses b=8 for synthetic clients and b=32 for IJCNN1, again with {seed_word(BENCHMARK_STOCHASTIC_SEEDS)} seeds.",
             "- Synthetic designs have exactly controlled pooled Gram condition numbers 1, 10, and 100, "
             "with the largest Gram eigenvalue fixed. IJCNN1 uses 32,000 examples and 22 features.",
             "- Smooth convex: L2-squared logistic regression; nonsmooth convex: L1 logistic regression; "
             "smooth nonconvex: logistic loss plus `lambda sum_j x_j^2/(1+x_j^2)`. All use lambda=0.01.",
             "- The total nonconvex objectives, not merely their penalties, were checked to have a negative "
             "Hessian eigenvalue on every dataset.",
             "- Convex reference optima use high-accuracy centralized Newton (L2) or FISTA plus support "
             "Newton polishing (L1).", "",
             "## Step-size tuning", "",
             "Here `eta` is the paper's parameter and `gamma=S_1`; therefore the initial shared step is "
             "`eta_1=eta/gamma`. Eta was tuned with gamma=1 and K=10. The search uses absolute integer "
             "powers of ten around the diagnostic local `1/L`, never the exact computed `1/L`, and extends "
             "until the selected value is interior. One eta is shared across K, and the kappa=10 eta is "
             "shared across synthetic condition numbers, so the K and conditioning comparisons do not "
             "retune away the effect being studied. The table below reports the exact-gradient choice and "
             "each active paper stochastic batch.", "", "| Dataset | Regime | Oracle | 1/L diagnostic | eta | Interior? |", "|---|---|---:|---:|---:|:---:|"]
    for key, t in table.items():
        if t["mode"] == "full":
            lines.append(f"| {t['dataset']} | {t['regime']} | exact | {t['diagnostic_1_over_L']:.3g} | "
                         f"{t['selected_eta']:g} | {'yes' if t['interior'] else 'NO'} |")
    for dataset in TUNE_DATASETS:
        for batch in paper_batches(dataset):
            batch_table = json.loads(BATCH_TUNING[batch].read_text())
            for regime in REGIMES:
                t = batch_table[f"{dataset}/{regime}/stochastic"]
                lines.append(f"| {dataset} | {regime} | stochastic b={batch} | "
                             f"{t['diagnostic_1_over_L']:.3g} | {t['selected_eta']:g} | "
                             f"{'yes' if t['interior'] else 'NO'} |")
    lines += ["", "## Representative empirical slopes", "",
              "The numbers below are ordinary least-squares slopes of log(metric) against log(T) over the "
              "second half of saved checkpoints. They are diagnostics over a finite horizon, not theorem "
              f"estimates. Stochastic entries use the pointwise arithmetic mean over {seed_word()} seeds at b=1.", "",
              "| Dataset | Regime | Oracle | K | Last | Uniform-random | Predicted |", "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in ("synthetic-kappa10", "ijcnn1-32"):
        for regime in REGIMES:
            for mode in MODES:
                for K in KS:
                    a = next(r for r in rows if r["dataset"] == dataset and r["regime"] == regime and
                             r["mode"] == mode and r["K"] == K and r["output"] == "last")
                    b = next(r for r in rows if r["dataset"] == dataset and r["regime"] == regime and
                             r["mode"] == mode and r["K"] == K and r["output"] == "random")
                    predicted = ("-1 to -1/2" if mode == "stochastic" and regime != "nonsmooth-convex"
                                 else f"{displayed_rate(regime, mode):.1f}")
                    lines.append(f"| {dataset} | {regime} | {mode} | {K} | {a['slope_second_half']:.2f} | "
                                 f"{b['slope_second_half']:.2f} | {predicted} |")
    def final_value(dataset, regime, mode, K):
        return next(r["final_mean"] for r in rows if r["dataset"] == dataset and r["regime"] == regime and
                    r["mode"] == mode and r["K"] == K and r["output"] == "random")

    syn_gains = [final_value("synthetic-kappa10", r, "stochastic", 1) /
                 final_value("synthetic-kappa10", r, "stochastic", 100) for r in REGIMES]
    real_gains = [final_value("ijcnn1-32", r, "stochastic", 1) /
                  final_value("ijcnn1-32", r, "stochastic", 100) for r in REGIMES]
    lines += ["", "## Results at a glance", "",
              "- In every exact-gradient smooth case, the uniform-random output has slope approximately "
              "-1.00, matching the theorem's nominal rate. The exact nonsmooth slopes range from about -0.41 to "
              "-0.61 around the predicted -0.5.",
              "- Exact smooth last iterates are faster than the theorem output: K=10 and K=100 commonly "
              "show a -2 slope, while K=1 reaches floating-point precision. This is empirical last-iterate "
              "behavior, not a claim of the theorem.",
              "- With stochastic gradients, smooth problems show the expected crossover between the 1/T "
              "term and the noise-controlled 1/sqrt(T) term. The uniform-random curve is much more stable "
              "than the last iterate.",
              f"- At b=1 on synthetic kappa=10, increasing K from 1 to 100 improves the final "
              f"uniform-random metric by factors {syn_gains[0]:.1f} (smooth convex), "
              f"{syn_gains[1]:.1f} (nonsmooth convex), and {syn_gains[2]:.1f} (smooth nonconvex). "
              f"On IJCNN1 the corresponding factors are {real_gains[0]:.1f}, "
              f"{real_gains[1]:.1f}, and {real_gains[2]:.1f}.",
             "- Varying the synthetic Gram condition number from 1 to 100 changes constants but not the "
             "qualitative rates. The conditioning panels make this visible without separately retuning eta.",
             f"- The eta ablation contains {eta_trajectory_count(table, batch_table_b1)} and the gamma "
             f"ablation {gamma_trajectory_count()} full-horizon trajectories. Their stochastic "
             f"rows use b=1 and the same {seed_word()} seeds as the primary paper figures.",
             "- The gamma response can be non-monotone because gamma changes the initial shared step eta/gamma. "
             "The exported ablation curves report the complete response rather than selecting gamma again.",
             f"- The appendix-only partial-participation study contains {partial_trajectory_count()} "
             f"trajectories. Every random-participation setting uses {seed_word(PAPER_ABLATION_SEEDS)} "
             "seeds; only the exact-gradient p=1 baseline is deterministic. "
             "Lower p raises the stochastic error level in every regime. It is separated because the "
             "present theorem and Algorithm 1 specify full participation.", "",
             "## Eta, gamma, and participation ablations", "",
             "All three ablations fix K=10 and use the full main-study horizon. The eta sweep fixes gamma=1 "
             "and tests {0.01, 0.1, 1, 10, 100} on synthetic kappa=10 and {0.1, 1, 10, 100, 1000} on "
             "IJCNN1. The gamma sweep fixes the tuned eta and tests {0.01, 0.1, 1, 10, 100}. Both use full "
             "participation. The participation sweep fixes tuned eta and gamma=1 and tests p in "
             "{0.25, 0.5, 1} with independent Bernoulli client sampling and the implementation's "
             "Horvitz--Thompson scaling.", "",
             "Exact eta and gamma rows are deterministic; their stochastic rows use b=1 and pointwise "
             f"means over seeds {seed_range(PAPER_ABLATION_SEEDS)}. In the participation study, exact rows at "
             f"p<1 also use all {seed_word(PAPER_ABLATION_SEEDS)} "
             "seeds because client selection is random. These participation results are an empirical robustness check, not a "
             "partial-participation theorem.", "",
             "## Reading the figures", "",
              "Each main figure is a 2x3 panel: columns are the three regularity regimes; rows are the last "
              "iterate and theorem-supported uniform-random output. Color denotes K. Dashed gray lines are "
              "slope guides only. For stochastic smooth cases, the guide is the asymptotic 1/sqrt(T) term "
             "of the mixed theorem bound. Stochastic bands show directional one-sided RMS standard "
             f"deviations over {seed_word()} seeds. Separate "
             "figures isolate conditioning, eta, gamma, and partial participation at K=10.",
              "",
              "The four paired sensitivity figures requested for direct comparison are `eta-synthetic-"
              "kappa10`, `eta-ijcnn1-32`, `K-synthetic-kappa10`, and `K-ijcnn1-32` (each as PDF and PNG). "
              "Their rows are exact and stochastic local gradients, their columns are smooth nonconvex, "
              "smooth convex, and nonsmooth convex, solid lines are uniform-random-output means, and "
             "dashed lines with sparse dots are last iterates. Primary stochastic curves are pointwise "
             f"means over {seed_word()} b=1 seeds; only major grid lines are shown. The appendix versions are "
             "`K-synthetic-kappa10-b8` and `K-ijcnn1-32-b32`. The corresponding appendix ablation figures "
             "are `gamma-<dataset>-<mode>` and `partial-<dataset>-<mode>`; the standalone PGFPlots paper "
             "document combines each ablation into the same requested 2x3 layout.", "",
              "## Scope", "",
              "These are rate illustrations, not a practical FL benchmark. The deterministic client "
              "partition has moderate margin-based heterogeneity held fixed across methods. The completed "
              "partial-participation study is appendix-only because the current theorem states the "
              "full-participation algorithm. No competing method or undocumented AdaFed variant is run."]
    def batch_final(batch, dataset, regime, K):
        _, ys = _batch_curves(batch, dataset, regime, K, "random")
        return float(np.mean(ys[:, -1]))

    lines += ["", f"## {seed_word(BENCHMARK_STOCHASTIC_SEEDS).capitalize()}-seed stochastic minibatch comparison", "",
              "The b=1 primary matrix and the size-matched appendix batches (b=8 synthetic, b=32 IJCNN1) "
              f"use the same {seed_word(BENCHMARK_STOCHASTIC_SEEDS)} seeds, K values, and horizons. Eta is tuned "
              "separately for each dataset family, regime, and batch size at K=10, then held fixed across K. "
              "Synthetic eta is tuned at kappa=10 and shared across condition numbers. Exact-gradient "
              "trajectories are common to both figure sets.", "",
              "Final uniform-random-output improvement factor (K=1 metric divided by K=100 metric):", "",
              "| Dataset | b | Smooth nonconvex | Smooth convex | Nonsmooth convex |",
              "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for batch in paper_batches(dataset):
            gains = {r: batch_final(batch, dataset, r, 1) /
                          batch_final(batch, dataset, r, 100) for r in REGIMES}
            lines.append(f"| {dataset} | {batch} | {gains['smooth-nonconvex']:.2f} | "
                         f"{gains['smooth-convex']:.2f} | {gains['nonsmooth-convex']:.2f} |")
    syn_b8_gains = {r: batch_final(8, "synthetic-kappa10", r, 1) /
                         batch_final(8, "synthetic-kappa10", r, 100) for r in REGIMES}
    lines += ["", f"At synthetic kappa=10 and b=8, the K=1-to-K=100 factors are "
              f"{syn_b8_gains['smooth-nonconvex']:.2f} (smooth nonconvex), "
              f"{syn_b8_gains['smooth-convex']:.2f} (smooth convex), and "
              f"{syn_b8_gains['nonsmooth-convex']:.2f} (nonsmooth convex). The convex separations "
              "are smaller than at b=1, consistent with reduced stochastic noise. The nonconvex "
              "separation is larger, but its selected eta changes from 1 at b=1 to 10 at b=8, so "
              "that cross-batch contrast is not a causal batch-size estimate."]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "REPORT.md")


def cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "tune", "main", "eta", "batch1", "batch32", "batch8",
                                        "gamma", "partial", "ijcnn-t10000", "export-ijcnn-t10000",
                                        "plot", "report", "all"))
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--force-tune", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "prepare": prepare()
    elif args.command == "tune": tune(force=args.force_tune)
    elif args.command == "main": main_runs(args.jobs)
    elif args.command == "eta": eta_runs(args.jobs)
    elif args.command == "batch1": batch1_runs(args.jobs)
    elif args.command == "batch32": batch32_runs(args.jobs)
    elif args.command == "batch8": batch8_runs(args.jobs)
    elif args.command == "gamma": gamma_runs(args.jobs)
    elif args.command == "partial": partial_runs(args.jobs)
    elif args.command == "ijcnn-t10000": ijcnn_t10000_runs(args.jobs)
    elif args.command == "export-ijcnn-t10000": export_ijcnn_t10000()
    elif args.command == "plot": plot()
    elif args.command == "report": report()
    else:
        prepare(); tune(force=args.force_tune); main_runs(args.jobs); eta_runs(args.jobs)
        batch1_runs(args.jobs); batch8_runs(args.jobs); batch32_runs(args.jobs); gamma_runs(args.jobs)
        partial_runs(args.jobs); ijcnn_t10000_runs(args.jobs); plot(); report()


if __name__ == "__main__":
    cli()
