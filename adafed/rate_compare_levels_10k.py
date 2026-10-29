"""Tune and compare adaptive FL by adaptation level over 10,000 rounds.

Usage: python -m adafed.rate_compare_levels_10k all --jobs 4
Output: rate-study/comparison-by-adaptation-10000/
"""

import argparse
import contextlib
import csv
import io
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import (adaptive_baselines, extra_adaptive_baselines as extra, fl,
               rate_compare_joint as joint, rate_compare_selected as selected,
               rate_study as study)


T = 10_000
ROOT = study.ROOT / "comparison-by-adaptation-10000"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TUNING = ROOT / "tuning.json"
DATASETS = selected.DATASETS
GROUPS = {
    "server": ("adafed", "fedadagrad", "fedadam", "fedur"),
    "client": ("adafed", "local_adaalter", "fedsps", "fedlada"),
    "both": ("adafed", "fafed", "fedada2"),
}
NEW_METHODS = ("fedadam",) + extra.METHODS
GRID = {
    "fedadam": ((1/128, 1/8, 2.), (.01, .1, 1., 10.)),
    "local_adaalter": ((.001, .01, .1, 1., 10.), (.01, .1, 1.)),
    "fedsps": ((.1, 1., 10., 100.), (.01, .1, 1., 10.)),
    "fedada2": ((.0001, .001, .01, .1, 1.), (.01, .1, 1., 10.)),
    "fedur": ((.001, .01, .1, 1.), (.1, .5, .9, 1.)),
}
BOUNDS = {
    "fedadam": ((1/128000, 20.), (.001, 100.)),
    "local_adaalter": ((1e-5, 100.), (1e-4, 10.)),
    "fedsps": ((.001, 1000.), (.0001, 100.)),
    "fedada2": ((1e-6, 10.), (.001, 100.)),
    "fedur": ((.0001, 10.), (.1, 1.)),
}
PARAMS = {
    "fedadam": ("client_scale", "server_eta"),
    "local_adaalter": ("client_scale", "epsilon"),
    "fedsps": ("polyak_c", "gamma_cap_scale"),
    "fedada2": ("client_scale", "server_eta"),
    "fedur": ("eta0_scale", "alpha"),
}


def key(dataset, regime, mode, method):
    return f"{dataset}/{regime}/{mode}/{method}"


def config(dataset, regime, mode, method, horizon, seed, a, b, out=None):
    if method == "fedadam":
        return selected.config(dataset, regime, mode, method, horizon, seed, a, b, out)
    L = float(study.data.load(dataset)["meta"]["L"])
    cfg = study.base_cfg(dataset, regime, mode, selected.K, horizon, seed, 1., out=out)
    if mode == "stochastic":
        cfg["b"] = study.BENCHMARK_BATCH[study.base_dataset(dataset)]
    cfg.update(algorithm=method, p=1.)
    if method == "local_adaalter":
        cfg.update(client_lr=a/L, client_scale=a, epsilon=b, b0=1.)
    elif method == "fedsps":
        cfg.update(polyak_c=a, gamma_cap_scale=b, gamma_cap=b/L, lower_bound=0.)
    elif method == "fedada2":
        cfg.update(client_lr=a/L, client_scale=a, server_eta=b,
                   client_offset=.001, server_offset=.001, beta1=.9)
    elif method == "fedur":
        cfg.update(eta0_scale=a, eta0=a/L, alpha=b, server_offset=.001,
                   beta1=.9, beta2=.999, smoothness_fallback=L)
    else:
        raise ValueError(method)
    return cfg


def _run_one(cfg):
    if cfg["algorithm"] in ("fedadam", "fedadagrad", "adafed"):
        return fl.run(cfg, write=cfg["out"] is not None)
    if cfg["algorithm"] in adaptive_baselines.METHODS:
        return adaptive_baselines.run(cfg, write=cfg["out"] is not None)
    return extra.run(cfg, write=cfg["out"] is not None)


def _score(summary, regime):
    value = (summary["final_mean_grad_norm_sq"] if regime == "smooth-nonconvex"
             else summary["final_mean_gap"])
    return value if summary["status"] == "done" and value is not None and math.isfinite(value) else None


def _best(records):
    finite = [r for r in records if r["criterion"] is not None]
    if not finite:
        return None
    best = min(r["criterion"] for r in finite)
    near = [r for r in finite if r["criterion"] <= best + .001 * max(abs(best), 1e-16)]
    return min(near, key=lambda r: (abs(math.log10(r["a"])) + abs(math.log10(r["b"])),
                                    r["criterion"], r["a"], r["b"]))


def _tune_cell(dataset, regime, mode, method, jobs):
    horizon = 300 if dataset.startswith("synthetic") else 150
    axes = [list(axis) for axis in GRID[method]]
    records = []
    for extension in range(7):
        tested = {(r["a"], r["b"]) for r in records}
        points = [(a, b) for a in axes[0] for b in axes[1] if (a, b) not in tested]
        cfgs = [config(dataset, regime, mode, method, horizon, 1901, a, b)
                for a, b in points]
        if method == "fedadam" and cfgs:
            with contextlib.redirect_stdout(io.StringIO()):
                summaries = fl.run_batch(cfgs, write=False, threads=1)
        elif cfgs:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                summaries = list(pool.map(_run_one, cfgs))
        else:
            summaries = []
        records += [{"a": a, "b": b, "criterion": _score(s, regime), "status": s["status"]}
                    for (a, b), s in zip(points, summaries)]
        best = _best(records)
        if best is None:
            raise RuntimeError(f"all candidates diverged: {key(dataset, regime, mode, method)}")
        expanded = False
        for i, name in enumerate(("a", "b")):
            lo, hi = BOUNDS[method][i]
            axis = axes[i]
            if best[name] == axis[0] and axis[0] / 10 >= lo * (1 - 1e-12):
                axis.insert(0, axis[0] / 10)
                expanded = True
            elif best[name] == axis[-1] and axis[-1] * 10 <= hi * (1 + 1e-12):
                axis.append(axis[-1] * 10)
                expanded = True
        if not expanded:
            break
    best = _best(records)
    return {"dataset": dataset, "regime": regime, "mode": mode, "method": method,
            "T_tuned": horizon, "seed_tuned": 1901,
            "criterion": "final uniform-random-output mean",
            "selection_rule": "within-0.1-percent-of-minimum-smallest-log-magnitude",
            "parameter_names": PARAMS[method], "initial_grid": GRID[method],
            "bounds": BOUNDS[method], "search_complete": True,
            "selected": dict(zip(PARAMS[method], (best["a"], best["b"]))),
            "selected_score": best["criterion"], "candidates": records}


def tune(jobs=4, datasets=None):
    ROOT.mkdir(parents=True, exist_ok=True)
    table = json.loads(TUNING.read_text()) if TUNING.exists() else {}
    for dataset in (DATASETS if datasets is None else datasets):
        for regime in study.REGIMES:
            for mode in study.MODES:
                for method in NEW_METHODS:
                    k = key(dataset, regime, mode, method)
                    if k in table and table[k].get("search_complete"):
                        continue
                    table[k] = _tune_cell(dataset, regime, mode, method, jobs)
                    TUNING.write_text(json.dumps(table, indent=2, allow_nan=False))
                    print(f"tuned {k}: {table[k]['selected']} score={table[k]['selected_score']:.4g}", flush=True)
    return table


def _selected_source(dataset, regime, mode, method, seed):
    return (study.ROOT / "comparison-selected-joint-10000" / "results" / dataset /
            regime / mode / method / f"seed{seed}")


def _selected_config(dataset, regime, mode, method, seed, joint_table, baseline_table):
    if method == "adafed":
        setting = joint_table[joint.key(dataset, regime, mode)]
        return joint.config(dataset, regime, mode, T, seed,
                            setting["selected_eta"], setting["selected_gamma"],
                            _selected_source(dataset, regime, mode, method, seed))
    setting = baseline_table[selected.key(dataset, regime, mode, method)]
    return selected.config(dataset, regime, mode, method, T, seed,
                           setting["selected_client_scale"], setting["selected_server_eta"],
                           _selected_source(dataset, regime, mode, method, seed))


def _validate_selected_runs(joint_table, baseline_table, datasets=DATASETS):
    """Validate the selected-method runs without copying their trajectories."""
    count = 0
    for dataset in datasets:
        for regime in study.REGIMES:
            for mode in study.MODES:
                seeds = ((study.EXACT_SEED,) if mode == "full"
                         else study.BENCHMARK_STOCHASTIC_SEEDS)
                for method in ("adafed", "fedadagrad", "fedlada", "fafed"):
                    for seed in seeds:
                        src = _selected_source(dataset, regime, mode, method, seed)
                        summary = json.loads((src / "run.json").read_text())
                        expected = _selected_config(dataset, regime, mode, method, seed,
                                                    joint_table, baseline_table)
                        if method in ("adafed", "fedadagrad"):
                            expected = fl.resolve(expected)
                        if summary["status"] != "done" or summary["final_round"] != T or summary["config"] != expected:
                            raise ValueError(f"stale selected-method result: {src}")
                        count += 1
    print(f"validated {count} selected-method runs in place", flush=True)


def run(jobs=4, datasets=None):
    """Tune and run the adaptation-level comparison on `datasets` (default: the two benchmark datasets), whose
    selected-method runs (rate_compare_10k) must be complete."""
    datasets = DATASETS if datasets is None else tuple(datasets)
    table = tune(jobs, datasets)
    joint_table = joint.tune(datasets=datasets)
    baseline_table = selected.tune(jobs, datasets=datasets)
    _validate_selected_runs(joint_table, baseline_table, datasets)
    configs = []
    for dataset in datasets:
        for regime in study.REGIMES:
            for mode in study.MODES:
                seeds = ((study.EXACT_SEED,) if mode == "full"
                         else study.BENCHMARK_STOCHASTIC_SEEDS)
                for method in NEW_METHODS:
                    rec = table[key(dataset, regime, mode, method)]
                    a, b = (rec["selected"][name] for name in PARAMS[method])
                    for seed in seeds:
                        out = RESULTS / dataset / regime / mode / method / f"seed{seed}"
                        cfg = config(dataset, regime, mode, method, T, seed, a, b, out)
                        marker = out / "run.json"
                        expected = fl.resolve(cfg) if method == "fedadam" else cfg
                        if marker.exists():
                            old = json.loads(marker.read_text())
                            if old["status"] == "done" and old["final_round"] == T and old["config"] == expected:
                                continue
                        configs.append(cfg)
    print(f"{len(configs)} new full runs pending", flush=True)
    if configs:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, result in enumerate(pool.map(_run_one, configs), 1):
                print(f"full run {i}/{len(configs)}: {result['config']['algorithm']} "
                      f"{result['config']['dataset']} {result['config']['mode']} "
                      f"{result['status']}", flush=True)


def _path(dataset, regime, mode, method, seed):
    if method in selected.METHODS:
        return _selected_source(dataset, regime, mode, method, seed)
    return RESULTS / dataset / regime / mode / method / f"seed{seed}"


def _series(dataset, regime, mode, method, column):
    seeds = ((study.EXACT_SEED,) if mode == "full"
             else study.BENCHMARK_STOCHASTIC_SEEDS)
    rows = [study.read_metrics(_path(dataset, regime, mode, method, s) / "metrics.csv")
            for s in seeds]
    x = rows[0]["round"]
    if any(not np.array_equal(x, r["round"]) for r in rows):
        raise ValueError("checkpoint mismatch")
    if len(x) != 1001 or x[-1] != T:
        raise ValueError("incomplete trajectory")
    return x, np.stack([r[column] for r in rows])


def plot():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    FIGURES.mkdir(parents=True, exist_ok=True)
    names = {"adafed": "AdaFed", "fedadagrad": "FedAdagrad", "fedadam": "FedAdam",
             "fedur": "FedUR", "local_adaalter": "Local AdaAlter",
             "fedsps": "FedSPS", "fedlada": "FedLADA", "fafed": "FAFED",
             "fedada2": r"FedAda$^2$"}
    colors = {"adafed": "#222222", "fedadagrad": "#D55E00", "fedadam": "#009E73",
              "fedur": "#CC79A7", "local_adaalter": "#D55E00",
              "fedsps": "#009E73", "fedlada": "#0072B2", "fafed": "#D55E00",
              "fedada2": "#0072B2"}
    for group, methods in GROUPS.items():
        if any(not _path(DATASETS[0], study.PANEL_REGIMES[0], "full", m, 2027).exists()
               for m in methods):
            print(f"skipping {group} figure: method results missing", flush=True)
            continue
        for dataset in DATASETS:
            fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5), sharex=True)
            for row, mode in enumerate(study.MODES):
                for col, regime in enumerate(study.PANEL_REGIMES):
                    ax = axes[row, col]
                    metric = study.regime_cfg(regime)["metric"]
                    for method in methods:
                        x, average = _series(dataset, regime, mode, method, "mean_" + metric)
                        _, last = _series(dataset, regime, mode, method, metric)
                        ax.loglog(x, np.maximum(np.mean(average, axis=0), 1e-16),
                                  color=colors[method], lw=2)
                        ax.loglog(x, np.maximum(np.mean(last, axis=0), 1e-16),
                                  color=colors[method], lw=1, ls="--", alpha=.5)
                    ax.grid(True, which="major", color=".85", lw=.6)
                    ax.grid(False, which="minor")
                    if row == 0:
                        ax.set_title(regime.replace("-", " ").title())
                    if row == 1:
                        ax.set_xlabel("Communication rounds $t$")
                    ax.set_ylabel(r"$\|\nabla F(x)\|^2$" if regime == "smooth-nonconvex"
                                  else r"$F(x)-F^\star$")
            fig.legend(handles=[Line2D([0], [0], color=colors[m], lw=2, label=names[m])
                                for m in methods], ncol=len(methods), loc="upper center",
                       bbox_to_anchor=(.5, .965), frameon=False)
            fig.text(.008, .665, "Exact local gradients", rotation=90, va="center", fontsize=10)
            fig.text(.008, .255, f"Stochastic local gradients\n(mean of {len(study.BENCHMARK_STOCHASTIC_SEEDS)} seeds)",
                     rotation=90, va="center", fontsize=10)
            fig.text(.5, .035, "Solid: uniform-random-output mean; dashed: last iterate",
                     ha="center", fontsize=9)
            fig.suptitle(f"{group.title()} adaptation, K=10, T={T:,} — "
                         f"{study._dataset_label(dataset)}", fontsize=13, y=1.005)
            fig.tight_layout(rect=(.04, .055, 1, .9), w_pad=1.2)
            for suffix in ("pdf", "png"):
                fig.savefig(FIGURES / f"{group}-{dataset}.{suffix}",
                            dpi=220, bbox_inches="tight")
            plt.close(fig)


def report_methods():
    """The methods the report covers: every method of every group, once."""
    return tuple(dict.fromkeys(m for group in GROUPS.values() for m in group))


def mode_seeds(mode):
    """The seeds a mode runs: the exact seed once, or all benchmark stochastic seeds."""
    return (study.EXACT_SEED,) if mode == "full" else study.BENCHMARK_STOCHASTIC_SEEDS


def report_plan(datasets=DATASETS):
    """Every (dataset, regime, mode, method, seed) the report reads, in report order."""
    return [(dataset, regime, mode, method, seed)
            for dataset in datasets
            for regime in study.PANEL_REGIMES
            for mode in study.MODES
            for method in report_methods()
            for seed in mode_seeds(mode)]


def expected_report_runs(datasets=DATASETS):
    """Runs the report expects, from the methods, problems, oracles and seed tuples it uses."""
    return (len(report_methods()) * len(datasets) * len(study.PANEL_REGIMES)
            * sum(len(mode_seeds(mode)) for mode in study.MODES))


def report(split="default"):
    """Validate every run of the comparison on the datasets of `split` and write summary.csv and group-summary.csv
    (summary-kmeans.csv and group-summary-kmeans.csv for the k-means split)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    datasets = study.split_datasets(DATASETS, split)
    suffix = study.split_suffix(split)
    method_set = report_methods()
    columns = ("dataset", "regime", "mode", "method", "seed", "status", "final_round",
               "final_gap", "final_grad_norm_sq", "final_mean_gap", "final_mean_grad_norm_sq", "wall_sec")
    with (ROOT / f"summary{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        count = 0
        for dataset in datasets:
            for regime in study.PANEL_REGIMES:
                for mode in study.MODES:
                    seeds = ((study.EXACT_SEED,) if mode == "full"
                             else study.BENCHMARK_STOCHASTIC_SEEDS)
                    for method in method_set:
                        for seed in seeds:
                            path = _path(dataset, regime, mode, method, seed) / "run.json"
                            result = json.loads(path.read_text())
                            cfg = result["config"]
                            if (result["status"] != "done" or result["final_round"] != T
                                    or cfg["T"] != T or cfg["K"] != selected.K
                                    or cfg["p"] != 1.0 or cfg["seed"] != seed
                                    or cfg["algorithm"] != method or cfg["dataset"] != dataset
                                    or cfg["mode"] != mode
                                    or cfg["penalty"] != study.regime_cfg(regime)["penalty"]):
                                raise ValueError(f"invalid run: {path}")
                            metrics = study.read_metrics(path.with_name("metrics.csv"))
                            if (not np.array_equal(metrics["round"], fl.eval_rounds(T, 10))
                                    or not all(np.isfinite(v).all() for v in metrics.values())):
                                raise ValueError(f"invalid metrics: {path}")
                            writer.writerow({"dataset": dataset, "regime": regime, "mode": mode,
                                             "method": method, "seed": seed,
                                             **{k: result.get(k) for k in columns[5:]}})
                            count += 1
    if count != expected_report_runs(datasets):
        raise ValueError(f"expected {expected_report_runs(datasets)} full runs, found {count}")
    group_rows = []
    for group, methods in GROUPS.items():
        for dataset in datasets:
            for regime in study.PANEL_REGIMES:
                metric = study.regime_cfg(regime)["metric"]
                for mode in study.MODES:
                    cell = []
                    for method in methods:
                        _, average = _series(dataset, regime, mode, method, "mean_" + metric)
                        _, last = _series(dataset, regime, mode, method, metric)
                        cell.append({"group": group, "dataset": dataset, "regime": regime,
                                     "mode": mode, "method": method, "n_seeds": len(average),
                                     "random_output_mean": float(np.mean(average[:, -1])),
                                     "last_mean": float(np.mean(last[:, -1]))})
                    for rank, row in enumerate(sorted(cell, key=lambda r: r["random_output_mean"]), 1):
                        row["rank_mean"] = rank
                    group_rows.extend(cell)
    with (ROOT / f"group-summary{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=group_rows[0].keys())
        writer.writeheader()
        writer.writerows(group_rows)
    print(f"validated {count} full runs and wrote both summaries", flush=True)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tune", "run", "plot", "report", "all"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command in ("tune", "run", "all"):
        tune(args.jobs) if args.command == "tune" else run(args.jobs)
    if args.command in ("plot", "all"):
        plot()
    if args.command in ("report", "all"):
        report()


if __name__ == "__main__":
    cli()
