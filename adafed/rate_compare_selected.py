"""AdaFed versus FedAdagrad, FedLADA, and FAFED on two rate-study datasets.

Usage: python -m adafed.rate_compare_selected all --jobs 4
Outputs are isolated under rate-study/comparison-selected/.
"""

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import adaptive_baselines, fl, rate_study as study


ROOT = study.ROOT / "comparison-selected"
TUNING = ROOT / "tuning.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATASETS = ("synthetic-kappa1", "ijcnn1-32")
METHODS = ("adafed", "fedadagrad", "fedlada", "fafed")
K = 10
GRIDS = {
    "fedadagrad": ((1 / 128, 1 / 32, 1 / 8, 1 / 2, 2), (.001, .01, .1, 1., 10.)),
    "fedlada": ((1e-5, 1e-4, 1e-3, 1e-2, 1e-1), (.1, 1., 10.)),
    "fafed": ((1e-5, 1e-4, 1e-3, 1e-2, 1e-1), (1.,)),
}
# Finite search domains prevent FedAdagrad's client/server rescaling from
# sending the client rate to zero and the server rate to infinity.
BOUNDS = {
    "fedadagrad": (1 / 128000, 20., .001, 100.),
    "fedlada": (1e-5, 10., .01, 10.),
    "fafed": (1e-5, .1, 1., 1.),
}


def key(dataset, regime, mode, method):
    return f"{dataset}/{regime}/{mode}/{method}"


def config(dataset, regime, mode, method, horizon, seed, c, server_eta=1., out=None):
    base = study.base_cfg(dataset, regime, mode, K, horizon, seed, 1., out=out)
    if mode == "stochastic":
        base["b"] = study.BENCHMARK_BATCH[study.base_dataset(dataset)]
    base.update(algorithm=method, eta=None, c=c, server_eta=server_eta,
                beta1=.9, beta2=.99, adapt_tau=.001, tau="1")
    if method in adaptive_baselines.METHODS:
        # Their algorithms use an explicit local learning rate, rather than
        # the FedOpt config's dimensionless client scale.
        L = float(study.data.load(dataset)["meta"]["L"])
        base.update(client_lr=c / L, alpha=.1, rho=.01,
                    epsilon_v=1e-8, p=1.)
        if method == "fafed":
            base["implementation"] = "paper-initial-step-v2"
    return base


def _run_one(cfg):
    if cfg["algorithm"] == "fedadagrad":
        return fl.run(cfg, write=cfg["out"] is not None)
    return adaptive_baselines.run(cfg, write=cfg["out"] is not None)


def _criterion(summary, regime):
    value = (summary.get("final_mean_grad_norm_sq") if regime == "smooth-nonconvex"
             else summary.get("final_mean_gap"))
    return value if summary["status"] == "done" and value is not None and math.isfinite(value) else math.inf


def _interior(records, selected, method):
    clients = sorted({r["client_scale"] for r in records})
    servers = sorted({r["server_eta"] for r in records})
    return (clients[0] < selected["client_scale"] < clients[-1]
            and (method == "fafed" or servers[0] < selected["server_eta"] < servers[-1]))


def tune(jobs=4, force=False, datasets=None):
    ROOT.mkdir(parents=True, exist_ok=True)
    table = {} if force or not TUNING.exists() else json.loads(TUNING.read_text())
    for dataset in (DATASETS if datasets is None else datasets):
        for regime in study.REGIMES:
            for mode in study.MODES:
                T = 300 if dataset.startswith("synthetic") else 150
                for method in METHODS[1:]:
                    k = key(dataset, regime, mode, method)
                    if k in table and not force and table[k].get("search_complete"):
                        continue
                    clients, servers = GRIDS[method]
                    min_c, max_c, min_e, max_e = BOUNDS[method]
                    records = ([] if force or k not in table else
                               [r for r in table[k]["candidates"]
                                if min_c <= r["client_scale"] <= max_c
                                and min_e <= r["server_eta"] <= max_e])
                    if records:
                        clients = sorted({r["client_scale"] for r in records})
                        servers = sorted({r["server_eta"] for r in records})
                    for extension in range(9):
                        points = [(c, eta) for c in clients for eta in servers]
                        new = [(c, eta) for c, eta in points if not any(
                            r["client_scale"] == c and r["server_eta"] == eta for r in records)]
                        cfgs = [config(dataset, regime, mode, method, T, 1901, c, eta)
                                for c, eta in new]
                        if not cfgs:
                            summaries = []
                        elif method == "fedadagrad":
                            summaries = fl.run_batch(cfgs, write=False, threads=1)
                        else:
                            with ProcessPoolExecutor(max_workers=jobs) as pool:
                                summaries = list(pool.map(_run_one, cfgs))
                        for (c, eta), summary in zip(new, summaries):
                            val = _criterion(summary, regime)
                            records.append({"client_scale": c, "server_eta": eta,
                                            "criterion": val if math.isfinite(val) else None,
                                            "status": summary["status"]})
                        best = min(records, key=lambda r: math.inf if r["criterion"] is None else r["criterion"])
                        if best["criterion"] is None:
                            raise RuntimeError(f"all tuning candidates diverged: {k}")
                        if _interior(records, best, method):
                            break
                        expanded = False
                        if best["client_scale"] == clients[0]:
                            candidate = clients[0] / 10
                            if candidate >= min_c:
                                clients = sorted([candidate, *clients])
                                expanded = True
                        elif best["client_scale"] == clients[-1]:
                            candidate = clients[-1] * 10
                            if candidate <= max_c:
                                clients = sorted([*clients, candidate])
                                expanded = True
                        if method != "fafed":
                            if best["server_eta"] == servers[0]:
                                candidate = servers[0] / 10
                                if candidate >= min_e:
                                    servers = sorted([candidate, *servers])
                                    expanded = True
                            elif best["server_eta"] == servers[-1]:
                                candidate = servers[-1] * 10
                                if candidate <= max_e:
                                    servers = sorted([*servers, candidate])
                                    expanded = True
                        if not expanded:
                            break
                    selected = min(records, key=lambda r: math.inf if r["criterion"] is None else r["criterion"])
                    table[k] = {"dataset": dataset, "regime": regime, "mode": mode,
                                "method": method, "T_tuned": T, "seed_tuned": 1901,
                                "criterion": "final uniform-random-output mean",
                                "search_complete": True,
                                "bounds": {"client_scale": [min_c, max_c],
                                           "server_eta": [min_e, max_e]},
                                "interior": _interior(records, selected, method),
                                "selected_client_scale": selected["client_scale"],
                                "selected_server_eta": selected["server_eta"],
                                "selected_score": selected["criterion"], "candidates": records}
                    TUNING.write_text(json.dumps(table, indent=2))
                    print(f"{k}: c={selected['client_scale']:g}, "
                          f"server={selected['server_eta']:g}, score={selected['criterion']:.3g}", flush=True)
    return table


def run(jobs=4):
    table = tune(jobs)
    configs = []
    for dataset in DATASETS:
        for regime in study.REGIMES:
            for mode in study.MODES:
                seeds = ((study.EXACT_SEED,) if mode == "full"
                         else study.BENCHMARK_STOCHASTIC_SEEDS)
                for method in METHODS[1:]:
                    setting = table[key(dataset, regime, mode, method)]
                    for seed in seeds:
                        out = RESULTS / dataset / regime / mode / method / f"seed{seed}"
                        configs.append(config(dataset, regime, mode, method,
                                              study.DATASETS[dataset]["T"], seed,
                                              setting["selected_client_scale"],
                                              setting["selected_server_eta"], out))
    pending = []
    for cfg in configs:
        marker = Path(cfg["out"]) / "run.json"
        if not marker.exists():
            pending.append(cfg)
        else:
            saved = json.loads(marker.read_text())
            if saved["status"] != "done" or saved["config"] != cfg:
                pending.append(cfg)
    print(f"{len(configs)-len(pending)} current, {len(pending)} pending", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, summary in enumerate(pool.map(_run_one, pending), 1):
                print(f"completed {i}/{len(pending)}: {summary['config']['out']}", flush=True)


def metrics_path(dataset, regime, mode, method, seed, adafed_results=None,
                 baseline_results=None):
    if method == "adafed":
        if adafed_results is not None:
            return (Path(adafed_results) / dataset / regime / mode / method /
                    f"seed{seed}" / "metrics.csv")
        root = (study.RESULTS / "main" if mode == "full"
                else study.RESULTS / f"batch{study.PRIMARY_BATCH}")
        mode_part = (mode,) if mode == "full" else ()
        return root.joinpath(dataset, regime, *mode_part, f"K{K}",
                             f"seed{seed}", "metrics.csv")
    root = RESULTS if baseline_results is None else Path(baseline_results)
    return root / dataset / regime / mode / method / f"seed{seed}" / "metrics.csv"


def curves(dataset, regime, mode, method, output, adafed_results=None,
           baseline_results=None):
    seeds = ((study.EXACT_SEED,) if mode == "full"
             else study.BENCHMARK_STOCHASTIC_SEEDS)
    col = ("mean_" if output == "random" else "") + study.regime_cfg(regime)["metric"]
    series = [study.read_metrics(metrics_path(dataset, regime, mode, method, seed,
                                             adafed_results, baseline_results))
              for seed in seeds]
    x = series[0]["round"]
    if any(not np.array_equal(s["round"], x) for s in series[1:]):
        raise ValueError("checkpoint rounds differ")
    return x, np.stack([s[col] for s in series])


def plot(figures=FIGURES, adafed_results=None, adafed_label="AdaFed",
         baseline_results=None, rounds=None):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"adafed": "#222222", "fedadagrad": "#D55E00",
              "fedlada": "#0072B2", "fafed": "#009E73"}
    names = {"adafed": adafed_label, "fedadagrad": "FedAdagrad",
             "fedlada": "FedLADA", "fafed": "FAFED"}
    for dataset in DATASETS:
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5), sharex=True)
        for row, mode in enumerate(study.MODES):
            for col, regime in enumerate(study.PANEL_REGIMES):
                ax = axes[row, col]
                for method in METHODS:
                    x, random = curves(dataset, regime, mode, method, "random",
                                       adafed_results, baseline_results)
                    _, last = curves(dataset, regime, mode, method, "last",
                                     adafed_results, baseline_results)
                    ax.loglog(x, np.maximum(np.mean(random, axis=0), 1e-16),
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
        fig.legend(handles=[Line2D([0], [0], color=colors[m], lw=2, label=names[m]) for m in METHODS],
                   ncol=4, loc="upper center", bbox_to_anchor=(.5, .965), frameon=False)
        fig.text(.008, .665, "Exact local gradients", rotation=90, va="center", fontsize=10)
        fig.text(.008, .255, f"Stochastic local gradients\n(mean of {len(study.BENCHMARK_STOCHASTIC_SEEDS)} seeds)",
                 rotation=90, va="center", fontsize=10)
        fig.text(.5, .035, "Solid: uniform-random-output mean; dashed: last iterate",
                 ha="center", fontsize=9)
        horizon = "" if rounds is None else f", T={rounds:,}"
        fig.suptitle(f"Full participation, K=10{horizon} — {study._dataset_label(dataset)}",
                     fontsize=13, y=1.005)
        fig.tight_layout(rect=(.04, .055, 1, .9), w_pad=1.2)
        stem = figures / f"selected-{dataset}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(stem, flush=True)


def report(root=ROOT, adafed_results=None, baseline_results=None, datasets=DATASETS, name="summary.csv"):
    path = Path(root) / name
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("dataset", "regime", "oracle", "method", "last_mean",
                         "uniform_random_output_mean", "seed_count"))
        for dataset in datasets:
            for regime in study.PANEL_REGIMES:
                for mode in study.MODES:
                    for method in METHODS:
                        _, last = curves(dataset, regime, mode, method, "last",
                                         adafed_results, baseline_results)
                        _, random = curves(dataset, regime, mode, method, "random",
                                           adafed_results, baseline_results)
                        writer.writerow((dataset, regime, mode, method,
                                         float(np.mean(last[:, -1])),
                                         float(np.mean(random[:, -1])), len(random)))
    print(path, flush=True)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tune", "run", "plot", "report", "all"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force-tune", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "tune":
        tune(args.jobs, args.force_tune)
    elif args.command == "run":
        run(args.jobs)
    elif args.command == "plot":
        plot()
    elif args.command == "report":
        report()
    else:
        tune(args.jobs, args.force_tune)
        run(args.jobs)
        plot()
        report()


if __name__ == "__main__":
    cli()
