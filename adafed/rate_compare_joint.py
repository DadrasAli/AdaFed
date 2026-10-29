"""Jointly tune AdaFed eta and gamma for the selected adaptive-FL comparison.

Run from the repository root:
    python -m adafed.rate_compare_joint all --jobs 4

The existing selected-comparison baselines are reused without modification.
New AdaFed runs and figures live in rate-study/comparison-selected-joint/.
"""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import fl, rate_compare_selected as comparison, rate_study as study


ROOT = study.ROOT / "comparison-selected-joint"
TUNING = ROOT / "tuning.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
GAMMAS = (0.01, 0.1, 1., 10., 100.)
ETA_GRIDS = {
    "synthetic-kappa1": (0.01, 0.1, 1., 10., 100.),
    "ijcnn1-32": (0.1, 1., 10., 100., 1000.),
}
BOUNDS = {
    "synthetic-kappa1": {"eta": (0.001, 1000.), "gamma": (0.001, 1000.)},
    "ijcnn1-32": {"eta": (0.01, 10000.), "gamma": (0.001, 1000.)},
}
SELECTION_RULE = "within-0.1-percent-of-minimum-smallest-log-magnitude"


def key(dataset, regime, mode):
    return f"{dataset}/{regime}/{mode}"


def config(dataset, regime, mode, horizon, seed, eta, gamma, out=None):
    cfg = study.base_cfg(dataset, regime, mode, comparison.K, horizon, seed,
                         eta, gamma=gamma, out=out)
    if mode == "stochastic":
        cfg["b"] = study.BENCHMARK_BATCH[study.base_dataset(dataset)]
    return cfg


def _criterion(summary, regime):
    metric = study.regime_cfg(regime)["metric"]
    value = (summary.get("final_mean_grad_norm_sq") if metric == "grad_norm_sq"
             else summary.get("final_mean_gap"))
    return (value if summary["status"] == "done" and value is not None
            and math.isfinite(value) else math.inf)


def _best(records):
    finite = [r for r in records if r["criterion"] is not None]
    if not finite:
        return records[0]
    minimum = min(r["criterion"] for r in finite)
    near = [r for r in finite
            if r["criterion"] <= minimum + 0.001 * max(abs(minimum), 1e-16)]
    return min(near, key=lambda r: (abs(math.log10(r["eta"]))
                                    + abs(math.log10(r["gamma"])),
                                    r["criterion"], r["eta"], r["gamma"]))


def _interior(records, selected):
    etas = sorted({r["eta"] for r in records})
    gammas = sorted({r["gamma"] for r in records})
    return (etas[0] < selected["eta"] < etas[-1]
            and gammas[0] < selected["gamma"] < gammas[-1])


def tune(force=False, datasets=None):
    """Tune AdaFed eta and gamma per (dataset, problem, gradient mode); the pick on a grid edge extends that
    grid by a factor 10 inside BOUNDS (up to 10 passes). `datasets` defaults to the comparison datasets;
    an IJCNN1 Dirichlet split takes ijcnn1-32's grids and bounds."""
    ROOT.mkdir(parents=True, exist_ok=True)
    table = {} if force or not TUNING.exists() else json.loads(TUNING.read_text())
    for dataset in (comparison.DATASETS if datasets is None else datasets):
        grid_key = study.base_dataset(dataset)
        limits = BOUNDS[grid_key]
        for regime in study.REGIMES:
            for mode in study.MODES:
                k = key(dataset, regime, mode)
                if (k in table and not force and table[k].get("search_complete")
                        and table[k].get("selection_rule") == SELECTION_RULE):
                    continue
                T = 300 if dataset.startswith("synthetic") else 150
                records = ([] if force or k not in table else
                           [r for r in table[k]["candidates"]
                            if limits["eta"][0] <= r["eta"] <= limits["eta"][1]
                            and limits["gamma"][0] <= r["gamma"] <= limits["gamma"][1]])
                etas = sorted({r["eta"] for r in records}) if records else list(ETA_GRIDS[grid_key])
                gammas = sorted({r["gamma"] for r in records}) if records else list(GAMMAS)
                for extension in range(11):
                    tested = {(r["eta"], r["gamma"]) for r in records}
                    points = [(eta, gamma) for eta in etas for gamma in gammas
                              if (eta, gamma) not in tested]
                    if points:
                        configs = [config(dataset, regime, mode, T, 1901, eta, gamma)
                                   for eta, gamma in points]
                        summaries = fl.run_batch(configs, write=False, threads=1)
                        for (eta, gamma), summary in zip(points, summaries):
                            value = _criterion(summary, regime)
                            records.append({"eta": eta, "gamma": gamma,
                                            "criterion": value if math.isfinite(value) else None,
                                            "status": summary["status"]})
                    selected = _best(records)
                    if selected["criterion"] is None:
                        raise RuntimeError(f"all AdaFed candidates diverged: {k}")
                    if _interior(records, selected) or extension == 10:
                        break
                    expanded = False
                    for field, values in (("eta", etas), ("gamma", gammas)):
                        low, high = limits[field]
                        if selected[field] == values[0] and values[0] / 10 >= low:
                            values.insert(0, values[0] / 10)
                            expanded = True
                        elif selected[field] == values[-1] and values[-1] * 10 <= high:
                            values.append(values[-1] * 10)
                            expanded = True
                    if not expanded:
                        break
                selected = _best(records)
                table[k] = {"dataset": dataset, "regime": regime, "mode": mode,
                            "K": comparison.K, "T_tuned": T, "seed_tuned": 1901,
                            "criterion": "final uniform-random-output mean",
                            "selection_rule": SELECTION_RULE,
                            "initial_eta_grid": ETA_GRIDS[grid_key],
                            "initial_gamma_grid": GAMMAS, "bounds": limits,
                            "search_complete": True,
                            "interior": _interior(records, selected),
                            "selected_eta": selected["eta"],
                            "selected_gamma": selected["gamma"],
                            "selected_score": selected["criterion"],
                            "candidates": records}
                TUNING.write_text(json.dumps(table, indent=2, allow_nan=False))
                print(f"{k}: eta={selected['eta']:g}, gamma={selected['gamma']:g}, "
                      f"score={selected['criterion']:.3g}, "
                      f"interior={table[k]['interior']}", flush=True)
    return table


def _run_one(cfg):
    return fl.run(cfg, write=True)


def run(jobs=4):
    table = tune()
    configs = []
    for dataset in comparison.DATASETS:
        for regime in study.REGIMES:
            for mode in study.MODES:
                selected = table[key(dataset, regime, mode)]
                seeds = ((study.EXACT_SEED,) if mode == "full"
                         else study.BENCHMARK_STOCHASTIC_SEEDS)
                for seed in seeds:
                    out = RESULTS / dataset / regime / mode / "adafed" / f"seed{seed}"
                    configs.append(config(dataset, regime, mode,
                                          study.DATASETS[dataset]["T"], seed,
                                          selected["selected_eta"],
                                          selected["selected_gamma"], out))
    pending = []
    for cfg in configs:
        marker = Path(cfg["out"]) / "run.json"
        if not marker.exists():
            pending.append(cfg)
        else:
            saved = json.loads(marker.read_text())
            if saved["status"] != "done" or saved["config"] != fl.resolve(cfg):
                pending.append(cfg)
    print(f"{len(configs)-len(pending)} current, {len(pending)} pending", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, summary in enumerate(pool.map(_run_one, pending), 1):
                print(f"completed {i}/{len(pending)}: {summary['config']['out']}",
                      flush=True)


def plot():
    comparison.plot(figures=FIGURES, adafed_results=RESULTS)


def report():
    ROOT.mkdir(parents=True, exist_ok=True)
    comparison.report(root=ROOT, adafed_results=RESULTS)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tune", "run", "plot", "report", "all"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force-tune", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "tune":
        tune(args.force_tune)
    elif args.command == "run":
        run(args.jobs)
    elif args.command == "plot":
        plot()
    elif args.command == "report":
        report()
    else:
        tune(args.force_tune)
        run(args.jobs)
        plot()
        report()


if __name__ == "__main__":
    cli()
