"""Repeat the jointly tuned selected comparison for 10,000 rounds.

Usage: python -m adafed.rate_compare_10k all --jobs 4
Output: rate-study/comparison-selected-joint-10000/
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import (adaptive_baselines, fl, rate_compare_joint as joint,
               rate_compare_selected as selected, rate_study as study)


T = 10_000
ROOT = study.ROOT / "comparison-selected-joint-10000"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def config(dataset, regime, mode, method, seed, joint_table, baseline_table):
    out = RESULTS / dataset / regime / mode / method / f"seed{seed}"
    if method == "adafed":
        setting = joint_table[joint.key(dataset, regime, mode)]
        return joint.config(dataset, regime, mode, T, seed,
                            setting["selected_eta"], setting["selected_gamma"], out)
    setting = baseline_table[selected.key(dataset, regime, mode, method)]
    return selected.config(dataset, regime, mode, method, T, seed,
                           setting["selected_client_scale"],
                           setting["selected_server_eta"], out)


def _run_one(cfg):
    if cfg["algorithm"] == "fedadagrad":
        return fl.run(cfg, write=True)
    if cfg["algorithm"] == "adafed":
        return fl.run(cfg, write=True)
    return adaptive_baselines.run(cfg, write=True)


def _saved_config(cfg):
    if cfg["algorithm"] in ("adafed", "fedadagrad"):
        return fl.resolve(cfg)
    return cfg


def run(jobs=4, datasets=None):
    """Tune and run the selected comparison on `datasets` (default: the two benchmark datasets)."""
    datasets = selected.DATASETS if datasets is None else tuple(datasets)
    joint_table = joint.tune(datasets=datasets)
    baseline_table = selected.tune(jobs, datasets=datasets)
    ROOT.mkdir(parents=True, exist_ok=True)
    settings = {"T": T, "K": selected.K,
                "adafed_tuning": {k: {"eta": v["selected_eta"],
                                       "gamma": v["selected_gamma"]}
                                  for k, v in joint_table.items()},
                "baseline_tuning": {k: {"c": v["selected_client_scale"],
                                         "server_eta": v["selected_server_eta"]}
                                    for k, v in baseline_table.items()}}
    (ROOT / "settings.json").write_text(json.dumps(settings, indent=2))
    configs = []
    for dataset in datasets:
        for regime in study.REGIMES:
            for mode in study.MODES:
                seeds = ((study.EXACT_SEED,) if mode == "full"
                         else study.BENCHMARK_STOCHASTIC_SEEDS)
                for method in selected.METHODS:
                    for seed in seeds:
                        configs.append(config(dataset, regime, mode, method, seed,
                                              joint_table, baseline_table))
    pending = []
    for cfg in configs:
        marker = Path(cfg["out"]) / "run.json"
        if not marker.exists():
            pending.append(cfg)
        else:
            saved = json.loads(marker.read_text())
            if saved["status"] != "done" or saved["config"] != _saved_config(cfg):
                pending.append(cfg)
    print(f"{len(configs)-len(pending)} current, {len(pending)} pending", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, summary in enumerate(pool.map(_run_one, pending), 1):
                print(f"completed {i}/{len(pending)}: {summary['config']['out']}",
                      flush=True)


def plot():
    selected.plot(figures=FIGURES, adafed_results=RESULTS,
                  baseline_results=RESULTS, rounds=T)


def report(split="default"):
    """summary.csv of the selected comparison (summary-kmeans.csv for the k-means split)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    selected.report(root=ROOT, adafed_results=RESULTS,
                    baseline_results=RESULTS, datasets=study.split_datasets(selected.DATASETS, split),
                    name=f"summary{study.split_suffix(split)}.csv")


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "plot", "report", "all"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.jobs)
    elif args.command == "plot":
        plot()
    elif args.command == "report":
        report()
    else:
        run(args.jobs)
        plot()
        report()


if __name__ == "__main__":
    cli()
