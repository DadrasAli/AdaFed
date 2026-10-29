"""Fast end-to-end checks for AdaFed and every adaptive baseline runner."""

import csv
import json
import tempfile
from pathlib import Path

from adafed import (adaptive_baselines, data, extra_adaptive_baselines, fl,
                    rate_compare_levels_10k as levels,
                    rate_compare_selected as selected, rate_study as study)


METHODS = (
    "adafed", "fedadagrad", "fedadam", "fedur", "local_adaalter",
    "fedsps", "fedlada", "fafed", "fedada2",
)


def config(method, mode, output):
    common = ("synthetic-kappa1", "smooth-convex", mode)
    if method == "adafed":
        cfg = study.base_cfg(*common, K=2, T=3, seed=17, eta=0.1, gamma=1.0, out=output)
    elif method in ("fedadagrad", "fedlada", "fafed"):
        a = {"fedadagrad": 0.1, "fedlada": 0.001, "fafed": 0.001}[method]
        cfg = selected.config(*common, method, 3, 17, a, 0.1 if method == "fedadagrad" else 1.0, output)
    else:
        parameters = {
            "fedadam": (0.1, 0.1),
            "fedur": (0.1, 0.5),
            "local_adaalter": (0.1, 0.1),
            "fedsps": (1.0, 1.0),
            "fedada2": (0.01, 0.1),
        }
        cfg = levels.config(*common, method, 3, 17, *parameters[method], output)
    cfg.update(K=2, T=3, eval_every=1, b=1)
    return cfg


def execute(cfg):
    method = cfg["algorithm"]
    if method in ("adafed", "fedadagrad", "fedadam"):
        return fl.run(cfg, write=True)
    if method in adaptive_baselines.METHODS:
        return adaptive_baselines.run(cfg, write=True)
    return extra_adaptive_baselines.run(cfg, write=True)


def run():
    dataset = data.DATA / "synthetic-kappa1.npz"
    if not dataset.exists():
        data.prepare("synthetic-kappa1")
    data.fstar("synthetic-kappa1", "l2sq", 0.01)
    completed = []
    with tempfile.TemporaryDirectory(prefix="adafed-smoke-") as temporary:
        root = Path(temporary)
        for mode in ("full", "stochastic"):
            for method in METHODS:
                output = root / mode / method
                summary = execute(config(method, mode, output))
                manifest = json.loads((output / "run.json").read_text())
                with (output / "metrics.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if summary["status"] != "done" or manifest["status"] != "done" or len(rows) != 3:
                    raise AssertionError(f"failed smoke run: {mode}/{method}")
                completed.append(f"{mode}/{method}")
    print(f"smoke test passed: {len(completed)} runs ({', '.join(METHODS)})")


if __name__ == "__main__":
    run()
