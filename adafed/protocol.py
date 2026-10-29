"""Machine-checkable lock for the experiment protocol used in the paper.

The numerical modules remain the executable source of truth.  This module is a
deliberately redundant guard: it fails loudly if a future refactor silently
changes a paper-defining constant, dataset partition, aggregation rule, or
benchmark group.
"""

from __future__ import annotations

import inspect
from copy import deepcopy

from . import (adaptive_baselines, extra_adaptive_baselines, fl, generate,
               rate_compare_10k, rate_compare_levels_10k,
               rate_compare_selected, rate_study)
from tools import export_benchmark_pgfplots, export_rate_study_pgfplots


FROZEN = {
    "methods": [
        "adafed", "fedadagrad", "fedadam", "fedlada", "fafed",
        "local_adaalter", "fedsps", "fedada2", "fedur",
    ],
    "datasets": {
        "synthetic-kappa1": {"clients": 32, "examples_per_client": 64,
                             "dimension": 32, "condition": 1.0, "rounds": 2000},
        "synthetic-kappa10": {"clients": 32, "examples_per_client": 64,
                              "dimension": 32, "condition": 10.0, "rounds": 2000},
        "synthetic-kappa100": {"clients": 32, "examples_per_client": 64,
                               "dimension": 32, "condition": 100.0, "rounds": 2000},
        "ijcnn1-32": {"clients": 32, "examples_per_client": 1000,
                      "dimension": 22, "rounds": 1000, "long_rounds": 10000,
                      "partition_block": 50, "exact_margin_key": False,
                      "partition_sha256": "0d6c70541218a2e22e1b996c3b5d33828b6496dec0fd5b9831b101819c92bb5c"},
    },
    "objectives": {
        "smooth-nonconvex": {"penalty": "nonconvex", "lambda": 0.01,
                             "metric": "grad_norm_sq"},
        "smooth-convex": {"penalty": "l2sq", "lambda": 0.01, "metric": "gap"},
        "nonsmooth-convex": {"penalty": "l1", "lambda": 0.01, "metric": "gap"},
    },
    "rates": {
        "K": [1, 10, 100],
        "exact_seed": 2027,
        "stochastic_seeds": list(range(2027, 2047)),
        "primary_batch": 1,
        "appendix_batch": {"synthetic": 8, "real": 32},
        "gamma": [0.01, 0.1, 1.0, 10.0, 100.0],
        "aggregation": "pointwise arithmetic mean",
        "error_band": "directional one-sided RMS standard deviation",
        "measurement": "pre-update iterate",
    },
    "benchmarks": {
        "datasets": ["synthetic-kappa1", "ijcnn1-32"],
        "K": 10,
        "rounds": 10000,
        "stochastic_seeds": list(range(2027, 2047)),
        "stochastic_batch": {"synthetic-kappa1": 16, "ijcnn1-32": 32},
        "groups": {
            "server": ["adafed", "fedadagrad", "fedadam", "fedur"],
            "client": ["adafed", "local_adaalter", "fedsps", "fedlada"],
            "both": ["adafed", "fafed", "fedada2"],
            "selected": ["adafed", "fedadagrad", "fedlada", "fafed"],
        },
    },
    "dirichlet": {
        "base": "ijcnn1-32", "alphas": [0.01, 0.1, 0.5, 1.0], "split_seed": 2021, "clients": 32,
        "datasets": ["ijcnn1-32-dirichlet0.01", "ijcnn1-32-dirichlet0.1", "ijcnn1-32-dirichlet0.5",
                     "ijcnn1-32-dirichlet1"],
        "reference_optima": {
            "ijcnn1-32-dirichlet0.01/l1/0.01": 0.4278836450209717,
            "ijcnn1-32-dirichlet0.01/l2sq/0.01": 0.4154526395889211,
            "ijcnn1-32-dirichlet0.1/l1/0.01": 0.4278836450209717,
            "ijcnn1-32-dirichlet0.1/l2sq/0.01": 0.41545263958892104,
            "ijcnn1-32-dirichlet0.5/l1/0.01": 0.4278836450209717,
            "ijcnn1-32-dirichlet0.5/l2sq/0.01": 0.41545263958892104,
            "ijcnn1-32-dirichlet1/l1/0.01": 0.4278836450209717,
            "ijcnn1-32-dirichlet1/l2sq/0.01": 0.41545263958892115,
        },
    },
    "kmeans": {
        "base": "ijcnn1-32", "dataset": "ijcnn1-32-kmeans", "clients": 32, "rows_per_client": 1000,
        "lambda": 0.01, "restarts": 10, "split_seed": 2021, "kept_restart": 1,
        "partition_sha256": "b3ac05b09eecebaeb4fb4edf48c38a3bb8a47b3fe426f6192c5502b8434398e3",
        "reference_optima": {
            "ijcnn1-32-kmeans/l1/0.01": 0.4278836450209717,
            "ijcnn1-32-kmeans/l2sq/0.01": 0.4154526395889211,
            "synthetic-kappa1-kmeans/l1/0.01": 0.4551244564739533,
            "synthetic-kappa1-kmeans/l2sq/0.01": 0.4409334313437286,
            "synthetic-kappa10-kmeans/l1/0.01": 0.47382480072297695,
            "synthetic-kappa10-kmeans/l2sq/0.01": 0.45935734908765125,
            "synthetic-kappa100-kmeans/l1/0.01": 0.48743008220058726,
            "synthetic-kappa100-kmeans/l2sq/0.01": 0.47750977596974997,
        },
        # The k-means splits of the synthetic datasets: the same construction (lambda, restarts, split seed as above), 32
        # clients of 64 rows; their tracked partitions are data/<dataset>-partition.json.
        "synthetic": {
            "synthetic-kappa1-kmeans": {
                "base": "synthetic-kappa1", "clients": 32, "rows_per_client": 64, "kept_restart": 0,
                "partition_sha256": "154fc565712695b3d8a2c564dd8e9c12f583217e3b5976b85a860e805e3390eb"},
            "synthetic-kappa10-kmeans": {
                "base": "synthetic-kappa10", "clients": 32, "rows_per_client": 64, "kept_restart": 7,
                "partition_sha256": "6f279e0ce66d310cba82475e5996c9c59242be785dffcd0913eba26f39b65adb"},
            "synthetic-kappa100-kmeans": {
                "base": "synthetic-kappa100", "clients": 32, "rows_per_client": 64, "kept_restart": 0,
                "partition_sha256": "883bf906416b45f77cc42c452e2ae95c2e23de528cc3a06830b9a569e4eb6625"},
        },
    },
    "reference_optima": {
        "ijcnn1-32/l1/0.01": 0.42788364502097165,
        "ijcnn1-32/l2sq/0.01": 0.4154526395889211,
        "synthetic-kappa1/l1/0.01": 0.45512445647395333,
        "synthetic-kappa1/l2sq/0.01": 0.4409334313437286,
        "synthetic-kappa10/l1/0.01": 0.4738248007229769,
        "synthetic-kappa10/l2sq/0.01": 0.4593573490876512,
        "synthetic-kappa100/l1/0.01": 0.48743008220058726,
        "synthetic-kappa100/l2sq/0.01": 0.47750977596975,
    },
}


def snapshot():
    """Return the protocol currently encoded by the executable modules."""
    synthetic = {}
    for name in ("synthetic-kappa1", "synthetic-kappa10", "synthetic-kappa100"):
        spec = generate.SYNTHETIC[name]
        synthetic[name] = {
            "clients": spec["clients"], "examples_per_client": spec["rows_per_client"],
            "dimension": spec["dim"], "condition": spec["condition"],
            "rounds": rate_study.DATASETS[name]["T"],
        }
    real = generate.LIBSVM["ijcnn1-32"]
    tracked = generate.read_kmeans_partition()[1]        # raises unless the tracked partition is intact
    default = generate.read_default_partition()[1]       # likewise the default split's
    synthetic_kmeans = {}
    for base in sorted(generate.SYNTHETIC):              # likewise every synthetic k-means split's
        doc = generate.read_kmeans_partition(base=base)[1]
        if (doc["lambda"], doc["restarts"], doc["split_seed"]) != (generate.KMEANS_LAMBDA, generate.KMEANS_RESTARTS,
                                                                   generate.SPLIT_SEED):
            raise AssertionError(f"the k-means partition of {base} was built with other settings")
        clients, rows_per_client = generate.split_shape(base)
        synthetic_kmeans[generate.kmeans_name(base)] = {
            "base": base, "clients": clients, "rows_per_client": rows_per_client,
            "kept_restart": doc["kept_restart"], "partition_sha256": doc["sha256"]}
    datasets = {
        **synthetic,
        "ijcnn1-32": {
            "clients": real["clients"], "examples_per_client": real["rows_per_client"],
            "dimension": real["dim"], "rounds": rate_study.DATASETS["ijcnn1-32"]["T"],
            "long_rounds": rate_study.IJCNN_LONG_T, "partition_block": real["block"],
            "exact_margin_key": real["exact_key"], "partition_sha256": default["sha256"],
        },
    }
    objectives = {
        name: {"penalty": rate_study.REGIMES[name]["penalty"],
               "lambda": rate_study.REGIMES[name]["lam"],
               "metric": rate_study.REGIMES[name]["metric"]}
        for name in rate_study.PANEL_REGIMES
    }
    groups = {name: list(methods) for name, methods in rate_compare_levels_10k.GROUPS.items()}
    groups["selected"] = list(rate_compare_selected.METHODS)
    return {
        "methods": list(fl.ALGORITHMS + adaptive_baselines.METHODS
                        + extra_adaptive_baselines.METHODS),
        "datasets": datasets,
        "objectives": objectives,
        "rates": {
            "K": list(rate_study.KS), "exact_seed": rate_study.EXACT_SEED,
            "stochastic_seeds": list(rate_study.PAPER_STOCHASTIC_SEEDS),
            "primary_batch": rate_study.PRIMARY_BATCH,
            "appendix_batch": deepcopy(rate_study.APPENDIX_BATCH_BY_KIND),
            "gamma": list(rate_study.GAMMAS),
            "aggregation": "pointwise arithmetic mean",
            "error_band": "directional one-sided RMS standard deviation",
            "measurement": "pre-update iterate",
        },
        "benchmarks": {
            "datasets": list(rate_compare_selected.DATASETS),
            "K": rate_compare_selected.K, "rounds": rate_compare_10k.T,
            "stochastic_seeds": list(rate_study.BENCHMARK_STOCHASTIC_SEEDS),
            "stochastic_batch": deepcopy(rate_study.BENCHMARK_BATCH),
            "groups": groups,
        },
        "dirichlet": {
            "base": generate.DIRICHLET_BASE, "alphas": list(generate.DIRICHLET_ALPHAS),
            "split_seed": generate.SPLIT_SEED, "clients": real["clients"],
            "datasets": [generate.dirichlet_name(a) for a in generate.DIRICHLET_ALPHAS],
            "reference_optima": deepcopy(FROZEN["dirichlet"]["reference_optima"]),
        },
        "kmeans": {
            "base": generate.DIRICHLET_BASE, "dataset": generate.KMEANS_NAME, "clients": real["clients"],
            "rows_per_client": real["rows_per_client"], "lambda": generate.KMEANS_LAMBDA,
            "restarts": generate.KMEANS_RESTARTS, "split_seed": generate.SPLIT_SEED,
            "kept_restart": tracked["kept_restart"], "partition_sha256": tracked["sha256"],
            "reference_optima": deepcopy(FROZEN["kmeans"]["reference_optima"]),
            "synthetic": synthetic_kmeans,
        },
        "reference_optima": deepcopy(FROZEN["reference_optima"]),
    }


def validate():
    """Raise ``AssertionError`` if code and the frozen paper protocol differ."""
    current = snapshot()
    if current != FROZEN:
        raise AssertionError(f"paper protocol drifted\nexpected={FROZEN!r}\ncurrent={current!r}")
    if export_rate_study_pgfplots.SEEDS != rate_study.PAPER_STOCHASTIC_SEEDS:
        raise AssertionError("rate exporter seed set differs from the run protocol")
    if set(generate.DATASETS) != set(FROZEN["datasets"]):
        raise AssertionError("an undocumented dataset was added to the release")
    if rate_study.data.FSTAR.exists():
        import json
        references = json.loads(rate_study.data.FSTAR.read_text())
        found = {key: references[key]["f_star"] for key in FROZEN["reference_optima"]}
        if found != FROZEN["reference_optima"]:
            raise AssertionError("data/fstar.json differs from the frozen paper references")
        dirichlet = FROZEN["dirichlet"]["reference_optima"]
        if any(key in references and references[key]["f_star"] != value for key, value in dirichlet.items()):
            raise AssertionError("data/fstar.json differs from the frozen Dirichlet references")
        kmeans = FROZEN["kmeans"]["reference_optima"]
        if any(key in references and references[key]["f_star"] != value for key, value in kmeans.items()):
            raise AssertionError("data/fstar.json differs from the frozen k-means references")
    expected_benchmark_seeds = {
        "full": (rate_study.EXACT_SEED,),
        "stochastic": rate_study.BENCHMARK_STOCHASTIC_SEEDS,
    }
    if export_benchmark_pgfplots.SEEDS != expected_benchmark_seeds:
        raise AssertionError("benchmark exporter seed set differs from the run protocol")
    if export_benchmark_pgfplots.BATCH != rate_study.BENCHMARK_BATCH:
        raise AssertionError("benchmark exporter batch sizes differ from the run protocol")
    for module in (rate_study, export_rate_study_pgfplots, export_benchmark_pgfplots):
        if "np.median" in inspect.getsource(module):
            raise AssertionError(f"median aggregation reintroduced in {module.__name__}")
    return current
