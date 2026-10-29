"""The k-means split of the whole pipeline: the synthetic k-means partitions (tracked, locked, built without rerunning
k-means, F* equal to the default split's), the rate and ablation plans and the benchmark plans of the k-means split (the
default plans on the k-means datasets), the CLI's handling of `--split kmeans`, the exporters and the k-means figure
sources (prefix-only renderings, and the fixed-eta sources rendered from the runs, which reproduce the default sources byte
for byte when given the default etas).
"""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from adafed import (cli, data, generate, protocol, rate_compare_10k, rate_compare_joint as joint,
                    rate_compare_levels_10k as levels, rate_compare_selected as selected, rate_study)
from tools import export_benchmark_pgfplots as bench, export_rate_study_pgfplots as rate_export, figure_sources

SYNTHETIC = sorted(generate.SYNTHETIC)
KMEANS_TUNE = tuple(generate.kmeans_name(d) for d in rate_study.TUNE_DATASETS)
DEFAULT_FIXED_ETAS = {"rates-synthetic-b1": (10, 10, 1, 1, 1, 1), "rates-synthetic-b8": (10, 10, 1, 10, 1, 1)}


def _info(doc):
    return {"restarts": [{"zeta2": z, "iterations": k} for z, k in zip(doc["restart_zeta2"], doc["restart_iterations"])],
            "best": doc["kept_restart"]}


class SyntheticPartitionTests(unittest.TestCase):
    def test_the_tracked_files(self):
        for base in SYNTHETIC:
            name = generate.kmeans_name(base)
            part, doc = generate.read_kmeans_partition(base=base)
            self.assertEqual(part.shape, (32, 64), base)
            self.assertTrue(np.array_equal(np.sort(part.ravel()), np.arange(2048)))
            self.assertTrue((np.diff(part, axis=1) > 0).all())                  # ascending within a client
            self.assertEqual(doc["dataset"], name)
            self.assertEqual((doc["lambda"], doc["restarts"], doc["split_seed"]), (0.01, 10, 2021))
            self.assertEqual(doc["kept_restart"], int(np.argmax(doc["restart_zeta2"])))
            self.assertEqual(doc["zeta2"], max(doc["restart_zeta2"]))
            self.assertEqual(generate.partition_digest(part), doc["sha256"])
            self.assertEqual(doc["sha256"], protocol.FROZEN["kmeans"]["synthetic"][name]["partition_sha256"])
            self.assertEqual(generate.kmeans_partition_path(base).name, f"{name}-partition.json")

    def test_the_text_round_trips(self):
        for base in SYNTHETIC:
            part, doc = generate.read_kmeans_partition(base=base)
            self.assertEqual(generate.kmeans_partition_text(part, _info(doc), base),
                             generate.kmeans_partition_path(base).read_text())

    def test_a_damaged_file_is_refused(self):
        part, doc = generate.read_kmeans_partition(base="synthetic-kappa10")
        with tempfile.TemporaryDirectory() as tmp:
            for what in ("digest", "row", "shape"):
                bad = json.loads(json.dumps(doc))
                if what == "digest":
                    bad["partition"][0][0], bad["partition"][1][0] = bad["partition"][1][0], bad["partition"][0][0]
                elif what == "row":
                    bad["partition"][0][0] = bad["partition"][0][1]
                else:
                    bad["partition"][0].pop()
                path = Path(tmp) / f"{what}.json"
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError, msg=what):
                    generate.read_kmeans_partition(path, base="synthetic-kappa10")
            with self.assertRaises(ValueError):                                 # an IJCNN1-shaped check refuses it too
                generate.read_kmeans_partition(generate.kmeans_partition_path("synthetic-kappa10"))

    def test_a_damaged_file_fails_the_protocol(self):
        _, doc = generate.read_kmeans_partition(base="synthetic-kappa100")
        doc["partition"][0][0], doc["partition"][1][0] = doc["partition"][1][0], doc["partition"][0][0]
        original = generate.kmeans_partition_path
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps(doc))
            with mock.patch.object(generate, "kmeans_partition_path",
                                   side_effect=lambda base: bad if base == "synthetic-kappa100" else original(base)):
                with self.assertRaises(ValueError):
                    protocol.validate()

    def test_the_build_uses_the_tracked_partition_without_rerunning_kmeans(self):
        for base in SYNTHETIC:
            name = generate.kmeans_name(base)
            with mock.patch.object(generate, "kmeans_split_info", side_effect=AssertionError("k-means was rerun")), \
                 mock.patch.object(generate, "balanced_kmeans", side_effect=AssertionError("k-means was rerun")):
                X, y, part, meta = generate.build(name, data.RAW, log=lambda s: None)
            A, labels, _, _ = generate.synthetic_design(base)
            tracked, doc = generate.read_kmeans_partition(base=base)
            self.assertEqual(X.shape, (32, 64, 32))
            self.assertTrue(np.array_equal(part, tracked))
            self.assertTrue(np.array_equal(X, A.numpy()[part]))
            self.assertTrue(np.array_equal(y, labels.numpy()[part]))
            self.assertEqual(meta["kmeans"]["sha256"], doc["sha256"])
            self.assertEqual(meta["kind"], "synthetic-kmeans")

    def test_the_default_synthetic_split_is_unchanged(self):
        """The default split is still the design dealt in blocks by the planted margin."""
        for base in SYNTHETIC:
            s = generate.SYNTHETIC[base]
            X, y, part, meta = generate.build(base, data.RAW, log=lambda s: None)
            A, labels, margin, _ = generate.synthetic_design(base)
            self.assertTrue(np.array_equal(part, generate.deal(margin, s["clients"], s["rows_per_client"], s["block"]).numpy()))
            self.assertTrue(np.array_equal(X, A.numpy()[part]))
            self.assertEqual(meta["kind"], "synthetic")

    def test_zeta2_at_xstar_is_the_recorded_one(self):
        for base in SYNTHETIC:
            A, y = generate.kmeans_rows(base, data.RAW)
            G, _ = generate.kmeans_gradients(A, y)
            part, doc = generate.read_kmeans_partition(base=base)
            z = float(((G[part].mean(1) - G.mean(0)) ** 2).sum(1).mean())
            self.assertAlmostEqual(z, doc["zeta2"], delta=1e-12 * doc["zeta2"])

    def test_fstar_entries_equal_the_default_splits(self):
        """The objective is the mean over all rows, so F* does not depend on the split."""
        table = json.loads(data.FSTAR.read_text())
        for base in SYNTHETIC:
            for pen in ("l2sq", "l1"):
                entry = table[f"{generate.kmeans_name(base)}/{pen}/0.01"]
                self.assertLess(abs(entry["f_star"] - table[f"{base}/{pen}/0.01"]["f_star"]), 1e-12, (base, pen))
                self.assertLess(entry["kkt_residual"], 1e-11)
                self.assertEqual(entry["f_star"], protocol.FROZEN["kmeans"]["reference_optima"][f"{generate.kmeans_name(base)}/{pen}/0.01"])


class NamingTests(unittest.TestCase):
    def test_split_names(self):
        self.assertEqual(generate.split_dataset("synthetic-kappa1", "default"), "synthetic-kappa1")
        self.assertEqual(generate.split_dataset("synthetic-kappa1", "kmeans"), "synthetic-kappa1-kmeans")
        self.assertEqual(generate.split_dataset("ijcnn1-32", "kmeans"), generate.KMEANS_NAME)
        self.assertEqual(generate.split_dataset("ijcnn1-32", "dirichlet0.1"), "ijcnn1-32-dirichlet0.1")
        with self.assertRaises(ValueError):
            generate.split_dataset("synthetic-kappa1", "dirichlet0.1")       # Dirichlet splits are IJCNN1's only
        with self.assertRaises(ValueError):
            generate.kmeans_name("ijcnn1-32-kmeans")
        self.assertEqual(generate.kmeans_base("synthetic-kappa10-kmeans"), "synthetic-kappa10")
        self.assertIsNone(generate.kmeans_base("synthetic-kappa10"))
        self.assertIsNone(generate.kmeans_base("ijcnn1-32-dirichlet0.1"))
        self.assertEqual(cli.dataset_of("kmeans", "synthetic-kappa1"), "synthetic-kappa1-kmeans")
        self.assertEqual(cli.dataset_of("kmeans"), "ijcnn1-32-kmeans")

    def test_protocol_settings_come_from_the_base(self):
        self.assertEqual(rate_study.base_dataset("synthetic-kappa100-kmeans"), "synthetic-kappa100")
        self.assertEqual(rate_study.base_dataset("ijcnn1-32-kmeans"), "ijcnn1-32")
        self.assertEqual(rate_study.paper_batches("synthetic-kappa10-kmeans"), (1, 8))
        self.assertEqual(rate_study.paper_batches("ijcnn1-32-kmeans"), (1, 32))
        self.assertEqual(rate_study.tuning_source("synthetic-kappa1-kmeans"), "synthetic-kappa10-kmeans")
        self.assertEqual(rate_study.tuning_source("synthetic-kappa1"), "synthetic-kappa10")
        self.assertEqual(rate_study.tuning_source("ijcnn1-32-kmeans"), "ijcnn1-32-kmeans")
        self.assertEqual(rate_study.split_suffix("kmeans"), "-kmeans")
        self.assertEqual(rate_study.split_suffix("default"), "")


def _tables(etas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Tuning records for the default and the k-means tuning datasets: default eta 1, k-means eta 3."""
    full, stochastic = {}, {}
    for dataset in rate_study.TUNE_DATASETS + KMEANS_TUNE:
        for regime in rate_study.REGIMES:
            for mode, table in (("full", full), ("stochastic", stochastic)):
                table[f"{dataset}/{regime}/{mode}"] = {
                    "selected_eta": 3.0 if dataset in KMEANS_TUNE else 1.0, "candidates": [{"eta": e} for e in etas]}
    return full, stochastic


def _plan(runs, split, etas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """(configs, the datasets tune was asked for, the datasets tune_batch was asked for) of one study, no run executed;
    `etas` are the tuning candidates of every record."""
    full, stochastic = _tables(etas)
    captured, tuned, batch_tuned = [], [], []
    with mock.patch.object(rate_study, "tune", side_effect=lambda **kw: tuned.append(kw.get("datasets")) or full), \
         mock.patch.object(rate_study, "tune_batch",
                           side_effect=lambda batch, **kw: batch_tuned.append(kw.get("datasets")) or stochastic), \
         mock.patch.object(rate_study, "_run_parallel", side_effect=lambda configs, jobs: captured.extend(configs)):
        runs(jobs=1, split=split)
    return captured, tuned, batch_tuned


class RatePlanTests(unittest.TestCase):
    STUDIES = (rate_study.main_runs, rate_study.eta_runs, rate_study.gamma_runs, rate_study.partial_runs,
               rate_study.batch1_runs, rate_study.batch8_runs, rate_study.batch32_runs)

    def test_kmeans_plans_are_the_default_plans_on_the_kmeans_datasets(self):
        grid = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)          # holds both eta figures' values: no eta is added
        for runs in self.STUDIES:
            default, _, _ = _plan(runs, "default", grid)
            kmeans, _, _ = _plan(runs, "kmeans", grid)
            self.assertTrue(default, runs.__name__)
            expected = []
            for cfg in default:
                name = generate.kmeans_name(cfg["dataset"])
                cfg = dict(cfg, dataset=name, out=cfg["out"].replace(f"/{cfg['dataset']}/", f"/{name}/"))
                if runs is not rate_study.eta_runs:           # the eta ablation runs the candidates, not the tuned eta
                    cfg["eta"] = 3.0
                expected.append(cfg)
            self.assertEqual(kmeans, expected, runs.__name__)

    def test_kmeans_tunes_only_the_kmeans_datasets(self):
        for runs in self.STUDIES:
            _, tuned, batch_tuned = _plan(runs, "kmeans")
            self.assertTrue(all(d == KMEANS_TUNE for d in tuned), runs.__name__)
            self.assertTrue(all(d is not None and set(d) <= set(KMEANS_TUNE) for d in batch_tuned), runs.__name__)
            _, tuned, batch_tuned = _plan(runs, "default")            # the default split: the calls it made before
            self.assertTrue(all(d == rate_study.TUNE_DATASETS for d in tuned), runs.__name__)
            self.assertTrue(all(d is None for d in batch_tuned), runs.__name__)

    def test_kmeans_eta_ablation_runs_the_figure_values_its_grid_lacks(self):
        """A k-means grid of 0.01..100 also runs IJCNN1's figure value 1000 (exact once, stochastic every seed); the
        synthetic figure's values are all in it; the default split runs exactly its candidates."""
        default, _, _ = _plan(rate_study.eta_runs, "default")
        kmeans, _, _ = _plan(rate_study.eta_runs, "kmeans")
        self.assertEqual({c["eta"] for c in default}, {0.01, 0.1, 1.0, 10.0, 100.0})
        extra = [c for c in kmeans if c["eta"] == 1000.0]
        self.assertEqual(len(kmeans), len(default) + len(extra))
        self.assertEqual({c["dataset"] for c in extra}, {"ijcnn1-32-kmeans"})
        self.assertEqual(len(extra), len(rate_study.REGIMES) * (1 + len(rate_study.PAPER_ABLATION_SEEDS)))
        self.assertTrue(all(c["out"].endswith(f"/eta1000/seed{c['seed']}") and c["K"] == 10 and c["gamma"] == 1.0
                            for c in extra))
        self.assertEqual([c for c in kmeans if c["eta"] != 1000.0],
                         [dict(c, dataset=generate.kmeans_name(c["dataset"]),
                               out=c["out"].replace(f"/{c['dataset']}/", f"/{generate.kmeans_name(c['dataset'])}/"))
                          for c in default])

    def test_figure_etas_are_the_exporters(self):
        for dataset in ("synthetic-kappa10", "ijcnn1-32", "synthetic-kappa10-kmeans", "ijcnn1-32-kmeans"):
            self.assertEqual(rate_study.figure_etas(dataset), tuple(float(e) for e in rate_export.eta_values(dataset)))

    def test_eta_ablation_values_are_the_planned_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_path, b1_path = Path(tmp) / "tuning.json", Path(tmp) / "tuning-b1.json"
            full, stochastic = _tables(etas=(100.0, 0.1, 10.0, 1.0, 1000.0, 10.0))
            full_path.write_text(json.dumps(full))
            b1_path.write_text(json.dumps(stochastic))
            with mock.patch.object(rate_study, "TUNING", full_path), \
                 mock.patch.dict(rate_study.BATCH_TUNING, {1: b1_path}):
                for mode in rate_study.MODES:
                    self.assertEqual(rate_study.eta_ablation_values("ijcnn1-32-kmeans", "smooth-convex", mode),
                                     (0.1, 1.0, 10.0, 100.0, 1000.0))
            full, stochastic = _tables()                                         # a grid of 0.01..100
            full_path.write_text(json.dumps(full))
            b1_path.write_text(json.dumps(stochastic))
            with mock.patch.object(rate_study, "TUNING", full_path), \
                 mock.patch.dict(rate_study.BATCH_TUNING, {1: b1_path}):
                for mode in rate_study.MODES:
                    self.assertEqual(rate_study.eta_ablation_values("ijcnn1-32-kmeans", "smooth-convex", mode),
                                     (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))
                    self.assertEqual(rate_study.eta_ablation_values("ijcnn1-32", "smooth-convex", mode),
                                     (0.01, 0.1, 1.0, 10.0, 100.0))

    def test_missing_eta_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_path, b1_path, results = Path(tmp) / "tuning.json", Path(tmp) / "tuning-b1.json", Path(tmp) / "results"
            full, stochastic = _tables()
            full_path.write_text(json.dumps(full))
            b1_path.write_text(json.dumps(stochastic))
            dataset, regime, figure = "ijcnn1-32-kmeans", "smooth-convex", (0.1, 1, 10, 100, 1000)
            for mode in rate_study.MODES:
                seeds = (rate_study.EXACT_SEED,) if mode == "full" else rate_study.PAPER_ABLATION_SEEDS
                for eta in figure:
                    for seed in seeds:
                        run = results / "eta-paper" / dataset / regime / mode / f"eta{eta:g}" / f"seed{seed}"
                        run.mkdir(parents=True)
                        (run / "run.json").write_text("{}")
            with mock.patch.object(rate_study, "TUNING", full_path), \
                 mock.patch.dict(rate_study.BATCH_TUNING, {1: b1_path}), \
                 mock.patch.object(rate_study, "RESULTS", results):
                for mode in rate_study.MODES:
                    self.assertEqual(rate_study.missing_eta_runs(dataset, regime, mode, figure), [])
                (results / "eta-paper" / dataset / regime / "stochastic" / "eta1000" / "seed2040" / "run.json").unlink()
                self.assertEqual(rate_study.missing_eta_runs(dataset, regime, "stochastic", figure), [1000])
                self.assertEqual(rate_study.missing_eta_runs(dataset, regime, "full", figure), [])
                self.assertEqual(rate_study.missing_eta_runs("ijcnn1-32", regime, "full", figure), [0.1, 1, 10, 100, 1000])

    def test_fixed_etas_are_read_from_the_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            dataset = "synthetic-kappa10-kmeans"
            etas = dict(zip([(m, r) for m in rate_study.MODES for r in rate_study.PANEL_REGIMES], (10., 1., .1, 1., 100., 1.)))
            for (mode, regime), eta in etas.items():
                for K in rate_study.KS:
                    seeds = (rate_study.EXACT_SEED,) if mode == "full" else rate_study.PAPER_STOCHASTIC_SEEDS
                    for seed in seeds:
                        run = (results / "main" / dataset / regime / mode if mode == "full" else
                               results / "batch8" / dataset / regime) / f"K{K}" / f"seed{seed}"
                        run.mkdir(parents=True)
                        (run / "run.json").write_text(json.dumps({"config": {"eta": eta}}))
            with mock.patch.object(rate_study, "RESULTS", results):
                self.assertEqual(rate_study.fixed_etas(dataset, 8), (10., 1., .1, 1., 100., 1.))
                odd = results / "batch8" / dataset / "smooth-convex" / "K100" / "seed2030" / "run.json"
                odd.write_text(json.dumps({"config": {"eta": 7.0}}))
                with self.assertRaises(ValueError):
                    rate_study.fixed_etas(dataset, 8)


class CliTests(unittest.TestCase):
    MATRIX = ("main_runs", "eta_runs", "batch1_runs", "batch8_runs", "batch32_runs", "gamma_runs", "partial_runs")

    def _rates(self, split):
        calls = []
        patches = [mock.patch.object(cli, "prepare", side_effect=lambda s: calls.append(("prepare", s))),
                   mock.patch.object(cli, "figure", side_effect=lambda *a, **k: calls.append(("figure",) + a)),
                   mock.patch.object(rate_study, "tune", side_effect=lambda **k: calls.append(("tune", k["datasets"]))),
                   mock.patch.object(rate_study, "ijcnn_t10000_runs",
                                     side_effect=lambda jobs, dataset: calls.append(("t10000", dataset))),
                   mock.patch.object(rate_study, "report", side_effect=lambda s: calls.append(("report", s)))]
        patches += [mock.patch.object(rate_study, name, side_effect=lambda jobs, s, name=name: calls.append((name, s)))
                    for name in self.MATRIX]
        for p in patches:
            p.start()
        try:
            cli.rates(3, False, split)
        finally:
            for p in patches:
                p.stop()
        return calls

    def test_rates_of_the_kmeans_split_run_the_whole_matrix(self):
        self.assertEqual(self._rates("kmeans"), [("prepare", "kmeans"), ("tune", KMEANS_TUNE)]
                         + [(name, "kmeans") for name in self.MATRIX]
                         + [("t10000", "ijcnn1-32-kmeans"), ("report", "kmeans")])
        self.assertEqual(self._rates("default"), [("prepare", "default"), ("tune", rate_study.TUNE_DATASETS)]
                         + [(name, "default") for name in self.MATRIX]
                         + [("t10000", "ijcnn1-32"), ("report", "default")])
        self.assertEqual(self._rates("dirichlet0.1"), [("figure", "rates-ijcnn-b1", "dirichlet0.1", 3, False)])

    def test_benchmarks_of_the_kmeans_split_run_every_group(self):
        for split, datasets in (("kmeans", ("synthetic-kappa1-kmeans", "ijcnn1-32-kmeans")), ("default", None)):
            calls = []
            with mock.patch.object(cli, "prepare", side_effect=lambda s: calls.append(("prepare", s))), \
                 mock.patch.object(rate_compare_10k, "run", side_effect=lambda j, datasets: calls.append(("selected", datasets))), \
                 mock.patch.object(rate_compare_10k, "report", side_effect=lambda s: calls.append(("selected report", s))), \
                 mock.patch.object(levels, "run", side_effect=lambda j, datasets: calls.append(("levels", datasets))), \
                 mock.patch.object(levels, "report", side_effect=lambda s: calls.append(("levels report", s))):
                cli.benchmarks(3, split)
            self.assertEqual(calls, [("prepare", split), ("selected", datasets), ("selected report", split),
                                     ("levels", datasets), ("levels report", split)])

    def test_prepare_builds_every_kmeans_dataset(self):
        with mock.patch.object(protocol, "validate"), mock.patch.object(rate_study, "prepare") as prepare:
            cli.prepare("kmeans")
        prepare.assert_called_once_with(tuple(generate.kmeans_name(d) for d in rate_study.DATASETS))

    def test_export_of_the_kmeans_split(self):
        with mock.patch.object(protocol, "validate"), mock.patch.object(cli, "check_eta_ablation") as check, \
             mock.patch.object(rate_export, "export") as rates, mock.patch.object(bench, "export") as benchmarks, \
             mock.patch.object(rate_study, "fixed_etas", side_effect=lambda d, b: (d, b)), \
             mock.patch.object(figure_sources, "write_fixed_eta_sources") as write:
            cli.export("kmeans")
        check.assert_called_once_with(("synthetic-kappa10-kmeans", "ijcnn1-32-kmeans"))
        rates.assert_called_once_with(datasets=("synthetic-kappa10-kmeans", "ijcnn1-32-kmeans"))
        benchmarks.assert_called_once_with(datasets=("synthetic-kappa1-kmeans", "ijcnn1-32-kmeans"))
        write.assert_called_once_with({"rates-synthetic-b1": ("synthetic-kappa10-kmeans", 1),
                                       "rates-synthetic-b8": ("synthetic-kappa10-kmeans", 8)})

    def test_the_eta_figure_values_must_have_run(self):
        datasets = ("synthetic-kappa10-kmeans", "ijcnn1-32-kmeans")
        asked = []
        with mock.patch.object(rate_study, "missing_eta_runs", side_effect=lambda d, r, m, e: asked.append((d, e)) or []):
            cli.check_eta_ablation(datasets)
        self.assertEqual({(d, tuple(e)) for d, e in asked}, {("synthetic-kappa10-kmeans", (0.01, 0.1, 1, 10, 100)),
                                                             ("ijcnn1-32-kmeans", (0.1, 1, 10, 100, 1000))})
        self.assertEqual(len(asked), 2 * len(rate_study.REGIMES) * len(rate_study.MODES))
        with mock.patch.object(rate_study, "missing_eta_runs",
                               side_effect=lambda d, r, m, e: [1000] if d == "ijcnn1-32-kmeans" else []):
            with self.assertRaisesRegex(RuntimeError, "ijcnn1-32-kmeans .*1000"):
                cli.check_eta_ablation(datasets)


class BenchmarkTests(unittest.TestCase):
    def test_configs_take_the_base_batch(self):
        self.assertEqual(joint.config("synthetic-kappa1-kmeans", "smooth-convex", "stochastic", 12, 7, 1.0, 1.0)["b"], 16)
        self.assertEqual(selected.config("ijcnn1-32-kmeans", "smooth-convex", "stochastic", "fedadagrad", 12, 7, 1.0)["b"], 32)
        self.assertEqual(levels.config("synthetic-kappa1-kmeans", "smooth-convex", "stochastic", "fedadam", 12, 7, 1., 1.)["b"], 16)
        self.assertEqual(bench.batch_of("synthetic-kappa1-kmeans"), 16)
        self.assertEqual(bench.batch_of("ijcnn1-32-kmeans"), 32)
        self.assertEqual(bench.batch_of("synthetic-kappa1"), 16)

    def test_levels_report_counts(self):
        kmeans = rate_study.split_datasets(levels.DATASETS, "kmeans")
        self.assertEqual(levels.expected_report_runs(kmeans), len(levels.report_plan(kmeans)))
        self.assertEqual(levels.expected_report_runs(kmeans), levels.expected_report_runs())

    def test_levels_run_tunes_only_its_datasets(self):
        kmeans = rate_study.split_datasets(levels.DATASETS, "kmeans")
        calls = []

        class Stop(Exception):
            pass

        with mock.patch.object(levels, "tune", side_effect=lambda jobs, datasets: calls.append(("levels", datasets)) or {}), \
             mock.patch.object(joint, "tune", side_effect=lambda **k: calls.append(("joint", k["datasets"])) or {}), \
             mock.patch.object(selected, "tune", side_effect=lambda jobs, **k: calls.append(("selected", k["datasets"])) or {}), \
             mock.patch.object(levels, "_validate_selected_runs", side_effect=lambda a, b, datasets: calls.append(("validate", datasets)) or (_ for _ in ()).throw(Stop())):
            with self.assertRaises(Stop):
                levels.run(2, datasets=kmeans)
        self.assertEqual(calls, [("levels", kmeans), ("joint", kmeans), ("selected", kmeans), ("validate", kmeans)])

    def test_benchmark_export_of_a_synthetic_kmeans_split(self):
        dataset = "synthetic-kappa1-kmeans"
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            results, out = Path(tmp) / "results", Path(tmp) / "out"
            for regime in bench.REGIMES:
                metric = bench.METRIC[regime]
                for mode in ("full", "stochastic"):
                    for method in sorted(bench.SELECTED_METHODS):
                        for seed in bench.SEEDS[mode]:
                            run = results / dataset / regime / mode / method / f"seed{seed}"
                            run.mkdir(parents=True)
                            cfg = {"dataset": dataset, "mode": mode, "algorithm": method, "seed": seed,
                                   "T": 10000, "K": 10, "p": 1.0}
                            if mode == "stochastic":
                                cfg["b"] = 16
                            (run / "run.json").write_text(json.dumps({"status": "done", "final_round": 10000, "config": cfg}))
                            a, b = rng.random(2) + 0.1
                            (run / "metrics.csv").write_text(f"round,{metric},mean_{metric}\n1,{a!r},{b!r}\n10000,{a!r},{b!r}\n")
            with mock.patch.object(bench, "SELECTED_RESULTS", results):
                paths = bench.export(out, groups=("selected",), datasets=(dataset,))
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(p.name.startswith(f"selected-{dataset}-") for p in out.glob("*.dat")))
            self.assertEqual(len(list(out.glob("*-band.dat"))), 12)


class RateExportTests(unittest.TestCase):
    def test_kmeans_tables_read_kmeans_runs_with_the_default_batches_and_values(self):
        read = []

        def aggregate(paths, regime):
            read.extend(paths)
            values = np.array([[0.5, 0.25]] * len(paths))
            return np.array([1, 10]), values.mean(0), values.mean(0), values, values

        datasets = ("synthetic-kappa10-kmeans", "ijcnn1-32-kmeans")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rate_export, "aggregate", side_effect=aggregate):
            rate_export.export(Path(tmp), datasets=datasets)
            written = {p.name for p in Path(tmp).glob("*.dat")}
        self.assertTrue(all(any(f"/{d}/" in str(p) for d in datasets) for p in read))
        self.assertTrue(all(name.startswith(datasets) for name in written))
        self.assertIn("synthetic-kappa10-kmeans-b8-smooth-convex-K100.dat", written)
        self.assertIn("ijcnn1-32-kmeans-b32-smooth-convex-K100.dat", written)
        self.assertIn("ijcnn1-32-kmeans-b1-smooth-convex-K100-band.dat", written)            # the 10,000-round panel
        self.assertIn("synthetic-kappa10-kmeans-eta-full-smooth-convex-0.01.dat", written)
        self.assertIn("ijcnn1-32-kmeans-eta-stochastic-smooth-convex-1000.dat", written)
        self.assertIn("ijcnn1-32-kmeans-gamma-full-nonsmooth-convex-100.dat", written)
        self.assertIn("synthetic-kappa10-kmeans-partial-stochastic-smooth-nonconvex-0.25.dat", written)
        self.assertTrue(any("/ijcnn-t10000/ijcnn1-32-kmeans/" in str(p) for p in read))
        self.assertEqual(rate_export.eta_values("ijcnn1-32-kmeans"), (0.1, 1, 10, 100, 1000))
        self.assertEqual(rate_export.eta_values("synthetic-kappa10-kmeans"), (0.01, 0.1, 1, 10, 100))


class KmeansFigureSourceTests(unittest.TestCase):
    def test_every_paper_figure_has_a_kmeans_source(self):
        self.assertEqual(set(figure_sources.PAPER_FIGURES), {p.stem for p in cli._sources("default")})

    def test_tracked_kmeans_sources_are_the_rendering_of_the_default_ones(self):
        for name in figure_sources.PAPER_FIGURES:
            if name in figure_sources.FIXED_ETA_FIGURES:
                continue
            path = figure_sources.source_path(name, "kmeans")
            self.assertEqual(path.read_text(), figure_sources.render_kmeans(name), path.name)

    def test_kmeans_sources_read_only_kmeans_tables(self):
        for name, (_, base, _) in figure_sources.PAPER_FIGURES.items():
            text = figure_sources.render_kmeans(name, DEFAULT_FIXED_ETAS.get(name))
            refs = re.findall(r"experiments-data/([^}]+)", text)
            self.assertTrue(refs, name)
            self.assertTrue(all(f"{generate.kmeans_name(base)}-" in r for r in refs), name)
            self.assertIn("\\input{../pdf-metadata.tex}", text)

    def test_fixed_eta_rendering_reproduces_the_default_sources(self):
        for name, etas in DEFAULT_FIXED_ETAS.items():
            path = figure_sources.source_path(name)
            self.assertEqual(figure_sources.render_with(name, fixed=etas), path.read_text(), name)    # byte for byte

    def test_fixed_eta_kmeans_sources_differ_only_in_tables_note_labels_and_bands(self):
        etas = (100, 0.1, 1000, 0.01, 1e10, 0.001)                               # each differs from the default's
        labels = ["10^2", "10^{-1}", "10^3", "10^{-2}", "10^{10}", "10^{-3}"]
        for name in figure_sources.FIXED_ETA_FIGURES:
            text = figure_sources.render_kmeans(name, etas)
            # the band additions (tested in test_kmeans_bands) taken out
            text = re.sub(r"  \\(?:MeanBand|LastBand)\{[^\n]*\n", "", text)
            text = text.replace(figure_sources.BAND_MACROS.search(text).group(0) + "\n", "", 1)
            text = text.replace(figure_sources.BAND_NOTE, "", 1).replace("groupplots,fillbetween", "groupplots", 1)
            kmeans = text.splitlines()
            default = figure_sources.source_path(name).read_text().splitlines()
            self.assertEqual(len(kmeans), len(default) + 3)                      # the three note lines
            changed = [k for k, d in zip(kmeans[5:], default[2:]) if k != d]
            self.assertEqual([c for c in changed if "experiments-data/" not in c],
                             [f"    at (rel axis cs:0.03,0.035) {{fixed $\\eta={label}$}};" for label in labels])

    def test_the_fixed_eta_sources_need_their_etas(self):
        with self.assertRaises(ValueError):
            figure_sources.render_kmeans("rates-synthetic-b1")
        with self.assertRaises(ValueError):
            figure_sources.render_kmeans("eta-synthetic", (1, 1, 1, 1, 1, 1))
        with self.assertRaises(ValueError):
            figure_sources.render_kmeans("rates-synthetic-b8", (1, 1, 1))

    def test_tex_power(self):
        self.assertEqual([figure_sources.tex_power(v) for v in (0.01, 0.1, 1, 10, 100, 1000, 1e10)],
                         ["10^{-2}", "10^{-1}", "1", "10", "10^2", "10^3", "10^{10}"])
        with self.assertRaises(ValueError):
            figure_sources.tex_power(3.0)

    def test_write_fixed_eta_sources(self):
        """The export writes them beside the default sources (here a copy of the folder)."""
        etas = {"rates-synthetic-b1": (1, 10, 100, 0.1, 1, 10), "rates-synthetic-b8": (10, 10, 1, 1, 0.01, 1)}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "figures" / "rates"
            folder.mkdir(parents=True)
            for name in figure_sources.FIXED_ETA_FIGURES + (figure_sources.BAND_SOURCE,):     # Figure 1: the band macros
                (folder / f"{name}.tex").write_text(figure_sources.source_path(name).read_text())
            with mock.patch.object(figure_sources, "ROOT", Path(tmp)):
                figure_sources.write_fixed_eta_sources(etas)
                for name in figure_sources.FIXED_ETA_FIGURES:
                    self.assertEqual((folder / f"{name}-kmeans.tex").read_text(), figure_sources.render_kmeans(name, etas[name]))


if __name__ == "__main__":
    unittest.main()
