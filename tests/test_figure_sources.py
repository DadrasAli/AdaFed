"""The Dirichlet figures: tracked sources, the Figure 2 band, the benchmark exporter, the CLI's split handling."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from adafed import cli, generate, rate_study
from tools import export_benchmark_pgfplots as bench, export_rate_study_pgfplots as rates, figure_sources


class FigureSourceTests(unittest.TestCase):
    def test_tracked_dirichlet_sources_are_the_rendering_of_the_default_ones(self):
        for name in figure_sources.FIGURES:
            for alpha in generate.DIRICHLET_ALPHAS:
                path = figure_sources.source_path(name, figure_sources.split_name(alpha))
                self.assertEqual(path.read_text(), figure_sources.render(name, alpha), path.name)

    def test_tracked_kmeans_sources_are_the_rendering_of_the_default_ones(self):
        for name in figure_sources.FIGURES:
            path = figure_sources.source_path(name, "kmeans")
            self.assertEqual(path.read_text(), figure_sources.render_kmeans(name), path.name)
            tables = re.findall(r"experiments-data/([^}]+\.dat)", path.read_text())
            self.assertTrue(tables)
            self.assertTrue(all("ijcnn1-32-kmeans-" in t for t in tables))
        self.assertEqual(figure_sources.source_path("rates-ijcnn-b1", "kmeans").name, "rates-ijcnn-b1-kmeans.tex")
        self.assertEqual(figure_sources.source_path("selected-ijcnn1-32", "kmeans").name, "selected-ijcnn1-32-kmeans.tex")

    def test_kmeans_sources_differ_from_the_dirichlet_ones_only_in_prefix_and_note(self):
        for name in figure_sources.FIGURES:
            kmeans = figure_sources.render_kmeans(name).splitlines()
            dirichlet = figure_sources.render(name, 0.1).replace("ijcnn1-32-dirichlet0.1", "ijcnn1-32-kmeans").splitlines()
            self.assertEqual(len(kmeans), len(dirichlet))
            self.assertEqual([k for k, d in zip(kmeans, dirichlet) if k != d and not k.startswith("%")], [])

    def test_dirichlet_source_names(self):
        self.assertEqual(figure_sources.source_path("rates-ijcnn-b1", "dirichlet0.1").name,
                         "rates-ijcnn-b1-dirichlet0.1.tex")
        self.assertEqual(figure_sources.source_path("selected-ijcnn1-32", "dirichlet1").name,
                         "selected-ijcnn1-32-dirichlet1.tex")
        self.assertEqual(figure_sources.source_path("selected-ijcnn1-32").name, "selected-ijcnn1-32.tex")

    def test_dirichlet_sources_read_only_their_own_tables(self):
        for name in figure_sources.FIGURES:
            for alpha in generate.DIRICHLET_ALPHAS:
                text = figure_sources.render(name, alpha)
                tables = re.findall(r"experiments-data/([^}]+\.dat)", text)
                self.assertTrue(tables)
                self.assertTrue(all("dirichlet" + f"{alpha:g}" + "-" in t for t in tables))
                self.assertIn("\\input{../pdf-metadata.tex}", text)

    def test_figure_two_draws_the_figure_one_band(self):
        text = figure_sources.source_path("selected-ijcnn1-32").read_text()
        self.assertIn("fillbetween", text)
        self.assertEqual(len(re.findall(r"\\MeanBand\{", text)), 12)       # 3 stochastic panels x 4 methods
        self.assertEqual(len(re.findall(r"\\LastBand\{", text)), 12)
        self.assertIn("fill opacity=0.22", text)
        self.assertIn("fill opacity=0.17", text)
        # the band tables it reads are those the benchmark exporter writes
        for regime in ("smooth-nonconvex", "smooth-convex", "nonsmooth-convex"):
            for method in ("adafed", "fedadagrad", "fedlada", "fafed"):
                self.assertIn(f"selected-ijcnn1-32-stochastic-{regime}-{method}-band.dat", text)
        # the exact panels have no band
        exact_panel = text.split("\\nextgroupplot")[1]
        self.assertNotIn("MeanBand", exact_panel)


class SeedTests(unittest.TestCase):
    def test_twenty_seeds_everywhere(self):
        seeds = tuple(range(2027, 2047))
        self.assertEqual(rate_study.PAPER_STOCHASTIC_SEEDS, seeds)
        self.assertEqual(rate_study.BENCHMARK_STOCHASTIC_SEEDS, seeds)
        self.assertEqual(rate_study.PAPER_ABLATION_SEEDS, seeds)
        self.assertEqual(rates.SEEDS, seeds)
        self.assertEqual(bench.SEEDS, {"full": (2027,), "stochastic": seeds})

    def test_exporter_band_is_the_library_band(self):
        values = np.random.default_rng(1).random((20, 7)) ** 3
        mean, low, high = rates.directional_band(values)
        m, lo, hi = rate_study._directional_std_band(values)
        np.testing.assert_array_equal(mean, m)
        np.testing.assert_array_equal(low, lo)
        np.testing.assert_array_equal(high, hi)


class BenchmarkExportTests(unittest.TestCase):
    def test_export_writes_means_and_bands_for_a_dirichlet_split(self):
        dataset = generate.dirichlet_name(0.5)
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            results, out = Path(tmp) / "results", Path(tmp) / "out"
            truth = {}
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
                                cfg["b"] = 32
                            (run / "run.json").write_text(json.dumps({"status": "done", "final_round": 10000, "config": cfg}))
                            random, last = rng.random(2) + 0.1, rng.random(2) + 0.1
                            truth.setdefault((regime, mode, method), []).append((random, last))
                            (run / "metrics.csv").write_text(
                                f"round,{metric},mean_{metric}\n1,{last[0]!r},{random[0]!r}\n10000,{last[1]!r},{random[1]!r}\n")
            with mock.patch.object(bench, "SELECTED_RESULTS", results):
                bench.export(out, groups=("selected",), datasets=(dataset,))
            tables = sorted(p.name for p in out.glob("*.dat"))
            self.assertEqual(len([t for t in tables if not t.endswith("-band.dat")]), 6)       # 3 regimes x 2 modes
            self.assertEqual(len([t for t in tables if t.endswith("-band.dat")]), 12)          # 3 regimes x 4 methods
            self.assertFalse(any("full" in t and t.endswith("-band.dat") for t in tables))
            regime, method = "smooth-convex", "adafed"
            random = np.stack([r for r, _ in truth[(regime, "stochastic", method)]])
            last = np.stack([l for _, l in truth[(regime, "stochastic", method)]])
            band = (out / f"selected-{dataset}-stochastic-{regime}-{method}-band.dat").read_text().split("\n")
            self.assertEqual(band[0], "round random_low random_high last_low last_high")
            _, lo, hi = rate_study._directional_std_band(random)
            _, llo, lhi = rate_study._directional_std_band(last)
            row = band[-2].split()                                                             # round 10000
            np.testing.assert_allclose([float(v) for v in row[1:]], [lo[-1], hi[-1], llo[-1], lhi[-1]], rtol=1e-15)
            mean_table = (out / f"selected-{dataset}-stochastic-{regime}.dat").read_text().split("\n")
            self.assertEqual(mean_table[0].split()[:3], ["round", "adafed_random", "adafed_last"])
            self.assertAlmostEqual(float(mean_table[-2].split()[1]), random.mean(axis=0)[-1], places=15)


class CliTests(unittest.TestCase):
    def test_splits(self):
        self.assertEqual(cli.SPLITS, ("default", "dirichlet0.01", "dirichlet0.1", "dirichlet0.5", "dirichlet1", "kmeans"))
        self.assertEqual(cli.dataset_of("default"), "ijcnn1-32")
        self.assertEqual(cli.dataset_of("kmeans"), "ijcnn1-32-kmeans")
        self.assertEqual(cli.dataset_of("dirichlet0.1"), "ijcnn1-32-dirichlet0.1")
        self.assertEqual(cli.dataset_of("dirichlet1"), "ijcnn1-32-dirichlet1")
        with self.assertRaises(ValueError):
            cli.dataset_of("dirichlet2")

    def test_commands_take_a_split(self):
        parser = cli._parser()
        for command in ("prepare", "rates", "benchmarks", "export", "figures"):
            self.assertEqual(parser.parse_args([command, "--split", "kmeans"]).split, "kmeans")
            self.assertEqual(parser.parse_args([command, "--split", "dirichlet0.5"]).split, "dirichlet0.5")
            self.assertEqual(parser.parse_args([command]).split, "default")
        args = parser.parse_args(["figure", "selected-ijcnn1-32", "--split", "dirichlet1", "--jobs", "3"])
        self.assertEqual((args.name, args.split, args.jobs), ("selected-ijcnn1-32", "dirichlet1", 3))
        args = parser.parse_args(["figure", "all", "--split", "all"])
        self.assertEqual((args.name, args.split), ("all", "all"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["rates", "--split", "all"])           # `all` splits only for `figure`
        with self.assertRaises(SystemExit):
            parser.parse_args(["rates", "--split", "dirichlet2"])

    def test_figure_all_runs_every_figure_of_every_split(self):
        seen = []
        with mock.patch.object(cli, "figure", side_effect=lambda n, s, j, f: seen.append((n, s))), \
             mock.patch.object(cli, "compile_figures", side_effect=lambda s, only=None: seen.append(("compile", s, only))):
            cli.main(["figure", "all", "--split", "all", "--jobs", "1"])
        built = [x for x in seen if x[0] != "compile"]
        self.assertEqual(len(built), 2 * len(cli.SPLITS))
        self.assertEqual(len(built), 12)
        self.assertEqual({s for _, s in built}, set(cli.SPLITS))
        self.assertEqual({n for n, _ in built}, set(cli.FIGURES))
        self.assertEqual(len([x for x in seen if x[0] == "compile"]), 12)

    def test_default_figures_do_not_compile_the_dirichlet_sources(self):
        names = {p.name for p in cli._sources("default")}
        self.assertIn("rates-ijcnn-b1.tex", names)
        self.assertFalse(any("dirichlet" in n or "kmeans" in n for n in names))
        self.assertEqual([p.name for p in cli._sources("default", only=("selected-ijcnn1-32",))],
                         ["selected-ijcnn1-32.tex"])
        self.assertEqual({p.name for p in cli._sources("dirichlet0.1")},
                         {"rates-ijcnn-b1-dirichlet0.1.tex", "selected-ijcnn1-32-dirichlet0.1.tex"})
        self.assertEqual({p.name for p in cli._sources("kmeans")},                       # the k-means split: every figure
                         {f"{name}-kmeans.tex" for name in figure_sources.PAPER_FIGURES})
        self.assertEqual([p.name for p in cli._sources("kmeans", only=("rates-ijcnn-b1",))], ["rates-ijcnn-b1-kmeans.tex"])

    def test_all_includes_every_other_split(self):
        calls = []
        with mock.patch.object(cli, "rates", side_effect=lambda j, f=False, s="default": calls.append(("rates", s))), \
             mock.patch.object(cli, "benchmarks", side_effect=lambda j, s="default": calls.append(("benchmarks", s))), \
             mock.patch.object(cli, "export", side_effect=lambda s="default", only=None: calls.append(("export", s))), \
             mock.patch.object(cli, "compile_figures", side_effect=lambda s="default", only=None: calls.append(("compile", s))):
            cli.main(["all", "--jobs", "1"])
        for kind in ("rates", "benchmarks", "export", "compile"):
            self.assertEqual([s for k, s in calls if k == kind], list(cli.SPLITS) if kind in ("export", "compile")
                             else list(cli.SPLITS))
        self.assertEqual(calls[:2], [("rates", "default"), ("benchmarks", "default")])


if __name__ == "__main__":
    unittest.main()
