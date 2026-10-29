"""Command-line interface for the frozen AdaFed paper artifact."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from . import (__version__, data, generate, protocol, rate_compare_10k, rate_compare_levels_10k,
               rate_compare_selected, rate_study)
from tools import export_benchmark_pgfplots, export_rate_study_pgfplots, figure_sources, smoke_test


ROOT = Path(__file__).resolve().parent.parent
PINNED = {"torch": "2.5.1", "numpy": "1.26.4", "matplotlib": "3.10.9"}

# Client splits: the paper's own (`default`), Dirichlet label skew of IJCNN1 for each alpha (Figures 1 and 2 only), and the
# balanced k-means split of every dataset (`kmeans`; the whole pipeline).
DIRICHLET_SPLITS = tuple(figure_sources.split_name(a) for a in generate.DIRICHLET_ALPHAS)
KMEANS_SPLIT = "kmeans"
SPLITS = ("default",) + DIRICHLET_SPLITS + (KMEANS_SPLIT,)
OTHER_SPLITS = SPLITS[1:]
FIGURES = tuple(figure_sources.FIGURES)          # rates-ijcnn-b1 (Figure 1), selected-ijcnn1-32 (Figure 2)


def dataset_of(split, base="ijcnn1-32"):
    """The dataset name of a split: ijcnn1-32, ijcnn1-32-dirichlet<alpha>, or ijcnn1-32-kmeans; with `base` (a paper
    dataset), that dataset under the default or the k-means split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose one of {', '.join(SPLITS)}")
    return generate.split_dataset(base, split)


def default_jobs():
    """Worker processes by default: the CPUs this process may use (all cores of a laptop), at most 8."""
    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:                           # macOS, Windows
        usable = os.cpu_count() or 1
    return max(1, min(usable, 8))


def prepare(split="default"):
    protocol.validate()
    if split == "default":
        rate_study.prepare()
    elif split == KMEANS_SPLIT:
        rate_study.prepare(rate_study.split_datasets(rate_study.DATASETS, split))
    else:
        rate_study.prepare((dataset_of(split),))


def rates(jobs, force_tune=False, split="default"):
    """The rate and ablation matrix of the default or the k-means split; for a Dirichlet split only its Figure 1 (the
    10,000-round IJCNN1 b=1 panel)."""
    if split in DIRICHLET_SPLITS:
        figure("rates-ijcnn-b1", split, jobs, force_tune, build=False)
        return
    prepare(split)
    rate_study.tune(force=force_tune, datasets=rate_study.split_datasets(rate_study.TUNE_DATASETS, split))
    rate_study.main_runs(jobs, split)
    rate_study.eta_runs(jobs, split)
    rate_study.batch1_runs(jobs, split)
    rate_study.batch8_runs(jobs, split)
    rate_study.batch32_runs(jobs, split)
    rate_study.gamma_runs(jobs, split)
    rate_study.partial_runs(jobs, split)
    rate_study.ijcnn_t10000_runs(jobs, dataset=dataset_of(split))
    rate_study.report(split)


def benchmarks(jobs, split="default"):
    """The adaptive-method benchmarks of the default or the k-means split; for a Dirichlet split only its Figure 2 (the
    selected comparison)."""
    if split in DIRICHLET_SPLITS:
        figure("selected-ijcnn1-32", split, jobs, build=False)
        return
    prepare(split)
    datasets = None if split == "default" else rate_study.split_datasets(rate_compare_selected.DATASETS, split)
    rate_compare_10k.run(jobs, datasets=datasets)
    rate_compare_10k.report(split)
    rate_compare_levels_10k.run(jobs, datasets=datasets)
    rate_compare_levels_10k.report(split)


def figure(name, split="default", jobs=4, force_tune=False, build=True):
    """One figure of one split: prepare, tune, run, and (with `build`) export its tables."""
    if name not in FIGURES:
        raise ValueError(f"unknown figure {name!r}; choose one of {', '.join(FIGURES)}")
    dataset = dataset_of(split)
    prepare(split)
    if name == "rates-ijcnn-b1":
        if force_tune:
            rate_study.tune(force=True, datasets=(dataset,))
            rate_study.tune_batch(rate_study.PRIMARY_BATCH, force=True, datasets=(dataset,))
        rate_study.ijcnn_t10000_runs(jobs, dataset=dataset, only=True)
    else:
        rate_compare_10k.run(jobs, datasets=(dataset,))
    if build:
        export(split, only=(name,))


def export(split="default", only=None):
    """Write the PGFPlots tables: everything for the default and the k-means split (for the k-means split also the sources
    of the fixed-eta figures, from its runs), or a Dirichlet split's two figures."""
    protocol.validate()
    dataset = dataset_of(split)
    if split == "default" and only is None:
        export_rate_study_pgfplots.export()
        export_benchmark_pgfplots.export()
        return
    if split == KMEANS_SPLIT and only is None:
        datasets = rate_study.split_datasets(export_rate_study_pgfplots.DATASETS, split)
        check_eta_ablation(datasets)
        export_rate_study_pgfplots.export(datasets=datasets)
        export_benchmark_pgfplots.export(datasets=rate_study.split_datasets(export_benchmark_pgfplots.DATASETS, split))
        synthetic = dataset_of(split, "synthetic-kappa10")
        figure_sources.write_fixed_eta_sources({name: rate_study.fixed_etas(synthetic, batch) for name, batch in
                                                zip(figure_sources.FIXED_ETA_FIGURES, rate_study.paper_batches(synthetic))})
        return
    for name in (only or FIGURES):
        if name == "rates-ijcnn-b1":
            export_rate_study_pgfplots.export_ijcnn_panel(dataset)
        else:
            export_benchmark_pgfplots.export(groups=("selected",), datasets=(dataset,))


def check_eta_ablation(datasets):
    """The eta figures draw the default figures' five etas; stop unless the eta ablation of every dataset, regime and
    oracle planned each of them and finished its runs for every seed."""
    for dataset in datasets:
        drawn = export_rate_study_pgfplots.eta_values(dataset)
        for regime in rate_study.REGIMES:
            for mode in rate_study.MODES:
                missing = rate_study.missing_eta_runs(dataset, regime, mode, drawn)
                if missing:
                    raise RuntimeError(f"{dataset} {regime} {mode}: the eta ablation has no complete runs at eta "
                                       f"{missing}, which the eta figure draws")


def _sources(split="default", only=None):
    """The standalone sources to compile: the named figures of a split, else all of them (for the default
    split: every one of the paper's own figures, not those of the other splits; for the k-means split its version of each)."""
    if split == KMEANS_SPLIT and only is None:
        return figure_sources.kmeans_sources()
    if split != "default" or only is not None:
        return [figure_sources.source_path(name, split) for name in (only or FIGURES)]
    sources = []
    for folder in (ROOT / "figures" / "rates", ROOT / "figures" / "benchmarks"):
        sources += [source for source in sorted(folder.glob("*.tex"))
                    if "-dirichlet" not in source.name and "-kmeans" not in source.name]
    return sources


def compile_figures(split="default", only=None):
    engine = shutil.which("pdflatex")
    if engine is None:
        raise RuntimeError("pdflatex is not on PATH; tables were exported but PDFs were not built")
    for source in _sources(split, only):
        print(f"compiling {source.relative_to(ROOT)}", flush=True)
        subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", source.name],
                       cwd=source.parent, check=True)


def doctor(strict=False):
    problems = []
    try:
        protocol.validate()
        print("protocol: OK (frozen paper specification matches executable code)")
    except AssertionError as error:
        problems.append(str(error))
        print(f"protocol: ERROR: {error}")
    print(f"python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        problems.append("Python 3.10 or newer is required")
    for package, wanted in PINNED.items():
        try:
            found = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found = "not installed"
        match = found == wanted
        print(f"{package}: {found} (pinned {wanted})" + ("" if match else " [mismatch]"))
        if strict and not match:
            problems.append(f"{package} {wanted} required, found {found}")
    present = [name for name in rate_study.DATASETS if (data.DATA / f"{name}.npz").exists()]
    print(f"datasets: {len(present)}/{len(rate_study.DATASETS)} prepared")
    print(f"pdflatex: {shutil.which('pdflatex') or 'not found (only needed to compile PDFs)'}")
    print(f"workspace: {ROOT}")
    if problems:
        raise SystemExit("doctor failed:\n- " + "\n- ".join(problems))


def run_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def _parser():
    parser = argparse.ArgumentParser(prog="adafed", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    split_help = ("client split: `default` (the paper's), or `dirichlet0.1`, `dirichlet0.5`, `dirichlet1`, "
                  "`dirichlet0.01` (Dirichlet label skew of IJCNN1; Figures 1 and 2 only), or `kmeans` (balanced "
                  "k-means of the per-example gradients of every dataset, 32 equal clients; the whole pipeline)")
    sub = commands.add_parser("prepare", help="build/check datasets and convex references")
    sub.add_argument("--split", choices=SPLITS, default="default", help=split_help)
    for name, help_text in (
        ("rates", "run the frozen AdaFed rate and ablation matrix"),
        ("benchmarks", "run the frozen adaptive-method benchmarks"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--jobs", type=int, default=default_jobs())
        sub.add_argument("--split", choices=SPLITS, default="default", help=split_help)
        if name == "rates":
            sub.add_argument("--force-tune", action="store_true")
    sub = commands.add_parser("export", help="validate runs and write the PGFPlots tables")
    sub.add_argument("--split", choices=SPLITS, default="default", help=split_help)
    sub = commands.add_parser("figures", help="export tables and compile standalone figures")
    sub.add_argument("--split", choices=SPLITS, default="default", help=split_help)
    sub.add_argument("--no-compile", action="store_true")
    sub = commands.add_parser("figure", help="run, export and compile one figure of one split (or `all` of them)")
    sub.add_argument("name", choices=FIGURES + ("all",),
                     help="rates-ijcnn-b1 (Figure 1), selected-ijcnn1-32 (Figure 2), or all (both)")
    sub.add_argument("--split", choices=SPLITS + ("all",), default="default",
                     help=split_help + "; `all`: the default split, the four Dirichlet splits and the k-means split")
    sub.add_argument("--jobs", type=int, default=default_jobs())
    sub.add_argument("--force-tune", action="store_true")
    sub.add_argument("--no-compile", action="store_true")
    commands.add_parser("smoke", help="run a short end-to-end algorithm check")
    commands.add_parser("test", help="run protocol and unit tests")
    sub = commands.add_parser("doctor", help="inspect the environment and protocol lock")
    sub.add_argument("--strict", action="store_true", help="require exactly pinned dependency versions")
    sub = commands.add_parser("protocol", help="print the frozen experiment specification")
    sub.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sub = commands.add_parser("all", help="run the complete paper pipeline, including the Dirichlet splits")
    sub.add_argument("--jobs", type=int, default=default_jobs())
    sub.add_argument("--force-tune", action="store_true")
    sub.add_argument("--no-compile", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.split)
    elif args.command == "rates":
        rates(args.jobs, args.force_tune, args.split)
    elif args.command == "benchmarks":
        benchmarks(args.jobs, args.split)
    elif args.command == "export":
        export(args.split)
    elif args.command == "figures":
        export(args.split)
        if not args.no_compile:
            compile_figures(args.split)
    elif args.command == "figure":
        for split in (SPLITS if args.split == "all" else (args.split,)):
            for name in (FIGURES if args.name == "all" else (args.name,)):
                figure(name, split, args.jobs, args.force_tune)
                if not args.no_compile:
                    compile_figures(split, only=(name,))
    elif args.command == "smoke":
        protocol.validate()
        smoke_test.run()
    elif args.command == "test":
        run_tests()
    elif args.command == "doctor":
        doctor(args.strict)
    elif args.command == "protocol":
        current = protocol.validate()
        print(json.dumps(current, indent=2) if args.json else _human_protocol(current))
    else:
        rates(args.jobs, args.force_tune)
        benchmarks(args.jobs)
        for split in OTHER_SPLITS:
            rates(args.jobs, args.force_tune, split)
            benchmarks(args.jobs, split)
        for split in SPLITS:
            export(split)
        if not args.no_compile:
            for split in SPLITS:
                compile_figures(split)


def _human_protocol(spec):
    rate = spec["rates"]
    benchmark = spec["benchmarks"]
    return "\n".join((
        "AdaFed frozen paper protocol",
        f"  datasets: {', '.join(spec['datasets'])}",
        f"  regimes: {', '.join(spec['objectives'])}",
        f"  rate K: {rate['K']}; stochastic batches: 1 (main), 8/32 (appendix)",
        f"  rate seeds: {rate['stochastic_seeds'][0]}--{rate['stochastic_seeds'][-1]}",
        f"  aggregation: {rate['aggregation']}",
        f"  bands: {rate['error_band']}",
        f"  benchmarks: K={benchmark['K']}, T={benchmark['rounds']}, "
        f"b={benchmark['stochastic_batch']}, "
        f"seeds={benchmark['stochastic_seeds'][0]}--{benchmark['stochastic_seeds'][-1]}",
    ))


if __name__ == "__main__":
    main()
