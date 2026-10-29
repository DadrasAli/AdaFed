#!/usr/bin/env python3
"""Standalone figure sources of the IJCNN1 Dirichlet splits and the k-means splits, derived from the default figures' sources.

Figure 1 (figures/rates/rates-ijcnn-b1.tex) and Figure 2 (figures/benchmarks/selected-ijcnn1-32.tex)
exist once per client split. A Dirichlet or k-means source is the default source with only the table prefix
changed (ijcnn1-32-... becomes ijcnn1-32-dirichlet<alpha>-... or ijcnn1-32-kmeans-...) and a comment line naming the split;
styles, axes, limits and the pdf-metadata include are the default figure's. The k-means split also has a version of every other
figure of the paper (<name>-kmeans.tex, reading the tables of the k-means split of the figure's dataset). The two synthetic
K-comparison figures print the tuned eta of each panel ("fixed eta = ..."); their k-means sources take it from the k-means
runs and are written by `python -m adafed export --split kmeans` (`write_fixed_eta_sources`), not tracked. Every k-means
source also shades a band behind each curve that is a mean over seeds (`with_bands`; the k-means Figures 1 and 2 have them
from the default sources), and the eta and gamma figures set their legend on one line (`with_one_line_legend`). The tracked
files
are checked against this rendering by tests/test_figure_sources.py and tests/test_kmeans_all.py. To rewrite them:

    python -m tools.figure_sources
"""

import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# figure name -> (folder, default source, table prefix the source reads, a one-line description)
FIGURES = {
    "rates-ijcnn-b1": ("rates", "rates-ijcnn-b1.tex", "experiments-data/ijcnn1-32-",
                       "AdaFed rates (K = 1, 10, 100) on IJCNN1"),
    "selected-ijcnn1-32": ("benchmarks", "selected-ijcnn1-32.tex", "experiments-data/selected-ijcnn1-32-",
                           "AdaFed, FedAdagrad, FedLADA and FAFED on IJCNN1"),
}

# Every figure of the paper (the standalone sources the default split compiles): name -> (folder, the dataset whose tables it
# reads, a one-line description). The k-means split has a version of each.
PAPER_FIGURES = {
    "rates-ijcnn-b1": ("rates", "ijcnn1-32", FIGURES["rates-ijcnn-b1"][3]),
    "rates-ijcnn-b32": ("rates", "ijcnn1-32", "AdaFed rates (K = 1, 10, 100; b = 32) on IJCNN1"),
    "rates-synthetic-b1": ("rates", "synthetic-kappa10", "AdaFed rates (K = 1, 10, 100) on synthetic data (kappa = 10)"),
    "rates-synthetic-b8": ("rates", "synthetic-kappa10", "AdaFed rates (K = 1, 10, 100; b = 8) on synthetic data (kappa = 10)"),
    "eta-ijcnn": ("rates", "ijcnn1-32", "AdaFed sensitivity to eta on IJCNN1"),
    "eta-synthetic": ("rates", "synthetic-kappa10", "AdaFed sensitivity to eta on synthetic data (kappa = 10)"),
    "gamma-ijcnn": ("rates", "ijcnn1-32", "AdaFed sensitivity to gamma on IJCNN1"),
    "gamma-synthetic": ("rates", "synthetic-kappa10", "AdaFed sensitivity to gamma on synthetic data (kappa = 10)"),
    "partial-ijcnn": ("rates", "ijcnn1-32", "AdaFed under partial participation on IJCNN1"),
    "partial-synthetic": ("rates", "synthetic-kappa10", "AdaFed under partial participation on synthetic data (kappa = 10)"),
    "selected-ijcnn1-32": ("benchmarks", "ijcnn1-32", FIGURES["selected-ijcnn1-32"][3]),
    "selected-synthetic-kappa1": ("benchmarks", "synthetic-kappa1",
                                  "AdaFed, FedAdagrad, FedLADA and FAFED on synthetic data (kappa = 1)"),
    "server-ijcnn1-32": ("benchmarks", "ijcnn1-32", "Server-level adaptation (AdaFed, FedAdagrad, FedAdam, FedUR) on IJCNN1"),
    "server-synthetic-kappa1": ("benchmarks", "synthetic-kappa1",
                                "Server-level adaptation (AdaFed, FedAdagrad, FedAdam, FedUR) on synthetic data (kappa = 1)"),
    "client-ijcnn1-32": ("benchmarks", "ijcnn1-32",
                         "Client-level adaptation (AdaFed, Local AdaAlter, FedSPS, FedLADA) on IJCNN1"),
    "client-synthetic-kappa1": ("benchmarks", "synthetic-kappa1",
                                "Client-level adaptation (AdaFed, Local AdaAlter, FedSPS, FedLADA) on synthetic data (kappa = 1)"),
    "both-ijcnn1-32": ("benchmarks", "ijcnn1-32", "Adaptation at both levels (AdaFed, FAFED, FedAda2) on IJCNN1"),
    "both-synthetic-kappa1": ("benchmarks", "synthetic-kappa1",
                              "Adaptation at both levels (AdaFed, FAFED, FedAda2) on synthetic data (kappa = 1)"),
}

# The figures whose panels print the eta their runs used ("fixed eta = ..."), exact row then stochastic row, columns smooth
# nonconvex, smooth convex, nonsmooth convex. Their k-means sources are written from the k-means runs at export.
FIXED_ETA_FIGURES = ("rates-synthetic-b1", "rates-synthetic-b8")
FIXED_ETA = re.compile(r"\{fixed \$\\eta=([^$]*)\$\}")

# The bands of the k-means figures: behind every curve that is a mean over seeds, the stochastic rows and, in the
# partial-participation figures, the exact rows with p < 1 (random client sampling); curves of exact gradients with full
# participation have none. Each reads the exporters' band table of its curve (`write_band`: the directional one-sided RMS
# deviation over the seeds), <table>-band.dat for a rate-study table and <table>-<method>-band.dat for a benchmark table,
# and is drawn with Figure 1's macros (\MeanBand behind the uniform-random mean, \LastBand behind the last iterate).
BAND_SOURCE = "rates-ijcnn-b1"
BAND_MACROS = re.compile(r"% Asymmetric one-sided standard-deviation bands.*?\n\\newcommand\{\\LastBand\}.*?\n\}\n", re.S)
PLOT = re.compile(r"  \\addplot\+\[([\w!]+),[^\]]*\]\n    table\[x=round,y=(\w+)\] \{(experiments-data/[^}]+)\.dat\};\n")
DRAWN_BAND = re.compile(r"\\(MeanBand|LastBand)\{\w+\}\{[\w!]+\}\{([^}]+)\}")
REGIME = re.compile(r"-(smooth-nonconvex|nonsmooth-convex|smooth-convex)(?:-|$)")
REGIME_TAG = {"smooth-nonconvex": "snc", "smooth-convex": "sc", "nonsmooth-convex": "nsc"}
BAND_NOTE = "% Shaded bands behind every curve that is a mean over seeds, drawn as in rates-ijcnn-b1.tex.\n"

# The eta and gamma figures' legend line is wider than the figure, so it wraps; their k-means sources set it on one line,
# scaled down to the figure's width (the log reports both widths).
LEGEND = "\n\\AdaFedFiveLegend\n"
ONE_LINE_LEGEND = r"""% The legend on one line, scaled down to the figure's width if it is wider.
\newsavebox{\LegendBox}
\newcommand{\OneLineLegend}[1]{%
  \sbox{\LegendBox}{#1}%
  \typeout{one-line legend: natural width \the\wd\LegendBox, line width \the\linewidth}%
  \ifdim\wd\LegendBox>\linewidth\resizebox{\linewidth}{!}{\usebox{\LegendBox}}\else\usebox{\LegendBox}\fi}
"""


KMEANS_SPLIT = "kmeans"


def split_name(alpha):
    return f"dirichlet{float(alpha):g}"


def source_path(name, split="default"):
    folder = PAPER_FIGURES[name][0]
    stem = name if split == "default" else f"{name}-{split}"
    return ROOT / "figures" / folder / f"{stem}.tex"


def kmeans_sources():
    """The k-means source of every paper figure."""
    return [source_path(name, KMEANS_SPLIT) for name in PAPER_FIGURES]


def tex_power(value):
    """A power of ten as the figures write it: 1, 10, 10^2 .. 10^9, 10^{-1}, 10^{10}."""
    k = round(math.log10(value))
    if abs(10.0 ** k - value) > 1e-12 * value:
        raise ValueError(f"eta {value!r} is not a power of ten")
    return "1" if k == 0 else "10" if k == 1 else f"10^{k}" if 2 <= k <= 9 else f"10^{{{k}}}"


def with_fixed_etas(text, etas):
    """`text` with its panels' "fixed eta = ..." labels set to `etas`, in the order the panels appear."""
    if len(FIXED_ETA.findall(text)) != len(etas):
        raise ValueError(f"the source has {len(FIXED_ETA.findall(text))} fixed-eta labels, {len(etas)} etas given")
    values = iter(etas)
    return FIXED_ETA.sub(lambda m: "{fixed $\\eta=" + tex_power(next(values)) + "$}", text)


def render_with(name, dataset=None, how=None, fixed=None):
    """The source of figure `name` from the default figure's source: its tables those of `dataset` (default: the figure's
    own, unchanged), a comment naming the split (`how`: its description and note lines) and, for a fixed-eta figure, the
    panels' etas `fixed` (default: unchanged). With neither, the default source itself."""
    folder, base, what = PAPER_FIGURES[name]
    default = f"{name}.tex"
    text = (ROOT / "figures" / folder / default).read_text()
    if fixed is not None:
        text = with_fixed_etas(text, fixed)
    if dataset is None:
        return text
    marker = "\\input{../pdf-metadata.tex}\n"
    if marker not in text:
        raise ValueError(f"{default} has no pdf-metadata include")
    table = re.compile(r"(experiments-data/(?:(?:selected|server|client|both)-)?)" + re.escape(base) + "-")
    text, count = table.subn(lambda m: m.group(1) + dataset + "-", text)
    if not count or count != text.count("experiments-data/"):
        raise ValueError(f"{default} does not read only tables of {base}")
    lines = list(how[1][:-1]) + [f"{how[1][-1]} Same layout, styles and limits as {default}."]
    note = f"% {what} {how[0]}: {dataset}\n" + "".join(f"% {line}\n" for line in lines)
    return text.replace(marker, marker + note, 1)


def band_of(table, column):
    """(macro, band table) of the curve `column` of `table` (its path without .dat) if the curve is a mean over seeds,
    else None. Benchmark columns are <method>_random and <method>_last, rate-study columns random and last."""
    stem = table.rsplit("/", 1)[-1]
    random_participation = re.search(r"-partial-full-.*-(0\.\d+)$", stem)
    if "-stochastic-" not in stem and not re.search(r"-b\d+-", stem) and not random_participation:
        return None
    macro = "MeanBand" if column.endswith("random") else "LastBand"
    method = column.rsplit("_", 1)[0] if "_" in column else None
    return macro, f"{table}-{method}-band.dat" if method else f"{table}-band.dat"


def with_bands(text):
    """`text` with a band behind every curve that is a mean over seeds: in each panel without bands, the \\MeanBand lines
    and then the \\LastBand lines of its curves before its first plot, and Figure 1's band macros and the fillbetween
    library in the preamble. A panel that draws bands already must draw exactly those of its curves."""
    head, *panels = text.split("\\nextgroupplot")
    out = []
    for panel in panels:
        curves = PLOT.findall(panel)
        if panel.count("\\addplot") != len(curves):
            raise ValueError(f"a panel has {panel.count(chr(92) + 'addplot')} plots, {len(curves)} of the expected form")
        bands = [(color, column, *band) for color, column, table in curves for band in [band_of(table, column)] if band]
        drawn = DRAWN_BAND.findall(panel)
        if drawn and sorted(drawn) != sorted((macro, band) for _, _, macro, band in bands):
            raise ValueError(f"a panel draws bands {sorted(drawn)}, its curves need {sorted(b[2:] for b in bands)}")
        if drawn or not bands:
            out.append(panel)
            continue
        keys, lines = {}, {"MeanBand": [], "LastBand": []}
        for color, column, macro, band in bands:
            # path names as Figures 1 and 2 write them: regime, row, Last for the last iterate, then the method or a letter
            key = keys.setdefault(band, re.sub("[^a-z]", "", column.rsplit("_", 1)[0]) if "_" in column
                                  else chr(ord("A") + len(keys)))
            name = (REGIME_TAG[REGIME.search(band).group(1)] + ("exact" if "-full-" in band else "")
                    + ("Last" if macro == "LastBand" else "") + key)
            lines[macro].append(f"  \\{macro}{{{name}}}{{{color}}}{{{band}}}\n")
        first = panel.index("  \\addplot")
        out.append(panel[:first] + "".join(lines["MeanBand"] + lines["LastBand"]) + panel[first:])
    if out == panels:
        return text
    if "\\newcommand{\\MeanBand}" not in head:
        source = (ROOT / "figures" / PAPER_FIGURES[BAND_SOURCE][0] / f"{BAND_SOURCE}.tex").read_text()
        if head.count("\\usepgfplotslibrary{groupplots}\n") != 1 or head.count("\\begin{document}") != 1:
            raise ValueError("the source does not load the groupplots library once before \\begin{document}")
        head = head.replace("\\usepgfplotslibrary{groupplots}\n", "\\usepgfplotslibrary{groupplots,fillbetween}\n")
        head = head.replace("\\begin{document}", BAND_MACROS.search(source).group(0) + "\n\\begin{document}")
    return "\\nextgroupplot".join([head] + out)


def with_one_line_legend(text):
    """`text` (an eta or gamma figure) with its legend set on one line, scaled down to the figure's width if wider."""
    if text.count(LEGEND) != 1 or text.count("\\begin{document}") != 1:
        raise ValueError("the source does not place \\AdaFedFiveLegend once on a line of its own")
    text = text.replace(LEGEND, "\n\\OneLineLegend{\\AdaFedFiveLegend}\n")
    return text.replace("\\begin{document}", ONE_LINE_LEGEND + "\n\\begin{document}")


def render(name, alpha):
    """The source of figure `name` for the Dirichlet split `alpha`, from the default figure's source."""
    return render_with(name, f"ijcnn1-32-{split_name(alpha)}",
                       (f"under Dirichlet label skew, alpha = {float(alpha):g}",
                        ("(32 clients, split seed 2021; the objective is the mean over all rows, as for every split, and the",
                         "server weights client i by n_i / n).")))


def render_kmeans(name, fixed=None):
    """The source of figure `name` for the k-means split of its dataset, from the default figure's source; a fixed-eta
    figure needs the etas of its k-means runs."""
    from adafed import generate
    base = PAPER_FIGURES[name][1]
    if (name in FIXED_ETA_FIGURES) != (fixed is not None):
        raise ValueError(f"{name}: the panels' etas are {'needed' if name in FIXED_ETA_FIGURES else 'not used'}")
    clients, rows = generate.split_shape(base)
    dataset = generate.kmeans_name(base)
    text = render_with(name, dataset,
                       ("under the balanced k-means split of the per-example gradients",
                        (f"({clients} clients of exactly {rows:,} rows, the partition of data/{dataset}-partition.json; the",
                         "objective is the mean over all rows, as for every split).")), fixed)
    banded = with_bands(text)
    if banded != text:
        note = f" Same layout, styles and limits as {name}.tex.\n"
        banded = banded.replace(note, note + BAND_NOTE, 1)
    return with_one_line_legend(banded) if name.startswith(("eta-", "gamma-")) else banded


def write_fixed_eta_sources(fixed):
    """Write the k-means sources of the fixed-eta figures; `fixed` maps each to the six etas of its k-means runs."""
    for name in FIXED_ETA_FIGURES:
        path = source_path(name, KMEANS_SPLIT)
        path.write_text(render_kmeans(name, fixed[name]))
        print("wrote", path.relative_to(ROOT))


def write_all():
    """Write the tracked sources: every Dirichlet source and every k-means source but the fixed-eta ones."""
    from adafed import generate
    for name in FIGURES:
        for alpha in generate.DIRICHLET_ALPHAS:
            path = source_path(name, split_name(alpha))
            path.write_text(render(name, alpha))
            print("wrote", path.relative_to(ROOT))
    for name in PAPER_FIGURES:
        if name not in FIXED_ETA_FIGURES:
            path = source_path(name, KMEANS_SPLIT)
            path.write_text(render_kmeans(name))
            print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    write_all()
