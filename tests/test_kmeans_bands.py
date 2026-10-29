"""The bands of the k-means figures: every curve that is a mean over seeds has its band (Figure 1's macros, the curve's
colour, the exporters' band table) and no other curve has one; the default and Dirichlet sources are unchanged; the eta
and gamma legends of the k-means split are set on one line. (That the exporters' band is rate_study's directional band is
tested in test_figure_sources.)
"""

import re
import unittest

from tools import figure_sources as fs

FIXED = {"rates-synthetic-b1": (10, 10, 1, 1, 1, 1), "rates-synthetic-b8": (10, 10, 1, 10, 1, 1)}
DRAWN = re.compile(r"\\(MeanBand|LastBand)\{(\w+)\}\{([\w!]+)\}\{([^}]+)\}")
# band tables each k-means figure draws: rate figures 3 regimes x (3 K or 5 values) stochastic, partial also its exact
# p < 1 rows; benchmark figures 3 regimes x their methods
BAND_TABLES = {"rates-ijcnn-b1": 9, "rates-ijcnn-b32": 9, "rates-synthetic-b1": 9, "rates-synthetic-b8": 9,
               "eta-ijcnn": 15, "eta-synthetic": 15, "gamma-ijcnn": 15, "gamma-synthetic": 15,
               "partial-ijcnn": 15, "partial-synthetic": 15,
               "selected-ijcnn1-32": 12, "selected-synthetic-kappa1": 12, "server-ijcnn1-32": 12,
               "server-synthetic-kappa1": 12, "client-ijcnn1-32": 12, "client-synthetic-kappa1": 12,
               "both-ijcnn1-32": 9, "both-synthetic-kappa1": 9}


def kmeans_sources():
    return {name: fs.render_kmeans(name, FIXED.get(name)) for name in fs.PAPER_FIGURES}


class BandRuleTests(unittest.TestCase):
    def test_band_of(self):
        d = "experiments-data/"
        cases = {
            (d + "ijcnn1-32-kmeans-b32-smooth-convex-K10", "random"):
                ("MeanBand", d + "ijcnn1-32-kmeans-b32-smooth-convex-K10-band.dat"),
            (d + "synthetic-kappa10-kmeans-b8-nonsmooth-convex-K1", "last"):
                ("LastBand", d + "synthetic-kappa10-kmeans-b8-nonsmooth-convex-K1-band.dat"),
            (d + "ijcnn1-32-kmeans-exact-smooth-convex-K10", "random"): None,
            (d + "ijcnn1-32-kmeans-eta-full-smooth-convex-0.1", "last"): None,
            (d + "ijcnn1-32-kmeans-eta-stochastic-smooth-convex-0.1", "last"):
                ("LastBand", d + "ijcnn1-32-kmeans-eta-stochastic-smooth-convex-0.1-band.dat"),
            (d + "synthetic-kappa10-kmeans-gamma-full-smooth-nonconvex-100", "random"): None,
            (d + "ijcnn1-32-kmeans-partial-full-smooth-convex-0.25", "random"):
                ("MeanBand", d + "ijcnn1-32-kmeans-partial-full-smooth-convex-0.25-band.dat"),
            (d + "ijcnn1-32-kmeans-partial-full-smooth-convex-1", "random"): None,
            (d + "client-ijcnn1-32-kmeans-stochastic-smooth-convex", "local_adaalter_last"):
                ("LastBand", d + "client-ijcnn1-32-kmeans-stochastic-smooth-convex-local_adaalter-band.dat"),
            (d + "server-synthetic-kappa1-kmeans-full-smooth-convex", "fedur_random"): None,
            (d + "both-ijcnn1-32-kmeans-stochastic-nonsmooth-convex", "fedada2_random"):
                ("MeanBand", d + "both-ijcnn1-32-kmeans-stochastic-nonsmooth-convex-fedada2-band.dat"),
        }
        for (table, column), expected in cases.items():
            self.assertEqual(fs.band_of(table, column), expected, (table, column))


class KmeansBandTests(unittest.TestCase):
    def test_every_mean_over_seeds_has_its_band_and_no_other_curve_has_one(self):
        for name, text in kmeans_sources().items():
            for panel in text.split("\\nextgroupplot")[1:]:
                curves = fs.PLOT.findall(panel)
                self.assertEqual(panel.count("\\addplot"), len(curves), name)
                needed = {fs.band_of(table, column): colour for colour, column, table in curves
                          if fs.band_of(table, column)}
                drawn = DRAWN.findall(panel)
                self.assertEqual(sorted((macro, band) for macro, _, _, band in drawn), sorted(needed), name)
                for macro, _, colour, band in drawn:
                    self.assertEqual(colour, needed[(macro, band)], (name, band))       # the curve's own colour
                if drawn:                                                                # behind the curves
                    self.assertLess(panel.index("Band{"), panel.index("\\addplot"), name)

    def test_band_tables_per_figure(self):
        for name, text in kmeans_sources().items():
            bands = {band for _, _, _, band in DRAWN.findall(text)}
            self.assertEqual(len(bands), BAND_TABLES[name], name)
            self.assertTrue(all("-kmeans-" in band and band.endswith("-band.dat") for band in bands), name)
        self.assertEqual(sum(BAND_TABLES.values()), 216)

    def test_band_paths_are_named_once_and_in_letters(self):
        for name, text in kmeans_sources().items():
            names = [path for _, path, _, _ in DRAWN.findall(text)]
            self.assertEqual(len(names), len(set(names)), name)
            self.assertTrue(all(re.fullmatch("[A-Za-z]+", path) for path in names), name)

    def test_the_macros_and_opacities_are_figure_ones(self):
        figure_one = fs.BAND_MACROS.search(fs.source_path(fs.BAND_SOURCE).read_text()).group(0)
        self.assertIn("fill opacity=0.22", figure_one.split("\\newcommand{\\LastBand}")[0])
        self.assertIn("fill opacity=0.17", figure_one.split("\\newcommand{\\LastBand}")[1])
        for name, text in kmeans_sources().items():
            self.assertEqual(text.count("\\newcommand{\\MeanBand}"), 1, name)
            self.assertEqual(text.count("\\newcommand{\\LastBand}"), 1, name)
            self.assertIn("\\usepgfplotslibrary{groupplots,fillbetween}\n", text)
            if name not in fs.FIGURES:                                  # added to a source that had no bands
                self.assertIn(figure_one, text, name)
                self.assertIn(fs.BAND_NOTE, text, name)

    def test_figures_one_and_two_render_as_before(self):
        for name in fs.FIGURES:
            text = fs.render_kmeans(name)
            self.assertEqual(fs.with_bands(text), text)
            self.assertNotIn(fs.BAND_NOTE, text)

    def test_rendering_is_idempotent(self):
        for name, text in kmeans_sources().items():
            self.assertEqual(fs.with_bands(text), text, name)

    def test_a_panel_with_some_of_its_bands_is_refused(self):
        text = fs.render_kmeans("selected-ijcnn1-32")
        line = re.search(r"  \\LastBand\{[^\n]*\n", text).group(0)
        with self.assertRaises(ValueError):
            fs.with_bands(text.replace(line, "", 1))


class UnchangedSourcesTests(unittest.TestCase):
    def test_default_sources_draw_bands_only_in_figures_one_and_two(self):
        for name in fs.PAPER_FIGURES:
            text = fs.source_path(name).read_text()
            self.assertEqual("\\MeanBand{" in text, name in fs.FIGURES, name)
            self.assertNotIn("OneLineLegend", text, name)

    def test_dirichlet_sources_are_unchanged_renderings(self):
        for name in fs.FIGURES:
            for alpha in (0.01, 0.1, 0.5, 1):
                text = fs.render(name, alpha)
                self.assertEqual(fs.source_path(name, fs.split_name(alpha)).read_text(), text)
                self.assertNotIn(fs.BAND_NOTE, text)


class LegendTests(unittest.TestCase):
    def test_eta_and_gamma_legends_are_set_on_one_line(self):
        for name, text in kmeans_sources().items():
            one_line = name.startswith(("eta-", "gamma-"))
            self.assertEqual(text.count("\\OneLineLegend{\\AdaFedFiveLegend}"), int(one_line), name)
            self.assertEqual(text.count("\\newcommand{\\OneLineLegend}"), int(one_line), name)
            if one_line:
                self.assertNotIn(fs.LEGEND, text, name)
                self.assertLess(text.index("\\newcommand{\\OneLineLegend}"), text.index("\\begin{document}"))

    def test_a_source_without_the_legend_line_is_refused(self):
        with self.assertRaises(ValueError):
            fs.with_one_line_legend(fs.render_kmeans("server-ijcnn1-32"))


if __name__ == "__main__":
    unittest.main()
