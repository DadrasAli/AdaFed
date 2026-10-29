import unittest
from unittest import mock

from adafed import rate_compare_levels_10k as levels
from adafed import rate_study


def _tables(etas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Tuning records of the shapes the ablation plans read: full, and the b=1 stochastic one."""
    full, stochastic = {}, {}
    for dataset in rate_study.TUNE_DATASETS:
        for regime in rate_study.REGIMES:
            for mode, table in (("full", full), ("stochastic", stochastic)):
                table[f"{dataset}/{regime}/{mode}"] = {
                    "selected_eta": 1.0, "candidates": [{"eta": eta} for eta in etas]}
    return full, stochastic


def _plan(runs, tables):
    """The configurations an ablation hands to the runner, with no run executed."""
    full, stochastic = tables
    captured = []
    with mock.patch.object(rate_study, "tune", return_value=full), \
         mock.patch.object(rate_study, "tune_batch", return_value=stochastic), \
         mock.patch.object(rate_study, "_run_parallel",
                           side_effect=lambda configs, jobs: captured.extend(configs)):
        runs(jobs=1)
    return captured


class LevelsBenchmarkRunCountTests(unittest.TestCase):
    def test_expected_count_equals_the_plan(self):
        plan = levels.report_plan()
        self.assertEqual(levels.expected_report_runs(), len(plan))
        self.assertEqual(len(set(plan)), len(plan))

    def test_expected_count_follows_the_current_constants(self):
        methods = len(levels.report_methods())
        seeds = len(rate_study.BENCHMARK_STOCHASTIC_SEEDS)
        self.assertEqual(levels.expected_report_runs(),
                         methods * len(levels.DATASETS) * len(rate_study.PANEL_REGIMES) * (1 + seeds))

    def test_expected_count_follows_a_changed_seed_tuple(self):
        for seeds in ((1, 2, 3), tuple(range(7))):
            with mock.patch.object(rate_study, "BENCHMARK_STOCHASTIC_SEEDS", seeds):
                self.assertEqual(levels.expected_report_runs(), len(levels.report_plan()))
                self.assertEqual(levels.expected_report_runs(),
                                 len(levels.report_methods()) * len(levels.DATASETS)
                                 * len(rate_study.PANEL_REGIMES) * (1 + len(seeds)))


class AblationTrajectoryCountTests(unittest.TestCase):
    def test_counts_equal_the_plans(self):
        tables = _tables()
        self.assertEqual(rate_study.gamma_trajectory_count(), len(_plan(rate_study.gamma_runs, tables)))
        self.assertEqual(rate_study.partial_trajectory_count(), len(_plan(rate_study.partial_runs, tables)))
        self.assertEqual(rate_study.eta_trajectory_count(*tables), len(_plan(rate_study.eta_runs, tables)))

    def test_counts_follow_a_changed_seed_tuple_and_eta_grid(self):
        tables = _tables(etas=(0.1, 1.0, 10.0))
        with mock.patch.object(rate_study, "PAPER_ABLATION_SEEDS", (5, 6, 7, 8)):
            self.assertEqual(rate_study.gamma_trajectory_count(), len(_plan(rate_study.gamma_runs, tables)))
            self.assertEqual(rate_study.partial_trajectory_count(),
                             len(_plan(rate_study.partial_runs, tables)))
            self.assertEqual(rate_study.eta_trajectory_count(*tables), len(_plan(rate_study.eta_runs, tables)))

    def test_counts_with_the_current_seeds(self):
        seeds = len(rate_study.PAPER_ABLATION_SEEDS)
        self.assertEqual(rate_study.gamma_trajectory_count(), 6 * 5 * (1 + seeds))
        self.assertEqual(rate_study.partial_trajectory_count(), 6 * (1 + 5 * seeds))
        self.assertEqual(rate_study.eta_trajectory_count(*_tables()), 6 * 5 * (1 + seeds))


class SeedTextTests(unittest.TestCase):
    def test_words_and_ranges_come_from_the_constants(self):
        self.assertEqual(rate_study.seed_word(), "twenty")
        self.assertEqual(rate_study.seed_range(), "2027--2046")
        self.assertEqual(rate_study.seed_word((1, 2, 3)), "three")
        self.assertEqual(rate_study.seed_word(tuple(range(40))), "40")
        with mock.patch.object(rate_study, "PAPER_STOCHASTIC_SEEDS", tuple(range(2027, 2037))):
            self.assertEqual(rate_study.seed_word(), "ten")
            self.assertEqual(rate_study.seed_range(), "2027--2036")


if __name__ == "__main__":
    unittest.main()
