import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from adafed import (fl, protocol, rate_compare_joint, rate_compare_levels_10k,
                    rate_compare_selected, rate_study)
from tools import export_rate_study_pgfplots as exporter


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_matches_code(self):
        self.assertEqual(protocol.validate(), protocol.FROZEN)

    def test_rate_orchestration_has_no_legacy_stochastic_suite(self):
        tuning = {}
        for dataset in rate_study.TUNE_DATASETS:
            for regime in rate_study.REGIMES:
                tuning[f"{dataset}/{regime}/full"] = {"selected_eta": 1.0}
        captured = []
        with mock.patch.object(rate_study, "tune", return_value=tuning), \
             mock.patch.object(rate_study, "_run_parallel",
                               side_effect=lambda configs, jobs: captured.extend(configs)):
            rate_study.main_runs(jobs=7)
        self.assertEqual(len(captured), 4 * 3 * 3)
        self.assertTrue(all(cfg["mode"] == "full" for cfg in captured))
        self.assertTrue(all(cfg["seed"] == 2027 for cfg in captured))
        self.assertEqual({cfg["b"] for cfg in captured}, {1})

    def test_primary_and_appendix_batches(self):
        self.assertEqual(rate_study.paper_batches("synthetic-kappa10"), (1, 8))
        self.assertEqual(rate_study.paper_batches("ijcnn1-32"), (1, 32))
        self.assertEqual(rate_study.BENCHMARK_BATCH,
                         {"synthetic-kappa1": 16, "ijcnn1-32": 32})

    def test_benchmark_constructors_set_the_frozen_batch(self):
        joint = rate_compare_joint.config(
            "synthetic-kappa1", "smooth-convex", "stochastic", 12, 7, 1.0, 1.0)
        selected = rate_compare_selected.config(
            "ijcnn1-32", "smooth-convex", "stochastic", "fedadagrad", 12, 7, 1.0)
        levels = rate_compare_levels_10k.config(
            "synthetic-kappa1", "smooth-convex", "stochastic", "fedadam", 12, 7, 1.0, 1.0)
        self.assertEqual(joint["b"], 16)
        self.assertEqual(selected["b"], 32)
        self.assertEqual(levels["b"], 16)

    def test_directional_band_is_nonnegative_and_contains_mean(self):
        samples = np.asarray([[0.0, 2.0, 7.0], [1.0, 4.0, 8.0], [5.0, 9.0, 12.0]])
        mean, low, high = exporter.directional_band(samples)
        np.testing.assert_allclose(mean, samples.mean(axis=0))
        self.assertTrue(np.all(low >= 0))
        self.assertTrue(np.all(low <= mean))
        self.assertTrue(np.all(mean <= high))

    def test_export_uses_arithmetic_mean(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for seed, values in enumerate(((1.0, 3.0), (5.0, 7.0))):
                path = Path(temporary) / f"seed{seed}.csv"
                path.write_text(
                    "round,objective,gap,grad_norm_sq,mean_gap,mean_grad_norm_sq,eta_t\n"
                    f"1,0,{values[0]},0,{values[0]},0,1\n"
                    f"2,0,{values[1]},0,{values[1]},0,1\n"
                )
                paths.append(path)
            rounds, random, last, _, _ = exporter.aggregate(paths, "smooth-convex")
        np.testing.assert_array_equal(rounds, [1, 2])
        np.testing.assert_allclose(random, [3.0, 5.0])
        np.testing.assert_allclose(last, [3.0, 5.0])


class CoreValidationTests(unittest.TestCase):
    def test_tau_rule(self):
        np.testing.assert_allclose(fl.tau_table("1", 3, 1), np.ones((3, 1)))
        expected = np.asarray([[1 / 9] * 10, [1 / 18] * 10, [1 / 27] * 10])
        np.testing.assert_allclose(fl.tau_table(rate_study.tau_for(10), 3, 10), expected)

    def test_unknown_setting_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            fl.resolve({"undocumented_variant": True})

    def test_invalid_participation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "p must be"):
            fl.resolve({"p": 0})


if __name__ == "__main__":
    unittest.main()
