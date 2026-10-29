"""The balanced k-means split: the balanced assignment step is exact (against brute force on small instances), the
construction is deterministic, every row is held exactly once and every client holds exactly n / clients rows; the tracked
partition file (data/ijcnn1-32-kmeans-partition.json) is intact, and the build of ijcnn1-32-kmeans uses it (32 clients of
1,000 rows, every row once, its SHA-256 checked) without rerunning k-means.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from adafed import data, generate


def balanced_labelings(n, K, size):
    """Every assignment of n rows to K clients with exactly `size` rows each."""
    if n == 0:
        yield ()
        return
    for rest in balanced_labelings(n - 1, K, size):
        counts = np.bincount(rest, minlength=K) if rest else np.zeros(K, int)
        for k in range(K):
            if counts[k] < size:
                yield rest + (k,)


def brute_force(C, size):
    n, K = C.shape
    return min(math.fsum(C[j, k] for j, k in enumerate(a)) for a in balanced_labelings(n, K, size))


class BalancedAssignmentTests(unittest.TestCase):
    def test_the_assignment_step_is_exact(self):
        rng = np.random.default_rng(3)
        for K, size in ((2, 4), (3, 3), (4, 2)):
            for case in range(20):
                C = rng.random((K * size, K)) if case % 2 else rng.integers(0, 4, (K * size, K)).astype(float)
                start = rng.permutation(np.repeat(np.arange(K), size))
                a, _ = generate.balanced_assignment(C, size, start)
                self.assertEqual(np.bincount(a, minlength=K).tolist(), [size] * K)
                self.assertAlmostEqual(float(C[np.arange(K * size), a].sum()), brute_force(C, size), places=12)

    def test_an_unbalanced_start_is_refused(self):
        with self.assertRaises(ValueError):
            generate.balanced_assignment(np.zeros((4, 2)), 2, [0, 0, 0, 1])


class KmeansSplitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.G = np.concatenate([rng.normal(c, 1.0, (100, 3)) for c in (-4.0, 0.0, 4.0, 8.0)])

    def test_deterministic_every_row_once_equal_clients(self):
        part, info = generate.kmeans_split_info(self.G, 8, restarts=3)
        again, info2 = generate.kmeans_split_info(self.G, 8, restarts=3)
        self.assertTrue(np.array_equal(part, again))
        self.assertEqual(info, info2)
        self.assertEqual(part.shape, (8, 50))
        self.assertTrue(np.array_equal(np.sort(part.ravel()), np.arange(400)))
        self.assertTrue((np.diff(part, axis=1) > 0).all())                  # ascending within a client
        zs = [r["zeta2"] for r in info["restarts"]]
        self.assertEqual(len(zs), 3)
        self.assertEqual(info["best"], int(np.argmax(zs)))
        means = self.G[part].mean(1)
        self.assertAlmostEqual(float(((means - self.G.mean(0)) ** 2).sum(1).mean()), max(zs), places=12)

    def test_a_kmeans_fixed_point(self):
        """At convergence no balanced assignment is cheaper for the final centres."""
        rng = np.random.default_rng(5)
        a, centres, _ = generate.balanced_kmeans(self.G, 8, 50, rng)
        C = generate.kmeans_costs(self.G, centres)
        b, _ = generate.balanced_assignment(C, 50, a)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.allclose(centres, np.stack([self.G[a == k].mean(0) for k in range(8)])))


class TrackedPartitionTests(unittest.TestCase):
    """data/ijcnn1-32-kmeans-partition.json: the shipped partition, which the build uses instead of rerunning k-means."""

    def test_the_tracked_file(self):
        part, doc = generate.read_kmeans_partition()
        self.assertEqual(part.shape, (32, 1000))
        self.assertTrue(np.array_equal(np.sort(part.ravel()), np.arange(32000)))
        self.assertTrue((np.diff(part, axis=1) > 0).all())                   # ascending within a client
        self.assertEqual(doc["sha256"], "b3ac05b09eecebaeb4fb4edf48c38a3bb8a47b3fe426f6192c5502b8434398e3")
        self.assertEqual(generate.partition_digest(part), doc["sha256"])
        self.assertEqual((doc["lambda"], doc["restarts"], doc["split_seed"]), (0.01, 10, 2021))
        self.assertEqual(doc["kept_restart"], int(np.argmax(doc["restart_zeta2"])))
        self.assertEqual(doc["zeta2"], max(doc["restart_zeta2"]))
        self.assertEqual(len(doc["restart_iterations"]), 10)

    def test_a_damaged_file_is_refused(self):
        part, doc = generate.read_kmeans_partition()
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
                    generate.read_kmeans_partition(path)

    def test_the_text_round_trips(self):
        part, doc = generate.read_kmeans_partition()
        info = {"restarts": [{"zeta2": z, "iterations": k} for z, k in zip(doc["restart_zeta2"], doc["restart_iterations"])],
                "best": doc["kept_restart"]}
        self.assertEqual(generate.kmeans_partition_text(part, info), generate.KMEANS_PARTITION.read_text())

    def test_regenerate_compares_with_the_tracked_partition(self):
        part, doc = generate.read_kmeans_partition()
        info = {"restarts": [{"zeta2": z, "iterations": k} for z, k in zip(doc["restart_zeta2"], doc["restart_iterations"])],
                "best": doc["kept_restart"]}
        other = part.copy()
        other[0, 0], other[1, 0] = other[1, 0], other[0, 0]
        with mock.patch.object(generate, "read_libsvm", return_value=(torch.zeros(32000, 22), torch.ones(32000))), \
             mock.patch.object(generate, "fetch", return_value="raw"), \
             mock.patch.object(generate, "kmeans_gradients", return_value=(None, None)):
            for got, equal in ((part, True), (other, False)):
                with mock.patch.object(generate, "kmeans_split_info", return_value=(got, info)):
                    result = generate.regenerate_kmeans("raw", log=lambda s: None)
                self.assertEqual(result["equal"], equal)
                self.assertEqual(result["tracked_sha256"], doc["sha256"])
                self.assertEqual(result["sha256"], generate.partition_digest(got))


class PipelineTests(unittest.TestCase):
    """The k-means split is treated as IJCNN1 itself where the protocol is read off the dataset name."""

    def test_protocol_settings_come_from_ijcnn1(self):
        from adafed import rate_study
        from tools import export_benchmark_pgfplots as bench
        self.assertTrue(generate.is_ijcnn1_split(generate.KMEANS_NAME))
        self.assertTrue(generate.is_ijcnn1_split("ijcnn1-32-dirichlet0.1"))
        self.assertFalse(generate.is_ijcnn1_split("synthetic-kappa1"))
        self.assertEqual(rate_study.base_dataset(generate.KMEANS_NAME), "ijcnn1-32")
        self.assertEqual(rate_study.BENCHMARK_BATCH[rate_study.base_dataset(generate.KMEANS_NAME)], 32)
        self.assertEqual(bench.batch_of(generate.KMEANS_NAME), 32)

    def test_fstar_entries_equal_the_default_splits(self):
        """Every split is the same 32,000 rows, so F* does not depend on the split."""
        table = json.loads(data.FSTAR.read_text())
        for pen in ("l2sq", "l1"):
            entry = table[f"{generate.KMEANS_NAME}/{pen}/0.01"]
            self.assertLess(abs(entry["f_star"] - table[f"ijcnn1-32/{pen}/0.01"]["f_star"]), 1e-12, pen)
            self.assertLess(entry["kkt_residual"], 1e-11)


@unittest.skipUnless((data.RAW / "ijcnn1.bz2").exists(), "needs the downloaded IJCNN1 file")
class Ijcnn1KmeansTests(unittest.TestCase):
    def test_the_build_uses_the_tracked_partition(self):
        X, y, part, meta = generate.build(generate.KMEANS_NAME, data.RAW, log=lambda s: None)
        tracked, doc = generate.read_kmeans_partition()
        self.assertEqual(X.shape, (32, 1000, 22))
        self.assertEqual(y.shape, (32, 1000))
        self.assertTrue(np.array_equal(part, tracked))
        self.assertEqual(generate.partition_digest(part), doc["sha256"])
        self.assertEqual(meta["kmeans"]["sha256"], doc["sha256"])
        g = generate.LIBSVM["ijcnn1-32"]
        A, labels = generate.read_libsvm(data.RAW / g["file"], g["dim"], g["rows"])
        self.assertTrue(np.array_equal(X, A.numpy()[part]))                   # every client holds its tracked rows
        self.assertTrue(np.array_equal(np.sort(part.ravel()), np.arange(32000)))

    def test_the_build_does_not_rerun_kmeans(self):
        with mock.patch.object(generate, "kmeans_split_info", side_effect=AssertionError("k-means was rerun")), \
             mock.patch.object(generate, "balanced_kmeans", side_effect=AssertionError("k-means was rerun")):
            generate.build(generate.KMEANS_NAME, data.RAW, log=lambda s: None)

    def test_a_damaged_tracked_file_stops_the_build(self):
        _, doc = generate.read_kmeans_partition()
        doc["partition"][0][0], doc["partition"][1][0] = doc["partition"][1][0], doc["partition"][0][0]
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps(doc))
            with mock.patch.object(generate, "KMEANS_PARTITION", bad):
                with self.assertRaises(ValueError):
                    generate.build(generate.KMEANS_NAME, data.RAW, log=lambda s: None)

    def test_zeta2_at_xstar_is_the_recorded_one(self):
        g = generate.LIBSVM["ijcnn1-32"]
        A, y = generate.read_libsvm(data.RAW / g["file"], g["dim"], g["rows"])
        G, _ = generate.kmeans_gradients(A, y)
        part, doc = generate.read_kmeans_partition()
        means = G[part].mean(1)
        z = float(((means - G.mean(0)) ** 2).sum(1).mean())
        self.assertAlmostEqual(z, doc["zeta2"], delta=1e-12 * doc["zeta2"])


if __name__ == "__main__":
    unittest.main()
