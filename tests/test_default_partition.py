"""The default ijcnn1-32 split is shipped (data/ijcnn1-32-partition.json): the tracked file is intact, `build` uses it
instead of sorting, a damaged file is refused, and the protocol locks its digest."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from adafed import generate, problem, protocol

DIGEST = "0d6c70541218a2e22e1b996c3b5d33828b6496dec0fd5b9831b101819c92bb5c"


def damaged(doc, what):
    bad = json.loads(json.dumps(doc))
    if what == "digest":
        bad["partition"][0][0], bad["partition"][1][0] = bad["partition"][1][0], bad["partition"][0][0]
    elif what == "row":
        bad["partition"][0][0] = bad["partition"][0][1]
    else:
        bad["partition"][0].pop()
    return bad


def fake_rows():
    """32,000 rows whose first feature is the row's own index, so a client's rows can be read off."""
    A = torch.zeros(32000, 22, dtype=torch.float64)
    A[:, 0] = torch.arange(32000)
    return A, torch.ones(32000, dtype=torch.float64)


class TrackedFileTests(unittest.TestCase):
    def test_the_tracked_file(self):
        part, doc = generate.read_default_partition()
        self.assertEqual(part.shape, (32, 1000))
        self.assertEqual(part.dtype, np.int64)
        self.assertTrue(np.array_equal(np.sort(part.ravel()), np.arange(32000)))      # every row once
        self.assertEqual(doc["sha256"], DIGEST)
        self.assertEqual(generate.partition_digest(part), DIGEST)
        self.assertEqual((doc["clients"], doc["rows"], doc["rows_per_client"]), (32, 32000, 1000))
        g = generate.LIBSVM["ijcnn1-32"]
        self.assertEqual((doc["reference_lambda"], doc["block"], doc["exact_key"]),
                         (g["reference_lambda"], g["block"], g["exact_key"]))

    def test_the_text_round_trips(self):
        part, doc = generate.read_default_partition()
        self.assertEqual(generate.default_partition_text(part, doc["lambda_max"], doc["reference_kkt"]),
                         generate.DEFAULT_PARTITION.read_text())

    def test_a_damaged_file_is_refused(self):
        _, doc = generate.read_default_partition()
        with tempfile.TemporaryDirectory() as tmp:
            for what in ("digest", "row", "shape"):
                path = Path(tmp) / f"{what}.json"
                path.write_text(json.dumps(damaged(doc, what)))
                with self.assertRaises(ValueError, msg=what):
                    generate.read_default_partition(path)

    def test_the_other_splits_keep_their_digests(self):
        self.assertEqual(generate.read_kmeans_partition()[1]["sha256"], protocol.FROZEN["kmeans"]["partition_sha256"])
        self.assertEqual(protocol.FROZEN["datasets"]["ijcnn1-32"]["partition_sha256"], DIGEST)


class BuildTests(unittest.TestCase):
    """build('ijcnn1-32') with the IJCNN1 file replaced by rows that carry their own index."""

    def build(self, **patches):
        with mock.patch.object(generate, "fetch", return_value="raw"), \
             mock.patch.object(generate, "read_libsvm", return_value=fake_rows()):
            return generate.build("ijcnn1-32", "raw", log=lambda s: None)

    def test_the_build_uses_the_tracked_partition(self):
        X, y, part, meta = self.build()
        tracked, doc = generate.read_default_partition()
        self.assertEqual(X.shape, (32, 1000, 22))
        self.assertEqual(y.shape, (32, 1000))
        self.assertTrue(np.array_equal(part, tracked))
        self.assertTrue(np.array_equal(X[:, :, 0], tracked.astype(float)))            # every client holds its tracked rows
        self.assertEqual(meta["lambda_max"], doc["lambda_max"])
        self.assertEqual(meta["reference_kkt"], doc["reference_kkt"])

    def test_the_build_does_not_sort(self):
        def refuse(*args, **kwargs):
            raise AssertionError("the build rebuilt the partition")
        with mock.patch.object(generate, "deal", side_effect=refuse), \
             mock.patch.object(generate, "exact_margins", side_effect=refuse), \
             mock.patch.object(generate, "construct_default_partition", side_effect=refuse), \
             mock.patch.object(problem, "solve_l1", side_effect=refuse), \
             mock.patch.object(torch, "argsort", side_effect=refuse), \
             mock.patch.object(torch, "sort", side_effect=refuse):
            X, _, part, _ = self.build()
        self.assertEqual(generate.partition_digest(part), DIGEST)

    def test_a_damaged_tracked_file_stops_the_build(self):
        _, doc = generate.read_default_partition()
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            for what in ("digest", "row", "shape"):
                bad.write_text(json.dumps(damaged(doc, what)))
                with mock.patch.object(generate, "DEFAULT_PARTITION", bad):
                    with self.assertRaises(ValueError, msg=what):
                        self.build()


class RegenerateTests(unittest.TestCase):
    def test_regenerate_compares_with_the_tracked_partition(self):
        part, doc = generate.read_default_partition()
        other = part.copy()
        other[0, 0], other[1, 0] = other[1, 0], other[0, 0]
        with mock.patch.object(generate, "read_libsvm", return_value=fake_rows()), \
             mock.patch.object(generate, "fetch", return_value="raw"):
            for got, equal in ((part, True), (other, False)):
                with mock.patch.object(generate, "construct_default_partition", return_value=(got, 0.5, 1e-18)):
                    result = generate.regenerate_default("raw", log=lambda s: None)
                self.assertEqual(result["equal"], equal)
                self.assertEqual(result["tracked_sha256"], DIGEST)
                self.assertEqual(result["sha256"], generate.partition_digest(got))

    def test_the_construction_deals_the_sorted_rows_in_blocks(self):
        """The construction on a tiny instance: rows sorted by the margin at the reference solution, dealt round robin."""
        g = dict(generate.LIBSVM["ijcnn1-32"], rows=8, clients=2, rows_per_client=4, block=2, dim=1)
        A = torch.tensor([[3.], [1.], [4.], [2.], [8.], [6.], [7.], [5.]], dtype=torch.float64)
        y = torch.ones(8, dtype=torch.float64)
        w = torch.tensor([1.], dtype=torch.float64)
        with mock.patch.dict(generate.LIBSVM, {"ijcnn1-32": g}), \
             mock.patch.object(problem, "solve_l1", return_value=(w, 0.0, 0.0)), \
             mock.patch.object(problem, "lipschitz", return_value=1.0):
            part, lam_max, res = generate.construct_default_partition("ijcnn1-32", A, y, log=lambda s: None)
        # ascending margins give the rows 1,3,0,2,7,5,6,4: blocks of two, dealt round robin
        self.assertEqual(part.tolist(), [[1, 3, 7, 5], [0, 2, 6, 4]])
        self.assertEqual(res, 0.0)


class ProtocolLockTests(unittest.TestCase):
    def test_the_digest_is_locked(self):
        self.assertEqual(protocol.validate()["datasets"]["ijcnn1-32"]["partition_sha256"], DIGEST)

    def test_a_different_partition_fails_validation(self):
        part, doc = generate.read_default_partition()
        other = part.copy()
        other[0, 0], other[1, 0] = other[1, 0], other[0, 0]
        doc["partition"] = other.tolist()
        doc["sha256"] = generate.partition_digest(other)                   # a consistent file, but a different split
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "other.json"
            path.write_text(json.dumps(doc))
            with mock.patch.object(generate, "DEFAULT_PARTITION", path):
                with self.assertRaises(AssertionError):
                    protocol.validate()


if __name__ == "__main__":
    unittest.main()
