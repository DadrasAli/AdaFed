"""The Dirichlet client splits of IJCNN1 including the small alphas 0.01 and 0.05, whose draws leave a client empty every time
and are repaired: the construction is deterministic, holds every row exactly once, leaves no client empty, the repair moves
examples only from the largest client, and alpha 0.1, 0.5 and 1 give exactly the splits they gave before the repair existed
(their digests below were taken from the code before the change).
"""

import hashlib
import unittest

import numpy as np

from adafed import data, generate

SYNTHETIC_DIGESTS = {0.1: "e142952387fe34ed7e3e3897ba9615b0f652f0d8e9829fc22fca9dd03bffb732",
                     0.5: "fda2b0bc6709f39a19bb93ef2f2b6585a6885202d352e1c9b26edfae8b90c9e6",
                     1.0: "0b13383140ecad94bc392b5180b9997483cfcb3a448c053dd588e60a332c1b8c"}
IJCNN1_DIGESTS = {0.1: "7b33ee846e072d20d76f3b3e40e2a87e7662b472c5643376f6c4e4618753ae03",
                  0.5: "501fbeee7252d2f3d496541b5ebec76a397abe4816e140ffe87ba2713bf530bd",
                  1.0: "597446ea60737e43689c99cf6575f773d9b1f2bcb9a439ca2a17280627ef93c8"}
IJCNN1_SMALL = {0.01: ("9aec0e156f132c5910853b755712a43ef8d7789bc37ca16961073fbce4fc4ea3", 26),
                0.05: ("0c6a4e62bb03a8b910a1712fd2dac7889853f864cfe956357e115eaa1a6d6181", 9)}


def digest(order, offsets):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(order, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(offsets, dtype=np.int64).tobytes())
    return h.hexdigest()


def check_repair(test, info, order, offsets):
    """Replay the repair: each move goes to the lowest-index empty client from the currently largest one."""
    sizes = list(info["draw_sizes"])
    for to, src, row in info["moves"]:
        test.assertEqual(sizes[to], 0)
        test.assertEqual(to, sizes.index(0))
        test.assertEqual(src, int(np.argmax(sizes)))
        sizes[src] -= 1
        sizes[to] += 1
        pos = int(np.flatnonzero(order == row)[0])
        test.assertTrue(offsets[to] <= pos < offsets[to + 1])               # the moved row ends in its new client
    test.assertEqual(sizes, np.diff(offsets).tolist())
    test.assertEqual(info["moved"], len(info["moves"]))


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.labels = rng.choice([-1.0, 1.0], size=3000, p=[0.9, 0.1])

    def test_alpha_01_05_1_are_the_splits_they_always_were(self):
        for alpha, want in SYNTHETIC_DIGESTS.items():
            order, offsets, info = generate.dirichlet_split_info(self.labels, 12, alpha)
            self.assertEqual(digest(order, offsets), want, alpha)
            self.assertFalse(info["repaired"])

    def test_a_tiny_alpha_is_repaired_deterministically(self):
        a = generate.dirichlet_split_info(self.labels, 12, 0.01)
        b = generate.dirichlet_split_info(self.labels, 12, 0.01)
        self.assertEqual(digest(a[0], a[1]), digest(b[0], b[1]))
        order, offsets, info = a
        self.assertTrue(info["repaired"])
        self.assertEqual(info["draws"], generate.DIRICHLET_MAX_REDRAWS + 1)
        self.assertGreater(info["moved"], 0)
        self.assertEqual(np.sort(order).tolist(), list(range(3000)))
        self.assertGreaterEqual(int(np.diff(offsets).min()), 1)
        check_repair(self, info, order, offsets)

    def test_the_redraw_cap_is_honoured_and_the_first_draw_is_the_one_repaired(self):
        order, offsets, info = generate.dirichlet_split_info(self.labels, 12, 0.01, max_redraws=3)
        self.assertEqual(info["draws"], 4)
        self.assertTrue(info["repaired"])
        first = generate.dirichlet_split_info(self.labels, 12, 0.01, max_redraws=0)
        self.assertEqual(first[2]["draw_sizes"], info["draw_sizes"])             # draw 1 either way
        self.assertGreaterEqual(int(np.diff(offsets).min()), 1)
        check_repair(self, info, order, offsets)

    def test_no_repair_when_a_draw_within_the_cap_is_complete(self):
        order, offsets, info = generate.dirichlet_split_info(self.labels, 12, 1.0)
        self.assertEqual((info["repaired"], info["moved"], info["moves"]), (False, 0, []))


@unittest.skipUnless((data.RAW / "ijcnn1.bz2").exists(), "data/raw/ijcnn1.bz2 is not downloaded")
class Ijcnn1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = generate.LIBSVM[generate.DIRICHLET_BASE]
        _, y = generate.read_libsvm(generate.fetch(generate.DIRICHLET_BASE, data.RAW, log=lambda s: None), g["dim"], g["rows"])
        cls.y = y.numpy()

    def test_alpha_01_05_1_give_exactly_their_current_splits(self):
        for alpha, want in IJCNN1_DIGESTS.items():
            order, offsets, info = generate.dirichlet_split_info(self.y, 32, alpha)
            self.assertEqual(digest(order, offsets), want, alpha)
            self.assertFalse(info["repaired"])
            self.assertEqual(int(offsets[-1]), 32000)

    def test_small_alphas_are_deterministic_complete_nonempty_and_repaired_from_the_largest(self):
        for alpha, (want, moved) in IJCNN1_SMALL.items():
            order, offsets, info = generate.dirichlet_split_info(self.y, 32, alpha)
            self.assertEqual(digest(order, offsets), want, alpha)
            self.assertEqual(np.sort(order).tolist(), list(range(32000)))
            self.assertGreaterEqual(int(np.diff(offsets).min()), 1)
            self.assertEqual((info["repaired"], info["moved"], info["draws"]), (True, moved, 1001))
            check_repair(self, info, order, offsets)

    def test_the_small_alphas_build_as_datasets(self):
        for alpha in generate.DIRICHLET_SMALL_ALPHAS:
            name = generate.dirichlet_name(alpha)
            self.assertEqual(generate.dirichlet_alpha(name), alpha)
            X, y, order, meta = generate.build(name, data.RAW, log=lambda s: None)
            self.assertEqual((X.shape, len(meta["offsets"])), ((32000, 22), 33))
            self.assertEqual(meta["builder"]["alpha"], alpha)
            self.assertNotIn(name, generate.DATASETS)
            self.assertEqual(alpha in generate.DIRICHLET_ALPHAS, alpha == 0.01)       # 0.01 is in the pipeline, 0.05 is only measured


if __name__ == "__main__":
    unittest.main()
