"""Datasets with unequal clients (the Dirichlet splits): the split, the pooled objective, and the ragged
paths of fl.py and adaptive_baselines.py against references and against the equal-client paths.

The objective is the mean over all n rows, F = sum_i (n_i / n) F_i + penalty, whatever the split; every
method's server weights client i by n_i / n. In-memory datasets only (nothing is written to data/); the
nonconvex penalty has no F*, so data/fstar.json is not touched either.
"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adafed import adaptive_baselines, data, fl, generate, problem, rate_study

NAMES = ("ragged-test-equal", "ragged-test-twin", "ragged-test-unequal", "ragged-test-const")
D_DIM = 4


def make(name, blocks, labels):
    """Register a dataset from per-client row arrays: equal-size ones as (N, n, d), others flat."""
    sizes = [len(b) for b in blocks]
    X = np.concatenate(blocks)
    y = np.concatenate(labels)
    meta = {"L": float(problem.lipschitz(torch.from_numpy(X)))}              # the pooled smoothness
    if len(set(sizes)) == 1 and not name.endswith("twin"):
        N, n = len(sizes), sizes[0]
        X, y = X.reshape(N, n, -1), y.reshape(N, n)
        data._cache[name] = {"X": X, "y": y, "partition": None, "B": X * (-y)[..., None], "meta": meta, "offsets": None}
    else:
        off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        data._cache[name] = {"X": X, "y": y, "partition": None, "B": X * (-y)[..., None], "meta": meta,
                             "offsets": off}


def read(path):
    with open(Path(path) / "metrics.csv") as f:
        return [{k: float(v) if v != "" else math.nan for k, v in r.items()} for r in csv.DictReader(f)]


class RaggedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        cls.eq = [rng.normal(size=(6, D_DIM)) for _ in range(5)]
        cls.eq_y = [rng.choice([-1.0, 1.0], size=6) for _ in range(5)]
        make("ragged-test-equal", cls.eq, cls.eq_y)
        make("ragged-test-twin", cls.eq, cls.eq_y)           # the same rows as flat data with offsets
        sizes = [1, 3, 9, 2, 17, 5]                          # one client larger than the block below
        cls.un = [rng.normal(size=(s, D_DIM)) for s in sizes]
        cls.un_y = [rng.choice([-1.0, 1.0], size=s) for s in sizes]
        make("ragged-test-unequal", cls.un, cls.un_y)
        # every client's rows are copies of one row: any minibatch of a client gives its exact gradient
        cls.const = [np.repeat(rng.normal(size=(1, D_DIM)), s, axis=0) for s in sizes]
        cls.const_y = [np.repeat(rng.choice([-1.0, 1.0], size=1), s) for s in sizes]
        make("ragged-test-const", cls.const, cls.const_y)
        cls._block = data.RAGGED_BLOCK
        data.RAGGED_BLOCK = 4                                # several blocks per client, a partial last block

    @classmethod
    def tearDownClass(cls):
        data.RAGGED_BLOCK = cls._block
        for n in NAMES:
            data._cache.pop(n, None)
        for k in [k for k in data._blocks if k[0] in NAMES]:
            data._blocks.pop(k)

    def cfg(self, dataset, algorithm="adafed", **kw):
        c = {"dataset": dataset, "penalty": "nonconvex", "lam": 0.05, "algorithm": algorithm, "mode": "full",
             "b": 3, "K": 3, "p": 1.0, "T": 8, "seed": 7, "eta": 1.0, "gamma": 0.5, "tau": "1/(2*t)",
             "eval_every": 1, "device": "cpu", "dtype": "float64"}
        if algorithm in ("fedlada", "fafed"):
            c.update(client_lr=0.5, alpha=0.1, rho=0.01, epsilon_v=1e-8, server_eta=1.0, c=1.0, tau="1", eta=None)
        c.update(kw)
        return c

    def go(self, cfg):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(cfg, out=tmp)
            (adaptive_baselines.run if cfg["algorithm"] in ("fedlada", "fafed") else fl.run)(cfg)
            return read(tmp)

    def assertSame(self, a, b, keys=("objective", "grad_norm_sq", "mean_grad_norm_sq"), tol=1e-11):
        self.assertEqual(len(a), len(b))
        for ra, rb in zip(a, b):
            for k in keys:
                self.assertLessEqual(abs(ra[k] - rb[k]), tol * max(abs(ra[k]), 1e-300), (k, ra, rb))

    # --- the split -------------------------------------------------------------------------------
    def test_dirichlet_split_is_deterministic_complete_and_nonempty(self):
        labels = np.where(np.random.default_rng(0).random(2000) < 0.1, 1.0, -1.0)
        for alpha in (0.1, 0.5, 1.0):
            o1, f1 = generate.dirichlet_split(labels, 12, alpha)
            o2, f2 = generate.dirichlet_split(labels, 12, alpha)
            np.testing.assert_array_equal(o1, o2)
            np.testing.assert_array_equal(f1, f2)
            np.testing.assert_array_equal(np.sort(o1), np.arange(len(labels)))
            self.assertEqual(len(f1), 13)
            self.assertTrue((np.diff(f1) > 0).all())
            for lo, hi in zip(f1[:-1], f1[1:]):
                self.assertTrue((np.diff(o1[lo:hi]) > 0).all())     # ascending inside a client
        self.assertFalse(np.array_equal(generate.dirichlet_split(labels, 12, 0.1)[0],
                                        generate.dirichlet_split(labels, 12, 0.1, seed=1)[0]))

    def test_dirichlet_names(self):
        self.assertEqual(generate.dirichlet_name(0.1), "ijcnn1-32-dirichlet0.1")
        self.assertEqual(generate.dirichlet_alpha("ijcnn1-32-dirichlet1"), 1.0)
        self.assertIsNone(generate.dirichlet_alpha("ijcnn1-32"))
        with self.assertRaises(ValueError):
            generate.dirichlet_alpha("ijcnn1-32-dirichlet01")
        self.assertNotIn(generate.dirichlet_name(0.1), generate.DATASETS)

    # --- the objective -----------------------------------------------------------------------------
    def test_objective_is_the_row_mean_and_the_n_i_weighted_client_mean(self):
        D = data.load("ragged-test-unequal")
        P = data.pooled(D, "l2sq", 0.1)
        x = torch.tensor([0.3, -0.2, 0.5, 0.1], dtype=torch.float64)
        rows = np.concatenate(self.un)
        labels = np.concatenate(self.un_y)
        row_mean = float(torch.nn.functional.softplus(torch.from_numpy(rows * -labels[:, None]) @ x).mean())
        by_client = [float(torch.nn.functional.softplus(torch.from_numpy(b * -y[:, None]) @ x).mean())
                     for b, y in zip(self.un, self.un_y)]
        n_i = np.array([len(b) for b in self.un])
        weighted = float(sum(n * f for n, f in zip(n_i, by_client)) / n_i.sum())
        penalty = float(0.5 * 0.1 * (x * x).sum())
        self.assertAlmostEqual(row_mean, weighted, places=14)             # local losses stay client means
        self.assertAlmostEqual(P.objective(x), row_mean + penalty, places=14)
        F, G = P.evaluate(x)
        self.assertAlmostEqual(float(F), row_mean + penalty, places=14)
        eps = 1e-6                                        # the gradient is the derivative of F
        for j in range(D_DIM):
            e = torch.zeros(D_DIM, dtype=torch.float64)
            e[j] = eps
            num = (P.objective(x + e) - P.objective(x - e)) / (2 * eps)
            _, g = P.smooth_value_grad(x)
            self.assertAlmostEqual(num, float(g[j] + 0.1 * x[j]), places=7)

    def test_the_objective_does_not_depend_on_the_partition(self):
        rows, labels = np.concatenate(self.un), np.concatenate(self.un_y)
        order = np.random.default_rng(2).permutation(len(rows))
        a = data.pooled(data.load("ragged-test-unequal"), "l1", 0.05)
        b = problem.Pooled(torch.from_numpy((rows * -labels[:, None])[order]), "l1", 0.05)
        x = torch.tensor([0.4, 0.1, -0.3, 0.2], dtype=torch.float64)
        self.assertAlmostEqual(float(a.evaluate(x)[0]), float(b.evaluate(x)[0]), places=14)

    def test_the_pooled_solver_reaches_its_optimum(self):
        P = data.pooled(data.load("ragged-test-unequal"), "l2sq", 0.1)
        x, f, res = problem.solve_l2sq(P, log=lambda s: None)
        self.assertLess(res, 1e-11)
        self.assertLessEqual(f, P.objective(x + 1e-3))

    def test_dirichlet_fstar_equal_the_default_splits_fstar(self):
        """Every split is the same 32,000 rows, so F* does not depend on the split."""
        table = json.loads(data.FSTAR.read_text())
        for pen in ("l2sq", "l1"):
            default = table[f"ijcnn1-32/{pen}/0.01"]["f_star"]
            for alpha in generate.DIRICHLET_ALPHAS:
                entry = table[f"{generate.dirichlet_name(alpha)}/{pen}/0.01"]
                self.assertLess(abs(entry["f_star"] - default), 1e-12, (pen, alpha))
                self.assertLess(entry["kkt_residual"], 1e-11)

    @unittest.skipUnless((data.RAW / "ijcnn1.bz2").exists(), "needs the downloaded IJCNN1 file")
    def test_recomputed_dirichlet_fstar_equal_the_default_splits_fstar(self):
        g = generate.LIBSVM[generate.DIRICHLET_BASE]
        A, y = generate.read_libsvm(generate.fetch(generate.DIRICHLET_BASE, data.RAW, log=lambda s: None), g["dim"], g["rows"])
        table = json.loads(data.FSTAR.read_text())
        self.assertEqual(len(y), 32000)
        for alpha in generate.DIRICHLET_ALPHAS:
            order, offsets = generate.dirichlet_split(y.numpy(), g["clients"], alpha)
            self.assertEqual(int(offsets[-1]), 32000)
            sel = torch.from_numpy(order)
            B = (A * (-y)[:, None])[sel]
            _, f2, _ = problem.solve_l2sq(problem.Pooled(B, "l2sq", 0.01), log=lambda s: None)
            _, f1, _ = problem.solve_l1(problem.Pooled(B, "l1", 0.01), float(problem.lipschitz(A)), log=lambda s: None)
            self.assertLess(abs(f2 - table["ijcnn1-32/l2sq/0.01"]["f_star"]), 1e-12)
            self.assertLess(abs(f1 - table["ijcnn1-32/l1/0.01"]["f_star"]), 1e-12)

    # --- the runners -------------------------------------------------------------------------------
    def test_ragged_path_equals_equal_path_for_equal_clients_exact(self):
        for p in (1.0, 0.6):
            a = self.go(self.cfg("ragged-test-equal", p=p))
            b = self.go(self.cfg("ragged-test-twin", p=p))
            self.assertSame(a, b)
            self.assertNotAlmostEqual(a[0]["objective"], a[-1]["objective"], places=6)

    def test_ragged_baselines_equal_equal_path_for_equal_clients_exact(self):
        for method in ("fedlada", "fafed"):
            a = self.go(self.cfg("ragged-test-equal", method))
            b = self.go(self.cfg("ragged-test-twin", method))
            self.assertSame(a, b)

    def test_adafed_unequal_clients_match_a_plain_reference(self):
        T, K, p, eta, gamma, lam = 6, 3, 0.7, 1.0, 0.5, 0.05
        c = self.cfg("ragged-test-unequal", T=T, K=K, p=p, eta=eta, gamma=gamma, lam=lam)
        got = self.go(c)
        N = len(self.un)
        masks = (torch.rand(T, N, generator=torch.Generator().manual_seed(c["seed"]), dtype=torch.float64) < p).numpy()
        tau = fl.tau_table(c["tau"], T, K)
        weight = np.array([len(b) for b in self.un]) / sum(len(b) for b in self.un)             # n_i / n
        P = data.pooled(data.load("ragged-test-unequal"), "nonconvex", lam)
        x, S = np.zeros(D_DIM), gamma
        for t in range(1, T + 1):
            F, G = P.evaluate(torch.from_numpy(x))
            self.assertLessEqual(abs(float(F) - got[t - 1]["objective"]), 1e-12 * float(F))
            self.assertLessEqual(abs(float(G) - got[t - 1]["grad_norm_sq"]), 1e-10 * float(G))
            eta_t = eta / S
            total = np.zeros(D_DIM)
            for i in np.flatnonzero(masks[t - 1]):
                B = self.un[i] * -self.un_y[i][:, None]
                xi = x.copy()
                for k in range(K):
                    g = B.T @ (1 / (1 + np.exp(-(B @ xi)))) / len(B) + 2 * lam * xi / (1 + xi * xi) ** 2
                    xi = xi - tau[t - 1, k] * eta_t * g
                total += weight[i] * (xi - x)
            step = total / p / math.fsum(tau[t - 1])
            x = x + step
            S *= math.sqrt(1 + (np.linalg.norm(step) / eta) ** 2)

    def _reference(self, kind, T, K, hp, lam=0.05):
        """The iterates x_1..x_T of FedAdagrad, FedLADA or FAFED (exact, full participation, p = 1) in numpy,
        the server weighting clients by n_i / n."""
        sizes = np.array([len(b) for b in self.un])
        w = (sizes / sizes.sum())[:, None]
        N = len(sizes)
        B = [self.un[i] * -self.un_y[i][:, None] for i in range(N)]

        def grad(i, x):
            return B[i].T @ (1 / (1 + np.exp(-(B[i] @ x)))) / len(B[i]) + 2 * lam * x / (1 + x * x) ** 2

        def grads(points):
            return np.stack([grad(i, points[i]) for i in range(N)])

        average = lambda a: (w * a).sum(0)
        x, xs = np.zeros(D_DIM), []
        if kind == "fedadagrad":
            lr = hp["c"] / data.load("ragged-test-unequal")["meta"]["L"]
            momentum, second = np.zeros(D_DIM), np.full(D_DIM, hp["adapt_tau"] ** 2)
            for t in range(T):
                xs.append(x.copy())
                local = np.tile(x, (N, 1))
                for _ in range(K):
                    local = local - lr * grads(local)
                delta = average(local) - x
                momentum = hp["beta1"] * momentum + (1 - hp["beta1"]) * delta
                second = second + delta ** 2
                x = x + hp["server_eta"] * momentum / (np.sqrt(second) + hp["adapt_tau"])
        elif kind == "fedlada":
            lr, a, b1, b2 = hp["client_lr"], hp["alpha"], 0.9, 0.99
            second, direction = np.full(D_DIM, hp["epsilon_v"] ** 2), np.zeros(D_DIM)
            for t in range(T):
                xs.append(x.copy())
                local = np.tile(x, (N, 1))
                m, v = np.zeros_like(local), np.tile(second, (N, 1))
                vhat = v.copy()
                for _ in range(K):
                    g = grads(local)
                    m = b1 * m + (1 - b1) * g
                    v = b2 * v + (1 - b2) * g * g
                    vhat = np.maximum(vhat, v)
                    local = local - lr * (a * m / np.sqrt(vhat) + (1 - a) * direction)
                mean = average(local)
                x_new = x + hp["server_eta"] * (mean - x)
                direction = (x - mean) / (lr * K)
                second = average(vhat)
                x = x_new
        else:
            lr, a, rho, b2 = hp["client_lr"], hp["alpha"], hp["rho"], 0.99
            g0 = grads(np.tile(x, (N, 1)))
            momentum = np.tile(average(g0), (N, 1))
            second = np.tile(average(g0 * g0), (N, 1))
            denom = np.sqrt(second[0]) + rho
            previous = np.tile(x, (N, 1))
            for t in range(1, T + 1):
                xs.append(x.copy())
                local = np.tile(x, (N, 1))
                if t == 1:
                    previous = local
                    local = local - lr * momentum
                for k in range(K):
                    g, g_old = grads(local), grads(previous)
                    momentum = g + (1 - a) * (momentum - g_old)
                    second = b2 * second + (1 - b2) * g * g
                    old = local
                    if k == K - 1:
                        mean_m, mean_v = average(momentum), average(second)
                        denom = np.sqrt(mean_v) + rho
                        local = np.tile(average(local - lr * mean_m / denom), (N, 1))
                        momentum, second = np.tile(mean_m, (N, 1)), np.tile(mean_v, (N, 1))
                    else:
                        local = local - lr * momentum / denom
                    previous = old
                x = local[0].copy()
        return xs

    def test_every_method_weights_the_clients_by_n_i_over_n(self):
        T, K, lam = 5, 3, 0.05
        P = data.pooled(data.load("ragged-test-unequal"), "nonconvex", lam)
        settings = {"fedadagrad": dict(c=0.5, server_eta=0.1, beta1=0.9, adapt_tau=1e-3),
                    "fedlada": dict(client_lr=0.5, alpha=0.1, epsilon_v=1e-8, server_eta=1.0),
                    "fafed": dict(client_lr=0.5, alpha=0.1, rho=0.01)}
        for method, hp in settings.items():
            cfg = self.cfg("ragged-test-unequal", method, T=T, K=K, lam=lam, **hp)
            got = self.go(cfg)
            want = self._reference(method, T, K, {**hp, "alpha": hp.get("alpha"), "rho": hp.get("rho", 0.01)}, lam)
            self.assertEqual(len(got), T)
            for t, x in enumerate(want):
                F, G = P.evaluate(torch.from_numpy(x))
                self.assertLessEqual(abs(float(F) - got[t]["objective"]), 1e-10 * float(F), (method, t))
                self.assertLessEqual(abs(float(G) - got[t]["grad_norm_sq"]), 1e-8 * max(float(G), 1e-12), (method, t))
            self.assertNotAlmostEqual(got[0]["objective"], got[-1]["objective"], places=6)

    def test_stochastic_minibatches_stay_inside_their_client(self):
        for algorithm in ("adafed", "fedlada", "fafed"):
            exact = self.go(self.cfg("ragged-test-const", algorithm))
            stoch = self.go(self.cfg("ragged-test-const", algorithm, mode="stochastic", b=2))
            self.assertSame(exact, stoch, tol=1e-10)

    def test_stochastic_runs_differ_between_seeds_and_agree_with_themselves(self):
        a = self.go(self.cfg("ragged-test-unequal", mode="stochastic", b=2))
        b = self.go(self.cfg("ragged-test-unequal", mode="stochastic", b=2))
        c = self.go(self.cfg("ragged-test-unequal", mode="stochastic", b=2, seed=8))
        self.assertEqual(a, b)
        self.assertNotEqual(a[-1]["objective"], c[-1]["objective"])

    def test_local_smoothness_of_unequal_clients(self):
        L = rate_study.local_smoothness("ragged-test-unequal", "smooth-convex")
        want = max(float(np.linalg.eigvalsh(b.T @ b)[-1]) / (4 * len(b)) for b in self.un) + 0.01
        self.assertAlmostEqual(L, want, places=12)


if __name__ == "__main__":
    unittest.main()
