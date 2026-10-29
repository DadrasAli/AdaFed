"""Heterogeneity of the IJCNN1 client splits: sizes, label skew, gradient dissimilarity, and whether the Dirichlet repair was used.

    python -m tools.heterogeneity            (from the repository root, PYTHONPATH=.)

Splits: the default ijcnn1-32 (32 clients of 1,000 rows), the Dirichlet splits ijcnn1-32-dirichlet<alpha>, alpha in 0.01, 0.05, 0.1,
0.5, 1, and the balanced k-means split ijcnn1-32-kmeans (32 clients of 1,000 rows; generate.kmeans_split_info) (datasets are built with
data.prepare when absent; it needs data/raw/ijcnn1.bz2; the k-means split is built from its tracked partition). With n_i the client sizes, n = 32,000,
p_i a client's positive fraction and p_bar the pooled positive fraction:
  size           min, median, max of n_i
  positives      min, median, max of the number of positive examples per client
  label skew     sum_i (n_i / n) |p_i - p_bar|, the unweighted mean of |p_i - p_bar|, and the number of single-class clients
  zeta^2         sum_i (n_i / n) ||grad l_i(x) - grad l(x)||^2, l_i client i's mean logistic data loss (no penalty), l the pooled
                 data loss (the n_i / n-weighted mean of the l_i); at x = 0 and at the minimisers x* of the smooth convex (l2sq) and
                 nonsmooth convex (l1) problems (lambda 0.01), each from the library's own solver (problem.solve_l2sq, solve_l1) on
                 the split's own pooled rows. The pooled objective is the same function on every split, so x* is the same
                 up to solver round-off; zeta^2 at x* of l1 uses that x* (a subgradient point: the data-loss gradients are exact).
  repair         whether the Dirichlet construction's repair was used (no draw within the 1,000 redraws left every client
                 non-empty), the number of examples it moved, and the number of draws made
A last row, "upper bound", holds at each of the three points the largest zeta^2 any split can have: the total spread
(1 / n) sum_j ||g_j - g_bar||^2 of the per-example data-loss gradients g_j (g_bar their mean), at the default split's x*; for any split
zeta^2 is the between-client part of that spread, so it cannot exceed it.
Prints one table and writes tools/heterogeneity-ijcnn1.csv.

    python -m tools.heterogeneity --synthetic

measures the default and the k-means split of each synthetic dataset (32 clients of 64 rows) the same way, with one upper-bound
row per dataset (at the default split's x*), and writes tools/heterogeneity-synthetic.csv.
"""

import csv
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from adafed import data, generate, problem

ALPHAS = (0.01, 0.05, 0.1, 0.5, 1.0)
LAM = 0.01
OUT = Path(__file__).resolve().parent / "heterogeneity-ijcnn1.csv"
SYNTHETIC_OUT = Path(__file__).resolve().parent / "heterogeneity-synthetic.csv"
COLUMNS = ("split", "clients", "size_min", "size_median", "size_max", "positives_min", "positives_median", "positives_max",
           "label_skew_weighted", "label_skew_unweighted",
           "single_class_clients", "zeta2_x0", "zeta2_xstar_l2sq", "zeta2_xstar_l1", "repair_used", "repair_moved", "draws")


def split_names():
    return [generate.DIRICHLET_BASE] + [generate.dirichlet_name(a) for a in ALPHAS] + [generate.KMEANS_NAME]


def rows_and_sizes(name):
    """Signed rows B = -y a (n, d) float64, labels y (n,), client sizes, in client order."""
    D = data.load(name)
    B, y, off = D["B"], np.asarray(D["y"]), D.get("offsets")
    if off is None:
        N, n_i = B.shape[:2]
        return torch.from_numpy(np.ascontiguousarray(B.reshape(N * n_i, -1))), y.reshape(-1), np.full(N, n_i)
    return torch.from_numpy(np.ascontiguousarray(B)), y, np.diff(off)


def zeta2(B, sizes, x):
    """sum_i (n_i / n) ||grad l_i(x) - grad l(x)||^2 for the mean logistic data losses."""
    s = torch.sigmoid(B @ x)
    G, lo = [], 0
    for n_i in sizes:
        G.append((B[lo:lo + n_i].T @ s[lo:lo + n_i]) / n_i)
        lo += n_i
    G = torch.stack(G)
    w = torch.from_numpy(sizes / sizes.sum()).to(G.dtype)
    g = (w[:, None] * G).sum(0)
    return float((w * ((G - g) ** 2).sum(1)).sum())


def spread(B, x):
    """(1 / n) sum_j ||g_j - g_bar||^2, g_j = grad of row j's logistic data loss at x: the upper bound of zeta^2 at x."""
    G = torch.sigmoid(B @ x)[:, None] * B
    return float(((G - G.mean(0)) ** 2).sum(1).mean())


def minimisers(name):
    D = data.load(name)
    xs = {}
    for pen in ("l2sq", "l1"):
        P = data.pooled(D, pen, LAM)
        quiet = lambda s: None
        x, f, res = problem.solve_l2sq(P, log=quiet) if pen == "l2sq" else problem.solve_l1(P, D["meta"]["L"], log=quiet)
        xs[pen] = x.detach().clone()
    return xs


def measure(name, repair):
    B, y, sizes = rows_and_sizes(name)
    n = int(sizes.sum())
    assert len(y) == n, "every row is in exactly one client"
    npos = np.array([int((y[lo:hi] > 0).sum()) for lo, hi in zip(np.cumsum(sizes) - sizes, np.cumsum(sizes))])
    pos = npos / sizes
    p_bar = float((y > 0).mean())
    d = np.abs(pos - p_bar)
    xs = minimisers(name)
    zero = torch.zeros(B.shape[1], dtype=B.dtype)
    used, moved, draws = repair
    return {"split": name, "clients": len(sizes), "size_min": int(sizes.min()), "size_median": float(statistics.median(sizes.tolist())),
            "size_max": int(sizes.max()), "positives_min": int(npos.min()), "positives_median": float(statistics.median(npos.tolist())),
            "positives_max": int(npos.max()), "label_skew_weighted": float((sizes / n * d).sum()), "label_skew_unweighted": float(d.mean()),
            "single_class_clients": int(((pos == 0) | (pos == 1)).sum()), "zeta2_x0": zeta2(B, sizes, zero),
            "zeta2_xstar_l2sq": zeta2(B, sizes, xs["l2sq"]), "zeta2_xstar_l1": zeta2(B, sizes, xs["l1"]),
            "repair_used": bool(used), "repair_moved": int(moved), "draws": int(draws)}


def main():
    g = generate.LIBSVM[generate.DIRICHLET_BASE]
    _, y = generate.read_libsvm(generate.fetch(generate.DIRICHLET_BASE, data.RAW, log=lambda s: None), g["dim"], g["rows"])
    repairs = {generate.DIRICHLET_BASE: (False, 0, 0), generate.KMEANS_NAME: (False, 0, 0)}
    for a in ALPHAS:
        _, _, info = generate.dirichlet_split_info(y.numpy(), g["clients"], a)
        repairs[generate.dirichlet_name(a)] = (info["repaired"], info["moved"], info["draws"])
    table = []
    for name in split_names():
        if not (data.DATA / f"{name}.npz").exists():
            data.prepare(name, log=lambda s: None)
        table.append(measure(name, repairs[name]))
    table.append(upper_bound(generate.DIRICHLET_BASE, "upper bound"))
    return _report(table, OUT)


def upper_bound(base, label):
    """The upper-bound row: the total spread of the per-example gradients at x = 0 and at the x* of `base`'s default split."""
    B, _, _ = rows_and_sizes(base)
    xs = minimisers(base)
    bound = {k: "" for k in COLUMNS}
    bound.update(split=label, zeta2_x0=spread(B, torch.zeros(B.shape[1], dtype=B.dtype)),
                 zeta2_xstar_l2sq=spread(B, xs["l2sq"]), zeta2_xstar_l1=spread(B, xs["l1"]))
    return bound


def synthetic_main():
    """The default and the k-means split of each synthetic dataset, each dataset followed by its upper bound."""
    table = []
    for base in sorted(generate.SYNTHETIC):
        for name in (base, generate.kmeans_name(base)):
            if not (data.DATA / f"{name}.npz").exists():
                data.prepare(name, log=lambda s: None)
            table.append(measure(name, (False, 0, 0)))
        table.append(upper_bound(base, f"upper bound ({base})"))
    return _report(table, SYNTHETIC_OUT)


def _report(table, out):
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in table:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    head = ["split", "clients", "min", "median", "max", "pos_min", "pos_med", "pos_max", "skew_w", "skew_u", "single", "zeta2(0)", "zeta2(x*l2sq)", "zeta2(x*l1)", "repair", "moved", "draws"]
    print(" ".join(f"{h:>14}" for h in head))
    for r in table:
        if r["split"].startswith("upper bound"):
            print(" ".join(f"{v:>14}" for v in [r["split"]] + [""] * 10 + [f"{r[k]:.4e}" for k in ("zeta2_x0", "zeta2_xstar_l2sq", "zeta2_xstar_l1")] + [""] * 3))
            continue
        print(" ".join(f"{v:>14}" for v in (r["split"], r["clients"], r["size_min"], f"{r['size_median']:g}", r["size_max"],
                                           r["positives_min"], f"{r['positives_median']:g}", r["positives_max"],
                                           f"{r['label_skew_weighted']:.4f}", f"{r['label_skew_unweighted']:.4f}", r["single_class_clients"],
                                           f"{r['zeta2_x0']:.4e}", f"{r['zeta2_xstar_l2sq']:.4e}", f"{r['zeta2_xstar_l1']:.4e}",
                                           "yes" if r["repair_used"] else "no", r["repair_moved"], r["draws"])))
    return table


if __name__ == "__main__":
    synthetic_main() if "--synthetic" in sys.argv[1:] else main()
