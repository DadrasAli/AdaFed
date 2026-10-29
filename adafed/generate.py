"""The datasets, built from scratch: controlled synthetic designs and LIBSVM data.

Synthetic (no RNG anywhere):
  design   a_ij = Phi^-1(vdc(i, p_j)), the Halton sequence (i = 1..n, p_j the j-th prime)
           through the normal quantile
  truth    `sparsity` non-zeros at coordinates (o * d) // s, value scale * (-1/2)^o
  labels   b_i = +1 if sigmoid(a_i . x_true) > vdc(i, p_d) else -1, with p_d the first
           prime the design does not use
  clients  rows sorted by the margin a_i . x_true (stable), dealt in blocks of `block`
           rows round robin: client c holds blocks c, c + N, c + 2N, ...

IJCNN1 is downloaded once from the public LIBSVM binary page and pinned by
SHA-256. Every objective uses the same fixed client partition. The default split "ijcnn1-32" sorts the rows by their margin
at the l1 reference solution (a floating-point sort key, which another CPU or BLAS can order differently) and deals them in
blocks; the partition it gives is tracked in data/ijcnn1-32-partition.json and `build` loads it (checking its SHA-256)
instead of sorting. `regenerate_default` reruns the construction (`construct_default_partition`) and reports whether it
reproduces the tracked partition.

A Dirichlet label-skew variant of IJCNN1 ("ijcnn1-32-dirichlet<alpha>", see `dirichlet_split`) has
the same rows and the same number of clients but unequal client sizes; it is stored as flat rows
plus client offsets. Its objective is the same as every split's, the mean over all rows; the server
weights client i by n_i / n. The construction redraws a draw that leaves a client empty at most 1,000
times; for small alpha (0.01, which the figures use, and 0.05, which is only measured:
`DIRICHLET_SMALL_ALPHAS`) no draw is complete and the first draw is repaired by moving single examples from the largest
client to each empty client (see `dirichlet_split_info`).

The maximum-heterogeneity split "ijcnn1-32-kmeans" (see `kmeans_split_info`) has the same rows in 32 clients of exactly 1,000
rows: a balanced k-means of the per-example data-loss gradients at the l2sq minimiser x*, which makes the client gradients at
x* as different as the method finds (zeta^2 at x* is the between-cluster spread, the k-means objective its complement).
The partition is tracked in data/ijcnn1-32-kmeans-partition.json and `build` loads it (checking its SHA-256) instead of
rerunning k-means; `regenerate_kmeans` reruns the construction and reports whether it reproduces the tracked partition.
Every synthetic dataset has its k-means split the same way ("synthetic-kappa<k>-kmeans": the same Halton rows in 32 clients
of exactly 64, by the same construction; tracked in data/synthetic-kappa<k>-kmeans-partition.json).
"""

import bz2
import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch

from . import problem

DTYPE = torch.float64

SYNTHETIC = {
    # Exact pooled Gram condition numbers.  Keeping lambda_max(A^T A/n)=1 makes
    # 1/L comparable across the three designs while only the conditioning changes.
    "synthetic-kappa1": dict(clients=32, rows_per_client=64, dim=32, sparsity=3,
                              signal_scale=2.0, block=32, condition=1.0),
    "synthetic-kappa10": dict(clients=32, rows_per_client=64, dim=32, sparsity=3,
                               signal_scale=2.0, block=32, condition=10.0),
    "synthetic-kappa100": dict(clients=32, rows_per_client=64, dim=32, sparsity=3,
                                signal_scale=2.0, block=32, condition=100.0),
}

LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM = {
    # IJCNN1 is large enough to be a meaningful real-data check and low-dimensional
    # enough that the K=100 exact-gradient experiment remains feasible on a workstation.
    "ijcnn1-32": dict(file="ijcnn1.bz2",
                      sha256="450dee8470b168e47b9fae1fc160396b8a809c675737837fce3acd51dcdc0481",
                      dim=22, rows=32000, clients=32, rows_per_client=1000, block=50,
                      row_normalize=False, reference_lambda=0.01, exact_key=False),
}

DATASETS = sorted(SYNTHETIC) + sorted(LIBSVM)

# Not part of DATASETS (the frozen protocol lists exactly the datasets above).
DIRICHLET_BASE = "ijcnn1-32"
DIRICHLET_PREFIX = DIRICHLET_BASE + "-dirichlet"
SPLIT_SEED = 2021          # seed of the Dirichlet client split (fixed; part of the protocol lock)
DIRICHLET_ALPHAS = (0.01, 0.1, 0.5, 1.0)   # the splits of the pipeline (CLI, protocol lock, figures)
DIRICHLET_SMALL_ALPHAS = (0.01, 0.05)      # the alphas whose draws are repaired; 0.05 is measured (tools/heterogeneity.py) but not used
DIRICHLET_MAX_REDRAWS = 1000
KMEANS_SUFFIX = "-kmeans"
KMEANS_BASES = (DIRICHLET_BASE,) + tuple(sorted(SYNTHETIC))   # the datasets with a balanced k-means split
DEFAULT_PARTITION = Path(__file__).resolve().parent.parent / "data" / (DIRICHLET_BASE + "-partition.json")   # tracked
KMEANS_LAMBDA = 0.01                       # x* is the l2sq minimiser of the pooled objective at this lambda
KMEANS_RESTARTS = 10                       # k-means++ restarts, all from one generator seeded with SPLIT_SEED
KMEANS_MAX_ITERATIONS = 1000


def kmeans_name(base):
    """The k-means split of `base` (IJCNN1 or a synthetic dataset): <base>-kmeans."""
    if base not in KMEANS_BASES:
        raise ValueError(f"no k-means split of {base!r}; one of {', '.join(KMEANS_BASES)}")
    return base + KMEANS_SUFFIX


def kmeans_base(name):
    """The dataset a k-means split name splits, None for any other name."""
    base = name[:-len(KMEANS_SUFFIX)] if name.endswith(KMEANS_SUFFIX) else None
    return base if base in KMEANS_BASES else None


def kmeans_partition_path(base):
    """data/<base>-kmeans-partition.json, the tracked partition of the k-means split of `base`."""
    return Path(__file__).resolve().parent.parent / "data" / (kmeans_name(base) + "-partition.json")


KMEANS_NAME = kmeans_name(DIRICHLET_BASE)
KMEANS_PARTITION = kmeans_partition_path(DIRICHLET_BASE)   # tracked


def split_dataset(base, split):
    """`base` (a paper dataset) under a client split: `default` is `base` itself, `kmeans` its balanced k-means split, and
    `dirichlet<alpha>` the Dirichlet label-skew split of IJCNN1 (IJCNN1 only)."""
    if split == "default":
        return base
    if split == "kmeans":
        return kmeans_name(base)
    if split.startswith("dirichlet") and base == DIRICHLET_BASE:
        return dirichlet_name(float(split[len("dirichlet"):]))
    raise ValueError(f"no split {split!r} of {base!r}")


def dirichlet_name(alpha):
    return f"{DIRICHLET_PREFIX}{float(alpha):g}"


def dirichlet_alpha(name):
    """alpha of a Dirichlet dataset name, None for any other name."""
    if not name.startswith(DIRICHLET_PREFIX):
        return None
    alpha = float(name[len(DIRICHLET_PREFIX):])
    if dirichlet_name(alpha) != name or not alpha > 0:
        raise ValueError(f"{name!r}: expected {DIRICHLET_PREFIX}<alpha> with alpha > 0, e.g. {dirichlet_name(0.5)}")
    return alpha


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

def primes(count):
    found, candidate = [], 2
    while len(found) < count:
        if all(candidate % p for p in found):
            found.append(candidate)
        candidate += 1
    return found


def van_der_corput(count, base):
    """vdc(i, base) for i = 1..count, digits accumulated lowest first."""
    index = torch.arange(1, count + 1, dtype=torch.int64)
    value = torch.zeros(count, dtype=DTYPE)
    denominator = 1.0
    while bool(index.any()):
        denominator *= base
        value += (index % base).to(DTYPE) / denominator
        index = index // base
    return value


def halton_normal_design(rows, dim):
    return torch.special.ndtri(torch.stack([van_der_corput(rows, p) for p in primes(dim)], dim=1))


def truth(dim, sparsity, signal_scale):
    x = torch.zeros(dim, dtype=DTYPE)
    for o in range(sparsity):
        x[(o * dim) // sparsity] = signal_scale * (-1.0) ** o * 0.5 ** o
    return x


def deal(key, clients, rows_per_client, block):
    """Rows sorted by `key` (stable), dealt in blocks round robin: (clients, rows_per_client)."""
    if rows_per_client % block:
        raise ValueError(f"block {block} must divide rows_per_client {rows_per_client}")
    order = torch.argsort(key, stable=True)
    return torch.stack([
        torch.cat([order[(g * clients + c) * block:(g * clients + c + 1) * block]
                   for g in range(rows_per_client // block)])
        for c in range(clients)])


def synthetic_design(name):
    """(A, labels, margin, x_true) of a synthetic dataset: its rows in Halton order, before any client split."""
    s = SYNTHETIC[name]
    n = s["clients"] * s["rows_per_client"]
    A = halton_normal_design(n, s["dim"])
    if "condition" in s:
        # Deterministically orthogonalise the centred Halton columns.  The scale
        # squares are geometric from 1 to 1/kappa, hence cond(A^T A/n)=kappa.
        Q, _ = torch.linalg.qr(A - A.mean(0), mode="reduced")
        scales = torch.logspace(0.0, -0.5 * math.log10(s["condition"]), s["dim"], dtype=DTYPE)
        A = math.sqrt(n) * Q * scales
    x_true = truth(s["dim"], s["sparsity"], s["signal_scale"])
    margin = A @ x_true
    labels = torch.where(torch.sigmoid(margin) > van_der_corput(n, primes(s["dim"] + 1)[s["dim"]]), 1.0, -1.0).to(DTYPE)
    return A, labels, margin, x_true


def _synthetic_meta(name, A, x_true):
    s = SYNTHETIC[name]
    n = len(A)
    gram = (A.T @ A) / n
    eig = torch.linalg.eigvalsh(gram)
    return {"kind": "synthetic", "generator": dict(s), "rows": n, "x_true": x_true.tolist(),
            "gram_condition": float(eig[-1] / eig[0]),
            "label_base": primes(s["dim"] + 1)[s["dim"]], "partition": "row indices of the Halton design"}


def build_synthetic(name):
    s = SYNTHETIC[name]
    A, labels, margin, x_true = synthetic_design(name)
    part = deal(margin, s["clients"], s["rows_per_client"], s["block"])
    meta = _synthetic_meta(name, A, x_true)
    return A[part].numpy(), labels[part].numpy(), part.numpy(), meta


# ---------------------------------------------------------------------------
# LIBSVM (IJCNN1)
# ---------------------------------------------------------------------------

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, raw_dir, log=print):
    """data/raw/<file>, downloaded once from the public LIBSVM page; its SHA-256 must match."""
    g = LIBSVM[name]
    path = Path(raw_dir) / g["file"]
    url = LIBSVM_URL + g["file"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading {url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    found = sha256_of(path)
    if found != g["sha256"]:
        raise ValueError(f"{path}: SHA-256 {found}, expected {g['sha256']}; delete it and re-run to re-fetch")
    return path


def read_libsvm(path, dim, rows):
    """The first `rows` rows, dense float64."""
    labels, row_ids, cols, vals = [], [], [], []
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rt", encoding="ascii") as f:
        for line in f:
            if len(labels) == rows:
                break
            fields = line.split()
            label = float(fields[0])
            if label not in (1.0, -1.0):
                raise ValueError(f"{path}:{len(labels) + 1}: label {label:g}")
            for field in fields[1:]:
                k, _, v = field.partition(":")
                row_ids.append(len(labels))
                cols.append(int(k) - 1)
                vals.append(float(v))
            labels.append(label)
    if len(labels) < rows:
        raise ValueError(f"{path} holds {len(labels)} rows, {rows} needed")
    A = torch.zeros((rows, dim), dtype=DTYPE)
    A[torch.tensor(row_ids), torch.tensor(cols)] = torch.tensor(vals, dtype=DTYPE)
    return A, torch.tensor(labels, dtype=DTYPE)


def exact_margins(A, w, digits=10):
    """a_i . w with w rounded to `digits` significant digits and each row summed exactly
    (math.fsum): the same key on any machine, whatever BLAS does with the order."""
    w = [float(f"{v:.{digits - 1}e}") for v in w.tolist()]
    A = A.numpy()
    return torch.tensor([math.fsum(A[i, j] * w[j] for j in np.flatnonzero(A[i])) for i in range(A.shape[0])],
                        dtype=DTYPE)


def construct_default_partition(name, A, y, log=print):
    """The default split's construction: solve the l1 reference problem, sort the rows by their margin at its solution
    (stable) and deal them in blocks round robin. Returns (partition (clients, rows_per_client), lambda_max, KKT residual).
    The sort key is floating-point (`exact_key` False), so another CPU or BLAS can order it differently: `build` does not
    call this but loads the tracked partition."""
    g = LIBSVM[name]
    lam_max = float((A.T @ y).abs().max()) / (2 * g["rows"])
    log(f"{name}: {g['rows']} rows x {g['dim']}, lambda_max {lam_max:.6g}; solving l1 at lambda "
        f"{g['reference_lambda']:g} ({g['reference_lambda'] / lam_max:.3f} lambda_max) for the partition")
    P = problem.Pooled(A * (-y)[:, None], "l1", g["reference_lambda"])
    w_ref, _, res = problem.solve_l1(P, problem.lipschitz(A), log=lambda s: None)
    log(f"{name}: reference solve KKT {res:.2e}, {int((w_ref != 0).sum())} non-zeros")
    key = exact_margins(A, w_ref) if g["exact_key"] else A @ w_ref
    return deal(key, g["clients"], g["rows_per_client"], g["block"]).numpy(), lam_max, res


def default_partition_text(part, lam_max, res):
    """The tracked file's text: the settings, the digest, the build's two reference numbers and the partition, one client
    per line."""
    g = LIBSVM[DIRICHLET_BASE]
    head = {"dataset": DIRICHLET_BASE, "clients": g["clients"], "rows": g["rows"], "rows_per_client": g["rows_per_client"],
            "sha256": partition_digest(part),
            "sha256_of": "the partition as int64 values, row-major: clients in order, the rows of a client in the order "
                         "they are dealt",
            "construction": "rows stably sorted by a_i . w_ref, w_ref the l1 minimiser of the pooled objective at "
                            "reference_lambda (problem.solve_l1), dealt in blocks of `block` rows round robin "
                            "(generate.deal)",
            "reference_lambda": g["reference_lambda"], "block": g["block"], "exact_key": g["exact_key"],
            "lambda_max": lam_max, "reference_kkt": res}
    lines = [f"{json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    rows = ",\n".join("  " + json.dumps([int(i) for i in client], separators=(",", ":")) for client in part)
    return "{\n" + "\n".join(lines) + '\n"partition": [\n' + rows + "\n]\n}\n"


def read_default_partition(path=None):
    """(part, document): the tracked partition of the default split (clients, rows_per_client) int64; its SHA-256 must
    match the file's, it must hold every one of the rows exactly once and have the clients' sizes of ijcnn1-32."""
    return _read_partition(DEFAULT_PARTITION if path is None else path)


def regenerate_default(raw_dir, log=print, path=None):
    """Rerun the default split's construction on the IJCNN1 rows and compare with the tracked partition: a dict with `equal`
    (the regenerated partition is the tracked one), `sha256` (of the regenerated one), `tracked_sha256`, `partition`,
    `lambda_max` and `reference_kkt`. Takes seconds. Nothing is written."""
    g = LIBSVM[DIRICHLET_BASE]
    A, y = read_libsvm(fetch(DIRICHLET_BASE, raw_dir, log), g["dim"], g["rows"])
    if g["row_normalize"]:
        A = A / torch.linalg.vector_norm(A, dim=1, keepdim=True)
    part, lam_max, res = construct_default_partition(DIRICHLET_BASE, A, y, log)
    tracked, doc = read_default_partition(path)
    return {"equal": bool(np.array_equal(part, tracked)), "sha256": partition_digest(part), "tracked_sha256": doc["sha256"],
            "partition": part, "lambda_max": lam_max, "reference_kkt": res}


def build_libsvm(name, raw_dir, log=print):
    """The rows of a LIBSVM dataset in clients by its tracked partition (the sort is not rerun)."""
    g = LIBSVM[name]
    if name != DIRICHLET_BASE:
        raise ValueError(f"no tracked partition for {name!r}")
    path = fetch(name, raw_dir, log)
    A, y = read_libsvm(path, g["dim"], g["rows"])
    if g["row_normalize"]:
        A = A / torch.linalg.vector_norm(A, dim=1, keepdim=True)
    part, doc = read_default_partition()
    log(f"{name}: tracked partition {doc['sha256'][:12]}")
    meta = {"kind": "libsvm", "builder": {**g, "url": LIBSVM_URL + g["file"]}, "rows": g["rows"],
            "lambda_max": doc["lambda_max"], "reference_kkt": doc["reference_kkt"],
            "partition": "row indices of the LIBSVM file"}
    sel = torch.from_numpy(part)
    return A[sel].numpy(), y[sel].numpy(), part, meta


def dirichlet_split_info(labels, clients, alpha, seed=SPLIT_SEED, max_redraws=DIRICHLET_MAX_REDRAWS):
    """(order, offsets, info): client i holds rows order[offsets[i]:offsets[i + 1]] (ascending row index).

    The label-skew construction of Hsu et al. (2019): for every class its rows are shuffled and cut across the
    clients by proportions q ~ Dirichlet(alpha * 1_clients); a draw that leaves a client empty is redrawn from the
    same generator, at most `max_redraws` times (so at most 1 + max_redraws draws). If no draw leaves every client
    non-empty, the first draw is taken and repaired: while a client is empty (lowest index first), one example,
    chosen by the same generator (it continues from where the last draw left it), moves to it from the currently
    largest client (lowest index among equals). Every row is held exactly once.

    info: {"draws": draws made, "repaired": bool, "moved": examples moved, "draw_sizes": client sizes of the draw
    taken before the repair, "moves": [(to_client, from_client, row index), ...] in order}.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    first = None
    for draw in range(1, max_redraws + 2):
        parts = [[] for _ in range(clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            q = rng.dirichlet(np.full(clients, alpha))
            cuts = (np.cumsum(q)[:-1] * len(idx)).astype(np.int64)
            for i, piece in enumerate(np.split(idx, cuts)):
                parts[i].append(piece)
        parts = [np.concatenate(ps) for ps in parts]
        if first is None:
            first = parts
        sizes = np.array([len(p) for p in parts])
        if sizes.min() > 0:
            break
    info = {"draws": draw, "repaired": False, "moved": 0, "draw_sizes": None, "moves": []}
    if sizes.min() == 0:
        parts = [p.copy() for p in first]
        info.update(repaired=True, draw_sizes=[len(p) for p in parts])
        while min(len(p) for p in parts) == 0:
            to = next(i for i, p in enumerate(parts) if len(p) == 0)
            src = int(np.argmax([len(p) for p in parts]))
            k = int(rng.integers(len(parts[src])))
            row = int(parts[src][k])
            parts[src] = np.delete(parts[src], k)
            parts[to] = np.array([row], dtype=np.int64)
            info["moves"].append((to, src, row))
        info["moved"] = len(info["moves"])
    sizes = np.array([len(p) for p in parts])
    order = np.concatenate([np.sort(p) for p in parts]).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    if int(offsets[-1]) != len(labels) or not np.array_equal(np.sort(order), np.arange(len(labels))):
        raise AssertionError("a Dirichlet split must hold every row exactly once")
    return order, offsets, info


def dirichlet_split(labels, clients, alpha, seed=SPLIT_SEED, max_redraws=DIRICHLET_MAX_REDRAWS):
    """(order, offsets) of dirichlet_split_info."""
    return dirichlet_split_info(labels, clients, alpha, seed, max_redraws)[:2]


def build_dirichlet(name, raw_dir, log=print):
    """Flat rows (M, d) in client order, labels, row indices, and meta with the client offsets."""
    alpha = dirichlet_alpha(name)
    g = LIBSVM[DIRICHLET_BASE]
    path = fetch(DIRICHLET_BASE, raw_dir, log)
    A, y = read_libsvm(path, g["dim"], g["rows"])
    order, offsets = dirichlet_split(y.numpy(), g["clients"], alpha)
    assert int(offsets[-1]) == g["rows"] == len(y), "every split uses the same rows"
    log(f"{name}: {g['clients']} clients, sizes {int(np.diff(offsets).min())}..{int(np.diff(offsets).max())}")
    meta = {"kind": "libsvm-dirichlet", "builder": {**g, "url": LIBSVM_URL + g["file"], "alpha": alpha,
                                                      "split_seed": SPLIT_SEED},
            "rows": g["rows"], "offsets": offsets.tolist(),
            "partition": "row indices of the LIBSVM file, clients in order, ascending within a client"}
    sel = torch.from_numpy(order)
    return A[sel].numpy(), y[sel].numpy(), order, meta


def _negative_cycle(w, tol):
    """(cycle, d) for the complete digraph on K nodes with arc weights w (K, K) (diagonal ignored): Bellman-Ford from a
    virtual source joined to every node at weight 0, a relaxation counting only when it lowers a distance by more than
    tol. cycle is a list of nodes [k_0, ..., k_m] (arcs k_i -> k_i+1 and k_m -> k_0) of total weight below 0, or None
    when no relaxation is left, d then holding the distances: d[v] <= d[u] + w[u, v] + tol for every arc."""
    K = len(w)
    w = w.copy()
    np.fill_diagonal(w, np.inf)
    d = np.zeros(K)
    pred = np.full(K, -1)
    for _ in range(K + 1):
        cand = d[:, None] + w
        best = cand.argmin(0)
        new = cand[best, np.arange(K)]
        upd = new < d - tol
        if not upd.any():
            return None, d
        d = np.where(upd, new, d)
        pred = np.where(upd, best, pred)
    for v in range(K):                     # still relaxing after K + 1 passes: the predecessor graph has a cycle
        seen = {}
        while v >= 0 and v not in seen:
            seen[v] = len(seen)
            v = int(pred[v])
        if v >= 0:
            path = list(seen)
            return path[seen[v]:][::-1], d
    raise AssertionError("Bellman-Ford still relaxing without a predecessor cycle")


def balanced_assignment(C, size, start, tol=0.0):
    """The assignment of n rows to K clients, every client exactly `size` rows (n = K * size), of least total cost
    sum_j C[j, a_j]: the transportation problem, solved exactly by cancelling negative cycles from the balanced
    assignment `start`.

    With the costs fixed, moving rows client to client along a cycle of clients keeps every client's size, and the
    assignment is optimal if and only if no such cycle lowers the cost (the residual network of the transportation
    problem has no negative cycle). Arc k -> k' weighs the cheapest move of one row of k to k', min_{j in k} C[j, k'] -
    C[j, k]; a cycle below -tol is cancelled by moving, along every arc, the t cheapest rows of its source client,
    t chosen to lower the total most (the arcs leave distinct clients, so their moves are independent). The result is
    certified: with d the final Bellman-Ford distances, every row's client minimises C[j, k] - d[k] up to tol, so by
    LP duality no balanced assignment costs less than this one minus n * tol. Returns (a, d).
    """
    C = np.asarray(C, dtype=np.float64)
    n, K = C.shape
    a = np.array(start, dtype=np.int64)
    if n != K * size or not np.array_equal(np.bincount(a, minlength=K), np.full(K, size)):
        raise ValueError(f"start must put exactly {size} of the {n} rows in each of the {K} clients")
    rows = np.arange(n)
    while True:
        order = np.argsort(a, kind="stable").reshape(K, size)          # order[k]: the rows of client k
        D = (C - C[rows, a][:, None])[order]                           # (K, size, K): cost change of moving a row
        cycle, d = _negative_cycle(D.min(1), tol)
        if cycle is None:
            break
        arcs = list(zip(cycle, cycle[1:] + cycle[:1]))
        picks, gain = [], np.zeros(size)
        for k, k2 in arcs:
            o = np.argsort(D[k, :, k2], kind="stable")
            picks.append(order[k][o])
            gain += np.cumsum(D[k, o, k2])
        t = int(gain.argmin()) + 1
        if not gain[t - 1] < -tol:
            raise AssertionError("a negative cycle that lowers nothing")
        for (k, k2), p in zip(arcs, picks):
            a[p[:t]] = k2
    r = C - d[None, :]
    if not (r[rows, a] <= r.min(1) + 2 * tol + 1e-15 * np.abs(r).max()).all():
        raise AssertionError("the balanced assignment is not certified optimal")
    return a, d


def kmeans_costs(G, centres):
    """C[j, k] = ||G[j] - centres[k]||^2."""
    return ((G * G).sum(1)[:, None] - 2.0 * (G @ centres.T) + (centres * centres).sum(1)[None, :]).clip(min=0.0)


def balanced_kmeans(G, clients, size, rng, max_iterations=KMEANS_MAX_ITERATIONS):
    """(a, centres, iterations): balanced k-means of the rows of G into `clients` clusters of exactly `size` rows.

    k-means++ initialisation (the first centre uniform, each next one a row drawn with probability proportional to
    its squared distance to the nearest centre so far); a first balanced assignment by giving each row, in an order
    drawn from rng, its nearest centre with room left; then alternately the exact balanced assignment step
    (balanced_assignment, warm-started from the previous assignment) and the centre step (cluster means), until the
    assignment no longer changes.
    """
    n = len(G)
    picks = [int(rng.integers(n))]
    near = ((G - G[picks[0]]) ** 2).sum(1)
    for _ in range(1, clients):
        j = int(rng.choice(n, p=near / near.sum()))
        picks.append(j)
        near = np.minimum(near, ((G - G[j]) ** 2).sum(1))
    centres = G[picks].copy()
    C = kmeans_costs(G, centres)
    room = np.full(clients, size)
    a = np.empty(n, dtype=np.int64)
    for j in rng.permutation(n):
        k = int(np.where(room > 0, C[j], np.inf).argmin())
        a[j] = k
        room[k] -= 1
    previous = None
    for iteration in range(1, max_iterations + 1):
        C = kmeans_costs(G, centres)
        a, _ = balanced_assignment(C, size, a, tol=1e-12 * float(C.max()))
        if previous is not None and np.array_equal(a, previous):
            return a, centres, iteration
        previous = a
        centres = np.stack([G[a == k].mean(0) for k in range(clients)])
    raise RuntimeError(f"balanced k-means did not settle in {max_iterations} iterations")


def kmeans_split_info(G, clients, restarts=KMEANS_RESTARTS, seed=SPLIT_SEED):
    """(part, info): part (clients, n / clients) the row indices of each client, ascending within a client.

    balanced_kmeans of the rows of G, `restarts` times from one generator default_rng(seed); the partition kept is the
    one with the largest zeta^2 = (1 / clients) sum_k ||mean of client k's rows - mean of all rows||^2 (the first among
    equals). info: {"restarts": [{"zeta2", "iterations"}, ...] in order, "best": index of the one kept}.
    """
    G = np.asarray(G, dtype=np.float64)
    n = len(G)
    if n % clients:
        raise ValueError(f"{n} rows do not split into {clients} equal clients")
    size = n // clients
    rng = np.random.default_rng(seed)
    g_bar = G.mean(0)
    runs, best, best_a = [], None, None
    for r in range(restarts):
        a, _, iterations = balanced_kmeans(G, clients, size, rng)
        means = np.stack([G[a == k].mean(0) for k in range(clients)])
        z = float(((means - g_bar) ** 2).sum(1).mean())
        runs.append({"zeta2": z, "iterations": iterations})
        if best is None or z > runs[best]["zeta2"]:
            best, best_a = r, a
    part = np.stack([np.flatnonzero(best_a == k) for k in range(clients)]).astype(np.int64)
    if not np.array_equal(np.sort(part.ravel()), np.arange(n)):
        raise AssertionError("a k-means split must hold every row exactly once")
    return part, {"restarts": runs, "best": best}


def kmeans_gradients(A, y, lam=KMEANS_LAMBDA):
    """(G, x*): G[j] = grad of row j's logistic data loss log(1 + exp(-y_j a_j . x)) at x*, the l2sq minimiser of the
    pooled objective at `lam` (problem.solve_l2sq on all rows; the penalty's gradient is the same for every row and is
    left out)."""
    B = A * (-y)[:, None]
    x, _, _ = problem.solve_l2sq(problem.Pooled(B, "l2sq", lam), log=lambda s: None)
    return (torch.sigmoid(B @ x)[:, None] * B).numpy(), x


def partition_digest(part):
    """SHA-256 of a partition as int64, row-major (clients in order, the rows of a client ascending)."""
    return hashlib.sha256(np.ascontiguousarray(part, dtype=np.int64).tobytes()).hexdigest()


def split_shape(base):
    """(clients, rows_per_client) of the paper's split of `base`, which every equal-client split of it keeps."""
    g = LIBSVM[base] if base in LIBSVM else SYNTHETIC[base]
    return g["clients"], g["rows_per_client"]


def kmeans_partition_text(part, info, base=DIRICHLET_BASE):
    """The tracked file's text: the settings, the digest, the restarts' zeta^2 and the partition, one client per line."""
    clients, rows_per_client = split_shape(base)
    runs = info["restarts"]
    head = {"dataset": kmeans_name(base), "clients": clients, "rows": clients * rows_per_client,
            "rows_per_client": rows_per_client,
            "sha256": partition_digest(part),
            "sha256_of": "the partition as int64 values, row-major: clients in order, the rows of a client ascending",
            "reference_point": "x*, the l2sq minimiser of the pooled objective (problem.solve_l2sq), at lambda",
            "lambda": KMEANS_LAMBDA, "split_seed": SPLIT_SEED, "restarts": len(runs), "kept_restart": info["best"],
            "zeta2": runs[info["best"]]["zeta2"], "restart_zeta2": [r["zeta2"] for r in runs],
            "restart_iterations": [r["iterations"] for r in runs]}
    lines = [f"{json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    rows = ",\n".join("  " + json.dumps([int(i) for i in client], separators=(",", ":")) for client in part)
    return "{\n" + "\n".join(lines) + '\n"partition": [\n' + rows + "\n]\n}\n"


def _read_partition(path, base=DIRICHLET_BASE):
    """(part, document) of a tracked partition file: (clients, rows_per_client) int64, every one of the rows exactly once,
    and the SHA-256 the file records."""
    clients, rows_per_client = split_shape(base)
    rows = clients * rows_per_client
    doc = json.loads(Path(path).read_text())
    part = np.array(doc["partition"], dtype=np.int64)
    if part.shape != (clients, rows_per_client):
        raise ValueError(f"{path}: partition of shape {part.shape}, expected {(clients, rows_per_client)}")
    if not np.array_equal(np.sort(part.ravel()), np.arange(rows)):
        raise ValueError(f"{path}: the partition must hold every one of the {rows} rows exactly once")
    if partition_digest(part) != doc["sha256"]:
        raise ValueError(f"{path}: the partition's SHA-256 is {partition_digest(part)}, the file records {doc['sha256']}")
    return part, doc


def read_kmeans_partition(path=None, base=DIRICHLET_BASE):
    """(part, document): the tracked partition of the k-means split of `base` (clients, rows_per_client) int64; its SHA-256
    must match the file's, it must hold every one of the rows exactly once and have the clients' sizes of `base`."""
    if path is None:
        path = KMEANS_PARTITION if base == DIRICHLET_BASE else kmeans_partition_path(base)
    return _read_partition(path, base)


def kmeans_rows(base, raw_dir, log=print):
    """(A, y): the rows of `base` before any client split (the first rows of the LIBSVM file, or the Halton design)."""
    if base in SYNTHETIC:
        A, y, _, _ = synthetic_design(base)
        return A, y
    g = LIBSVM[base]
    return read_libsvm(fetch(base, raw_dir, log), g["dim"], g["rows"])


def regenerate_kmeans(raw_dir, log=print, path=None, base=DIRICHLET_BASE):
    """Rerun the k-means construction on the rows of `base` and compare with the tracked partition: a dict with `equal`
    (the regenerated partition is the tracked one), `sha256` (of the regenerated one), `tracked_sha256` (None when no
    partition is tracked yet), `partition` and `info`. Takes about two minutes for IJCNN1. Nothing is written."""
    A, y = kmeans_rows(base, raw_dir, log)
    G, _ = kmeans_gradients(A, y)
    part, info = kmeans_split_info(G, split_shape(base)[0])
    tracked_path = path if path is not None else (KMEANS_PARTITION if base == DIRICHLET_BASE else kmeans_partition_path(base))
    if not Path(tracked_path).exists():
        return {"equal": False, "sha256": partition_digest(part), "tracked_sha256": None, "partition": part, "info": info}
    tracked, doc = read_kmeans_partition(path, base)
    return {"equal": bool(np.array_equal(part, tracked)), "sha256": partition_digest(part), "tracked_sha256": doc["sha256"],
            "partition": part, "info": info}


def build_kmeans(name, raw_dir, log=print):
    """The rows in the clients of the tracked partition, (N, n_i, d) like the paper's split of the base dataset (IJCNN1: 32
    clients of 1,000; synthetic: 32 of 64). K-means is not rerun."""
    base = kmeans_base(name)
    if base in SYNTHETIC:
        A, y, _, x_true = synthetic_design(base)
        part, doc = read_kmeans_partition(base=base)
        log(f"{name}: tracked partition {doc['sha256'][:12]}, restart {doc['kept_restart']} of {doc['restarts']}, "
            f"zeta^2 at x* {doc['zeta2']:.6g}")
        meta = _synthetic_meta(base, A, x_true)
        meta.update(kind="synthetic-kmeans", kmeans={"sha256": doc["sha256"], "kept_restart": doc["kept_restart"],
                                                     "zeta2": doc["zeta2"], "restart_zeta2": doc["restart_zeta2"],
                                                     "restart_iterations": doc["restart_iterations"], "lambda": doc["lambda"],
                                                     "restarts": doc["restarts"], "split_seed": doc["split_seed"]},
                    partition=f"row indices of the Halton design, ascending within a client (data/{name}-partition.json)")
        sel = torch.from_numpy(part)
        return A[sel].numpy(), y[sel].numpy(), part, meta
    g = LIBSVM[DIRICHLET_BASE]
    path = fetch(DIRICHLET_BASE, raw_dir, log)
    A, y = read_libsvm(path, g["dim"], g["rows"])
    part, doc = read_kmeans_partition()
    log(f"{name}: tracked partition {doc['sha256'][:12]}, restart {doc['kept_restart']} of {doc['restarts']}, "
        f"zeta^2 at x* {doc['zeta2']:.6g}")
    meta = {"kind": "libsvm-kmeans", "builder": {**g, "url": LIBSVM_URL + g["file"], "lambda": doc["lambda"],
                                                   "restarts": doc["restarts"], "split_seed": doc["split_seed"]},
            "rows": g["rows"], "kmeans": {"sha256": doc["sha256"], "kept_restart": doc["kept_restart"],
                                           "zeta2": doc["zeta2"], "restart_zeta2": doc["restart_zeta2"],
                                           "restart_iterations": doc["restart_iterations"]},
            "partition": "row indices of the LIBSVM file, ascending within a client (data/ijcnn1-32-kmeans-partition.json)"}
    sel = torch.from_numpy(part)
    return A[sel].numpy(), y[sel].numpy(), part, meta


def is_ijcnn1_split(name):
    """IJCNN1 itself or one of its client splits (a Dirichlet split or the k-means split)."""
    return name == DIRICHLET_BASE or name == KMEANS_NAME or dirichlet_alpha(name) is not None


def build(name, raw_dir, log=print):
    if kmeans_base(name) is not None:
        return build_kmeans(name, raw_dir, log)
    if dirichlet_alpha(name) is not None:
        return build_dirichlet(name, raw_dir, log)
    if name in SYNTHETIC:
        return build_synthetic(name)
    if name in LIBSVM:
        return build_libsvm(name, raw_dir, log)
    raise ValueError(f"unknown dataset {name!r}; known: {DATASETS}")
