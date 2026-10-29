"""Datasets, built by this project (adafed/generate.py) and cached as data/<name>.npz.

The npz holds X (N, n_i, d) and y (N, n_i) in client order, `partition` (N, n_i): each
client row's index in the source (the Halton design, or the LIBSVM file), and
`metadata`, a JSON string (the generator's settings and L = lambda_max(A^T A) / (4n)).

F* lives in data/fstar.json, keyed "<dataset>/<penalty>/<lam>", computed on first use.

A dataset with unequal clients (the Dirichlet splits of generate.py) is stored flat: X (M, d) and y (M,)
in client order, `offsets` (N + 1) with client i holding rows offsets[i]:offsets[i + 1]. load() then
returns 2-D "X"/"B" and "offsets". The objective is the same for every split: the mean over all n rows,
F(x) = sum_i (n_i / n) F_i(x) + penalty with F_i the client's mean loss, so F, F* and L are those of the
pooled rows whatever the partition. For equal clients "offsets" is None and nothing changes.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import generate, problem

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
FSTAR = DATA / "fstar.json"

_cache = {}


def lam_key(lam):
    return repr(float(lam))


def prepare(name, log=print):
    """Build data/<name>.npz from scratch (IJCNN1: from the public LIBSVM file)."""
    t0 = time.time()
    X, y, partition, meta = generate.build(name, RAW, log)
    if X.ndim == 2:                        # unequal clients: flat rows and offsets
        offsets = np.asarray(meta["offsets"], dtype=np.int64)
        N, d = len(offsets) - 1, X.shape[1]
        assert X.shape[0] == generate.LIBSVM[generate.DIRICHLET_BASE]["rows"], "every split uses the same rows"
        L = problem.lipschitz(torch.from_numpy(X))
        meta.update({"name": name, "N": N, "n_i": None, "d": d, "L": L})
        extra = {"offsets": offsets}
        shape = f"N {N}, sizes {int(np.diff(offsets).min())}..{int(np.diff(offsets).max())}"
    else:
        N, n_i, d = X.shape
        L = problem.lipschitz(torch.from_numpy(X.reshape(N * n_i, d)))
        meta.update({"name": name, "N": N, "n_i": n_i, "d": d, "L": L})
        extra = {}
        shape = f"N {N} x n_i {n_i}"
    DATA.mkdir(exist_ok=True)
    tmp = DATA / f"{name}.tmp{os.getpid()}.npz"
    np.savez(tmp, X=X, y=y, partition=partition, metadata=np.array(json.dumps(meta)), **extra)
    tmp.replace(DATA / f"{name}.npz")
    _cache.pop(name, None)
    log(f"{name}: {shape}, d {d}, L = {L!r} -> data/{name}.npz ({time.time() - t0:.1f}s)")


def load(name):
    """{'X', 'y', 'partition', 'B' (numpy), 'meta' (dict)}, cached per process.

    B = -y * X, the signed rows every computation uses (exact for +-1 labels).
    Loaded in a sweep's parent before it forks, so the workers share its pages.
    """
    if name not in _cache:
        path = DATA / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m adafed prepare {name}`")
        with np.load(path) as z:
            X, y = z["X"], z["y"]
            if not np.isin(y, (-1.0, 1.0)).all():
                raise ValueError(f"{path}: labels must be -1 or +1")
            offsets = z["offsets"] if "offsets" in z.files else None
            _cache[name] = {"X": X, "y": y, "partition": z["partition"], "B": X * (-y)[..., None],
                            "meta": json.loads(str(z["metadata"])), "offsets": offsets}
    return _cache[name]


def pooled(D, pen, lam, dtype=None):
    """problem.Pooled over all rows of a loaded dataset: the mean over every row, whatever the client split."""
    B = torch.from_numpy(D["B"])
    B = B.to(dtype) if dtype is not None else B
    return problem.Pooled(B.reshape(-1, B.shape[-1]), pen, lam)


RAGGED_BLOCK = 128
_blocks = {}


def ragged_blocks(name, block=None):
    """Unequal clients as equal blocks: (Xb (nb, block, d) signed rows, zero-padded at the end of each
    client; client (nb,) the client of each block; inv (nb,) = 1 / n_client). A zero row has no
    gradient, so a client's local gradient is the sum over its blocks of block^T (inv * sigmoid(block x)).
    """
    block = block or RAGGED_BLOCK
    if (name, block) not in _blocks:
        D = load(name)
        B, off = D["B"], D.get("offsets")
        sizes = np.diff(off)
        counts = -(-sizes // block)
        client = np.repeat(np.arange(len(sizes)), counts)
        Xb = np.zeros((int(counts.sum()), block, B.shape[1]), dtype=B.dtype)
        b = 0
        for i, (lo, hi) in enumerate(zip(off[:-1], off[1:])):
            for start in range(lo, hi, block):
                rows = B[start:min(start + block, hi)]
                Xb[b, :len(rows)] = rows
                b += 1
        _blocks[(name, block)] = {"Xb": Xb, "client": client, "inv": 1.0 / sizes[client]}
    return _blocks[(name, block)]


def _read_fstar():
    return json.loads(FSTAR.read_text()) if FSTAR.exists() else {}


def _write_fstar(table):
    DATA.mkdir(exist_ok=True)
    tmp = FSTAR.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(table, indent=1, sort_keys=True))
    tmp.replace(FSTAR)


def fstar(name, pen, lam, compute=True, log=print):
    """F* for (dataset, penalty, lam); None for a nonconvex penalty."""
    if not problem.penalty(pen)[2]:
        return None
    key = f"{name}/{pen}/{lam_key(lam)}"
    table = _read_fstar()
    if key in table:
        return table[key]["f_star"]
    if not compute:
        raise KeyError(key)
    D = load(name)
    P = pooled(D, pen, lam)
    log(f"computing F* for {key}")
    t0 = time.time()
    if pen == "l2sq":
        x, f, res = problem.solve_l2sq(P, log=log)
        how = "damped Newton"
    else:
        x, f, res = problem.solve_l1(P, D["meta"]["L"], log=log)
        how = "FISTA + adaptive restart, support Newton polish"
    log(f"F* {key} = {f!r}, KKT {res:.2e}, {time.time() - t0:.1f}s")
    table = _read_fstar()
    table[key] = {"f_star": f, "kkt_residual": res, "source": how, "nnz": int((x != 0).sum())}
    _write_fstar(table)
    return f
