"""One AdaFed, FedAdagrad, or FedAdam run, fully vectorised.

All sampled clients take their K local steps together as batched operations
on X[sampled] (S, n_i, d). Metrics stay in memory and are written once.

Row t is measured on the paper's pre-update iterate x_t.  The exact cumulative
mean of the per-round metric is accumulated online, so no model trajectory needs
to be retained on disk.

Datasets with unequal clients (data.load()["offsets"] is not None; the Dirichlet splits) take a
second path: flat rows, a client's minibatch drawn uniformly with replacement from its own n_i
rows, exact local gradients as sums over data.ragged_blocks, and the server weights client i by
n_i / n (the objective is the mean over all n rows) instead of 1 / N.  Equal-client datasets never
enter it.
"""

import ast
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import data, problem

DEFAULTS = {
    "dataset": "synthetic-kappa10",
    "penalty": "l1",
    "lam": 0.03,
    "algorithm": "adafed",   # adafed | fedadagrad | fedadam
    "mode": "full",          # full | stochastic
    "b": 8,                  # batch size, stochastic mode
    "K": 1,
    "p": 1.0,
    "T": 10000,
    "seed": 0,
    "c": 1.0,                # AdaFed: eta = c * gamma / L; FedAvg/FedOpt: client lr = c / L
    "eta": None,             # AdaFed paper parameter eta; overrides c * gamma / L
    "gamma": 0.1,            # AdaFed accumulator initialization S_1
    "server_eta": 1.0,       # FedOpt server learning rate
    "beta1": 0.9,            # FedOpt first-moment decay
    "beta2": 0.99,           # FedAdam/FedYogi second-moment decay
    "adapt_tau": 1e-3,       # FedOpt denominator offset; v_0 = adapt_tau**2
    "tau": "1/t",            # expression in t, T, K, k (k = 0..K-1)
    "eval_every": 10,
    "device": "cpu",
    "dtype": "float64",
    "out": None,
}


# ---------------------------------------------------------------------------
# tau(t, T, K, k): a small arithmetic language, evaluated on the whole (T, K) grid
# ---------------------------------------------------------------------------

_FUNCS = {"min": np.minimum, "max": np.maximum, "sqrt": np.sqrt}


def tau_table(expr, T, K):
    """tau at every (t, k), t = 1..T, k = 0..K-1, as a float64 (T, K) array."""
    env = {"t": np.arange(1, T + 1, dtype=float)[:, None], "T": float(T), "K": float(K),
           "k": np.arange(K, dtype=float)[None, :]}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = ev(node.left), ev(node.right)
            return {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
                    ast.Div: np.divide, ast.Pow: np.power}[type(node.op)](a, b)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            args = [ev(a) for a in node.args]
            f = _FUNCS[node.func.id]
            return f(*args) if f is np.sqrt else _reduce(f, args)
        raise ValueError(f"tau {expr!r}: unsupported element {ast.dump(node)[:60]}")

    with np.errstate(all="ignore"):
        tab = np.broadcast_to(np.asarray(ev(ast.parse(str(expr), mode="eval")), dtype=float), (T, K)).copy()
    if not (np.isfinite(tab).all() and (tab > 0).all()):
        raise ValueError(f"tau {expr!r} is not finite and positive on every (t, k)")
    return tab


def _reduce(f, args):
    out = args[0]
    for a in args[1:]:
        out = f(out, a)
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

FEDOPT_ALGORITHMS = ("fedadagrad", "fedadam")
ALGORITHMS = ("adafed",) + FEDOPT_ALGORITHMS


def resolve(cfg):
    out = dict(DEFAULTS)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    out.update(cfg)
    for k in ("K", "T", "b", "seed", "eval_every"):
        out[k] = int(out[k])
    for k in ("lam", "p", "c", "gamma", "server_eta", "beta1", "beta2", "adapt_tau"):
        out[k] = float(out[k])
    if out["eta"] is not None:
        out["eta"] = float(out["eta"])
    out["tau"] = str(out["tau"])
    if out["algorithm"] not in ALGORITHMS:
        raise ValueError(f"algorithm {out['algorithm']!r}: one of {', '.join(ALGORITHMS)}")
    if out["mode"] not in ("full", "stochastic"):
        raise ValueError(f"mode {out['mode']!r}: full or stochastic")
    if not 0 < out["p"] <= 1:
        raise ValueError("p must be in (0, 1]")
    if out["algorithm"] in FEDOPT_ALGORITHMS:
        if not (out["c"] > 0 and out["server_eta"] > 0 and out["adapt_tau"] > 0):
            raise ValueError("FedOpt client/server learning rates and adapt_tau must be positive")
        if not (0 <= out["beta1"] < 1 and 0 <= out["beta2"] < 1):
            raise ValueError("FedOpt beta1 and beta2 must be in [0, 1)")
    return out


def eval_rounds(T, every):
    r = set(range(every, T + 1, every)) | {1, T}
    return np.array(sorted(r), dtype=np.int64)


# Settings that may differ between the runs of one batch; everything else is shared.
BATCH_KEYS = ("c", "eta", "gamma", "tau", "server_eta", "out")


def batch_key(cfg):
    """Runs with equal keys see the same sampled clients and batches every round."""
    cfg = resolve(cfg)
    return tuple((k, cfg[k]) for k in sorted(cfg) if k not in BATCH_KEYS)


def run(cfg, write=True):
    """Run one configuration. Returns the summary dict (also run.json)."""
    return run_batch([cfg], write=write)[0]


def _budget_bytes(dev):
    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info(dev)
        return int(0.5 * free)
    return int(float(os.environ.get("ADAFED_CPU_MEM_GB", "2")) * 2**30)


def run_batch(cfgs, write=True, threads=1):
    """Run configurations that share everything but c/eta, gamma, tau, server_eta and output.

    Iterates are (sampled clients, R, d): each local step is one batched matrix-matrix
    product per client for all R runs. The participation and IID minibatch draws are those a
    single run with this seed makes, so each run's output is the unbatched one up to
    floating-point summation order. A run whose values go non-finite leaves the batch
    at that round, marked diverged. R is split into chunks that fit in memory.
    `threads` is torch's intra-op (BLAS and elementwise) thread count for this batch.
    """
    cfgs = [resolve(c) for c in cfgs]
    key = batch_key(cfgs[0])
    if any(batch_key(c) != key for c in cfgs):
        raise ValueError("a batch's runs may differ only in " + ", ".join(BATCH_KEYS))
    c0 = cfgs[0]
    thread_count = max(1, int(threads))
    if torch.get_num_threads() != thread_count:
        torch.set_num_threads(thread_count)
    dev = torch.device(c0["device"])
    dtype = getattr(torch, c0["dtype"])
    D = data.load(c0["dataset"])
    ragged = D.get("offsets") is not None
    if ragged:
        N, d = len(D.get("offsets")) - 1, D["B"].shape[1]
        n = data.RAGGED_BLOCK
    else:
        N, n, d = D["B"].shape
    K, b = c0["K"], c0["b"]
    m = b if c0["mode"] == "stochastic" else n
    gen = torch.Generator().manual_seed(c0["seed"])
    cnt_max = int((torch.rand(c0["T"], N, generator=gen, dtype=torch.float64) < c0["p"]).sum(1).max())
    if ragged and c0["mode"] != "stochastic":
        cnt_max = max(cnt_max, len(data.ragged_blocks(c0["dataset"])["client"]))   # blocks, not clients
    per_run = torch.finfo(dtype).bits // 8 * (4 * max(cnt_max, 1) * d + 3 * max(cnt_max, 1) * m + 4 * d)
    chunk = max(1, _budget_bytes(dev) // per_run)
    out = []
    for i in range(0, len(cfgs), chunk):
        out += _run_chunk(cfgs[i:i + chunk], write)
    return out


def _run_chunk(cfgs, write):
    c0 = cfgs[0]
    R = len(cfgs)
    dev = torch.device(c0["device"])
    dtype = getattr(torch, c0["dtype"])
    f64 = torch.float64
    D = data.load(c0["dataset"])
    L = float(D["meta"]["L"])
    fstar = data.fstar(c0["dataset"], c0["penalty"], c0["lam"])
    _, _, convex = problem.penalty(c0["penalty"])
    grad_add = problem.grad_add(c0["penalty"])
    lam = c0["lam"]

    # Signed rows B = -y * X (exact for +-1 labels): margins z = B x, gradient B^T w.
    # Training uses `dtype`; F, the gap and the gradient norm always use a float64 copy.
    X = torch.from_numpy(D["B"]).to(dev, dtype)  # shared with the sweep's other workers on cpu float64
    X64 = X if dtype == f64 else torch.from_numpy(D["B"]).to(dev, f64)
    ragged = D.get("offsets") is not None
    K, T, p, b = c0["K"], c0["T"], c0["p"], c0["b"]
    stochastic = c0["mode"] == "stochastic"
    if ragged:
        off = torch.from_numpy(D.get("offsets"))
        N, d = len(off) - 1, X.shape[1]
        sizes_cpu = off[1:] - off[:-1]
        pooled = problem.Pooled(X64, c0["penalty"], lam)
        client_weight = (sizes_cpu.to(f64) / float(sizes_cpu.sum())).to(dev, dtype)                  # n_i / n
        n = data.RAGGED_BLOCK
        if not stochastic:
            RB = data.ragged_blocks(c0["dataset"])
            Xb = torch.from_numpy(RB["Xb"]).to(dev, dtype)
            blk_client = torch.from_numpy(RB["client"]).to(dev)
            blk_inv = torch.from_numpy(RB["inv"]).to(dev, dtype)
            blk_client_np = RB["client"]
            n = Xb.shape[1]
    else:
        N, n, d = X.shape
        pooled = problem.Pooled(X64.view(N * n, d), c0["penalty"], lam)

    m = b if stochastic else n
    adafed = c0["algorithm"] == "adafed"
    fedopt = c0["algorithm"] in FEDOPT_ALGORITHMS
    if adafed:
        eta_np = np.array([c["eta"] if c["eta"] is not None else c["c"] * c["gamma"] / L for c in cfgs])
        eta = torch.tensor(eta_np, dtype=f64, device=dev)
        S = torch.tensor([c["gamma"] for c in cfgs], dtype=f64, device=dev)
        exprs = sorted({c["tau"] for c in cfgs})
        tabs = np.stack([tau_table(e, T, K) for e in exprs])                         # (E, T, K)
        tsum = np.array([[math.fsum(row) for row in tab] for tab in tabs])            # (E, T)
        eidx = np.array([exprs.index(c["tau"]) for c in cfgs])
    else:
        lr_np = np.array([c["c"] / L for c in cfgs])
        lr = torch.tensor(lr_np, dtype=dtype, device=dev)
        server_eta_np = np.array([c["server_eta"] for c in cfgs])
        server_eta = torch.tensor(server_eta_np, dtype=dtype, device=dev)
        momentum = torch.zeros(R, d, dtype=dtype, device=dev)
        second = torch.full((R, d), c0["adapt_tau"] ** 2, dtype=dtype, device=dev)

    # Bernoulli participation, drawn up front on the CPU so cpu and cuda agree.
    gen = torch.Generator().manual_seed(c0["seed"])
    masks = (torch.rand(T, N, generator=gen, dtype=torch.float64) < p).numpy()
    all_idx = torch.arange(N, device=dev)
    batch_gen = torch.Generator().manual_seed(c0["seed"] ^ 0x5DEECE66D)
    g0 = torch.full((1, 1, 1), 1.0 / m, dtype=dtype, device=dev)  # mean's backward, expanded per step
    cmax = max(1, int(masks.sum(1).max()))
    bufs = {}  # per batch width: the local step's arrays, allocated once (views [:cnt] per round)

    def buffers(Ra):
        if bufs.get("Ra") != Ra:
            bufs.clear()
            bufs.update(Ra=Ra, xs=torch.empty(cmax, Ra, d, dtype=dtype, device=dev),
                        g=torch.empty(cmax, Ra, d, dtype=dtype, device=dev),
                        p=torch.empty(cmax, Ra, d, dtype=dtype, device=dev),
                        z=torch.empty(cmax, Ra, m, dtype=dtype, device=dev),
                        w=torch.empty(cmax, Ra, m, dtype=dtype, device=dev))
        return bufs

    ev_rounds = eval_rounds(T, c0["eval_every"])
    is_eval = np.zeros(T + 1, dtype=bool)
    is_eval[ev_rounds] = True
    E = len(ev_rounds)
    F_log, G_log, eta_log, MF_log, MG_log = (np.full((E, R), math.nan) for _ in range(5))
    F_sum, G_sum = np.zeros(R), np.zeros(R)
    n_rows = np.zeros(R, dtype=np.int64)          # evaluated rows written per run
    status, last = ["done"] * R, [T] * R
    act = np.arange(R)                             # runs still in the batch
    x = torch.zeros(R, d, dtype=dtype, device=dev)
    e = 0
    tag = _tag(c0, R)
    mark_every = max(1, T // 10)
    next_mark = mark_every
    t0 = time.perf_counter()

    def drop(bad_local, t):
        nonlocal act, x, S, eta, lr, server_eta, momentum, second
        for j in act[bad_local]:
            status[j], last[j] = "diverged", t
        keep = ~bad_local
        kt = torch.from_numpy(keep).to(dev)
        act, x = act[keep], x[kt]
        if adafed:
            S, eta = S[kt], eta[kt]
        else:
            lr = lr[kt]
            server_eta, momentum, second = server_eta[kt], momentum[kt], second[kt]
        print(f"{tag} {int(bad_local.sum())} run(s) diverged at round {t}; {len(act)} left", flush=True)

    for t in range(1, T + 1):
        if not len(act):
            break
        Ra = len(act)
        sel = masks[t - 1]
        cnt = int(sel.sum())
        if adafed:
            eta_t = eta / S                                                         # (Ra,) float64
        # Algorithm 1 outputs a uniform draw from x_1,...,x_T.  Evaluate x_t before
        # its update and accumulate the conditional expectation exactly.
        F_pre, G_pre = pooled.evaluate_many(x.to(f64))
        Fh, Gh = F_pre.cpu().numpy(), G_pre.cpu().numpy()
        F_sum[act] += Fh
        G_sum[act] += Gh
        bad = ~(np.isfinite(Fh) & np.isfinite(Gh))
        if bad.any():
            drop(bad, t)
            if not len(act):
                break
            Fh, Gh = Fh[~bad], Gh[~bad]
            if adafed:
                eta_t = eta_t[torch.from_numpy(~bad).to(dev)]
            Ra = len(act)  # a diverged run left after Ra was set at the top of the round
        if is_eval[t]:
            F_log[e, act], G_log[e, act] = Fh, Gh
            MF_log[e, act], MG_log[e, act] = F_sum[act] / t, G_sum[act] / t
            if adafed:
                eta_log[e, act] = eta_t.cpu().numpy()
            n_rows[act] = e + 1
            e += 1
        if t == T:
            break
        if cnt:
            if cnt == N and not ragged:
                idx, Xs = all_idx, X
            else:
                idx = torch.from_numpy(np.flatnonzero(sel)).to(dev)
                Xs = None
            if stochastic and ragged:
                # Each client's minibatch: uniform with replacement from its own n_i rows (CPU draw).
                ic = torch.from_numpy(np.flatnonzero(sel))
                n_loc = sizes_cpu[ic].view(1, cnt, 1)
                u = torch.rand((K, cnt, b), generator=batch_gen, dtype=torch.float64)
                rows = torch.minimum((u * n_loc).long(), n_loc - 1) + off[ic].view(1, cnt, 1)
                XB = X[rows.to(dev)]                                                       # (K, S, b, d)
            elif stochastic:
                # Independent uniform-with-replacement minibatches at every oracle call.
                # Draw on CPU so CPU and CUDA runs use the same oracle realization.
                rows = torch.randint(n, (K, cnt, b), generator=batch_gen).to(dev)
                XB = X[idx[None, :, None], rows]                                           # (K, S, b, d)
            elif ragged:
                if cnt == N:
                    bs_client, bs_X, bs_inv = blk_client, Xb, blk_inv
                else:                                                                      # blocks of the sampled clients
                    keep = np.flatnonzero(sel[blk_client_np])
                    kt = torch.from_numpy(keep).to(dev)
                    pos = torch.from_numpy(np.cumsum(sel) - 1).to(dev)
                    bs_client, bs_X, bs_inv = pos[blk_client[kt]], Xb[kt], blk_inv[kt]
                nbs = len(bs_client)
            elif Xs is None:
                Xs = X[idx]
            if adafed:
                taus = torch.from_numpy(tabs[eidx[act], t - 1]).to(dev)             # (Ra, K) float64
                steps = (taus * eta_t[:, None]).to(dtype)
            gk = g0.expand(cnt, Ra, m)
            B_ = buffers(Ra)
            xs, g, pb, z, w = (B_[k][:cnt] for k in ("xs", "g", "p", "z", "w"))
            xs.copy_(x.unsqueeze(0).expand(cnt, Ra, d))                             # (S, Ra, d)
            if ragged and not stochastic:
                # Exact local gradients of unequal clients: one batched product over blocks,
                # then the blocks of a client are summed (index_add) into its gradient.
                zb = torch.empty(nbs, Ra, n, dtype=dtype, device=dev)
                wb = torch.empty_like(zb)
                gb = torch.empty(nbs, Ra, d, dtype=dtype, device=dev)
                gkb = bs_inv.view(nbs, 1, 1).expand(nbs, Ra, n)
                for k in range(K):
                    torch.bmm(xs.index_select(0, bs_client), bs_X.transpose(1, 2), out=zb)
                    problem.logistic_weights(zb, gkb, out=wb)
                    torch.bmm(wb, bs_X, out=gb)
                    g.zero_().index_add_(0, bs_client, gb)
                    grad_add(g, xs, lam, pb)
                    xs.addcmul_(g, (steps[:, k] if adafed else lr).view(1, Ra, 1), value=-1.0)
            else:
                for k in range(K):
                    Xk = XB[k] if stochastic else Xs
                    torch.bmm(xs, Xk.transpose(1, 2), out=z)                        # (S, Ra, m)
                    problem.logistic_weights(z, gk, out=w)
                    torch.bmm(w, Xk, out=g)                                         # (S, Ra, d)
                    grad_add(g, xs, lam, pb)
                    xs.addcmul_(g, (steps[:, k] if adafed else lr).view(1, Ra, 1), value=-1.0)
            if adafed:
                if ragged:        # sum_{i in S} (n_i / (n p)) (x_i - x), over the sum of tau
                    scale = torch.from_numpy(1.0 / tsum[eidx[act], t - 1]).to(dev, dtype)
                    step = xs.sub_(x).mul_((client_weight[idx] / p).view(cnt, 1, 1)).sum(0).mul_(scale[:, None])
                else:
                    scale = torch.from_numpy(1.0 / (N * tsum[eidx[act], t - 1])).to(dev, dtype)
                    step = xs.sub_(x).mul_(1.0 / p).sum(0).mul_(scale[:, None])    # (Ra, d)
                st = step.to(f64)
                nsq = (st * st).sum(1)
                x = x + step
                S = S * torch.sqrt(1.0 + (torch.sqrt(nsq) / eta) ** 2)
                bad = ~torch.isfinite(S).cpu().numpy()
                if bad.any():
                    drop(bad, t)
                    if adafed:
                        eta_t = eta_t[torch.from_numpy(~bad).to(dev)]
                    if not len(act):
                        break
            else:
                if ragged:        # the n_i-weighted mean of the sampled clients
                    w_s = client_weight[idx]
                    delta = (xs * w_s.view(cnt, 1, 1)).sum(0) / w_s.sum() - x
                else:
                    delta = xs.mean(0) - x
                momentum = c0["beta1"] * momentum + (1 - c0["beta1"]) * delta
                delta_sq = delta.square()
                if c0["algorithm"] == "fedadagrad":
                    second = second + delta_sq
                else:
                    second = c0["beta2"] * second + (1 - c0["beta2"]) * delta_sq
                x = x + server_eta[:, None] * momentum / (second.sqrt() + c0["adapt_tau"])
        if is_eval[t] and t >= next_mark and len(act):
            next_mark = (t // mark_every + 1) * mark_every
            ok = np.isfinite(Fh)
            if fstar is not None and ok.any():
                gaps = Fh[ok] - fstar
                stat = f"gap min {gaps.min():.3e} mean {np.mean(gaps):.3e}"
            else:
                stat = f"grad^2 min {Gh[ok].min():.3e}" if ok.any() else ""
            print(f"{tag} round {t}/{T} ({100 * t // T}%) {len(act)}/{R} running, {stat}, "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    device_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else _cpu_name()
    summaries = []
    for j, cfg in enumerate(cfgs):
        k = n_rows[j]
        F_np, G_np = F_log[:k, j], G_log[:k, j]
        rounds = ev_rounds[:k]
        gap = F_np - fstar if fstar is not None else None
        mean_gap = MF_log[:k, j] - fstar if fstar is not None else None
        summary = {
            "config": cfg,
            "status": status[j],
            "rounds_run": int(last[j]),
            "wall_sec": wall,
            "batch_size": R,
            "L": L,
            "f_star": fstar,
            "eta": float(eta_np[j]) if adafed else None,
            "lr": None if adafed else float(lr_np[j]),
            "server_eta": float(server_eta_np[j]) if fedopt else None,
            "score": score_of(status[j], gap, G_np, convex and fstar is not None),
            "final_round": int(rounds[-1]) if k else None,
            "final_objective": float(F_np[-1]) if k else None,
            "final_gap": float(gap[-1]) if gap is not None and k else None,
            "final_grad_norm_sq": float(G_np[-1]) if k else None,
            "final_eta_t": float(eta_log[k - 1, j]) if adafed and k else None,
            "final_mean_gap": float(mean_gap[-1]) if mean_gap is not None and k else None,
            "final_mean_grad_norm_sq": float(MG_log[k - 1, j]) if k else None,
            "device_name": device_name,
        }
        if write:
            o = Path(cfg["out"] or default_out(cfg))
            o.mkdir(parents=True, exist_ok=True)
            write_csv(o / "metrics.csv", rounds, F_np, gap, G_np,
                      eta_log[:k, j] if adafed else None, mean_gap, MG_log[:k, j])
            tmp = o / "run.json.tmp"
            tmp.write_text(json.dumps(summary, indent=1, default=str))
            tmp.replace(o / "run.json")  # run.json appears last: it marks the run finished
            summary["out"] = str(o)
        summaries.append(summary)
    return summaries


def _tag(cfg, R=1):
    if R > 1 and cfg["out"]:
        return _tag(cfg).rsplit("/", 2)[0] + f" [{R} runs]"
    if cfg["out"]:
        out = Path(cfg["out"]).resolve()
        for base in (data.ROOT / "results", data.ROOT):
            try:
                return str(out.relative_to(base))
            except ValueError:
                pass
        return str(out)
    return str(default_out(cfg).relative_to("results"))


def score_of(status, gap, G, use_gap):
    """Mean log10 gap (clamped at 1e-16) or mean log10 grad_norm_sq; inf if diverged."""
    if status != "done":
        return math.inf
    v = gap if use_gap else G
    return float(np.mean(np.log10(np.maximum(v, 1e-16))))


MEAN_COLUMNS = ("mean_gap", "mean_grad_norm_sq")


def add_mean_columns(path):
    """Validate that exact cumulative means are present.

    They cannot be reconstructed from checkpointed values. Correct runs accumulate every
    pre-update round online in `_run_chunk`.
    """
    path = Path(path)
    with open(path) as f:
        head = f.readline().strip().split(",")
    if all(c in head for c in MEAN_COLUMNS):
        return False
    raise RuntimeError(f"{path}: exact random-output means are absent and cannot be recovered from checkpoints; rerun")


def write_csv(path, rounds, F, gap, G, eta, mean_gap, mean_G):
    """Write checkpointed last-iterate values and exact all-round cumulative means."""
    cols = ["round", "objective", "gap", "grad_norm_sq"] + (["eta_t"] if eta is not None else []) \
           + ["mean_gap", "mean_grad_norm_sq"]
    lines = [",".join(cols)]
    for i, r in enumerate(rounds):
        row = [str(int(r)), repr(float(F[i])), "" if gap is None else repr(float(gap[i])), repr(float(G[i]))]
        if eta is not None:
            row.append(repr(float(eta[i])))
        row += ["" if mean_gap is None else repr(float(mean_gap[i])), repr(float(mean_G[i]))]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


def default_out(cfg):
    base = (f"eta{cfg['eta']:g}" if cfg["algorithm"] == "adafed" and cfg["eta"] is not None
            else f"c{cfg['c']:g}")
    if cfg["algorithm"] == "adafed":
        tuned = base + f"_gamma{cfg['gamma']:g}_tau{_slug(cfg['tau'])}"
    elif cfg["algorithm"] in FEDOPT_ALGORITHMS:
        tuned = (base + f"_server{cfg['server_eta']:g}_beta1{cfg['beta1']:g}"
                 f"_beta2{cfg['beta2']:g}_offset{cfg['adapt_tau']:g}")
    else:
        tuned = base
    mode = "full" if cfg["mode"] == "full" else f"stoch-b{cfg['b']}"
    return (Path("results") / "run" / f"{cfg['dataset']}-{cfg['penalty']}-lam{cfg['lam']:g}" /
            f"{cfg['algorithm']}-{mode}-K{cfg['K']}-p{cfg['p']:g}-{tuned}-T{cfg['T']}-s{cfg['seed']}")


def _slug(s):
    return "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in str(s).replace(" ", ""))


def _cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "cpu"
