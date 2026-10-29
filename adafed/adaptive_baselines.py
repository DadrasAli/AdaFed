"""Full-participation FedLADA and FAFED for the rate-study objectives.

FedLADA follows Algorithm 3 of Sun et al. (arXiv:2308.00522).
FAFED follows Algorithm 2 of Wu et al. (AAAI 2023), with q=K and a
communication-round checkpoint after each synchronization. The same batch is
used for both gradients in its STORM-style tracker.

Datasets with unequal clients (data.load()["offsets"] is not None) draw each client's minibatch
from its own n_i rows, compute exact local gradients over data.ragged_blocks, and give FAFED's
initial pass min(n_i, bK) rows per client; the server averages the clients with weights n_i / n (the
objective is the mean over all n rows) instead of 1 / N.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import data, fl, problem


METHODS = ("fedlada", "fafed")


def run(cfg, write=True):
    if cfg["algorithm"] not in METHODS or cfg["p"] != 1.0:
        raise ValueError("this runner supports full-participation FedLADA or FAFED")
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    D = data.load(cfg["dataset"])
    X = torch.from_numpy(D["B"])
    ragged = D.get("offsets") is not None
    if ragged:
        off = torch.from_numpy(D.get("offsets"))
        N, d = len(off) - 1, X.shape[1]
        n_loc = (off[1:] - off[:-1])
        client_weight = (n_loc.to(X.dtype) / float(n_loc.sum())).view(N, 1)                      # n_i / n
        if cfg["mode"] == "full":
            RB = data.ragged_blocks(cfg["dataset"])
            Xb = torch.from_numpy(RB["Xb"])
            blk_client = torch.from_numpy(RB["client"])
            blk_inv = torch.from_numpy(RB["inv"]).to(X.dtype)
    else:
        N, n, d = X.shape
    pooled = data.pooled(D, cfg["penalty"], cfg["lam"])

    def average(t):
        """The server's average over clients: 1 / N each, or n_i / n for unequal clients."""
        return (client_weight * t).sum(0) if ragged else t.mean(0)
    fstar = data.fstar(cfg["dataset"], cfg["penalty"], cfg["lam"])
    pen_grad = problem.penalty(cfg["penalty"])[1]
    method, mode = cfg["algorithm"], cfg["mode"]
    K, T, b = cfg["K"], cfg["T"], cfg["b"]
    lr, server_eta = cfg["client_lr"], cfg.get("server_eta", 1.0)
    beta1, beta2 = cfg.get("beta1", .9), cfg.get("beta2", .99)
    alpha, rho = cfg.get("alpha", .1), cfg.get("rho", .01)
    if not (0 < lr and 0 < server_eta and 0 < alpha < 1 and rho > 0):
        raise ValueError("learning rates, alpha and rho must be positive")
    batch_gen = torch.Generator().manual_seed(cfg["seed"] ^ 0x5DEECE66D)
    init_gen = torch.Generator().manual_seed(cfg["seed"] ^ 0xDA7A51)
    clients = torch.arange(N)[:, None]
    x = torch.zeros(d, dtype=torch.float64)
    checkpoints = set(fl.eval_rounds(T, cfg["eval_every"]).tolist())
    rows = []
    F_sum = G_sum = 0.0
    status = "done"

    def gradient(points, batch, row_weights=None):
        if batch is None:                       # unequal clients, exact: sums over the clients' blocks
            z = torch.bmm(Xb, points.index_select(0, blk_client).unsqueeze(-1)).squeeze(-1)
            w = torch.sigmoid(z) * blk_inv[:, None]
            gb = torch.bmm(w.unsqueeze(1), Xb).squeeze(1)
            return torch.zeros(N, d, dtype=points.dtype).index_add_(0, blk_client, gb) + pen_grad(points, cfg["lam"])
        z = torch.bmm(batch, points.unsqueeze(-1)).squeeze(-1)
        w = torch.sigmoid(z) * row_weights if row_weights is not None else torch.sigmoid(z) / batch.shape[1]
        return torch.bmm(w.unsqueeze(1), batch).squeeze(1) + pen_grad(points, cfg["lam"])

    def sample(generator=batch_gen, size=b):
        if mode == "full":
            return None if ragged else X
        if ragged:                              # uniform with replacement from the client's own rows
            ids = torch.minimum((torch.rand((N, size), generator=generator, dtype=torch.float64)
                                 * n_loc[:, None]).long(), n_loc[:, None] - 1)
            return X[off[:-1, None] + ids]
        ids = torch.randint(n, (N, size), generator=generator)
        return X[clients, ids]

    if method == "fedlada":
        second = torch.full((d,), cfg.get("epsilon_v", 1e-8) ** 2)
        global_direction = torch.zeros(d, dtype=torch.float64)
    else:
        # Paper initialization: one larger batch estimates initial first and
        # second moments, then both moments are averaged across clients.
        if ragged and mode != "full":
            # min(n_i, bK) rows per client: draw bK, weight the first min(n_i, bK) by 1 / min(n_i, bK)
            size = b * K
            s_i = torch.clamp(n_loc, max=size)
            keep = (torch.arange(size)[None, :] < s_i[:, None]).to(X.dtype) / s_i[:, None].to(X.dtype)
            g_init = gradient(x.expand(N, d), sample(init_gen, size), keep)
        else:
            g_init = gradient(x.expand(N, d), sample(init_gen, b * K if ragged else min(n, b * K)))
        momentum = average(g_init).expand(N, d).clone()
        second = average(g_init.square()).expand(N, d).clone()
        denom = second[0].sqrt() + rho
        previous = x.expand(N, d).clone()

    start = time.perf_counter()
    for t in range(1, T + 1):
        value, grad_sq = pooled.evaluate(x)
        F, G = float(value), float(grad_sq)
        if not (math.isfinite(F) and math.isfinite(G)):
            status = "diverged"
            break
        F_sum += F
        G_sum += G
        if t in checkpoints:
            rows.append((t, F, F - fstar if fstar is not None else None, G,
                         F_sum / t - fstar if fstar is not None else None, G_sum / t))
        if t == T:
            break

        local = x.expand(N, d).clone()
        if method == "fedlada":
            m = torch.zeros_like(local)
            v = second.expand(N, d).clone()
            vhat = v.clone()
            for _ in range(K):
                g = gradient(local, sample())
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g.square()
                vhat = torch.maximum(vhat, v)
                local = local - lr * (alpha * m / vhat.sqrt() + (1 - alpha) * global_direction)
            local_mean = average(local)
            new_x = x + server_eta * (local_mean - x)
            global_direction = (x - local_mean) / (lr * K)
            second = average(vhat)
            x = new_x
        else:
            if t == 1:
                # Algorithm 2 takes x_1 = x_0 - eta_0 m_0 before its first
                # q tracked steps; the first synchronization thus follows
                # this initialization step and K tracker updates.
                previous = local
                local = local - lr * momentum
            for k in range(K):
                batch = sample()
                g = gradient(local, batch)
                g_old = gradient(previous, batch)
                momentum = g + (1 - alpha) * (momentum - g_old)
                second = beta2 * second + (1 - beta2) * g.square()
                old = local
                if k == K - 1:
                    # Sync moments and model every K local steps (q=K).
                    mean_m = average(momentum)
                    mean_v = average(second)
                    denom = mean_v.sqrt() + rho
                    local = average(local - lr * mean_m / denom).expand(N, d).clone()
                    momentum = mean_m.expand(N, d).clone()
                    second = mean_v.expand(N, d).clone()
                else:
                    local = local - lr * momentum / denom
                previous = old
            x = local[0].clone()
        if not torch.isfinite(x).all():
            status = "diverged"
            break

    wall = time.perf_counter() - start
    summary = {"config": cfg, "status": status, "rounds_run": t, "wall_sec": wall,
               "final_round": rows[-1][0] if rows else None,
               "final_gap": rows[-1][2] if rows else None,
               "final_grad_norm_sq": rows[-1][3] if rows else None,
               "final_mean_gap": rows[-1][4] if rows else None,
               "final_mean_grad_norm_sq": rows[-1][5] if rows else None}
    if write:
        out = Path(cfg["out"])
        out.mkdir(parents=True, exist_ok=True)
        fl.write_csv(out / "metrics.csv", np.array([r[0] for r in rows]),
                     np.array([r[1] for r in rows]),
                     np.array([r[2] for r in rows]) if fstar is not None else None,
                     np.array([r[3] for r in rows]), None,
                     np.array([r[4] for r in rows]) if fstar is not None else None,
                     np.array([r[5] for r in rows]))
        tmp = out / "run.json.tmp"
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(out / "run.json")
    return summary
