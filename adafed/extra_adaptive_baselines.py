"""Additional full-participation baselines for the rate-study comparison.

Local AdaAlter: Xie et al., Algorithm 3, arXiv:1911.09030.
FedSPS: Mukherjee et al., Algorithm 1, arXiv:2307.06306.
FedAda2: Lee et al., Algorithm 1 with client and server AdaGrad,
arXiv:2410.18117. This is FedAda2, not the SM3-based FedAda2++.
FedUR: Zhang et al., Algorithm 1 and Eqs. (12), (26), (28), (29), (32),
IEEE Transactions on Signal Processing 71 (2023), DOI:10.1109/TSP.2023.3292497.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import data, fl, problem


METHODS = ("local_adaalter", "fedsps", "fedada2", "fedur")


def run(cfg, write=True):
    if cfg["algorithm"] not in METHODS or cfg["p"] != 1.0:
        raise ValueError("this runner requires a supported method and full participation")
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    D = data.load(cfg["dataset"])
    X = torch.from_numpy(D["B"])
    N, n, d = X.shape
    pooled = problem.Pooled(X.reshape(N * n, d), cfg["penalty"], cfg["lam"])
    fstar = data.fstar(cfg["dataset"], cfg["penalty"], cfg["lam"])
    pen_value, pen_grad, _ = problem.penalty(cfg["penalty"])
    K, T, b = cfg["K"], cfg["T"], cfg["b"]
    method, mode = cfg["algorithm"], cfg["mode"]
    generator = torch.Generator().manual_seed(cfg["seed"] ^ 0x5DEECE66D)
    clients = torch.arange(N)[:, None]
    x = torch.zeros(d, dtype=torch.float64)
    checkpoints = set(fl.eval_rounds(T, cfg["eval_every"]).tolist())
    rows = []
    F_sum = G_sum = 0.0
    status = "done"
    if method == "local_adaalter":
        shared_v = torch.full((d,), cfg["b0"] ** 2, dtype=torch.float64)
    elif method == "fedada2":
        server_m = torch.zeros(d, dtype=torch.float64)
        server_v = torch.full((d,), cfg["server_offset"] ** 2, dtype=torch.float64)
    elif method == "fedur":
        local_m = torch.zeros((N, d), dtype=torch.float64)
        server_m = torch.zeros(d, dtype=torch.float64)
        server_v = torch.zeros(d, dtype=torch.float64)
        eta0 = cfg["eta0"]

    def sample():
        if mode == "full":
            return X
        indices = torch.randint(n, (N, b), generator=generator)
        return X[clients, indices]

    def local_value_grad(points, batch, values=False):
        z = torch.bmm(batch, points.unsqueeze(-1)).squeeze(-1)
        weights = torch.sigmoid(z) / batch.shape[1]
        grad = torch.bmm(weights.unsqueeze(1), batch).squeeze(1)
        grad = grad + pen_grad(points, cfg["lam"])
        if values:
            loss = torch.nn.functional.softplus(z).mean(1) + pen_value(points, cfg["lam"])
            return loss, grad
        return grad

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
        if method == "local_adaalter":
            # Denominators remain based on the last synchronization throughout
            # the K local steps; the squared gradients are synchronized at end.
            local_v = shared_v.expand(N, d).clone()
            for k in range(K):
                grad = local_value_grad(local, sample())
                denom = torch.sqrt(shared_v + (k + 1) * cfg["epsilon"] ** 2)
                local = local - cfg["client_lr"] * grad / denom
                local_v += grad.square()
            x = local.mean(0)
            shared_v = local_v.mean(0)
        elif method == "fedsps":
            for _ in range(K):
                loss, grad = local_value_grad(local, sample(), values=True)
                norm_sq = grad.square().sum(1)
                polyak = loss / (cfg["polyak_c"] * norm_sq.clamp_min(1e-30))
                step = torch.minimum(polyak, torch.full_like(polyak, cfg["gamma_cap"]))
                local = local - step[:, None] * grad
            x = local.mean(0)
        elif method == "fedada2":
            # FedAda2 resets client preconditioners every round while the
            # server AdaGrad accumulator persists across communication rounds.
            local_v = torch.zeros((N, d), dtype=torch.float64)
            for _ in range(K):
                grad = local_value_grad(local, sample())
                local_v += grad.square()
                local = local - cfg["client_lr"] * grad / (local_v.sqrt() + cfg["client_offset"])
            delta = local.mean(0) - x
            server_m = cfg["beta1"] * server_m + (1 - cfg["beta1"]) * delta
            server_v += delta.square()
            x = x + cfg["server_eta"] * server_m / (server_v.sqrt() + cfg["server_offset"])
        else:
            # FedUR's local momentum persists across rounds even though the
            # client models restart from the latest global model. Its
            # bias-correction exponent counts *local iterations*, not rounds.
            x_start = x
            global_start_grad = pooled.smooth_value_grad(x)[1] + pen_grad(x, cfg["lam"])
            for k in range(K):
                grad = local_value_grad(local, sample())
                local_m = cfg["beta1"] * local_m + (1 - cfg["beta1"]) * grad
                step_number = (t - 1) * K + k + 1
                local = local - eta0 * local_m / (1 - cfg["beta1"] ** step_number)
            # Eq. (28): aggregate client pseudo-momenta, then Eqs. (26),(29).
            midpoint = (t - 1) * K + K / 2
            mhat = (x - local.mean(0)) / (eta0 * K)
            new_m = mhat * (1 - cfg["beta1"] ** midpoint)
            inferred_grad = (new_m - cfg["beta1"] * server_m) / (1 - cfg["beta1"])
            server_v = cfg["beta2"] * server_v + (1 - cfg["beta2"]) * inferred_grad.square()
            vhat = server_v / (1 - cfg["beta2"] ** midpoint)
            x = x + (local.mean(0) - x) / (vhat.sqrt() + cfg["server_offset"])
            server_m = new_m
            # Algorithm 1 line 29 estimates global smoothness from client
            # secants. Eq. (32) updates eta0 and the paper prescribes a
            # positive fallback plus a bounded learning-rate interval.
            client_beta = ((grad - global_start_grad).norm(dim=1) /
                           (local - x_start).norm(dim=1).clamp_min(1e-12))
            beta_hat = float(client_beta.mean())
            if not math.isfinite(beta_hat) or beta_hat <= 1e-12:
                beta_hat = cfg["smoothness_fallback"]
            new_global_grad = pooled.smooth_value_grad(x)[1] + pen_grad(x, cfg["lam"])
            alpha = cfg["alpha"]
            proposed = (-(alpha * float(new_global_grad.sum()) + (1 - alpha)) /
                        (alpha * beta_hat * d))
            eta0 = min(cfg["eta0"] * 10, max(cfg["eta0"] / 10,
                                             proposed if math.isfinite(proposed) else cfg["eta0"] / 10))
        if not torch.isfinite(x).all():
            status = "diverged"
            break

    summary = {"config": cfg, "status": status, "rounds_run": t,
               "wall_sec": time.perf_counter() - start,
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
        marker = out / "run.json"
        tmp = marker.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, allow_nan=False))
        tmp.replace(marker)
    return summary
