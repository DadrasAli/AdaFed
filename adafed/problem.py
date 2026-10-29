"""The objective F(x) = (1/n) sum_j log(1 + exp(-b_j a_j^T x)) + penalty(x),
its pooled evaluation, and the reference solvers for F*.

Gradients follow torch autograd's own arithmetic (softplus_backward with
threshold 20, then the product with -b), so a local step here is the step
autograd takes, up to BLAS summation order.
"""

import math

import torch

_softplus_backward = torch.ops.aten.softplus_backward


# ---------------------------------------------------------------------------
# Penalties: value(x) -> (...,), grad(x) -> x.shape; x may be (d,) or (S, d).
# ---------------------------------------------------------------------------

def _l2sq_value(x, lam):
    return 0.5 * lam * (x * x).sum(-1)


def _l2sq_grad(x, lam):
    return lam * x


def _l1_value(x, lam):
    return lam * x.abs().sum(-1)


def _l1_grad(x, lam):
    # sign(0) = 0: the minimum-norm subgradient, and what autograd gives |x|.
    return lam * torch.sign(x)


def _nonconvex_value(x, lam):
    """Smooth saturating penalty lam * sum_j x_j^2 / (1 + x_j^2)."""
    x2 = x * x
    return lam * (x2 / (1.0 + x2)).sum(-1)


def _nonconvex_grad(x, lam):
    x2 = x * x
    return 2.0 * lam * x / ((1.0 + x2) ** 2)


PENALTIES = {
    # name: (value, (sub)gradient, convex)
    "l2sq": (_l2sq_value, _l2sq_grad, True),
    "l1": (_l1_value, _l1_grad, True),
    "nonconvex": (_nonconvex_value, _nonconvex_grad, False),
}


# In-place g += (sub)gradient of the penalty at x, using `buf` (x's shape) as scratch:
# what the batched local step calls, so no array is allocated per step.
def _l2sq_grad_add(g, x, lam, buf):
    g.add_(x, alpha=lam)


def _l1_grad_add(g, x, lam, buf):
    torch.sign(x, out=buf)
    g.add_(buf, alpha=lam)


GRAD_ADD = {"l2sq": _l2sq_grad_add, "l1": _l1_grad_add}


def _nonconvex_grad_add(g, x, lam, buf):
    torch.mul(x, x, out=buf)
    buf.add_(1.0).square_()
    buf.reciprocal_().mul_(x).mul_(2.0 * lam)
    g.add_(buf)


GRAD_ADD["nonconvex"] = _nonconvex_grad_add


def penalty_smoothness_bound(name, lam):
    """Global Hessian-norm bound contributed by a smooth penalty."""
    if name == "l2sq":
        return float(lam)
    if name == "nonconvex":
        # r''(u) = 2 lam (1 - 3u^2)/(1 + u^2)^3, whose absolute value is <= 2 lam.
        return 2.0 * float(lam)
    return 0.0


def grad_add(name):
    """The in-place gradient update; a penalty without one falls back to its grad function."""
    if name in GRAD_ADD:
        return GRAD_ADD[name]
    grad = penalty(name)[1]
    return lambda g, x, lam, buf: g.add_(grad(x, lam))


def penalty(name):
    if name not in PENALTIES:
        raise ValueError(f"penalty {name!r}: choose one of {sorted(PENALTIES)}")
    return PENALTIES[name]


# ---------------------------------------------------------------------------
# The logistic part
# ---------------------------------------------------------------------------

def logistic_weights(z, inv_n, out=None):
    """d loss / dz for loss = mean softplus(z): autograd's own softplus_backward.

    `inv_n` is a tensor of z's shape filled with 1/n (mean's backward).
    """
    if out is None:
        return _softplus_backward(inv_n, z, 1.0, 20.0)
    return torch.ops.aten.softplus_backward.grad_input(inv_n, z, 1.0, 20.0, grad_input=out)


class Pooled:
    """All rows at once, as signed rows B = -y * A: F(x), the full gradient, the reference solves."""

    def __init__(self, B, pen_name, lam):
        self.B = B
        self.n = B.shape[0]
        self.inv_n = torch.full((self.n,), 1.0 / self.n, dtype=B.dtype, device=B.device)
        self.pen_value, self.pen_grad, self.convex = penalty(pen_name)
        self.pen_name = pen_name
        self.lam = float(lam)

    def smooth_value_grad(self, x):
        z = self.B @ x
        loss = torch.nn.functional.softplus(z).mean()
        g = self.B.T @ logistic_weights(z, self.inv_n)
        return loss, g

    def evaluate(self, x):
        """(F(x), ||grad F(x)||^2) as 0-d tensors; l1 uses sign(0) = 0."""
        loss, g = self.smooth_value_grad(x)
        g = g + self.pen_grad(x, self.lam)
        return loss + self.pen_value(x, self.lam), (g * g).sum()

    def evaluate_many(self, X):
        """F and ||grad F||^2 at each row of X (R, d), as (R,) tensors."""
        z = X @ self.B.T                                                   # (R, n)
        loss = torch.nn.functional.softplus(z).mean(1)
        w = logistic_weights(z, self.inv_n.expand(X.shape[0], self.n))
        g = (w @ self.B).add_(self.pen_grad(X, self.lam))
        return loss + self.pen_value(X, self.lam), (g * g).sum(1)

    def values_many(self, X):
        """F at each row of X (R, d), as an (R,) tensor."""
        return torch.nn.functional.softplus(X @ self.B.T).mean(1) + self.pen_value(X, self.lam)

    def objective(self, x):
        return float(torch.nn.functional.softplus(self.B @ x).mean()) + float(self.pen_value(x, self.lam))


def lipschitz(A, tol=1e-15, iters=5000):
    """L = lambda_max(A^T A) / (4 n), by power iteration on A^T A."""
    v = torch.ones(A.shape[1], dtype=A.dtype, device=A.device)
    v = v / torch.linalg.vector_norm(v)
    prev = 0.0
    for _ in range(iters):
        w = A.T @ (A @ v)
        val = float(torch.linalg.vector_norm(w))
        v = w / val
        if abs(val - prev) <= tol * val:
            break
        prev = val
    return val / (4.0 * A.shape[0])


# ---------------------------------------------------------------------------
# Reference optimum F*
# ---------------------------------------------------------------------------

def kkt_residual(P, x):
    _, g = P.smooth_value_grad(x)
    if P.pen_name == "l2sq":
        return float((g + P.lam * x).abs().max())
    if P.pen_name == "l1":
        nz = x != 0
        r = torch.where(nz, (g + P.lam * torch.sign(x)).abs(), (g.abs() - P.lam).clamp(min=0))
        return float(r.max())
    raise ValueError(f"no F* for penalty {P.pen_name!r}")


def _hessian(P, x, cols=None):
    s = torch.sigmoid(P.B @ x)
    D = s * (1 - s) / P.n
    A = P.B if cols is None else P.B[:, cols]
    B = A * D.sqrt()[:, None]
    return B.T @ B


def solve_l2sq(P, tol=1e-12, max_iter=100, log=print):
    """Damped Newton from 0; stops at max |grad F| < tol."""
    x = torch.zeros(P.B.shape[1], dtype=P.B.dtype)
    f = P.objective(x)
    for it in range(max_iter):
        res = kkt_residual(P, x)
        log(f"  newton {it}: F {f:.17g} kkt {res:.3e}")
        if res < tol:
            return x, f, res
        _, g = P.smooth_value_grad(x)
        g = g + P.lam * x
        H = _hessian(P, x)
        H.diagonal().add_(P.lam)
        step = torch.linalg.solve(H, g)
        t = 1.0
        while True:
            xn = x - t * step
            fn = P.objective(xn)
            if fn <= f - 1e-4 * t * float(g @ step) or t < 1e-12:
                break
            t *= 0.5
        if fn >= f and res < 1e3 * tol:
            # At machine precision the objective no longer decreases; take the
            # full Newton step anyway and let the residual decide.
            xn, fn = x - step, P.objective(x - step)
        x, f = xn, fn
    res = kkt_residual(P, x)
    raise RuntimeError(f"l2sq Newton did not reach KKT {tol:g} (at {res:.3e})")


def _soft(v, level):
    return torch.sign(v) * (v.abs() - level).clamp(min=0)


def _polish_l1(P, x, iters=50):
    """Newton on the support of x with its signs fixed; a side branch."""
    x = x.clone()
    for _ in range(5):
        S = torch.nonzero(x != 0).flatten()
        if S.numel() == 0:
            return x
        sgn = torch.sign(x[S])
        for _ in range(iters):
            _, g = P.smooth_value_grad(x)
            gS = g[S] + P.lam * sgn
            H = _hessian(P, x, S)
            evals, evecs = torch.linalg.eigh(H)
            keep = evals > evals.max() * 1e-12
            step = evecs[:, keep] @ ((evecs[:, keep].T @ gS) / evals[keep])
            xn = x.clone()
            xn[S] = x[S] - step
            x = xn
            if float(step.abs().max()) < 1e-17 * max(1.0, float(x.abs().max())):
                break
        flipped = torch.sign(x[S]) != sgn
        if not bool(flipped.any()):
            return x
        x[S[flipped]] = 0.0  # a coordinate pushed across its own sign leaves the support
    return x


def solve_l1(P, L, tol=1e-12, max_iter=2_000_000, check_every=200, log=print):
    """FISTA with adaptive (gradient) restart, polishing on the support at checks."""
    d = P.B.shape[1]
    x = torch.zeros(d, dtype=P.B.dtype)
    z, t = x.clone(), 1.0
    step = 1.0 / (L + 0.0)
    best = (math.inf, None)
    for it in range(1, max_iter + 1):
        _, g = P.smooth_value_grad(z)
        xn = _soft(z - step * g, step * P.lam)
        if float((z - xn) @ (xn - x)) > 0:  # O'Donoghue-Candes restart
            t = 1.0
            z = xn.clone()
        else:
            tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
            z = xn + ((t - 1) / tn) * (xn - x)
            t = tn
        x = xn
        if it % check_every == 0:
            res = kkt_residual(P, x)
            if res < 1e-5:
                xp = _polish_l1(P, x)
                rp = kkt_residual(P, xp)
                if rp < best[0]:
                    best = (rp, xp)
                log(f"  fista {it}: kkt {res:.3e}, polished {rp:.3e}, nnz {int((xp != 0).sum())}")
                if rp < tol:
                    return xp, P.objective(xp), rp
            elif it % (50 * check_every) == 0:
                log(f"  fista {it}: kkt {res:.3e}")
            if res < tol:
                return x, P.objective(x), res
    raise RuntimeError(f"l1 FISTA did not reach KKT {tol:g} (best {best[0]:.3e})")
