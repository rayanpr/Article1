from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from src.optimization.ems import BatterySpec


def box_argmin(c: np.ndarray, batt: BatterySpec) -> np.ndarray:
    c = np.asarray(c, dtype=float).ravel()
    h = len(c)
    lo = np.full(h, -batt.p_ch)
    ub = np.full(h, batt.p_dis)
    a = np.zeros((h, h))
    for t in range(h):
        a[t, : t + 1] = 1.0
    a_ub = np.vstack([a, -a])
    b_ub = np.concatenate(
        [np.full(h, batt.e_max - batt.e0), np.full(h, batt.e0 - batt.e_min * batt.e_max)]
    )
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(lo[t], ub[t]) for t in range(h)],
        method="highs",
    )
    if not res.success:
        return lo.copy()
    return res.x


def box_argmax(c: np.ndarray, batt: BatterySpec) -> np.ndarray:
    return box_argmin(-np.asarray(c, dtype=float), batt)


def _spo_terms(c_hat: np.ndarray, c: np.ndarray, batt: BatterySpec) -> tuple[float, np.ndarray]:
    x_hat = box_argmin(c_hat, batt)
    x_star = box_argmax(c - 2.0 * c_hat, batt)
    loss = float(np.dot(c - 2.0 * c_hat, x_star) + 2.0 * np.dot(c_hat, x_hat) - np.dot(c, x_hat))
    grad = 2.0 * (x_hat - x_star)
    return max(loss, 0.0), grad


def spo_plus_loss(price_hat: np.ndarray, price_true: np.ndarray, batt: BatterySpec) -> float:
    c_hat = np.asarray(price_hat, dtype=float).ravel()
    c = np.asarray(price_true, dtype=float).ravel()
    loss, _ = _spo_terms(c_hat, c, batt)
    return loss


def spo_plus_loss_torch(price_hat, price_true, batt: BatterySpec):
    import torch

    c_hat = price_hat.detach().cpu().numpy()
    c = price_true.detach().cpu().numpy()
    batch = c_hat.shape[0]
    losses = np.empty(batch, dtype=np.float32)
    grads = np.empty_like(c_hat, dtype=np.float32)
    for i in range(batch):
        losses[i], grads[i] = _spo_terms(c_hat[i], c[i], batt)
    scale = float(np.mean(np.abs(c))) * c_hat.shape[1] + 1e-8
    loss_t = torch.tensor(losses.mean() / scale, dtype=price_hat.dtype, device=price_hat.device)
    g_t = torch.tensor(grads / (batch * scale), dtype=price_hat.dtype, device=price_hat.device)

    class _SPOLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ph, pt):
            ctx.save_for_backward(g_t)
            return loss_t

        @staticmethod
        def backward(ctx, grad_out):
            (g,) = ctx.saved_tensors
            return grad_out * g, None

    return _SPOLoss.apply(price_hat, price_true)
