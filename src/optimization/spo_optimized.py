from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linprog
from typing import Tuple

from src.optimization.ems import BatterySpec


def box_argmin(c: np.ndarray, batt: BatterySpec) -> np.ndarray:
    """Vectorized version of box argmin for batch processing."""
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
    """Vectorized version of box argmax for batch processing."""
    return box_argmin(-np.asarray(c, dtype=float), batt)


def _spo_terms_batch(c_hat: np.ndarray, c: np.ndarray, batt: BatterySpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized SPO terms computation for batch processing.

    Implements the SPO+ loss of Elmachtoub & Grigas (2022), Definition 3:

        l_SPO+(c_hat, c) = max_{u in X} (c - 2 c_hat)^T u + 2 c_hat^T w*(c) - z*(c)
                         = (2 c_hat - c)^T ( w*(c) - w*(2 c_hat - c) )

    with subgradient 2 ( w*(c) - w*(2 c_hat - c) ).

    Args:
        c_hat: Predicted costs, shape (batch_size, horizon)
        c: True costs, shape (batch_size, horizon)
        batt: Battery specifications
        
    Returns:
        losses: SPO+ losses, shape (batch_size,)
        grads: Subgradients, shape (batch_size, horizon)
    """
    batch_size = c_hat.shape[0]
    horizon = c_hat.shape[1]
    
    # Vectorized computation of the two oracle decisions
    w_c = np.zeros((batch_size, horizon))    # w*(c) : true-cost oracle
    w_mod = np.zeros((batch_size, horizon))  # w*(2 c_hat - c) : modified-cost problem
    
    for i in range(batch_size):
        w_c[i] = box_argmin(c[i], batt)
        w_mod[i] = box_argmin(2.0 * c_hat[i] - c[i], batt)
    
    # Vectorized loss computation: (2 c_hat - c)^T (w*(c) - w*(2 c_hat - c))
    raw = np.einsum("bh,bh->b", 2.0 * c_hat - c, w_c - w_mod)

    # Vectorized subgradient computation
    grads = 2.0 * (w_c - w_mod)

    # The theory guarantees losses >= 0 (l_SPO+ >= l_SPO >= 0); clip numerical noise only
    losses = np.where(raw < 0.0, 0.0, raw)
    grads[raw < 0.0] = 0.0

    return losses, grads


def spo_plus_loss_torch_optimized(price_hat: torch.Tensor, price_true: torch.Tensor, batt: BatterySpec) -> torch.Tensor:
    """
    Optimized SPO+ loss implementation using vectorized operations.

    Implements the SPO+ loss of Elmachtoub & Grigas (2022), Definition 3, with
    subgradient 2 ( w*(c) - w*(2 c_hat - c) ). The loss is non-negative and
    vanishes when the prediction is exact.

    Args:
        price_hat: Predicted prices, shape (batch_size, horizon)
        price_true: True prices, shape (batch_size, horizon)
        batt: Battery specifications
        
    Returns:
        SPO+ loss tensor
    """
    # Convert to numpy for batch processing
    c_hat = price_hat.detach().cpu().numpy().astype(np.float64)
    c = price_true.detach().cpu().numpy().astype(np.float64)
    
    # Batch computation
    losses, grads = _spo_terms_batch(c_hat, c, batt)
    
    # Normalize by scale factor
    scale = float(np.mean(np.abs(c))) * c.shape[1] + 1e-8
    loss_t = torch.tensor(losses.mean() / scale, dtype=price_hat.dtype, device=price_hat.device)
    
    # Store gradients for backward pass
    grads_scaled = grads / (c.shape[0] * scale)
    g_t = torch.tensor(grads_scaled, dtype=price_hat.dtype, device=price_hat.device)
    
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


def spo_plus_loss_torch_original(price_hat: torch.Tensor, price_true: torch.Tensor, batt: BatterySpec) -> torch.Tensor:
    """
    Original SPO+ loss implementation for comparison.
    """
    import torch

    c_hat = price_hat.detach().cpu().numpy().astype(np.float64)
    c = price_true.detach().cpu().numpy().astype(np.float64)
    batch = c_hat.shape[0]
    losses = np.empty(batch, dtype=np.float64)
    grads = np.empty_like(c_hat, dtype=np.float64)
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


def _spo_terms(c_hat: np.ndarray, c: np.ndarray, batt: BatterySpec) -> Tuple[float, np.ndarray]:
    """SPO terms for a single instance (Definition 3, Elmachtoub & Grigas 2022)."""
    w_c = box_argmin(c, batt)  # w*(c) : true-cost oracle
    w_mod = box_argmin(2.0 * c_hat - c, batt)  # w*(2 c_hat - c) : modified-cost problem
    loss = float(np.dot(2.0 * c_hat - c, w_c - w_mod))
    grad = 2.0 * (w_c - w_mod)
    if loss < 0.0:  # numerical noise only (theory gives loss >= 0)
        return 0.0, np.zeros_like(c_hat)
    return loss, grad


def benchmark_spo_implementations(batch_sizes: list, horizon: int = 24, n_runs: int = 10) -> dict:
    """
    Benchmark the optimized vs original SPO implementations.
    
    Args:
        batch_sizes: List of batch sizes to test
        horizon: Prediction horizon
        n_runs: Number of runs for each batch size
        
    Returns:
        Benchmark results dictionary
    """
    import time
    batt = BatterySpec()
    
    results = {}
    
    for batch_size in batch_sizes:
        # Generate test data
        price_hat = torch.randn(batch_size, horizon, dtype=torch.float32)
        price_true = torch.randn(batch_size, horizon, dtype=torch.float32)
        
        # Warm up
        for _ in range(5):
            _ = spo_plus_loss_torch_optimized(price_hat, price_true, batt)
            _ = spo_plus_loss_torch_original(price_hat, price_true, batt)
        
        # Benchmark optimized version
        times_opt = []
        for _ in range(n_runs):
            start = time.time()
            _ = spo_plus_loss_torch_optimized(price_hat, price_true, batt)
            times_opt.append(time.time() - start)
        
        # Benchmark original version
        times_orig = []
        for _ in range(n_runs):
            start = time.time()
            _ = spo_plus_loss_torch_original(price_hat, price_true, batt)
            times_orig.append(time.time() - start)
        
        results[batch_size] = {
            "optimized_mean": np.mean(times_opt),
            "optimized_std": np.std(times_opt),
            "original_mean": np.mean(times_orig),
            "original_std": np.std(times_orig),
            "speedup": np.mean(times_orig) / np.mean(times_opt)
        }
    
    return results