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
    
    Args:
        c_hat: Predicted costs, shape (batch_size, horizon)
        c: True costs, shape (batch_size, horizon)
        batt: Battery specifications
        
    Returns:
        losses: SPO losses, shape (batch_size,)
        grads: Gradients, shape (batch_size, horizon)
    """
    batch_size = c_hat.shape[0]
    horizon = c_hat.shape[1]
    
    # Vectorized computation of x_hat and x_star
    x_hat = np.zeros((batch_size, horizon))
    x_star = np.zeros((batch_size, horizon))
    
    for i in range(batch_size):
        x_hat[i] = box_argmin(c_hat[i], batt)
        x_star[i] = box_argmax(c[i] - 2.0 * c_hat[i], batt)
    
    # Vectorized loss computation
    losses = np.zeros(batch_size)
    for i in range(batch_size):
        losses[i] = float(np.dot(c[i] - 2.0 * c_hat[i], x_star[i]) + 
                          2.0 * np.dot(c_hat[i], x_hat[i]) - np.dot(c[i], x_hat[i]))
    
    losses = np.maximum(losses, 0.0)
    
    # Vectorized gradient computation
    grads = 2.0 * (x_hat - x_star)
    
    return losses, grads


def spo_plus_loss_torch_optimized(price_hat: torch.Tensor, price_true: torch.Tensor, batt: BatterySpec) -> torch.Tensor:
    """
    Optimized SPO+ loss implementation using vectorized operations.
    
    Args:
        price_hat: Predicted prices, shape (batch_size, horizon)
        price_true: True prices, shape (batch_size, horizon)
        batt: Battery specifications
        
    Returns:
        SPO+ loss tensor
    """
    # Convert to numpy for batch processing
    c_hat = price_hat.detach().cpu().numpy()
    c = price_true.detach().cpu().numpy()
    
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


def _spo_terms(c_hat: np.ndarray, c: np.ndarray, batt: BatterySpec) -> Tuple[float, np.ndarray]:
    """Original SPO terms computation for single instance."""
    x_hat = box_argmin(c_hat, batt)
    x_star = box_argmax(c - 2.0 * c_hat, batt)
    loss = float(np.dot(c - 2.0 * c_hat, x_star) + 2.0 * np.dot(c_hat, x_hat) - np.dot(c, x_hat))
    grad = 2.0 * (x_hat - x_star)
    return max(loss, 0.0), grad


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