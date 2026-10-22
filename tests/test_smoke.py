from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import datasets_status, load_processed
from src.forecasting.metrics import crps_from_quantiles, evaluate_forecast, pinball_loss, DEFAULT_TAUS
from src.forecasting.preprocessing import add_time_features, build_supervised, make_splits
from src.optimization.ems import BatterySpec, evaluate_schedule, solve_ems
from src.optimization.spo import box_argmin, spo_plus_loss, spo_plus_loss_torch
from src.optimization.tsro import tsro_ccg


def test_datasets_present():
    status = datasets_status()
    assert status["processed_2022"]["present"] is True
    assert status["ieee33"]["present"] and status["ieee69"]["present"]
    assert status["citylearn"]["present"]


def test_build_supervised_shapes():
    df = add_time_features(load_processed())
    xs, ys, idx = build_supervised(df, lookback=24, horizon=24)
    n = len(df) - 24 - 24 + 1
    assert xs.shape == (n, 24, 8)
    assert ys.shape == (n, 24, 2)
    assert idx.shape == (n,)
    masks = make_splits(df.index)
    assert sum(int(m.sum()) for m in masks.values()) == len(df)


def test_metrics_bounds():
    y = np.array([1.0, 2.0, 3.0])
    q = np.stack([y - 1, y - 0.5, y, y + 0.5, y + 1], axis=-1)
    assert pinball_loss(y, q, DEFAULT_TAUS) >= 0
    assert crps_from_quantiles(y, q) >= 0
    ev = evaluate_forecast(y, q, DEFAULT_TAUS)
    assert 0.0 <= ev["coverage_80"] <= 1.0


def test_ems_lp_and_schedule():
    t = np.arange(24)
    net = 20.0 + 10.0 * np.sin(2 * np.pi * t / 24)
    price = 0.15 + 0.05 * np.sin(2 * np.pi * (t - 6) / 24)
    batt = BatterySpec()
    sol = solve_ems(net, price, batt)
    assert sol["status"] in ("optimal", "optimal_inaccurate")
    assert np.isfinite(sol["cost"])
    cost_sched = evaluate_schedule(sol["discharge"] - sol["charge"], net, price, batt)
    assert abs(cost_sched - sol["cost"]) < 1e-2


def test_spo_plus_nonnegative_and_grad():
    rng = np.random.default_rng(0)
    price_hat = np.abs(rng.normal(0.2, 0.05, 24))
    price_true = np.abs(rng.normal(0.2, 0.05, 24))
    batt = BatterySpec()
    assert spo_plus_loss(price_hat, price_true, batt) >= 0
    u = box_argmin(price_hat, batt)
    assert u.min() >= -batt.p_ch - 1e-6 and u.max() <= batt.p_dis + 1e-6
    import torch

    ph = torch.tensor(price_hat[None, :], dtype=torch.float32, requires_grad=True)
    pt = torch.tensor(price_true[None, :], dtype=torch.float32)
    loss = spo_plus_loss_torch(ph, pt, batt)
    loss.backward()
    assert torch.isfinite(ph.grad).all()


def test_tsro_converges():
    t = np.arange(24)
    net = np.abs(20.0 + 10.0 * np.sin(2 * np.pi * t / 24))
    price = 0.15 + 0.05 * np.sin(2 * np.pi * (t - 6) / 24)
    out = tsro_ccg(net, price, 0.2 * net + 1.0, BatterySpec(), n_iter=3)
    assert out["worst_cost"] >= out["nominal_cost"] - 1e-6
    assert len(out["scenarios"]) >= 1
