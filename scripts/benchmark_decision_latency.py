"""Benchmark inference and decision-resolution latency on the saved Article1 artifacts.

The benchmark is deliberately local and hardware-specific.  It reports medians and
p95 values for a single-window TFT forecast, a 64-window TFT batch, one HiGHS EMS
solve, and the repository's heuristic C&CG routine.

Output:
    results/decision_latency_benchmark.json
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.paths import RESULTS_DIR  # noqa: E402
from scripts.train_evaluate import H, load_arrays, load_tft, predict  # noqa: E402
from src.optimization.ems import BatterySpec, solve_ems  # noqa: E402
from src.optimization.tsro import tsro_ccg  # noqa: E402

MODEL_NAME = "tft_spo_exact"
N_WINDOWS = 50
N_REPEATS = 50
N_WARMUP = 5


def summarize(samples_ms: list[float]) -> dict:
    values = np.asarray(samples_ms, dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def timed(callable_, repeats: int = N_REPEATS, warmup: int = N_WARMUP) -> list[float]:
    for _ in range(warmup):
        callable_()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        callable_()
        samples.append((time.perf_counter_ns() - t0) / 1_000_000.0)
    return samples


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arr = load_arrays()
    model = load_tft(MODEL_NAME, arr)
    q = predict(model, arr["X_test"])
    sm = arr["scaler_mean"]
    ss = arr["scaler_std"]
    q_real = q * ss[None, None, :, None] + sm[None, None, :, None]

    n_available = arr["X_test"].shape[0]
    n = min(N_WINDOWS, n_available)
    indices = np.linspace(0, n_available - 1, n, dtype=int)

    x_single = torch.tensor(arr["X_test"][:1], dtype=torch.float32)
    x_batch = torch.tensor(arr["X_test"][:64], dtype=torch.float32)

    @torch.inference_mode()
    def tft_single() -> None:
        model(x_single)

    @torch.inference_mode()
    def tft_batch() -> None:
        model(x_batch)

    batt = BatterySpec()
    net_hat = q_real[0, :, 0, 2]
    price_hat = q_real[0, :, 1, 2]
    delta = 0.2 * np.abs(net_hat) + 1.0

    def highs_solve() -> None:
        sol = solve_ems(net_hat, price_hat, batt)
        if sol["status"] not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(sol["status"])

    def ccg_solve() -> None:
        tsro_ccg(net_hat, price_hat, delta, batt, n_iter=5)

    measurements = {
        "tft_single_window": summarize(timed(tft_single)),
        "tft_batch_64": summarize(timed(tft_batch)),
        "ems_highs_single_window": summarize(timed(highs_solve)),
        "heuristic_ccg_single_window": summarize(timed(ccg_solve)),
    }
    output = {
        "metadata": {
            "model": MODEL_NAME,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "n_windows": int(n),
            "n_repeats": N_REPEATS,
            "n_warmup": N_WARMUP,
            "timing_scope": (
                "Local CPU latency; the heuristic C&CG routine evaluates three "
                "dispatch rules and does not call a nonlinear solver."
            ),
        },
        "measurements": measurements,
    }
    out = RESULTS_DIR / "decision_latency_benchmark.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Output: {out}")
    for name, values in measurements.items():
        print(
            f"{name}: median={values['median_ms']:.3f} ms, "
            f"p95={values['p95_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
