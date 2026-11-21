"""Sweep of a budgeted uncertainty set for the microgrid EMS experiment.

The script evaluates the implemented heuristic C&CG dispatcher on the 241
operational test windows.  The uncertainty set is

    U_Gamma = {xi : |xi_t - n_hat_t| <= delta_t,
               sum_t |xi_t - n_hat_t| / delta_t <= Gamma},

with Gamma in [0, d] and d = 24.  The worst-case net profile is selected
hour by hour under the budget; this is the same separable approximation used
by src.optimization.tsro, but with an explicit budget on simultaneous
deviations.

Outputs:
    results/robustness_budget_sweep.json
    figures/robustness_budget_sweep.png
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.paths import FIGURES_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from scripts.train_evaluate import H, TAUS, load_arrays, load_tft, predict  # noqa: E402
from src.optimization.ems import BatterySpec, evaluate_schedule  # noqa: E402
from src.optimization.tsro import default_rules, hourly_cost  # noqa: E402

SEED = 42
EXPORT_RATIO = 0.5
DELTA_SCALE = 0.2
DELTA_FLOOR_KWH = 1.0
GAMMAS = (0, 2, 4, 8, 12, 16, 20, 24)
MODEL_NAME = "tft_spo_exact"


def budget_worst_case_net(
    net_hat: np.ndarray,
    delta: np.ndarray,
    schedule: np.ndarray,
    price: np.ndarray,
    gamma: int,
    export_ratio: float = EXPORT_RATIO,
) -> tuple[np.ndarray, dict]:
    """Return a budget-feasible worst-case net profile.

    For each hour, the upper and lower bounds are evaluated separately.  The
    largest positive hourly cost increments are selected, up to Gamma hours.
    Ties are deterministic (hours are sorted by increment, then by index).
    """
    h = len(net_hat)
    nominal_cost = np.array(
        [hourly_cost(net_hat[t], schedule[t], price[t], export_ratio) for t in range(h)]
    )
    lower = net_hat - delta
    upper = net_hat + delta
    lower_cost = np.array(
        [hourly_cost(lower[t], schedule[t], price[t], export_ratio) for t in range(h)]
    )
    upper_cost = np.array(
        [hourly_cost(upper[t], schedule[t], price[t], export_ratio) for t in range(h)]
    )

    candidates: list[tuple[float, int, int, float]] = []
    for t in range(h):
        lo_inc = float(lower_cost[t] - nominal_cost[t])
        hi_inc = float(upper_cost[t] - nominal_cost[t])
        if hi_inc >= lo_inc:
            candidates.append((hi_inc, t, 1, float(upper[t])))
        else:
            candidates.append((lo_inc, t, -1, float(lower[t])))

    selected = sorted(candidates, key=lambda x: (-x[0], x[1]))[: max(0, min(int(gamma), h))]
    worst = net_hat.copy()
    deviations: list[dict] = []
    used_budget = 0.0
    for increment, t, direction, value in selected:
        if increment <= 0:
            continue
        worst[t] = value
        used_budget += abs(value - net_hat[t]) / max(delta[t], 1e-12)
        deviations.append(
            {"hour": int(t), "direction": int(direction), "cost_increment_eur": increment}
        )
    return worst, {
        "budget_used": used_budget,
        "budget_limit": int(gamma),
        "deviations": deviations,
    }


def tsro_ccg_budget(
    net_hat: np.ndarray,
    price_hat: np.ndarray,
    delta: np.ndarray,
    gamma: int,
    batt: BatterySpec,
    export_ratio: float = EXPORT_RATIO,
    n_iter: int = 5,
) -> dict:
    """Heuristic column-and-constraint generation over active scenarios."""
    rules = default_rules()
    scenarios = [net_hat.copy()]
    history: list[dict] = []
    schedule = None
    chosen = None
    for iteration in range(n_iter):
        best = None
        for rule_name, rule in rules.items():
            candidate = rule(net_hat, price_hat, batt)
            robust_cost = max(
                evaluate_schedule(candidate, scenario, price_hat, batt, export_ratio)
                for scenario in scenarios
            )
            if best is None or robust_cost < best["cost"] - 1e-9:
                best = {"cost": robust_cost, "schedule": candidate, "rule": rule_name}
        schedule = best["schedule"]
        chosen = best["rule"]
        worst_net, budget_info = budget_worst_case_net(
            net_hat, delta, schedule, price_hat, gamma, export_ratio
        )
        new_cost = evaluate_schedule(
            schedule, worst_net, price_hat, batt, export_ratio
        )
        history.append(
            {
                "iteration": iteration,
                "rule": chosen,
                "robust_cost_eur": float(best["cost"]),
                "new_scenario_cost_eur": float(new_cost),
                "budget_used": budget_info["budget_used"],
            }
        )
        if any(np.allclose(worst_net, scenario, atol=1e-9) for scenario in scenarios):
            break
        scenarios.append(worst_net)
    nominal_cost = evaluate_schedule(schedule, net_hat, price_hat, batt, export_ratio)
    worst_cost = max(
        evaluate_schedule(schedule, scenario, price_hat, batt, export_ratio)
        for scenario in scenarios
    )
    return {
        "schedule": schedule,
        "rule": chosen,
        "scenarios": scenarios,
        "history": history,
        "nominal_cost": nominal_cost,
        "worst_cost": worst_cost,
    }


def aggregate(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "q05": float(np.quantile(a, 0.05)),
        "q95": float(np.quantile(a, 0.95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    arr = load_arrays()
    model = load_tft(MODEL_NAME, arr)
    q_test = predict(model, arr["X_test"])
    scaler_mean = arr["scaler_mean"]
    scaler_std = arr["scaler_std"]

    y_real = arr["y_test"] * scaler_std + scaler_mean
    q_real = q_test * scaler_std[None, None, :, None] + scaler_mean[None, None, :, None]
    step = max(1, len(y_real) // 240)
    indices = list(range(0, len(y_real), step))
    batt = BatterySpec()

    records: dict[str, dict] = {}
    rule_counts = {str(g): Counter() for g in GAMMAS}
    for gamma in GAMMAS:
        records[str(gamma)] = {
            "nominal_cost_eur": [],
            "worst_case_cost_eur": [],
            "realized_cost_eur": [],
            "robustness_premium_pct": [],
            "iterations": [],
        }

    per_window: list[dict] = []
    for window_index in indices:
        net_hat = q_real[window_index, :, 0, 2]
        price_hat = q_real[window_index, :, 1, 2]
        net_true = y_real[window_index, :, 0]
        price_true = y_real[window_index, :, 1]
        delta = DELTA_SCALE * np.abs(net_hat) + DELTA_FLOOR_KWH
        item = {"window": int(window_index), "gammas": {}}
        for gamma in GAMMAS:
            result = tsro_ccg_budget(
                net_hat, price_hat, delta, gamma, batt, n_iter=5
            )
            nominal = float(result["nominal_cost"])
            worst = float(result["worst_cost"])
            realized = float(
                evaluate_schedule(
                    result["schedule"], net_true, price_true, batt, EXPORT_RATIO
                )
            )
            premium = 100.0 * (worst - nominal) / max(abs(nominal), 1e-12)
            rec = records[str(gamma)]
            rec["nominal_cost_eur"].append(nominal)
            rec["worst_case_cost_eur"].append(worst)
            rec["realized_cost_eur"].append(realized)
            rec["robustness_premium_pct"].append(premium)
            rec["iterations"].append(len(result["history"]))
            rule_counts[str(gamma)][result["rule"]] += 1
            item["gammas"][str(gamma)] = {
                "nominal_cost_eur": nominal,
                "worst_case_cost_eur": worst,
                "realized_cost_eur": realized,
                "robustness_premium_pct": premium,
                "rule": result["rule"],
                "iterations": len(result["history"]),
            }
        per_window.append(item)

    summary: dict[str, dict] = {}
    for gamma in GAMMAS:
        rec = records[str(gamma)]
        summary[str(gamma)] = {
            "gamma": gamma,
            "gamma_over_d": gamma / H,
            "nominal_cost_eur": aggregate(rec["nominal_cost_eur"]),
            "worst_case_cost_eur": aggregate(rec["worst_case_cost_eur"]),
            "realized_cost_eur": aggregate(rec["realized_cost_eur"]),
            "robustness_premium_pct": aggregate(rec["robustness_premium_pct"]),
            "iterations": aggregate([float(x) for x in rec["iterations"]]),
            "selected_rules": dict(rule_counts[str(gamma)]),
        }

    output = {
        "metadata": {
            "model": MODEL_NAME,
            "seed": SEED,
            "n_windows": len(indices),
            "window_step_hours": step,
            "horizon_hours": H,
            "delta_definition": "delta_t = 0.2 * abs(net_hat_t) + 1 kWh",
            "uncertainty_set": (
                "U_Gamma = {xi: |xi_t-net_hat_t| <= delta_t and "
                "sum_t |xi_t-net_hat_t|/delta_t <= Gamma}"
            ),
            "gamma_values": list(GAMMAS),
            "metric_definition": (
                "robustness_premium_pct = 100 * (worst_case_cost - nominal_cost) "
                "/ nominal_cost"
            ),
            "algorithm_scope": (
                "Heuristic C&CG over three dispatch rules with a separable, "
                "budget-feasible worst-case net profile; not a full bilevel TSRO solver."
            ),
        },
        "summary": summary,
        "per_window": per_window,
    }
    out_json = RESULTS_DIR / "robustness_budget_sweep.json"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Publication-style figure -----------------------------------------
    x = np.array([g / H for g in GAMMAS])
    nominal_mean = np.array([summary[str(g)]["nominal_cost_eur"]["mean"] for g in GAMMAS])
    worst_mean = np.array([summary[str(g)]["worst_case_cost_eur"]["mean"] for g in GAMMAS])
    realized_mean = np.array([summary[str(g)]["realized_cost_eur"]["mean"] for g in GAMMAS])
    premium_mean = np.array([summary[str(g)]["robustness_premium_pct"]["mean"] for g in GAMMAS])
    premium_std = np.array([summary[str(g)]["robustness_premium_pct"]["std"] for g in GAMMAS])

    navy = "#1B3A5C"
    gold = "#B8860B"
    red = "#B23A48"
    slate = "#4A5568"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#AAB0B7",
            "axes.grid": True,
            "grid.color": "#E3E7EB",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, nominal_mean, "o-", color=navy, lw=2, label="Coût nominal")
    ax.plot(x, worst_mean, "o-", color=red, lw=2, label="Pire cas sous budget")
    ax.plot(x, realized_mean, "s--", color=gold, lw=1.8, label="Coût réalisé")
    ax.set_title("A. Coût opérationnel journalier")
    ax.set_xlabel(r"Budget d'incertitude $\Gamma/d$")
    ax.set_ylabel("Coût moyen (EUR / 24 h)")
    ax.set_xticks(x)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(x, premium_mean, "o-", color=red, lw=2, label="Prime de robustesse")
    ax.fill_between(
        x,
        premium_mean - premium_std,
        premium_mean + premium_std,
        color=red,
        alpha=0.13,
        label="± 1 écart-type",
    )
    ax.axhline(0, color=slate, lw=0.9)
    ax.set_title("B. Prime de robustesse")
    ax.set_xlabel(r"Budget d'incertitude $\Gamma/d$")
    ax.set_ylabel("Surcoût pire cas / nominal (%)")
    ax.set_xticks(x)
    ax.legend(loc="best")

    fig.suptitle(
        "Sensibilité de la planification à l'incertitude — 241 fenêtres de test",
        fontsize=12,
        fontweight="bold",
        color=navy,
    )
    out_png = FIGURES_DIR / "robustness_budget_sweep.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"JSON: {out_json}")
    print(f"Figure: {out_png}")
    for gamma in GAMMAS:
        s = summary[str(gamma)]
        print(
            f"Gamma/d={gamma/H:.3f}: nominal={s['nominal_cost_eur']['mean']:.2f} EUR, "
            f"worst={s['worst_case_cost_eur']['mean']:.2f} EUR, "
            f"premium={s['robustness_premium_pct']['mean']:.2f}%"
        )


if __name__ == "__main__":
    main()
