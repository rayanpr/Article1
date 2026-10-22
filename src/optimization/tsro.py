from __future__ import annotations

import numpy as np

from src.optimization.ems import BatterySpec, evaluate_schedule


def hourly_cost(net_t: float, u_t: float, price_t: float, export_ratio: float) -> float:
    net_after = net_t - u_t
    return price_t * max(0.0, net_after) - export_ratio * price_t * max(0.0, -net_after)


def worst_case_net(
    net_hat: np.ndarray, delta: np.ndarray, schedule: np.ndarray, price: np.ndarray, export_ratio: float
) -> np.ndarray:
    lo = net_hat - delta
    hi = net_hat + delta
    c_lo = np.array([hourly_cost(lo[t], schedule[t], price[t], export_ratio) for t in range(len(net_hat))])
    c_hi = np.array([hourly_cost(hi[t], schedule[t], price[t], export_ratio) for t in range(len(net_hat))])
    return np.where(c_hi >= c_lo, hi, lo)


def default_rules() -> dict:
    def greedy_discharge(net_hat, price_hat, batt):
        h = len(net_hat)
        u = np.zeros(h)
        energy = batt.e0 - batt.e_min * batt.e_max
        for t in np.argsort(-price_hat):
            d = min(batt.p_dis, energy * batt.eta_dis)
            u[t] = d
            energy -= d / batt.eta_dis
            if energy <= 1e-9:
                break
        return u

    def greedy_charge_surplus(net_hat, price_hat, batt):
        h = len(net_hat)
        u = np.zeros(h)
        room = (batt.e_max - batt.e0) * batt.eta_ch
        for t in np.argsort(price_hat):
            c = min(batt.p_ch, room)
            u[t] = -c
            room -= c * batt.eta_ch
            if room <= 1e-9:
                break
        return u

    def net_aware(net_hat, price_hat, batt):
        h = len(net_hat)
        u = np.zeros(h)
        energy = batt.e0 - batt.e_min * batt.e_max
        e_max_usable = batt.e_max - batt.e_min * batt.e_max
        for t in np.argsort(-np.maximum(net_hat, 0.0)):
            surplus = max(0.0, -net_hat[t])
            if surplus > 1e-9:
                room = (e_max_usable - energy) * batt.eta_ch
                c = min(batt.p_ch, surplus, max(0.0, room))
                u[t] = -c
                energy += c * batt.eta_ch
        for t in np.argsort(-price_hat):
            if u[t] < 0.0:
                continue
            room_e = max(0.0, energy - 0.0) * batt.eta_dis
            d = min(batt.p_dis, room_e, max(0.0, net_hat[t]))
            u[t] = d
            energy -= d / batt.eta_dis
            if energy <= 1e-9:
                break
        return u

    return {"greedy_discharge": greedy_discharge, "greedy_charge": greedy_charge_surplus, "net_aware": net_aware}


def tsro_ccg(
    net_hat: np.ndarray,
    price_hat: np.ndarray,
    delta: np.ndarray,
    batt: BatterySpec,
    export_ratio: float = 0.5,
    n_iter: int = 5,
    sched_rules: dict | None = None,
) -> dict:
    scenarios = [net_hat.copy()]
    history = []
    schedule = None
    chosen = None
    for it in range(n_iter):
        best = None
        for rule_name, rule in (sched_rules or default_rules()).items():
            u = rule(net_hat, price_hat, batt)
            worst_cost = max(evaluate_schedule(u, y, price_hat, batt, export_ratio) for y in scenarios)
            if best is None or worst_cost < best["cost"] - 1e-9:
                best = {"cost": worst_cost, "schedule": u, "rule": rule_name}
        schedule = best["schedule"]
        chosen = best["rule"]
        y_wc = worst_case_net(net_hat, delta, schedule, price_hat, export_ratio)
        new_scen_cost = evaluate_schedule(schedule, y_wc, price_hat, batt, export_ratio)
        history.append(
            {"iter": it, "rule": chosen, "robust_cost": best["cost"], "new_scenario_cost": new_scen_cost}
        )
        if any(np.allclose(y_wc, s, atol=1e-9) for s in scenarios):
            break
        scenarios.append(y_wc)
    nominal_cost = evaluate_schedule(schedule, net_hat, price_hat, batt, export_ratio)
    worst_cost = max(evaluate_schedule(schedule, y, price_hat, batt, export_ratio) for y in scenarios)
    return {
        "schedule": schedule,
        "rule": chosen,
        "scenarios": scenarios,
        "history": history,
        "nominal_cost": nominal_cost,
        "worst_cost": worst_cost,
    }
