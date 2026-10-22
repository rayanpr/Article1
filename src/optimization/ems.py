from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BatterySpec:
    e_max: float = 100.0
    e_min: float = 0.1
    p_ch: float = 30.0
    p_dis: float = 30.0
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    soc0: float = 0.5

    @property
    def e0(self) -> float:
        return self.soc0 * self.e_max


def solve_ems(net: np.ndarray, price: np.ndarray, batt: BatterySpec, export_ratio: float = 0.5):
    import cvxpy as cp

    h = len(net)
    g = cp.Variable(h, nonneg=True)
    x = cp.Variable(h, nonneg=True)
    ch = cp.Variable(h, nonneg=True)
    dis = cp.Variable(h, nonneg=True)
    e = cp.Variable(h + 1)

    cons = [
        e[0] == batt.e0,
        e[1:] == e[:-1] + batt.eta_ch * ch - dis / batt.eta_dis,
        e >= batt.e_min * batt.e_max,
        e <= batt.e_max,
        ch <= batt.p_ch,
        dis <= batt.p_dis,
        g - x == net + ch - dis,
    ]
    cost = cp.sum(cp.multiply(price, g) - cp.multiply(price * export_ratio, x))
    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver=cp.HIGHS, warm_start=True)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": prob.status, "cost": float("inf")}
    return {
        "status": prob.status,
        "cost": float(prob.value),
        "import": g.value,
        "export": x.value,
        "charge": ch.value,
        "discharge": dis.value,
        "soc": e.value,
    }


def clairvoyant_cost(net: np.ndarray, price: np.ndarray, batt: BatterySpec, export_ratio: float = 0.5) -> float:
    return float(solve_ems(net, price, batt, export_ratio)["cost"])


def evaluate_schedule(
    schedule: np.ndarray, net: np.ndarray, price: np.ndarray, batt: BatterySpec, export_ratio: float = 0.5
) -> float:
    h = len(net)
    soc = np.empty(h + 1)
    soc[0] = batt.e0
    imp = np.zeros(h)
    exp = np.zeros(h)
    for t in range(h):
        dis = max(0.0, min(batt.p_dis, schedule[t]))
        ch = max(0.0, min(batt.p_ch, -schedule[t]))
        room_ch = (batt.e_max - soc[t]) * batt.eta_ch
        room_dis = (soc[t] - batt.e_min * batt.e_max) * batt.eta_dis
        dis = min(dis, room_dis)
        ch = min(ch, room_ch)
        soc[t + 1] = soc[t] + batt.eta_ch * ch - dis / batt.eta_dis
        net_t = net[t] + ch - dis
        imp[t] = max(0.0, net_t)
        exp[t] = max(0.0, -net_t)
    return float(np.sum(price * imp - price * export_ratio * exp))
