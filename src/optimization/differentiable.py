from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    HAS_CVXPYLAYERS = True
except Exception:
    HAS_CVXPYLAYERS = False


def build_ems_layer(horizon: int = 24, e_max: float = 100.0, e_min: float = 0.1,
                    p_ch: float = 30.0, p_dis: float = 30.0,
                    eta_ch: float = 0.95, eta_dis: float = 0.95, soc0: float = 0.5,
                    export_ratio: float = 0.5):
    if not HAS_CVXPYLAYERS:
        raise ImportError("cvxpylayers indisponible (installer diffcp — voir .venv311)")
    net = cp.Parameter(horizon)
    price = cp.Parameter(horizon, nonneg=True)
    g = cp.Variable(horizon, nonneg=True)
    x = cp.Variable(horizon, nonneg=True)
    ch = cp.Variable(horizon, nonneg=True)
    dis = cp.Variable(horizon, nonneg=True)
    e = cp.Variable(horizon + 1)
    cons = [
        e[0] == soc0 * e_max,
        e[1:] == e[:-1] + eta_ch * ch - dis / eta_dis,
        e >= e_min * e_max,
        e <= e_max,
        ch <= p_ch,
        dis <= p_dis,
        g - x == net + ch - dis,
    ]
    cost = cp.sum(cp.multiply(price, g) - cp.multiply(price * export_ratio, x))
    prob = cp.Problem(cp.Minimize(cost), cons)
    layer = CvxpyLayer(prob, parameters=[net, price], variables=[g, x, ch, dis, e])
    return layer


def solve_with_layer(layer, net, price):
    import torch

    g, x, ch, dis, e = layer(net, price, solver_args={"solve_method": "highs"})
    return {"import": g, "export": x, "charge": ch, "discharge": dis, "soc": e}
