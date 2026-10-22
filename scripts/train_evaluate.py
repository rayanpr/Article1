from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.forecasting.copula import GaussianCopulaScenarios
from src.forecasting.metrics import (
    DEFAULT_TAUS,
    conformal_interval,
    crps_from_scenarios,
    evaluate_forecast,
)
from src.forecasting.tft import TFTQuantile
from src.optimization.ems import BatterySpec, evaluate_schedule, solve_ems
from src.optimization.spo import spo_plus_loss_torch

SEED = 42
H = 24
TAUS = DEFAULT_TAUS
LAMBDA_SPO = 0.3


def set_seed(s: int = SEED) -> None:
    np.random.seed(s)
    torch.manual_seed(s)


def load_arrays() -> dict:
    z = np.load(PROCESSED_DIR / "arrays_2022.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def to_loaders(arr: dict, batch: int = 64) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        ds = TensorDataset(
            torch.tensor(arr[f"X_{split}"], dtype=torch.float32),
            torch.tensor(arr[f"y_{split}"], dtype=torch.float32),
        )
        out[split] = DataLoader(ds, batch_size=batch, shuffle=(split == "train"))
    return out


def train_phase(model, loaders, epochs, lr, phase2: bool, batt: BatterySpec, log: list) -> TFTQuantile:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_val = None, float("inf")
    for ep in range(epochs):
        model.train()
        tr = 0.0
        for xb, yb in loaders["train"]:
            opt.zero_grad()
            q = model(xb)
            loss = model.pinball(yb, q)
            if phase2:
                price_hat = q[:, :, 1, 2]
                price_true = yb[:, :, 1]
                stride = max(1, price_hat.shape[0] // 16)
                loss = loss + LAMBDA_SPO * spo_plus_loss_torch(price_hat[::stride], price_true[::stride], batt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss.detach()) * len(xb)
        tr /= len(loaders["train"].dataset)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                q = model(xb)
                vl += float(model.pinball(yb, q)) * len(xb)
        vl /= len(loaders["val"].dataset)
        log.append({"phase": 2 if phase2 else 1, "epoch": ep, "train_loss": tr, "val_pinball": vl})
        if vl < best_val:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_tft(arr: dict, loaders: dict, epochs1: int = 12, epochs2: int = 6) -> tuple:
    set_seed()
    model = TFTQuantile(n_features=arr["X_train"].shape[-1], horizon=H, n_targets=2, hidden=64, taus=TAUS)
    batt = BatterySpec()
    log: list = []
    t0 = time.time()
    model = train_phase(model, loaders, epochs1, 1e-3, False, batt, log)
    torch.save(copy.deepcopy(model.state_dict()), MODELS_DIR / "tft_pinball.pt")
    model = train_phase(model, loaders, epochs2, 3e-4, True, batt, log)
    torch.save(model.state_dict(), MODELS_DIR / "tft_spo.pt")
    return model, log, time.time() - t0


def load_tft(name: str, arr: dict) -> TFTQuantile:
    model = TFTQuantile(n_features=arr["X_train"].shape[-1], horizon=H, n_targets=2, hidden=64, taus=TAUS)
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def predict(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    qs = []
    for i in range(0, len(X), 256):
        q = model(torch.tensor(X[i : i + 256], dtype=torch.float32))
        qs.append(q.numpy())
    return np.concatenate(qs, axis=0)


class PersistenceBaseline:
    def predict(self, X: np.ndarray) -> np.ndarray:
        last = X[:, -1:, :2]
        q = np.repeat(last[..., None], len(TAUS), axis=-1)
        return np.repeat(q, H, axis=1)

    def fit(self, *a, **k):
        return self


class LightGBMQuantile:
    def __init__(self):
        import lightgbm as lgb

        self.lgb = lgb
        self.models = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMQuantile":
        n, t, f = X.shape
        xf = X.reshape(n * t, f)
        for ti, tgt in enumerate([0, 1]):
            yt = y[:, :, ti].reshape(n * t)
            for qi, tau in enumerate(TAUS):
                m = self.lgb.LGBMRegressor(
                    objective="quantile", alpha=float(tau), n_estimators=100,
                    learning_rate=0.05, num_leaves=31, verbose=-1,
                )
                m.fit(xf, yt)
                self.models[(ti, qi)] = m
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n, t, f = X.shape
        xf = X.reshape(n * t, f)
        out = np.empty((n, H, 2, len(TAUS)))
        for ti in range(2):
            for qi in range(len(TAUS)):
                out[:, :, ti, qi] = self.models[(ti, qi)].predict(xf).reshape(n, H)
        return out


class XGBoostQuantile:
    def __init__(self):
        import xgboost as xgb

        self.xgb = xgb
        self.models = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostQuantile":
        n, t, f = X.shape
        xf = X.reshape(n * t, f)
        for ti, tgt in enumerate([0, 1]):
            yt = y[:, :, ti].reshape(n * t)
            for qi, tau in enumerate(TAUS):
                m = self.xgb.XGBRegressor(
                    objective="reg:quantileerror", quantile_alpha=float(tau),
                    n_estimators=100, learning_rate=0.05, max_depth=6,
                    tree_method="hist", verbosity=0,
                )
                m.fit(xf, yt)
                self.models[(ti, qi)] = m
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n, t, f = X.shape
        xf = X.reshape(n * t, f)
        out = np.empty((n, H, 2, len(TAUS)))
        for ti in range(2):
            for qi in range(len(TAUS)):
                out[:, :, ti, qi] = self.models[(ti, qi)].predict(xf).reshape(n, H)
        return out


def conformalize(q_val: np.ndarray, y_val: np.ndarray, q_test: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    q_test = q_test.copy()
    for ti in range(2):
        med = q_val[:, :, ti, 2]
        lo, hi = conformal_interval(y_val[:, :, ti], med, q_test[:, :, ti, 2], q_test[:, :, ti, 2], alpha=alpha)
        width_lo = np.maximum(0.0, q_test[:, :, ti, 2] - lo)
        width_hi = np.maximum(0.0, hi - q_test[:, :, ti, 2])
        q_test[:, :, ti, 0] = q_test[:, :, ti, 2] - width_lo
        q_test[:, :, ti, -1] = q_test[:, :, ti, 2] + width_hi
    return q_test


def copula_scenarios(model, arr: dict, n_scen: int = 100) -> tuple:
    q_tr = predict(model, arr["X_train"])
    resid_tr = arr["y_train"] - q_tr[:, :, :, 2]
    flat = resid_tr.reshape(-1, 2)
    cop = GaussianCopulaScenarios().fit(flat)
    q_te = predict(model, arr["X_test"])
    resid_te = arr["y_test"] - q_te[:, :, :, 2]
    rng = np.random.default_rng(SEED)
    scen = np.empty((arr["y_test"].shape[0], H, 2, n_scen))
    for t in range(H):
        s = cop.sample(n_scen, rng)
        scen[:, t, :, :] = resid_te[:, t][:, :, None] + s.T[None, :, :]
    return scen


def operational_metrics(q: np.ndarray, y_std: np.ndarray, arr: dict, scaler_mean, scaler_std) -> dict:
    y = y_std * scaler_std + scaler_mean
    med = q[:, :, :, 2] * scaler_std + scaler_mean
    net_hat = med[:, :, 0]
    price_hat = med[:, :, 1]
    batt = BatterySpec()
    cost_pred, cost_true, cost_oracle = [], [], []
    step = max(1, len(y) // 240)
    for i in range(0, len(y), step):
        sol = solve_ems(net_hat[i], price_hat[i], batt)
        if sol["status"] not in ("optimal", "optimal_inaccurate"):
            continue
        sched = sol["discharge"] - sol["charge"]
        cp_ = evaluate_schedule(sched, y[i, :, 0], price_hat[i], batt)
        co = evaluate_schedule(sched, y[i, :, 0], y[i, :, 1], batt)
        ref = solve_ems(y[i, :, 0], y[i, :, 1], batt)
        cost_pred.append(cp_)
        cost_true.append(co)
        cost_oracle.append(ref["cost"] if ref["status"] in ("optimal", "optimal_inaccurate") else np.nan)
    cost_pred = np.array(cost_pred)
    cost_true = np.array(cost_true)
    cost_oracle = np.array(cost_oracle)
    valid = ~np.isnan(cost_oracle)
    return {
        "cost_pred_price": float(np.mean(cost_pred)),
        "cost_true_price": float(np.mean(cost_true)),
        "cost_oracle": float(np.mean(cost_oracle[valid])),
        "regret": float(np.mean(cost_true[valid] - cost_oracle[valid])),
        "regret_pct": float(np.mean((cost_true[valid] - cost_oracle[valid]) / (np.abs(cost_oracle[valid]) + 1e-9) * 100)),
        "cost_std": float(np.std(cost_true)),
        "n_windows": int(valid.sum()),
    }


def main() -> dict:
    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arr = load_arrays()
    loaders = to_loaders(arr)
    scaler_mean = arr["scaler_mean"]
    scaler_std = arr["scaler_std"]

    results: dict = {"taus": TAUS.tolist(), "lambda_spo": LAMBDA_SPO}

    model, log, train_time = train_tft(arr, loaders)
    results["train_time_s"] = round(train_time, 1)
    results["train_log"] = log

    q_test_tft = predict(model, arr["X_test"])
    y_test = arr["y_test"]

    model_pb = load_tft("tft_pinball", arr)
    q_test_pb = predict(model_pb, arr["X_test"])

    preds = {
        "persistence": PersistenceBaseline().predict(arr["X_test"]),
        "lightgbm": LightGBMQuantile().fit(arr["X_train"], arr["y_train"]).predict(arr["X_test"]),
        "xgboost": XGBoostQuantile().fit(arr["X_train"], arr["y_train"]).predict(arr["X_test"]),
        "tft_pinball": q_test_pb,
        "tft_pinball_spo": q_test_tft,
    }

    q_val_tft = predict(model, arr["X_val"])
    preds["tft_conf"] = conformalize(q_val_tft, arr["y_val"], q_test_tft, alpha=0.2)

    metrics = {}
    for name, q in preds.items():
        m_net = evaluate_forecast(y_test[:, :, 0], q[:, :, 0, :], TAUS)
        m_price = evaluate_forecast(y_test[:, :, 1], q[:, :, 1, :], TAUS)
        op = operational_metrics(q, y_test, arr, scaler_mean, scaler_std)
        metrics[name] = {"net": m_net, "price": m_price, "operational": op}

    scen = copula_scenarios(model, arr, n_scen=100)
    scen_real = scen * scaler_std[None, None, :, None] + scaler_mean[None, None, :, None]
    y_real = y_test * scaler_std + scaler_mean
    crps_scen = crps_from_scenarios(y_real[:, :, 0], scen_real[:, :, 0, :])
    crps_scen_price = crps_from_scenarios(y_real[:, :, 1], scen_real[:, :, 1, :])
    metrics["tft_copula_scenarios"] = {
        "crps_scenarios_net": float(crps_scen),
        "crps_scenarios_price": float(crps_scen_price),
        "n_scenarios": int(scen.shape[-1]),
    }

    from src.optimization.tsro import tsro_ccg

    i0 = 0
    net_hat = preds["tft_pinball_spo"][i0, :, 0, 2] * scaler_std[0] + scaler_mean[0]
    price_hat = preds["tft_pinball_spo"][i0, :, 1, 2] * scaler_std[1] + scaler_mean[1]
    tsro = tsro_ccg(net_hat, price_hat, 0.2 * np.abs(net_hat) + 1.0, BatterySpec(), n_iter=4)
    results["tsro_demo"] = {
        "rule": tsro["rule"],
        "nominal_cost": round(tsro["nominal_cost"], 2),
        "worst_case_cost": round(tsro["worst_cost"], 2),
        "n_scenarios_ccg": len(tsro["scenarios"]),
        "iterations": tsro["history"],
    }

    results["metrics"] = metrics
    (RESULTS_DIR / "experiment_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    np.save(RESULTS_DIR / "y_test_real.npy", y_real)
    np.save(
        RESULTS_DIR / "q_test_tft.npy",
        preds["tft_pinball_spo"] * scaler_std[None, None, :, None] + scaler_mean[None, None, :, None],
    )

    print(json.dumps({k: v for k, v in results.items() if k != "train_log"}, indent=2, ensure_ascii=False, default=float))
    return results


if __name__ == "__main__":
    main()
