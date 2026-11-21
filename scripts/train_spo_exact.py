from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0]))
sys.path.insert(0, str(_HERE))

from config.paths import MODELS_DIR, RESULTS_DIR
from src.forecasting.metrics import evaluate_forecast
from src.optimization.differentiable import build_ems_layer, spo_exact_loss
from train_evaluate import (
    H,
    SEED,
    TAUS,
    load_arrays,
    load_tft,
    operational_metrics,
    predict,
    set_seed,
    to_loaders,
)

LAMBDA_GRID = [0.05, 0.1, 0.3]
BETA = 1.0
EPOCHS_MAX = 6
PATIENCE = 2
LR = 3e-4
STRIDE = 8
BATCH = 64
EXPORT_RATIO = 0.5


def decision_scale(arr: dict) -> float:
    m, s = arr["scaler_mean"], arr["scaler_std"]
    y = arr["y_train"] * s + m
    return float(np.mean(np.sum(np.abs(y[:, :, 1] * y[:, :, 0]), axis=1)))


def run_lambda(lam: float, arr: dict, layer, scale: float, beta: float = BETA,
               epochs_max: int = EPOCHS_MAX, stride: int = STRIDE, smoke: bool = False) -> dict:
    set_seed()
    loaders = to_loaders(arr, batch=BATCH)
    model = load_tft("tft_pinball", arr)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sm, ss = arr["scaler_mean"], arr["scaler_std"]
    q0 = predict(model, arr["X_val"])
    init_regret = operational_metrics(q0, arr["y_val"], arr, sm, ss)["regret_pct"]
    best_state = copy.deepcopy(model.state_dict())
    best_regret, bad, log = init_regret, 0, [{"epoch": -1, "val_regret_pct": round(init_regret, 4), "phase": "depart"}]
    print(f"  lam={lam} beta={beta} depart val_regret={init_regret:.3f}%", flush=True)
    t0 = time.time()
    epochs = 1 if smoke else epochs_max
    for ep in range(epochs):
        model.train()
        tr, nb = 0.0, 0
        for xb, yb in loaders["train"]:
            opt.zero_grad()
            q = model(xb)
            loss = model.pinball(yb, q)
            loss = loss + lam * spo_exact_loss(
                layer,
                q[::stride, :, 0, 2], q[::stride, :, 1, 2],
                yb[::stride, :, 0], yb[::stride, :, 1],
                sm, ss, export_ratio=EXPORT_RATIO, beta=beta, scale=scale,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss.detach()) * len(xb)
            nb += len(xb)
        tr /= nb

        model.eval()
        q_val = predict(model, arr["X_val"])
        op = operational_metrics(q_val, arr["y_val"], arr, sm, ss)
        val_pb = float(loaders_val_pinball(model, arr))
        log.append({"epoch": ep, "train_loss": round(tr, 5),
                    "val_pinball": round(val_pb, 5),
                    "val_regret_pct": round(op["regret_pct"], 4)})
        print(f"  lam={lam} ep={ep} train={tr:.4f} val_pb={val_pb:.4f} "
              f"val_regret={op['regret_pct']:.3f}%", flush=True)

        if op["regret_pct"] < best_regret - 1e-4:
            best_regret = op["regret_pct"]
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  lam={lam} early stop apres ep {ep}", flush=True)
                break
    return {"lambda": lam, "best_val_regret_pct": round(best_regret, 4),
            "init_val_regret_pct": round(init_regret, 4),
            "epochs_run": len(log) - 1, "train_time_s": round(time.time() - t0, 1),
            "log": log, "best_state": best_state}


def loaders_val_pinball(model, arr: dict) -> float:
    model.eval()
    q = predict(model, arr["X_val"])
    y = torch.tensor(arr["y_val"], dtype=torch.float32)
    qt = torch.tensor(q, dtype=torch.float32)
    return float(model.pinball(y, qt))


def env_info() -> dict:
    import cvxpy
    import cvxpylayers
    import diffcp

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cvxpy": cvxpy.__version__,
        "cvxpylayers": getattr(cvxpylayers, "__version__", "?"),
        "diffcp": getattr(diffcp, "__version__", "?"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 lambda, 1 epoque")
    ap.add_argument("--lambdas", type=float, nargs="*", default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--tag", type=str, default="", help="suffixe pour fichiers de sortie")
    args = ap.parse_args()

    beta = BETA if args.beta is None else args.beta
    epochs_max = EPOCHS_MAX if args.epochs is None else args.epochs
    stride = STRIDE if args.stride is None else args.stride

    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arr = load_arrays()
    scale = decision_scale(arr)
    layer = build_ems_layer(H, export_ratio=EXPORT_RATIO)
    grid = list(args.lambdas) if args.lambdas else ([0.1] if args.smoke else LAMBDA_GRID)

    print(f"echelle de normalisation = {scale:.1f} | lambdas = {grid} | beta = {beta} "
          f"| epochs = {epochs_max} | stride = {stride}", flush=True)
    results: dict = {
        "config": {
            "lambda_grid": grid, "beta_recourse": beta, "epochs_max": epochs_max,
            "patience": PATIENCE, "lr": LR, "stride": stride, "batch": BATCH,
            "export_ratio": EXPORT_RATIO, "seed": SEED, "scale": round(scale, 2),
            "start_checkpoint": "tft_pinball.pt", "solver": "ECOS",
            "env": env_info(),
        },
        "grid": [],
    }

    t_all = time.time()
    best_overall = (None, float("inf"), None)
    for lam in grid:
        r = run_lambda(lam, arr, layer, scale, beta=beta, epochs_max=epochs_max,
                       stride=stride, smoke=args.smoke)
        state = r.pop("best_state")
        results["grid"].append(r)
        if r["best_val_regret_pct"] < best_overall[1]:
            best_overall = (state, r["best_val_regret_pct"], lam)

    best_state, best_val, chosen = best_overall
    print(f"lambda retenu = {chosen} (val regret {best_val:.3f}%)", flush=True)
    results["chosen_lambda"] = chosen
    results["best_val_regret_pct"] = best_val

    if not args.smoke and best_state is not None:
        model = load_tft("tft_pinball", arr)
        model.load_state_dict(best_state)
        torch.save(copy.deepcopy(best_state), MODELS_DIR / f"tft_spo_exact{args.tag}.pt")

        sm, ss = arr["scaler_mean"], arr["scaler_std"]
        q_test = predict(model, arr["X_test"])
        q_val = predict(model, arr["X_val"])
        results["val"] = {
            "net": evaluate_forecast(arr["y_val"][:, :, 0], q_val[:, :, 0, :], TAUS),
            "price": evaluate_forecast(arr["y_val"][:, :, 1], q_val[:, :, 1, :], TAUS),
            "pinball": round(loaders_val_pinball(model, arr), 5),
        }
        results["test"] = {
            "net": evaluate_forecast(arr["y_test"][:, :, 0], q_test[:, :, 0, :], TAUS),
            "price": evaluate_forecast(arr["y_test"][:, :, 1], q_test[:, :, 1, :], TAUS),
            "operational": operational_metrics(q_test, arr["y_test"], arr, sm, ss),
        }

        base_file = RESULTS_DIR / "experiment_results.json"
        if base_file.exists():
            base = json.loads(base_file.read_text(encoding="utf-8"))
            results["baseline_test_regret_pct"] = {
                k: v["operational"]["regret_pct"]
                for k, v in base.get("metrics", {}).items() if "operational" in v
            }

    results["total_time_s"] = round(time.time() - t_all, 1)
    if args.smoke:
        out = RESULTS_DIR / "spo_exact_smoke.json"
    elif args.tag:
        out = RESULTS_DIR / f"spo_exact{args.tag}.json"
    else:
        out = RESULTS_DIR / "spo_exact_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "grid"},
                     indent=2, ensure_ascii=False, default=float), flush=True)
    print(f"resultats ecrits : {out}", flush=True)


if __name__ == "__main__":
    main()
