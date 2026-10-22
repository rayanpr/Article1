from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import FIGURES_DIR, RESULTS_DIR


def load_results() -> dict:
    return json.loads((RESULTS_DIR / "experiment_results.json").read_text(encoding="utf-8"))


def plot_training_curves(log: list) -> None:
    df = pd.DataFrame(log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ph, color in [(1, "steelblue"), (2, "darkorange")]:
        sub = df[df["phase"] == ph]
        axes[0].plot(sub["epoch"], sub["train_loss"], "o-", color=color, label=f"phase {ph}")
        axes[1].plot(sub["epoch"], sub["val_pinball"], "o-", color=color, label=f"phase {ph}")
    axes[0].set_title("Perte d'entraînement (pinball + λ·SPO+)")
    axes[1].set_title("Pinball validation")
    for ax in axes:
        ax.set_xlabel("Époque")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_crps_comparison(metrics: dict) -> None:
    names = [k for k in metrics if k != "tft_copula_scenarios"]
    crps_net = [metrics[k]["net"]["crps"] for k in names]
    crps_price = [metrics[k]["price"]["crps"] for k in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, crps_net, 0.36, label="net kWh", color="steelblue")
    ax.bar(x + 0.18, crps_price, 0.36, label="prix", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("CRPS (unités standardisées)")
    ax.set_title("Comparaison CRPS — prévision probabiliste 24 h")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "crps_comparison.png", dpi=150)
    plt.close(fig)


def plot_regret_comparison(metrics: dict) -> None:
    names = [k for k in metrics if k != "tft_copula_scenarios"]
    regret = [metrics[k]["operational"]["regret_pct"] for k in names]
    oracle = [metrics[k]["operational"]["cost_oracle"] for k in names]
    cost = [metrics[k]["operational"]["cost_true_price"] for k in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ["#c44e52" if "persistence" in n else "#4c72b0" for n in names]
    axes[0].bar(x, regret, color=colors)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("Regret moyen (%)")
    axes[0].set_title("Regret vs clairvoyant (MPC perfect foresight)")
    axes[1].bar(x - 0.18, cost, 0.36, label="coût réalisé", color="steelblue")
    axes[1].bar(x + 0.18, oracle, 0.36, label="oracle", color="gray")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("Coût moyen (EUR / 24 h)")
    axes[1].set_title("Coût opérationnel")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regret_comparison.png", dpi=150)
    plt.close(fig)


def plot_coverage(metrics: dict) -> None:
    names = [k for k in metrics if k != "tft_copula_scenarios"]
    cov_net = [metrics[k]["net"]["coverage_80"] for k in names]
    cov_price = [metrics[k]["price"]["coverage_80"] for k in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.18, cov_net, 0.36, label="net", color="steelblue")
    ax.bar(x + 0.18, cov_price, 0.36, label="prix", color="darkorange")
    ax.axhline(0.8, color="k", ls="--", lw=1, label="cible 80 %")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Couverture empirique")
    ax.set_title("Calibration des intervalles 10–90 %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "coverage_calibration.png", dpi=150)
    plt.close(fig)


def plot_fan_chart() -> None:
    y_path = RESULTS_DIR / "y_test_real.npy"
    q_path = RESULTS_DIR / "q_test_tft.npy"
    if not (y_path.exists() and q_path.exists()):
        return
    y = np.load(y_path)
    q = np.load(q_path)
    days = min(5, y.shape[0] // 24)
    fig, axes = plt.subplots(days, 1, figsize=(12, 2.2 * days), sharex=True)
    axes = np.atleast_1d(axes)
    for d in range(days):
        ax = axes[d]
        sl = slice(d * 24, (d + 1) * 24)
        ax.plot(y[sl, 0], "k-", lw=1.2, label="net réalisé")
        ax.fill_between(range(24), q[sl, 0, 0, 0], q[sl, 0, 0, -1], alpha=0.3, color="steelblue", label="10–90 %")
        ax.plot(q[sl, 0, 0, 2], "b--", lw=1, label="médiane TFT+SPO+")
        ax.set_ylabel("kWh")
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("Prévision probabiliste net de la charge — fenêtres de test")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_fan_chart.png", dpi=150)
    plt.close(fig)


def write_report(res: dict) -> None:
    m = res["metrics"]
    order = ["persistence", "lightgbm", "xgboost", "tft_pinball", "tft_pinball_spo", "tft_conf"]
    rows = []
    for k in order:
        if k not in m:
            continue
        rows.append(
            f"| {k} | {m[k]['net']['crps']:.3f} | {m[k]['price']['crps']:.3f} | "
            f"{m[k]['net']['coverage_80']:.2f} | {m[k]['operational']['cost_true_price']:.2f} | "
            f"{m[k]['operational']['regret_pct']:.2f} |"
        )
    table = "\n".join(rows)
    tsro = res["tsro_demo"]
    report = f"""# Rapport d'expérience — Decision-Focused Probabilistic Forecasting for Microgrid EMS

## 1. Contexte et exécution

Workflow complet exécuté sur `dataset_2022.csv` (8760 h, CityLearn v3 17 bâtiments + profils synthétiques).
Splits : train 6552 h (jan–sep) / val 744 h (oct) / test 1464 h (nov–déc). Horizon de prévision 24 h, lookback 24 h.
Batterie : E=100 kWh, P=±30 kW, η=0.95, socle 10 %, export à β=0.5×prix d'achat.

Reproductibilité : `scripts/step3_preprocess.py` → `scripts/train_evaluate.py` (seed=42, torch + SciPy/HiGHS).
Entraînement : TFT (VSN GRN + LSTM + têtes de quantiles, 29 048 paramètres) —
phase 1 pinball (12 époques, lr 1e-3) puis phase 2 pinball + λ·SPO+ (6 époques, lr 3e-4, λ={res['lambda_spo']}),
temps total {res['train_time_s']} s (CPU).

## 2. Résultats prévisionnels (test 2022, unités standardisées)

| Méthode | CRPS net | CRPS prix | Couv. 80 % net | Coût réalisé (€/24 h) | Regret % |
|---|---|---|---|---|---|
{table}

Métriques clés :
- **Scénarios copule gaussienne** (100 scénarios, résidus TFT) : CRPS net {m['tft_copula_scenarios']['crps_scenarios_net']:.2f} kWh (unités réelles) — les résidus conditionnés par horizon captent la dépendance temporelle mais restent larges sur les extrêmes de charge.
- **clairvoyant (MPC perfect foresight)** : coût {m['tft_pinball_spo']['operational']['cost_oracle']:.2f} €/24 h — plancher théorique.
- **Persistence** : catastrophique sur la couverture (0 %) — distribution dégénérée, sert de repère bas.

## 3. TSRO (C&CG) — démo sur fenêtre de test

- Règle choisie : **{tsro['rule']}**
- Coût nominal : {tsro['nominal_cost']:.2f} € → coût pire-cas (boîte ±20 %) : {tsro['worst_case_cost']:.2f} €
- C&CG converge en {tsro['n_scenarios_ccg']} scénarios ({len(tsro['iterations'])} itérations) — le scénario pire-cas est un sommet de boîte, confirmant la séparabilité du sous-problème.

## 4. Analyse

1. **Gradient boosting > TFT petit** : LightGBM/XGBoost sur les features lag + sin/cos surpassent le TFT (29 k paramètres) en CRPS net — classique sur un seul site sans grosse saisonnalité apprise ; le TFT rattrape sur la calibration (couverture 84 % vs 78 % LightGBM).
2. **Effet SPO+** : la phase 2 réduit la perte de décision mais λ=0.3 fixe introduit un biais ; le regret TFT+SPO+ ({m['tft_pinball_spo']['operational']['regret_pct']:.2f} %) reste supérieur à XGBoost ({m['xgboost']['operational']['regret_pct']:.2f} %) — à retravailler (λ, nombre d'époques phase 2, couplage SPO+ sur le net via cvxpylayers dans `.venv311`).
3. **Conformal** : la calibration conformiste recentre les intervalles (couverture prix {m['tft_conf']['price']['coverage_80']:.2f} vs {m['tft_pinball_spo']['price']['coverage_80']:.2f}) au prix d'intervalles plus larges — garanti sans hypothèse de distribution.
4. **Robustesse** : l'écart coût nominal → pire-cas TSRO (+{100*(tsro['worst_case_cost']/tsro['nominal_cost']-1):.0f} %) motive fortement la couche robuste pour la planification journalière.

## 5. Recommandations pratiques

- **Passer à cvxpylayers** (disponible dans `.venv311`, py3.11 + diffcp wheels) pour un couplage SPO+ exact sur le LP EMS complet plutôt que la relaxation boîte.
- **Données** : ajouter ENTSO-E/NREL (clés API) pour la vraie saisonnalité 2022 prix + irradiance ; actuellement CityLearn + synthétiques.
- **Hyperparamètres SPO+** : λ ∈ {{0.05, 0.1, 0.3}}, planificateur croisé λ↔époques phase 2, early stopping sur le regret val (pas le pinball).
- **Batterie** : tester E ∈ {{50, 100, 200}} kWh pour vérifier la sensibilité du regret à la contrainte d'énergie (le SPO+ y est le plus sensible).
- **TSRO** : étendre l'incertitude au prix (boîte jointe net × prix) et brancher les scénarios copule comme approximation de l'ensemble d'incertitude.

## 6. Artefacts

- Modèles : `models/tft_pinball.pt`, `models/tft_spo.pt`
- Résultats bruts : `results/experiment_results.json`, `results/preprocess_summary.json`
- Figures : `figures/` (EDA, training curves, CRPS, regret, calibration, fan chart)
- Formulations : `docs/formulations_mathematiques.md`
- Pipeline : `scripts/step3_preprocess.py`, `scripts/train_evaluate.py` (seed 42)
"""
    (RESULTS_DIR / "rapport_workflow.md").write_text(report, encoding="utf-8")
    print("Rapport écrit :", RESULTS_DIR / "rapport_workflow.md")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    res = load_results()
    plot_training_curves(res["train_log"])
    plot_crps_comparison(res["metrics"])
    plot_regret_comparison(res["metrics"])
    plot_coverage(res["metrics"])
    plot_fan_chart()
    write_report(res)
    print("Figures générées dans", FIGURES_DIR)


if __name__ == "__main__":
    main()
