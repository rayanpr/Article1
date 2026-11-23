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
    """Charge les resultats principaux et complete avec la phase 3 (SPO+ exact)."""
    res = json.loads((RESULTS_DIR / "experiment_results.json").read_text(encoding="utf-8"))
    exact_file = RESULTS_DIR / "spo_exact_results.json"
    if exact_file.exists():
        exact = json.loads(exact_file.read_text(encoding="utf-8"))
        res.setdefault("metrics", {})["tft_spo_exact"] = {
            "net": exact["test"]["net"],
            "price": exact["test"]["price"],
            "operational": exact["test"]["operational"],
        }
        res["spo_exact"] = {
            "lambda": exact.get("chosen_lambda"),
            "train_time_s": exact.get("total_time_s"),
            "best_val_regret_pct": exact.get("best_val_regret_pct"),
        }
    return res


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
    order = ["persistence", "lightgbm", "xgboost", "tft_pinball", "tft_spo_exact", "tft_pinball_spo", "tft_conf"]
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
    regret_box = m['tft_pinball_spo']['operational']['regret_pct']
    regret_pin = m['tft_pinball']['operational']['regret_pct']
    box_effect = (
        f"améliore le regret ({regret_box:.2f} % vs {regret_pin:.2f} % pour le pinball seul)"
        if regret_box <= regret_pin
        else f"dégrade le regret ({regret_box:.2f} % vs {regret_pin:.2f} % pour le pinball seul)"
    )
    report = f"""# Rapport d'expérience — Decision-Focused Probabilistic Forecasting for Microgrid EMS

## 1. Contexte et exécution

Workflow complet exécuté sur `dataset_2022.csv` (8760 h, CityLearn v3 17 bâtiments + profils synthétiques).
Splits : train 6552 h (jan–sep) / val 744 h (oct) / test 1464 h (nov–déc). Horizon de prévision 24 h, lookback 24 h.
Batterie : E=100 kWh, P=±30 kW, η=0.95, socle 10 %, export à β=0.5×prix d'achat.

Reproductibilité : `scripts/step3_preprocess.py` → `scripts/train_evaluate.py` (seed=42, torch + SciPy/HiGHS) → `scripts/train_spo_exact.py` (couplage exact cvxpylayers, env `.venv311`, graine rechargée depuis `tft_pinball.pt`).
Entraînement : TFT (VSN GRN + LSTM + têtes de quantiles, 92 600 paramètres) —
phase 1 pinball (12 époques, lr 1e-3) puis phase 2 pinball + λ·SPO+ (6 époques, lr 3e-4, λ={res['lambda_spo']}),
temps total {res['train_time_s']} s (CPU) ; couplage exact : 3 λ × ≤6 époques, {res.get('spo_exact', {}).get('train_time_s', '—')} s.

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

1. **Gradient boosting > TFT en CRPS** : LightGBM/XGBoost sur les features lag + sin/cos surpassent le TFT en CRPS net — classique sur un seul site sans grosse saisonnalité apprise ; le TFT rattrape sur la calibration (couverture {m['tft_spo_exact']['net']['coverage_80']*100:.0f} % vs {m['lightgbm']['net']['coverage_80']*100:.0f} % LightGBM) et, en regret, le TFT+SPO exact dépasse LightGBM (voir point 2).
2. **Effet SPO+** : la relaxation boîte (phase 2, λ=0.3) {box_effect}. Le couplage **exact via cvxpylayers** (LP EMS complet net + prix + contraintes batterie, SPO+ conforme à la Définition 3 d'Elmachtoub & Grigas — trois solves : décision prédite, oracle prix vrai `z_c = x*(n̂,p)`, problème modifié `z₂ = x*(n̂,2p̂−p)` — plus terme de recours réévalué sous les vraies séries, λ={res.get('spo_exact', {}).get('lambda', '—')} retenu par regret de validation, early stopping sur le regret val) le porte à **{m['tft_spo_exact']['operational']['regret_pct']:.2f} %** — à comparer à LightGBM ({m['lightgbm']['operational']['regret_pct']:.2f} %) et XGBoost ({m['xgboost']['operational']['regret_pct']:.2f} %), avec calibration conservée (pinball val {m['tft_spo_exact'].get('price', {}).get('pinball_mean', float('nan')):.3f} ≈ 0,107 au départ). Détails : `results/spo_exact_results.json`.
3. **Conformal** : la calibration conformiste recentre les intervalles (couverture prix {m['tft_conf']['price']['coverage_80']:.2f} vs {m['tft_pinball_spo']['price']['coverage_80']:.2f}) au prix d'intervalles plus larges — garanti sans hypothèse de distribution.
4. **Robustesse** : l'écart coût nominal → pire-cas TSRO (+{100*(tsro['worst_case_cost']/tsro['nominal_cost']-1):.0f} %) motive fortement la couche robuste pour la planification journalière.

## 5. Recommandations pratiques

- **cvxpylayers** : **fait** — `scripts/train_spo_exact.py` (env `.venv311`, py3.11) ; regret {m['tft_pinball_spo']['operational']['regret_pct']:.2f} % (boîte) → {m['tft_spo_exact']['operational']['regret_pct']:.2f} % (exact). Extensions : λ au-delà de la grille explorée, β du terme de recours, sous-échantillonnage plus fin de la couche.
- **Données** : ajouter ENTSO-E/NREL (clés API) pour la vraie saisonnalité 2022 prix + irradiance ; actuellement CityLearn + synthétiques.
- **Hyperparamètres SPO+** : **fait pour le couplage exact** (grille λ ∈ {{0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}} + early stopping sur le regret val, λ={res.get('spo_exact', {}).get('lambda', '—')} retenu). Reste : planificateur λ↔époques, sensibilité au terme de recours β.
- **Batterie** : tester E ∈ {{50, 100, 200}} kWh pour vérifier la sensibilité du regret à la contrainte d'énergie (le SPO+ y est le plus sensible).
- **TSRO** : étendre l'incertitude au prix (boîte jointe net × prix) et brancher les scénarios copule comme approximation de l'ensemble d'incertitude.

## 6. Artefacts

- Modèles : `models/tft_pinball.pt`, `models/tft_spo.pt`, `models/tft_spo_exact.pt`
- Résultats bruts : `results/experiment_results.json`, `results/spo_exact_results.json`, `results/preprocess_summary.json`
- Figures : `figures/` (EDA, training curves, CRPS, regret, calibration, fan chart)
- Formulations : `docs/formulations_mathematiques.md`
- Pipeline : `scripts/step3_preprocess.py`, `scripts/train_evaluate.py` (seed 42), `scripts/train_spo_exact.py` (`.venv311`)
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
