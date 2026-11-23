"""Build publication-style figures from the saved Article1 JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

NAVY = "#1B3A5C"
GOLD = "#B8860B"
RED = "#B23A48"
SLATE = "#4A5568"
GREEN = "#3D7A57"
LIGHT = "#DCE4EB"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def style() -> None:
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


def regret_figure(base: dict) -> None:
    labels = [
        "Persistence",
        "LightGBM",
        "XGBoost",
        "TFT\npinball",
        "TFT+SPO+\nboîte",
        "TFT+SPO+\nexact",
        "TFT\nconformal",
    ]
    keys = [
        "persistence",
        "lightgbm",
        "xgboost",
        "tft_pinball",
        "tft_pinball_spo",
        "tft_spo_exact",
        "tft_conf",
    ]
    values = [base["metrics"][k]["operational"]["regret_pct"] for k in keys]
    colors = [LIGHT, "#9FB3C5", "#7F98AF", "#AFC4D4", "#D4DCE4", GOLD, "#C7D2DC"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, edgecolor=NAVY, linewidth=0.6)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8, color=SLATE)
    ax.set_ylabel("Regret moyen vs oracle clairvoyant (%)")
    ax.set_title("Performance décisionnelle sur 241 fenêtres de test (seed 42)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    gain = 100.0 * (values[3] - values[5]) / values[3]
    ax.annotate(
        f"TFT+SPO+ exact : −{gain:.1f} %\nde regret vs TFT pinball",
        xy=(5, values[5]),
        xytext=(3.5, max(values) * 0.88),
        arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 1.2},
        fontsize=8.5,
        color=NAVY,
    )
    out = FIGURES / "professional_regret_comparison.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def tradeoff_figure(base: dict) -> None:
    methods = {
        "Persistence": ("persistence", "#AAB7C4", (5, 4)),
        "LightGBM": ("lightgbm", "#6E8DA7", (8, 8)),
        "XGBoost": ("xgboost", "#4F7393", (8, -18)),
        "TFT pinball": ("tft_pinball", NAVY, (8, 8)),
        "TFT+SPO+ boîte": ("tft_pinball_spo", "#9BA7B2", (-82, 12)),
        "TFT+SPO+ exact": ("tft_spo_exact", GOLD, (-45, 12)),
        "TFT conformal": ("tft_conf", "#B8C3CD", (-85, -20)),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for label, (key, color, offset) in methods.items():
        m = base["metrics"][key]
        x = m["net"]["crps"]
        y = m["operational"]["regret_pct"]
        ax.scatter(x, y, s=72, color=color, edgecolor="white", linewidth=0.8, label=label, zorder=3)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=7.5, color=SLATE)
    ax.set_xlabel("CRPS net (unités standardisées)")
    ax.set_ylabel("Regret opérationnel (%)")
    ax.set_title("Le meilleur CRPS n'est pas nécessairement la meilleure décision")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    out = FIGURES / "professional_crps_regret_tradeoff.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def _sweep_sources() -> list[tuple[str, float]]:
    """Balayages lambda complementaires (fichier JSON, lambda cible)."""
    return [
        ("spo_exact_lam05b10.json", 0.5),
        ("spo_exact_lam10b10.json", 1.0),
        ("spo_exact_lam15b10.json", 1.5),
        ("spo_exact_lam20b10.json", 2.0),
        ("spo_exact_lam2.json", 3.0),
    ]


def _lambda_rows() -> list[tuple[float, float, float | None]]:
    """(lambda, regret validation, regret test) pour chaque lambda explore.

    Source primaire : grille de reference `spo_exact_results.json` (chaque ligne
    contient le regret de validation et le regret de test du meilleur etat pour
    ce lambda). Les balayages complementaires ne servent qu'a completer les
    points absents de la grille de reference.
    """
    rows: list[tuple[float, float, float | None]] = []
    primary = load("spo_exact_results.json")
    for r in primary.get("grid", []):
        rows.append((
            float(r["lambda"]),
            float(r["best_val_regret_pct"]),
            r.get("test_regret_pct"),
        ))

    for filename, target_lambda in _sweep_sources():
        if not (RESULTS / filename).exists():
            continue
        if any(abs(target_lambda - r[0]) < 1e-12 for r in rows):
            continue
        data = load(filename)
        row = next(
            (r for r in data.get("grid", []) if abs(r["lambda"] - target_lambda) < 1e-12),
            None,
        )
        if row is None:
            continue
        test_regret = None
        if abs(data.get("chosen_lambda", -1.0) - target_lambda) < 1e-12:
            test_regret = data["test"]["operational"]["regret_pct"]
        rows.append((target_lambda, float(row["best_val_regret_pct"]), test_regret))

    rows.sort(key=lambda r: r[0])
    return rows


def lambda_figure() -> None:
    """Regret de validation et de test en fonction de lambda (selection sur validation).

    Le regret de test n'est disponible que pour les lambdas dont le modele
    correspondant a ete evalue sur le jeu de test ; les points sans test
    correspondant sont simplement absents de la serie 'Regret de test'.
    """
    rows = _lambda_rows()
    x = np.array([r[0] for r in rows])
    val = np.array([r[1] for r in rows])
    test = np.array([np.nan if r[2] is None else r[2] for r in rows])
    selected_idx = int(np.argmin(val))

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    ax.plot(x, val, "o-", color=NAVY, lw=2, label="Régret de validation")
    ax.plot(x, test, "s--", color=GOLD, lw=2, label="Régret de test")
    ax.scatter(x[selected_idx], val[selected_idx], s=105, facecolor="white", edgecolor=GREEN, linewidth=2, zorder=4)
    ax.annotate(
        f"Sélection par validation\n$\\lambda={x[selected_idx]:g}$",
        xy=(x[selected_idx], val[selected_idx]),
        xytext=(0.85, 3.25),
        arrowprops={"arrowstyle": "->", "color": GREEN},
        color=GREEN,
        fontsize=8.5,
    )
    ax.set_xlabel("Poids décisionnel $\\lambda$ (SPO+ exact)")
    ax.set_ylabel("Regret (%)")
    ax.set_title("Exploration de $\\lambda$ : sélection sur validation, mesure sur test")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.minorticks_off()
    ax.legend(loc="best")
    out = FIGURES / "professional_lambda_sweep.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def validation_trajectory_figure() -> None:
    """Plot phase-3 validation regret by epoch from the saved training logs."""
    sources: list[tuple[float, str, float]] = []
    primary = load("spo_exact_results.json")
    for r in primary.get("grid", []):
        sources.append((float(r["lambda"]), "spo_exact_results.json", float(r["lambda"])))
    for filename, target_lambda in _sweep_sources():
        if not (RESULTS / filename).exists():
            continue
        if any(abs(target_lambda - s[0]) < 1e-12 for s in sources):
            continue
        sources.append((target_lambda, filename, target_lambda))
    sources.sort(key=lambda s: s[0])

    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, (lam, filename, target_lambda) in enumerate(sources):
        data = load(filename)
        row = next(
            r for r in data["grid"] if abs(r["lambda"] - target_lambda) < 1e-12
        )
        epochs = [entry["epoch"] for entry in row["log"]]
        regrets = [entry["val_regret_pct"] for entry in row["log"]]
        color = cmap(i / max(1, len(sources) - 1))
        ax.plot(
            epochs,
            regrets,
            "o-",
            color=color,
            lw=1.8,
            markersize=4,
            label=f"$\\lambda={lam:g}$",
        )
        best_epoch = epochs[int(np.argmin(regrets))]
        best_value = min(regrets)
        ax.scatter(
            [best_epoch],
            [best_value],
            s=70,
            facecolor="white",
            edgecolor=color,
            linewidth=1.7,
            zorder=4,
        )
    ax.set_xlabel("Époque de la phase 3 (−1 : modèle initial)")
    ax.set_ylabel("Regret de validation (%)")
    ax.set_title("Trajectoires de validation du regret selon $\\lambda$ (graine 42)")
    ax.set_xticks(range(-1, 8))
    ax.legend(ncol=3, loc="best", fontsize=8)
    ax.invert_yaxis()
    out = FIGURES / "professional_validation_trajectories.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    base = load("experiment_results.json")
    regret_figure(base)
    tradeoff_figure(base)
    lambda_figure()
    validation_trajectory_figure()
    for name in (
        "professional_regret_comparison.png",
        "professional_crps_regret_tradeoff.png",
        "professional_lambda_sweep.png",
        "professional_validation_trajectories.png",
    ):
        print(FIGURES / name)


if __name__ == "__main__":
    main()
