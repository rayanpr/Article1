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

from config.paths import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR
from src.data.loaders import load_processed
from src.forecasting.preprocessing import (
    KNOWN_COLS,
    STANDARDIZE_COLS,
    TARGET_COLS,
    Standardizer,
    add_time_features,
    build_supervised,
    make_splits,
    split_arrays,
)


def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(
        axes,
        ["load_kwh", "pv_kwh", "price_eur_kwh"],
        ["Charge brute (kWh)", "Production PV (kWh)", "Prix spot (EUR/kWh)"],
    ):
        sns.histplot(df[col], bins=60, kde=True, ax=ax, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_distributions.png", dpi=150)
    plt.close(fig)


def plot_seasonal_profiles(df: pd.DataFrame) -> None:
    prof = df[TARGET_COLS].copy()
    prof["hour"] = prof.index.hour
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, TARGET_COLS):
        g = prof.groupby("hour")[col]
        mean = g.mean()
        ax.plot(mean.index, mean.values, label="moyenne", color="crimson")
        ax.fill_between(mean.index, g.quantile(0.1), g.quantile(0.9), alpha=0.25, color="crimson")
        ax.set_title(f"Profil horaire — {col}")
        ax.set_xlabel("Heure")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_seasonal_profiles.png", dpi=150)
    plt.close(fig)


def plot_week_heatmap(df: pd.DataFrame) -> None:
    prof = pd.DataFrame(
        {"net": df["net_kwh"], "dow": df.index.dayofweek, "hour": df.index.hour}
    )
    piv = prof.groupby(["dow", "hour"])["net"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(piv, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Charge nette moyenne (kWh) — jour × heure")
    ax.set_xlabel("Heure")
    ax.set_ylabel("Jour semaine (0=lundi)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_heatmap_net.png", dpi=150)
    plt.close(fig)


def plot_split_timeline(df: pd.DataFrame, masks: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df.index, df["net_kwh"], lw=0.4, color="gray")
    for name, color in [("train", "#4c72b0"), ("val", "#dd8452"), ("test", "#55a868")]:
        m = masks[name]
        ax.axvspan(df.index[m][0], df.index[m][-1], alpha=0.25, color=color, label=name)
    ax.legend()
    ax.set_title("Splits temporels train / validation / test")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_splits.png", dpi=150)
    plt.close(fig)


def main() -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed()
    n0 = len(df)
    df = df.dropna()
    df = add_time_features(df)
    masks = make_splits(df.index)

    sc = Standardizer(STANDARDIZE_COLS).fit(df.loc[masks["train"], TARGET_COLS])
    df_std = df.copy()
    df_std[TARGET_COLS] = sc.transform(df[TARGET_COLS])

    xs, ys, idx = build_supervised(df_std, lookback=24, horizon=24)
    arr = split_arrays(xs, ys, idx, masks)
    np.savez_compressed(
        PROCESSED_DIR / "arrays_2022.npz",
        **{k: v for k, v in arr.items()},
        scaler_mean=sc.mean_,
        scaler_std=sc.std_,
        timestamps=df.index.astype(str).to_numpy(),
    )

    plot_distributions(df)
    plot_seasonal_profiles(df)
    plot_week_heatmap(df)
    plot_split_timeline(df, masks)

    summary = {
        "n_rows_raw": n0,
        "n_rows_after_dropna": int(len(df)),
        "n_features_time": KNOWN_COLS,
        "targets": TARGET_COLS,
        "scaler_mean": sc.mean_.round(5).tolist(),
        "scaler_std": sc.std_.round(5).tolist(),
        "splits": {k: int(v.sum()) for k, v in masks.items()},
        "arrays": {k: list(v.shape) for k, v in arr.items() if hasattr(v, "shape")},
        "figures": [
            "eda_distributions.png",
            "eda_seasonal_profiles.png",
            "eda_heatmap_net.png",
            "eda_splits.png",
        ],
    }
    (RESULTS_DIR / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
