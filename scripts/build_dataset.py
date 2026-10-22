#!/usr/bin/env python3
"""Assemble data/processed/dataset_2022.csv a partir des sources deja presentes dans data/raw.

Sources utilisees (aucun telechargement) :
  - data/raw/citylearn/{load,pv,prices}_profiles.csv : 17 batiments, horaire 2022
  - data/raw/profiles/synthetic_profiles_2022.csv    : PV/charge/prix synthetiques 8760 h
Sorties :
  - data/processed/dataset_2022.csv            : serie horaire agrgee (load, pv, price)
  - data/processed/dataset_2022_buildings.csv  : charge et pv par batiment
  - data/processed/dataset_2022_summary.json   : metadonnees
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import CITYLEARN_DIR, PROCESSED_DIR, PROFILES_DIR, START, YEAR

N_HOURS = 8760


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.apply(pd.to_numeric, errors="coerce")


def build() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    index = pd.date_range(START, periods=N_HOURS, freq="h")

    load_b = _load(CITYLEARN_DIR / "load_profiles.csv").reindex(index)
    pv_b = _load(CITYLEARN_DIR / "pv_profiles.csv").reindex(index)
    price = _load(CITYLEARN_DIR / "prices.csv").reindex(index)["electricity_pricing"]

    n_missing = int(load_b.isna().sum().sum() + pv_b.isna().sum().sum() + price.isna().sum())
    load_b = load_b.ffill().bfill()
    pv_b = pv_b.ffill().bfill()
    price = price.ffill().bfill()

    syn = _load(PROFILES_DIR / "synthetic_profiles_2022.csv").reindex(index)

    aggregate = pd.DataFrame(
        {
            "load_kwh": load_b.sum(axis=1),
            "pv_kwh": pv_b.sum(axis=1),
            "price_eur_kwh": price,
            "load_syn_kwh": syn.filter(like="load_").sum(axis=1),
            "pv_syn_kwh": syn.filter(like="pv_").sum(axis=1),
            "price_syn_eur_kwh": syn["price"],
        },
        index=index,
    )
    aggregate.index.name = "timestamp"
    aggregate["net_kwh"] = aggregate["load_kwh"] - aggregate["pv_kwh"]

    if int(aggregate.isna().sum().sum()) != 0:
        raise SystemExit(f"NaN residuels dans l'aggregate : {aggregate.isna().sum().to_dict()}")

    out = PROCESSED_DIR / f"dataset_{YEAR}.csv"
    aggregate.to_csv(out)

    buildings = pd.concat(
        {"load_kwh": load_b, "pv_kwh": pv_b}, names=["variable", "building"], axis=1
    )
    buildings.index.name = "timestamp"
    out_b = PROCESSED_DIR / f"dataset_{YEAR}_buildings.csv"
    buildings.to_csv(out_b)

    summary = {
        "year": YEAR,
        "n_hours": int(len(aggregate)),
        "n_buildings": int(load_b.shape[1]),
        "csv": str(out),
        "csv_buildings": str(out_b),
        "filled_missing_values": n_missing,
        "load_kwh_mean": round(float(aggregate["load_kwh"].mean()), 3),
        "pv_kwh_max": round(float(aggregate["pv_kwh"].max()), 3),
        "price_eur_kwh_mean": round(float(aggregate["price_eur_kwh"].mean()), 4),
        "price_syn_eur_kwh_mean": round(float(aggregate["price_syn_eur_kwh"].mean()), 4),
        "sources": ["citylearn", "synthetic_profiles"],
    }
    (PROCESSED_DIR / f"dataset_{YEAR}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
