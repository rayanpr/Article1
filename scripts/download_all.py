#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import ensure_dirs
from src.data import (
    download_citylearn,
    download_entsoe,
    download_ieee33,
    download_ieee69,
    download_nrel_pv,
    download_opsd,
    generate_price_profile,
    generate_load_profile,
    generate_pv_profile,
)
import pandas as pd

from config.paths import PROFILES_DIR, START


def main() -> None:
    ensure_dirs()
    report = {}

    steps = [
        ("ieee33", lambda: download_ieee33()),
        ("ieee69", lambda: download_ieee69()),
        ("synthetic", lambda: generate_all_synthetic()),
        ("opsd", lambda: download_opsd(countries=("FR",))),
        ("nrel", lambda: download_nrel_pv()),
        ("entsoe", lambda: download_entsoe(country_code="FR")),
        ("citylearn", lambda: download_citylearn()),
    ]

    for name, fn in steps:
        print(f"\n=== [{name}] ===")
        try:
            result = fn()
            report[name] = result
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        except Exception as e:
            report[name] = {"status": "error", "error": str(e)}
            print(f"ERREUR: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  RAPPORT FINAL")
    print("=" * 60)
    for name, result in report.items():
        status = result.get("status", "ok") if isinstance(result, dict) else "ok"
        print(f"  {name:<12} -> {status}")


def generate_all_synthetic(n_buildings=5, n_days=365):
    profiles = {}
    for i in range(n_buildings):
        profiles[f"pv_{i+1}"] = generate_pv_profile(n_days=n_days, seed=42 + i)
        profiles[f"load_{i+1}"] = generate_load_profile(n_days=n_days, seed=100 + i)
    profiles["price"] = generate_price_profile(n_days=n_days, seed=0)
    df = pd.DataFrame(profiles)
    df.index.name = "timestamp"
    out = PROFILES_DIR / f"synthetic_profiles_{START[:4]}.csv"
    df.to_csv(out)
    return {"status": "ok", "path": str(out), "shape": df.shape}


if __name__ == "__main__":
    main()
