from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import CITYLEARN_DIR, IEEE33_DIR, IEEE69_DIR, PROCESSED_DIR, PROFILES_DIR, YEAR


def load_processed(year: int = YEAR) -> pd.DataFrame:
    path = PROCESSED_DIR / f"dataset_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset traite absent : {path}. Lancer scripts/build_dataset.py")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        df = df.ffill().bfill()
    return df


def load_processed_summary(year: int = YEAR) -> dict:
    path = PROCESSED_DIR / f"dataset_{year}_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_citylearn() -> dict:
    out = {}
    for name in ("load_profiles", "pv_profiles", "net_load_profiles", "storage_profiles", "prices"):
        p = CITYLEARN_DIR / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")
    return out


def load_synthetic_profiles() -> pd.DataFrame:
    p = PROFILES_DIR / "synthetic_profiles_2022.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")


def load_ieee(case: str = "33") -> dict:
    d = IEEE33_DIR if case == "33" else IEEE69_DIR
    out = {}
    for name in ("bus", "branch", "gen", "load", "line"):
        p = d / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def datasets_status() -> dict:
    status = {}
    try:
        df = load_processed()
        status["processed_2022"] = {"present": True, "shape": list(df.shape), "nan": int(df.isna().sum().sum())}
    except FileNotFoundError:
        status["processed_2022"] = {"present": False}
    status["ieee33"] = {"present": bool(load_ieee("33"))}
    status["ieee69"] = {"present": bool(load_ieee("69"))}
    cl = load_citylearn()
    status["citylearn"] = {"present": bool(cl), "files": list(cl.keys())}
    status["synthetic_profiles"] = {"present": (PROFILES_DIR / "synthetic_profiles_2022.csv").exists()}
    status["opsd_fr"] = {"present": (PROFILES_DIR / "opsd_fr_2020.csv").exists()}
    status["prices_dir"] = {"present": any((PROFILES_DIR.parent / "prices").glob("*.csv"))}
    status["meteorology_dir"] = {"present": any((PROFILES_DIR.parent / "meteorology").glob("*.csv"))}
    return status


if __name__ == "__main__":
    print(json.dumps(datasets_status(), indent=2, ensure_ascii=False))
