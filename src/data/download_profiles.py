from __future__ import annotations

import io

import pandas as pd
import requests

from config.paths import (
    METEO_DIR,
    PRICES_DIR,
    PROFILES_DIR,
    TIMEZONE,
    YEAR,
    load_api_keys,
)


def download_nrel_pv(latitude=48.8566, longitude=2.3522, year=YEAR, panel_area=10, efficiency=0.20, pr=0.85):
    keys = load_api_keys()
    api_key = keys.get("NREL_API_KEY", "")
    if not api_key:
        return {"status": "skipped", "reason": "NREL_API_KEY absent (config/keys.env)"}
    url = (
        "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"
        f"?wkt=POINT({longitude}+{latitude})"
        f"&names={year}&leap_day=false&interval=60&utc=false"
        "&full_name=Researcher&email=researcher@university.edu"
        "&affiliation=University&mailing_list=false&reason=research"
        f"&api_key={api_key}"
        "&attributes=ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), skiprows=2)
    ghi_col = next(c for c in df.columns if c.strip().upper() == "GHI")
    df["pv_production_kWh"] = df[ghi_col] * panel_area * efficiency * pr / 1000
    out = METEO_DIR / f"nrel_solar_{year}.csv"
    df.to_csv(out, index=False)
    return {"status": "ok", "path": str(out), "shape": df.shape}


def download_entsoe(country_code="FR"):
    keys = load_api_keys()
    api_key = keys.get("ENTSOE_API_KEY", "")
    if not api_key:
        return {"status": "skipped", "reason": "ENTSOE_API_KEY absent (config/keys.env)"}
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key)
    start = pd.Timestamp(f"{YEAR}-01-01 00:00", tz=TIMEZONE)
    end = pd.Timestamp(f"{YEAR}-12-31 23:59", tz=TIMEZONE)

    results = {}
    prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    prices.name = "price_EUR_MWh"
    path = PRICES_DIR / f"entsoe_day_ahead_prices_{country_code}_{YEAR}.csv"
    prices.to_csv(path, header=True)
    results["prices"] = {"path": str(path), "rows": len(prices)}

    try:
        load = client.query_load(country_code, start=start, end=end)
        path = PRICES_DIR / f"entsoe_load_{country_code}_{YEAR}.csv"
        load.to_csv(path)
        results["load"] = {"path": str(path), "rows": len(load)}
    except Exception as e:
        results["load"] = {"error": str(e)}

    try:
        solar = client.query_generation(country_code, start=start, end=end, psr_type="B16")
        path = METEO_DIR / f"entsoe_solar_{country_code}_{YEAR}.csv"
        solar.to_csv(path)
        results["solar"] = {"path": str(path), "rows": len(solar)}
    except Exception as e:
        results["solar"] = {"error": str(e)}

    results["status"] = "ok"
    return results


def download_opsd(countries=("FR",), year=YEAR):
    url = (
        "https://data.open-power-system-data.org/time_series/latest/"
        "time_series_60min_singleindex.csv"
    )
    cache = PROFILES_DIR / "opsd_time_series_full.csv"
    if cache.exists():
        print(f"Cache OPSD trouve : {cache}")
        df = pd.read_csv(cache, index_col=0, parse_dates=True, low_memory=False)
    else:
        print("Telechargement OPSD time_series (peut prendre plusieurs minutes)...")
        df = pd.read_csv(url, index_col=0, parse_dates=True, low_memory=False)
        df.to_csv(cache)
    results = {"status": "ok", "coverage": f"{df.index.min()} -> {df.index.max()}", "sources": {}}
    for c in countries:
        cols = [col for col in df.columns if col.startswith(f"{c}_")]
        if not cols:
            results["sources"][c] = {"error": "aucune colonne"}
            continue
        sub = df[cols].dropna(how="all")
        available_years = sorted(sub.index.year.unique())
        target_year = year if year in available_years else available_years[-1]
        sub_year = sub[sub.index.year == target_year]
        out = PROFILES_DIR / f"opsd_{c.lower()}_{target_year}.csv"
        sub_year.to_csv(out)
        results["sources"][c] = {
            "path": str(out),
            "shape": sub_year.shape,
            "year": target_year,
            "available_years": f"{available_years[0]}-{available_years[-1]}",
            "note": None if target_year == year else f"annee {year} indisponible, fallback {target_year}",
        }
    return results
