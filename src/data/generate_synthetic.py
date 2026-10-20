from __future__ import annotations

import numpy as np
import pandas as pd

from config.paths import END, PROFILES_DIR, START


def _index(n_days: int) -> pd.DatetimeIndex:
    return pd.date_range(START, periods=n_days * 24, freq="h")


def generate_pv_profile(n_days=365, peak_kw=50.0, seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    hours = np.arange(n_days * 24)
    day_of_year = hours // 24
    hour_of_day = hours % 24
    solar_angle = np.pi * (hour_of_day - 6) / 12
    daylight = (solar_angle > 0) & (solar_angle < np.pi)
    seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    ghi = np.where(daylight, np.sin(solar_angle) * seasonal, 0)
    ghi = ghi + rng.normal(0, 0.05, len(hours))
    ghi = np.clip(ghi, 0, 1)
    return pd.Series(ghi * peak_kw, index=_index(n_days), name="pv_kW")


def generate_load_profile(n_days=365, peak_kw=30.0, seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    hours = np.arange(n_days * 24)
    hour_of_day = hours % 24
    day_of_year = hours // 24
    morning_peak = np.exp(-0.5 * ((hour_of_day - 8) / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hour_of_day - 19) / 2.0) ** 2)
    load = 0.3 + 0.4 * morning_peak + 0.6 * evening_peak
    load = load * (1.0 + 0.2 * np.cos(2 * np.pi * day_of_year / 365))
    load = load + rng.normal(0, 0.05, len(hours))
    load = np.clip(load, 0.1, 1.5)
    return pd.Series(load * peak_kw, index=_index(n_days), name="load_kW")


def generate_price_profile(n_days=365, seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    hours = np.arange(n_days * 24)
    hour_of_day = hours % 24
    peak = 30.0 * np.exp(-0.5 * ((hour_of_day - 19) / 2.0) ** 2)
    valley = -15.0 * np.exp(-0.5 * ((hour_of_day - 3) / 2.0) ** 2)
    price = 50.0 + peak + valley
    price = price + rng.normal(0, 5, len(hours))
    price = np.clip(price, 10, 200)
    return pd.Series(price, index=_index(n_days), name="price_EUR_MWh")


def generate_all(n_buildings=5, n_days=365):
    profiles = {}
    for i in range(n_buildings):
        profiles[f"pv_{i+1}"] = generate_pv_profile(n_days=n_days, seed=42 + i)
        profiles[f"load_{i+1}"] = generate_load_profile(n_days=n_days, seed=100 + i)
    profiles["price"] = generate_price_profile(n_days=n_days, seed=0)
    df = pd.DataFrame(profiles)
    df.index.name = "timestamp"
    out = PROFILES_DIR / f"synthetic_profiles_{START[:4]}.csv"
    df.to_csv(out)
    return {"path": str(out), "shape": df.shape, "period": f"{START} -> {END}"}
