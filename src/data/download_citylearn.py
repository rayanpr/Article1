from __future__ import annotations

import numpy as np
import pandas as pd

from config.paths import CITYLEARN_DIR, START

DATASET = "citylearn_challenge_2022_phase_all_plus_evs"


def download_citylearn():
    try:
        from citylearn.citylearn import CityLearnEnv
        from citylearn.agents.baseline import BusinessAsUsualAgent
    except ImportError:
        return {"status": "skipped", "reason": "citylearn non installe (pip install citylearn)"}

    env = CityLearnEnv(DATASET, episode_time_steps=8760)
    observations, _ = env.reset(seed=42)
    agent = BusinessAsUsualAgent(env)

    n_buildings = len(env.buildings)
    demand = np.zeros((8760, n_buildings))
    pv_gen = np.zeros((8760, n_buildings))
    net_exchange = np.zeros((8760, n_buildings))
    storage = np.zeros((8760, n_buildings))

    terminated = truncated = False
    step = 0
    while not (terminated or truncated) and step < 8760:
        actions = agent.predict(observations)
        observations, _, terminated, truncated, _ = env.step(actions)
        for i, b in enumerate(env.buildings):
            nec = float(np.asarray(b.net_electricity_consumption).ravel()[step])
            pv = -float(np.asarray(b.solar_generation).ravel()[step])
            st = float(np.asarray(b.electrical_storage.electricity_consumption).ravel()[step])
            net_exchange[step, i] = nec
            pv_gen[step, i] = pv
            storage[step, i] = st
            demand[step, i] = nec - st + pv
        step += 1

    time_index = pd.date_range(START, periods=step, freq="h")
    cols = [f"building_{i+1}" for i in range(n_buildings)]

    demand_df = pd.DataFrame(demand[:step], index=time_index, columns=cols)
    pv_df = pd.DataFrame(pv_gen[:step], index=time_index, columns=cols)
    net_df = pd.DataFrame(net_exchange[:step], index=time_index, columns=cols)
    storage_df = pd.DataFrame(storage[:step], index=time_index, columns=cols)

    demand_df.to_csv(CITYLEARN_DIR / "load_profiles.csv")
    pv_df.to_csv(CITYLEARN_DIR / "pv_profiles.csv")
    net_df.to_csv(CITYLEARN_DIR / "net_load_profiles.csv")
    storage_df.to_csv(CITYLEARN_DIR / "storage_profiles.csv")

    price_df = None
    pricing = getattr(env.buildings[0], "pricing", None)
    if pricing is not None:
        raw = getattr(pricing, "electricity_pricing", None)
        if raw is None:
            raw = getattr(pricing, "_electricity_pricing", None)
        if raw is not None:
            raw = np.asarray(raw, dtype=float).ravel()

            def _maybe(attr):
                v = getattr(pricing, attr, None)
                if v is None:
                    v = getattr(pricing, f"_{attr}", None)
                return np.asarray(v, dtype=float).ravel() if v is not None else None

            price_df = pd.DataFrame(
                {
                    "electricity_pricing": raw,
                    "predicted_1": _maybe("electricity_pricing_predicted_1"),
                    "predicted_2": _maybe("electricity_pricing_predicted_2"),
                    "predicted_3": _maybe("electricity_pricing_predicted_3"),
                },
                index=pd.date_range(START, periods=len(raw), freq="h"),
            )
            price_df.to_csv(CITYLEARN_DIR / "prices.csv")

    try:
        kpis = env.evaluate_v2()
        kpis.to_csv(CITYLEARN_DIR / "kpis_bau.csv", index=False)
    except Exception:
        try:
            kpis = pd.DataFrame(env.evaluate())
            kpis.to_csv(CITYLEARN_DIR / "kpis_bau.csv", index=False)
        except Exception as e:
            kpis = {"error": str(e)}

    env.close()
    return {
        "status": "ok",
        "n_buildings": n_buildings,
        "n_steps": step,
        "demand_mean_kwh": round(float(demand[:step].mean()), 3),
        "pv_max_kwh": round(float(pv_gen[:step].max()), 3),
        "load": str(CITYLEARN_DIR / "load_profiles.csv"),
        "pv": str(CITYLEARN_DIR / "pv_profiles.csv"),
        "net": str(CITYLEARN_DIR / "net_load_profiles.csv"),
        "storage": str(CITYLEARN_DIR / "storage_profiles.csv"),
        "prices": str(CITYLEARN_DIR / "prices.csv") if price_df is not None else None,
        "kpis": str(CITYLEARN_DIR / "kpis_bau.csv"),
    }
