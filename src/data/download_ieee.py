from __future__ import annotations

import re

import pandas as pd
import requests

from config.paths import IEEE33_DIR, IEEE69_DIR

MATPOWER_BASE = "https://raw.githubusercontent.com/MATPOWER/matpower/master/data"
BUS_COLUMNS = [
    "bus_i", "type", "Pd_MW", "Qd_MVAr", "Gs", "Bs",
    "area", "Vm", "Va", "baseKV", "zone", "Vmax", "Vmin",
]
BRANCH_COLUMNS = [
    "fbus", "tbus", "r_pu", "x_pu", "b_pu",
    "rateA_MVA", "rateB_MVA", "rateC_MVA",
    "ratio", "angle", "status", "angmin", "angmax",
]
GEN_COLUMNS = [
    "bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase",
    "status", "Pmax", "Pmin",
]


def parse_matpower_case(filepath):
    content = open(filepath, encoding="utf-8").read()

    def extract_matrix(name):
        match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", content, re.DOTALL)
        if not match:
            return None
        rows = []
        for line in match.group(1).strip().split("\n"):
            line = line.strip().rstrip(";").strip()
            if not line or line.startswith("%"):
                continue
            line = re.sub(r"%.*$", "", line).strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if vals:
                rows.append(vals)
        return pd.DataFrame(rows)

    def _set_columns(df, base_columns):
        if df is None or df.empty:
            return df
        n = df.shape[1]
        if n <= len(base_columns):
            df.columns = base_columns[:n]
        else:
            df.columns = base_columns + [f"extra_{i}" for i in range(n - len(base_columns))]
        return df

    bus_df = _set_columns(extract_matrix("bus"), BUS_COLUMNS)
    branch_df = _set_columns(extract_matrix("branch"), BRANCH_COLUMNS)
    gen_df = _set_columns(extract_matrix("gen"), GEN_COLUMNS)
    return bus_df, branch_df, gen_df


NOTE_CASE69_KW = "charges exprimees en kW (convention MATPOWER case69.m)"


def _fetch_case(case_name: str, dest_dir) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{case_name}.m"
    url = f"{MATPOWER_BASE}/{case_name}.m"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_text(response.text, encoding="utf-8")
    return str(dest)


def download_ieee33():
    mpath = _fetch_case("case33bw", IEEE33_DIR)
    bus_df, branch_df, gen_df = parse_matpower_case(mpath)
    bus_df.to_csv(IEEE33_DIR / "bus_matpower.csv", index=False)
    branch_df.to_csv(IEEE33_DIR / "branch_matpower.csv", index=False)
    if gen_df is not None:
        gen_df.to_csv(IEEE33_DIR / "gen_matpower.csv", index=False)
    try:
        import pandapower.networks as pn

        net = pn.case33bw()
        net.bus.to_csv(IEEE33_DIR / "bus.csv", index=False)
        net.line.to_csv(IEEE33_DIR / "line.csv", index=False)
        net.load.to_csv(IEEE33_DIR / "load.csv", index=False)
        source = "pandapower"
        n_buses, n_branches = len(net.bus), len(net.line)
        p_mw, q_mvar = float(net.load["p_mw"].sum()), float(net.load["q_mvar"].sum())
        vn_kv = float(net.bus["vn_kv"].iloc[0])
    except Exception:
        bus_df.to_csv(IEEE33_DIR / "bus.csv", index=False)
        branch_df.to_csv(IEEE33_DIR / "branch.csv", index=False)
        source = "matpower"
        n_buses, n_branches = len(bus_df), len(branch_df)
        p_mw, q_mvar = float(bus_df["Pd_MW"].sum()), float(bus_df["Qd_MVAr"].sum())
        vn_kv = float(bus_df["baseKV"].mode().iloc[0])
    return {
        "case": "IEEE 33-bus (case33bw)",
        "source": source,
        "n_buses": n_buses,
        "n_branches": n_branches,
        "Pd_total_MW": round(p_mw, 3),
        "Qd_total_MVAr": round(q_mvar, 3),
        "baseKV": vn_kv,
    }


def download_ieee69():
    mpath = _fetch_case("case69", IEEE69_DIR)
    bus_df, branch_df, gen_df = parse_matpower_case(mpath)
    bus_df.to_csv(IEEE69_DIR / "bus.csv", index=False)
    branch_df.to_csv(IEEE69_DIR / "branch.csv", index=False)
    if gen_df is not None and not gen_df.empty:
        gen_df.to_csv(IEEE69_DIR / "gen.csv", index=False)
    return {
        "case": "IEEE 69-bus",
        "source": "matpower",
        "n_buses": len(bus_df),
        "n_branches": len(branch_df),
        "Pd_total": round(float(bus_df["Pd_MW"].sum()), 3),
        "Qd_total": round(float(bus_df["Qd_MVAr"].sum()), 3),
        "unit": "kW/kVAr",
        "note": NOTE_CASE69_KW,
        "baseKV": float(bus_df["baseKV"].mode().iloc[0]),
    }
