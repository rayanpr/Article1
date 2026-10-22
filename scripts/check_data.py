import pandas as pd

checks = []

b33 = pd.read_csv("data/raw/ieee33/bus.csv")
checks.append(("IEEE33 bus", b33.shape, ""))
l33 = pd.read_csv("data/raw/ieee33/line.csv")
checks.append(("IEEE33 line", l33.shape, ""))
load33 = pd.read_csv("data/raw/ieee33/load.csv")
pd33 = load33[[c for c in load33.columns if "p_mw" in c.lower()]].sum().sum()
checks.append(("IEEE33 load", load33.shape, f"Pd={pd33:.3f} MW"))

b69 = pd.read_csv("data/raw/ieee69/bus.csv")
pd69 = b69[[c for c in b69.columns if c.startswith("Pd")]].sum().sum()
checks.append(("IEEE69 bus", b69.shape, f"Pd={pd69:.1f} kW"))

syn = pd.read_csv("data/raw/profiles/synthetic_profiles_2022.csv", index_col=0, parse_dates=True)
pv_cols = [c for c in syn.columns if c.startswith("pv")]
checks.append(("Synthetiques", syn.shape, f"NaN={int(syn.isna().sum().sum())} | pv={syn[pv_cols].min().min():.1f}-{syn[pv_cols].max().max():.1f} kW"))

opsd = pd.read_csv("data/raw/profiles/opsd_fr_2020.csv", index_col=0, parse_dates=True)
checks.append(("OPSD FR 2020", opsd.shape, f"cols={list(opsd.columns)}"))

ld = pd.read_csv("data/raw/citylearn/load_profiles.csv", index_col=0, parse_dates=True)
checks.append(("CityLearn load", ld.shape, f"NaN={int(ld.isna().sum().sum())} | mean={ld.mean().mean():.2f} kWh"))
pv = pd.read_csv("data/raw/citylearn/pv_profiles.csv", index_col=0, parse_dates=True)
checks.append(("CityLearn pv", pv.shape, f"max={pv.max().max():.2f} kWh"))
pr = pd.read_csv("data/raw/citylearn/prices.csv", index_col=0, parse_dates=True)
checks.append(("CityLearn prices", pr.shape, f"mean={pr['electricity_pricing'].mean():.3f}"))

from pathlib import Path

processed = Path("data/processed/dataset_2022.csv")
if processed.exists():
    ds = pd.read_csv(processed, index_col=0, parse_dates=True)
    checks.append(
        (
            "Processed 2022",
            ds.shape,
            f"NaN={int(ds.isna().sum().sum())} | net_mean={ds['net_kwh'].mean():.1f} kWh",
        )
    )
else:
    checks.append(("Processed 2022", "-", "ABSENT -> python scripts/build_dataset.py"))

for name, shape, info in checks:
    print(f"{name:<22} {str(shape):<15} {info}")
