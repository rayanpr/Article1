from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

IEEE33_DIR = RAW_DIR / "ieee33"
IEEE69_DIR = RAW_DIR / "ieee69"
CITYLEARN_DIR = RAW_DIR / "citylearn"
PRICES_DIR = RAW_DIR / "prices"
PROFILES_DIR = RAW_DIR / "profiles"
METEO_DIR = RAW_DIR / "meteorology"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CONFIG_DIR = ROOT / "config"

YEAR = 2022
START = f"{YEAR}-01-01"
END = f"{YEAR}-12-31 23:59:59"
FREQ = "h"
TIMEZONE = "Europe/Paris"

KEYS_FILE = CONFIG_DIR / "keys.env"


def ensure_dirs() -> None:
    for d in (
        IEEE33_DIR,
        IEEE69_DIR,
        CITYLEARN_DIR,
        PRICES_DIR,
        PROFILES_DIR,
        METEO_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def load_api_keys() -> dict:
    keys = {"NREL_API_KEY": "", "ENTSOE_API_KEY": ""}
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            keys[k.strip()] = v.strip()
    return keys
