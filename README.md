# Article 1 — Decision-Focused Probabilistic Forecasting for Microgrid EMS

Projet de reproduction/extension de l'article *Decision-Focused Probabilistic Forecasting for Microgrid EMS*.
Guide de prérequis : `outils_mathematiques_article1_ems.md`.

## Structure

```
Article1/
├── config/
│   ├── paths.py              # Chemins du projet + chargement des clés API
│   └── keys.env.example      # Modèle de clés API (copier vers keys.env)
├── data/
│   ├── raw/
│   │   ├── ieee33/           # Réseau IEEE 33-bus (MATPOWER)
│   │   ├── ieee69/           # Réseau IEEE 69-bus (MATPOWER + pandapower)
│   │   ├── citylearn/        # Profils CityLearn v3 (load/PV/prices/KPIs)
│   │   ├── prices/           # Prix spot (ENTSO-E, OPSD)
│   │   ├── profiles/         # Profils PV/charge/prix (synthétiques + OPSD)
│   │   └── meteorology/      # Irradiance/météo (NREL NSRDB, ENTSO-E solaire)
│   └── processed/            # Jeux de données prêts pour l'entraînement
├── src/
│   ├── data/                 # Pipeline de téléchargement/génération
│   ├── forecasting/          # TFT / prévision probabiliste
│   ├── optimization/         # EMS, SPO+, cvxpylayers, TSRO/C&CG
│   └── utils/
├── scripts/
│   ├── download_all.py       # Orchestrateur : tous les datasets
│   ├── verify_install.py     # Vérification de l'environnement
│   └── check_data.py         # Contrôle d'intégrité des CSV
├── notebooks/
├── models/                   # Checkpoints entraînés
├── results/                  # Résultats d'expériences
├── figures/
├── tests/
└── requirements.txt
```

## Démarrage rapide

```powershell
# 1. Environnement (Python 3.12 recommandé)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt

# 2. Clés API (optionnel : NREL, ENTSO-E)
Copy-Item config\keys.env.example config\keys.env
# Éditer config\keys.env

# 3. Téléchargement des datasets
python scripts\download_all.py

# 4. Vérification environnement + données
python scripts\verify_install.py
python scripts\check_data.py
```

## Datasets

| Dataset | Source | Entrée API | Statut |
|---|---|---|---|
| IEEE 33-bus | MATPOWER + pandapower `case33bw` | non | ✅ importé (33 bus, 37 lignes, 3.715 MW) |
| IEEE 69-bus | MATPOWER `case69.m` | non | ✅ importé (69 bus, 68 lignes, 3802.1 **kW** — convention case69) |
| Profils synthétiques PV/charge/prix | généré localement | non | ✅ 8760 h × 11 colonnes |
| OPSD France horaire | open-power-system-data.org | non | ✅ FR 2020 (6575 h × 4) — couverture source 2015–2020, cache complet 89 Mo |
| NREL NSRDB (irradiance) | developer.nrel.gov | `NREL_API_KEY` | ⏸ en attente de clé |
| ENTSO-E (prix/charge/solaire) | transparency.entsoe.eu | `ENTSOE_API_KEY` | ⏸ en attente de clé |
| CityLearn v3 | package `citylearn` 3.0.2 | non | ✅ 17 bâtiments, 8759 h, export charge/PV/net/stockage/prix/KPIs |

### Conventions CityLearn v3 (important)

- `net_electricity_consumption` : échange réseau net (positif = import), inclut PV + stockage
- `solar_generation` : convention **négative** (génération PV = consumption négative) → exporté en positif dans `pv_profiles.csv`
- `demande brute` = `net − stockage_cons + pv_gen` → exporté dans `load_profiles.csv`
- Historiques indexés par le pas de temps `[t]` (pas `[-1]`)

## Prérequis connus / limitations

| Élément | Détail | Solution |
|---|---|---|
| `cvxpylayers` / `diffcp` | build C++ requis (CMake : `CMAKE_CXX_COMPILER not set`) | Installer *Visual Studio Build Tools* (workload C++), puis `pip install diffcp cvxpylayers` — **ou** utiliser Python 3.11 (wheels diffcp disponibles) |
| CityLearn pins | `citylearn 3.0.2` exige `numpy<2` et `scikit-learn<=1.2.2` (pas de wheels py3.12) | Installé en `--no-deps` avec numpy 2.x + sklearn 1.9 — fonctionnel sur les tests |
| OPSD 2022 | le jeu `time_series` public s'arrête en 09/2020 | Fallback automatique 2020 + cache local ; pour 2022 utiliser ENTSO-E (clé API) |
| Gurobi | licence requise | Fallback : HiGHS/ECOS/SCS/OSQP/CLARABEL déjà installés |

## Conventions

- Chemins centralisés dans `config/paths.py` (jamais de chemins en dur dans le code)
- Scripts exécutables depuis la racine : `python scripts\download_all.py`
- Données brutes dans `data/raw/`, données dérivées dans `data/processed/`
