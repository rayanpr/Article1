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
│   ├── data/                 # Pipeline de téléchargement/génération + loaders
│   │   └── loaders.py        # Chargement datasets (processed/CityLearn/IEEE/synthétiques)
│   ├── forecasting/          # TFT / prévision probabiliste
│   │   ├── preprocessing.py  # Features sin/cos, splits, standardisation, supervised
│   │   ├── tft.py            # TFT quantile (VSN GRN + LSTM + têtes de quantiles)
│   │   ├── copula.py         # Scénarios multivariés — copule gaussienne
│   │   └── metrics.py        # Pinball, CRPS, couverture, conformal prediction
│   └── optimization/         # EMS, SPO+, cvxpylayers, TSRO/C&CG
│       ├── ems.py            # LP EMS (dispatch batterie) — HiGHS
│       ├── spo.py            # SPO+ loss + gradient analytique (autograd)
│       ├── tsro.py           # TSRO C&CG (pire-cas boîte, règles de dispatch)
│       └── differentiable.py # Couche cvxpylayers + perte SPO+ exacte (voir .venv311)
├── docs/
│   └── formulations_mathematiques.md  # LP/QP/MILP, TSRO, CRPS, SPO+, cvxpylayers (LaTeX)
├── scripts/
│   ├── download_all.py       # Orchestrateur : tous les datasets
│   ├── build_dataset.py      # Assemblage de data/processed/ à partir des sources présentes
│   ├── verify_install.py     # Vérification de l'environnement
│   ├── check_data.py         # Contrôle d'intégrité des CSV
│   ├── step3_preprocess.py   # Étape 3 : nettoyage, sin/cos, splits, figures EDA
│   ├── train_evaluate.py     # Étapes 5–6 : entraînement TFT+SPO+, baselines, évaluation
│   ├── train_spo_exact.py    # Phase 3 : couplage SPO+ exact cvxpylayers (.venv311)
│   └── step7_report.py       # Étape 7 : figures comparatives + rapport structuré
├── models/                   # Checkpoints entraînés (.pt)
├── results/                  # Résultats d'expériences (JSON, .npy, rapport)
├── figures/                  # Figures EDA + comparatives (PNG)
├── tests/                    # Tests (pytest)
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

# 5. Jeu de données d'entraînement (data/processed/) — sans clé API
python scripts\build_dataset.py
```

## Pipeline ML (workflow complet, sans clé API)

```powershell
# Étape 3 : nettoyage, features sin/cos, splits, figures EDA
python scripts\step3_preprocess.py

# Étapes 5–6 : entraînement TFT (pinball → pinball+SPO+), baselines, évaluation
python scripts\train_evaluate.py

# Étape 6bis : couplage SPO+ exact sur le LP EMS (env .venv311, ~35 min CPU)
.venv311\Scripts\python scripts\train_spo_exact.py

# Étape 7 : figures comparatives + rapport structuré
python scripts\step7_report.py
```

- **Modèle** : TFT quantile (92,6 k paramètres) — phase 1 pinball (12 époques) puis phase 2 pinball + λ·SPO+ (6 époques, λ=0.3), seed=42, ~2 min CPU.
- **SPO+ boîte** : conforme à la Définition 3 d'Elmachtoub & Grigas (2022) — perte `(2ĉ−c)ᵀ(w*(c) − w*(2ĉ−c))`, gradient analytique `2(x*(c) − x*(2ĉ−c))` via `torch.autograd.Function` (LP SciPy/HiGHS).
- **SPO+ exact** (`train_spo_exact.py`, env `.venv311`) : LP EMS différenciable via cvxpylayers/diffcp (ECOS, repli SCS), loss = pinball + λ·(SPO+ à trois solves : décision prédite, oracle prix vrai `z_c = x*(n̂,p)`, problème modifié `z₂ = x*(n̂,2p̂−p)` + terme de recours sous vraies séries), grille λ ∈ {0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}, *early stopping* sur le regret val — checkpoint `models/tft_spo_exact.pt`.
- **Vérification de la convention SPO+** : `scripts/verify_spo_convention.py` (comparaison à la Définition 3, propriétés P1–P5, chemin `cvxpylayers`) — voir `docs/verification_spo_sign.md`.
- **Baselines** : persistence, LightGBM quantile, XGBoost quantile, TFT pinball-only, conformal, clairvoyant (oracle).
- **Sorties** : `models/tft_*.pt`, `results/experiment_results.json`, `results/spo_exact_results.json`, `results/rapport_workflow.md`, `figures/`.
- **Formulations mathématiques** (LP/QP/MILP, TSRO, CRPS, SPO+, cvxpylayers) : `docs/formulations_mathematiques.md`.

### Résultats de référence (test nov–déc 2022, 241 fenêtres de 24 h)

| Méthode | CRPS net (std) | Regret vs oracle |
|---|---|---|
| Persistence | 1.178 | 16.3 % |
| LightGBM | **0.167** | 3.1 % |
| XGBoost | 0.194 | 2.6 % |
| TFT pinball | 0.189 | 3.8 % |
| **TFT+SPO+ exact (cvxpylayers)** | 0.200 | **2.5 %** |
| TFT+SPO+ (boîte) | 0.188 | 3.3 % |

> Résultats recalculés après alignement de la perte SPO+ sur la Définition 3
> d'Elmachtoub & Grigas (2022) — voir `docs/verification_spo_sign.md`.
> λ de la phase 3 sélectionné par regret de validation parmi
> {0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0} → **λ = 3.0**
> (regret val 3.01 %, regret test 2.50 %).

Clairvoyant (perfect foresight) : 54.08 €/24 h. TSRO C&CG converge en 2 scénarios (nominal 49 € → pire-cas +63 %).
Détails et recommandations : `results/rapport_workflow.md`.

## Datasets

| Dataset | Source | Entrée API | Statut |
|---|---|---|---|
| IEEE 33-bus | MATPOWER + pandapower `case33bw` | non | ✅ importé (33 bus, 37 lignes, 3.715 MW) |
| IEEE 69-bus | MATPOWER `case69.m` | non | ✅ importé (69 bus, 68 lignes, 3802.1 **kW** — convention case69) |
| Profils synthétiques PV/charge/prix | généré localement | non | ✅ 8760 h × 11 colonnes |
| OPSD France horaire | open-power-system-data.org | non | ✅ FR 2020 (6575 h × 4) — couverture source 2015–2020, cache complet 89 Mo |
| NREL NSRDB (irradiance) | developer.nrel.gov | `NREL_API_KEY` | ⏸ non utilisé (pas de clé) — irradiance via profils PV synthétiques + CityLearn |
| ENTSO-E (prix/charge/solaire) | transparency.entsoe.eu | `ENTSOE_API_KEY` | ⏸ non utilisé (pas de clé) — prix via CityLearn + profil synthétique |
| CityLearn v3 | package `citylearn` 3.0.2 | non | ✅ 17 bâtiments, 8759 h, export charge/PV/net/stockage/prix/KPIs |
| **Jeu traité 2022** | `scripts/build_dataset.py` | non | ✅ `data/processed/dataset_2022.csv` : 8760 h × 7 colonnes (load/pv/price réels + synthétiques + net) |

### Conventions CityLearn v3 (important)

- `net_electricity_consumption` : échange réseau net (positif = import), inclut PV + stockage
- `solar_generation` : convention **négative** (génération PV = consumption négative) → exporté en positif dans `pv_profiles.csv`
- `demande brute` = `net − stockage_cons + pv_gen` → exporté dans `load_profiles.csv`
- Historiques indexés par le pas de temps `[t]` (pas `[-1]`)

## Prérequis connus / limitations

| Élément | Détail | Solution |
|---|---|---|
| `cvxpylayers` / `diffcp` | build C++ requis (CMake : `CMAKE_CXX_COMPILER not set`) | Installer *Visual Studio Build Tools* (workload C++), puis `pip install diffcp cvxpylayers` — **ou** utiliser Python 3.11 (wheels diffcp disponibles) |
| Environnement `.venv311` validé | `_diffcp` compilé numpy 1.x **plante en access violation sous numpy 2.x** | Python 3.11 + `numpy==1.26.4` (`<2`), `cvxpy==1.6.7`, `cvxpylayers==1.1.0` (`pip install --no-deps cvxpylayers`), `diffcp==1.0.22`, `torch` CPU ; solveur de la couche : **ECOS** (diffcp ne supporte qu'ECOS/SCS) |
| CityLearn pins | `citylearn 3.0.2` exige `numpy<2` et `scikit-learn<=1.2.2` (pas de wheels py3.12) | Installé en `--no-deps` avec numpy 2.x + sklearn 1.9 — fonctionnel sur les tests |
| OPSD 2022 | le jeu `time_series` public s'arrête en 09/2020 | Fallback automatique 2020 + cache local ; pour 2022 utiliser ENTSO-E (clé API) |
| Gurobi | licence requise | Fallback : HiGHS/ECOS/SCS/OSQP/CLARABEL déjà installés |

## Conventions

- Chemins centralisés dans `config/paths.py` (jamais de chemins en dur dans le code)
- Scripts exécutables depuis la racine : `python scripts\download_all.py`
- Données brutes dans `data/raw/`, données dérivées dans `data/processed/`
