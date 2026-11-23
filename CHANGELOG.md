# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scripts/verify_spo_convention.py` : vérification numérique de la perte SPO+ contre la
  Définition 3 d'Elmachtoub & Grigas (2022) — valeurs, non-négativité, domination de la
  perte SPO, sous-gradient, convexité, chemin `cvxpylayers`.
- `docs/verification_spo_sign.md` : analyse complète de la convention SPO+ (diagnostic
  initial « signe inversé » infirmé, écart réel avec la Définition 3, correctifs).
- Tests unitaires `TestSPOElmachtoubGrigas` verrouillant la conformité à la Définition 3.

### Changed
- Perte SPO+ (modèle boîte, `src/optimization/spo.py` / `spo_optimized.py`) :
  `(2ĉ−c)ᵀ(w*(c) − w*(2ĉ−c))` au lieu de l'expression substituant `w*(ĉ)` et `cᵀw*(ĉ)`
  à `w*(c)` et `z*(c)` — sous-gradient `2(w*(c) − w*(2ĉ−c))`.
- Perte SPO+ exacte (`src/optimization/differentiable.py`) : trois résolutions du LP EMS
  (décision prédite, oracle prix vrai `z_c = x*(n̂,p)`, problème modifié `z₂ = x*(n̂,2p̂−p)`),
  solutions détachées (constantes par morceaux en coefficients de coût).
- Grille λ de la phase 3 étendue de `{0.05, 0.1, 0.3}` à
  `{0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}` (8 époques max, patience 2) :
  la perte enfin conforme à la Définition 3 déplace l'optimum de λ ; la sélection
  du modèle final se fait par regret de **validation**, chaque λ étant évalué
  sur le test (`test_regret_pct` dans `results/spo_exact_results.json`).
- Documentation (`docs/formulations_mathematiques.md`, `README.md`, rapport PDF §V)
  alignée sur la Définition 3.
- Entraînement et résultats expérimentaux recalculés avec la perte corrigée.

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Compatibilité `diffcp 1.0.x` / `cvxpylayers ≥ 1.1` : le chemin « sans gradient » de
  `cvxpylayers` appelait `diffcp.solve_only_batch`, absent de diffcp 1.0.x ; l'échec
  silencieux rendait la perte SPO+ exacte nulle (entraînement réduit au pinball).
- Robustesse de la couche EMS : en cas d'échec transitoire du solveur ECOS sur un
  lot dégénéré, le triplet de résolutions est refait avec SCS au lieu d'abandonner
  le lot (gradient SPO+ nul pour ce lot).
- Précision numérique : chemins batch et unitaire de la perte SPO+ en `float64`
  (le chemin unitaire était en `float32`).
- Assertions de forme erronées dans `tests/test_spo_optimization.py` (`box_argmin` /
  `box_argmax` vérifiaient une longueur fixe de 24 au lieu de la longueur de l'entrée).

### Security
- Nothing yet

## [1.0.0] - 2026-11-20

### Added
- Complete implementation of SPO+ optimization with vectorization
- Robust models with Monte Carlo Dropout
- Comprehensive robustness analysis
- MLflow integration for experiment tracking
- Advanced visualization tools
- Complete benchmarking system
- Full test suite with unit tests
- Professional documentation and README
- Analysis of uncertainty budgeted robustness
- Decision latency benchmarking
- Comprehensive dashboard generation

### Changed
- Improved performance of SPO+ optimization (2-5x speedup)
- Enhanced model robustness against adversarial attacks
- Better memory management and GPU support
- Improved visualization capabilities
- Enhanced documentation and examples

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Fixed syntax errors in robust_models.py
- Corrected missing parentheses in code
- Fixed MLflow integration script

### Security
- Nothing yet
