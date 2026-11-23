# Vérification de la perte SPO+ — Analyse définitive (2026-11-23)

> **Mise à jour importante.** Le premier diagnostic (« la perte a le signe inversé,
> elle est négative, il faut ré-entraîner ») était **faux** : il reposait sur des
> scripts de vérification buggués. L'analyse ci-dessous est la version vérifiée
> contre la source primaire (arXiv:1710.08005v5, *Smart "Predict, then Optimize"*,
> Elmachtoub & Grigas, 2022).

---

## 1. Référence : la Définition 3 d'Elmachtoub & Grigas

Pour un problème de **minimisation** `P(c) : z*(c) = min_{w∈S} cᵀw`, avec
`w*(c) = argmin_{w∈S} cᵀw` :

**Définition 3 (SPO+ loss)**

```
ℓ_SPO+(ĉ, c) := max_{w∈S} { cᵀw − 2ĉᵀw } + 2ĉᵀ w*(c) − z*(c)
```

où `z*(c) = cᵀ w*(c)` est la valeur optimale sous le coût vrai. Forme équivalente
utilisée dans le code (puisque `max_w (c−2ĉ)ᵀw = −min_w (2ĉ−c)ᵀw`) :

```
ℓ_SPO+(ĉ, c) = (2ĉ − c)ᵀ ( w*(c) − w*(2ĉ − c) )
```

**Proposition 3** : `ℓ_SPO ≤ ℓ_SPO+` (borne supérieure de la perte SPO),
`ℓ_SPO+` est convexe en `ĉ`, et `ℓ_SPO+(c,c) = 0`.

**Proposition 3.3 (sous-gradient)** :

```
∇_ĉ ℓ_SPO+ = 2 ( w*(c) − w*(2ĉ − c) )
```

> Point clé : la définition fait intervenir **l'oracle sur le coût vrai** `w*(c)`
> et `z*(c)`, et **non** la décision induite par la prédiction `w*(ĉ)`.

---

## 2. Pourquoi le premier diagnostic était faux

Les scripts `verify_spo_sign.py` / `verify_spo_sign_simple.py` concluaient à un
« signe inversé » à cause de **deux bugs** :

| Bug | Effet |
|-----|-------|
| Dans `ems_test()` / `sensitivity_test()` : après avoir résolu le problème à prix modifié, le script relit `g.value` et `x.value` **pour les deux solutions** (les mêmes variables CVXPY sont écrasées par le 2ᵉ solve) | `grad1 == grad2` ⇒ perte **identiquement nulle** ⇒ « perte non positive » ⇒ faux diagnostic |
| Dans `simple_mathematical_test()` : `x1 = c_true` est présenté comme « solution de `min c·x s.t. 0≤x≤1` » | solution **hors du domaine** (2 > 1) au lieu de `x = 0` ⇒ test non valide |

En réalité, la perte implémentée **était déjà non négative** : l'ancienne formule
`(c−2ĉ)ᵀ(x*_mod − x̂)` avec `x*_mod = argmax (c−2ĉ)ᵀx` l'est par construction
(le maximum est atteint en `x*_mod`), et le code appliquait déjà
`np.maximum(losses, 0.0)`.

---

## 3. Le vrai écart (et sa nature)

L'implémentation d'origine calculait :

```
ℓ_impl = (c−2ĉ)ᵀ(x*_mod − x̂) + 2ĉᵀ x̂ − cᵀ x̂
```

soit, en comparant à la Définition 3 :

| Terme | E&G (correct) | Ancienne implémentation |
|-------|---------------|-------------------------|
| 1er terme | `max_w (c−2ĉ)ᵀw` | `max_w (c−2ĉ)ᵀw` ✔ |
| 2ᵉ terme | `2ĉᵀ w*(c)` (oracle coût vrai) | `2ĉᵀ x̂ = 2ĉᵀ w*(ĉ)` ✘ |
| 3ᵉ terme | `− z*(c) = −cᵀ w*(c)` | `− cᵀ x̂ = −cᵀ w*(ĉ)` ✘ |
| Gradient | `2(w*(c) − w*(2ĉ−c))` | `2(x̂ − x*(2ĉ−c))` ✘ |

Ce n'était donc **pas un problème de signe**, mais un **écart avec la définition
de la source** : la perte n'était ni la borne supérieure convexe de la perte SPO,
ni convexe, et son sous-gradient n'était pas celui de la littérature.

---

## 4. Correction appliquée

* `src/optimization/spo.py`, `src/optimization/spo_optimized.py` (modèle boîte) :
  deux solves par instance — `w*(c) = argmin cᵀx` et `w*(2ĉ−c) = argmin (2ĉ−c)ᵀx` —
  perte `(2ĉ−c)ᵀ(w*(c) − w*(2ĉ−c))`, sous-gradient `2(w*(c) − w*(2ĉ−c))`.
* `src/optimization/differentiable.py` (LP EMS via `cvxpylayers`) : **trois** solves
  (décision prédite `z₁` pour le recours, oracle `z_c = x*(n̂, p)`, problème modifié
  `z₂ = x*(n̂, 2p̂−p)`), perte `(2p̂−p)ᵀ(ψ(z_c) − ψ(z₂))`, solutions détachées
  (constantes par morceaux en coefficients de coût).
* Correctif de compatibilité `diffcp 1.0.x` / `cvxpylayers ≥ 1.1` : le chemin « sans
  gradient » de `cvxpylayers` appelle `diffcp.solve_only_batch`, absent de diffcp 1.0.x
  (le solve de l'oracle échouait silencieusement, transformant la perte en 0).

---

## 5. Vérification numérique (`scripts/verify_spo_convention.py`)

| Test | Propriété | Résultat |
|------|-----------|----------|
| P0 | valeurs = Définition 3 | écart max **0.000e+00** ✔ |
| P1 | `ℓ_SPO+(c,c) = 0` | **0.000e+00** ✔ |
| P2 | `ℓ_SPO+ ≥ ℓ_SPO ≥ 0` | min `ℓ_SPO` = 2.81e+01, min `ℓ_SPO+` = 1.82e+02 ✔ |
| P3 | gradient = sous-gradient E&G | écart max **4.8e-08** ✔ |
| P4 | convexité en `ĉ` | violation max **−3.0e+00** (≤ 0) ✔ |
| P5 | chemin `cvxpylayers` (LP EMS) | valeur : err. rel. **2.9e-09**, gradient : err. max **9.6e-07** ✔ |

Commande :

```powershell
.venv\Scripts\python.exe scripts\verify_spo_convention.py      # modèle boîte
.venv311\Scripts\python.exe scripts\verify_spo_convention.py   # + chemin EMS/cvxpylayers
```

Les mêmes propriétés sont verrouillées par des tests unitaires dans
`tests/test_spo_optimization.py` (classe `TestSPOElmachtoubGrigas`).

---

## 6. Conséquences

1. Les modèles `tft_spo.pt` (phase 2, SPO+ boîte) et `tft_spo_exact*.pt`
   (phase 3, SPO+ exact) **ont été ré-entraînés** avec la perte conforme à la
   Définition 3 ; les résultats expérimentaux ont été recalculés.
2. La formulation du rapport (section V, équation 7) et
   `docs/formulations_mathematiques.md` (§5, §6.1) ont été mises à jour.
3. Le diagnostic « signe inversé » ne doit pas être repris dans le mémoire :
   la perte n'a jamais été négative, c'est la **définition** qui était inexacte.

### 6.1 Résultats après correction (test nov–déc 2022, 241 fenêtres)

| Méthode | CRPS net | Regret vs oracle |
|---------|----------|------------------|
| Persistence | 1,178 | 16,27 % |
| LightGBM quantile | **0,167** | 3,10 % |
| XGBoost quantile | 0,194 | 2,56 % |
| TFT pinball (phase 1) | 0,189 | 3,82 % |
| TFT + SPO+ boîte (phase 2) | 0,188 | 3,30 % |
| **TFT + SPO+ exact (phase 3)** | 0,200 | **2,50 %** |

La phase 2 (relaxation boîte) **améliore** désormais le regret (3,30 % contre
3,82 % pour le pinball seul) alors qu'elle le dégradait (4,21 %) avec
l'implémentation non conforme.

### 6.2 Effet de bord : la perte SPO+ exacte était en réalité nulle avant correction

Le correctif de compatibilité `diffcp.solve_only_batch` a révélé que le chemin
« sans gradient » de `cvxpylayers ≥ 1.1` (utilisé pour l'oracle `z_c`) échouait
avec diffcp 1.0.x. L'exception était avalée par le `try/except` de
`spo_exact_loss`, qui renvoyait alors une perte nulle : **la phase 3
n'entraînait donc que la perte pinball** (et non `pinball + λ·SPO+`).

C'est ce qui explique que la correction déplace l'optimum de λ : l'ancienne
grille `{0,05 ; 0,1 ; 0,3}` avait été calibrée sur un entraînement réduit au
pur pinball. Avec la vraie perte SPO+, le balayage complet donne (sélection
sur **validation**, test reporté a posteriori) :

| λ | 0,05 | 0,1 | 0,3 | 0,5 | 1,0 | 1,5 | 2,0 | 3,0 | 5,0 |
|---|------|-----|-----|-----|-----|-----|-----|-----|-----|
| regret validation (%) | 3,876 | 3,733 | 3,514 | 3,384 | 3,239 | 3,096 | 3,091 | **3,015** | 3,020 |
| regret test (%) | 3,129 | 2,945 | 2,836 | 2,717 | 2,685 | 2,660 | 2,674 | **2,503** | 2,591 |

La sélection par validation (λ = 3,0) est ici confirmée par le test : c'est
aussi la valeur qui minimise le regret de test. La grille de référence de la
phase 3 a donc été étendue à `{0,05 ; 0,1 ; 0,3 ; 0,5 ; 1 ; 1,5 ; 2 ; 3 ; 5}`
(8 époques max, patience 2).

### 6.3 Robustesse numérique

En cas d'échec transitoire du solveur ECOS sur un lot dégénéré, le triplet de
résolutions (décision prédite, oracle, problème modifié) est refait avec **SCS**
plutôt que d'abandonner le lot (le gradient SPO+ serait alors nul pour ce lot).

## 7. Référence

[1] A. N. Elmachtoub and P. Grigas, "Smart 'Predict, then Optimize'", *Management
Science*, vol. 68, no. 5, pp. 2875–2902, 2022, doi: `10.1287/mnsc.2020.3922`
(préprint : arXiv:1710.08005v5, §3.2, Définition 3 et Propositions 3 et 3.3).

---

*Document généré le 2026-11-23. Source de vérité du dépôt :
`docs/verification_spo_sign.md`.*
