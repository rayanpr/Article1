# Formulations mathématiques — EMS microgrid & prévision decision-focused

> Correspondances code : `src/optimization/ems.py` (LP/QP/MILP), `src/optimization/tsro.py` (TSRO/C&CG),
> `src/optimization/spo.py` (SPO+), `src/optimization/differentiable.py` (cvxpylayers),
> `src/forecasting/metrics.py` (pinball/CRPS/conformal).

## 1. Modèle physique du microgrid

Horizon $H$ (24 h, pas horaire). Grandeurs nettes : charge brute $L_t \ge 0$, production PV $P_t^{pv} \ge 0$,
charge nette $y_t = L_t - P_t^{pv}$ (positif = import). Batterie : état de charge $e_t$, puissance charge $c_t \ge 0$,
décharge $d_t \ge 0$, rendements $\eta^{ch}, \eta^{dis} \in (0,1]$, capacité $E^{\max}$, socle $E^{\min} = \alpha E^{\max}$.

$$
e_{t+1} = e_t + \eta^{ch} c_t - \frac{d_t}{\eta^{dis}}, \qquad
E^{\min} \le e_t \le E^{\max}, \qquad 0 \le c_t \le P^{ch},\ 0 \le d_t \le P^{dis}
$$

Échange réseau : import $g_t \ge 0$, export $x_t \ge 0$, tarif d'achat $p_t$, tarif de revente $r_t = \beta p_t$ ($\beta = 0.5$) :

$$
g_t - x_t = y_t + c_t - d_t
$$

## 2. Problème LP — dispatch économique (perfect foresight)

$$
\min_{g,x,c,d,e} \ \sum_{t=1}^{H} p_t\, g_t - \beta\, p_t\, x_t
\quad \text{s.c. phys (1)–(3)}
$$

LP linéaire : solution exacte via HiGHS (`solve_ems`, `cp.HIGHS`). C'est le **clairvoyant** (oracle) utilisé pour le regret.

### 2.1 Version QP — dégradation batterie

Régularisation quadratique du cycle de vie (souvent utilisée pour décourager les micro-cycles) :

$$
\min \ \sum_t p_t g_t - \beta p_t x_t + \frac{\gamma}{2}\sum_t (c_t^2 + d_t^2)
$$

QP convexe ($\gamma > 0$), solveur OSQP/CLARABEL/ECOS.

### 2.2 Version MILP — unit commitment batterie

Binaire $z_t \in \{0,1\}$ (batterie active) + big-$M$ :

$$
c_t \le P^{ch} z_t, \quad d_t \le P^{dis} z_t, \quad \min \ \text{coût LP} + \kappa \sum_t z_t
$$

MILP (HiGHS-branch-and-bound, ou Gurobi si licence). $\kappa$ modélise le coût de démarrage/Usure.

## 3. Two-Stage Robust Optimization (TSRO) — C&CG

Incertitude structurée sur la charge nette, boîte budgétée type Bertsimas–Sim :

$$
\mathcal{U}(\hat y, \Delta, \Gamma) = \left\{ y : y_t = \hat y_t + \delta_t z_t,\ 0 \le z_t \le 1,\ \sum_t z_t \le \Gamma \right\}
$$

- **Stage 1** (here-and-now) : décision $u = (c,d)$ (dispatch batterie) choisie avant $y$.
- **Stage 2** (recourse) : import/export ajustés après réalisation.

Forme min-max :

$$
\min_{u \in \mathcal{X}} \ \max_{y \in \mathcal{U}} \ Q(u,y), \qquad
Q(u,y) = \sum_t p_t \big(y_t - u_t\big)^{+} - \beta p_t \big(y_t - u_t\big)^{-}
$$

Algorithme C&CG (column-and-constraint generation) :

1. Master : $\min \tau$ s.c. $\tau \ge Q(u, y^{s})\ \forall s \in \mathcal{S}$ (scénarios accumulés).
2. Sous-problème : $y^{s+1} = \arg\max_{y\in\mathcal{U}} Q(u^\star, y)$ (séparable par heure → sommet de boîte).
3. Ajout de la contrainte, itération jusqu'à stabilité de $\tau$ (gap $< \epsilon$).

Implémentation : `tsro_ccg` (sous-problème explicite, sommets optimaux par comparaison $\hat y_t \pm \Delta_t$).

## 4. Prévision probabiliste — quantiles, CRPS, calibration

### 4.1 Quantiles et pinball loss

Pour un niveau $\tau \in (0,1)$, le $\tau$-quantile conditionnel minimise :

$$
\mathcal{L}_\tau(y, \hat q_\tau) = \rho_\tau(y - \hat q_\tau), \qquad
\rho_\tau(u) = u\,(\tau - \mathbf{1}\{u < 0\})
$$

Multi-quantiles ($\tau \in \{0.1, 0.25, 0.5, 0.75, 0.9\}$) : somme moyenne des pinballs.

### 4.2 CRPS

Continuous Ranked Probability Score :

$$
\mathrm{CRPS}(F, y) = \int_{\mathbb{R}} \big(F(z) - \mathbf{1}\{z \ge y\}\big)^2 dz
= \mathbb{E}|X - y| - \tfrac{1}{2}\mathbb{E}|X - X'|
$$

- Estimateur par scénarios (copule) : `crps_from_scenarios` (deux termes empiriques).
- Approximation par quantiles intégrée : `crps_from_quantiles` (somme des pinballs × 2).

### 4.3 Calibration conformiste (split conformal)

Sur le jeu de calibration $\mathcal{C}$ (validation) :

$$
\hat q = \mathrm{Quantile}_{1-\alpha}\Big(\big|y_i - \hat q^{med}_i\big|\Big)_{i \in \mathcal{C}},
\qquad I^{conf}_t = \big[\hat q^{med}_t - \hat q,\ \hat q^{med}_t + \hat q\big]
$$

Garantie marginale de couverture $1-\alpha$ sous échantillonnage exchangeable. Code : `conformal_interval`.

## 5. SPO+ — Smart Predict-then-Optimize (Elmachtoub & Grigas, 2022)

Problème d'énergie paramétré par un vecteur de coût $c$ (ici prix) :

$$
x^\star(c) = \arg\min_{x \in \mathcal{X}} c^\top x, \qquad
\mathcal{X} = \{ u \in [-P^{ch}, P^{dis}]^H : E^{\min}-e_0 \le \textstyle\sum_{s \le t} u_s \le E^{\max}-e_0 \}
$$

($u_t = c_t - d_t$ ; polytope couplé, contraintes cumulées.)

**SPOT loss** (non différenciable, Définition 2 d'Elmachtoub & Grigas) :

$$
\mathcal{L}_{SPOT}(\hat c, c) = c^\top x^\star(\hat c) - c^\top x^\star(c) \;\ge\; 0
$$

(régret de la décision induite par la prédiction : $x^\star(\hat c)$ est la décision
réellement implémentée, $x^\star(c)$ l'oracle sous le coût vrai.)

**SPO+** (Définition 3, arXiv:1710.08005 §3.2 — convexe, dérivable presque partout) :

$$
\mathcal{L}_{SPO+}(\hat c, c) = \max_{x \in \mathcal{X}} \big(c - 2\hat c\big)^\top x
+ 2\hat c^\top x^\star(c) - z^\star(c),
\qquad z^\star(c) = \min_{x\in\mathcal{X}} c^\top x = c^\top x^\star(c)
$$

soit, avec $x^\star(2\hat c - c) = \arg\min_x (2\hat c - c)^\top x$ :

$$
\boxed{\;\mathcal{L}_{SPO+}(\hat c, c) = \big(2\hat c - c\big)^\top \big(x^\star(c) - x^\star(2\hat c - c)\big)\;}
$$

> **Attention — erreur corrigée (2026-11-23).** Une version antérieure du code et de
> ce document utilisait $x^\star(\hat c)$ et $c^\top x^\star(\hat c)$ à la place de
> $x^\star(c)$ et $z^\star(c)$ dans la définition de SPO+. Ce n'était pas un
> simple problème de signe (la perte restait non négative), mais un écart avec la
> Définition 3 : la convexité de $\mathcal{L}_{SPO+}$ et le sous-gradient
> d'Elmachtoub & Grigas ne s'appliquaient pas. La formule ci-dessus est
> conforme à la source et est vérifiée numériquement par
> `scripts/verify_spo_convention.py` (voir §5.3).

Propriétés (Elmachtoub & Grigas, Proposition 3) : $\mathcal{L}_{SPOT} \le \mathcal{L}_{SPO+}$,
$\mathcal{L}_{SPO+}$ convexe en $\hat c$, $\mathcal{L}_{SPO+}(c,c) = 0$.

### 5.1 Gradient (sous-gradient)

Le max est atteint en $x^\star(2\hat c - c)$ (car $\min_{x}(2\hat c - c)^\top x$), et la
solution d'un LP est constante par morceaux en coefficients de coût ; d'où
(Proposition 3.3) :

$$
\nabla_{\hat c}\,\mathcal{L}_{SPO+} = 2\big(x^\star(c) - x^\star(2\hat c - c)\big)
$$

Chaînage avec le forecaster $\hat c_\theta$ : $\nabla_\theta = \nabla_{\hat c} \cdot \partial \hat c_\theta / \partial \theta$.
Implémentation : `spo.spo_plus_loss` (LP SciPy/HiGHS) + `spo_plus_loss_torch` (deux solves
par instance : $x^\star(c)$ et $x^\star(2\hat c - c)$ ; gradient analytique détaché).

### 5.2 Coût de calcul

Le SPO+ d'Elmachtoub & Grigas ne nécessite que **deux resolutions de LP par instance**
(oracle sur le coût vrai + problème modifié) : le même nombre que la version
précédente, mais les deux solutions intervenant dans la perte sont celles de la
théorie.

### 5.3 Vérification numérique

`scripts/verify_spo_convention.py` (exécutable avec `.venv` et `.venv311`) contrôle :

| Test | Propriété |
|------|-----------|
| P0 | valeurs implémentées = Définition 3 (écart < 1e-6) |
| P1 | $\mathcal{L}_{SPO+}(c,c) = 0$ |
| P2 | $\mathcal{L}_{SPO+} \ge \mathcal{L}_{SPOT} \ge 0$ |
| P3 | gradient implémenté = sous-gradient $2(x^\star(c) - x^\star(2\hat c - c))$ |
| P4 | convexité de $\mathcal{L}_{SPO+}$ en $\hat c$ |
| P5 | chemin `cvxpylayers` (LP EMS complet) : même valeur et même gradient |

## 6. Couche d'optimisation différentielle (cvxpylayers)

Le LP EMS est traduit en couche $\mathbb{R}^{2H} \to \mathbb{R}^{5H}$ (sorties $g, x, ch, dis, e$) :

```python
net, price = cp.Parameter(H), cp.Parameter(H)   # sans nonneg : 2*p_hat - p contient des négatifs
# ... variables g, x, c, d, e + contraintes physiques ...
cons += [g <= 500, x <= 500]                   # bornitude (prix négatifs : LP non borné sinon)
prob = cp.Problem(cp.Minimize(cost), cons)
layer = CvxpyLayer(prob, parameters=[net, price], variables=[g, x, c, d, e])
```

Différenciation implicite (diffcp) via le système KKT / conditions de sensibilité du point optimal :

$$
\frac{\partial z^\star}{\partial \theta}
= -\big[\nabla_{zz}^2 \mathcal{L}\big]^{-1}
\begin{bmatrix} \nabla_{z\theta}^2 \mathcal{L} \\ \nabla_{\lambda\theta}^2 \mathcal{L} \end{bmatrix}
$$

Solveur : **ECOS** uniquement (`diffcp` ne supporte qu'ECOS/SCS). Entraînement via `scripts/train_spo_exact.py` (env `.venv311`, `numpy<2` obligatoire — voir README). Fallback : SPO+ analytique (Section 5) sans dépendance C++.

### 6.1 Perte SPO+ exacte sur le LP EMS

Deux composantes, normalisées par $\text{scale} = \frac{1}{N}\sum_i\sum_t |p_t n_t| \approx 114{,}1$ :

1. **SPO+ (Définition 3)** — trois résolutions du LP EMS, toutes avec la **même**
   charge nette prédite $\hat n$ (ensemble faisable $\mathcal{X}$ fixe, exigence
   d'Elmachtoub & Grigas) :

   | symbole | problème résolu | rôle |
   |---------|------------------|------|
   | $z_1 = x^\star(\hat n, \hat p)$ | prix prédit | décision implémentée (terme de recours) |
   | $z_c = x^\star(\hat n, p)$ | prix vrai | oracle $w^\star(c)$ de la Définition 3 |
   | $z_2 = x^\star(\hat n, 2\hat p - p)$ | prix modifié | $w^\star(2\hat c - c)$ |

   Avec $\psi(z) = g - \beta x$ (gradient partiel du coût par rapport au prix) :

   $$
   \ell_{\text{SPO+}} = \big(2\hat p - p\big)^\top\big(\psi(z_c) - \psi(z_2)\big) \;\ge\; 0
   $$

   Gradient par rapport à $\hat p$ : $2\big(\psi(z_c) - \psi(z_2)\big)$ (terme explicite ;
   les solutions d'un LP sont constantes par morceaux en coefficients de coût, donc
   $z_c$ et $z_2$ sont détachés dans le graphe de differentiation).

2. **Terme de recours** (signal net exact) : planning $(ch_1, dis_1)$ réévalué sous les vraies séries — $b = n_{\text{true}} + ch_1 - dis_1$, coût $\sum_t p_{\text{true},t}(\mathrm{relu}(b_t) - \beta\,\mathrm{relu}(-b_t))$.

$\mathcal{L} = \mathcal{L}_{pinball} + \lambda\,(\ell_{\text{SPO+}} + \beta\,\ell_{\text{recours}})$, $\beta = 1$, grille $\lambda \in \{0{,}05; 0{,}1; 0{,}3; 0{,}5; 1; 1{,}5; 2; 3; 5\}$, *early stopping* sur le regret val (patience 2, 8 époques max), sélection finale du modèle par le **regret de validation**.

> **Note — grille λ étendue (2026-11-23).** La perte étant enfin conforme à la
> Définition 3 (et le shim `diffcp` rétablissant le gradient), l'optimum de la
> grille d'origine $\{0{,}05; 0{,}1; 0{,}3\}$ n'est plus le meilleur : le regret
> de validation décroît jusqu'à $\lambda \approx 3$. La grille complète explorée
> est donc utilisée pour la sélection, et chaque $\lambda$ est évalué sur le
> jeu de test (colonnes `test_regret_pct` de `results/spo_exact_results.json`).

Résolution numérique : **ECOS** en solveur principal ; en cas d'échec transitoire
sur un lot dégénéré, le triplet de résolutions est refait avec **SCS** (diffcp
supporte les deux) plutôt que d'abandonner le lot.

## 7. Pipeline d'entraînement decision-focused (3 phases)

1. **Phase 1 — warm-up forecast-focused** : $\min_\theta \mathcal{L}_{pinball}(\hat q_\theta, y)$.
2. **Phase 2 — fine-tuning decision-focused (voie boîte)** :
   $\min_\theta \ \mathcal{L}_{pinball} + \lambda\,\mathcal{L}_{SPO+}(\hat p_\theta, p)$
   ($\lambda$ croissant 0 → 0.5 ; SPO+ sur le prix, proxy du coût d'achat).
3. **Phase 3 — couplage exact (`.venv311`)** : reprise de `tft_pinball.pt` avec la perte de la section~6.1 (couche cvxpylayers, SPO+ + recours) ; grille $\lambda \times$ *early stopping* sur le regret val → `tft_spo_exact.pt`.
4. **Post-calibration** : conformal sur la validation.

## 8. Métriques d'évaluation

| Métrique | Définition | Rôle |
|---|---|---|
| Pinball | $\mathbb{E}\,\rho_\tau$ multi-$\tau$ | qualité marginale des quantiles |
| CRPS | Section 4.2 | qualité de la distribution complète |
| Couverture 80 % | $\mathbb{P}(y \in [q_{0.1}, q_{0.9}])$ | calibration |
| Coût opérationnel | $\sum p g - \beta p x$ via LP EMS | décision réelle |
| Regret | $J(\hat x) - J(x^\star)$ (vs clairvoyant) | coût de l'incertitude |
| Robustesse | coût pire-cas TSRO + écart type Monte-Carlo | stabilité sous choc |
