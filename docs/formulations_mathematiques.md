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

**SPOT loss** (non différenciable) :

$$
\mathcal{L}_{SPOT}(\hat c, c) = c^\top x^\star(\hat c) - c^\top x^\star(c)
$$

**SPO+** (borne inférieure convexe, dérivable presque partout) :

$$
\mathcal{L}_{SPO+}(\hat c, c) = \max_{x \in \mathcal{X}} \big(c - 2\hat c\big)^\top x
+ 2\hat c^\top x^\star(\hat c) - c^\top x^\star(\hat c)
$$

Propriétés : $\mathcal{L}_{SPOT} \le \mathcal{L}_{SPO+} \le 2\,\mathcal{L}_{SPOT}$ (facteur 2).

### 5.1 Gradient

Le max est atteint en $x^\star(2\hat c - c)$ (car $\min_{x}(2\hat c - c)^\top x$) :

$$
\nabla_{\hat c}\,\mathcal{L}_{SPO+} = 2\big(x^\star(\hat c) - x^\star(2\hat c - c)\big)
$$

Chaînage avec le forecaster $\hat c_\theta$ : $\nabla_\theta = \nabla_{\hat c} \cdot \partial \hat c_\theta / \partial \theta$.
Implémentation : `spo.spo_plus_loss` (LP SciPy/HiGHS batch) + `spo_plus_loss_torch` (tenseur détaché pour la régularisation décisionnelle dans la phase 2).

## 6. Couche d'optimisation différentielle (cvxpylayers)

Le LP EMS est traduit en couche $\mathbb{R}^{2H} \to \mathbb{R}^{H}$ :

```python
net, price = cp.Parameter(H), cp.Parameter(H, nonneg=True)
# ... variables g, x, c, d, e + contraintes physiques ...
prob = cp.Problem(cp.Minimize(cost), cons)
layer = CvxpyLayer(prob, parameters=[net, price], variables=[g, x, c, d, e])
```

Différenciation implicite (diffcp) via le système KKT / conditions de sensibilité du point optimal :

$$
\frac{\partial z^\star}{\partial \theta}
= -\big[\nabla_{zz}^2 \mathcal{L}\big]^{-1}
\begin{bmatrix} \nabla_{z\theta}^2 \mathcal{L} \\ \nabla_{\lambda\theta}^2 \mathcal{L} \end{bmatrix}
$$

Disponible dans `.venv311` (diffcp wheels py3.11). Fallback : SPO+ analytique (Section 5) sans dépendance C++.

## 7. Pipeline d'entraînement decision-focused (2 phases)

1. **Phase 1 — warm-up forecast-focused** : $\min_\theta \mathcal{L}_{pinball}(\hat q_\theta, y)$.
2. **Phase 2 — fine-tuning decision-focused** :
   $\min_\theta \ \mathcal{L}_{pinball} + \lambda\,\mathcal{L}_{SPO+}(\hat p_\theta, p)$
   ($\lambda$ croissant 0 → 0.5 ; SPO+ sur le prix, proxy du coût d'achat).
3. **Post-calibration** : conformal sur la validation.

## 8. Métriques d'évaluation

| Métrique | Définition | Rôle |
|---|---|---|
| Pinball | $\mathbb{E}\,\rho_\tau$ multi-$\tau$ | qualité marginale des quantiles |
| CRPS | Section 4.2 | qualité de la distribution complète |
| Couverture 80 % | $\mathbb{P}(y \in [q_{0.1}, q_{0.9}])$ | calibration |
| Coût opérationnel | $\sum p g - \beta p x$ via LP EMS | décision réelle |
| Regret | $J(\hat x) - J(x^\star)$ (vs clairvoyant) | coût de l'incertitude |
| Robustesse | coût pire-cas TSRO + écart type Monte-Carlo | stabilité sous choc |
