# Outils Mathématiques, Prérequis et Datasets — Article 1 : Decision-Focused Probabilistic Forecasting for Microgrid EMS

---

## Résumé Exécutif

Ce document technique constitue un guide de référence complet pour aborder l'article de recherche intitulé *Decision-Focused Probabilistic Forecasting for Microgrid EMS*. Il couvre l'ensemble des prérequis mathématiques nécessaires — algèbre linéaire, optimisation convexe, optimisation stochastique et robuste, probabilités et apprentissage automatique — ainsi que les outils spécifiques à l'article : la perte SPO+ (Smart Predict-then-Optimize), les couches d'optimisation différentiables, l'architecture Temporal Fusion Transformer, la formulation complète de l'EMS et la décomposition robuste bi-niveau [1].

Le document fournit également un guide pratique et exhaustif pour l'installation du stack logiciel Python (PyTorch, CVXPY, cvxpylayers, PyEPO, Gurobi/HiGHS, Pyomo) et pour le téléchargement des datasets de référence : réseaux IEEE 33-bus et 69-bus, environnement CityLearn v3, profils PV/charge/prix issus de NREL, ENTSO-E et Open Power Systems Data [1] [2]. Un plan d'apprentissage sur quatre semaines est proposé en fin de document pour guider le lecteur de manière progressive.

Ce guide s'adresse aux chercheurs et ingénieurs souhaitant reproduire, étendre ou comparer les méthodes de l'article sur des benchmarks standardisés. Il constitue une base de travail autonome, avec des équations LaTeX complètes, des blocs de code Python exécutables et des tableaux récapitulatifs.

---

## Table des Matières

1. [Prérequis Mathématiques Fondamentaux](#1-prérequis-mathématiques-fondamentaux)
   - 1.1 [Algèbre Linéaire](#11-algèbre-linéaire)
   - 1.2 [Optimisation Convexe](#12-optimisation-convexe)
   - 1.3 [Optimisation Stochastique et Robuste](#13-optimisation-stochastique-et-robuste)
   - 1.4 [Probabilités et Statistiques](#14-probabilités-et-statistiques)
   - 1.5 [Apprentissage Automatique](#15-apprentissage-automatique)
2. [Outils Mathématiques Spécifiques à l'Article](#2-outils-mathématiques-spécifiques-à-larticle)
   - 2.1 [SPO+ (Smart Predict-then-Optimize)](#21-spo-smart-predict-then-optimize)
   - 2.2 [Couches d'Optimisation Différentiables (cvxpylayers)](#22-couches-doptimisation-différentiables-cvxpylayers)
   - 2.3 [Temporal Fusion Transformer (TFT)](#23-temporal-fusion-transformer-tft)
   - 2.4 [Formulation EMS Complète](#24-formulation-ems-complète)
   - 2.5 [Two-Stage Robust Optimization (TSRO)](#25-two-stage-robust-optimization-tsro)
3. [Stack Logiciel et Librairies](#3-stack-logiciel-et-librairies)
4. [Guide Complet de Téléchargement des Datasets](#4-guide-complet-de-téléchargement-des-datasets)
   - 4.1 [Dataset IEEE 33-bus](#41-dataset-ieee-33-bus)
   - 4.2 [Dataset IEEE 69-bus](#42-dataset-ieee-69-bus)
   - 4.3 [CityLearn v3](#43-citylearn-v3)
   - 4.4 [Profils PV/Charge/Prix](#44-profils-pvchargeprix)
   - 4.5 [Script Python Complet](#45-script-python-complet-pour-télécharger-et-préparer-tous-les-datasets)
5. [Plan d'Apprentissage Recommandé](#5-plan-dapprentissage-recommandé)

---

## 1. Prérequis Mathématiques Fondamentaux

### 1.1 Algèbre Linéaire

L'algèbre linéaire est le socle de toutes les formulations d'optimisation et d'apprentissage automatique utilisées dans l'EMS. Cette section rappelle les notions essentielles et leur rôle concret dans le contexte de l'article.

#### 1.1.1 Vecteurs et Matrices

Un **vecteur** $\mathbf{x} \in \mathbb{R}^n$ représente, dans le contexte EMS, un état du système à un instant $t$ : puissances injectées, état de charge (SoC) des batteries, prix de l'énergie, etc. Une **matrice** $A \in \mathbb{R}^{m \times n}$ encode les contraintes linéaires du problème d'optimisation (bilan de puissance, limites de lignes, etc.).

**Opérations fondamentales :**

- Produit scalaire : $\langle \mathbf{c}, \mathbf{x} \rangle = \mathbf{c}^\top \mathbf{x} = \sum_{i=1}^n c_i x_i$ (fonction objectif linéaire)
- Produit matrice-vecteur : $A\mathbf{x} = \mathbf{b}$ (système de contraintes)
- Transposée : $(AB)^\top = B^\top A^\top$
- Inverse : $A^{-1}$ (si $A$ est carrée et inversible), $A A^{-1} = I$

#### 1.1.2 Normes

| Norme | Définition | Usage EMS |
|---|---|---|
| $\ell_1$ | $\|\mathbf{x}\|_1 = \sum_i |x_i|$ | Régularisation sparse, pénalité d'imbalance |
| $\ell_2$ | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | Régularisation Ridge, distance euclidienne |
| $\ell_\infty$ | $\|\mathbf{x}\|_\infty = \max_i |x_i|$ | Ensembles d'incertitude box (robust opt.) |
| Frobenius | $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ | Régularisation de matrices de poids |

#### 1.1.3 Décompositions Matricielles

**Décomposition en Valeurs Singulières (SVD) :**

$$A = U \Sigma V^\top$$

où $U \in \mathbb{R}^{m \times m}$ et $V \in \mathbb{R}^{n \times n}$ sont orthogonales, et $\Sigma \in \mathbb{R}^{m \times n}$ est diagonale avec les valeurs singulières $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$. La SVD est utilisée pour la réduction de dimensionnalité des profils de charge/PV (PCA), la compression de matrices de covariance dans les modèles probabilistes, et l'analyse de sensibilité des contraintes réseau.

**Décomposition de Cholesky :**

Pour une matrice symétrique définie positive $\Sigma \succ 0$ :

$$\Sigma = L L^\top$$

où $L$ est une matrice triangulaire inférieure. Cette décomposition est indispensable pour :
- Générer des échantillons gaussiens multivariés : $\mathbf{z} = \boldsymbol{\mu} + L\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$
- Résoudre efficacement des systèmes linéaires dans les QP
- Paramétrer les ambiguity sets dans la DRO (Distributionally Robust Optimization)

**Décomposition LU :**

$$A = LU$$

Utilisée dans les solveurs LP/QP pour résoudre $A\mathbf{x} = \mathbf{b}$ en $O(n^3)$.

#### 1.1.4 Opérations Matricielles dans l'EMS

Dans la formulation EMS, les contraintes de bilan de puissance sur un réseau de $N$ nœuds s'écrivent :

$$\mathbf{B} \boldsymbol{\theta} = \mathbf{P}_{\text{inj}}$$

où $\mathbf{B} \in \mathbb{R}^{N \times N}$ est la matrice susceptance (matrice de Laplacien pondéré), $\boldsymbol{\theta} \in \mathbb{R}^N$ le vecteur des angles de phase, et $\mathbf{P}_{\text{inj}} \in \mathbb{R}^N$ le vecteur des injections nettes. Les flux de lignes sont donnés par :

$$\mathbf{f} = \mathbf{H} \mathbf{P}_{\text{inj}}, \quad \mathbf{H} = \mathbf{B}_f \mathbf{B}^{-1}$$

où $\mathbf{H} \in \mathbb{R}^{L \times N}$ est la matrice de sensibilité (PTDF — Power Transfer Distribution Factors), $L$ étant le nombre de lignes [1].

---

### 1.2 Optimisation Convexe

#### 1.2.1 Fonctions Convexes

Une fonction $f : \mathbb{R}^n \to \mathbb{R}$ est **convexe** si et seulement si :

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}), \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^n, \lambda \in [0,1]$$

**Exemples dans l'EMS :**
- Coût d'achat d'énergie : $f(p) = c \cdot p$ (linéaire, donc convexe)
- Coût de dégradation batterie : $f(p_c, p_d) = \alpha (p_c^2 + p_d^2)$ (quadratique convexe)
- Pénalité de déséquilibre : $f(\Delta p) = \beta |\Delta p|$ (norme $\ell_1$, convexe)

**Conditions de convexité (fonctions différentiables) :**
- Ordre 1 : $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x})$
- Ordre 2 : $\nabla^2 f(\mathbf{x}) \succeq 0$ (hessienne semi-définie positive)

#### 1.2.2 Dualité de Lagrange

Pour un problème primal :

$$\min_{\mathbf{x}} \; f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \; i=1,\ldots,m, \quad h_j(\mathbf{x}) = 0, \; j=1,\ldots,p$$

Le **Lagrangien** est :

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x})$$

La **fonction duale** est :

$$d(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$$

Le **problème dual** est : $\max_{\boldsymbol{\lambda} \geq 0, \boldsymbol{\nu}} d(\boldsymbol{\lambda}, \boldsymbol{\nu})$.

**Saut de dualité** : $p^* - d^* \geq 0$. Sous les conditions de Slater (qualification des contraintes), $p^* = d^*$ (**dualité forte**), ce qui est exploité dans les algorithmes de décomposition de Benders et C&CG pour la TSRO [1].

#### 1.2.3 Conditions KKT

Les conditions de Karush-Kuhn-Tucker sont **nécessaires et suffisantes** pour les problèmes convexes avec dualité forte :

$$\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) = 0 \quad \text{(stationnarité)}$$
$$g_i(\mathbf{x}^*) \leq 0 \quad \text{(admissibilité primale)}$$
$$\lambda_i^* \geq 0 \quad \text{(admissibilité duale)}$$
$$\lambda_i^* g_i(\mathbf{x}^*) = 0 \quad \text{(complémentarité)}$$
$$h_j(\mathbf{x}^*) = 0 \quad \text{(contraintes d'égalité)}$$

Dans le contexte des couches d'optimisation différentiables (cvxpylayers), les conditions KKT sont différenciées implicitement pour calculer $\partial \mathbf{x}^* / \partial \boldsymbol{\theta}$ [1].

#### 1.2.4 Programmes Linéaires (LP)

$$\min_{\mathbf{x}} \; \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \leq \mathbf{b}, \quad \mathbf{x} \geq 0$$

**Exemple EMS (dispatch journalier simplifié) :**

$$\min_{p_g, p_c, p_d, p_{\text{imp}}} \; \sum_{t=1}^T \left( c_g p_{g,t} + c_{\text{imp}} p_{\text{imp},t} \right)$$

$$\text{s.t.} \quad p_{g,t} + p_{d,t} - p_{c,t} + p_{\text{imp},t} = d_t - \hat{p}_{\text{pv},t}, \quad \forall t$$

$$0 \leq p_{g,t} \leq P_g^{\max}, \quad 0 \leq p_{c,t} \leq P_c^{\max}, \quad 0 \leq p_{d,t} \leq P_d^{\max}$$

Résolu en temps polynomial par la méthode du simplexe ou les méthodes de points intérieurs.

#### 1.2.5 Programmes Quadratiques (QP)

$$\min_{\mathbf{x}} \; \frac{1}{2} \mathbf{x}^\top Q \mathbf{x} + \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \leq \mathbf{b}$$

avec $Q \succeq 0$. Utilisé pour modéliser les coûts de dégradation quadratiques des batteries ou les coûts de régulation de fréquence.

#### 1.2.6 Programmes Mixtes Entiers (MILP)

$$\min_{\mathbf{x}, \mathbf{z}} \; \mathbf{c}^\top \mathbf{x} + \mathbf{d}^\top \mathbf{z} \quad \text{s.t.} \quad A\mathbf{x} + B\mathbf{z} \leq \mathbf{b}, \quad \mathbf{x} \geq 0, \quad \mathbf{z} \in \{0,1\}^k$$

Les variables binaires $\mathbf{z}$ modélisent les décisions d'engagement/démarrage des générateurs (unit commitment), les modes de charge/décharge exclusifs de la batterie, et les décisions d'îlotage du microréseau. La résolution utilise le branch-and-bound avec des relaxations LP [1].

| Type | Variables | Solveurs | Complexité |
|---|---|---|---|
| LP | Continues | HiGHS, Gurobi, GLPK | Polynomial |
| QP | Continues | OSQP, Gurobi, ECOS | Polynomial |
| MILP | Mixtes entier/continu | Gurobi, HiGHS, CPLEX | NP-difficile |
| SOCP | Continues (cône) | ECOS, SCS, Mosek | Polynomial |

---

### 1.3 Optimisation Stochastique et Robuste

#### 1.3.1 Espérance Mathématique et Programmation Stochastique

Soit $\xi \in \Xi \subseteq \mathbb{R}^d$ un vecteur aléatoire (incertitudes : production PV, charge, prix) de distribution $\mathbb{P}$. Le **programme stochastique bi-niveau** (two-stage) est :

$$\min_{\mathbf{x} \in \mathcal{X}} \; \mathbf{c}^\top \mathbf{x} + \mathbb{E}_{\xi \sim \mathbb{P}}\left[ Q(\mathbf{x}, \xi) \right]$$

où la **fonction de recours** est :

$$Q(\mathbf{x}, \xi) = \min_{\mathbf{y} \in \mathcal{Y}(\xi)} \; \mathbf{q}^\top \mathbf{y} \quad \text{s.t.} \quad A\mathbf{x} + B\mathbf{y} \geq \mathbf{b}(\xi)$$

- $\mathbf{x}$ : décisions de **premier niveau** (day-ahead : engagement, bids)
- $\mathbf{y}(\xi)$ : décisions de **second niveau** (real-time : recours, ajustements)
- $\mathcal{X}$ : ensemble admissible des décisions de premier niveau
- $\mathcal{Y}(\xi)$ : ensemble admissible des décisions de recours (dépend de $\xi$)

En pratique, $\mathbb{E}[\cdot]$ est approchée par une **Sample Average Approximation (SAA)** sur $S$ scénarios $\{\xi^{(s)}\}_{s=1}^S$ :

$$\min_{\mathbf{x}} \; \mathbf{c}^\top \mathbf{x} + \frac{1}{S} \sum_{s=1}^S Q(\mathbf{x}, \xi^{(s)})$$

#### 1.3.2 Ensembles d'Incertitude et Optimisation Robuste

L'**optimisation robuste** remplace la distribution $\mathbb{P}$ par un ensemble d'incertitude $\mathcal{U}$ et optimise contre le pire cas :

$$\min_{\mathbf{x} \in \mathcal{X}} \; \mathbf{c}^\top \mathbf{x} + \max_{\xi \in \mathcal{U}} \; Q(\mathbf{x}, \xi)$$

**Types d'ensembles d'incertitude courants :**

| Type | Définition | Propriétés |
|---|---|---|
| Box | $\mathcal{U} = \{\xi : \|\xi - \hat{\xi}\|_\infty \leq \Gamma\}$ | Simple, conservateur |
| Ellipsoïdal | $\mathcal{U} = \{\xi : (\xi-\hat{\xi})^\top \Sigma^{-1} (\xi-\hat{\xi}) \leq \kappa^2\}$ | Moins conservateur, SOCP |
| Polyédral | $\mathcal{U} = \{\xi : F\xi \leq \mathbf{g}\}$ | Flexible, LP-compatible |
| Budget | $\mathcal{U} = \{\xi : \|\xi - \hat{\xi}\|_1 \leq \Gamma_1, \|\xi\|_\infty \leq \Gamma_\infty\}$ | Contrôle le niveau de robustesse |

#### 1.3.3 Optimisation Robuste Max-Min (TSRO)

La formulation **Two-Stage Robust Optimization** est :

$$\min_{\mathbf{x} \in \mathcal{X}} \; \mathbf{c}^\top \mathbf{x} + \max_{\xi \in \mathcal{U}} \min_{\mathbf{y} \in \mathcal{Y}(\xi)} \; \mathbf{q}^\top \mathbf{y} \quad \text{s.t.} \quad A\mathbf{x} + B\mathbf{y} \geq \mathbf{b}(\xi)$$

Ce problème tri-niveau est résolu par des algorithmes de décomposition (C&CG, Benders) détaillés en Section 2.5 [1].

#### 1.3.4 DRO — Distributionally Robust Optimization

La DRO optimise contre le pire cas sur un **ambiguity set** $\mathcal{D}$ de distributions :

$$\min_{\mathbf{x} \in \mathcal{X}} \; \mathbf{c}^\top \mathbf{x} + \sup_{\mathbb{P} \in \mathcal{D}} \; \mathbb{E}_{\xi \sim \mathbb{P}}\left[ Q(\mathbf{x}, \xi) \right]$$

**Ambiguity sets courants :**

- **Moment-based** : $\mathcal{D} = \{\mathbb{P} : \mathbb{E}[\xi] = \boldsymbol{\mu}, \mathbb{E}[(\xi-\boldsymbol{\mu})(\xi-\boldsymbol{\mu})^\top] \preceq \Sigma\}$
- **Wasserstein ball** : $\mathcal{D} = \{\mathbb{P} : W_p(\mathbb{P}, \hat{\mathbb{P}}_N) \leq \epsilon\}$ où $W_p$ est la distance de Wasserstein d'ordre $p$ et $\hat{\mathbb{P}}_N$ la distribution empirique sur $N$ échantillons
- **$\phi$-divergence** : $\mathcal{D} = \{\mathbb{P} : D_\phi(\mathbb{P} \| \hat{\mathbb{P}}) \leq \rho\}$ (KL-divergence, $\chi^2$-divergence)

La DRO permet de réduire le conservatisme de la robustesse pure tout en maintenant des garanties de performance sous incertitude distributionnelle [1].

---

### 1.4 Probabilités et Statistiques

#### 1.4.1 Distributions Conditionnelles

La **distribution conditionnelle** $\mathbb{P}(\xi_{t+1:t+H} | \mathcal{F}_t)$ de la trajectoire future $\xi_{t+1:t+H}$ sachant l'historique $\mathcal{F}_t$ est l'objet central des prévisions probabilistes. Dans l'EMS, $\xi$ peut représenter la production PV, la charge, ou les prix spot.

Pour une distribution gaussienne conditionnelle :

$$\xi_{t+h} | \mathcal{F}_t \sim \mathcal{N}(\mu_{t+h|t}, \sigma_{t+h|t}^2)$$

La densité conditionnelle est :

$$p(\xi | \mathcal{F}_t) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(\xi - \mu)^2}{2\sigma^2}\right)$$

#### 1.4.2 Quantiles et Intervalles de Prédiction

Le **quantile d'ordre $\tau \in (0,1)$** de la distribution conditionnelle est :

$$q_\tau = F^{-1}(\tau) = \inf\{y : F(y) \geq \tau\}$$

où $F$ est la CDF (Cumulative Distribution Function). Un **intervalle de prédiction** de niveau $(1-\alpha)$ est :

$$\text{PI}_{1-\alpha} = [q_{\alpha/2}, q_{1-\alpha/2}]$$

La **régression par quantile** minimise la **pinball loss** (aussi appelée quantile loss) :

$$\mathcal{L}_\tau(\hat{q}_\tau, y) = \begin{cases} \tau (y - \hat{q}_\tau) & \text{si } y \geq \hat{q}_\tau \\ (1-\tau)(\hat{q}_\tau - y) & \text{si } y < \hat{q}_\tau \end{cases}$$

#### 1.4.3 CRPS — Continuous Ranked Probability Score

Le **CRPS** est un score de vérification probabiliste qui mesure la qualité d'une prévision probabiliste $F$ par rapport à une observation $y$ :

$$\text{CRPS}(F, y) = \int_{-\infty}^{+\infty} \left[ F(z) - \mathbf{1}(z \geq y) \right]^2 dz$$

Équivalence avec la somme des pinball losses :

$$\text{CRPS}(F, y) = 2 \int_0^1 \mathcal{L}_\tau(F^{-1}(\tau), y) \, d\tau$$

Le CRPS est un **score propre** (proper scoring rule) : il est minimisé en espérance si et seulement si $F$ est la vraie distribution. Cependant, minimiser le CRPS ne garantit pas l'optimalité des décisions EMS — c'est précisément la motivation de SPO+ [1].

#### 1.4.4 Calibration Probabiliste

Une prévision probabiliste est **calibrée** si les quantiles empiriques correspondent aux quantiles théoriques :

$$\mathbb{P}(y \leq \hat{q}_\tau) = \tau, \quad \forall \tau \in (0,1)$$

Le **PIT (Probability Integral Transform)** histogram est l'outil de diagnostic standard : si $u_t = F_t(y_t)$ est uniformément distribué sur $[0,1]$, la prévision est calibrée. La calibration est nécessaire mais pas suffisante pour la valeur décisionnelle [1].

---

### 1.5 Apprentissage Automatique

#### 1.5.1 Réseaux de Neurones

Un réseau de neurones à $L$ couches est une composition de fonctions :

$$f_\theta(\mathbf{x}) = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})$$

où chaque couche $f_l(\mathbf{h}) = \sigma_l(W_l \mathbf{h} + \mathbf{b}_l)$, avec $W_l \in \mathbb{R}^{d_l \times d_{l-1}}$ la matrice de poids, $\mathbf{b}_l \in \mathbb{R}^{d_l}$ le biais, et $\sigma_l$ la fonction d'activation.

**Fonctions d'activation courantes :**

| Activation | Formule | Usage |
|---|---|---|
| ReLU | $\max(0, x)$ | Couches cachées standard |
| GELU | $x \cdot \Phi(x)$ | Transformers (TFT) |
| Sigmoid | $1/(1+e^{-x})$ | Sorties probabilistes $\in [0,1]$ |
| Softmax | $e^{x_i}/\sum_j e^{x_j}$ | Distributions de probabilité |
| ELU | $x$ si $x>0$, $\alpha(e^x-1)$ sinon | Réseaux profonds |

#### 1.5.2 Descente de Gradient et Backpropagation

L'entraînement minimise une loss function $\mathcal{L}(\theta)$ par descente de gradient stochastique (SGD) :

$$\theta_{k+1} = \theta_k - \eta_k \nabla_\theta \mathcal{L}(\theta_k)$$

où $\eta_k > 0$ est le taux d'apprentissage. La **backpropagation** calcule $\nabla_\theta \mathcal{L}$ par la règle de la chaîne :

$$\frac{\partial \mathcal{L}}{\partial W_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \cdot \frac{\partial \mathbf{h}_L}{\partial \mathbf{h}_{L-1}} \cdots \frac{\partial \mathbf{h}_{l+1}}{\partial \mathbf{h}_l} \cdot \frac{\partial \mathbf{h}_l}{\partial W_l}$$

**Optimiseurs avancés :**

- **Adam** : $m_k = \beta_1 m_{k-1} + (1-\beta_1)g_k$, $v_k = \beta_2 v_{k-1} + (1-\beta_2)g_k^2$, $\theta_{k+1} = \theta_k - \eta \hat{m}_k / (\sqrt{\hat{v}_k} + \epsilon)$
- **AdamW** : Adam avec weight decay découplé
- **LBFGS** : quasi-Newton, utilisé pour les fine-tunings de petits modèles

#### 1.5.3 Loss Functions

| Loss | Formule | Usage |
|---|---|---|
| MSE | $\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2$ | Régression point |
| MAE | $\frac{1}{N}\sum_i |y_i - \hat{y}_i|$ | Régression robuste |
| Pinball | $\mathcal{L}_\tau(\hat{q}, y)$ | Régression quantile |
| Cross-entropy | $-\sum_k y_k \log \hat{p}_k$ | Classification |
| NLL gaussienne | $\frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$ | Sortie probabiliste |
| SPO+ | Voir Section 2.1 | Decision-focused |

#### 1.5.4 Régularisation

- **L2 (Ridge)** : $\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\theta\|_2^2$
- **L1 (Lasso)** : $\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\theta\|_1$
- **Dropout** : mise à zéro aléatoire de neurones avec probabilité $p$ pendant l'entraînement
- **Batch Normalization** : $\hat{h} = (h - \mu_B)/\sigma_B$, puis $y = \gamma \hat{h} + \beta$
- **Early stopping** : arrêt de l'entraînement quand la loss de validation cesse de décroître

---

## 2. Outils Mathématiques Spécifiques à l'Article

### 2.1 SPO+ (Smart Predict-then-Optimize)

#### 2.1.1 Motivation : Le Regret Décisionnel

Dans le pipeline **predict-then-optimize** classique, un prédicteur $\hat{\mathbf{c}} = f_\theta(\mathbf{x})$ estime les coûts $\mathbf{c}$ (ou les paramètres d'incertitude), puis un solveur calcule la décision optimale $\mathbf{z}^*(\hat{\mathbf{c}}) = \arg\min_{\mathbf{z} \in \mathcal{Z}} \hat{\mathbf{c}}^\top \mathbf{z}$.

Le **regret décisionnel** (SPO loss) mesure le coût supplémentaire dû à l'imprécision de la prédiction :

$$\ell_{\text{SPO}}(\hat{\mathbf{c}}, \mathbf{c}) = \mathbf{c}^\top \mathbf{z}^*(\hat{\mathbf{c}}) - \mathbf{c}^\top \mathbf{z}^*(\mathbf{c})$$

où $\mathbf{z}^*(\mathbf{c}) = \arg\min_{\mathbf{z} \in \mathcal{Z}} \mathbf{c}^\top \mathbf{z}$ est la décision clairvoyante (oracle). Le problème est que $\ell_{\text{SPO}}$ est **non-convexe et discontinue** en $\hat{\mathbf{c}}$, ce qui rend la descente de gradient directe impossible [1].

#### 2.1.2 Formulation SPO+

La **surrogate loss SPO+** est une relaxation convexe de $\ell_{\text{SPO}}$ :

$$\ell_{\text{SPO+}}(\hat{\mathbf{c}}, \mathbf{c}) = \max_{\mathbf{z} \in \mathcal{Z}} \left\{ (2\hat{\mathbf{c}} - \mathbf{c})^\top \mathbf{z} \right\} - 2\hat{\mathbf{c}}^\top \mathbf{z}^*(\mathbf{c}) + \mathbf{c}^\top \mathbf{z}^*(\mathbf{c})$$

En développant :

$$\ell_{\text{SPO+}}(\hat{\mathbf{c}}, \mathbf{c}) = \underbrace{\max_{\mathbf{z} \in \mathcal{Z}} (2\hat{\mathbf{c}} - \mathbf{c})^\top \mathbf{z}}_{\text{terme de support}} - 2\hat{\mathbf{c}}^\top \mathbf{z}^*(\mathbf{c}) + \mathbf{c}^\top \mathbf{z}^*(\mathbf{c})$$

**Propriétés clés :**
- $\ell_{\text{SPO+}}$ est **convexe** en $\hat{\mathbf{c}}$ (max de fonctions linéaires)
- $\ell_{\text{SPO+}} \geq \ell_{\text{SPO}}$ (upper bound du regret)
- $\ell_{\text{SPO+}} = \ell_{\text{SPO}}$ quand $\hat{\mathbf{c}} = \mathbf{c}$ (consistance)

#### 2.1.3 Calcul du Gradient

Le gradient de $\ell_{\text{SPO+}}$ par rapport à $\hat{\mathbf{c}}$ est :

$$\nabla_{\hat{\mathbf{c}}} \ell_{\text{SPO+}}(\hat{\mathbf{c}}, \mathbf{c}) = 2\left(\mathbf{z}^*(2\hat{\mathbf{c}} - \mathbf{c}) - \mathbf{z}^*(\mathbf{c})\right)$$

où $\mathbf{z}^*(2\hat{\mathbf{c}} - \mathbf{c}) = \arg\min_{\mathbf{z} \in \mathcal{Z}} (2\hat{\mathbf{c}} - \mathbf{c})^\top \mathbf{z}$ est obtenu en résolvant un LP/QP auxiliaire. Ce gradient est **piecewise constant** (sous-gradient) et peut être utilisé directement dans SGD/Adam [1].

#### 2.1.4 Intégration dans la Boucle d'Entraînement

```python
# Pseudo-code SPO+ training loop
for epoch in range(num_epochs):
    for X_batch, c_batch in dataloader:
        # Forward pass : prédiction des coûts
        c_hat = model(X_batch)  # shape: (batch, n_vars)
        
        # Résolution du LP auxiliaire SPO+
        # z_spo = argmin_{z in Z} (2*c_hat - c_batch)^T z
        z_spo = solve_lp(2 * c_hat - c_batch, constraints)
        
        # Décision oracle (clairvoyante)
        z_star = solve_lp(c_batch, constraints)
        
        # SPO+ loss
        loss = torch.mean(
            torch.sum((2*c_hat - c_batch) * z_spo, dim=1)
            - 2 * torch.sum(c_hat * z_star, dim=1)
            + torch.sum(c_batch * z_star, dim=1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()  # gradient = 2*(z_spo - z_star)
        optimizer.step()
```

#### 2.1.5 Extension Probabiliste

Dans le cadre probabiliste de l'article, $\hat{\mathbf{c}}$ est remplacé par une distribution prédictive $\hat{F}$ (quantiles, scénarios). La loss SPO+ est étendue en espérance sur les scénarios :

$$\ell_{\text{SPO+}}^{\text{prob}}(\hat{F}, \mathbf{c}) = \mathbb{E}_{\hat{\xi} \sim \hat{F}}\left[\ell_{\text{SPO+}}(\hat{\xi}, \mathbf{c})\right]$$

---

### 2.2 Couches d'Optimisation Différentiables (cvxpylayers)

#### 2.2.1 Principe

Une **couche d'optimisation différentiable** encapsule un programme convexe paramétrique comme une couche de réseau de neurones, permettant de propager les gradients à travers le solveur [1].

Le programme paramétrique est de la forme :

$$\mathbf{z}^*(\boldsymbol{\theta}) = \arg\min_{\mathbf{z}} \; f(\mathbf{z}; \boldsymbol{\theta}) \quad \text{s.t.} \quad g_i(\mathbf{z}; \boldsymbol{\theta}) \leq 0, \; h_j(\mathbf{z}; \boldsymbol{\theta}) = 0$$

où $\boldsymbol{\theta}$ sont les paramètres (sorties du réseau de neurones, e.g., prévisions de charge/PV).

#### 2.2.2 Forward Pass

Le **forward pass** résout le programme convexe pour obtenir $\mathbf{z}^*(\boldsymbol{\theta})$ via un solveur (ECOS, SCS, Gurobi). La solution $\mathbf{z}^*$ est utilisée pour calculer la loss décisionnelle.

#### 2.2.3 Backward Pass — Différentiation Implicite

Le **backward pass** calcule $\partial \mathbf{z}^* / \partial \boldsymbol{\theta}$ par **différentiation implicite** des conditions KKT. En différenciant le système KKT :

$$\begin{pmatrix} \nabla^2_{\mathbf{z}\mathbf{z}} \mathcal{L} & \nabla_{\mathbf{z}} g^\top & \nabla_{\mathbf{z}} h^\top \\ \text{diag}(\boldsymbol{\lambda}) \nabla_{\mathbf{z}} g & \text{diag}(g) & 0 \\ \nabla_{\mathbf{z}} h & 0 & 0 \end{pmatrix} \begin{pmatrix} d\mathbf{z} \\ d\boldsymbol{\lambda} \\ d\boldsymbol{\nu} \end{pmatrix} = -\frac{\partial}{\partial \boldsymbol{\theta}} \begin{pmatrix} \nabla_{\mathbf{z}} \mathcal{L} \\ \text{diag}(\boldsymbol{\lambda}) g \\ h \end{pmatrix} d\boldsymbol{\theta}$$

On obtient $d\mathbf{z}/d\boldsymbol{\theta}$ en résolvant ce système linéaire, ce qui est $O(n^3)$ mais ne nécessite pas de différencier le solveur lui-même [1].

#### 2.2.4 Exemple avec cvxpylayers

```python
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

# Définition du programme EMS (dispatch BESS)
n_t = 24  # horizon temporel
p_c = cp.Variable(n_t, nonneg=True)   # puissance de charge
p_d = cp.Variable(n_t, nonneg=True)   # puissance de décharge
soc  = cp.Variable(n_t + 1)           # état de charge

# Paramètres (sorties du prédicteur)
p_pv_hat = cp.Parameter(n_t)          # prévision PV
p_load_hat = cp.Parameter(n_t)        # prévision charge
price = cp.Parameter(n_t)             # prix de l'énergie

# Paramètres physiques
eta_c, eta_d = 0.95, 0.95
soc_min, soc_max = 0.1, 0.9
p_max = 50.0  # kW
soc_init = 0.5

# Contraintes
constraints = [
    soc == soc_init,
    soc[n_t] == soc_init,  # état final = état initial
]
for t in range(n_t):
    constraints += [
        soc[t+1] == soc[t] + eta_c * p_c[t] - (1/eta_d) * p_d[t],
        soc[t+1] >= soc_min,
        soc[t+1] <= soc_max,
        p_c[t] <= p_max,
        p_d[t] <= p_max,
        # Bilan de puissance (import réseau = load - PV - décharge + charge)
        p_load_hat[t] - p_pv_hat[t] - p_d[t] + p_c[t] >= 0,
    ]

# Fonction objectif : minimiser le coût d'achat réseau
p_import = p_load_hat - p_pv_hat - p_d + p_c
objective = cp.Minimize(price @ p_import)

problem = cp.Problem(objective, constraints)

# Création de la couche différentiable
ems_layer = CvxpyLayer(
    problem,
    parameters=[p_pv_hat, p_load_hat, price],
    variables=[p_c, p_d, soc]
)

# Utilisation dans PyTorch (forward + backward automatique)
pv_pred = torch.randn(24, requires_grad=True)
load_pred = torch.randn(24, requires_grad=True)
price_tensor = torch.rand(24)

p_c_sol, p_d_sol, soc_sol = ems_layer(pv_pred, load_pred, price_tensor)

# La loss peut être n'importe quelle fonction différentiable de la solution
loss = (price_tensor * (load_pred - pv_pred - p_d_sol + p_c_sol)).sum()
loss.backward()  # gradients propagés jusqu'à pv_pred et load_pred
```

---

### 2.3 Temporal Fusion Transformer (TFT)

#### 2.3.1 Architecture Générale

Le **Temporal Fusion Transformer** est une architecture de deep learning pour les séries temporelles multivariées qui combine des mécanismes d'attention multi-têtes, des réseaux récurrents (LSTM), et des mécanismes de sélection de variables [1].

L'architecture se décompose en cinq blocs principaux :

```
Entrées (covariables statiques + temporelles passées/futures)
    │
    ▼
[1] Variable Selection Networks (VSN)
    │  → Sélection pondérée des variables pertinentes
    ▼
[2] Encodeur LSTM (passé) + Décodeur LSTM (futur)
    │  → Capture des dépendances temporelles locales
    ▼
 Gated Residual Networks (GRN)
    │  → Transformations non-linéaires avec skip connections
    ▼
 Multi-Head Attention (Temporal Self-Attention)
    │  → Capture des dépendances temporelles longue portée
    ▼
 Têtes de sortie quantile
    │  → Prédiction de quantiles multiples
    ▼
Sorties : {q̂_τ(t+h) : τ ∈ {0.1, 0.5, 0.9}, h = 1,...,H}
```

#### 2.3.2 Mécanisme d'Attention

Le **Scaled Dot-Product Attention** est :

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

où $Q = XW_Q$, $K = XW_K$, $V = XW_V$ sont les matrices de requêtes, clés et valeurs, et $d_k$ est la dimension des clés. Le **Multi-Head Attention** concatène $H$ têtes d'attention :

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Dans le TFT, l'attention est **interprétable** : les poids d'attention $\alpha_{t,t'} = \text{softmax}(q_t k_{t'}^\top / \sqrt{d_k})$ indiquent l'importance du pas de temps $t'$ pour la prédiction au pas $t$.

#### 2.3.3 Gated Residual Network (GRN)

Le GRN est le bloc de base du TFT :

$$\text{GRN}(\mathbf{a}, \mathbf{c}) = \text{LayerNorm}\left(\mathbf{a} + \text{GLU}\left(\eta_1(\mathbf{a}, \mathbf{c})\right)\right)$$

$$\eta_1(\mathbf{a}, \mathbf{c}) = W_1 \text{ELU}(W_2 \mathbf{a} + W_3 \mathbf{c} + \mathbf{b}_2) + \mathbf{b}_1$$

$$\text{GLU}(\mathbf{h}) = \sigma(\mathbf{h}_{[:d/2]}) \odot \mathbf{h}_{[d/2:]}$$

où $\mathbf{c}$ est un contexte optionnel (covariables statiques), $\text{GLU}$ est la Gated Linear Unit, et $\odot$ est le produit élément par élément.

#### 2.3.4 Sorties Probabilistes et Quantile Loss

Le TFT produit des **prédictions de quantiles multiples** pour chaque horizon $h$ :

$$\hat{q}_\tau(t+h) = f_\tau(\mathbf{x}_{1:t}, \mathbf{s})$$

La **quantile loss** (pinball loss) pour l'entraînement est :

$$\mathcal{L}_{\text{TFT}} = \sum_{h=1}^H \sum_{\tau \in \mathcal{T}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}_\tau\left(\hat{q}_\tau^{(i)}(t_i+h), y_{t_i+h}^{(i)}\right)$$

avec $\mathcal{T} = \{0.1, 0.2, \ldots, 0.9\}$ typiquement. Pour l'intégration avec SPO+, la loss est remplacée ou augmentée par la surrogate décisionnelle [1].

#### 2.3.5 Variable Selection Network (VSN)

Le VSN sélectionne dynamiquement les variables d'entrée pertinentes :

$$\mathbf{v}_\chi = \text{softmax}\left(\text{GRN}_\chi(\boldsymbol{\xi}, \mathbf{c}_s)\right)$$

$$\tilde{\boldsymbol{\xi}} = \sum_{j=1}^{m_\chi} v_\chi^{(j)} \tilde{\boldsymbol{\xi}}^{(j)}$$

où $\boldsymbol{\xi}$ est le vecteur de toutes les variables d'entrée, $\mathbf{c}_s$ le contexte statique, et $v_\chi^{(j)}$ le poids de sélection de la $j$-ème variable. Les poids $v_\chi^{(j)}$ sont interprétables et permettent d'identifier les variables les plus informatives pour l'EMS (e.g., température, heure, prix).

---

### 2.4 Formulation EMS Complète

#### 2.4.1 Variables de Décision

| Variable | Symbole | Domaine | Description |
|---|---|---|---|
| Puissance de charge BESS | $p_{c,t}$ | $[0, P_c^{\max}]$ | Puissance absorbée par la batterie |
| Puissance de décharge BESS | $p_{d,t}$ | $[0, P_d^{\max}]$ | Puissance fournie par la batterie |
| État de charge | $\text{SoC}_t$ | $[\text{SoC}^{\min}, \text{SoC}^{\max}]$ | Énergie stockée normalisée |
| Import réseau | $p_{\text{imp},t}$ | $[0, P_{\text{imp}}^{\max}]$ | Puissance achetée au réseau |
| Export réseau | $p_{\text{exp},t}$ | $[0, P_{\text{exp}}^{\max}]$ | Puissance vendue au réseau |
| Délestage charge | $p_{\text{curt},t}$ | $[0, d_t]$ | Charge non servie (pénalité) |
| Curtailment PV | $p_{\text{pv,curt},t}$ | $[0, p_{\text{pv},t}]$ | Production PV écrêtée |
| Engagement générateur | $u_{g,t}$ | $\{0,1\}$ | Décision binaire ON/OFF |

#### 2.4.2 Fonction Objectif

$$\min_{\mathbf{x}} \; \sum_{t=1}^T \left[ c_{\text{imp},t} p_{\text{imp},t} - c_{\text{exp},t} p_{\text{exp},t} + c_{\text{deg}} (p_{c,t}^2 + p_{d,t}^2) + c_{\text{curt}} p_{\text{curt},t} + c_{\text{su}} u_{g,t}(1-u_{g,t-1}) \right]$$

où :
- $c_{\text{imp},t}$ : prix d'achat de l'énergie au temps $t$ (€/kWh)
- $c_{\text{exp},t}$ : prix de vente de l'énergie au temps $t$ (€/kWh)
- $c_{\text{deg}}$ : coût de dégradation de la batterie (€/kW²)
- $c_{\text{curt}}$ : pénalité de délestage (€/kWh)
- $c_{\text{su}}$ : coût de démarrage du générateur (€)

#### 2.4.3 Contraintes BESS

**Dynamique de l'état de charge :**

$$\text{SoC}_{t+1} = \text{SoC}_t + \frac{\eta_c \Delta t}{E_{\text{cap}}} p_{c,t} - \frac{\Delta t}{\eta_d E_{\text{cap}}} p_{d,t}, \quad \forall t \in \{1,\ldots,T\}$$

**Bornes sur le SoC :**

$$\text{SoC}^{\min} \leq \text{SoC}_t \leq \text{SoC}^{\max}, \quad \forall t$$

**Condition initiale et finale :**

$$\text{SoC}_1 = \text{SoC}_T = \text{SoC}_0$$

**Limites de puissance :**

$$0 \leq p_{c,t} \leq P_c^{\max}, \quad 0 \leq p_{d,t} \leq P_d^{\max}, \quad \forall t$$

**Exclusion mutuelle charge/décharge (avec variables binaires) :**

$$p_{c,t} \leq P_c^{\max} \delta_t, \quad p_{d,t} \leq P_d^{\max} (1 - \delta_t), \quad \delta_t \in \{0,1\}, \quad \forall t$$

#### 2.4.4 Bilan de Puissance

$$p_{\text{pv},t} - p_{\text{pv,curt},t} + p_{d,t} - p_{c,t} + p_{\text{imp},t} - p_{\text{exp},t} + p_{g,t} = d_t - p_{\text{curt},t}, \quad \forall t$$

#### 2.4.5 Contraintes Réseau (DC Power Flow)

Pour un réseau de $N$ nœuds et $L$ lignes :

$$\mathbf{f}_t = \mathbf{H} \mathbf{P}_{\text{inj},t}, \quad \forall t$$

$$-\mathbf{f}^{\max} \leq \mathbf{f}_t \leq \mathbf{f}^{\max}, \quad \forall t$$

$$\mathbf{P}_{\text{inj},t} = \mathbf{P}_{\text{gen},t} - \mathbf{d}_t + \mathbf{p}_{\text{curt},t}$$

où $\mathbf{H} \in \mathbb{R}^{L \times N}$ est la matrice PTDF et $\mathbf{f}^{\max} \in \mathbb{R}^L$ les limites de transit [1].

#### 2.4.6 Contraintes Générateur

$$P_g^{\min} u_{g,t} \leq p_{g,t} \leq P_g^{\max} u_{g,t}, \quad \forall t$$

$$p_{g,t} - p_{g,t-1} \leq \text{RU}_g, \quad p_{g,t-1} - p_{g,t} \leq \text{RD}_g, \quad \forall t$$

---

### 2.5 Two-Stage Robust Optimization (TSRO)

#### 2.5.1 Formulation Min-Max

La formulation TSRO pour l'EMS est :

$$\min_{\mathbf{x} \in \mathcal{X}} \; \mathbf{c}^\top \mathbf{x} + \max_{\boldsymbol{\xi} \in \mathcal{U}} \min_{\mathbf{y} \in \mathcal{Y}(\mathbf{x}, \boldsymbol{\xi})} \; \mathbf{q}^\top \mathbf{y}$$

avec :
- $\mathbf{x}$ : décisions day-ahead (engagement, bids, planification BESS)
- $\boldsymbol{\xi} = (\xi_{\text{pv}}, \xi_{\text{load}}, \xi_{\text{price}})$ : vecteur d'incertitude
- $\mathcal{U}$ : ensemble d'incertitude polyédral ou ellipsoïdal
- $\mathbf{y}$ : recours real-time (ajustements de puissance, délestage)
- $\mathcal{Y}(\mathbf{x}, \boldsymbol{\xi})$ : ensemble admissible du recours (dépend de $\mathbf{x}$ et $\boldsymbol{\xi}$)

**Ensemble d'incertitude budget :**

$$\mathcal{U} = \left\{ \boldsymbol{\xi} : \xi_i^- \leq \xi_i \leq \xi_i^+, \; \sum_{i=1}^n \frac{\xi_i - \hat{\xi}_i}{\xi_i^+ - \xi_i^-} \leq \Gamma \right\}$$

où $\Gamma \in [0, n]$ est le **budget d'incertitude** contrôlant le niveau de conservatisme [1].

#### 2.5.2 Algorithme Column & Constraint Generation (C&CG)

L'algorithme C&CG résout le problème TSRO par itérations maître/sous-problème :

**Problème maître (Master Problem — MP) :**

$$\min_{\mathbf{x}, \eta} \; \mathbf{c}^\top \mathbf{x} + \eta$$

$$\text{s.t.} \quad \eta \geq \mathbf{q}^\top \mathbf{y}^{(k)}, \quad A\mathbf{x} + B\mathbf{y}^{(k)} \geq \mathbf{b}(\boldsymbol{\xi}^{(k)}), \quad \mathbf{y}^{(k)} \in \mathcal{Y}, \quad k = 1,\ldots,K$$

**Sous-problème (Subproblem — SP) :**

$$\max_{\boldsymbol{\xi} \in \mathcal{U}} \min_{\mathbf{y} \in \mathcal{Y}(\mathbf{x}^*, \boldsymbol{\xi})} \; \mathbf{q}^\top \mathbf{y}$$

Le SP est un problème max-min qui, sous certaines conditions (recours complet, $\mathcal{Y}$ polyédral), peut être reformulé en un seul MILP par dualisation du problème interne :

$$\max_{\boldsymbol{\xi} \in \mathcal{U}, \boldsymbol{\lambda} \geq 0} \; \boldsymbol{\lambda}^\top (\mathbf{b}(\boldsymbol{\xi}) - A\mathbf{x}^*)$$

$$\text{s.t.} \quad B^\top \boldsymbol{\lambda} \leq \mathbf{q}, \quad \boldsymbol{\xi} \in \mathcal{U}$$

**Algorithme C&CG complet :**

```
Initialisation : K = 0, LB = -∞, UB = +∞, ε = 1e-4

Répéter :
    1. Résoudre MP → obtenir (x*, η*)
       LB = c^T x* + η*
    
    2. Résoudre SP avec x = x* → obtenir (ξ*, v*)
       UB = min(UB, c^T x* + v*)
    
    3. Si (UB - LB) / |UB| ≤ ε : STOP (convergence)
    
    4. Ajouter nouvelle colonne et contrainte au MP :
       - Nouvelle variable y^(K+1)
       - Contrainte : η ≥ q^T y^(K+1)
       - Contrainte : Ax* + By^(K+1) ≥ b(ξ*)
       K = K + 1

Retourner x* (solution robuste optimale)
```

#### 2.5.3 Décomposition de Benders

La **décomposition de Benders** est une alternative au C&CG, particulièrement adaptée quand le sous-problème est un LP :

**Coupe de Benders (optimality cut) :**

$$\eta \geq \mathbf{q}^\top \mathbf{y}^* + (\boldsymbol{\lambda}^*)^\top (B\mathbf{y}^* - \mathbf{b}(\boldsymbol{\xi}^*) + A\mathbf{x})$$

**Coupe de faisabilité (feasibility cut) :**

$$(\mathbf{r}^*)^\top (\mathbf{b}(\boldsymbol{\xi}^*) - A\mathbf{x}) \leq 0$$

où $\mathbf{r}^*$ est un rayon extrême du cône dual. La convergence est garantie en un nombre fini d'itérations pour les LP [1].

---

## 3. Stack Logiciel et Librairies

### 3.1 Vue d'Ensemble

| Librairie | Version | Rôle | Lien GitHub |
|---|---|---|---|
| Python | 3.10+ | Langage de base | https://python.org |
| PyTorch | 2.x | Deep learning, autograd | https://github.com/pytorch/pytorch |
| CVXPY | 1.4+ | Modélisation optimisation convexe | https://github.com/cvxpy/cvxpy |
| cvxpylayers | 0.1.6+ | Couches d'optimisation différentiables | https://github.com/cvxgrp/cvxpylayers |
| PyEPO | 0.3+ | SPO+, decision-focused learning | https://github.com/khalil-research/PyEPO |
| Gurobi | 11.x | Solveur MILP/QP commercial | https://github.com/Gurobi |
| HiGHS | 1.7+ | Solveur LP/MILP open-source | https://github.com/ERGO-Code/HiGHS |
| Pyomo | 6.x | Modélisation optimisation générale | https://github.com/Pyomo/pyomo |
| NumPy | 1.26+ | Calcul numérique | https://github.com/numpy/numpy |
| Pandas | 2.x | Manipulation de données | https://github.com/pandas-dev/pandas |
| Matplotlib | 3.8+ | Visualisation | https://github.com/matplotlib/matplotlib |
| CityLearn | 3.0.2 | Environnement EMS benchmark | https://github.com/citylearn-project/CityLearn |

### 3.2 Installation Complète

#### 3.2.1 Création de l'Environnement Virtuel

```bash
# Créer et activer un environnement virtuel Python 3.10+
python3.10 -m venv venv_ems
source venv_ems/bin/activate  # Linux/macOS
# venv_ems\Scripts\activate   # Windows

# Mise à jour de pip
pip install --upgrade pip setuptools wheel
```

#### 3.2.2 Installation via pip

```bash
# ─── Core scientific stack ───────────────────────────────────────────────────
pip install numpy>=1.26 pandas>=2.0 matplotlib>=3.8 scipy>=1.11

# ─── PyTorch (CPU) ───────────────────────────────────────────────────────────
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ─── PyTorch (GPU CUDA 12.1) ─────────────────────────────────────────────────
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ─── Optimisation convexe ────────────────────────────────────────────────────
pip install cvxpy>=1.4
pip install cvxpylayers>=0.1.6

# ─── Decision-focused learning ───────────────────────────────────────────────
pip install pyepo>=0.3.6

# ─── Solveurs open-source ────────────────────────────────────────────────────
pip install highspy>=1.7          # HiGHS LP/MILP solver
pip install ecos>=2.0.12          # ECOS conic solver
pip install scs>=3.2              # SCS large-scale solver
pip install osqp>=0.6             # OSQP quadratic solver

# ─── Modélisation optimisation ───────────────────────────────────────────────
pip install pyomo>=6.7

# ─── Time series & forecasting ───────────────────────────────────────────────
pip install pytorch-forecasting>=1.0   # TFT implementation
pip install pytorch-lightning>=2.0
pip install darts>=0.27                # Alternative TFT

# ─── CityLearn v3 ────────────────────────────────────────────────────────────
python -m pip install citylearn        # version 3.0.2

# ─── Données et utilitaires ──────────────────────────────────────────────────
pip install requests tqdm h5py netCDF4 openpyxl xlrd
pip install scikit-learn>=1.3
pip install seaborn plotly
```

#### 3.2.3 Installation via conda

```bash
# Créer l'environnement conda
conda create -n ems_env python=3.10 -y
conda activate ems_env

# Canaux recommandés
conda config --add channels conda-forge
conda config --add channels pytorch

# Installation des packages principaux
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install numpy pandas matplotlib scipy scikit-learn -y
conda install -c conda-forge cvxpy pyomo highspy -y

# Packages pip uniquement
pip install cvxpylayers pyepo pytorch-forecasting pytorch-lightning citylearn
```

#### 3.2.4 Installation de Gurobi (Licence Académique)

```bash
# Installation du package Python
pip install gurobipy>=11.0

# Activation de la licence académique (nécessite un compte Gurobi)
# 1. Créer un compte sur https://www.gurobi.com/academia/academic-program-and-licenses/
# 2. Télécharger la licence gurobi.lic
# 3. Placer le fichier dans ~/gurobi.lic (Linux/macOS) ou C:\gurobi\gurobi.lic (Windows)

# Vérification
python -c "import gurobipy; m = gurobipy.Model(); print('Gurobi OK')"
```

#### 3.2.5 Vérification de l'Installation

```python
# verify_install.py
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

import cvxpy as cp
print(f"CVXPY: {cp.__version__}")
print(f"CVXPY solvers: {cp.installed_solvers()}")

from cvxpylayers.torch import CvxpyLayer
print("cvxpylayers: OK")

import pyepo
print(f"PyEPO: {pyepo.__version__}")

import citylearn
print(f"CityLearn: {citylearn.__version__}")

import pyomo.environ as pyo
print(f"Pyomo: {pyo.__version__}")

import numpy as np
import pandas as pd
print(f"NumPy: {np.__version__}, Pandas: {pd.__version__}")

print("\n✅ Toutes les librairies sont installées correctement.")
```

---

## 4. Guide Complet de Téléchargement des Datasets

### 4.1 Dataset IEEE 33-bus

#### 4.1.1 Description

Le réseau **IEEE 33-bus** est un réseau de distribution radial de référence composé de 33 nœuds, 32 branches, une tension nominale de 12.66 kV, une charge totale de 3.715 MW et 2.300 MVAr. Il est utilisé dans l'article pour valider les contraintes de flux de puissance et les algorithmes de dispatch EMS [1].

**Caractéristiques principales :**

| Paramètre | Valeur |
|---|---|
| Nombre de nœuds | 33 |
| Nombre de branches | 32 |
| Tension nominale | 12.66 kV |
| Charge totale active | 3.715 MW |
| Charge totale réactive | 2.300 MVAr |
| Topologie | Radiale |
| Format | MATPOWER (.m) |

#### 4.1.2 Téléchargement depuis MATPOWER GitHub

```bash
# Téléchargement direct du fichier MATPOWER
wget -O case33bw.m \
  "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case33bw.m"

# Vérification
head -20 case33bw.m
```

#### 4.1.3 Lecture et Conversion en Python

```python
# parse_ieee33.py
import numpy as np
import pandas as pd
import re

def parse_matpower_case(filepath):
    """
    Parse un fichier MATPOWER .m et retourne les données bus/branch/gen.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    def extract_matrix(name, content):
        pattern = rf'mpc\.{name}\s*=\s*\[(.*?)\];'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return None
        rows = []
        for line in match.group(1).strip().split('\n'):
            line = line.strip().rstrip(';').strip()
            if line and not line.startswith('%'):
                vals = [float(x) for x in line.split()]
                if vals:
                    rows.append(vals)
        return np.array(rows)
    
    bus_data    = extract_matrix('bus', content)
    branch_data = extract_matrix('branch', content)
    gen_data    = extract_matrix('gen', content)
    
    # Colonnes MATPOWER bus : [bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
    bus_df = pd.DataFrame(bus_data, columns=[
        'bus_i', 'type', 'Pd_MW', 'Qd_MVAr', 'Gs', 'Bs',
        'area', 'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin'
    ])
    
    # Colonnes MATPOWER branch : [fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax]
    branch_df = pd.DataFrame(branch_data, columns=[
        'fbus', 'tbus', 'r_pu', 'x_pu', 'b_pu',
        'rateA_MVA', 'rateB_MVA', 'rateC_MVA',
        'ratio', 'angle', 'status', 'angmin', 'angmax'
    ])
    
    return bus_df, branch_df, gen_data

# Utilisation
bus_df, branch_df, gen_data = parse_matpower_case('case33bw.m')
print(f"Nœuds : {len(bus_df)}, Branches : {len(branch_df)}")
print(f"Charge totale : {bus_df['Pd_MW'].sum():.3f} MW")
print(bus_df[['bus_i', 'Pd_MW', 'Qd_MVAr']].head(10))

# Sauvegarde CSV
bus_df.to_csv('ieee33_bus.csv', index=False)
branch_df.to_csv('ieee33_branch.csv', index=False)
print("✅ IEEE 33-bus sauvegardé en CSV.")
```

---

### 4.2 Dataset IEEE 69-bus

#### 4.2.1 Description

Le réseau **IEEE 69-bus** est un réseau de distribution radial plus grand, utilisé pour tester la scalabilité des algorithmes EMS. Il comporte 69 nœuds, 68 branches, une tension nominale de 12.66 kV, une charge totale de 3.802 MW et 2.695 MVAr [1].

| Paramètre | Valeur |
|---|---|
| Nombre de nœuds | 69 |
| Nombre de branches | 68 |
| Tension nominale | 12.66 kV |
| Charge totale active | 3.802 MW |
| Charge totale réactive | 2.695 MVAr |
| Topologie | Radiale |

#### 4.2.2 Téléchargement

```bash
# Option 1 : MATPOWER GitHub
wget -O case69.m \
  "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case69.m"

# Option 2 : Figshare (données avec profils temporels)
# DOI: 10.6084/m9.figshare.XXXXXXX (voir publication de référence)
# wget -O ieee69_data.zip "https://figshare.com/ndownloader/articles/XXXXXXX/1"

# Option 3 : Pandapower (conversion automatique)
pip install pandapower
python -c "
import pandapower.networks as pn
net = pn.case69()
print(net)
net.bus.to_csv('ieee69_bus.csv')
net.line.to_csv('ieee69_line.csv')
print('✅ IEEE 69-bus chargé via pandapower.')
"
```

#### 4.2.3 Lecture avec pandapower

```python
# load_ieee69.py
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np

# Chargement du réseau
net = pn.case69()

print("=== IEEE 69-bus Network ===")
print(f"Nœuds (bus) : {len(net.bus)}")
print(f"Lignes (line) : {len(net.line)}")
print(f"Charge totale : {net.load['p_mw'].sum():.3f} MW")
print(f"Tension nominale : {net.bus['vn_kv'].iloc:.2f} kV")

# Extraction de la matrice d'admittance (pour DC power flow)
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pd2ppc import _pd2ppc
ppc, ppci = _pd2ppc(net)
Ybus, Yf, Yt = makeYbus(ppci['baseMVA'], ppci['bus'], ppci['branch'])

# Sauvegarde
net.bus.to_csv('ieee69_bus.csv')
net.line.to_csv('ieee69_line.csv')
net.load.to_csv('ieee69_load.csv')
print("✅ IEEE 69-bus sauvegardé.")
```

---

### 4.3 CityLearn v3

#### 4.3.1 Description

**CityLearn v3** est un environnement Gymnasium open-source pour la coordination énergétique de bâtiments, la réponse à la demande et l'apprentissage par renforcement [2]. La version 3.0.2 (septembre 2026) intègre :

- Bâtiments résidentiels et commerciaux avec profils de charge réels
- Systèmes PV avec production horaire
- Batteries (BESS) avec modèles de dégradation
- Véhicules électriques (EVs) avec contraintes de départ
- Charges flexibles (thermiques, électroménagers)
- Limites électriques de réseau (contraintes de phase)
- Partage d'énergie local entre bâtiments
- KPIs standardisés : coût total, énergie importée, violations de service

Le dataset principal `citylearn_challenge_2022_phase_all_plus_evs` contient des données horaires sur une année complète pour un quartier de bâtiments [2].

#### 4.3.2 Installation

```bash
# Installation de CityLearn v3
python -m pip install citylearn

# Vérification de la version
python -c "import citylearn; print(f'CityLearn version: {citylearn.__version__}')"
# Attendu : CityLearn version: 3.0.2
```

#### 4.3.3 Code Python pour Charger le Dataset

```python
# load_citylearn.py
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.baseline import BusinessAsUsualAgent
import pandas as pd
import numpy as np

# ─── 1. Chargement de l'environnement ────────────────────────────────────────
env = CityLearnEnv(
    "citylearn_challenge_2022_phase_all_plus_evs",
    episode_time_steps=8760,   # 1 an complet (heures)
    interface="flat",
    render_mode="none",
)

print("=== CityLearn v3 Environment ===")
print(f"Nombre de bâtiments : {len(env.buildings)}")
print(f"Pas de temps par épisode : {env.time_steps}")
print(f"Espace d'observation : {env.observation_space}")
print(f"Espace d'action : {env.action_space}")

# ─── 2. Extraction des profils de charge et PV ───────────────────────────────
observations, info = env.reset(seed=42)

# Collecte des données sur un épisode complet
load_profiles = []
pv_profiles = []
soc_profiles = []
timestamps = []

agent = BusinessAsUsualAgent(env)
terminated = truncated = False
step = 0

while not (terminated or truncated):
    actions = agent.predict(observations)
    observations, rewards, terminated, truncated, info = env.step(actions)
    
    # Extraction des données par bâtiment
    for i, building in enumerate(env.buildings):
        if step == 0:
            load_profiles.append([])
            pv_profiles.append([])
            soc_profiles.append([])
        
        load_profiles[i].append(building.net_electricity_consumption)
        if building.pv is not None:
            pv_profiles[i].append(building.pv.electricity_generation)
        if building.electrical_storage is not None:
            soc_profiles[i].append(building.electrical_storage.soc[-1])
    
    step += 1

# ─── 3. Conversion en DataFrames ─────────────────────────────────────────────
time_index = pd.date_range('2022-01-01', periods=step, freq='h')

load_df = pd.DataFrame(
    np.array(load_profiles).T,
    index=time_index,
    columns=[f'building_{i+1}_load_kWh' for i in range(len(env.buildings))]
)

pv_df = pd.DataFrame(
    np.array([p for p in pv_profiles if len(p) == step]).T,
    index=time_index,
    columns=[f'building_{i+1}_pv_kWh' for i in range(len(pv_profiles))]
)

print(f"\nProfils de charge : {load_df.shape}")
print(f"Profils PV : {pv_df.shape}")
print(load_df.describe())

# ─── 4. Évaluation des KPIs ──────────────────────────────────────────────────
kpis = env.evaluate_v2()
selected_kpis = [
    "district_cost_total_control_eur",
    "district_energy_grid_total_import_control_kwh",
    "district_ev_performance_departure_min_acceptable_feasible_ratio",
    "district_electrical_service_phase_violations_energy_total_kwh",
]
kpi_results = kpis.loc[
    (kpis["level"] == "district") & kpis["cost_function"].isin(selected_kpis),
    ["cost_function", "value"],
]
print("\n=== KPIs District ===")
print(kpi_results.to_string(index=False))

# ─── 5. Sauvegarde ───────────────────────────────────────────────────────────
load_df.to_csv('citylearn_load_profiles.csv')
pv_df.to_csv('citylearn_pv_profiles.csv')
kpi_results.to_csv('citylearn_kpis_baseline.csv', index=False)
print("\n✅ Données CityLearn sauvegardées.")
env.close()
```

#### 4.3.4 Extraction des Prix de l'Énergie

```python
# extract_citylearn_prices.py
from citylearn.citylearn import CityLearnEnv
import pandas as pd

env = CityLearnEnv(
    "citylearn_challenge_2022_phase_all_plus_evs",
    episode_time_steps=8760,
)
env.reset(seed=42)

# Extraction des prix depuis le premier bâtiment
building = env.buildings
pricing_data = building.pricing

if pricing_data is not None:
    prices_df = pd.DataFrame({
        'electricity_pricing': pricing_data.electricity_pricing,
        'electricity_pricing_predicted_6h': pricing_data.electricity_pricing_predicted_6h,
        'electricity_pricing_predicted_12h': pricing_data.electricity_pricing_predicted_12h,
        'electricity_pricing_predicted_24h': pricing_data.electricity_pricing_predicted_24h,
    })
    prices_df.index = pd.date_range('2022-01-01', periods=len(prices_df), freq='h')
    prices_df.to_csv('citylearn_prices.csv')
    print(f"Prix extraits : {prices_df.shape}")
    print(prices_df.describe())

env.close()
```

---

### 4.4 Profils PV/Charge/Prix

#### 4.4.1 NREL — National Renewable Energy Laboratory

**NREL NSRDB (National Solar Radiation Database)** fournit des données météorologiques et d'irradiance solaire horaires pour les États-Unis et le monde.

```bash
# Installation de l'API NREL
pip install nrel-pysam requests

# Téléchargement via l'API NSRDB (nécessite une clé API gratuite)
# Inscription : https://developer.nrel.gov/signup/
```

```python
# download_nrel_pv.py
import requests
import pandas as pd
import io

# Configuration API NREL NSRDB
API_KEY = "VOTRE_CLE_API_NREL"  # https://developer.nrel.gov/signup/
LATITUDE = 37.7749   # San Francisco (exemple)
LONGITUDE = -122.4194
YEAR = 2022

url = (
    f"https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"
    f"?wkt=POINT({LONGITUDE}+{LATITUDE})"
    f"&names={YEAR}"
    f"&leap_day=false"
    f"&interval=60"
    f"&utc=false"
    f"&full_name=Researcher"
    f"&email=researcher@university.edu"
    f"&affiliation=University"
    f"&mailing_list=false"
    f"&reason=research"
    f"&api_key={API_KEY}"
    f"&attributes=ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle"
)

response = requests.get(url)
if response.status_code == 200:
    # Les 2 premières lignes sont des métadonnées
    df = pd.read_csv(io.StringIO(response.text), skiprows=2)
    df.to_csv('nrel_solar_data.csv', index=False)
    print(f"✅ NREL NSRDB téléchargé : {df.shape}")
    print(df[['GHI', 'DNI', 'DHI', 'Temperature']].describe())
else:
    print(f"Erreur : {response.status_code}")

# Calcul de la production PV (modèle simplifié)
def pv_production(ghi, panel_area=10, efficiency=0.20, pr=0.85):
    """Production PV en kWh."""
    return ghi * panel_area * efficiency * pr / 1000

df['pv_production_kWh'] = pv_production(df['GHI'])
df.to_csv('nrel_pv_production.csv', index=False)
```

#### 4.4.2 ENTSO-E Transparency Platform

**ENTSO-E** fournit des données de prix spot, de charge et de production renouvelable pour l'Europe.

```bash
# Installation de la librairie entsoe-py
pip install entsoe-py
```

```python
# download_entsoe.py
from entsoe import EntsoePandasClient
import pandas as pd

# Clé API ENTSO-E (inscription gratuite : https://transparency.entsoe.eu/)
client = EntsoePandasClient(api_key='VOTRE_CLE_API_ENTSOE')

start = pd.Timestamp('2022-01-01', tz='Europe/Paris')
end   = pd.Timestamp('2022-12-31', tz='Europe/Paris')
country_code = 'FR'  # France

# ─── Prix spot day-ahead ──────────────────────────────────────────────────────
prices = client.query_day_ahead_prices(country_code, start=start, end=end)
prices.name = 'price_EUR_MWh'
prices.to_csv('entsoe_day_ahead_prices_FR_2022.csv', header=True)
print(f"Prix ENTSO-E : {len(prices)} heures")
print(prices.describe())

# ─── Charge totale ────────────────────────────────────────────────────────────
load = client.query_load(country_code, start=start, end=end)
load.to_csv('entsoe_load_FR_2022.csv')
print(f"Charge ENTSO-E : {load.shape}")

# ─── Production solaire ───────────────────────────────────────────────────────
solar = client.query_generation(country_code, start=start, end=end,
                                 psr_type='B16')  # B16 = Solar
solar.to_csv('entsoe_solar_FR_2022.csv')
print(f"Production solaire ENTSO-E : {solar.shape}")
```

#### 4.4.3 Open Power Systems Data (OPSD)

**OPSD** fournit des données open-source de charge, production renouvelable et prix pour l'Europe.

```bash
# Téléchargement direct des fichiers CSV OPSD
# Charge et production renouvelable (résolution horaire)
wget -O opsd_time_series.zip \
  "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"

# Prix spot
wget -O opsd_prices.csv \
  "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
```

```python
# load_opsd.py
import pandas as pd

# Chargement du fichier OPSD (résolution horaire)
url = ("https://data.open-power-system-data.org/time_series/latest/"
       "time_series_60min_singleindex.csv")

print("Téléchargement OPSD (peut prendre quelques minutes)...")
df = pd.read_csv(url, index_col=0, parse_dates=True, low_memory=False)

# Sélection des colonnes pertinentes pour la France
cols_fr = [c for c in df.columns if c.startswith('FR_')]
df_fr = df[cols_fr].dropna(how='all')

print(f"Colonnes France disponibles : {cols_fr}")
print(f"Période : {df_fr.index.min()} → {df_fr.index.max()}")

# Extraction 2022
df_fr_2022 = df_fr['2022']
df_fr_2022.to_csv('opsd_france_2022.csv')
print(f"✅ OPSD France 2022 : {df_fr_2022.shape}")
```

---

### 4.5 Script Python Complet pour Télécharger et Préparer Tous les Datasets

```python
#!/usr/bin/env python3
"""
download_all_datasets.py
Script complet pour télécharger et préparer tous les datasets
nécessaires à l'article Decision-Focused Probabilistic Forecasting
for Microgrid EMS.

Usage : python download_all_datasets.py
"""

import os
import sys
import subprocess
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

(DATA_DIR / "ieee33").mkdir(exist_ok=True)
(DATA_DIR / "ieee69").mkdir(exist_ok=True)
(DATA_DIR / "citylearn").mkdir(exist_ok=True)
(DATA_DIR / "pv_profiles").mkdir(exist_ok=True)
(DATA_DIR / "prices").mkdir(exist_ok=True)

print("=" * 60)
print("  TÉLÉCHARGEMENT DES DATASETS — EMS Decision-Focused")
print("=" * 60)

# ─── 1. IEEE 33-bus ───────────────────────────────────────────────────────────
print("\n[1/5] Téléchargement IEEE 33-bus (MATPOWER)...")
url_33 = ("https://raw.githubusercontent.com/MATPOWER/matpower/"
           "master/data/case33bw.m")
r = requests.get(url_33, timeout=30)
if r.status_code == 200:
    with open(DATA_DIR / "ieee33" / "case33bw.m", 'w') as f:
        f.write(r.text)
    print("  ✅ case33bw.m téléchargé.")
else:
    print(f"  ⚠️  Erreur HTTP {r.status_code} pour IEEE 33-bus.")

# ─── 2. IEEE 69-bus via pandapower ───────────────────────────────────────────
print("\n[2/5] Chargement IEEE 69-bus (pandapower)...")
try:
    import pandapower.networks as pn
    net69 = pn.case69()
    net69.bus.to_csv(DATA_DIR / "ieee69" / "bus.csv")
    net69.line.to_csv(DATA_DIR / "ieee69" / "line.csv")
    net69.load.to_csv(DATA_DIR / "ieee69" / "load.csv")
    print(f"  ✅ IEEE 69-bus : {len(net69.bus)} nœuds, {len(net69.line)} lignes.")
except ImportError:
    print("  ⚠️  pandapower non installé. Exécuter : pip install pandapower")

# ─── 3. CityLearn v3 ─────────────────────────────────────────────────────────
print("\n[3/5] Chargement CityLearn v3...")
try:
    from citylearn.citylearn import CityLearnEnv
    from citylearn.agents.baseline import BusinessAsUsualAgent
    
    env = CityLearnEnv(
        "citylearn_challenge_2022_phase_all_plus_evs",
        episode_time_steps=8760,
        interface="flat",
        render_mode="none",
    )
    
    agent = BusinessAsUsualAgent(env)
    observations, _ = env.reset(seed=42)
    terminated = truncated = False
    
    all_obs = []
    while not (terminated or truncated):
        actions = agent.predict(observations)
        observations, rewards, terminated, truncated, info = env.step(actions)
        all_obs.append(observations)
    
    # Sauvegarde des observations
    obs_df = pd.DataFrame(all_obs)
    obs_df.to_csv(DATA_DIR / "citylearn" / "observations_bau.csv", index=False)
    
    # KPIs
    kpis = env.evaluate_v2()
    kpis.to_csv(DATA_DIR / "citylearn" / "kpis_bau.csv", index=False)
    
    print(f"  ✅ CityLearn v3 : {len(env.buildings)} bâtiments, "
          f"{env.time_steps} pas de temps.")
    env.close()
    
except ImportError:
    print("  ⚠️  citylearn non installé. Exécuter : pip install citylearn")
except Exception as e:
    print(f"  ⚠️  Erreur CityLearn : {e}")

# ─── 4. Profils PV synthétiques (si NREL non disponible) ─────────────────────
print("\n[4/5] Génération de profils PV/charge synthétiques...")

def generate_pv_profile(n_days=365, peak_kw=50.0, seed=42):
    """Génère un profil PV synthétique horaire."""
    np.random.seed(seed)
    hours = np.arange(n_days * 24)
    day_of_year = hours // 24
    hour_of_day = hours % 24
    
    # Irradiance solaire (modèle sinusoïdal + saisonnalité)
    solar_angle = np.pi * (hour_of_day - 6) / 12
    daylight = (solar_angle > 0) & (solar_angle < np.pi)
    seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    ghi = np.where(daylight, np.sin(solar_angle) * seasonal, 0)
    ghi += np.random.normal(0, 0.05, len(hours)).clip(-0.1, 0.1)
    ghi = np.clip(ghi, 0, 1)
    
    return pd.Series(
        ghi * peak_kw,
        index=pd.date_range('2022-01-01', periods=len(hours), freq='h'),
        name='pv_kW'
    )

def generate_load_profile(n_days=365, peak_kw=30.0, seed=42):
    """Génère un profil de charge résidentielle synthétique."""
    np.random.seed(seed)
    hours = np.arange(n_days * 24)
    hour_of_day = hours % 24
    day_of_year = hours // 24
    
    # Profil journalier typique (double pic matin/soir)
    morning_peak = np.exp(-0.5 * ((hour_of_day - 8) / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hour_of_day - 19) / 2.0) ** 2)
    base_load = 0.3
    
    load = base_load + 0.4 * morning_peak + 0.6 * evening_peak
    
    # Saisonnalité (plus de charge en hiver)
    seasonal = 1.0 + 0.2 * np.cos(2 * np.pi * day_of_year / 365)
    load *= seasonal
    
    # Bruit
    load += np.random.normal(0, 0.05, len(hours))
    load = np.clip(load, 0.1, 1.5)
    
    return pd.Series(
        load * peak_kw,
        index=pd.date_range('2022-01-01', periods=len(hours), freq='h'),
        name='load_kW'
    )

def generate_price_profile(n_days=365, seed=42):
    """Génère un profil de prix spot synthétique (€/MWh)."""
    np.random.seed(seed)
    hours = np.arange(n_days * 24)
    hour_of_day = hours % 24
    
    # Prix de base avec pic en soirée
    base_price = 50.0
    peak = 30.0 * np.exp(-0.5 * ((hour_of_day - 19) / 2.0) ** 2)
    valley = -15.0 * np.exp(-0.5 * ((hour_of_day - 3) / 2.0) ** 2)
    
    price = base_price + peak + valley
    price += np.random.normal(0, 5, len(hours))
    price = np.clip(price, 10, 200)
    
    return pd.Series(
        price,
        index=pd.date_range('2022-01-01', periods=len(hours), freq='h'),
        name='price_EUR_MWh'
    )

# Génération pour plusieurs bâtiments/nœuds
n_buildings = 5
profiles = {}
for i in range(n_buildings):
    profiles[f'pv_{i+1}'] = generate_pv_profile(seed=42+i)
    profiles[f'load_{i+1}'] = generate_load_profile(seed=100+i)

profiles['price'] = generate_price_profile(seed=0)

profiles_df = pd.DataFrame(profiles)
profiles_df.to_csv(DATA_DIR / "pv_profiles" / "synthetic_profiles_2022.csv")
print(f"  ✅ Profils synthétiques générés : {profiles_df.shape}")

# ─── 5. Résumé et vérification ────────────────────────────────────────────────
print("\n[5/5] Vérification des fichiers téléchargés...")
print("\n" + "=" * 60)
print("  RÉSUMÉ DES DATASETS")
print("=" * 60)

datasets = {
    "IEEE 33-bus": DATA_DIR / "ieee33" / "case33bw.m",
    "IEEE 69-bus (bus)": DATA_DIR / "ieee69" / "bus.csv",
    "IEEE 69-bus (line)": DATA_DIR / "ieee69" / "line.csv",
    "CityLearn observations": DATA_DIR / "citylearn" / "observations_bau.csv",
    "CityLearn KPIs": DATA_DIR / "citylearn" / "kpis_bau.csv",
    "Profils synthétiques": DATA_DIR / "pv_profiles" / "synthetic_profiles_2022.csv",
}

for name, path in datasets.items():
    status = "✅" if path.exists() else "❌"
    size = f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else "N/A"
    print(f"  {status} {name:<30} {size}")

print("\n✅ Téléchargement terminé. Données dans ./data/")
```

---

## 5. Plan d'Apprentissage Recommandé

### 5.1 Ressources par Outil Mathématique

#### 5.1.1 Algèbre Linéaire

| Ressource | Type | Niveau | Lien |
|---|---|---|---|
| *Linear Algebra* — Gilbert Strang (MIT) | Livre + cours vidéo | Débutant–Intermédiaire | https://ocw.mit.edu/18-06 |
| *Matrix Computations* — Golub & Van Loan | Livre | Avancé | ISBN 978-1421407944 |
| 3Blue1Brown — Essence of Linear Algebra | Vidéos YouTube | Débutant | https://youtube.com/3blue1brown |
| NumPy Linear Algebra Tutorial | Tutoriel | Pratique | https://numpy.org/doc/stable/reference/routines.linalg.html |

#### 5.1.2 Optimisation Convexe

| Ressource | Type | Niveau | Lien |
|---|---|---|---|
| *Convex Optimization* — Boyd & Vandenberghe | Livre (gratuit en ligne) | Intermédiaire–Avancé | https://web.stanford.edu/~boyd/cvxbook/ |
| Stanford EE364a — Convex Optimization | Cours vidéo | Intermédiaire | https://www.youtube.com/playlist?list=PL3940DD956CDF0622 |
| CVXPY Documentation & Examples | Tutoriel | Pratique | https://www.cvxpy.org/examples/ |
| *Introduction to Linear Optimization* — Bertsimas & Tsitsiklis | Livre | Intermédiaire | ISBN 978-1886529199 |

#### 5.1.3 Optimisation Stochastique et Robuste

| Ressource | Type | Niveau | Lien |
|---|---|---|---|
| *Introduction to Stochastic Programming* — Birge & Louveaux | Livre | Avancé | ISBN 978-1461402367 |
| *Robust Optimization* — Ben-Tal, El Ghaoui & Nemirovski | Livre | Avancé | ISBN 978-0691143682 |
| *Data-Driven Optimization* — Esfahani & Kuhn | Article survey | Avancé | https://arxiv.org/abs/1811.07155 |
| Pyomo Stochastic Programming Tutorial | Tutoriel | Pratique | https://pyomo.readthedocs.io/en/stable/pysp.html |

#### 5.1.4 Probabilités et Statistiques

| Ressource | Type | Niveau | Lien |
|---|---|---|---|
| *Probabilistic Forecasting and Bayesian Data Assimilation* — Reich & Cotter | Livre | Intermédiaire | ISBN 978-1107663916 |
| *Energy Forecasting for Planning and Management* — Weron | Livre | Intermédiaire | ISBN 978-0470014011 |
| properscoring Python library | Librairie | Pratique | https://github.com/TheClimateCorporation/properscoring |
| *Forecasting: Principles and Practice* — Hyndman & Athanasopoulos | Livre (gratuit) | Débutant–Intermédiaire | https://otexts.com/fpp3/ |

#### 5.1.5 Apprentissage Automatique et Deep Learning

| Ressource | Type | Niveau | Lien |
|---|---|---|---|
| *Deep Learning* — Goodfellow, Bengio & Courville | Livre (gratuit) | Intermédiaire–Avancé | https://www.deeplearningbook.org/ |
| PyTorch Tutorials officiels | Tutoriel | Pratique | https://pytorch.org/tutorials/ |
| *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* — Lim et al. | Article | Avancé | https://arxiv.org/abs/1912.09363 |
| pytorch-forecasting TFT Tutorial | Tutoriel | Pratique | https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html |

#### 5.1.6 Decision-Focused Learning (SPO+, cvxpylayers)

| Ressource | Type | Niveau | Lien |
|---|---|---|---|
| *Smart "Predict, then Optimize"* — Elmachtoub & Grigas | Article fondateur | Avancé | https://arxiv.org/abs/1710.08005 |
| *Differentiable Convex Optimization Layers* — Agrawal et al. | Article | Avancé | https://arxiv.org/abs/1910.12430 |
| PyEPO Documentation & Tutorials | Tutoriel | Pratique | https://khalil-research.github.io/PyEPO/ |
| cvxpylayers GitHub Examples | Tutoriel | Pratique | https://github.com/cvxgrp/cvxpylayers/tree/master/examples |
| *Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities* — Mandi et al. | Survey | Avancé | https://arxiv.org/abs/2307.13565 |

---

### 5.2 Ordre d'Apprentissage Suggéré sur 4 Semaines

#### Semaine 1 — Fondations Mathématiques

| Jour | Thème | Ressource | Durée |
|---|---|---|---|
| Lun | Algèbre linéaire : vecteurs, matrices, normes | Strang MIT OCW, Lectures 1–5 | 3h |
| Mar | Algèbre linéaire : SVD, Cholesky, PTDF | Golub & Van Loan Ch. 2–3 | 3h |
| Mer | Optimisation convexe : fonctions convexes, KKT | Boyd & Vandenberghe Ch. 1–5 | 4h |
| Jeu | LP, QP, MILP : formulations et solveurs | CVXPY examples + HiGHS | 3h |
| Ven | Probabilités : distributions, quantiles, CRPS | Hyndman FPP3 Ch. 9–11 | 3h |
| Sam | **TP** : Résoudre un LP EMS avec CVXPY | Code Section 1.2.4 | 2h |
| Dim | Révision + exercices | — | 2h |

#### Semaine 2 — Optimisation Stochastique et Robuste

| Jour | Thème | Ressource | Durée |
|---|---|---|---|
| Lun | Programmation stochastique bi-niveau | Birge & Louveaux Ch. 1–3 | 3h |
| Mar | Ensembles d'incertitude, DRO | Ben-Tal et al. Ch. 1–2 | 3h |
| Mer | TSRO : formulation min-max | Article [1], Section 2.5 | 3h |
| Jeu | Algorithme C&CG : implémentation | Pyomo + HiGHS | 4h |
| Ven | Décomposition de Benders | Birge & Louveaux Ch. 5 | 3h |
| Sam | **TP** : TSRO sur IEEE 33-bus | Code Section 2.5 + 4.1 | 3h |
| Dim | Révision + exercices | — | 2h |

#### Semaine 3 — Deep Learning et Prévision Probabiliste

| Jour | Thème | Ressource | Durée |
|---|---|---|---|
| Lun | Réseaux de neurones, backpropagation | Goodfellow Ch. 6–8 | 3h |
| Mar | PyTorch : tenseurs, autograd, modules | PyTorch Tutorials | 3h |
| Mer | TFT : architecture, attention, GRN | Lim et al. 2021 + pytorch-forecasting | 4h |
| Jeu | Entraînement TFT sur profils PV/charge | pytorch-forecasting tutorial | 4h |
| Ven | Évaluation probabiliste : CRPS, calibration | properscoring library | 2h |
| Sam | **TP** : TFT sur données CityLearn | Code Section 4.3 + 2.3 | 3h |
| Dim | Révision + exercices | — | 2h |

#### Semaine 4 — Decision-Focused Learning et Intégration

| Jour | Thème | Ressource | Durée |
|---|---|---|---|
| Lun | SPO+ : formulation, gradient, implémentation | Elmachtoub & Grigas 2022 | 3h |
| Mar | PyEPO : SPO+ avec PyTorch | PyEPO documentation | 3h |
| Mer | cvxpylayers : forward/backward pass | Agrawal et al. 2019 + examples | 4h |
| Jeu | Intégration TFT + cvxpylayers + SPO+ | Code Section 2.1–2.2 | 4h |
| Ven | Benchmark : predict-then-optimize vs SPO+ | Article [1], Section 2 | 3h |
| Sam | **TP Final** : Pipeline complet sur IEEE 33-bus + CityLearn | Tous les codes | 4h |
| Dim | Rédaction notes + documentation | — | 2h |

---

### 5.3 Tableau Récapitulatif des Ressources

| Outil | Livre de référence | Cours en ligne | Librairie Python |
|---|---|---|---|
| Algèbre linéaire | Strang (MIT OCW) | MIT 18.06 | `numpy.linalg` |
| Optimisation convexe | Boyd & Vandenberghe | Stanford EE364a | `cvxpy` |
| Optimisation stochastique | Birge & Louveaux | — | `pyomo`, `cvxpy` |
| Optimisation robuste | Ben-Tal et al. | — | `pyomo`, `cvxpy` |
| Probabilités/Statistiques | Hyndman FPP3 | — | `scipy.stats`, `properscoring` |
| Deep learning | Goodfellow et al. | fast.ai | `torch`, `torch.nn` |
| TFT | Lim et al. (2021) | pytorch-forecasting docs | `pytorch-forecasting` |
| SPO+ | Elmachtoub & Grigas (2022) | PyEPO docs | `pyepo` |
| cvxpylayers | Agrawal et al. (2019) | cvxpylayers examples | `cvxpylayers` |
| EMS/Microgrid | Article [1] | CityLearn docs | `citylearn`, `pandapower` |

---

### 5.4 Checklist de Validation des Prérequis

Avant de commencer l'implémentation de l'article, vérifier que vous êtes capable de :

**Algèbre linéaire :**
- [ ] Calculer la SVD d'une matrice et interpréter les valeurs singulières
- [ ] Effectuer une décomposition de Cholesky et générer des échantillons gaussiens
- [ ] Construire et utiliser la matrice PTDF pour un réseau de distribution

**Optimisation convexe :**
- [ ] Formuler et résoudre un LP de dispatch EMS avec CVXPY
- [ ] Écrire les conditions KKT d'un QP et vérifier la solution
- [ ] Modéliser un MILP avec variables binaires d'engagement

**Optimisation stochastique/robuste :**
- [ ] Implémenter une SAA (Sample Average Approximation) pour un problème bi-niveau
- [ ] Construire un ensemble d'incertitude budget et résoudre le problème robuste
- [ ] Coder l'algorithme C&CG pour un TSRO simple

**Probabilités/Statistiques :**
- [ ] Calculer le CRPS pour une prévision probabiliste
- [ ] Tracer un diagramme de calibration (reliability diagram)
- [ ] Entraîner un modèle de régression par quantile

**Deep learning :**
- [ ] Entraîner un TFT sur des données de séries temporelles avec pytorch-forecasting
- [ ] Interpréter les poids d'attention et les importances de variables du TFT
- [ ] Calculer et visualiser les intervalles de prédiction

**Decision-focused learning :**
- [ ] Implémenter la loss SPO+ avec PyEPO sur un problème de dispatch
- [ ] Créer une couche d'optimisation différentiable avec cvxpylayers
- [ ] Comparer le regret décisionnel de SPO+ vs MSE sur un benchmark EMS

---

*Document rédigé le 15 octobre 2026. Basé sur les insights de l'article de recherche [1] et la documentation officielle de CityLearn v3 [2].*

## References

[1]“Article 1 Deep Insights — Decision-Focused Forecasting for Microgrid EMS.”

[2]“CityLearn v3 GitHub Repository.” [Online]. Available: https://github.com/citylearn-project/CityLearn