# PLAN D'ATTAQUE DÉTAILLÉ — ARTICLE 1
## Decision-Focused Probabilistic Forecasting for Microgrid Energy Management Systems

**Document de planification stratégique**  
**Date:** 30 mars 2026  
**Statut:** Plan détaillé pour soumission journal Q1

---

## 1. TITRE DÉFINITIF ET VARIANTES

### Option 1 (Recommandée) :
**"End-to-End Decision-Focused Probabilistic Forecasting for Robust Microgrid Energy Management"**

*Justification :* Met en avant l'approche end-to-end, le caractère décisionnel, et la robustesse. Cible directement les communautés EMS, forecasting, et optimization.

### Option 2 :
**"Integrating Probabilistic Forecasting and Two-Stage Robust Optimization via Decision-Focused Learning for Microgrid Scheduling"**

*Justification :* Plus technique, explicite sur la méthodologie (TSRO + DFL). Convient aux journaux très techniques (IEEE Transactions).

### Option 3 :
**"Bridging the Forecast-to-Decision Gap: A Decision-Aware Framework for Microgrid Energy Management Under Uncertainty"**

*Justification :* Accent sur le gap identifié dans la littérature. Plus conceptuel, adapté aux journaux interdisciplinaires (Applied Energy, Energy).

---

## 2. ABSTRACT COMPLET (180 mots)

**Version proposée :**

> Microgrid energy management systems (EMS) traditionally rely on a predict-then-optimize pipeline, where forecasting models are trained using statistical metrics (e.g., RMSE, CRPS) independently of downstream operational decisions. This separation often leads to suboptimal schedules and increased operational costs, as forecast errors that matter most for decision-making are not explicitly penalized during training. We propose an **end-to-end decision-focused framework** that jointly trains a probabilistic forecaster with a two-stage robust optimization solver for day-ahead microgrid scheduling. Our approach integrates a differentiable Smart Predict-and-Optimize (SPO) surrogate loss that directly minimizes operational regret by backpropagating decision costs through the optimization layer. The probabilistic forecaster generates calibrated prediction intervals for solar generation, load demand, and electricity prices, which parameterize a tractable two-stage robust optimization problem with recourse. Extensive experiments on IEEE 33-bus and 69-bus test feeders, as well as CityLearn building clusters, demonstrate that our method reduces realized operational costs by **12–18%** compared to predict-then-optimize baselines and achieves superior robustness under forecast uncertainty. We release open-source code and benchmark scenarios to facilitate reproducible research in decision-focused EMS.

**Éléments clés couverts :**
- ✅ Contexte (predict-then-optimize limitation)
- ✅ Problème (forecast-decision mismatch)
- ✅ Méthode (end-to-end, SPO loss, TSRO)
- ✅ Résultats attendus (12–18% cost reduction)
- ✅ Impact (open-source benchmark)

---

## 3. INTRODUCTION DÉTAILLÉE

### 3.1 Contexte et motivation (3–4 paragraphes)

#### Paragraphe 1 : Transition énergétique et défis des microgrids
*Contenu :*
- Croissance des énergies renouvelables intermittentes (PV, éolien) dans les microgrids
- Nécessité d'une gestion optimale des ressources énergétiques distribuées (DER) : batteries, générateurs, charges flexibles
- Défis : incertitude des prévisions RES, variabilité de la demande, prix de l'électricité volatils
- Importance économique : réduction des coûts opérationnels, maximisation de l'autoconsommation, participation aux marchés

*Citations attendues :* Études sur l'intégration RES, statistiques de croissance PV/éolien, cas d'usage microgrids (IEEE, CityLearn).

#### Paragraphe 2 : Paradigme predict-then-optimize et ses limites
*Contenu :*
- Description du pipeline traditionnel : (1) entraîner un modèle de prévision avec métriques statistiques (RMSE, MAE, CRPS), (2) utiliser les prévisions comme entrées d'un optimiseur EMS
- Problème fondamental : **metric mismatch** — les erreurs de prévision qui minimisent CRPS ne minimisent pas nécessairement le coût opérationnel
- Exemple illustratif : une erreur de prévision PV à midi (pic de production) a un impact décisionnel différent d'une erreur à minuit
- Conséquence : sous-optimalité des décisions, coûts opérationnels élevés, risques de délestage

*Citations attendues :* Travaux sur predict-then-optimize limitations, études empiriques montrant le mismatch forecast-decision.

#### Paragraphe 3 : Émergence du decision-focused learning
*Contenu :*
- Paradigme alternatif : **decision-focused learning** (DFL) ou **Smart Predict-and-Optimize (SPO)**
- Principe : entraîner le modèle de prévision en minimisant directement le regret décisionnel (coût réalisé vs coût optimal clairvoyant)
- Avantages théoriques : alignement forecast-decision, réduction du regret opérationnel
- Applications émergentes : bidding énergétique, dispatch de batteries, MPC
- Défis ouverts : scalabilité, différentiabilité des solveurs, intégration avec optimisation robuste/stochastique

*Citations attendues :* Papiers fondateurs SPO/SPO+, applications energy systems, solver-in-the-loop methods.

#### Paragraphe 4 : Optimisation robuste et prévision probabiliste
*Contenu :*
- Prévision probabiliste : quantiles, intervalles de prédiction, distributions prédictives
- Optimisation robuste (TSRO, DRO) : garantir la faisabilité sous incertitude, réduire le conservatisme via ensembles d'ambiguïté
- Synergie potentielle : prévisions probabilistes calibrées → ensembles d'incertitude informés → décisions robustes et économiques
- Gap actuel : peu de travaux intègrent end-to-end forecasting probabiliste + TSRO + DFL

*Citations attendues :* Méthodes de prévision probabiliste (quantile regression, conformal prediction), TSRO/DRO pour microgrids.

### 3.2 Problématique précise

**Énoncé formel :**

> Comment concevoir un système EMS pour microgrids qui **intègre de manière end-to-end** la prévision probabiliste et l'optimisation robuste, en entraînant le module de prévision avec un objectif décisionnel qui minimise le regret opérationnel réalisé, tout en garantissant la faisabilité et la robustesse des décisions sous incertitude ?

**Sous-questions :**
1. Comment construire une fonction de perte différentiable qui capture le coût opérationnel d'un problème TSRO ?
2. Comment modéliser efficacement les dépendances multivariées (PV, charge, prix) pour alimenter l'optimiseur robuste ?
3. Comment équilibrer conservatisme (robustesse) et performance économique dans l'entraînement end-to-end ?
4. Comment garantir la scalabilité computationnelle pour des réseaux réalistes (IEEE 33/69-bus) ?

### 3.3 Lacunes identifiées dans la littérature

**Liste détaillée des gaps :**

1. **Metric mismatch non résolu :**
   - La plupart des études EMS utilisent des prévisions entraînées avec CRPS/MAE sans considération du coût décisionnel
   - Peu de travaux quantifient empiriquement l'écart de performance entre forecast-focused et decision-focused training dans le contexte EMS

2. **Intégration limitée SPO + optimisation robuste :**
   - Les méthodes SPO/SPO+ existantes ciblent principalement l'optimisation stochastique ou déterministe
   - L'intégration avec TSRO/DRO reste peu explorée, notamment pour les problèmes bi-niveaux avec recours

3. **Modélisation des dépendances multivariées :**
   - Les approches probabilistes traitent souvent PV, charge, prix de manière indépendante
   - Les méthodes de génération de scénarios joints (copules, GANs) sont coûteuses et peu intégrées dans des pipelines DFL

4. **Scalabilité computationnelle :**
   - Les solveurs différentiables pour MILP/MIQP sont limités en taille
   - Les approximations convexes ou surrogates pour backpropagation manquent de validation sur réseaux réalistes

5. **Absence de benchmarks standardisés :**
   - Pas de protocole d'évaluation unifié comparant predict-then-optimize, SPO, SPO+, DRL sur mêmes données EMS
   - Manque de code open-source et de datasets reproductibles

6. **Garanties théoriques faibles :**
   - Peu de bornes théoriques reliant calibration des prévisions probabilistes (coverage) et regret décisionnel
   - Connexion entre conformal prediction et ensembles d'ambiguïté DRO non formalisée

### 3.4 Contributions principales de l'article

**Liste numérotée et détaillée :**

**Contribution 1 : Framework end-to-end intégré**
- Nous proposons le premier framework qui couple de manière end-to-end :
  - Un module de prévision probabiliste multivariée (quantile regression NN ou conformal prediction)
  - Un solveur TSRO bi-niveau avec recours pour EMS microgrid
  - Une fonction de perte SPO différentiable adaptée au contexte robuste
- Architecture modulaire permettant de remplacer les composants (forecaster, optimizer, loss)

**Contribution 2 : SPO loss différentiable pour TSRO**
- Développement d'une fonction de perte surrogate qui approxime le regret opérationnel d'un problème TSRO
- Méthode de calcul de gradients via :
  - Différentiation implicite du problème d'optimisation convexe (second stage)
  - Surrogate linéaire pour le problème minimax (first stage)
- Validation de la convergence et de la stabilité numérique

**Contribution 3 : Génération de scénarios décision-aware**
- Méthode de génération de scénarios multivariés qui préserve les dépendances critiques pour les décisions EMS
- Approche low-rank copula ou conditional sampling calibrée sur données historiques
- Réduction du nombre de scénarios nécessaires (de 1000+ à 50–100) sans perte de qualité décisionnelle

**Contribution 4 : Analyse théorique et bornes de regret**
- Établissement de bornes théoriques reliant :
  - Calibration des intervalles de prédiction (coverage empirique)
  - Taille des ensembles d'ambiguïté DRO
  - Regret opérationnel espéré
- Connexion formelle entre conformal prediction et DRO moment-based

**Contribution 5 : Évaluation extensive et benchmark open-source**
- Expériences sur 3 environnements :
  - IEEE 33-bus microgrid (réseau radial, contraintes réseau)
  - IEEE 69-bus microgrid (plus grande échelle)
  - CityLearn building cluster (multi-bâtiments, charges flexibles)
- Comparaison systématique avec 6 baselines :
  1. Predict-then-optimize (CRPS-trained)
  2. Predict-then-optimize (MAE-trained)
  3. SPO (stochastic programming)
  4. SPO+ (improved surrogate)
  5. Deep Reinforcement Learning (SAC, PPO)
  6. Model Predictive Control (MPC) avec perfect foresight (upper bound)
- Métriques multiples : CRPS, pinball loss, operational cost, regret, feasibility rate, robustness score
- Release de code Python (PyTorch + Pyomo/CVXPY), datasets, configurations solveur

**Contribution 6 : Insights pratiques pour déploiement**
- Analyse de sensibilité : impact de la taille de l'ensemble d'incertitude, du niveau de conservatisme, de l'horizon de prévision
- Recommandations pour le tuning hyperparamètres (learning rate, batch size, scenario count)
- Étude d'ablation : importance relative de chaque composant (probabilistic forecasting, decision-focused loss, robust optimization)

### 3.5 Structure de l'article

**Plan détaillé :**

1. **Introduction** (comme ci-dessus)
2. **Related Work** (revue de littérature structurée)
3. **Problem Formulation** (formulation mathématique EMS)
4. **Methodology** (architecture end-to-end, SPO loss, algorithme)
5. **Experimental Setup** (datasets, baselines, métriques, implémentation)
6. **Results and Analysis** (résultats quantitatifs, ablation, robustesse)
7. **Discussion** (implications, limitations, perspectives)
8. **Conclusion**
9. **References**
10. **Appendices** (preuves théoriques, détails algorithmes, hyperparamètres)

---

## 4. REVUE DE LITTÉRATURE (RELATED WORK) — SECTIONS DÉTAILLÉES

### 4.1 Prévision pour EMS microgrids

#### 4.1.1 Méthodes de prévision déterministe
*Contenu :*
- Modèles statistiques classiques : ARIMA, SARIMA, exponential smoothing pour charge et prix
- Méthodes machine learning : Random Forest, Gradient Boosting, SVR pour prévision PV
- Deep learning : LSTM, GRU, Transformer pour séries temporelles multivariées
- Limites : prévisions ponctuelles, pas d'information d'incertitude, inadaptées pour optimisation stochastique/robuste

*Citations clés :* Études comparatives forecasting PV/load, benchmarks (GEFCom, Kaggle).

#### 4.1.2 Prévision probabiliste
*Contenu :*
- **Quantile regression :** modèles linéaires, NN avec pinball loss, Gradient Boosting Quantile
  - Avantages : interprétabilité, calibration directe
  - Limites : quantiles indépendants, pas de modélisation de dépendance temporelle/multivariée
- **Conformal prediction :** garanties de coverage distribution-free
  - Méthodes : split conformal, full conformal, adaptive conformal
  - Limites : coverage marginale seulement, peut être conservative
- **Deep learning UQ :** ensembles, Monte Carlo Dropout, Bayesian NN, normalizing flows
  - Avantages : flexibilité, capture de patterns complexes
  - Limites : calibration nécessaire, coût computationnel
- **Génération de scénarios :** GANs, VAEs, copulas pour scénarios multivariés
  - Avantages : préserve dépendances
  - Limites : scalabilité, validation difficile

*Citations clés :* Papiers fondateurs quantile regression, conformal prediction (Vovk et al.), deep UQ (Gal & Ghahramani), scenario generation pour energy.

#### 4.1.3 Limites pour EMS
*Synthèse critique :*
- La plupart des méthodes optimisent des métriques statistiques (CRPS, Winkler score, coverage) sans lien direct avec le coût opérationnel
- Les prévisions probabilistes sont souvent utilisées en post-processing, pas intégrées dans l'entraînement du modèle
- Peu d'études évaluent l'impact de la qualité de prévision probabiliste sur les décisions EMS réelles

### 4.2 Optimisation robuste et stochastique pour EMS

#### 4.2.1 Programmation stochastique
*Contenu :*
- **Two-stage stochastic programming :** décisions first-stage (day-ahead), recours second-stage (real-time)
- Formulation : $\min_{x} c^T x + \mathbb{E}_{\xi \sim P}[Q(x, \xi)]$ où $Q(x, \xi) = \min_{y} q^T y$ s.t. $Ax + By \geq b(\xi)$
- Méthodes de résolution : L-shaped, Benders decomposition, scenario-based approximation
- Applications EMS : unit commitment, dispatch batteries, demand response
- Limites : nécessite distribution $P$ connue ou estimée, sensible à la qualité des scénarios

*Citations clés :* Birge & Louveaux (stochastic programming), applications microgrid scheduling.

#### 4.2.2 Optimisation robuste (TSRO)
*Contenu :*
- **Two-stage robust optimization :** $\min_{x} c^T x + \max_{\xi \in \mathcal{U}} Q(x, \xi)$
- Ensembles d'incertitude $\mathcal{U}$ : box, ellipsoidal, polyhedral, budget-constrained
- Avantages : garanties worst-case, pas besoin de distribution exacte
- Limites : conservatisme, coût computationnel (minimax)
- Méthodes de résolution : column-and-constraint generation, affine decision rules, robust counterpart reformulation
- Applications EMS : robust unit commitment, network-aware dispatch, islanding

*Citations clés :* Ben-Tal & Nemirovski (robust optimization), Bertsimas & Sim (budget uncertainty), applications microgrid.

#### 4.2.3 Distributionally Robust Optimization (DRO)
*Contenu :*
- Compromis entre stochastic et robust : $\min_{x} c^T x + \sup_{P \in \mathcal{D}} \mathbb{E}_{\xi \sim P}[Q(x, \xi)]$
- Ensembles d'ambiguïté $\mathcal{D}$ : moment-based, Wasserstein, $\phi$-divergence
- Avantages : moins conservatif que TSRO, data-driven
- Limites : complexité théorique, tractabilité dépend de $\mathcal{D}$
- Applications EMS : dispatch sous incertitude RES, bidding

*Citations clés :* Mohajerin Esfahani & Kuhn (DRO), Wasserstein DRO, applications energy.

#### 4.2.4 Comparaison et positionnement
*Tableau comparatif :*

| Approche | Garanties | Conservatisme | Tractabilité | Besoin de données |
|----------|-----------|---------------|--------------|-------------------|
| Stochastic Programming | Espérance optimale | Faible | Moyenne (scénarios) | Distribution $P$ |
| TSRO | Worst-case | Élevé | Difficile (minimax) | Ensemble $\mathcal{U}$ |
| DRO | Worst-case sur $\mathcal{D}$ | Moyen | Moyenne | Moments/distance |

*Positionnement de notre approche :*
- Nous utilisons TSRO pour garanties robustesse + prévisions probabilistes pour informer $\mathcal{U}$
- Réduction du conservatisme via calibration des intervalles de prédiction
- Tractabilité via approximations convexes et scenario reduction

### 4.3 Apprentissage orienté décision (SPO, SPO+, couches différentiables)

#### 4.3.1 Smart Predict-and-Optimize (SPO)
*Contenu :*
- **Principe :** entraîner le prédicteur $\pi_\theta$ en minimisant le regret décisionnel
- Regret : $\text{Regret}(\theta) = \mathbb{E}[\text{Cost}(x^*(\pi_\theta(\xi)), \xi) - \text{Cost}(x^*(\xi), \xi)]$
  où $x^*(\cdot)$ est la solution optimale du problème de décision
- Problème : $x^*$ est non-différentiable (solution d'un programme linéaire/quadratique)
- Solution SPO : utiliser une fonction de perte surrogate linéaire
  $L_{\text{SPO}}(\theta) = (c - c^*)^T (x^*(\hat{c}) - x^*(c^*))$
  où $\hat{c} = \pi_\theta(\xi)$ est la prédiction, $c^*$ est le vrai coût

*Citations clés :* Elmachtoub & Grigas (SPO), applications inventory, scheduling.

#### 4.3.2 SPO+ et variantes
*Contenu :*
- **SPO+ :** amélioration de SPO avec surrogate convexe
  $L_{\text{SPO+}}(\theta) = \max_{x \in \mathcal{X}} (2\hat{c} - c^*)^T x - \hat{c}^T x^*(\hat{c})$
- Avantages : convexe en $\hat{c}$, meilleure convergence
- **Task-based loss :** variantes pour problèmes spécifiques (portfolio, bidding)
- **Perturbed optimization :** ajouter du bruit pour rendre $x^*$ différentiable (Berthet et al.)

*Citations clés :* Elmachtoub & Grigas (SPO+), Berthet et al. (perturbed optimization).

#### 4.3.3 Couches d'optimisation différentiables
*Contenu :*
- **OptNet :** intégrer un solveur QP comme couche neuronale, différentiation via KKT
- **cvxpylayers :** différentiation automatique de problèmes convexes (CVXPY)
- **Differentiable MPC :** backpropagation à travers un contrôleur MPC
- Applications energy : dispatch batteries, MPC pour bâtiments, bidding

*Citations clés :* Amos & Kolter (OptNet), Agrawal et al. (cvxpylayers), applications energy MPC.

#### 4.3.4 Applications aux systèmes énergétiques
*Contenu :*
- **Bidding énergétique :** SPO pour optimiser les offres day-ahead
- **Dispatch batteries :** neural optimizers avec decision-focused training
- **MPC bâtiments :** fine-tuning de foundation models avec task-based loss
- **Résultats empiriques :** réductions de coût de 5–15% vs predict-then-optimize

*Citations clés :* Papier arXiv:2512.12755 (end-to-end microgrid), autres applications energy SPO.

### 4.4 Positionnement par rapport à la littérature

#### Tableau comparatif des approches existantes

| Référence | Forecasting | Optimization | Decision-focused | Robustness | Benchmark |
|-----------|-------------|--------------|------------------|------------|-----------|
| [Ref A] | Point (LSTM) | Stochastic | Non | Non | Propriétaire |
| [Ref B] | Quantile | TSRO | Non | Oui | IEEE 33-bus |
| [Ref C] | Ensemble | MPC | Oui (MPC loss) | Non | CityLearn |
| arXiv:2512.12755 | Probabilistic | TSRO | Oui (SPO-like) | Oui | IEEE 33/69 |
| **Notre approche** | **Probabilistic (conformal)** | **TSRO + DRO** | **Oui (SPO+ adapté)** | **Oui** | **IEEE 33/69 + CityLearn** |

#### Différences clés et innovations

1. **Intégration end-to-end complète :**
   - Contrairement à [Ref B] qui utilise des prévisions pré-entraînées, nous entraînons conjointement forecaster et optimizer

2. **SPO loss adapté au contexte robuste :**
   - Extension de SPO+ pour problèmes TSRO bi-niveaux (non traité dans littérature SPO classique)

3. **Génération de scénarios décision-aware :**
   - Méthode de scenario reduction guidée par sensibilité décisionnelle (nouveau)

4. **Analyse théorique forecast-decision :**
   - Bornes de regret reliant calibration et robustesse (contribution théorique originale)

5. **Benchmark reproductible multi-environnements :**
   - Protocole d'évaluation unifié sur 3 testbeds avec code open-source

---

## 5. MÉTHODOLOGIE COMPLÈTE

### 5.1 Formulation du problème EMS

#### 5.1.1 Notations et ensembles

**Indices et ensembles :**
- $t \in \mathcal{T} = \{1, \ldots, T\}$ : périodes temporelles (e.g., heures sur horizon day-ahead, $T=24$)
- $i \in \mathcal{N}$ : nœuds du réseau microgrid
- $g \in \mathcal{G}$ : générateurs dispatchables (diesel, micro-turbines)
- $s \in \mathcal{S}$ : systèmes de stockage (batteries)
- $\xi_t = (\xi_t^{\text{PV}}, \xi_t^{\text{load}}, \xi_t^{\text{price}})$ : vecteur d'incertitude à $t$ (génération PV, demande, prix)

**Variables de décision :**

*First-stage (day-ahead) :*
- $x = \{p_{g,t}, u_{g,t}, p_{s,t}^{\text{ch}}, p_{s,t}^{\text{dch}}, \text{SoC}_{s,t}\}_{t \in \mathcal{T}}$
  - $p_{g,t}$ : puissance générateur $g$ à $t$
  - $u_{g,t} \in \{0,1\}$ : état on/off générateur $g$
  - $p_{s,t}^{\text{ch}}, p_{s,t}^{\text{dch}}$ : charge/décharge batterie $s$
  - $\text{SoC}_{s,t}$ : état de charge batterie $s$

*Second-stage (real-time recours) :*
- $y(\xi) = \{p_{i,t}^{\text{curt}}(\xi), p_{i,t}^{\text{shed}}(\xi), p_{t}^{\text{grid}}(\xi)\}_{t \in \mathcal{T}}$
  - $p_{i,t}^{\text{curt}}(\xi)$ : curtailment PV au nœud $i$
  - $p_{i,t}^{\text{shed}}(\xi)$ : délestage charge au nœud $i$
  - $p_{t}^{\text{grid}}(\xi)$ : import/export réseau principal

#### 5.1.2 Fonction objectif

**Two-stage robust formulation :**

$$
\min_{x \in \mathcal{X}} \left\{ C_1(x) + \max_{\xi \in \mathcal{U}} Q(x, \xi) \right\}
$$

où :

**First-stage cost :**
$$
C_1(x) = \sum_{t \in \mathcal{T}} \left[ \sum_{g \in \mathcal{G}} (c_g p_{g,t} + c_g^{\text{start}} u_{g,t}(1 - u_{g,t-1})) + \sum_{s \in \mathcal{S}} c_s^{\text{deg}} (p_{s,t}^{\text{ch}} + p_{s,t}^{\text{dch}}) \right]
$$

**Second-stage cost (recourse) :**
$$
Q(x, \xi) = \min_{y \in \mathcal{Y}(x, \xi)} \sum_{t \in \mathcal{T}} \left[ c_t^{\text{grid}}(\xi) p_t^{\text{grid}}(\xi) + \sum_{i \in \mathcal{N}} (c^{\text{curt}} p_{i,t}^{\text{curt}}(\xi) + c^{\text{shed}} p_{i,t}^{\text{shed}}(\xi)) \right]
$$

**Ensemble d'incertitude $\mathcal{U}$ :**

Nous utilisons un ensemble d'incertitude **budget-constrained** informé par les prévisions probabilistes :

$$
\mathcal{U} = \left\{ \xi : \xi_t^j \in [\underline{\xi}_t^j, \overline{\xi}_t^j], \; \sum_{t,j} \frac{|\xi_t^j - \hat{\xi}_t^j|}{\overline{\xi}_t^j - \underline{\xi}_t^j} \leq \Gamma \right\}
$$

où :
- $\hat{\xi}_t^j$ : prévision médiane pour variable $j$ à $t$
- $[\underline{\xi}_t^j, \overline{\xi}_t^j]$ : intervalle de prédiction (e.g., 90% PI)
- $\Gamma$ : budget d'incertitude (contrôle le conservatisme)

#### 5.1.3 Contraintes

**First-stage constraints $\mathcal{X}$ :**

1. **Limites générateurs :**
   $$u_{g,t} P_g^{\min} \leq p_{g,t} \leq u_{g,t} P_g^{\max}, \quad \forall g, t$$

2. **Rampe générateurs :**
   $$-R_g^{\text{down}} \leq p_{g,t} - p_{g,t-1} \leq R_g^{\text{up}}, \quad \forall g, t$$

3. **Dynamique batteries :**
   $$\text{SoC}_{s,t} = \text{SoC}_{s,t-1} + \eta_s^{\text{ch}} p_{s,t}^{\text{ch}} \Delta t - \frac{p_{s,t}^{\text{dch}}}{\eta_s^{\text{dch}}} \Delta t, \quad \forall s, t$$

4. **Limites SoC :**
   $$\text{SoC}_s^{\min} \leq \text{SoC}_{s,t} \leq \text{SoC}_s^{\max}, \quad \forall s, t$$

5. **Limites charge/décharge :**
   $$0 \leq p_{s,t}^{\text{ch}} \leq P_s^{\text{ch,max}}, \quad 0 \leq p_{s,t}^{\text{dch}} \leq P_s^{\text{dch,max}}, \quad \forall s, t$$

6. **Contraintes de couplage (pas de charge et décharge simultanées) :**
   $$p_{s,t}^{\text{ch}} \cdot p_{s,t}^{\text{dch}} = 0, \quad \forall s, t$$
   (relaxée en pratique via big-M ou variables binaires)

**Second-stage constraints $\mathcal{Y}(x, \xi)$ :**

1. **Bilan de puissance nodal :**
   $$\sum_{g \in \mathcal{G}_i} p_{g,t} + \xi_{i,t}^{\text{PV}} - p_{i,t}^{\text{curt}}(\xi) + \sum_{s \in \mathcal{S}_i} (p_{s,t}^{\text{dch}} - p_{s,t}^{\text{ch}}) + p_{i,t}^{\text{grid}}(\xi) = \xi_{i,t}^{\text{load}} - p_{i,t}^{\text{shed}}(\xi), \quad \forall i, t$$

2. **Limites réseau :**
   $$-P^{\text{grid,max}} \leq p_t^{\text{grid}}(\xi) \leq P^{\text{grid,max}}, \quad \forall t$$

3. **Limites curtailment/shedding :**
   $$0 \leq p_{i,t}^{\text{curt}}(\xi) \leq \xi_{i,t}^{\text{PV}}, \quad 0 \leq p_{i,t}^{\text{shed}}(\xi) \leq \xi_{i,t}^{\text{load}}, \quad \forall i, t$$

4. **Contraintes réseau (optionnel, pour network-aware) :**
   - Limites de flux de ligne (DC power flow approximation)
   - Limites de tension

#### 5.1.4 Reformulation tractable

Le problème TSRO bi-niveau est NP-hard en général. Nous utilisons les techniques suivantes pour le rendre tractable :

1. **Dualité forte :** Si $Q(x, \xi)$ est un LP, on peut reformuler le minimax via dualité :
   $$\max_{\xi \in \mathcal{U}} Q(x, \xi) = \max_{\xi \in \mathcal{U}} \min_{y} \{q^T y : Ax + By \geq b(\xi)\}$$
   devient un problème de programmation linéaire avec contraintes bilinéaires.

2. **Column-and-constraint generation (C&CG) :**
   - Résoudre itérativement un master problem (optimisation sur $x$) et un subproblem (recherche du pire scénario $\xi^*$)
   - Ajouter des contraintes de faisabilité (cuts) au master problem

3. **Affine decision rules (ADR) :**
   - Restreindre les décisions de recours à des fonctions affines de $\xi$ :
     $$y_t(\xi) = y_t^0 + \sum_{j} y_t^j \xi_t^j$$
   - Réduit le problème à un LP de taille polynomiale

4. **Scenario-based approximation :**
   - Remplacer $\max_{\xi \in \mathcal{U}}$ par $\max_{\xi \in \Xi}$ où $\Xi$ est un ensemble fini de scénarios extrêmes
   - Résoudre un MILP avec contraintes pour chaque scénario

**Notre choix :** Nous utilisons une combinaison de **ADR** pour le recours et **scenario-based approximation** avec génération de scénarios décision-aware (voir section 5.3).

### 5.2 Module de prévision probabiliste

#### 5.2.1 Architecture du forecaster

**Input features :**
- Historique des observations : $\{\xi_{t-H:t-1}\}$ (fenêtre de $H$ pas de temps)
- Variables exogènes : heure du jour, jour de la semaine, saison, météo (température, irradiance, nébulosité)
- Embeddings temporels : encodage cyclique (sin/cos) pour capturer périodicité

**Architecture neuronale :**

Nous utilisons un **Temporal Fusion Transformer (TFT)** adapté pour prévision probabiliste :

1. **Variable selection network :** sélectionne les features pertinentes
2. **LSTM encoder :** encode l'historique temporel
3. **Multi-head attention :** capture les dépendances long-terme
4. **Quantile output layer :** prédit $K$ quantiles $\{\tau_1, \ldots, \tau_K\}$ (e.g., $\tau = [0.05, 0.25, 0.5, 0.75, 0.95]$)

**Output :**
- Pour chaque variable $j$ et horizon $t$ : $\hat{q}_{t}^{j}(\tau_k) = f_\theta(\xi_{t-H:t-1}, \text{exog}_t)_k^j$
- Intervalles de prédiction : $[\hat{q}_t^j(\tau_{\alpha/2}), \hat{q}_t^j(\tau_{1-\alpha/2})]$ pour niveau $1-\alpha$

#### 5.2.2 Loss function standard (baseline)

**Pinball loss (quantile loss) :**

$$
L_{\text{pinball}}(\theta) = \frac{1}{NTK} \sum_{n=1}^N \sum_{t=1}^T \sum_{k=1}^K \sum_{j} \rho_{\tau_k}(\xi_{n,t}^j - \hat{q}_{n,t}^j(\tau_k))
$$

où $\rho_\tau(e) = \tau \max(e, 0) + (1-\tau) \max(-e, 0)$ est la pinball loss.

**CRPS (Continuous Ranked Probability Score) :**

$$
\text{CRPS}(\hat{F}_t, \xi_t) = \int_{-\infty}^{+\infty} (\hat{F}_t(z) - \mathbb{1}\{z \geq \xi_t\})^2 dz
$$

approximé via quantiles :
$$
\text{CRPS} \approx \sum_{k=1}^K w_k \rho_{\tau_k}(\xi_t - \hat{q}_t(\tau_k))
$$

**Calibration post-processing (conformal prediction) :**

Après entraînement avec pinball loss, on calibre les intervalles sur un ensemble de validation :

1. Calculer les résidus non-conformité : $r_n = \max(\underline{q}_n - \xi_n, \xi_n - \overline{q}_n)$
2. Déterminer le quantile empirique : $\hat{r}_\alpha = \text{Quantile}_{1-\alpha}(\{r_n\})$
3. Intervalles calibrés : $[\underline{q}_t - \hat{r}_\alpha, \overline{q}_t + \hat{r}_\alpha]$

Garantie : $\mathbb{P}(\xi_t \in [\underline{q}_t - \hat{r}_\alpha, \overline{q}_t + \hat{r}_\alpha]) \geq 1 - \alpha$ (coverage marginale).

#### 5.2.3 Génération de scénarios multivariés

**Problème :** Les quantiles univariés ne capturent pas les dépendances entre PV, charge, prix.

**Solution : Low-rank Gaussian copula**

1. **Marginal transformation :**
   - Transformer chaque variable en uniforme via CDF empirique : $u_t^j = \hat{F}_t^j(\xi_t^j)$

2. **Copula fitting :**
   - Modéliser la dépendance via une copule Gaussienne avec matrice de corrélation $\Sigma$ :
     $$\mathbf{z}_t = \Phi^{-1}(\mathbf{u}_t) \sim \mathcal{N}(0, \Sigma)$$
   - Estimer $\Sigma$ sur données historiques (ou via low-rank approximation $\Sigma \approx LL^T$)

3. **Scenario generation :**
   - Échantillonner $\mathbf{z}_t^{(s)} \sim \mathcal{N}(0, \Sigma)$, $s = 1, \ldots, S$
   - Transformer en uniforme : $\mathbf{u}_t^{(s)} = \Phi(\mathbf{z}_t^{(s)})$
   - Inverser les marginales : $\xi_t^{j,(s)} = (\hat{F}_t^j)^{-1}(u_t^{j,(s)})$

**Scenario reduction décision-aware :**

Pour réduire $S$ scénarios à $S' \ll S$ sans perte de qualité décisionnelle :

1. Résoudre le problème EMS pour chaque scénario $\xi^{(s)}$ et obtenir la décision optimale $x^{*(s)}$
2. Calculer la sensibilité décisionnelle : $\Delta C^{(s)} = C(x^{*(s)}, \xi^{(s)}) - \min_s C(x^{*(s)}, \xi^{(s)})$
3. Clustering des scénarios dans l'espace des décisions (k-means sur $\{x^{*(s)}\}$)
4. Sélectionner $S'$ représentants (centres de clusters) avec poids proportionnels à $\Delta C^{(s)}$

### 5.3 Optimisation robuste bi-niveau

#### 5.3.1 Reformulation avec ADR

**Affine decision rules pour le recours :**

$$
y_t(\xi) = y_t^0 + \sum_{j} Y_t^j (\xi_t^j - \hat{\xi}_t^j)
$$

où $y_t^0$ et $Y_t^j$ sont des variables de décision.

**Problème reformulé :**

$$
\min_{x, y^0, Y} \left\{ C_1(x) + \max_{\xi \in \mathcal{U}} \sum_t q^T y_t(\xi) \right\}
$$

s.t. contraintes first-stage et contraintes second-stage pour tout $\xi \in \mathcal{U}$.

**Tractabilité :** Le problème devient un LP/MILP de taille polynomiale en $T$ et $|\mathcal{U}|$.

#### 5.3.2 Column-and-constraint generation (C&CG)

**Algorithme itératif :**

**Initialization :** $\mathcal{U}_{\text{active}} = \emptyset$, $k = 0$

**Repeat until convergence :**

1. **Master problem :** Résoudre
   $$
   \min_{x, \eta} \left\{ C_1(x) + \eta : \eta \geq Q(x, \xi^{(s)}), \; \forall \xi^{(s)} \in \mathcal{U}_{\text{active}} \right\}
   $$
   Obtenir solution candidate $\bar{x}^{(k)}$, $\bar{\eta}^{(k)}$

2. **Subproblem (adversarial) :** Résoudre
   $$
   \xi^{(k)} = \arg\max_{\xi \in \mathcal{U}} Q(\bar{x}^{(k)}, \xi)
   $$
   Obtenir pire scénario $\xi^{(k)}$ et coût $Q^{(k)} = Q(\bar{x}^{(k)}, \xi^{(k)})$

3. **Convergence check :**
   - Si $Q^{(k)} - \bar{\eta}^{(k)} \leq \epsilon$ : **STOP**, solution optimale $\bar{x}^{(k)}$
   - Sinon : ajouter $\xi^{(k)}$ à $\mathcal{U}_{\text{active}}$, $k \leftarrow k+1$

**Complexité :** Nombre d'itérations typiquement $O(10-100)$ pour convergence.

#### 5.3.3 Approximation scenario-based

**Alternative computationnellement efficace :**

Remplacer $\max_{\xi \in \mathcal{U}}$ par $\max_{\xi \in \Xi}$ où $\Xi = \{\xi^{(1)}, \ldots, \xi^{(S)}\}$ est un ensemble fini de scénarios.

**Problème équivalent :**

$$
\min_{x, \eta} \left\{ C_1(x) + \eta : \eta \geq Q(x, \xi^{(s)}), \; s = 1, \ldots, S \right\}
$$

**Génération de $\Xi$ :**
- Scénarios extrêmes : coins de l'ensemble d'incertitude $\mathcal{U}$
- Scénarios probabilistes : échantillonnés via copula (section 5.2.3)
- Scénarios historiques : réalisations passées

**Notre approche hybride :**
- $\Xi = \Xi_{\text{extreme}} \cup \Xi_{\text{prob}}$
- $\Xi_{\text{extreme}}$ : 10–20 scénarios extrêmes (worst-case PV, load, prix)
- $\Xi_{\text{prob}}$ : 30–80 scénarios probabilistes (copula + scenario reduction)

### 5.4 Couplage décision-prévision (SPO loss différentiable)

#### 5.4.1 Regret opérationnel

**Définition du regret :**

Pour une réalisation $\xi$, le regret de la décision $x$ basée sur prévision $\hat{\xi}$ est :

$$
\text{Regret}(x, \xi) = C(x, \xi) - C(x^*(\xi), \xi)
$$

où :
- $C(x, \xi) = C_1(x) + Q(x, \xi)$ : coût total réalisé
- $x^*(\xi)$ : décision optimale avec connaissance parfaite de $\xi$

**Objectif decision-focused :**

$$
\min_\theta \mathbb{E}_{\xi \sim P_{\text{true}}} [\text{Regret}(x^*(\hat{\xi}_\theta), \xi)]
$$

où $\hat{\xi}_\theta = \pi_\theta(\text{features})$ est la prévision du modèle.

**Problème :** $x^*(\hat{\xi}_\theta)$ est non-différentiable par rapport à $\theta$.

#### 5.4.2 SPO loss surrogate

**SPO loss classique (pour LP) :**

$$
L_{\text{SPO}}(\theta) = (c(\xi) - c(\hat{\xi}_\theta))^T (x^*(\hat{\xi}_\theta) - x^*(\xi))
$$

où $c(\cdot)$ est le vecteur de coûts du problème de décision.

**Limitation :** Ne s'applique pas directement aux problèmes TSRO bi-niveaux.

**Notre extension : SPO-TSRO loss**

Nous proposons une fonction de perte surrogate qui approxime le regret TSRO :

$$
L_{\text{SPO-TSRO}}(\theta) = \underbrace{C_1(x^*(\hat{\xi}_\theta))}_{\text{first-stage cost}} + \underbrace{\max_{s \in \Xi} Q(x^*(\hat{\xi}_\theta), \xi^{(s)})}_{\text{worst-case recourse}} - \underbrace{C_1(x^*(\xi)) - \max_{s \in \Xi} Q(x^*(\xi), \xi^{(s)})}_{\text{clairvoyant cost}}
$$

**Simplification pour différentiabilité :**

1. **Approximation du max :** Utiliser softmax pour rendre le max différentiable :
   $$\max_s Q(x, \xi^{(s)}) \approx \frac{1}{\beta} \log \sum_s \exp(\beta Q(x, \xi^{(s)}))$$
   où $\beta$ est un paramètre de température.

2. **Gradient via différentiation implicite :**
   - Pour $x^*(\hat{\xi}_\theta)$, utiliser le théorème de l'enveloppe (envelope theorem) :
     $$\frac{\partial C(x^*(\hat{\xi}_\theta), \xi)}{\partial \theta} = \frac{\partial C(x^*(\hat{\xi}_\theta), \xi)}{\partial \hat{\xi}_\theta} \frac{\partial \hat{\xi}_\theta}{\partial \theta}$$
   - Le terme $\frac{\partial C}{\partial \hat{\xi}_\theta}$ est calculé via les multiplicateurs de Lagrange (dual variables) du problème d'optimisation.

#### 5.4.3 Gradient computation via cvxpylayers

**Implémentation pratique :**

Nous utilisons **cvxpylayers** (Agrawal et al., 2019) pour différentiation automatique :

```python
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Define optimization problem as CVXPY problem
x = cp.Variable(n_vars)
xi_param = cp.Parameter(n_uncertain)
objective = cp.Minimize(cost_function(x, xi_param))
constraints = [constraint1(x, xi_param), constraint2(x), ...]
problem = cp.Problem(objective, constraints)

# Create differentiable layer
cvxpy_layer = CvxpyLayer(problem, parameters=[xi_param], variables=[x])

# In forward pass
xi_pred = forecaster(features)  # PyTorch tensor
x_opt, = cvxpy_layer(xi_pred)  # Solve optimization, get solution

# Compute cost with true realization
cost_realized = compute_cost(x_opt, xi_true)

# Backward pass (automatic differentiation)
loss = cost_realized.mean()
loss.backward()  # Gradients flow through cvxpy_layer to forecaster
```

**Limitations et solutions :**

- **MILP non-différentiable :** Pour les variables binaires (unit commitment), nous utilisons une relaxation continue pendant l'entraînement et arrondissons en évaluation.
- **Infeasibility :** Ajouter des variables de slack avec pénalités élevées pour garantir la faisabilité.
- **Scalabilité :** Limiter la taille du problème d'optimisation (horizon court, agrégation spatiale) pendant l'entraînement.

#### 5.4.4 Hybrid loss function

**Combinaison forecast accuracy + decision cost :**

$$
L_{\text{total}}(\theta) = \lambda L_{\text{forecast}}(\theta) + (1 - \lambda) L_{\text{SPO-TSRO}}(\theta)
$$

où :
- $L_{\text{forecast}}(\theta)$ : pinball loss ou CRPS (garantit calibration)
- $L_{\text{SPO-TSRO}}(\theta)$ : regret opérationnel (optimise décisions)
- $\lambda \in [0, 1]$ : hyperparamètre de trade-off

**Stratégie d'entraînement en deux phases :**

1. **Phase 1 (warm-up) :** $\lambda = 1$ (pure forecast training) pour 50 epochs
   - Objectif : obtenir des prévisions calibrées et stables
2. **Phase 2 (fine-tuning) :** $\lambda = 0.3$ (decision-focused) pour 100 epochs
   - Objectif : affiner les prévisions pour minimiser le regret opérationnel

### 5.5 Algorithme d'entraînement end-to-end

#### 5.5.1 Pseudo-code détaillé

```
Algorithm: End-to-End Decision-Focused Training for Microgrid EMS

Input:
  - Training data: {(features_n, xi_n)}_{n=1}^N
  - Forecaster architecture: pi_theta (TFT with quantile outputs)
  - Optimization problem: TSRO-EMS(xi_pred)
  - Hyperparameters: lambda, beta, learning_rate, batch_size, epochs

Output:
  - Trained forecaster parameters theta*

Initialization:
  - Initialize theta randomly
  - Initialize optimizer (Adam with learning_rate)

# Phase 1: Forecast-focused warm-up
for epoch = 1 to 50:
    for batch in DataLoader(training_data, batch_size):
        # Forward pass: predict quantiles
        xi_pred = pi_theta(batch.features)
        
        # Compute pinball loss
        L_forecast = pinball_loss(xi_pred, batch.xi_true)
        
        # Backward pass
        L_forecast.backward()
        optimizer.step()
        optimizer.zero_grad()

# Phase 2: Decision-focused fine-tuning
for epoch = 51 to 150:
    for batch in DataLoader(training_data, batch_size):
        # Forward pass: predict quantiles
        xi_pred = pi_theta(batch.features)
        
        # Generate scenarios from predicted quantiles
        scenarios_pred = generate_scenarios(xi_pred)  # Copula sampling
        scenarios_true = batch.xi_true
        
        # Solve TSRO-EMS with predicted scenarios
        x_pred = solve_TSRO_EMS(scenarios_pred)  # via cvxpylayers
        
        # Compute realized cost with true scenarios
        cost_realized = compute_cost(x_pred, scenarios_true)
        
        # Compute clairvoyant cost (oracle)
        x_oracle = solve_TSRO_EMS(scenarios_true)
        cost_oracle = compute_cost(x_oracle, scenarios_true)
        
        # SPO-TSRO loss (regret)
        L_SPO = (cost_realized - cost_oracle).mean()
        
        # Forecast loss (for calibration)
        L_forecast = pinball_loss(xi_pred, batch.xi_true)
        
        # Combined loss
        L_total = lambda * L_forecast + (1 - lambda) * L_SPO
        
        # Backward pass
        L_total.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    # Validation and early stopping
    if epoch % 10 == 0:
        val_regret = evaluate_on_validation_set(pi_theta)
        if val_regret < best_val_regret:
            best_val_regret = val_regret
            save_checkpoint(theta)
        else:
            patience_counter += 1
            if patience_counter > patience_limit:
                break  # Early stopping

# Post-processing: Conformal calibration
theta* = load_best_checkpoint()
calibrate_prediction_intervals(pi_theta*, calibration_set)

return theta*
```

#### 5.5.2 Détails d'implémentation

**Génération de scénarios dans la boucle d'entraînement :**

```python
def generate_scenarios(xi_pred_quantiles, n_scenarios=50):
    """
    Generate multivariate scenarios from predicted quantiles.
    
    Args:
        xi_pred_quantiles: [batch, T, n_vars, n_quantiles]
        n_scenarios: number of scenarios to generate
    
    Returns:
        scenarios: [batch, n_scenarios, T, n_vars]
    """
    batch_size, T, n_vars, n_quantiles = xi_pred_quantiles.shape
    
    # Fit marginal CDFs (piecewise linear interpolation)
    marginal_cdfs = fit_piecewise_linear_cdf(xi_pred_quantiles)
    
    # Sample from Gaussian copula
    z = torch.randn(batch_size, n_scenarios, T, n_vars)  # Standard normal
    z_correlated = z @ cholesky_correlation_matrix  # Apply correlation
    u = torch.distributions.Normal(0, 1).cdf(z_correlated)  # Uniform [0,1]
    
    # Inverse transform via marginal CDFs
    scenarios = marginal_cdfs.inverse(u)
    
    return scenarios
```

**Résolution TSRO avec cvxpylayers :**

```python
def solve_TSRO_EMS(scenarios):
    """
    Solve two-stage robust EMS problem.
    
    Args:
        scenarios: [batch, n_scenarios, T, n_vars]
    
    Returns:
        x_opt: [batch, n_first_stage_vars]
    """
    # Define CVXPY problem (done once, outside training loop)
    # x: first-stage variables
    # y: second-stage variables (for each scenario)
    # scenarios_param: CVXPY parameter
    
    # Objective: min C1(x) + max_s Q(x, scenario_s)
    # Approximated as: min C1(x) + (1/beta) * log(sum_s exp(beta * Q(x, scenario_s)))
    
    # Solve via cvxpylayers
    x_opt, = cvxpy_layer(scenarios)
    
    return x_opt
```

**Calcul du coût réalisé :**

```python
def compute_cost(x_first_stage, scenarios_true):
    """
    Compute realized operational cost.
    
    Args:
        x_first_stage: [batch, n_first_stage_vars]
        scenarios_true: [batch, n_scenarios_true, T, n_vars]
    
    Returns:
        cost: [batch]
    """
    # First-stage cost
    cost_first = compute_first_stage_cost(x_first_stage)
    
    # Second-stage cost (worst-case over true scenarios)
    cost_second = []
    for s in range(scenarios_true.shape[1]):
        y_opt = solve_second_stage(x_first_stage, scenarios_true[:, s])
        cost_s = compute_second_stage_cost(y_opt, scenarios_true[:, s])
        cost_second.append(cost_s)
    
    cost_second = torch.stack(cost_second, dim=1).max(dim=1)[0]  # Worst-case
    
    total_cost = cost_first + cost_second
    return total_cost
```

#### 5.5.3 Considérations computationnelles

**Complexité par itération :**
- Forward pass forecaster : $O(H \cdot d_{\text{model}}^2)$ (TFT)
- Génération scénarios : $O(S \cdot T \cdot n_{\text{vars}}^2)$ (copula)
- Résolution TSRO : $O(S \cdot n_{\text{vars}}^3)$ (LP solver)
- Backward pass : $O(H \cdot d_{\text{model}}^2 + S \cdot n_{\text{vars}}^3)$ (autodiff + implicit diff)

**Temps d'entraînement estimé :**
- Dataset : 1000 jours, batch_size=32, 150 epochs
- Temps par epoch : ~10 min (GPU V100)
- Temps total : ~25 heures

**Optimisations :**
1. **Parallel scenario solving :** Résoudre les $S$ scénarios en parallèle (GPU)
2. **Warm-start solver :** Utiliser la solution de l'itération précédente comme initialisation
3. **Reduced horizon :** Entraîner sur horizon 12h au lieu de 24h, évaluer sur 24h
4. **Mixed precision training :** FP16 pour le forecaster, FP32 pour l'optimiseur

---

## 6. SETUP EXPÉRIMENTAL ET SIMULATION

### 6.1 Datasets utilisés

#### 6.1.1 IEEE 33-bus microgrid

**Description :**
- Réseau radial de distribution 33 nœuds, 12.66 kV
- Topologie : 1 point de connexion au réseau principal, 32 nœuds de charge
- Modifications pour microgrid :
  - 10 nœuds avec PV (capacité totale 2 MW)
  - 3 systèmes de stockage (500 kWh chacun, P_max = 200 kW)
  - 2 générateurs diesel (500 kW, 300 kW)

**Données temporelles :**
- Horizon : 365 jours (2023), résolution 1 heure
- Profils PV : données NREL NSRDB (irradiance → puissance via modèle PVWatts)
- Profils charge : données OpenEI Commercial Building Load Profiles
- Prix électricité : données marché day-ahead (ISO New England)

**Preprocessing :**
- Normalisation : z-score par variable
- Train/Val/Test split : 70%/15%/15% (chronologique)
- Features : 168h historique (1 semaine) + variables exogènes (météo, calendrier)

#### 6.1.2 IEEE 69-bus microgrid

**Description :**
- Réseau radial 69 nœuds, 12.66 kV (plus grande échelle)
- Modifications :
  - 20 nœuds avec PV (capacité totale 5 MW)
  - 5 systèmes de stockage (1 MWh chacun)
  - 4 générateurs diesel

**Données :** Même sources que IEEE 33-bus, échelle adaptée.

#### 6.1.3 CityLearn building cluster

**Description :**
- Environnement de simulation multi-bâtiments (CityLearn Challenge 2022)
- 9 bâtiments commerciaux/résidentiels avec :
  - Charges électriques et thermiques
  - Systèmes HVAC flexibles
  - Batteries individuelles
  - Panneaux PV sur toits

**Données :**
- Horizon : 365 jours, résolution 1 heure
- Profils charge/PV : données synthétiques calibrées sur bâtiments réels
- Prix : tarifs time-of-use + demand charges

**Spécificités :**
- Couplage thermique-électrique (HVAC)
- Contraintes de confort (température intérieure)
- Objectif : minimiser coût + émissions CO2

### 6.2 Métriques d'évaluation

#### 6.2.1 Métriques de prévision

**1. CRPS (Continuous Ranked Probability Score) :**
$$
\text{CRPS} = \frac{1}{NT} \sum_{n,t} \int_{-\infty}^{+\infty} (\hat{F}_{n,t}(z) - \mathbb{1}\{z \geq \xi_{n,t}\})^2 dz
$$

**2. Pinball loss (quantile score) :**
$$
\text{Pinball}_\tau = \frac{1}{NT} \sum_{n,t} \rho_\tau(\xi_{n,t} - \hat{q}_{n,t}(\tau))
$$

**3. Interval coverage :**
$$
\text{Coverage}_\alpha = \frac{1}{NT} \sum_{n,t} \mathbb{1}\{\xi_{n,t} \in [\hat{q}_{n,t}(\alpha/2), \hat{q}_{n,t}(1-\alpha/2)]\}
$$
Cible : $1 - \alpha$ (e.g., 90% pour $\alpha=0.1$)

**4. Interval width :**
$$
\text{Width}_\alpha = \frac{1}{NT} \sum_{n,t} (\hat{q}_{n,t}(1-\alpha/2) - \hat{q}_{n,t}(\alpha/2))
$$
Plus étroit = meilleur (à coverage égale)

#### 6.2.2 Métriques décisionnelles

**1. Operational cost (coût réalisé) :**
$$
\text{Cost}_{\text{realized}} = \frac{1}{N} \sum_{n=1}^N C(x_n^*, \xi_n^{\text{true}})
$$
où $x_n^*$ est la décision basée sur prévision $\hat{\xi}_n$.

**2. Regret opérationnel :**
$$
\text{Regret} = \frac{1}{N} \sum_{n=1}^N [C(x_n^*, \xi_n^{\text{true}}) - C(x_n^{\text{oracle}}, \xi_n^{\text{true}})]
$$
où $x_n^{\text{oracle}}$ est la décision avec connaissance parfaite.

**3. Regret relatif (%) :**
$$
\text{Regret}_{\%} = \frac{\text{Regret}}{\frac{1}{N} \sum_n C(x_n^{\text{oracle}}, \xi_n^{\text{true}})} \times 100
$$

**4. Feasibility rate :**
$$
\text{Feasibility} = \frac{1}{N} \sum_{n=1}^N \mathbb{1}\{\text{all constraints satisfied for } x_n^*, \xi_n^{\text{true}}\}
$$

**5. Robustness score :**
$$
\text{Robustness} = \frac{1}{N} \sum_{n=1}^N \mathbb{1}\{\text{no load shedding or curtailment for } x_n^*, \xi_n^{\text{true}}\}
$$

#### 6.2.3 Métriques de comparaison

**1. Cost reduction vs baseline :**
$$
\Delta \text{Cost}_{\%} = \frac{\text{Cost}_{\text{baseline}} - \text{Cost}_{\text{ours}}}{\text{Cost}_{\text{baseline}}} \times 100
$$

**2. Pareto efficiency (cost vs robustness) :**
- Tracer frontière Pareto : coût vs taux de délestage
- Comparer position des méthodes sur cette frontière

### 6.3 Baselines de comparaison

#### Baseline 1 : Predict-then-Optimize (CRPS-trained)
- **Forecaster :** TFT entraîné avec CRPS loss
- **Optimizer :** TSRO avec intervalles de prédiction calibrés (conformal)
- **Justification :** État de l'art actuel en pratique

#### Baseline 2 : Predict-then-Optimize (MAE-trained)
- **Forecaster :** TFT entraîné avec MAE (prévisions ponctuelles)
- **Optimizer :** TSRO avec intervalles construits via quantiles empiriques des résidus
- **Justification :** Approche simple, largement utilisée

#### Baseline 3 : SPO (stochastic programming)
- **Forecaster :** TFT avec SPO loss pour problème stochastique (non robuste)
- **Optimizer :** Two-stage stochastic programming (espérance sur scénarios)
- **Justification :** Decision-focused learning sans robustesse

#### Baseline 4 : SPO+ (improved surrogate)
- **Forecaster :** TFT avec SPO+ loss (surrogate convexe)
- **Optimizer :** Two-stage stochastic programming
- **Justification :** Meilleure convergence que SPO classique

#### Baseline 5 : Deep Reinforcement Learning (SAC)
- **Agent :** Soft Actor-Critic (SAC) avec réseau de politique et de valeur
- **State :** Historique observations + prévisions ponctuelles
- **Action :** Décisions de dispatch (continues)
- **Reward :** Négatif du coût opérationnel
- **Justification :** Approche end-to-end alternative, populaire en EMS

#### Baseline 6 : Deep Reinforcement Learning (PPO)
- **Agent :** Proximal Policy Optimization
- **Justification :** Autre algorithme DRL de référence

#### Baseline 7 : Model Predictive Control (MPC) avec perfect foresight
- **Forecaster :** Connaissance parfaite de $\xi$ (oracle)
- **Optimizer :** TSRO (mais sans incertitude)
- **Justification :** Borne supérieure de performance (upper bound)

### 6.4 Outils de simulation

#### 6.4.1 Stack logiciel

**Langage :** Python 3.10

**Frameworks :**
- **PyTorch 2.0 :** Deep learning (forecaster, training loop)
- **PyTorch Lightning :** Structuration du code, logging, checkpointing
- **CVXPY 1.3 :** Modélisation problèmes d'optimisation convexe
- **cvxpylayers :** Différentiation automatique de CVXPY
- **Pyomo 6.5 :** Modélisation MILP (pour C&CG, comparaisons)
- **Gurobi 10.0 / CPLEX 22.1 :** Solveurs commerciaux (MILP, QP)
- **MOSEK 10.0 :** Solveur pour problèmes coniques (DRO)

**Librairies auxiliaires :**
- **pandas, numpy :** Manipulation de données
- **scikit-learn :** Preprocessing, métriques
- **matplotlib, seaborn, plotly :** Visualisation
- **wandb :** Tracking expériences, hyperparameter tuning
- **hydra :** Gestion configurations

#### 6.4.2 Architecture du code

```
project/
├── config/
│   ├── dataset/
│   │   ├── ieee33.yaml
│   │   ├── ieee69.yaml
│   │   └── citylearn.yaml
│   ├── model/
│   │   ├── tft.yaml
│   │   └── lstm.yaml
│   ├── optimizer/
│   │   ├── tsro.yaml
│   │   └── stochastic.yaml
│   └── experiment.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataloaders.py
├── models/
│   ├── forecaster.py  # TFT, LSTM, etc.
│   ├── optimizer.py   # TSRO, stochastic, DRO
│   └── end_to_end.py  # Integrated model
├── losses/
│   ├── forecast_losses.py  # Pinball, CRPS
│   ├── spo_losses.py       # SPO, SPO+, SPO-TSRO
│   └── hybrid_loss.py
├── training/
│   ├── trainer.py
│   └── callbacks.py
├── evaluation/
│   ├── metrics.py
│   └── visualize.py
├── baselines/
│   ├── predict_then_optimize.py
│   ├── spo.py
│   ├── drl.py  # SAC, PPO
│   └── mpc.py
├── utils/
│   ├── scenario_generation.py
│   ├── conformal.py
│   └── helpers.py
├── experiments/
│   ├── run_experiment.py
│   └── hyperparameter_search.py
└── notebooks/
    ├── exploratory_analysis.ipynb
    └── results_visualization.ipynb
```

#### 6.4.3 Environnement de calcul

**Hardware :**
- **CPU :** 32 cores (Intel Xeon ou AMD EPYC)
- **GPU :** 4x NVIDIA V100 32GB ou A100 40GB
- **RAM :** 256 GB
- **Storage :** 2 TB SSD

**Parallélisation :**
- **Data parallelism :** Entraînement forecaster sur multi-GPU (PyTorch DDP)
- **Scenario parallelism :** Résolution TSRO pour différents scénarios en parallèle
- **Hyperparameter search :** Ray Tune avec 8 workers parallèles

---

## 7. RÉSULTATS ATTENDUS ET ANALYSE

### 7.1 Résultats quantitatifs attendus

#### 7.1.1 Tableau comparatif principal (IEEE 33-bus)

| Méthode | CRPS ↓ | Coverage (%) | Cost (k$) ↓ | Regret (%) ↓ | Feasibility (%) ↑ | Robustness (%) ↑ |
|---------|--------|--------------|-------------|--------------|-------------------|------------------|
| MPC (oracle) | — | — | **45.2** | **0.0** | **100.0** | **100.0** |
| **Ours (SPO-TSRO)** | 0.082 | 91.2 | **52.3** | **4.8** | **99.7** | **96.3** |
| SPO+ (stochastic) | 0.079 | 90.8 | 55.1 | 7.2 | 98.5 | 92.1 |
| SPO (stochastic) | 0.081 | 90.5 | 56.4 | 8.5 | 98.2 | 91.3 |
| P-then-O (CRPS) | **0.075** | **92.1** | 59.7 | 12.1 | 99.1 | 89.7 |
| P-then-O (MAE) | 0.095 | 88.3 | 62.3 | 15.3 | 97.8 | 87.2 |
| DRL (SAC) | — | — | 61.8 | 14.7 | 96.5 | 85.4 |
| DRL (PPO) | — | — | 64.2 | 17.1 | 95.8 | 83.9 |

**Observations attendues :**
- Notre méthode réduit le coût de **12.4%** vs P-then-O (CRPS) et **15.9%** vs P-then-O (MAE)
- Regret proche de l'oracle (4.8% vs 0%)
- Meilleure robustesse (96.3%) que toutes les baselines sauf oracle
- CRPS légèrement supérieur à P-then-O (trade-off forecast accuracy vs decision quality)

#### 7.1.2 Résultats sur IEEE 69-bus et CityLearn

**IEEE 69-bus (plus grande échelle) :**

| Méthode | Cost (k$) ↓ | Regret (%) ↓ | Computation time (s) |
|---------|-------------|--------------|----------------------|
| **Ours (SPO-TSRO)** | **128.7** | **5.2** | 3.8 |
| SPO+ | 136.4 | 8.1 | 3.2 |
| P-then-O (CRPS) | 145.9 | 13.5 | 2.1 |
| DRL (SAC) | 149.3 | 15.8 | 0.5 |

**CityLearn building cluster :**

| Méthode | Cost ($) ↓ | CO2 (kg) ↓ | Comfort violations (%) ↓ |
|---------|------------|------------|--------------------------|
| **Ours (SPO-TSRO)** | **8,420** | **1,235** | **2.1** |
| SPO+ | 8,950 | 1,310 | 2.8 |
| P-then-O (CRPS) | 9,680 | 1,425 | 3.5 |
| DRL (SAC) | 9,340 | 1,380 | 4.2 |

**Insights :**
- Gains relatifs similaires sur les 3 environnements (12–18% cost reduction)
- Scalabilité démontrée (IEEE 69-bus avec temps de calcul raisonnable)
- Généralisation à des objectifs multiples (coût + CO2 + confort)

### 7.2 Analyse d'ablation prévue

#### 7.2.1 Importance des composants

**Tableau d'ablation :**

| Configuration | Cost (k$) ↓ | Regret (%) ↓ | Description |
|---------------|-------------|--------------|-------------|
| **Full model** | **52.3** | **4.8** | SPO-TSRO loss + probabilistic + copula |
| w/o decision-focused loss | 59.7 | 12.1 | CRPS loss only (= P-then-O) |
| w/o probabilistic forecasting | 57.2 | 9.6 | Point forecasts + empirical intervals |
| w/o copula (independent) | 54.8 | 6.3 | Marginal quantiles, no joint modeling |
| w/o robust optimization | 55.1 | 7.2 | Stochastic programming (= SPO+) |
| w/o scenario reduction | 53.1 | 5.4 | All 1000 scenarios (10x slower) |

**Conclusions attendues :**
- **Decision-focused loss** : contribution majeure (12.4% cost reduction)
- **Probabilistic forecasting** : importante pour robustesse (9.6% → 4.8% regret)
- **Copula** : gain modéré mais significatif (6.3% → 4.8%)
- **Robust optimization** : essentiel pour robustesse (7.2% → 4.8%)
- **Scenario reduction** : bon compromis qualité/vitesse

#### 7.2.2 Sensibilité aux hyperparamètres

**1. Trade-off parameter $\lambda$ (forecast vs decision loss) :**

| $\lambda$ | CRPS ↓ | Cost (k$) ↓ | Regret (%) ↓ |
|-----------|--------|-------------|--------------|
| 1.0 (pure forecast) | **0.075** | 59.7 | 12.1 |
| 0.7 | 0.078 | 56.2 | 8.5 |
| 0.5 | 0.080 | 54.1 | 6.7 |
| **0.3** | **0.082** | **52.3** | **4.8** |
| 0.1 | 0.089 | 51.8 | 4.5 |
| 0.0 (pure decision) | 0.095 | 51.2 | 4.3 |

**Observation :** $\lambda = 0.3$ optimal (bon équilibre calibration + décision).

**2. Budget d'incertitude $\Gamma$ (conservatisme TSRO) :**

| $\Gamma$ | Cost (k$) ↓ | Feasibility (%) ↑ | Robustness (%) ↑ |
|----------|-------------|-------------------|------------------|
| 0.5 (peu conservatif) | 50.8 | 97.2 | 88.5 |
| 1.0 | 51.5 | 98.9 | 93.7 |
| **2.0** | **52.3** | **99.7** | **96.3** |
| 3.0 | 54.7 | 99.9 | 98.1 |
| 5.0 (très conservatif) | 58.2 | 100.0 | 99.2 |

**Observation :** $\Gamma = 2.0$ optimal (robustesse élevée sans sur-conservatisme).

**3. Nombre de scénarios $S$ :**

| $S$ | Cost (k$) ↓ | Computation time (s) | Regret (%) ↓ |
|-----|-------------|----------------------|--------------|
| 10 | 54.1 | 0.8 | 6.5 |
| 25 | 52.9 | 1.5 | 5.3 |
| **50** | **52.3** | **2.4** | **4.8** |
| 100 | 52.1 | 4.7 | 4.7 |
| 200 | 52.0 | 9.3 | 4.7 |

**Observation :** $S = 50$ optimal (convergence qualité, temps raisonnable).

### 7.3 Robustesse et généralisation

#### 7.3.1 Robustesse aux erreurs de prévision

**Expérience :** Ajouter du bruit artificiel aux prévisions et mesurer la dégradation de performance.

**Résultats attendus :**

| Noise level (% RMSE) | Cost increase (%) — Ours | Cost increase (%) — P-then-O |
|----------------------|--------------------------|------------------------------|
| 0% (baseline) | 0.0 | 0.0 |
| +10% | 2.3 | 5.8 |
| +20% | 4.9 | 12.7 |
| +30% | 7.8 | 21.3 |

**Conclusion :** Notre méthode est plus robuste aux erreurs de prévision (dégradation plus lente).

#### 7.3.2 Généralisation à de nouvelles conditions

**Expérience :** Entraîner sur 2023, tester sur 2024 (distribution shift).

**Résultats attendus :**

| Méthode | Regret 2023 (%) | Regret 2024 (%) | Degradation |
|---------|-----------------|-----------------|-------------|
| **Ours** | 4.8 | 6.2 | +1.4 |
| SPO+ | 7.2 | 10.5 | +3.3 |
| P-then-O | 12.1 | 17.8 | +5.7 |

**Conclusion :** Meilleure généralisation grâce à l'optimisation robuste.

#### 7.3.3 Transfert entre environnements

**Expérience :** Entraîner sur IEEE 33-bus, fine-tuner sur IEEE 69-bus (10% données).

**Résultats attendus :**

| Méthode | Regret IEEE 69 (from scratch) | Regret IEEE 69 (transfer) |
|---------|-------------------------------|---------------------------|
| **Ours** | 5.2 | **3.8** |
| P-then-O | 13.5 | 11.2 |

**Conclusion :** Le transfer learning est efficace (réduction de 26% du regret).

### 7.4 Visualisations prévues

#### Figure 1 : Comparaison des coûts opérationnels
- Boxplot des coûts journaliers pour chaque méthode (IEEE 33-bus, 365 jours)
- Mise en évidence de la réduction de variance avec notre approche

#### Figure 2 : Frontière Pareto coût vs robustesse
- Scatter plot : axe X = coût moyen, axe Y = taux de robustesse
- Montrer que notre méthode domine les baselines

#### Figure 3 : Évolution temporelle d'une journée type
- 4 subplots : (a) prévision PV vs réalisé, (b) décisions de dispatch, (c) SoC batteries, (d) coût cumulé
- Comparer Ours vs P-then-O vs Oracle

#### Figure 4 : Analyse d'ablation
- Bar chart : contribution de chaque composant au gain de performance

#### Figure 5 : Sensibilité aux hyperparamètres
- 3 subplots : (a) $\lambda$ vs cost/CRPS, (b) $\Gamma$ vs cost/robustness, (c) $S$ vs cost/time

#### Figure 6 : Calibration des prévisions
- Reliability diagram : coverage empirique vs coverage théorique pour différents niveaux $\alpha$

#### Figure 7 : Heatmap des scénarios
- Visualisation des scénarios générés (PV vs charge) avec densité de probabilité

---

## 8. CONCLUSION ET PERSPECTIVES

### 8.1 Synthèse des contributions

**Rappel des contributions majeures :**

1. **Framework end-to-end intégré** couplant prévision probabiliste et optimisation robuste via decision-focused learning
2. **SPO-TSRO loss** : première fonction de perte différentiable pour problèmes TSRO bi-niveaux
3. **Génération de scénarios décision-aware** : méthode de scenario reduction guidée par sensibilité décisionnelle
4. **Analyse théorique** : bornes de regret reliant calibration et robustesse
5. **Benchmark reproductible** : évaluation extensive sur 3 environnements avec code open-source

**Impact scientifique attendu :**
- Combler le gap entre communautés forecasting, optimization, et machine learning
- Établir un nouveau standard pour l'évaluation des méthodes EMS
- Fournir des outils pratiques pour le déploiement industriel

### 8.2 Limitations identifiées

**Limitations techniques :**
1. **Scalabilité :** Temps de calcul élevé pour réseaux très larges (>100 nœuds)
2. **Approximations :** Relaxation des variables binaires pendant l'entraînement
3. **Hypothèses :** Modèle de batterie simplifié (pas de dégradation non-linéaire)
4. **Données :** Nécessite historique suffisant pour calibration (>1 an)

**Limitations méthodologiques :**
1. **Copula Gaussienne :** Peut ne pas capturer dépendances non-linéaires complexes
2. **Horizon fixe :** Entraînement sur 24h, pas d'adaptation dynamique de l'horizon
3. **Incertitude épistémique :** Pas de quantification de l'incertitude du modèle lui-même

### 8.3 Perspectives de recherche

**Court terme (extensions directes) :**
1. **Copules non-Gaussiennes :** Vine copulas, copules archimediennes pour dépendances complexes
2. **Optimisation multi-objectif :** Pareto-optimal trade-offs coût/émissions/confort
3. **Online learning :** Adaptation en temps réel aux changements de distribution
4. **Uncertainty quantification :** Bayesian neural networks pour épistémique + aleatoric

**Moyen terme (nouvelles directions) :**
1. **Multistage robust optimization :** Extension à 3+ stages (intraday, real-time)
2. **Network-aware dispatch :** Intégration complète des contraintes AC power flow
3. **Coordination multi-microgrids :** Optimisation distribuée avec communication limitée
4. **Apprentissage fédéré :** Entraînement décentralisé préservant la confidentialité

**Long terme (vision) :**
1. **Foundation models pour EMS :** Pré-entraînement sur large corpus de données microgrids
2. **Causal inference :** Modélisation causale pour robustesse aux interventions
3. **Explainability :** Interprétation des décisions pour acceptabilité opérateurs
4. **Certification formelle :** Garanties de sûreté vérifiables pour déploiement critique

---

## 9. JOURNAUX CIBLES

### Option 1 (Recommandée) : IEEE Transactions on Smart Grid

**Profil :**
- **Impact Factor :** 9.6 (2023)
- **Quartile :** Q1 (Energy & Fuels, Engineering Electrical & Electronic)
- **Scope :** Smart grid technologies, EMS, renewable integration, optimization
- **Audience :** Chercheurs et ingénieurs en systèmes électriques
- **Délai de réponse :** 3–4 mois (première décision)
- **Taux d'acceptation :** ~25%

**Justification :**
- Excellent fit pour notre sujet (microgrid EMS + optimization + forecasting)
- Forte visibilité dans la communauté power systems
- Accepte articles méthodologiques avec validation extensive

**Format :**
- Longueur : 10–12 pages (format IEEE double-colonne)
- Références : 40–60 typiquement
- Figures : 8–12

### Option 2 : Applied Energy

**Profil :**
- **Impact Factor :** 11.2 (2023)
- **Quartile :** Q1 (Energy & Fuels, Engineering Chemical)
- **Scope :** Energy systems, optimization, renewable energy, policy
- **Audience :** Interdisciplinaire (ingénieurs, économistes, policy makers)
- **Délai de réponse :** 2–3 mois
- **Taux d'acceptation :** ~20%

**Justification :**
- IF plus élevé, grande visibilité
- Accepte approches interdisciplinaires (ML + optimization + energy)
- Valorise les contributions pratiques et benchmarks

**Format :**
- Longueur : 8,000–12,000 mots
- Références : 50–80
- Figures : 10–15

### Option 3 : IEEE Transactions on Power Systems

**Profil :**
- **Impact Factor :** 7.3 (2023)
- **Quartile :** Q1 (Engineering Electrical & Electronic)
- **Scope :** Power system operation, planning, optimization
- **Audience :** Spécialistes power systems
- **Délai de réponse :** 4–5 mois
- **Taux d'acceptation :** ~22%

**Justification :**
- Journal de référence pour optimisation power systems
- Valorise rigueur mathématique et validation sur réseaux réalistes
- Moins orienté ML que IEEE TSG (peut nécessiter plus de détails techniques)

**Format :**
- Longueur : 10–12 pages
- Références : 30–50
- Figures : 6–10

### Option 4 : Energy and AI (Elsevier)

**Profil :**
- **Impact Factor :** 5.8 (2023, nouveau journal)
- **Quartile :** Q1 (Energy & Fuels, Computer Science AI)
- **Scope :** AI/ML pour systèmes énergétiques
- **Audience :** Chercheurs ML + energy
- **Délai de réponse :** 2–3 mois
- **Taux d'acceptation :** ~30% (plus accessible)

**Justification :**
- Focus spécifique AI + energy (excellent fit)
- Valorise code open-source et reproductibilité
- Processus de review plus rapide

**Format :**
- Longueur : flexible (8–15 pages)
- Références : 40–60
- Figures : 8–12

### Option 5 : Nature Energy (stretch goal)

**Profil :**
- **Impact Factor :** 56.7 (2023)
- **Quartile :** Q1 (top journal)
- **Scope :** Breakthrough research en énergie
- **Audience :** Large (scientifiques, policy makers, grand public)
- **Délai de réponse :** 1–2 mois (desk reject rapide si pas fit)
- **Taux d'acceptation :** ~8%

**Justification :**
- Prestige maximal
- Nécessite impact transformationnel démontré
- Format court (3,000 mots + supplementary)

**Stratégie :**
- Soumettre si résultats exceptionnels (>20% cost reduction, déploiement réel)
- Préparer version longue pour IEEE TSG en backup

### Recommandation finale

**Stratégie de soumission :**

1. **Cible primaire :** **IEEE Transactions on Smart Grid**
   - Meilleur fit scope + audience + format
   - Probabilité d'acceptation raisonnable avec travail solide

2. **Backup 1 :** **Applied Energy**
   - Si IEEE TSG rejette pour raisons de scope (trop ML)
   - Adapter pour audience plus large

3. **Backup 2 :** **Energy and AI**
   - Si besoin de publication plus rapide
   - Valoriser aspect reproductibilité et code open-source

---

## 10. CALENDRIER DÉTAILLÉ SUR 4 MOIS

### Mois 1 : Implémentation et expériences préliminaires

#### Semaine 1 (1–7 avril 2026)
**Objectifs :**
- Setup environnement de développement
- Préparation des datasets

**Tâches détaillées :**
- [ ] Installer stack logiciel (PyTorch, CVXPY, cvxpylayers, Gurobi)
- [ ] Télécharger et préprocesser données IEEE 33-bus (NREL, OpenEI)
- [ ] Créer dataloaders PyTorch avec features engineering
- [ ] Implémenter train/val/test split chronologique
- [ ] Visualisations exploratoires (profils PV/charge/prix)

**Livrables :**
- Datasets preprocessed (IEEE 33-bus)
- Notebook d'analyse exploratoire

#### Semaine 2 (8–14 avril)
**Objectifs :**
- Implémentation du forecaster baseline

**Tâches :**
- [ ] Implémenter architecture TFT (Temporal Fusion Transformer)
- [ ] Entraîner avec pinball loss (baseline)
- [ ] Évaluer CRPS, coverage, interval width
- [ ] Implémenter conformal calibration
- [ ] Sauvegarder checkpoints et métriques

**Livrables :**
- Forecaster baseline entraîné (CRPS ~0.075)
- Métriques de validation

#### Semaine 3 (15–21 avril)
**Objectifs :**
- Implémentation du solveur TSRO

**Tâches :**
- [ ] Modéliser problème EMS en CVXPY (first-stage + second-stage)
- [ ] Implémenter ensemble d'incertitude budget-constrained
- [ ] Tester résolution avec scénarios fixes
- [ ] Valider faisabilité et temps de calcul
- [ ] Implémenter affine decision rules (ADR)

**Livrables :**
- Solveur TSRO fonctionnel
- Validation sur cas test simple

#### Semaine 4 (22–28 avril)
**Objectifs :**
- Baseline Predict-then-Optimize

**Tâches :**
- [ ] Intégrer forecaster + TSRO (pipeline séquentiel)
- [ ] Évaluer sur test set (cost, regret, feasibility)
- [ ] Implémenter baseline P-then-O (MAE)
- [ ] Comparer P-then-O (CRPS) vs P-then-O (MAE)
- [ ] Documenter résultats baseline

**Livrables :**
- Résultats baseline P-then-O (cost ~59.7 k$)
- Tableau comparatif préliminaire

### Mois 2 : Développement du framework end-to-end

#### Semaine 5 (29 avril – 5 mai)
**Objectifs :**
- Implémentation génération de scénarios

**Tâches :**
- [ ] Implémenter Gaussian copula (estimation $\Sigma$)
- [ ] Génération de scénarios multivariés (PV, charge, prix)
- [ ] Valider préservation des dépendances (corrélations)
- [ ] Implémenter scenario reduction décision-aware
- [ ] Tester avec différents $S$ (10, 25, 50, 100)

**Livrables :**
- Module de génération de scénarios
- Visualisations (heatmaps, scatter plots)

#### Semaine 6 (6–12 mai)
**Objectifs :**
- Implémentation SPO-TSRO loss

**Tâches :**
- [ ] Implémenter calcul du regret opérationnel
- [ ] Développer surrogate différentiable (softmax approximation)
- [ ] Intégrer cvxpylayers pour différentiation automatique
- [ ] Tester gradient flow (backward pass)
- [ ] Valider convergence sur toy problem

**Livrables :**
- SPO-TSRO loss fonctionnelle
- Tests unitaires gradient computation

#### Semaine 7 (13–19 mai)
**Objectifs :**
- Entraînement end-to-end (phase 1)

**Tâches :**
- [ ] Implémenter boucle d'entraînement (Algorithm 5.5.1)
- [ ] Phase 1 : warm-up avec pinball loss (50 epochs)
- [ ] Monitoring : loss curves, CRPS, coverage
- [ ] Checkpointing et early stopping
- [ ] Validation sur val set

**Livrables :**
- Forecaster pré-entraîné (phase 1)
- Logs wandb

#### Semaine 8 (20–26 mai)
**Objectifs :**
- Entraînement end-to-end (phase 2)

**Tâches :**
- [ ] Phase 2 : fine-tuning avec SPO-TSRO loss (100 epochs)
- [ ] Tuning hyperparamètre $\lambda$ (grid search)
- [ ] Évaluation sur test set (cost, regret)
- [ ] Comparaison avec baseline P-then-O
- [ ] Analyse des résultats préliminaires

**Livrables :**
- Modèle end-to-end entraîné
- Résultats préliminaires (cost ~52–54 k$)

### Mois 3 : Expériences complètes et baselines

#### Semaine 9 (27 mai – 2 juin)
**Objectifs :**
- Implémentation baselines SPO/SPO+

**Tâches :**
- [ ] Implémenter SPO loss (stochastic programming)
- [ ] Implémenter SPO+ loss (surrogate convexe)
- [ ] Entraîner forecasters avec SPO et SPO+
- [ ] Évaluer sur test set
- [ ] Comparer avec notre méthode

**Livrables :**
- Baselines SPO/SPO+ entraînées
- Tableau comparatif mis à jour

#### Semaine 10 (3–9 juin)
**Objectifs :**
- Implémentation baselines DRL

**Tâches :**
- [ ] Implémenter agent SAC (Soft Actor-Critic)
- [ ] Implémenter agent PPO (Proximal Policy Optimization)
- [ ] Entraîner agents (1M steps)
- [ ] Évaluer sur test set
- [ ] Comparer avec méthodes optimization-based

**Livrables :**
- Baselines DRL entraînées
- Analyse comparative DRL vs optimization

#### Semaine 11 (10–16 juin)
**Objectifs :**
- Expériences IEEE 69-bus et CityLearn

**Tâches :**
- [ ] Préparer datasets IEEE 69-bus et CityLearn
- [ ] Adapter code pour nouveaux environnements
- [ ] Entraîner notre méthode sur IEEE 69-bus
- [ ] Entraîner notre méthode sur CityLearn
- [ ] Évaluer toutes les baselines sur les 2 environnements

**Livrables :**
- Résultats sur 3 environnements (IEEE 33, 69, CityLearn)
- Démonstration de généralisation

#### Semaine 12 (17–23 juin)
**Objectifs :**
- Analyse d'ablation et sensibilité

**Tâches :**
- [ ] Ablation : w/o decision-focused loss, w/o probabilistic, w/o copula, etc.
- [ ] Sensibilité : $\lambda$, $\Gamma$, $S$
- [ ] Robustesse : ajout de bruit, distribution shift
- [ ] Transfert : IEEE 33 → IEEE 69
- [ ] Générer toutes les figures et tableaux

**Livrables :**
- Tableaux d'ablation et sensibilité
- Figures de visualisation (8–12 figures)

### Mois 4 : Rédaction et soumission

#### Semaine 13 (24–30 juin)
**Objectifs :**
- Rédaction sections principales

**Tâches :**
- [ ] Rédiger Introduction (4 pages)
- [ ] Rédiger Related Work (3 pages)
- [ ] Rédiger Problem Formulation (2 pages)
- [ ] Rédiger Methodology (4 pages)
- [ ] Première relecture et corrections

**Livrables :**
- Draft sections 1–4 (13 pages)

#### Semaine 14 (1–7 juillet)
**Objectifs :**
- Rédaction résultats et discussion

**Tâches :**
- [ ] Rédiger Experimental Setup (2 pages)
- [ ] Rédiger Results and Analysis (4 pages)
- [ ] Rédiger Discussion (2 pages)
- [ ] Rédiger Conclusion (1 page)
- [ ] Intégrer toutes les figures et tableaux

**Livrables :**
- Draft complet (24 pages)

#### Semaine 15 (8–14 juillet)
**Objectifs :**
- Analyse théorique et appendices

**Tâches :**
- [ ] Rédiger preuves théoriques (bornes de regret)
- [ ] Rédiger appendices (détails algorithmes, hyperparamètres)
- [ ] Vérifier cohérence notations mathématiques
- [ ] Compléter références bibliographiques (50–60 refs)
- [ ] Relecture technique complète

**Livrables :**
- Draft avec appendices (30 pages total)

#### Semaine 16 (15–21 juillet)
**Objectifs :**
- Finalisation et soumission

**Tâches :**
- [ ] Relecture finale (grammaire, style, clarté)
- [ ] Vérification format IEEE (template, figures, références)
- [ ] Préparation code open-source (GitHub repo)
- [ ] Rédaction README et documentation code
- [ ] Soumission à IEEE Transactions on Smart Grid
- [ ] Préparation cover letter

**Livrables :**
- Article final soumis
- Code open-source publié (GitHub + Zenodo DOI)

---

## ANNEXES

### A. Checklist de validation

**Avant soumission, vérifier :**

- [ ] Tous les résultats sont reproductibles (seed fixé, code documenté)
- [ ] Toutes les figures ont légendes claires et sont référencées dans le texte
- [ ] Tous les tableaux ont captions et sont discutés
- [ ] Toutes les équations sont numérotées et expliquées
- [ ] Toutes les notations sont définies à la première utilisation
- [ ] Toutes les références sont complètes (auteurs, année, titre, venue, DOI)
- [ ] Les contributions sont clairement énoncées dans l'introduction
- [ ] Les limitations sont discutées honnêtement
- [ ] Le code est disponible (GitHub) avec licence appropriée
- [ ] Les datasets sont décrits avec sources et preprocessing
- [ ] Les hyperparamètres sont documentés (appendice)
- [ ] Les temps de calcul sont reportés
- [ ] Les comparaisons avec baselines sont équitables (mêmes données, même hardware)
- [ ] Les résultats statistiques incluent écarts-types ou intervalles de confiance
- [ ] L'abstract est autonome et informatif
- [ ] La conclusion répond aux questions de l'introduction

### B. Ressources bibliographiques clés

**Decision-focused learning :**
- Elmachtoub & Grigas (2022). Smart "Predict, then Optimize". Management Science.
- Wilder et al. (2019). Melding the Data-Decisions Pipeline. AAAI.
- Berthet et al. (2020). Learning with Differentiable Perturbed Optimizers. NeurIPS.

**Robust optimization :**
- Ben-Tal & Nemirovski (2002). Robust Optimization. Princeton University Press.
- Bertsimas & Sim (2004). The Price of Robustness. Operations Research.
- Mohajerin Esfahani & Kuhn (2018). Data-driven Distributionally Robust Optimization. Mathematical Programming.

**Probabilistic forecasting :**
- Gneiting & Katzfuss (2014). Probabilistic Forecasting. Annual Review of Statistics.
- Vovk et al. (2005). Algorithmic Learning in a Random World. Springer.
- Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. ICML.

**Microgrid EMS :**
- Olivares et al. (2014). Trends in Microgrid Control. IEEE Trans. Smart Grid.
- Parisio et al. (2014). Stochastic Model Predictive Control for Economic/Environmental Operation Management of Microgrids. ACC.
- arXiv:2512.12755 (2024). End-to-End Approach for Microgrid Probabilistic Forecasting and Robust Operation via Decision-focused Learning.

### C. Outils et librairies

**Optimization :**
- CVXPY: https://www.cvxpy.org/
- cvxpylayers: https://github.com/cvxgrp/cvxpylayers
- Pyomo: http://www.pyomo.org/
- Gurobi: https://www.gurobi.com/

**Machine Learning :**
- PyTorch: https://pytorch.org/
- PyTorch Lightning: https://lightning.ai/
- PyTorch Forecasting: https://pytorch-forecasting.readthedocs.io/

**Benchmarks :**
- CityLearn: https://github.com/intelligent-environments-lab/CityLearn
- IEEE test feeders: https://site.ieee.org/pes-testfeeders/

**Experiment tracking :**
- Weights & Biases: https://wandb.ai/
- MLflow: https://mlflow.org/

---

## CONCLUSION DU PLAN D'ATTAQUE

Ce plan d'attaque détaillé fournit une feuille de route complète pour la rédaction de l'Article 1 sur le **Decision-Focused Probabilistic Forecasting for Microgrid Energy Management Systems**.

**Points forts du plan :**
1. ✅ Structure claire et exhaustive (10 sections principales)
2. ✅ Méthodologie rigoureuse (formulation mathématique complète, algorithmes détaillés)
3. ✅ Expériences complètes (3 environnements, 7 baselines, ablation, sensibilité)
4. ✅ Contributions originales (SPO-TSRO loss, scenario reduction décision-aware, analyse théorique)
5. ✅ Calendrier réaliste (4 mois, semaine par semaine)
6. ✅ Cibles de publication identifiées (IEEE TSG recommandé)

**Prochaines étapes immédiates :**
1. Valider le plan avec co-auteurs/superviseurs
2. Commencer Semaine 1 (setup environnement + datasets)
3. Suivre le calendrier avec discipline
4. Ajuster si nécessaire en fonction des résultats préliminaires

**Facteurs de succès :**
- Rigueur méthodologique (reproductibilité, comparaisons équitables)
- Contributions claires et originales (pas d'incrémental)
- Validation extensive (3 environnements, 7 baselines)
- Code open-source (impact communauté)
- Rédaction soignée (clarté, figures de qualité)

**Risques identifiés et mitigations :**
- **Risque :** Temps de calcul trop élevé → **Mitigation :** Optimisations (GPU, warm-start, horizon réduit)
- **Risque :** Résultats décevants → **Mitigation :** Ablation pour identifier problèmes, tuning hyperparamètres
- **Risque :** Retard calendrier → **Mitigation :** Buffer de 2 semaines, prioriser expériences critiques

**Objectif final :**
Publication dans **IEEE Transactions on Smart Grid** (Q1, IF 9.6) avec code open-source et benchmark reproductible, établissant un nouveau standard pour l'évaluation des méthodes EMS decision-focused.

---

**Document préparé le 30 mars 2026**  
**Prêt pour démarrage immédiat de l'implémentation**