# Rossmann Store Sales Forecasting : 

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Regressor-brightgreen?style=flat-square)](https://lightgbm.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-GridSearchCV-orange?style=flat-square)](https://scikit-learn.org)

---

## Problem Statement :  

Predict **daily store-level sales** across 1,115 Rossmann drug stores using structured retail features. The goal is to build a robust regression model that captures nonlinear demand behaviour across promotions, holidays, store types, and time.

**Input Features :**
- Store type and assortment class
- Promotion status and promotion intervals
- Competition distance and opening dates
- Holiday indicators (state, school, public)
- Temporal signals (day, month, year, day of week)

**Target :** `Sales` : daily revenue per store (€).

This is a **nonlinear regression problem** involving regime shifts, interaction effects, and heavy target skew: a near-perfect use case for gradient boosting.

---

## Dataset : 

From the [Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) competition.
Two files are required;

| File | Description |
|---|---|
| `train.csv` | Historical daily sales per store |
| `store.csv` | Store metadata (type, assortment, competition, promotions) |

> The two files are **merged on `Store` ID** before any processing: `train.csv` alone is missing key feature columns.

---

## EDA : 

### Sales Distribution : 

Sales exhibits strong **right skew** with a long tail of extreme revenue days. This causes heteroscedastic noise, gradient instability during training, and large residual dominance risk.

**Fix :** Log-transform the target via `np.log1p(Sales)` before training, reversed with `np.expm1()` at evaluation.

![Sales Distribution](edad10.png)

### Correlation Heatmap : 

Most pairwise linear correlations are weak. This confirms that linear models will underfit and nonlinear tree-based partitioning is required.

![Correlation Heatmap](eda10.png)

---

## Feature Engineering : 

| Step | Detail |
|---|---|
| Date parsing | `Year`, `Month` extracted; raw `Date` column dropped |
| Closed store removal | Rows where `Open == 0` excluded |
| Log transform | `Sales = np.log1p(Sales)` to stabilize variance |
| Categorical encoding | `StoreType`, `Assortment`, `StateHoliday`, `PromoInterval` label-encoded via `cat.codes` |
| Missing value fill | `fillna(0)` applied after encoding |

---

## Model -> LightGBM : 

### LightGBM over XGBoost : 

The Rossmann dataset has approximately one million training rows across 1,115 stores, with mixed feature types, categorical signals, and strong nonlinear interactions. This rules out three natural candidates cleanly; 

- **Linear regression** cannot model regime shifts (promotion on vs off, store type differences) or interaction effects between features.
- **Classical Gradient Boosting** (sklearn) is too slow at this scale; it performs exact greedy split search over raw feature values, which becomes computationally prohibitive at one million rows.
- **XGBoost** is excellent and shares the same mathematical optimization objective as LightGBM, but it grows trees level-wise, being conservative and balanced everywhere. It is built for stability on very large, noisy datasets where safe exploration matters more than speed. Here, that conservatism would cost training time without a meaningful accuracy gain.

**LightGBM is the right fit :**

- It is the lighter, faster variant of XGBoost's math, optimized specifically for large structured tabular datasets.
- It grows trees leaf-wise, targeting exactly where the model is failing most, rather than growing uniformly across the whole tree.
- It introduces GOSS and EFB which reduce computation dramatically while preserving gradient fidelity.
- Native categorical support means less preprocessing overhead on a feature-heavy retail dataset.

---

### LightGBM vs XGBoost :

Both are gradient boosting frameworks sharing the same core mathematical objective (regularized second-order loss minimization). The differences are in how efficiently and selectively that objective is optimized; 

| | LightGBM | XGBoost |
|---|---|---|
| **Tree growth strategy** | Leaf-wise (best-first) | Level-wise (depth-first) |
| **Split search** | Histogram-based binning | Exact greedy or approximate |
| **Training speed** | Faster on large datasets | Slower; more memory hungry |
| **Memory usage** | Lower (binned histograms) | Higher (stores exact values) |
| **Categorical features** | Native support | Requires manual encoding |
| **Overfitting risk** | Higher (deep asymmetric trees) | Lower (balanced level-wise growth) |
| **Small datasets** | Can overfit more easily | More stable |
| **Target dataset size** | Large to very large | Medium to large |
| **Convergence style** | Aggressive: finds best leaf fast | Conservative: grows everywhere evenly |

XGBoost grows every level of the tree uniformly before going deeper, keeping the tree balanced and conservative. LightGBM always questions which single leaf, anywhere in the tree, would reduce loss the most right now, and then splits that one. The result is faster convergence and lower training loss, but with asymmetric, sometimes very deep trees on one branch. On a dataset at the scale of Rossmann, LightGBM is typically both faster and more accurate precisely because of this aggressive targeting.

**Additional reasons LightGBM wins here specifically :**
- At ~1 million rows, XGBoost's exact greedy search becomes overkill in compute cost; histogram binning is sufficient and far faster.
- The retail feature space has many categorical columns; LightGBM handles these natively without inflating dimensionality.
- Promotional spikes and store-level fixed effects create localized high-error regions; leaf-wise growth focuses capacity exactly there.
- GOSS ensures gradients are computed on the most informative samples, not wasted on already well-predicted ones.
- EFB compresses sparse features without any information loss, reducing effective feature dimensionality.

---

### GOSS -> Gradient-Based One-Side Sampling : 

In standard gradient boosting, every training sample participates in every tree's gradient computation. GOSS changes this; instead of using all $N$ samples, it keeps only the samples that matter most for learning.

**Algorithm :**

- Rank all samples by the absolute value of their gradient $|g_i|$.
- Always keep the top $a\%$ of samples with the largest gradients; these are the samples the model is currently getting most wrong.
- From the remaining low-gradient samples (already well-predicted), randomly keep only $b\%$, and upweight their contribution by $\frac{1-a}{b}$ to correct for the sampling bias.

**Mathematical Working :**

- High-gradient samples carry the most information about where the model needs to improve. Dropping them would distort the gradient signal.
- Low-gradient samples contribute little; the model already handles them well. Subsampling them aggressively has minimal effect on gradient estimates.
- The upweighting factor $\frac{1-a}{b}$ ensures the gradient statistics remain unbiased in expectation.

Hence LightGBM computes gradients on a fraction of the data per tree while maintaining nearly the same gradient fidelity as full-data methods. At one million rows, this is a substantial practical speedup.

---

### EFB -> Exclusive Feature Bundling : 

Real-world tabular datasets, especially after one-hot encoding categoricals, often contain many sparse features where most values are zero.
EFB exploits this:

**Key Intuition :** if two features are mutually exclusive (they are never both nonzero for the same sample), they carry no overlapping information and can be bundled into a single feature without any loss.

**Algorithm :**

- Build a graph where features are nodes and edges represent co-occurrence conflicts (both nonzero at the same time).
- Find near-exclusive bundles (small conflict rate allowed as a hyperparameter) using a greedy graph coloring algorithm.
- Merge each bundle into one synthetic feature by offsetting value ranges so the original features remain distinguishable within the bundle.

**Significance :** instead of searching splits over $F$ sparse features separately, LightGBM searches over $B \ll F$ dense bundles. The split search space shrinks dramatically with no meaningful information loss, since exclusive features cannot simultaneously influence a split anyway.

For a retail dataset with many one-hot encoded store attributes, EFB can reduce the effective feature count substantially, making histogram construction and split search faster at every boosting round.

---

### Additive Ensemble : 

Predictions are built as a sum of weak learners;

$$\hat{y}_i = \sum_{t=1}^{T} f_t(x_i)$$

Each tree $f_t$ partitions feature space into leaves. Inside leaf $j$;

$$f_t(x_i) = w_j$$

LightGBM approximates a **piecewise constant nonlinear function** over the full feature space.

---

### Regularized Objective : 

At boosting iteration $t$;

$$\text{Obj}^{(t)} = \sum_i L\left(y_i,\ \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

With regularization;

$$\Omega(f_t) = \gamma \cdot T + \frac{1}{2} \lambda \sum_j w_j^2$$

- $\gamma$ : penalises number of leaves (controls tree complexity)
- $\lambda$ : penalises leaf weight magnitude (prevents aggressive overcorrection)

This objective is mathematically identical to XGBoost's. What makes LightGBM lighter is not the math but how it computes the gradients and searches for splits: via GOSS, EFB, and histogram binning, all operating on the same underlying objective.

---

### Second-Order Optimization : 

The loss is approximated via Taylor expansion :

$$L \approx g_i \cdot f_t(x_i) + \frac{1}{2} h_i \cdot f_t(x_i)^2$$

Where;
- $g_i = \partial L / \partial \hat{y}_i$ : first-order gradient (direction and magnitude of prediction error).
- $h_i = \partial^2 L / \partial \hat{y}_i^2$ : second-order Hessian (curvature of the loss; how confident we are in the gradient direction).

This enables **Newton-style weight updates** rather than vanilla gradient descent, leading to more precise and faster convergence.

---

### Optimal Leaf Weights :

Aggregating over leaf $j$;

$$G_j = \sum_{i \in j} g_i, \quad H_j = \sum_{i \in j} h_i$$

The optimal leaf output is;

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

- Large $|G_j|$ : region poorly predicted, stronger correction applied.
- Large $\lambda$ : shrinks the update, safer step taken.
- Large $H_j$ : stable curvature, confident update issued.

---

### Split Gain Formula : 

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma$$

A split is accepted only if $\text{Gain} > 0$ : every split must reduce the objective by more than the complexity cost $\gamma$ of adding a new leaf.

---

### Leaf-Wise Growth : 

Unlike level-wise trees (XGBoost default), LightGBM **always splits the leaf with the highest gain**, regardless of depth. This leads to faster training loss reduction and deeper partitions in high-error regions, with overfitting risk mitigated via `num_leaves` and `max_depth` constraints.

---

### Histogram Optimization : 

Continuous feature values are discretized into bins. Gradient and Hessian statistics are aggregated per bin; split search becomes a **linear scan over bins** rather than over raw values, reducing per-split search from $O(N)$ to $O(B)$ where $B \ll N$ is the number of bins. This is the foundational speedup that makes LightGBM practical at scale.

---

## Time and Space Complexity : 

### Training Complexity : 

Each boosting round fits one tree. With histogram binning, the split search at each node scans $B$ bins (typically 255) rather than $N$ raw values. For a tree of depth $d$ with $T$ total trees;

$$O\left(T \cdot d \cdot N \cdot B\right)$$

where $N$ is the number of training samples, $d$ is max depth, and $B$ is the number of histogram bins. Since $B \ll N$ (e.g., $B = 255$ vs $N \approx 10^6$), this is dramatically faster than the $O(T \cdot N \log N)$ of exact greedy methods like vanilla Gradient Boosting or XGBoost's exact mode.

With GOSS active, $N$ is further replaced by $a \cdot N + b \cdot (1-a) \cdot N \ll N$ per round, adding another multiplier reduction on top of histogram binning.

With Grid Search over $C$ parameter combinations and $K$ cross-validation folds;

$$O\left(C \times K \times T \cdot d \cdot N \cdot B\right)$$

This is why the tuned model takes significantly longer to train despite LightGBM's inherent efficiency.

### Prediction Complexity : 

At inference, each sample traverses all $T$ trees, following one root-to-leaf path per tree;

$$O(T \cdot \text{depth})$$

With $T$ in the hundreds and depth in single digits, per-sample inference is sub-millisecond, consistent with the very low latency observed in results.

### Space Complexity : 

LightGBM stores histogram statistics (gradient and hessian sums per bin per feature) rather than raw data. Model storage covers all tree structures, split thresholds, and leaf outputs;

$$O\left(T \cdot 2^d + F \cdot B\right)$$

where $F$ is the number of features and $B$ is bins per feature. The histogram term $F \cdot B$ dominates during training but is discarded post-training: only the tree structures ($T \cdot 2^d$ nodes) are retained in the saved model. EFB reduces $F$ further, compressing the histogram workspace.

---

## Hyperparameter Tuning : 

Grid search with 5-fold Cross Validation;

```python
param_grid = {
    "n_estimators":   [300, 600],
    "learning_rate":  [0.03, 0.05, 0.1],
    "num_leaves":     [31, 63],
    "max_depth":      [-1, 10]
}
```

Scoring metric : `neg_mean_squared_error`. Best parameters printed after grid search completes.

---

## Results :

| Model | RMSE (€) | R² | Training Time | Inference Latency |
|---|---|---|---|---|
| Base LightGBM | 684.398745 | 0.950694 | Fast | Very low |
| Tuned LightGBM | 470.709523 | 0.976677 | Slower (CV) | Marginally higher |

> Tuning reduces bias at the cost of increased computational time. Inference latency remains negligible for both: prediction complexity is $O(T \cdot \text{depth})$.

---

## Residual Analysis : 

Residuals distributed around zero with no strong systematic trend; the model is largely unbiased across the prediction range. Variance increases for high sales values, indicating **multiplicative noise** at extreme revenue points (expected for log-scale targets).

![Residual Scatter Plot](respd10.png)

Error distribution is heavy-tailed due to rare extreme promotional spikes. The model underestimates outlier events, which is expected behaviour for ensemble regressors without explicit outlier modelling.

![Error Distribution](errordistd10.png)

---

## Feature Importance :

Top predictive signals from LightGBM split-gain importance;

1. **Promotion status** : dominant regime-switch effect
2. **Store ID** : captures store-level fixed effects
3. **Day of week** : strong cyclic demand signal
4. **Month / Year** : seasonal and trend components
5. **Competition distance** : structural competitive pressure
6. **Store type and assortment** : baseline revenue tier

![Feature Importance](fimpd10.png)

---

## Failure Cases and Limitations : 

**Extreme promotional events are underpredicted :**
Promotional spikes that produce outlier sales days are severely underrepresented in the training data. The model has seen very few examples of such events, so it cannot reliably estimate their magnitude. Even with log-transform stabilizing the target, rare spikes remain in the tail of the distribution, and the model reverts toward the bulk of its training signal rather than extrapolating to extreme values. This is visible in the heavy-tailed error distribution.

**Structural regime shifts are invisible to the model :**
Sudden changes such as a store refit, a regional competitor opening nearby, or a change in assortment class mid-dataset represent discontinuities in the data generating process. The model was trained on historical patterns and has no mechanism to detect or adapt to such structural breaks. If the relationship between features and sales changes, predictions will be systematically biased until retraining occurs with post-shift data.

**High leaf count overfitting is a real risk :**
LightGBM's leaf-wise growth can produce very deep, asymmetric trees when `num_leaves` is large and training data per leaf becomes sparse. In such cases, the model memorizes noise rather than learning generalizable patterns. This risk is mitigated by `num_leaves`, `max_depth`, `min_data_in_leaf`, and `lambda` constraints, but requires careful tuning: the gap between training and validation performance widens quickly if these are not properly set.

**Extrapolation beyond the training range is not possible :**
Boosted trees cannot extrapolate. If future sales exceed the historical maximum seen during training (e.g., a record-breaking promotional week), the model will cap predictions at or below the highest values it learned from. This is a fundamental limitation of all tree-based methods: they interpolate within the convex hull of training data but have no mechanism to reason beyond it.

**Heavy tail errors persist despite log-transform :**
The log-transform on the target significantly improves gradient stability and compresses the dynamic range of the target, but it does not eliminate the challenge of extreme outliers. After reversing the transform with `expm1()`, errors in the tail of the distribution are amplified back to their original scale. The model may look well-calibrated in log-space while still producing large absolute errors for the highest-sales days in real units.

---

## Key Takeaways : 

- **Leaf-wise boosting** reduces bias faster than level-wise growth for retail demand data.
- **Histogram optimization** makes LightGBM practical at scale with no meaningful accuracy loss.
- **GOSS and EFB** further reduce computation by focusing gradient computation and feature search where they matter most.
- **Second-order updates** (Newton steps) stabilize training vs first-order gradient descent.
- **Log-transform on target** is non-negotiable for right-skewed sales data: gradients behave far better in log space.
