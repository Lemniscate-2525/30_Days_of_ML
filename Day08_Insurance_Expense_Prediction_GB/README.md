# Insurance Expense Prediction : 

## Problem : 

Insurance companies estimate expected medical expenses to design premium pricing strategies, manage financial risk, and identify high-risk customer segments. This project builds a regression model using **Gradient Boosting** to predict annual insurance expenses from demographic and lifestyle features.

Areas of Focus : Structured EDA, residual learning and functional gradient descent, learning rate and ensemble depth tradeoffs, baseline vs tuned model comparison, bias-variance behavior, and computational complexity analysis.

---

## Dataset :  

**Source :** `insurance.csv`, 1338 records, 7 columns.

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the insured individual |
| `sex` | Categorical | Male or female |
| `bmi` | Numeric | Body Mass Index |
| `children` | Numeric | Number of dependents covered |
| `smoker` | Categorical | Whether the individual smokes |
| `region` | Categorical | Residential region in the US |
| `expenses` | Numeric | **Target: annual medical insurance cost** |

Categorical features (`sex`, `smoker`, `region`) were one-hot encoded using `pd.get_dummies(drop_first = True)` before modeling.

---

## EDA :
 
### Correlation Heatmap : 

![Correlation Heatmap](corrd8.png)

Key observations :

- `smoker_yes` has by far the strongest positive correlation with `expenses` (0.79), confirming it is the dominant predictor.
- `age` (0.30) and `bmi` (0.20) show moderate positive correlation with expenses.
- Regional features and `sex_male` are near zero in correlation, contributing little direct linear signal.
- The weak linear correlations across most features confirm that a non-linear model is the right choice here.

---

### Pairplot : 

![Pairplot](pp.png)

Observations : 

- `expenses` is right-skewed, with a long tail of high-cost individuals.
- Smokers form a clearly distinct high-expense cluster, visually separable from non-smokers.
- In the `age` vs `expenses` scatter, two parallel bands emerge: smokers consistently occupy the upper band.
- These interaction-driven, non-linear patterns are exactly what Gradient Boosting is built to capture.

---

## Gradient Boosting over Linear Regression : 

A linear model assumes;

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

where $\mathbf{w}$ is the learned weight vector, $\mathbf{x}$ is the input feature vector, and $b$ is the bias term. This produces a single flat hyperplane through the feature space.

Real insurance expense behavior breaks this assumption cleanly : 

- Smoker status causes a sharp, stepwise jump in expenses, not a smooth linear shift.
- BMI interacts with smoker status to amplify costs non-linearly; a high-BMI smoker is disproportionately expensive relative to what adding the two effects separately would predict.
- Age and BMI together create layered interaction effects that a hyperplane simply cannot represent.

The result is **systematic bias and structured residuals**; the linear model is not just noisy, it is consistently wrong in predictable, patterned ways. Gradient Boosting handles this by learning non-linear boundaries and feature interactions directly through its tree-based structure.

---

## Gradient Boosting Intuition : 

Gradient Boosting builds a prediction function sequentially. Each new tree corrects the errors of all previous trees, and the ensemble grows more accurate with each stage.

**Step 1->Initializing :** with a constant prediction, the mean of all training targets;

$$F_0(x) = \bar{y}$$

where $\bar{y}$ is the mean of $y$ across all training samples. This is the dumbest possible starting point as we predict the same value for everyone.

**Step 2->Computing residuals :**, i.e., Errors that the current model still gets wrong;

$$r_1 = y - F_0(x)$$

**Step 3->Fit a tree :** $h_1(x)$ to those residuals, then update the model:

$$F_1(x) = F_0(x) + \eta \cdot h_1(x)$$

where $\eta$ (eta) is the **learning rate**, a scalar between 0 and 1 that controls how much each tree's correction is trusted.

**After $M$ total stages**, the final model is; 

$$F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)$$

where $\sum_{m=1}^{M}$ denotes summing the contributions of all $M$ trees. Each tree $h_m(x)$ targets the residuals left by the previous ensemble, so the model progressively closes the gap between its predictions and the true values.

---

## Functional Gradient Descent : 

Training minimizes total prediction loss.
For regression using Mean Squared Error;

$$\mathcal{L} = \sum_{i=1}^{n} \left( y_i - F(x_i) \right)^2$$

where $n$ is the number of training samples, $y_i$ is the actual expense for sample $i$, and $F(x_i)$ is the model's current prediction for that sample. $\mathcal{L}$ is the total loss we want to drive toward zero.

The **gradient** of this loss with respect to the prediction $F(x_i)$ tells us how the loss changes as we nudge our prediction for sample $i$:

$$\frac{\partial \mathcal{L}}{\partial F(x_i)} = -2\left( y_i - F(x_i) \right)$$

This quantity points in the direction that increases loss fastest. To decrease loss, we move in the **opposite direction**, the negative gradient:

$$-\frac{\partial \mathcal{L}}{\partial F(x_i)} = 2\left( y_i - F(x_i) \right)$$

Dropping the constant factor of 2, this simplifies to;

$$\text{negative gradient} \approx y_i - F(x_i) = \text{residual}$$

This is the core insight: **fitting a tree on residuals is exactly gradient descent in function space.** Each boosting stage trains a tree to approximate the negative gradient, then takes a step of size $\eta$ in that direction.

The model update rule is;

$$F_{\text{new}}(x) = F_{\text{old}}(x) + \eta \cdot h(x)$$

where $h(x)$ is the new tree approximating the negative gradient, and $\eta$ controls the step size. A smaller $\eta$ means each tree contributes less, requiring more trees to converge but resulting in a smoother, more stable optimization path with lower overfitting risk.

---

## Key Hyperparameters : 

**`n_estimators` ($M$) :** the total number of boosting stages. More trees reduce bias; too many risk overfitting and increase training time proportionally.

**`max_depth` :** the maximum depth of each individual tree. Shallow trees (depth 2 or 3) act as weak learners, which is intentional. Deep trees risk memorizing noise and are harder to correct in later stages.

**`learning_rate` ($\eta$) :** shrinkage applied to each tree's contribution. Lower $\eta$ demands more trees but keeps the optimization stable and resistant to overfitting.

| $\eta$ | Trees Needed | Overfitting Risk |
|---|---|---|
| High (0.3+) | Few | High |
| Medium (0.1) | Moderate | Balanced |
| Low (0.01) | Many | Low |

---

## Baseline Model : 

A baseline Gradient Boosting model was trained first with manually chosen parameters :

```
n_estimators = 200,  learning_rate = 0.05,  max_depth = 3
```

This is standard practice: establish a reference point before tuning. The baseline already captures non-linear patterns well due to Gradient Boosting's inherent strength, but its hyperparameters are not optimally calibrated for this dataset. All improvements from tuning are measured relative to this baseline.

---

## Hyperparameter Tuning via Grid Search with Cross-Validation : 

After the baseline, Grid Search with 5-fold Cross-Validation was used to find the optimal parameter combination systematically.

**Search grid :**

```python
param_grid = {
    'n_estimators':  [100, 200, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth':     [2, 3, 4]
}
```

**Best parameters found :**

```
learning_rate = 0.1,  max_depth = 2,  n_estimators = 100
```

Why this combination works: a higher learning rate (0.1) allows faster convergence with fewer trees; shallow trees (depth = 2) act as proper weak learners and limit overfitting; fewer estimators reduce computational cost while maintaining strong predictive performance. Together, this achieves a clean balance between bias reduction and variance control.

---

## Model Performance : 

| Model | RMSE | $R^2$ | Training Time | Inference Latency |
|---|---|---|---|---|
| Baseline Gradient Boosting | 4313.93 | 0.8801 | 0.34 s | 0.000012 s/sample |
| Tuned Gradient Boosting | 4335.87 | 0.8789 | 38.12 s | 0.000016 s/sample |

**Reading the results :** both models explain approximately 88% of the variance in insurance expenses ($R^2 \approx 0.88$). The tuned model's RMSE is marginally higher on this specific test split, which is not a contradiction. Grid Search optimizes for cross-validated MSE across training folds, not raw test RMSE. The tuned model is more reliable and generalizes more consistently across unseen data; the baseline may have benefited slightly from how this particular 80/20 split fell.

The large jump in training time (0.34 s to 38.12 s) reflects the full cost of Grid Search: $3 \times 3 \times 3 = 27$ parameter combinations, each trained across $K = 5$ cross-validation folds, for a total of $27 \times 5 = 135$ complete model fits.

---

## Feature Importance : 

![Feature Importance (Baseline)](fimp.png)

![Feature Importance (Tuned)](fimpt.png)

Both baseline and tuned models agree strongly on feature rankings;

- **`smoker_yes`** dominates with approximately 0.70 importance, confirming what the heatmap showed: smoking status is the single largest driver of insurance expenses.
- **`bmi`** is second at approximately 0.17, capturing the non-linear BMI-expense relationship.
- **`age`** is third at approximately 0.11.
- `children`, regional dummies, and `sex_male` contribute negligibly in both models.

The consistency between baseline and tuned importance scores confirms these rankings are stable properties of the data, not artifacts of any particular hyperparameter choice.

---

## Boosting Stage Error Curve : 

![Boosting Stage Error](bse.png)

MSE starts above $1.3 \times 10^8$ and drops sharply in the first 20 iterations, then levels off and converges near $0.2 \times 10^8$ by iteration 100. Most predictive power is gained early; the model does not overfit as iterations increase (no upward curve at the end). 100 estimators is sufficient, which is consistent with what Grid Search selected.

---

## Residual Analysis :  

![Residual Plot](resd8.png)

The residual plot shows $\text{residual} = y - \hat{y}$ (actual minus predicted) against the predicted value $\hat{y}$.

- Most residuals cluster tightly around zero for mid-range predictions (roughly 0 to 15,000), indicating the model is well-calibrated in that region.
- Larger, more scattered residuals appear at both ends: the model struggles more with very low and very high expense individuals.
- There is no strong systematic curved trend in the residuals, which means overall bias is low; the model is not consistently over or under-predicting in a patterned way.
- The cluster of large positive residuals at low predicted values corresponds to smokers whose expenses were underestimated, a known failure mode detailed below.

---

## Time and Space Complexity : 

### Training Complexity : 

Fitting a single decision tree on $N$ samples costs $O(N \log N)$, where $N \log N$ accounts for the sorting operations performed at each candidate split. With $T$ trees total:

$$O(T \cdot N \log N)$$

For Grid Search with Cross-Validation, this multiplies by the number of parameter combinations and folds. The search grid here covers $3 \times 3 \times 3 = 27$ combinations (3 values each for `n_estimators`, `learning_rate`, and `max_depth`), each trained across $K = 5$ folds:

$$O(27 \times 5 \times T \cdot N \log N) = O(135 \cdot T \cdot N \log N)$$

This factor of 135 is exactly why tuning took approximately 38 seconds versus 0.34 seconds for a single baseline fit.

### Prediction Complexity : 

At inference, every input must traverse all $T$ trees. Each traversal costs $O(\text{depth})$, since the input follows one root-to-leaf path per tree:

$$O(T \cdot \text{depth})$$

With $T = 100$ and depth $= 2$, inference is very fast, consistent with the approximately 0.000016 s/sample latency observed.

### Space Complexity :

The model stores all $T$ trees in memory. Each tree with maximum depth $d$ has at most $2^d - 1$ nodes:

$$O(T \cdot 2^d)$$

Memory usage grows linearly with $T$ and exponentially with depth, which is another reason shallow trees are preferred at scale.

---

## Failure Cases : 

**Extreme expense outliers are underpredicted :**
The model consistently underestimates expenses for the highest-cost individuals, visible as large positive residuals in the upper portion of the residual plot. These cases, typically high-BMI smokers with compounding risk factors, lie far from the bulk of the training distribution. Gradient Boosting, like most ensemble methods, regresses toward the center of the training data and cannot fully stretch its predictions to rare, extreme values.

**Feature interactions may not be fully captured :**
While Gradient Boosting handles non-linearity well, shallow trees (depth = 2) can only model pairwise interactions at each individual stage. Higher-order interactions, such as the combined effect of age, BMI, smoking status, and number of children all together, may not be fully represented. Deeper trees would capture these but at the cost of increased overfitting risk, which is why this tradeoff is managed through tuning rather than simply increasing depth.

**Extrapolation failure beyond the training distribution :**
The dataset covers ages 18 to 64, BMI roughly 16 to 53, and expenses up to approximately $63,000. For individuals whose features fall outside these observed ranges, the model has no basis for prediction and will silently produce unreliable estimates without any warning. Tree-based models cannot extrapolate; they can only interpolate within the space they have seen during training.

**Overfitting with too many trees :**
Without careful tuning, increasing `n_estimators` indefinitely causes the model to begin fitting noise in the training data rather than the true signal. The boosting stage error curve plateaus around 100 iterations; training beyond this point on this dataset adds computational cost with no accuracy benefit and degrades generalization on unseen data. This is precisely why the Grid Search selected 100 estimators over the larger options of 200 and 400.
