# Insurance Expense Prediction

##  Project Overview

Insurance companies must estimate expected medical expenses of customers to design premium pricing strategies, manage financial risk, and identify high-risk segments.

This project builds a **non-linear regression model using Gradient Boosting** to predict annual insurance expenses from demographic and lifestyle features.

Focus areas:

- Structured Data Exploratory Data Analysis  
- Residual Learning & Functional Gradient Descent  
- Learning Rate and Ensemble Depth Tradeoffs  
- Training vs Inference Efficiency  
- Bias–Variance Behavior  
- Failure Case Analysis  
- Computational Complexity  

---

##  Problem Statement

The objective is to predict **annual medical insurance expenses** using features such as:

- age  
- sex  
- bmi  
- number of children  
- smoker status  
- region  

Insurance pricing exhibits strong **non-linear relationships and feature interactions**, making Gradient Boosting an effective modeling approach.

---

##  Dataset

Target Variable:

**expenses — annual insurance medical cost**

Regression Task: Continuous Value Prediction  

---

##  Exploratory Data Analysis

### Correlation Heatmap

![Correlation Heatmap](corr.png)

Observations:

- Smoking status shows strongest positive relationship with expenses  
- BMI interaction with age contributes significantly  
- Weak purely linear correlations suggest need for nonlinear models  

---

### Pairplot Visualization

![Pairplot](pair.png)

Insights:

- Expense distribution is right-skewed  
- Smokers form a high-expense cluster  
- Feature interaction patterns visible  

---

## Why Linear Regression Underperforms

Linear models assume:

**y = wᵀx + b**

But real expense behavior shows:

- stepwise increase for smokers  
- nonlinear BMI influence  
- interaction between multiple features  

This leads to **systematic bias and structured residuals.**

---

##  Gradient Boosting Intuition

Gradient Boosting constructs prediction function sequentially.

Initial model:

**F₀(x) = mean(y)**

Residual:

**r₁ = y − F₀(x)**

Train first tree on residual.

Update rule:

**F₁(x) = F₀(x) + η · Tree₁(x)**

After M stages:

**F_M(x) = F₀(x) + η Σ Tree_m(x)**

Each tree learns to **correct previous prediction errors.**

---

##  Functional Gradient Descent Mathematics

Loss Function (Mean Squared Error):

**L = Σ (y − F(x))²**

Gradient:

**∂L/∂F = −2(y − F(x))**

Negative Gradient Direction:

**Residual ≈ y − prediction**

Thus Gradient Boosting performs **Gradient Descent in Function Space.**

---

##  Learning Rate (η) — Step Size Interpretation

Update:

**F_new(x) = F_old(x) + η · Tree(x)**

Learning rate controls how far prediction moves opposite gradient direction.

| Learning Rate | Trees Needed | Overfitting Risk |
|--------------|-------------|----------------|
| High | Few | High |
| Medium | Moderate | Balanced |
| Low | Many | Low |

---

##  Important Hyperparameters — Conceptual Role

### n_estimators  
Number of boosting stages.

- More trees reduce bias  
- Excessive trees increase overfitting risk  
- Directly increases training time  

### max_depth  
Controls complexity of each tree.

- Shallow → weak learner  
- Moderate → captures interactions  
- Deep → risk of memorization  

### learning_rate  
Controls shrinkage of gradient step.

### subsample  
Random fraction of data per tree → variance reduction.

---

##  Model Performance Comparison

| Model | RMSE | R² Score | Training Time | Inference Latency |
|------|------|---------|--------------|----------------|
| Baseline Gradient Boosting | 4313.93 | 0.8801 | 0.50 sec | 1.81e-05 sec |
| Tuned Gradient Boosting | (add) | (add) | (add) | (add) |

---

## Training vs Inference

Training:

- Sequential tree building  
- Computationally intensive  
- Depends on number of estimators and depth  

Inference:

- Must traverse all trees  
- Latency proportional to ensemble size  

---

##  Time Complexity

Training Complexity:

**O(T · N log N)**  

Where:

- T → number of trees  
- N → number of samples  

Prediction Complexity:

**O(T · depth)**  

---

## Space Complexity

Model stores all trees in ensemble:

**O(T · nodes_per_tree)**  

Memory usage increases linearly with ensemble size.

---

##  Boosting Stage Error Curve

![Boost Curve](boost_curve.png)

Error reduces as boosting progresses until convergence.

---

##  Residual Plot

![Residual Plot](residual.png)

Random residual distribution indicates reduced bias.

---

##  Feature Importance

![Feature Importance](feat_imp.png)

Top predictors:

- smoker status  
- bmi  
- age  

---

## Failure Case Analysis

- Extreme expense outliers underpredicted  
- Sensitive to learning rate tuning  
- Large ensembles increase memory footprint  
- Cannot extrapolate beyond training distribution  

---

##  Key Learnings

- Gradient Boosting reduces bias via residual learning  
- Sequential trees approximate nonlinear functions  
- Learning rate stabilizes optimization  
- Ensemble depth controls interaction modeling  
- Tradeoff exists between accuracy and computational cost  

---

##  Future Improvements

- Compare with XGBoost / LightGBM  
- Log transform target for skew handling  
- SHAP interpretability  
- Deployment as pricing prediction API  

---
