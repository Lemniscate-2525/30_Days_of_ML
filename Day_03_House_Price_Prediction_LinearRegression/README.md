# House Price Prediction using Linear Regression

## Project Overview

Predicting house prices is a classic regression problem in machine learning.

In this project we build a Linear Regression model to estimate housing prices based on features such as median income, average rooms, population, and location.

The project focuses on building a clean machine learning pipeline including preprocessing, scaling, training and evaluation.

---

## Dataset

Dataset used: California Housing Dataset

Features include:

| Feature | Description |
|--------|-------------|
| MedInc | Median income |
| HouseAge | Average house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block population |
| AveOccup | Average occupancy |
| Latitude | Latitude |
| Longitude | Longitude |

Target variable:

| Variable | Meaning |
|----------|--------|
| Price | Median house value |

---

## Machine Learning Pipeline

Raw Dataset  
↓  
Exploratory Data Analysis  
↓  
Feature Scaling  
↓  
Train-Test Split  
↓  
Linear Regression Model  
↓  
Model Evaluation  

---

## Feature Scaling

StandardScaler was used to normalize features.

Scaling ensures features have:

Mean(mu) = 0  
Standard deviation(sigma) = 1

This improves gradient descent convergence and prevents large-scale features from dominating training. Scaling is usually not required here as the model is linear and not distance based, but here scaling smoothens the optimization hence it is used. 

---

## Linear Regression Intuition

Linear regression models the relationship between features and target values using a linear equation:

y = wᵀx + b

Where:

x → input features  
w → model weights  
b → bias

The model learns weights that minimize **Mean Squared Error (MSE)**.(Loss Fn)

---

---

## ⏱ Time Complexity

Training complexity for Linear Regression depends on solving the **least squares optimization problem**.

The analytical solution is given by the **Normal Equation:**

$$w = (X^TX)^{-1} X^Ty$$

| Symbol | Meaning |
|---|---|
| X | Feature matrix |
| y | Target vector |
| w | Model weights |

**Main computational steps:**

**1. Computing the matrix multiplication XᵀX**
```
O(n × d²)
```

**2. Matrix inversion of (XᵀX)**
```
O(d³)
```

**3. Multiplying with Xᵀy**
```
O(n × d)
```

**Overall training complexity:**
```
O(n d² + d³)
```

| Variable | Meaning |
|---|---|
| n | Number of samples |
| d | Number of features |

Since most real-world datasets have **n >> d**, the dominant term is typically:
```
O(n d²)
```

---

## ⏱ Prediction Complexity

Prediction requires computing the **dot product** between the feature vector and learned weights:

$$\hat{y} = w^Tx + b$$

**Prediction complexity per sample:**
```
O(d)
```

**For n samples:**
```
O(n × d)
```

---

## 💾 Space Complexity

Linear Regression stores **one weight per feature** and a bias term.
```
O(d)
```
 **d** -> is the number of features.

## Evaluation Metrics

| Metric | Meaning |
|------|--------|
MSE | Mean squared error |
RMSE | Root mean squared error |
R² | Proportion of variance explained |

---

# Results 

| Metric | Value |
|------|------|
RMSE | ~0.74 |
R² | ~0.57 |

---

## Key Learnings

• Importance of feature scaling for linear models  
• Understanding regression evaluation metrics  
• Visualizing model predictions vs actual values  
• Building structured ML pipelines

---
