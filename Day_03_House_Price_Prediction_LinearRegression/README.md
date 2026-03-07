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

Mean = 0  
Standard deviation = 1

This improves gradient descent convergence and prevents large-scale features from dominating training.

---

## Linear Regression Intuition

Linear regression models the relationship between features and target values using a linear equation:

y = wᵀx + b

Where:

x → input features  
w → model weights  
b → bias

The model learns weights that minimize **Mean Squared Error (MSE)**.

---

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
