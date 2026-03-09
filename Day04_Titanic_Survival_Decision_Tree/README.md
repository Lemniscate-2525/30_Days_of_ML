# Titanic Survival Prediction

## Problem

Predict whether a passenger survived the Titanic disaster.

Binary classification problem.

Target variable:

Survived  
0 → Did not survive  
1 → Survived

Dataset size: 891 samples.

---

# Dataset

Titanic Dataset

https://www.kaggle.com/competitions/titanic/data

Used file:

train.csv

---

# Machine Learning Pipeline

1. Data Loading
2. Data Exploration
3. Handling Missing Values
4. Feature Engineering
5. Encoding Categorical Variables
6. Train Test Split
7. Decision Tree Training
8. Model Evaluation
9. Tree Visualization
10. Feature Importance Analysis

---

# Handling Missing Values

Age column contained missing values.

We replaced them with the median age.

Cabin had many missing values (~77%).

Instead of using the raw cabin value, the deck information was extracted.

Deck corresponds to the first letter of the cabin number.

Example:

C85 → Deck C

Missing values were labeled as "Unknown".

---

# Feature Encoding

Categorical features were converted to numeric values using one-hot encoding.

Example:

Embarked values:

S, C, Q

Converted to binary columns:

Embarked_C  
Embarked_Q

The first category was dropped to avoid multicollinearity.

---

# Decision Tree Model

The model used:

```
DecisionTreeClassifier(max_depth = 4)
```

Tree depth was limited to control overfitting.

Decision trees tend to grow very deep and memorize training data.

Restricting the depth improves generalization.

---

# Mathematical Concepts

## Entropy

Entropy measures impurity in a node.

Formula:

Entropy = − Σ p log₂(p)

If all samples belong to one class → entropy = 0.

If classes are evenly mixed → entropy = 1.

---

## Information Gain

Trees choose splits using Information Gain.

Information Gain =

Entropy(parent) − Weighted Entropy(children)

The split with highest information gain is chosen.

---

## Gini Impurity

Alternative impurity measure used by CART trees.

Gini = 1 − Σ(p²)

Lower Gini indicates purer nodes.

---

# Overfitting in Decision Trees

Decision Trees easily overfit because they can create very deep structures.

The tree continues splitting until leaf nodes become pure.

This may lead to rules that memorize the training dataset rather than learning general patterns.

Example of overfitting rule:

If PassengerID = 438 → Survived

Such rules do not generalize.

To control overfitting we limit tree depth.

---

# Model Evaluation

Metrics used:

Accuracy  
Precision  
Recall  
F1 Score  
Confusion Matrix

---

# Cross Validation

To ensure the model generalizes well and is not dependent on a single train-test split, 5-fold cross validation was performed.

The dataset is divided into 5 folds. The model is trained on 4 folds and validated on the remaining fold. This process repeats until every fold has served as the validation set.

Example output:

```
Cross Validation Scores: [0.80 0.83 0.79 0.82 0.81]
Mean CV Score: 0.81
Std Dev: 0.014
```

The mean score provides a more reliable estimate of model performance compared to a single train-test split.

# Model Visualization

## Decision Tree Structure

![Decision Tree](images/tree.png)

The decision tree visualization helps understand how the model splits features such as sex, passenger class, and fare to determine survival probability.

---

## Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

The confusion matrix shows the number of correct and incorrect predictions for each class.

# Feature Importance

Decision Trees provide feature importance scores.

Important predictors typically include:

Sex  
Pclass  
Fare  
Age

Gender is usually the strongest predictor of survival.

---

# Time Complexity

Training complexity:

O(F * N log N)

Where

F = number of features  
N = number of samples

The algorithm evaluates possible splits across features and sorts feature values.

Prediction complexity:

O(depth)

Each prediction follows a path from root to leaf.

---

# Space Complexity

Space complexity depends on number of nodes in the tree.

Worst case:

O(N)

When tree grows very deep.

---

# Key Learning Points

Handling missing values  
Categorical encoding  
Decision tree structure  
Overfitting control  
Model interpretability
