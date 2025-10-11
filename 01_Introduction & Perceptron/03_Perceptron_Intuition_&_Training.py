# ------------------------------------------------------------
# PERCEPTRON — Intuition & Training
# ------------------------------------------------------------

# ------------------------------------------------------------
# What is a Perceptron?
# ------------------------------------------------------------
# The perceptron is the simplest type of artificial neuron (a single-layer neural network).
# It takes input features, multiplies them by weights, adds a bias, and applies an activation function.
# It acts as a **binary classifier** — separates data into two classes using a straight line (2D), plane (3D), or hyperplane (nD).

# ------------------------------------------------------------
# Key Concepts
# ------------------------------------------------------------
# Weight (w): 
#     - The importance assigned to each input feature.
#     - Adjusted during training to minimize errors.

# Bias (b): 
#     - A constant term added to shift the decision boundary.
#     - Helps the model make correct predictions even when all inputs are zero.

# Parameters:
#     - Model’s tunable values: weights (w1, w2, ...) and bias (b).

# Dot Product:
#     - Mathematical operation that combines inputs and weights.
#     - Formula: w·x = (w1*x1 + w2*x2 + ... + wn*xn)

# Activation Function:
#     - Decides whether the neuron should “fire” or not.
#     - In perceptron, we use a **step function**:
#         f(x) = 1 if (w·x + b) ≥ 0 else 0

# ------------------------------------------------------------
# Perceptron Equation
# ------------------------------------------------------------
# y = f(w·x + b)
# For 2D → forms a line
# For 3D → forms a plane
# For nD → forms a hyperplane
# ➤ Used only for **linearly separable datasets**.

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
# Benefits:
# - Simple and easy to implement
# - Fast for small datasets
# - Works well on linear data

# Uses:
# - Binary classification (e.g., pass/fail, spam/not spam)
# - Foundation for complex neural networks

# Disadvantages:
# - Fails on nonlinear data
# - Cannot learn XOR or curved boundaries

# Where to Use:
# - When data is roughly linearly separable

# ------------------------------------------------------------
# Example: Training & Prediction
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

# Load dataset
df = pd.read_csv('placement.csv')
print(df.shape)
print(df.head())

# Scatter plot of data
sns.scatterplot(x='cgpa', y='resume_score', hue='placed', data=df)

# Split into features and target
X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

# Initialize and train Perceptron
p = Perceptron()
p.fit(X, y)

# Model parameters
print("Weights:", p.coef_)
print("Bias:", p.intercept_)

# Plot decision boundary
plot_decision_regions(X.values, y.values, clf=p, legend=2)
plt.xlabel('CGPA')
plt.ylabel('Resume Score')
plt.title('Perceptron Decision Boundary')
plt.show()

# ------------------------------------------------------------
# Concept Recap:
# ------------------------------------------------------------
# - Perceptron finds a line/plane to separate classes.
# - Works only for linearly separable datasets.
# - Equation: w1*x1 + w2*x2 + b = 0 → forms a decision boundary.
