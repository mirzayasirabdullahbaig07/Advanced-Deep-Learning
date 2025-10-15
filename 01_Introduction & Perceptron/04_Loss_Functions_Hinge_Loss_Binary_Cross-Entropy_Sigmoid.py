# =========================================================
# Loss Functions in Machine Learning
# =========================================================

# Overview:
# -------------
# A **Loss Function** (or Cost Function) quantifies *how well or poorly*
# a machine learning model performs. It measures the difference between 
# the model’s predictions and the actual ground truth labels (y_true).

# We use loss functions to:
# Judge model performance numerically  
# Guide optimization algorithms (like Gradient Descent)  
# Help the model *learn* by minimizing this loss value  

# Common loss functions for classification:
# - Hinge Loss (used in SVM)
# - Binary Cross-Entropy (used in Logistic Regression)
# - Sigmoid Activation (used in probabilistic models)

# -----------------------------------------------------------
# Problem with Perceptron Trick
# -----------------------------------------------------------
# - The **Perceptron Algorithm** updates weights only based on misclassifications.
# - It doesn’t measure *how wrong* a prediction is — it’s binary: right or wrong.
# - So, we can’t *quantify* how well the model performs.
# - It also has **convergence issues** if the data is not linearly separable.
# → That’s why we use differentiable loss functions like **Hinge Loss** or **Cross-Entropy**.

# =========================================================


# -------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, hinge_loss

# -------------------------------------------------------
# CREATE A SIMPLE BINARY CLASSIFICATION DATASET
# -------------------------------------------------------
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=15,
    random_state=41,
)

# Visualize the dataset
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=100, edgecolors='k')
plt.title("Binary Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# =======================================================
# HINGE LOSS
# =======================================================
"""
Definition:
--------------
Hinge Loss is primarily used in **Support Vector Machines (SVM)**.

Formula:
---------
L(y, ŷ) = max(0, 1 - y * ŷ)

Where:
- y ∈ {-1, +1} → true label
- ŷ → predicted value (not probability, but raw score)
- If y and ŷ have the same sign and margin ≥ 1 → loss = 0
- Else → penalized based on how far it is from the correct margin

Benefit:
- Encourages a margin (confidence zone) between classes.
- Makes SVMs robust against small classification errors.

When to Use:
- When you care about classification *margin*.
- Common in Linear SVMs or Maximum Margin Classifiers.
"""

# Manual example
y_true = np.array([1, -1, 1, -1])
y_pred = np.array([0.9, -0.8, 0.2, 0.1])  # predicted scores
hinge = np.maximum(0, 1 - y_true * y_pred)
print("Manual Hinge Loss:", np.mean(hinge))

# Using sklearn
print("Sklearn Hinge Loss:", hinge_loss(y_true, y_pred))


# =======================================================
# BINARY CROSS-ENTROPY (LOG LOSS)
# =======================================================
"""
Definition:
--------------
Binary Cross-Entropy measures the dissimilarity between true labels
and predicted probabilities.

Formula:
---------
L(y, ŷ) = -[ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]

Where:
- y ∈ {0, 1}
- ŷ ∈ [0, 1] (predicted probability)
- If ŷ is close to y → small loss
- If ŷ is far from y → large loss

Benefits:
- Differentiable → good for gradient descent.
- Encourages probabilistic predictions.
- Penalizes confident wrong predictions heavily.

When to Use:
- Logistic Regression
- Neural Networks (Binary Classification)
"""

# Manual example
y_true = np.array([1, 0, 1, 0])
y_pred_prob = np.array([0.9, 0.2, 0.8, 0.3])
bce_loss = -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))
print("Manual Binary Cross-Entropy Loss:", bce_loss)

# Using sklearn
print("Sklearn Log Loss:", log_loss(y_true, y_pred_prob))


# =======================================================
# SIGMOID FUNCTION
# =======================================================
"""
Definition:
--------------
Sigmoid converts a real-valued number into a probability (0 → 1).

Formula:
---------
σ(z) = 1 / (1 + e^(-z))

Benefits:
- Converts raw scores into probabilities.
- Useful in Logistic Regression and Neural Networks.

When to Use:
- Output layer for binary classification models.
"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z), color='red')
plt.title("Sigmoid Function")
plt.xlabel("z (raw score)")
plt.ylabel("σ(z) (probability)")
plt.grid(True)
plt.show()


# =======================================================
# Real-World Use Cases
# =======================================================
"""
Hinge Loss:
--------------
- Used in spam detection using SVM
- Used in face verification (margin-based learning)
- Used for document classification

Binary Cross-Entropy:
------------------------
- Used in logistic regression for email classification
- Used in binary image classification
- Used in deep learning models (e.g., binary sentiment analysis)

Sigmoid:
-----------
- Converts logits → probabilities
- Used in the last layer of binary neural networks
- Helps interpret outputs in terms of confidence
"""

# =======================================================
# Small Practical Demo with SGDClassifier
# =======================================================
"""
We’ll compare Linear Classifiers trained using:
1. Hinge Loss  → SVM style
2. Log Loss    → Logistic Regression style
"""

# Model with hinge loss (SVM)
svm_model = SGDClassifier(loss='hinge', max_iter=1000, random_state=42)
svm_model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.7, linestyles=['-'])
plt.title("Hinge Loss (SVM-like Classifier)")
plt.show()
