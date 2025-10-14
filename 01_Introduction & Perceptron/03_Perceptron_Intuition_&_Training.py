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

# ===================================================================
# PERCEPTRON TRAINING — Step-by-Step Explanation
# ===================================================================

# ------------------------------------------------------------
# Goal:
# ------------------------------------------------------------
# Train a perceptron model to find a straight line (decision boundary)
# that separates two classes (binary classification).

# ------------------------------------------------------------
# Concept Behind Training:
# ------------------------------------------------------------
# - The perceptron tries to find weights (w1, w2, ...) and bias (b)
#   that correctly separate the data into two regions:
#       Positive region: (w·x + b) > 0
#       Negative region: (w·x + b) < 0
#       On the line:      (w·x + b) = 0
#
# - We adjust (update) the weights repeatedly until all points
#   are correctly classified or until we reach the maximum number of iterations (epochs).

# ------------------------------------------------------------
# Perceptron Update Rule:
# ------------------------------------------------------------
# weights = weights + learning_rate * (y_true - y_pred) * x
#
# Explanation:
# - If a **positive point** is predicted as negative → ADDITION
# - If a **negative point** is predicted as positive → SUBTRACTION
# - The learning rate (lr) controls how big the update steps are.
# - Repeat this process using a loop for many epochs (e.g., 1000 times).

# ------------------------------------------------------------
# Dataset Creation (using sklearn)
# ------------------------------------------------------------
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# Generate a simple, linearly separable dataset
X, y = make_classification(
    n_samples=100,          # number of data points
    n_features=2,           # number of input features (x1, x2)
    n_informative=1,        # number of useful features
    n_redundant=0,          # no duplicate features
    n_classes=2,            # two output classes (0 and 1)
    n_clusters_per_class=1, # one cluster per class
    random_state=41,        # reproducibility
    hypercube=False,
    class_sep=10            # distance between the classes
)

# Visualize the dataset
plt.figure(figsize=(10,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=100)
plt.title("Linearly Separable Data for Perceptron Training")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ------------------------------------------------------------
# Perceptron Function Definition
# ------------------------------------------------------------
def perceptron(X, y):
    # Step 1: Insert bias term (x0 = 1 for every sample)
    X = np.insert(X, 0, 1, axis=1)

    # Step 2: Initialize weights with ones
    weights = np.ones(X.shape[1])

    # Step 3: Set learning rate
    lr = 0.1

    # Step 4: Train for 1000 iterations (epochs)
    for i in range(1000):
        # Pick a random sample index
        j = np.random.randint(0, 100)

        # Step 5: Calculate prediction using dot product
        y_hat = step(np.dot(X[j], weights))

        # Step 6: Update weights using perceptron learning rule
        weights = weights + lr * (y[j] - y_hat) * X[j]

    # Step 7: Return bias and coefficients
    return weights[0], weights[1:]


# ------------------------------------------------------------
# Step Activation Function
# ------------------------------------------------------------
def step(z):
    """Return 1 if z > 0, else 0."""
    return 1 if z > 0 else 0


# ------------------------------------------------------------
# Train the Model
# ------------------------------------------------------------
intercept_, coef_ = perceptron(X, y)
print("Weights:", coef_)
print("Bias:", intercept_)

# ------------------------------------------------------------
# Plot the Decision Boundary
# ------------------------------------------------------------
# Equation of line: w1*x1 + w2*x2 + b = 0
# Rearranged as: x2 = -(w1/w2)*x1 - (b/w2)

m = -(coef_[0] / coef_[1])       # slope of the line
b = -(intercept_ / coef_[1])     # intercept

x_input = np.linspace(-3, 3, 100)
y_input = m * x_input + b

# Visualize the line separating the two classes
plt.figure(figsize=(10,6))
plt.plot(x_input, y_input, color='red', linewidth=3, label='Decision Boundary')
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=100)
plt.ylim(-3, 2)
plt.title("Perceptron Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# ------------------------------------------------------------
# Summary:
# ------------------------------------------------------------
# - A perceptron separates two classes using a linear equation.
# - Weights and bias are updated repeatedly using the perceptron rule.
# - Positive region → (w·x + b) > 0
# - Negative region → (w·x + b) < 0
# - On the line → (w·x + b) = 0
# - Learning rate controls the step size of updates.
# - The process continues until convergence or max epochs reached.
# ===================================================================
