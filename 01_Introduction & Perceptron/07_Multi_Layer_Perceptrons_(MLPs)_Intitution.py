# ============================================================
# Multi-Layer Perceptrons (MLPs) — Intuition & Concepts
# ============================================================

# ------------------------------------------------------------
# 1 Why MLPs?
# ------------------------------------------------------------
# A Single-Layer Perceptron (SLP) can only handle *linearly separable* data,
# such as AND and OR gates.
# But XOR is *non-linear*, meaning no straight line can separate its classes.
# To solve this, we introduce *hidden layers* — forming a Multi-Layer Perceptron (MLP).
# Hidden layers allow the network to learn *non-linear decision boundaries*.

# ------------------------------------------------------------
# 2 How MLP Creates Non-Linear Decision Boundaries
# ------------------------------------------------------------
# Structure:
# - Input Layer: Takes input features (x1, x2, ...)
# - Hidden Layers: Apply non-linear transformations (activation functions)
# - Output Layer: Produces the final prediction
#
# Each neuron computes:
#     z = w1*x1 + w2*x2 + b
#
# Then applies a non-linear activation function (e.g., Sigmoid):
#     a = 1 / (1 + e^(-z))
#
# Because of this non-linearity, the network can "bend" its decision boundary,
# creating curved or complex shapes that a simple perceptron cannot.

# ------------------------------------------------------------
# 3 How It Works (Step-by-Step Flow)
# ------------------------------------------------------------
# Step 1: Forward Propagation
#   - Input → Hidden layer (Linear + Non-linear)
#   - Hidden → Output layer
#   - Compute prediction
#
# Step 2: Loss Calculation
#   - Compare predicted output with actual output using a loss function
#     (e.g., Binary Cross-Entropy)
#
# Step 3: Backward Propagation
#   - Compute gradients (partial derivatives)
#   - Update weights using gradient descent:
#       w = w - η * ∂L/∂w
#
# The process repeats until the model minimizes the loss.

# ------------------------------------------------------------
# 4 Sigmoid Activation Function
# ------------------------------------------------------------
# Formula:
#     σ(z) = 1 / (1 + e^(-z))
#
# Derivative:
#     σ'(z) = σ(z) * (1 - σ(z))
#
# Benefits:
# Non-linear — captures complex relationships
# Smooth and differentiable — suitable for gradient descent
# Outputs between (0, 1) — interpretable as probabilities
#
# Disadvantages:
#  Vanishing gradients (when z is very large or very small)
#  Slower convergence in deep networks
#  Outputs not zero-centered (may slow learning)

# ------------------------------------------------------------
# 5 Real-World Examples
# ------------------------------------------------------------
# - Email Spam Detection → Non-linear relation between words & labels
# - Image Classification → Pixels form complex non-linear patterns
# - Stock Market Prediction → Price movements are non-linear

# ------------------------------------------------------------
# 6 Multiclass Classification
# ------------------------------------------------------------
# For multiple output classes (e.g., digits 0–9):
# - The output layer has *multiple neurons* (one per class)
# - Use *Softmax Activation*:
#       P(y_i) = e^(z_i) / Σ(e^(z_j))
# - Use *Cross-Entropy Loss* for training

# ------------------------------------------------------------
# 7 Adding Layers & Nodes
# ------------------------------------------------------------
# Concept                      | Effect
# ------------------------------------------------------------
# Adding more perceptrons       → Captures more features
# Adding hidden layers          → Enables hierarchical learning
# Adding output nodes           → Supports multi-class outputs
#
# Adding hidden layers allows the network to represent *non-linear* boundaries,
# but too many layers may cause overfitting or slow training.

# ------------------------------------------------------------
# 8 Visualization Example: XOR with MLP
# ------------------------------------------------------------
# The following example shows how an MLP captures the XOR pattern
# using a non-linear decision boundary.

"""
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Input (XOR data)
X = xor_data.iloc[:, 0:2].values
y = xor_data.iloc[:, -1].values

# MLP with one hidden layer of 4 neurons and sigmoid activation
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', max_iter=1000)
mlp.fit(X, y)

# Plot the decision boundary
plot_decision_regions(X, y, clf=mlp, legend=2)
plt.title("MLP Decision Boundary for XOR (Non-Linear)")
plt.show()
"""

# ------------------------------------------------------------
# 9 Summary Table
# ------------------------------------------------------------
# Concept               | Single Perceptron | Multi-Layer Perceptron
# ------------------------------------------------------------
# Layers                | 1                 | 2 or more
# Decision Boundary     | Linear            | Non-linear
# Activation Function   | Step              | Sigmoid / ReLU / Tanh
# Can Solve XOR?        | ❌                | ✅
# Learning Type         | Simple            | Deep (Backpropagation)

# ------------------------------------------------------------
# Key Takeaway:
# ------------------------------------------------------------
# Adding hidden layers + non-linear activations (like sigmoid)
# allows the MLP to learn non-linear decision boundaries,
# enabling it to solve complex problems that a single perceptron cannot.
# ------------------------------------------------------------
