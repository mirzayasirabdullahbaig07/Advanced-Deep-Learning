# ------------------------------------------------------------
# Multi-Layer Perceptron (MLP) — Explained
# ------------------------------------------------------------

# 1. What is a Multi-Layer Perceptron?

# A Multi-Layer Perceptron (MLP) is a type of feed-forward neural network.
# It consists of multiple layers of neurons:
#   - Input Layer  (L0)
#   - One or more Hidden Layers (L1, L2, ...)
#   - Output Layer (L3)
#
# Each neuron in one layer is connected to all neurons in the next layer
# using weighted connections (W).
# These weights are the learnable parameters during training.

# ------------------------------------------------------------
# 2. Components of an MLP

# Example: (from your diagram)
#   Input layer (4 features): x1, x2, x3, x4
#   Hidden Layer 1: 3 neurons (each neuron has 4 weights + 1 bias)
#   Hidden Layer 2: 2 neurons (each has 3 weights + 1 bias)
#   Output Layer: 1 neuron (2 weights + 1 bias)

# Total trainable parameters = weights + biases
#   = (4*3 + 3) + (3*2 + 2) + (2*1 + 1)
#   = 15 + 8 + 3 = 26 trainable parameters

# ------------------------------------------------------------
# 3. How Bias Works (and how it’s “donated”)
#
# Each neuron computes a weighted sum of its inputs and adds a bias term:
#      z = (w1*x1 + w2*x2 + w3*x3 + ... + wn*xn) + b
#
# The bias (b) is like an intercept term in linear regression.
# It allows the neuron to shift the activation function
# up or down, making the model more flexible.

# Without a bias, all neurons must pass through the origin (0,0),
# which limits the model’s ability to learn complex patterns.

# Bias values are also trainable parameters — updated during backpropagation.

# ------------------------------------------------------------
# 4. How an MLP Learns

# Step 1: Forward Propagation
#   - Inputs are passed layer by layer.
#   - Each neuron applies: 
#       z = W·X + b
#       a = activation(z)
#     where activation() could be sigmoid, ReLU, or tanh.
#
# Step 2: Loss Calculation
#   - Compare predicted output ŷ with actual output y.
#
# Step 3: Backpropagation
#   - Compute gradients (∂Loss/∂W, ∂Loss/∂b)
#   - Update weights and biases using gradient descent.

# ------------------------------------------------------------
# 5. Benefits of MLPs

#  Can model non-linear relationships (unlike simple perceptrons)
#  Can solve problems like XOR which are not linearly separable
#  Works well for classification and regression tasks
#  Forms the foundation of modern Deep Learning architectures

# ------------------------------------------------------------
# 6. Intuition Recap (as per your diagram)

# Layer L0 → L1:
#   - 4 input neurons connected to 3 hidden neurons (12 weights)
#   - Each hidden neuron adds 1 bias → 3 biases
#
# Layer L1 → L2:
#   - 3 neurons connected to 2 hidden neurons (6 weights)
#   - Each hidden neuron adds 1 bias → 2 biases
#
# Layer L2 → L3:
#   - 2 neurons connected to 1 output neuron (2 weights)
#   - Output neuron adds 1 bias → 1 bias
#
# Total = 26 trainable parameters (weights + biases)

# ------------------------------------------------------------
# Summary

# - MLPs are networks with multiple hidden layers.
# - Bias helps neurons learn flexible decision boundaries.
# - More layers → more ability to learn complex, non-linear patterns.
# - The cost: higher computation, risk of overfitting, need for more data.
# ------------------------------------------------------------
