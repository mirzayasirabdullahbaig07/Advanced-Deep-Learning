# ====================================================================================================
# FORWARD PROPAGATION IN NEURAL NETWORKS
# ====================================================================================================

# ---------------------------------------------------------------
# What is Forward Propagation?
# ---------------------------------------------------------------
# Forward Propagation is the **core mechanism** through which a neural network
# takes an input, passes it through its layers (neurons + activations),
# and produces an output (prediction).
#
# It is called “forward” because the data moves **forward** through the network —
# from input → hidden layers → output — without any feedback (no error correction yet).
#
# During forward propagation:
#   1. Each neuron receives input data.
#   2. The neuron computes a weighted sum: Z = W*X + b
#   3. An activation function is applied to introduce non-linearity: A = f(Z)
#   4. The result (A) becomes input for the next layer.
#
# In the end, the final layer outputs predictions ŷ (y-hat).
#
# ---------------------------------------------------------------
# When to Use Forward Propagation
# ---------------------------------------------------------------
# You use forward propagation every time you:
#    Make predictions from a trained neural network
#    Compute the output before calculating loss in training
#    Run inference on unseen data (testing phase)
#    Initialize the process before backward propagation (training phase)
#
# Example:
#   - In training → forward pass + loss + backward pass
#   - In prediction → only forward pass

# ---------------------------------------------------------------
# Benefits & Advantages of Forward Propagation
# ---------------------------------------------------------------
#  Converts complex input data into meaningful predictions.
#  Helps the model learn hidden representations of data.
#  Works with any network size (small MLPs → deep CNNs → Transformers).
#  It’s computationally efficient and parallelizable on GPUs.
#  Provides foundation for backpropagation (gradient computation).

# ---------------------------------------------------------------
#  Maths Behind Forward Propagation
# ---------------------------------------------------------------
# For a simple 2-layer neural network:
#
# Layer 1 (Hidden Layer):
#     Z₁ = W₁·X + b₁
#     A₁ = f(Z₁)       → activation function (ReLU, sigmoid, etc.)
#
# Layer 2 (Output Layer):
#     Z₂ = W₂·A₁ + b₂
#     A₂ = f(Z₂) = ŷ   → predicted output
#
# Here:
#     W → weight matrix
#     b → bias vector
#     f → activation function
#     ŷ → network output (prediction)
#
# ---------------------------------------------------------------
# Example (Step-by-Step Forward Propagation)
# ---------------------------------------------------------------

import numpy as np

# Suppose we are building a small neural network for binary classification.
# We have:
#   - 3 input features
#   - 2 neurons in hidden layer
#   - 1 neuron in output layer

# Step 1: Define input data (X) and true output (y)
X = np.array([[0.5, 0.8, 0.2]])  # shape (1, 3)
y_true = np.array([[1]])         # target label

# Step 2: Initialize weights and biases randomly
np.random.seed(42)
W1 = np.random.randn(3, 2)   # weights for input → hidden (3 inputs → 2 neurons)
b1 = np.random.randn(1, 2)   # biases for hidden layer (1x2)
W2 = np.random.randn(2, 1)   # weights for hidden → output (2 neurons → 1 output)
b2 = np.random.randn(1, 1)   # bias for output layer (1x1)

# Step 3: Define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Step 4: Forward Propagation Calculations

# Hidden layer computations
Z1 = np.dot(X, W1) + b1          # Linear combination
A1 = relu(Z1)                    # Apply ReLU activation

# Output layer computations
Z2 = np.dot(A1, W2) + b2         # Linear combination
A2 = sigmoid(Z2)                 # Apply Sigmoid activation for binary output

# Final prediction
y_pred = A2

print("Input (X):", X)
print("Hidden layer linear output (Z1):", Z1)
print("Hidden layer activated output (A1):", A1)
print("Output layer linear output (Z2):", Z2)
print("Predicted output (ŷ):", y_pred)

# ---------------------------------------------------------------
# Explanation of the Flow
# ---------------------------------------------------------------
# 1 Input Layer:
#     - Takes the input vector X = [x1, x2, x3]
#
# 2 Hidden Layer:
#     - Each neuron computes a weighted sum:
#         Z1 = W1·X + b1
#     - ReLU removes negative values, introducing non-linearity.
#
# 3 Output Layer:
#     - Combines hidden outputs using another set of weights:
#         Z2 = W2·A1 + b2
#     - Sigmoid converts this into a probability (between 0 and 1).
#
# 4 Final Output (ŷ):
#     - The value represents the model’s confidence in class 1.
#
# ---------------------------------------------------------------
# Suppose (for visualization):
# ---------------------------------------------------------------
# X = [0.5, 0.8, 0.2]
#
# Hidden layer:
#   Z1 = (0.5*w11 + 0.8*w21 + 0.2*w31) + b1
#   A1 = ReLU(Z1)
#
# Output layer:
#   Z2 = (A1_1*w12 + A1_2*w22) + b2
#   A2 = Sigmoid(Z2) → gives probability of class = 1
#
# ---------------------------------------------------------------
# What Happens During Forward Propagation
# ---------------------------------------------------------------
# - Each neuron transforms input into an intermediate representation.
# - The network "learns" weights (W) and biases (b) that reduce the error.
# - During prediction, forward propagation uses those learned parameters
#   to generate outputs without any weight updates.
#
# ---------------------------------------------------------------
# During Training (Full Process)
# ---------------------------------------------------------------
# Step 1: Forward Propagation (compute output ŷ)
# Step 2: Compute Loss (e.g., Binary Cross-Entropy)
# Step 3: Backward Propagation (compute gradients)
# Step 4: Update weights using gradient descent
# Step 5: Repeat for many epochs until convergence
#
# ---------------------------------------------------------------
# Benefits of Forward Propagation
# ---------------------------------------------------------------
#  Converts inputs to outputs efficiently
#  Enables deep feature learning layer by layer
#  Works universally for any network architecture
#  Foundation for backpropagation and model optimization
#  Parallelizable (especially on GPUs)

# ---------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------
#  Does not learn on its own — needs backward propagation for updates
#  Computationally heavy for very deep networks
#  Can suffer from vanishing/exploding gradients (in deep nets)
#  Depends on proper weight initialization & activation choice

# ---------------------------------------------------------------
# Summary Formula (for L layers)
# ---------------------------------------------------------------
# For layer l:
#     Z[l] = W[l] * A[l-1] + b[l]
#     A[l] = f(Z[l])
#
# Repeat this until the last layer (L):
#     A[L] = ŷ
#
# ---------------------------------------------------------------
# Final Note
# ---------------------------------------------------------------
# Forward propagation is the **heart of prediction** in neural networks.
# It is where learning "shows its results" — the output depends on the
# trained weights from previous learning (backpropagation).
#
# In simple words:
#     Forward Propagation = Data Flow
#     Backward Propagation = Learning Flow
