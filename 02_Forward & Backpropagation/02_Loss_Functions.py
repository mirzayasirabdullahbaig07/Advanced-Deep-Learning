#  LOSS FUNCTION IN DEEP LEARNING

# A Loss Function is a mathematical method used to evaluate
# how well your model is performing.
# It measures the difference between predicted and actual values.

#  If loss value is HIGH → Model performance is POOR
#  If loss value is LOW → Model performance is GOOD

#  How to reduce loss?
# By updating model parameters (weights and biases)
# using optimization algorithms like Gradient Descent.

#  Why do we need a loss function?
# "You can’t improve what you can’t measure." – Peter Drucker


# ------------------------------------------------------------
#  TYPES OF LOSS FUNCTIONS IN DEEP LEARNING
# ------------------------------------------------------------

# 1 REGRESSION LOSSES
#    - Mean Squared Error (MSE)
#    - Mean Absolute Error (MAE)
#    - Huber Loss

# 2 CLASSIFICATION LOSSES
#    - Binary Cross Entropy
#    - Categorical Cross Entropy
#    - Hinge Loss

# 3 AUTOENCODERS
#    - KL Divergence

# 4 GANs (Generative Adversarial Networks)
#    - Discriminator Loss
#    - Generator Loss
#    - Min–Max Loss

# 5 OBJECT DETECTION
#    - Focal Loss

# 6 EMBEDDINGS / METRIC LEARNING
#    - Triplet Loss


# ------------------------------------------------------------
#  LOSS FUNCTION vs COST FUNCTION
# ------------------------------------------------------------

# Loss Function → measures error for a SINGLE training example.
# Cost Function → measures the AVERAGE error across the entire dataset (batch).


# ------------------------------------------------------------
#  MEAN SQUARED ERROR (MSE) / L2 LOSS / SQUARED LOSS
# ------------------------------------------------------------

# Formula: MSE = (1/n) * Σ (y_true - y_pred)²

# Why square the error?
# → To make all values positive (no negative errors)
# → It forms a quadratic function — smooth and differentiable

# Intuition:
# - If points are far from the line → large updates in weights & bias
# - If points are near the line → small updates

#  Advantages:
# - Easy to interpret
# - Always differentiable
# - Only one global minimum (convex)

#  Disadvantages:
# - Sensitive to outliers (large errors dominate the loss)
# - Squared error units differ from target variable units


# ------------------------------------------------------------
#  MEAN ABSOLUTE ERROR (MAE) / L1 LOSS
# ------------------------------------------------------------

# Formula: MAE = (1/n) * Σ |y_true - y_pred|
# (Square replaced by absolute value)

#  Advantages:
# - Easy to understand
# - Same unit as target variable
# - Robust to outliers

#  Disadvantages:
# - Not differentiable at zero (requires subgradient)
# - Can be slower to converge

#  Quick tip:
# - If no outliers → use MSE
# - If outliers exist → use MAE


# ------------------------------------------------------------
#  HUBER LOSS
# ------------------------------------------------------------

# It combines both MSE and MAE behaviors:
# - For small errors → acts like MSE
# - For large errors (outliers) → acts like MAE
# - Controlled by a threshold δ (delta)

#  Advantages:
# - Robust to outliers
# - Differentiable everywhere
# - Best of both MSE and MAE


# ------------------------------------------------------------
#  BINARY CROSS ENTROPY (for binary classification)
# ------------------------------------------------------------

# Used when there are two classes (Yes/No, 0/1)
# Activation Function: Sigmoid

# Formula for loss function:
#     L = - [y * log(y_pred) + (1 - y) * log(1 - y_pred)]

# Formula for cost function:
#     J = -(1/m) * Σ [y * log(y_pred) + (1 - y) * log(1 - y_pred)]

#  Advantages:
# - Differentiable and works well with sigmoid output
# - Interpretable as probability loss

#  Disadvantages:
# - Can have multiple local minima
# - Sensitive to poor initialization


# ------------------------------------------------------------
#  CATEGORICAL CROSS ENTROPY (for multi-class classification)
# ------------------------------------------------------------

# Used when there are more than two classes (e.g. Yes, No, Maybe)
# Activation Function: Softmax

# Formula (single example):
#     L = - Σ [y_true * log(y_pred)]

# Formula (all examples):
#     J = -(1/m) * Σ Σ [y_true * log(y_pred)]

#  Advantages:
# - Works perfectly for multi-class classification
# - Probabilistic interpretation

#  Disadvantages:
# - Sensitive to incorrect labels or noisy data
# - Requires softmax output normalization


# ------------------------------------------------------------
# SUMMARY CHEAT SHEET
# ------------------------------------------------------------
# REGRESSION       → MSE, MAE, Huber
# CLASSIFICATION   → BCE, CCE, Hinge
# AUTOENCODER      → KL Divergence
# GANs             → Discriminator/Generator Loss
# DETECTION        → Focal Loss
# EMBEDDINGS       → Triplet Loss
