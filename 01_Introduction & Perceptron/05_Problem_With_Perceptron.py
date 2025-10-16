#  Perceptron: Understanding Linear Separability using Logic Gates
# ------------------------------------------------------------

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

# ------------------------------------------------------------
#  Create datasets for AND, OR, and XOR gates
# ------------------------------------------------------------

# OR gate truth table
or_data = pd.DataFrame({
    'input1': [1, 1, 0, 0],
    'input2': [1, 0, 1, 0],
    'output': [1, 1, 1, 0]
})

# AND gate truth table
and_data = pd.DataFrame({
    'input1': [1, 1, 0, 0],
    'input2': [1, 0, 1, 0],
    'output': [1, 0, 0, 0]
})

# XOR gate truth table
xor_data = pd.DataFrame({
    'input1': [1, 1, 0, 0],
    'input2': [1, 0, 1, 0],
    'output': [0, 1, 1, 0]
})

# ------------------------------------------------------------
#  Visualize input and output patterns for each gate
# ------------------------------------------------------------

# AND Gate
sns.scatterplot(x=and_data['input1'], y=and_data['input2'], hue=and_data['output'], s=200)
plt.title("AND Gate Data")
plt.show()

# OR Gate
sns.scatterplot(x=or_data['input1'], y=or_data['input2'], hue=or_data['output'], s=200)
plt.title("OR Gate Data")
plt.show()

# XOR Gate
sns.scatterplot(x=xor_data['input1'], y=xor_data['input2'], hue=xor_data['output'], s=200)
plt.title("XOR Gate Data")
plt.show()

# ------------------------------------------------------------
#  Train Perceptron models for each gate
# ------------------------------------------------------------

clf_and = Perceptron()
clf_or = Perceptron()
clf_xor = Perceptron()

clf_and.fit(and_data.iloc[:, 0:2].values, and_data.iloc[:, -1].values)
clf_or.fit(or_data.iloc[:, 0:2].values, or_data.iloc[:, -1].values)
clf_xor.fit(xor_data.iloc[:, 0:2].values, xor_data.iloc[:, -1].values)

# ------------------------------------------------------------
#  Check model parameters (weights and bias)
# ------------------------------------------------------------

print("AND Gate Weights:", clf_and.coef_)
print("AND Gate Intercept:", clf_and.intercept_)

print("OR Gate Weights:", clf_or.coef_)
print("OR Gate Intercept:", clf_or.intercept_)

print("XOR Gate Weights:", clf_xor.coef_)
print("XOR Gate Intercept:", clf_xor.intercept_)

# ------------------------------------------------------------
#  Plot decision boundary for AND gate
# ------------------------------------------------------------

x = np.linspace(-1, 1, 5)
y = -x + 1  # approximate line for visualization
plt.plot(x, y)
sns.scatterplot(x=and_data['input1'], y=and_data['input2'], hue=and_data['output'], s=200)
plt.title("AND Gate Decision Boundary (Linear)")
plt.show()

# ------------------------------------------------------------
#  Plot decision boundary for OR gate
# ------------------------------------------------------------

x1 = np.linspace(-1, 1, 5)
y1 = -x1 + 0.5  # approximate line for visualization
plt.plot(x1, y1)
sns.scatterplot(x=or_data['input1'], y=or_data['input2'], hue=or_data['output'], s=200)
plt.title("OR Gate Decision Boundary (Linear)")
plt.show()

# ------------------------------------------------------------
#  Try decision boundary for XOR gate
# ------------------------------------------------------------

plot_decision_regions(xor_data.iloc[:, 0:2].values, xor_data.iloc[:, -1].values, clf=clf_xor, legend=2)
plt.title("XOR Gate - Perceptron Decision Regions")
plt.show()

# ------------------------------------------------------------
#  Problem with Perceptron
# ------------------------------------------------------------

#  Perceptron works only when the data is linearly separable.
#    That means a single straight line (or hyperplane) can divide the classes.

#  AND and OR gates are linearly separable — 
#    you can draw a straight line to separate 0s and 1s.

#  XOR gate is *not linearly separable* —
#    it requires a non-linear decision boundary (like a curve or multiple lines).
#    A single perceptron cannot model XOR.

#  To solve XOR, we need multiple perceptrons stacked together → i.e. a Neural Network.

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
#  Perceptron succeeds on: AND, OR gates
#  Perceptron fails on: XOR gate
#  Reason: XOR requires non-linear separation → use MLP (Multi-Layer Perceptron)
# ------------------------------------------------------------
