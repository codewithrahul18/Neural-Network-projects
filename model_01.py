import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# ==========================================
# PART 1: The "From Scratch" Perceptron (Logic Gates)
# ==========================================
class CustomPerceptron:
    """
    A simple Perceptron classifier implemented from scratch using NumPy.
    """
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # 1. Initialize weights (n_features) and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Training Loop
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Calculate linear output: z = w.x + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._step_function(linear_output)

                # Perceptron Update Rule: w = w + lr * (y - y_hat) * x
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._step_function(linear_output)

    def _step_function(self, x):
        return np.where(x >= 0, 1, 0)

def visualize_boundary(X, y, model, title="Decision Boundary"):
    """Helper to visualize 2D decision boundaries"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green')))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(('red', 'green')))
    plt.title(title)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)

# ==========================================
# PART 2: Execution (Logic Gate Demo)
# ==========================================
print("--- 1. Training Custom Perceptron on AND Gate ---")

# AND Gate Data
X_gate = np.array([[0,0], [0,1], [1,0], [1,1]])
y_gate = np.array([0, 0, 0, 1]) # AND logic

# Train
custom_p = CustomPerceptron(learning_rate=0.1, n_iterations=10)
custom_p.fit(X_gate, y_gate)

# Test
print(f"Learned Weights: {custom_p.weights}")
print(f"Learned Bias: {custom_p.bias}")
print(f"Predictions for [0,0], [0,1], [1,0], [1,1]: {custom_p.predict(X_gate)}")

# Visualize
plt.figure(figsize=(6, 5))
visualize_boundary(X_gate, y_gate, custom_p, "Custom Perceptron: AND Gate")
plt.show()

# ==========================================
# PART 3: The "Pro" Scikit-Learn Pipeline
# ==========================================
print("\n--- 2. Training Scikit-Learn Perceptron on Complex Data ---")

# 1. Generate a harder dataset
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, 
                           n_redundant=5, n_classes=2, random_state=42)

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale Data (CRITICAL STEP usually missing in basic tutorials)
# Perceptrons converge much faster when data is centered and scaled.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create Model
clf = Perceptron(
    max_iter=1000,
    eta0=0.1,
    random_state=42,
    tol=1e-3,
    early_stopping=True, # Pro feature: Stop if validation score doesn't improve
    validation_fraction=0.1
)

# 5. Train
clf.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {acc:.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Check convergence
print(f"Converged in {clf.n_iter_} epochs.")