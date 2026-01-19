import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving the model (Feature 4)

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. DATA LOADING (Real World Data)
# ==========================================
print("--- 1. Loading Medical Dataset ---")
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names  # ['malignant', 'benign']

# Convert to DataFrame for a quick look (Pro habit)
df = pd.DataFrame(X, columns=data.feature_names)
df['diagnosis'] = y
print(f"Dataset Shape: {df.shape}")
print(f"Features used: {list(data.feature_names[:4])} ... and more.")
print("-" * 30)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. PIPELINE & HYPERPARAMETER TUNING (Feature 1 & 2)
# ==========================================
print("--- 2. Building Production Pipeline & Tuning ---")

# A Pipeline ensures we apply the exact same scaling to Training and New Data
pipe = Pipeline([
    ('scaler', StandardScaler()),        # Step 1: Normalize data
    ('model', Perceptron(random_state=42)) # Step 2: The Classifier
])

# Define a search space for the best parameters
# The computer will try all combinations to find the best settings
param_grid = {
    'model__eta0': [0.0001, 0.001, 0.01, 0.1, 1.0], # Learning rates
    'model__max_iter': [1000, 2000],                # Max epochs
    'model__penalty': [None, 'l2', 'l1']            # Regularization to prevent overfitting
}

# Run Grid Search (Cross Validation)
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best Parameters Found: {grid.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid.best_score_:.2%}")

# ==========================================
# 3. ADVANCED EVALUATION (Feature 3)
# ==========================================
print("\n--- 3. Evaluating Model Performance ---")

# Predict using the best found model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix: Cancer Detection')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Interpretation for the User
# In medical cases, "Recall" for Malignant is the most critical metric.
# We want to know: Of all actual cancer cases, how many did we catch?

# ==========================================
# 4. DEPLOYMENT (Feature 4)
# ==========================================
print("\n--- 4. Saving Model for Deployment ---")
model_filename = 'breast_cancer_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved to '{model_filename}'. You can now use this in a web app.")

# Simulate loading and using the model on a new patient
loaded_model = joblib.load(model_filename)
# Fake patient data (randomly generated for demo)
new_patient = X_test[0].reshape(1, -1) 
prediction = loaded_model.predict(new_patient)
result = target_names[prediction[0]]

print(f"\n[Test] New Patient Diagnosis: {result.upper()}")