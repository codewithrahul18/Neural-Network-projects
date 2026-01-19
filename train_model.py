import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load Data
print("--- Loading Data ---")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Build Pipeline (Scaling + Model)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Perceptron(random_state=42))
])

# 3. Hyperparameter Tuning (GridSearch)
print("--- Tuning Model ---")
param_grid = {
    'model__eta0': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'model__max_iter': [1000, 2000],
    'model__penalty': [None, 'l2', 'l1']
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# 4. Evaluate
print(f"Best Params: {grid.best_params_}")
best_model = grid.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
print("\nClassification Report:\n")
print(classification_report(y_test, best_model.predict(X_test), target_names=data.target_names))

# 5. Save Model
joblib.dump(best_model, 'breast_cancer_model.pkl')
print("Model saved to 'breast_cancer_model.pkl'")