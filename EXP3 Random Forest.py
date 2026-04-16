import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# -------------------------------
# Step 1: Load Existing Datasets
# -------------------------------

# Classification Dataset: Iris
iris = load_iris()
X_clf = iris.data
y_clf = iris.target

# Regression Dataset: Diabetes
diabetes = load_diabetes()
X_reg = diabetes.data
y_reg = diabetes.target

# -------------------------------
# Step 2: Split Data
# -------------------------------

# Classification split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Regression split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# -------------------------------
# Step 3: Train Random Forest Classifier
# -------------------------------

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_clf, y_train_clf)
y_pred_clf = rf_classifier.predict(X_test_clf)

# -------------------------------
# Step 4: Train Random Forest Regressor
# -------------------------------

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_regressor.predict(X_test_reg)

# -------------------------------
# Step 5: Evaluate Models
# -------------------------------

print("Model Evaluation")

# Classification Accuracy
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Random Forest Classifier Accuracy (Iris): {accuracy:.2f}")

# Regression MSE
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Random Forest Regressor MSE (Diabetes): {mse:.2f}")