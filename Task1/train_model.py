#!/usr/bin/env python

# Required imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lg
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load Iris dataset
iris = load_iris()

x = iris.data
y = iris.target
# Explore x and y

# Split data (80% / 20%)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=123
)

# Choose classifier
logicReg = lg(max_iter=200)
# Train model
logicReg.fit(x_train, y_train)
# Make predictions
y_pred = logicReg.predict(x_test)
print("Actual:", y_test[:10])
print("Predicted:", y_pred[:10])
print("")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:", cm, "\n")
# Save model.joblib
joblib.dump(logicReg, "model.joblib")
print("\nModel saved to model.joblib")
# Load model.joblib
loaded_model = joblib.load("model.joblib")
# Make predictions again
loaded_y_pred = loaded_model.predict(x_test)
# Verify accuracy
loaded_accuracy = accuracy_score(y_test, loaded_y_pred)
print("Loaded Model Accuracy:", loaded_accuracy)