#!/usr/bin/env python

# Required imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Explore X and y

# Split data (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=123
)

# Choose classifier

# Train model

# Make predictions

# Calculate accuracy

# Create confusion matrix

# Save model.joblib

# Load model.joblib

# Make predictions again

# Verify accuracy
