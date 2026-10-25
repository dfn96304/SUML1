from pathlib import Path

import joblib


_model = joblib.load(Path(__file__).resolve().parent.parent / "model.joblib")
_species = ("setosa", "versicolor", "virginica")


def predict(features):
    """Predict species from sepal length, sepal width, petal length, petal width."""
    prediction = _model.predict([features])[0]
    return _species[int(prediction)]
