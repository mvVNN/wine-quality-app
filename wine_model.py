"""Chargement du modèle et prédiction de la qualité d'un vin blanc."""
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

MODEL_PATH = Path(__file__).parent / "models" / "svm_wine.joblib"


def load_model(path=MODEL_PATH):
    """Charge le pipeline (StandardScaler + SVM) sauvegardé par train_model.py."""
    return joblib.load(path)


def predict_wine(model, values):
    """Retourne si le vin est bon (note >= 7) et la probabilité estimée qu'il le soit.

    `values` est un dictionnaire {nom_de_la_variable: valeur} contenant les 11 variables.
    """
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Variables manquantes : {', '.join(missing)}")

    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    label = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return {"good": label == 1, "probability": probability}
