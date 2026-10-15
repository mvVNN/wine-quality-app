import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wine_model import FEATURES, load_model, predict_wine


@pytest.fixture
def model(tmp_path):
    # Petit modèle factice entraîné sur des données aléatoires, pour tester sans le CSV.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    y = (X["alcohol"] > 0).astype(int)
    pipeline = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    pipeline.fit(X, y)
    path = tmp_path / "model.joblib"
    import joblib

    joblib.dump(pipeline, path)
    return load_model(path)


def sample_values():
    return {f: 0.0 for f in FEATURES}


def test_predict_returns_label_and_probability(model):
    result = predict_wine(model, sample_values())
    assert isinstance(result["good"], bool)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_raises_when_feature_missing(model):
    values = sample_values()
    del values["alcohol"]
    with pytest.raises(ValueError, match="alcohol"):
        predict_wine(model, values)
