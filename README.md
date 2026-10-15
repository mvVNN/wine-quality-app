# Qualité du vin blanc : application web

Application Streamlit qui prédit si un vin blanc est bon (note ≥ 7) à partir de ses 11 mesures physico-chimiques, avec le SVM optimisé du projet.

## Installation

```bash
pip install -r requirements.txt
```

## 1. Entraîner et sauvegarder le modèle

Le fichier `winequality-white.csv` (séparateur `;`) doit être présent. Le script reproduit le pipeline du notebook (suppression des doublons, découpage 75/25 stratifié, standardisation ajustée sur l'entraînement, SVM avec C=1 et gamma=0,1).

```bash
python train_model.py --csv winequality-white.csv
```

Le script affiche les métriques sur le jeu de test et enregistre `models/svm_wine.joblib`.

## 2. Lancer l'application

```bash
streamlit run app.py
```

## 3. Tests

```bash
pytest
```

## Structure

- `wine_model.py` : liste des variables, chargement du modèle, fonction de prédiction
- `train_model.py` : entraînement et sauvegarde du modèle
- `app.py` : interface Streamlit
- `tests/` : tests unitaires de la fonction de prédiction
