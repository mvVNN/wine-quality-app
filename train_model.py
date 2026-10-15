"""Reproduit le pipeline du notebook et sauvegarde le modèle SVM optimisé
"""
import argparse

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wine_model import FEATURES, MODEL_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Fichier winequality-white.csv (séparateur ;)")
    parser.add_argument("--out", default=str(MODEL_PATH))
    args = parser.parse_args()

    data = pd.read_csv(args.csv, sep=";").drop_duplicates().reset_index(drop=True)
    data["good_wine"] = (data["quality"] >= 7).astype(int)

    X = data[FEATURES]
    y = data["good_wine"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1, gamma=0.1, probability=True, class_weight="balanced", random_state=42),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Précision : {precision_score(y_test, y_pred):.3f}")
    print(f"Rappel    : {recall_score(y_test, y_pred):.3f}")
    print(f"F1        : {f1_score(y_test, y_pred):.3f}")
    print(f"AUC       : {roc_auc_score(y_test, y_proba):.3f}")

    joblib.dump(model, args.out)
    print(f"Modèle sauvegardé : {args.out}")


if __name__ == "__main__":
    main()
