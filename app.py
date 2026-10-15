"""Application web : prédit si un vin blanc est bon à partir de ses mesures"""
import streamlit as st

from wine_model import MODEL_PATH, load_model, predict_wine

SLIDERS = [
    ("fixed acidity", "Acidité fixe (g/L)", 3.8, 14.2, 6.9, 0.1),
    ("volatile acidity", "Acidité volatile (g/L)", 0.08, 1.1, 0.28, 0.01),
    ("citric acid", "Acide citrique (g/L)", 0.0, 1.66, 0.33, 0.01),
    ("residual sugar", "Sucre résiduel (g/L)", 0.6, 65.8, 6.4, 0.1),
    ("chlorides", "Chlorures (g/L)", 0.009, 0.346, 0.046, 0.001),
    ("free sulfur dioxide", "Dioxyde de soufre libre (mg/L)", 2.0, 289.0, 35.0, 1.0),
    ("total sulfur dioxide", "Dioxyde de soufre total (mg/L)", 9.0, 440.0, 138.0, 1.0),
    ("density", "Densité", 0.987, 1.039, 0.994, 0.0001),
    ("pH", "pH", 2.72, 3.82, 3.19, 0.01),
    ("sulphates", "Sulfates (g/L)", 0.22, 1.08, 0.49, 0.01),
    ("alcohol", "Alcool (% vol.)", 8.0, 14.2, 10.5, 0.1),
]


@st.cache_resource
def get_model():
    return load_model()


st.set_page_config(page_title="Qualité du vin blanc", page_icon="🍷")
st.title("Qualité du vin blanc")
st.write(
    "Renseignez les mesures physico-chimiques d'un vin. Le modèle (SVM optimisé) estime "
    "si sa note sera supérieure ou égale à 7."
)

if not MODEL_PATH.exists():
    st.error(
        "Modèle introuvable. Lancez d'abord `python train_model.py --csv winequality-white.csv`."
    )
    st.stop()

model = get_model()

values = {}
left, right = st.columns(2)
for i, (name, label, low, high, default, step) in enumerate(SLIDERS):
    column = left if i % 2 == 0 else right
    values[name] = column.slider(label, min_value=low, max_value=high, value=default, step=step)

if st.button("Prédire", type="primary"):
    result = predict_wine(model, values)
    if result["good"]:
        st.success(f"Bon vin prédit (probabilité estimée : {result['probability']:.0%})")
    else:
        st.error(f"Mauvais vin prédit (probabilité estimée de bon vin : {result['probability']:.0%})")
    st.caption(
        "Le SVM a un rappel de 82 % sur le jeu de test : il détecte la plupart des bons vins, "
        "mais environ 57 % des vins qu'il annonce bons sont en réalité mauvais."
    )
