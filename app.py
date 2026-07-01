import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loading import load_data
from preprocessing import create_target, split_data
from models import decision_tree_model
from sklearn.metrics import accuracy_score

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# -------------------- Train Model Once --------------------
@st.cache_resource
def train_model():

    df = load_data()
    df = create_target(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = decision_tree_model()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, df, X_train, accuracy


model, df, X_train, accuracy = train_model()

# -------------------- Sidebar --------------------

st.sidebar.title("🍷 Wine Quality Prediction")

st.sidebar.success("Machine Learning Project")

st.sidebar.write("**Algorithm:** Decision Tree")

st.sidebar.write(f"**Accuracy:** {accuracy*100:.2f}%")

st.sidebar.write("**Dataset:** Wine Quality Dataset")

# -------------------- Title --------------------

st.title("🍷 Wine Quality Prediction System")

st.write(
    "Predict whether a wine is **GOOD** or **BAD** based on its chemical properties."
)

st.markdown("---")

# -------------------- Dataset Preview --------------------

with st.expander("📄 View Dataset"):

    st.dataframe(df.head())

# -------------------- Input Form --------------------

st.subheader("Enter Wine Characteristics")

col1, col2 = st.columns(2)

with col1:

    fixed_acidity = st.number_input(
        "Fixed Acidity", value=7.4)

    volatile_acidity = st.number_input(
        "Volatile Acidity", value=0.70)

    citric_acid = st.number_input(
        "Citric Acid", value=0.00)

    residual_sugar = st.number_input(
        "Residual Sugar", value=1.90)

    chlorides = st.number_input(
        "Chlorides", value=0.076)

    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide", value=11)

with col2:

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide", value=34)

    density = st.number_input(
        "Density",
        value=0.9978,
        format="%.4f"
    )

    ph = st.number_input(
        "pH", value=3.51)

    sulphates = st.number_input(
        "Sulphates", value=0.56)

    alcohol = st.number_input(
        "Alcohol", value=9.4)

st.markdown("---")

# -------------------- Prediction --------------------

if st.button("🔍 Predict Wine Quality", use_container_width=True):

    user_data = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]],
    columns=X_train.columns)

    prediction = model.predict(user_data)

    probability = model.predict_proba(user_data)

    confidence = probability.max() * 100

    st.subheader("Prediction")

    if prediction[0] == 1:

        st.success("✅ GOOD WINE")

        st.balloons()

    else:

        st.error("❌ BAD WINE")

    st.info(f"Confidence : {confidence:.2f}%")

# -------------------- Feature Importance --------------------

st.markdown("---")

st.subheader("📊 Feature Importance")

importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": model.feature_importances_

}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    ax=ax
)

st.pyplot(fig)

# -------------------- Footer --------------------

st.markdown("---")

st.caption("Developed using Streamlit | Wine Quality Prediction")