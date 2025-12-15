import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Leachate Prediction App",
    layout="wide"
)

# ==================================================
# Dark Theme Styling
# ==================================================
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Load Model Artifacts
# ==================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

# ==================================================
# Sidebar Inputs
# ==================================================
st.sidebar.title("User Inputs")

st.sidebar.subheader("Rock Characteristics")
user_inputs = {}
for feature in feature_cols:
    user_inputs[feature] = st.sidebar.number_input(
        feature,
        value=0.3
    )

st.sidebar.subheader("Sequence S")
st.sidebar.write("Format: rain/snow;acidity;temperature")

events_text = st.sidebar.text_area(
    "Enter events (one per line)",
    "rain;0.6;12\nsnow;0.3;2\nrain;0.8;15"
)

# ==================================================
# Layout
# ==================================================
left_col, right_col = st.columns([1.3, 1])

# ==================================================
# LEFT SIDE → 3D Visualization
# ==================================================
with left_col:
    st.subheader("3D Environmental Representation")

    try:
        df = pd.read_csv("preprocessed_data.csv")

        fig = px.scatter_3d(
            df,
            x=df.columns[0],
            y=df.columns[1],
            z=df.columns[2],
            color=df.columns[2],
            template="plotly_dark",
            title="3D View of Environmental Factors"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            "This 3D visualization helps non-technical users understand "
            "how environmental factors interact together."
        )

    except:
        st.warning("3D data is not available.")

# ==================================================
# Explanation Helpers (SHAP-Inspired, Simple)
# ==================================================
def risk_label(pred):
    if pred < 0.25:
        return "LOW RISK"
    elif pred < 0.5:
        return "MODERATE RISK"
    elif pred < 0.75:
        return "HIGH RISK"
    else:
        return "VERY HIGH RISK"


def explain_event_with_shap_simple(pred, inputs):
    reasons = []

    avg_value = np.mean(list(inputs.values()))

    if avg_value < 0.3:
        reasons.append("The event had very low acidity, helping keep the leachate low.")
    elif avg_value < 0.6:
        reasons.append("Moderate acidity slightly increased the leachate level.")
    else:
        reasons.append("High acidity strongly increased leachate formation.")

    if inputs.get("Na", 0.3) < 0.4 or inputs.get("K", 0.3) < 0.4:
        reasons.append("Lower potassium and sodium levels helped control the leachate.")
    else:
        reasons.append("Higher salt content contributed to leachate formation.")

    reasons.append("Overall stable water chemistry helped regulate the system.")

    return reasons

# ==================================================
# RIGHT SIDE → Prediction Logic
# ==================================================
with right_col:
    st.title("Leachate Prediction Application")

    st.write(
        """
        This application predicts **leachate risk step-by-step**
        using a trained machine-learning model.

        The explanation is written in **simple language**
        so that non-technical users can understand the result.
        """
    )

    # Parse events
    events = []
    for line in events_text.split("\n"):
        if line.strip():
            try:
                p, a, t = line.split(";")
                events.append({
                    "precip": p.lower(),
                    "acidity": float(a),
                    "temp": float(t)
                })
            except:
                st.error(f"Invalid event format: {line}")

    # Event impact logic
    def apply_event(state, event):
        new_state = state.copy()

        if event["precip"] == "rain":
            new_state *= 1.05
        else:
            new_state *= 1.03

        new_state += event["acidity"] * 0.02
        new_state += event["temp"] * 0.001

        return new_state

    # Run prediction
    if st.button("Run Prediction"):
        input_df = pd.DataFrame([user_inputs])
        current_state = scaler.transform(input_df)

        for i, event in enumerate(events):
            pred = model.predict(current_state)[0]
            label = risk_label(pred)

            st.markdown(f"""
### Explanation  
**Predicted Leachate:** {pred*100:.2f} → **{label}**

This event is **{label.split()[0]}** because:
""")

            reasons = explain_event_with_shap_simple(pred, user_inputs)
            for r in reasons:
                st.write(f"• {r}")

            current_state = apply_event(current_state, event)

        # Feature Importance
        st.subheader("Key Factors Influencing the Prediction")

        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance")

        fig, ax = plt.subplots()
        ax.barh(imp_df["Feature"], imp_df["Importance"])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
