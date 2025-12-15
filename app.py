import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Leachate Prediction App",
    layout="centered"
)

# --------------------------------------------------
# Load trained model and preprocessing objects
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

# --------------------------------------------------
# App title and description
# --------------------------------------------------
st.title("Leachate Prediction Application")

st.write(
    """
    This application predicts **leachate formation** based on:
    - Rock characteristics
    - A sequence of environmental events (Sequence S)

    **Sequence S** means:
    The user enters multiple events one after another.
    Each event must contain ONLY:
    - rain or snow
    - acidity value
    - temperature after the event
    """
)

st.write(
    """
    **Example input (one event per line):**
    ```
    rain;0.6;12
    snow;0.3;2
    rain;0.8;15
    ```
    """
)

# --------------------------------------------------
# Rock selection
# --------------------------------------------------
st.header("Rock Selection")

predefined_rocks = {
    "Custom": None,
    "Basalt": {f: 0.30 for f in feature_cols},
    "Granite": {f: 0.20 for f in feature_cols},
    "Limestone": {f: 0.40 for f in feature_cols},
}

rock_choice = st.selectbox(
    "Select a rock type",
    list(predefined_rocks.keys())
)

# --------------------------------------------------
# Rock characteristics input
# --------------------------------------------------
st.header("Rock Characteristics")

user_inputs = {}

for feature in feature_cols:
    default_value = 0.0 if rock_choice == "Custom" else predefined_rocks[rock_choice][feature]
    user_inputs[feature] = st.number_input(
        label=feature,
        value=float(default_value)
    )

# --------------------------------------------------
# Sequence of Events S input
# --------------------------------------------------
st.header("Sequence of Events S")

events_text = st.text_area(
    "Enter events (one per line)",
    "rain;0.6;12\nsnow;0.3;2\nrain;0.8;15"
)

events = []

for line in events_text.split("\n"):
    line = line.strip()
    if line:
        try:
            precip, acidity, temp = line.split(";")
            precip = precip.lower()
            acidity = float(acidity)
            temp = float(temp)

            if precip not in ["rain", "snow"]:
                st.error(f"Invalid precipitation type: {precip}")
            else:
                events.append({
                    "precip": precip,
                    "acidity": acidity,
                    "temp": temp
                })
        except:
            st.error(f"Invalid format: {line}")

# --------------------------------------------------
# Event impact logic
# --------------------------------------------------
def apply_event(state, event):
    new_state = state.copy()

    # precipitation effect
    if event["precip"] == "rain":
        new_state *= 1.05
    elif event["precip"] == "snow":
        new_state *= 1.03

    # acidity effect
    new_state += event["acidity"] * 0.02

    # temperature effect
    new_state += event["temp"] * 0.001

    return new_state

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
if st.button("Run Prediction") and len(events) > 0:
    st.subheader("Predictions After Each Event")

    input_df = pd.DataFrame([user_inputs])
    current_state = scaler.transform(input_df)

    predictions = []

    for i, event in enumerate(events):
        prediction = model.predict(current_state)[0]
        predictions.append(prediction)

        st.write(
            f"**After Event {i+1}** "
            f"({event['precip']}, acidity={event['acidity']}, temp={event['temp']}°C): "
            f"**{prediction:.4f}**"
        )

        # ------------------------------------------
        # Explanation for NON-CS users
        # ------------------------------------------
        if prediction < 0.25:
            explanation = "Leachate formation is very low. Environmental impact is minimal."
        elif prediction < 0.5:
            explanation = "Leachate formation is moderate. Some monitoring is recommended."
        elif prediction < 0.75:
            explanation = "Leachate formation is high. Preventive action should be considered."
        else:
            explanation = "Leachate formation is very high. Immediate action is recommended."

        st.write(f"**Explanation:** {explanation}")

        current_state = apply_event(current_state, event)

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    st.subheader("Final Risk Summary")

    final_prediction = predictions[-1]

    if final_prediction < 0.25:
        summary = "Overall leachate risk is LOW."
    elif final_prediction < 0.5:
        summary = "Overall leachate risk is MODERATE."
    elif final_prediction < 0.75:
        summary = "Overall leachate risk is HIGH."
    else:
        summary = "Overall leachate risk is VERY HIGH."

    st.write(f"**Final Prediction Value:** {final_prediction:.4f}")
    st.write(f"**Overall Assessment:** {summary}")

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------
    st.subheader("What Influenced the Prediction?")

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    st.pyplot(fig)

