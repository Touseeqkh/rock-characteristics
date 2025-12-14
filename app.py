%%writefile app.py
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
# Title & description
# --------------------------------------------------
st.title("Leachate Prediction Application")
st.write("""
This application predicts leachate evolution based on rock characteristics
and a sequence of environmental events (Sequence S).
Each event must only contain:
- "rain" or "snow"
- acidity (numeric, e.g., 0.6)
- temperature after the event (numeric, e.g., 12)

Example sequence:
Event 1: rain;0.6;12
Event 2: snow;0.3;2
Event 3: rain;0.8;15
""")

# --------------------------------------------------
# Rock Selection
# --------------------------------------------------
st.header("Rock Selection")
predefined_rocks = {
    "Custom": None,
    "Basalt": {f: 0.30 for f in feature_cols},
    "Granite": {f: 0.20 for f in feature_cols},
    "Limestone": {f: 0.40 for f in feature_cols},
}
rock_choice = st.selectbox("Select a rock type", list(predefined_rocks.keys()))

# --------------------------------------------------
# Rock characteristics input
# --------------------------------------------------
st.header("Rock Characteristics")
user_inputs = {}
for feature in feature_cols:
    default_value = 0.0 if rock_choice == "Custom" else predefined_rocks[rock_choice][feature]
    user_inputs[feature] = st.number_input(label=feature, value=float(default_value))

# --------------------------------------------------
# Sequence of events S input
# --------------------------------------------------
st.header("Sequence of Events S")
st.write("Enter one event per line using format: precipitation;acidity;temperature")
st.write("Example: rain;0.6;12")

events_text = st.text_area(
    "Enter events (one per line)",
    "rain;0.6;12\nsnow;0.3;2\nrain;0.8;15"
)

events_list = []
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
                events_list.append({"precip": precip, "acidity": acidity, "temp": temp})
        except:
            st.error(f"Invalid format: {line}")

# --------------------------------------------------
# Event impact logic
# --------------------------------------------------
def apply_event(state, event):
    """Modify input features based on the event"""
    state_mod = state.copy()

    # Example adjustments (you can refine based on real domain logic)
    # Adjust first three features as example
    if event["precip"] == "rain":
        state_mod[0, 0] *= 1.05
    elif event["precip"] == "snow":
        state_mod[0, 0] *= 1.03

    state_mod[0, 1] += event["acidity"]
    state_mod[0, 2] = event["temp"]

    return state_mod

# --------------------------------------------------
# Preprocessed data display (optional)
# --------------------------------------------------
st.header("Preprocessed Data")
try:
    preprocessed_df = pd.read_csv("preprocessed_data.csv")  # adjust file extension if needed
    st.dataframe(preprocessed_df.head())
except:
    st.warning("preprocessed_data file not found or unreadable")

# --------------------------------------------------
# Scenario selection from Excel (optional)
# --------------------------------------------------
st.header("Predefined Scenarios (from Excel)")
try:
    excel_data = pd.read_excel("semson_si.xlsx")
    scenario_names = excel_data["Scenario Name"].unique()
    scenario_choice = st.selectbox("Select a scenario", scenario_names)
    scenario_events = excel_data[excel_data["Scenario Name"] == scenario_choice][["Precipitation","Acidity","Temperature"]]
    st.write(scenario_events)
except:
    st.info("semson_si.xlsx not found or unreadable")

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
predictions = []

if st.button("Run Prediction") and events_list:
    st.subheader("Leachate Predictions After Each Event")

    input_df = pd.DataFrame([user_inputs])
    current_state = scaler.transform(input_df)

    for i, event in enumerate(events_list):
        pred = model.predict(current_state)[0]
        predictions.append(pred)

        st.write(f"**After event {i+1} ({event['precip']}, acidity={event['acidity']}, temp={event['temp']})**: {pred:.4f}")

        # Plain-language explanation
        if pred < 0.2:
            explanation = "Leachate formation is very low. Your rock and conditions are safe."
        elif pred < 0.5:
            explanation = "Leachate formation is moderate. Some care might be needed."
        elif pred < 0.8:
            explanation = "Leachate formation is high. Caution is advised, monitor closely."
        else:
            explanation = "Leachate formation is very high! Immediate action may be required."

        st.write(f"Explanation: {explanation}")

        current_state = apply_event(current_state, event)

    # Overall risk summary
    final_pred = predictions[-1]
    st.subheader("Overall Leachate Risk After Sequence S")
    if final_pred < 0.2:
        overall = "Low risk of leachate. The system is stable."
    elif final_pred < 0.5:
        overall = "Moderate risk. Pay attention to conditions."
    elif final_pred < 0.8:
        overall = "High risk. Take preventive measures."
    else:
        overall = "Very high risk! Immediate action needed."
    st.write(f"Final prediction: {final_pred:.4f} → {overall}")

    # Feature importance visualization
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=False)
    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
