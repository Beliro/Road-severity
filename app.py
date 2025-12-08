import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide"
)

# Load the saved model and unique values


@st.cache_resource
def load_data():
    model = joblib.load('rta_model_pipeline.joblib')
    unique_values = joblib.load('unique_values.joblib')
    return model, unique_values


try:
    model, unique_values = load_data()
except FileNotFoundError:
    st.error("Please run the notebook code to save 'rta_model_pipeline.joblib' first!")
    st.stop()

# --- Title and Header ---
st.title("🚑 Road Accident Severity Prediction App")
st.markdown("""
This application uses **Machine Learning (Random Forest)** to predict the severity of a road accident 
based on environmental conditions, driver details, and vehicle characteristics.
""")
st.divider()

# --- Input Form ---
# We split inputs into logical sections to make it "Presentable"

# Section 1: Key Accident Info (Top Row)
st.subheader("📝 Accident Details")
col1, col2, col3 = st.columns(3)

with col1:
    hour = st.slider("Hour of Day (24h)", 0, 23, 17)
    day = st.selectbox("Day of Week", unique_values['Day_of_week'])
    cause = st.selectbox("Cause of Accident",
                         unique_values['Cause_of_accident'])

with col2:
    vehicles = st.number_input("Vehicles Involved", 1, 10, 2)
    casualties = st.number_input("Number of Casualties", 1, 10, 1)
    collision = st.selectbox(
        "Type of Collision", unique_values['Type_of_collision'])

with col3:
    light = st.selectbox("Light Conditions", unique_values['Light_conditions'])
    weather = st.selectbox("Weather Conditions",
                           unique_values['Weather_conditions'])
    road_surface = st.selectbox(
        "Road Surface", unique_values['Road_surface_conditions'])

st.divider()

# Section 2: Driver & Vehicle Info (Expandable to keep UI clean)
with st.expander("🚙 Driver & Vehicle Details (Advanced)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        age_band = st.selectbox(
            "Driver Age Band", unique_values['Age_band_of_driver'])
        sex = st.selectbox("Driver Gender", unique_values['Sex_of_driver'])
        education = st.selectbox(
            "Education Level", unique_values['Educational_level'])
        experience = st.selectbox(
            "Driving Experience", unique_values['Driving_experience'])

    with c2:
        vehicle_type = st.selectbox(
            "Vehicle Type", unique_values['Type_of_vehicle'])
        service_year = st.selectbox(
            "Vehicle Service Year", unique_values['Service_year_of_vehicle'])
        defect = st.selectbox(
            "Vehicle Defect", unique_values['Defect_of_vehicle'])
        owner = st.selectbox(
            "Vehicle Owner", unique_values['Owner_of_vehicle'])

    with c3:
        junction = st.selectbox(
            "Junction Type", unique_values['Types_of_Junction'])
        road_align = st.selectbox(
            "Road Alignment", unique_values['Road_allignment'])
        movement = st.selectbox(
            "Vehicle Movement", unique_values['Vehicle_movement'])
        pedestrian = st.selectbox(
            "Pedestrian Movement", unique_values['Pedestrian_movement'])

# Use defaults for the less critical columns to simplify the form
# (We need to provide ALL columns expected by the model)
default_inputs = {
    'Vehicle_driver_relation': unique_values['Vehicle_driver_relation'][0],
    'Area_accident_occured': unique_values['Area_accident_occured'][0],
    'Lanes_or_Medians': unique_values['Lanes_or_Medians'][0],
    'Road_surface_type': unique_values['Road_surface_type'][0],
    'Casualty_class': unique_values['Casualty_class'][0],
    'Sex_of_casualty': unique_values['Sex_of_casualty'][0],
    'Age_band_of_casualty': unique_values['Age_band_of_casualty'][0],
    'Casualty_severity': unique_values['Casualty_severity'][0],
    'Work_of_casuality': unique_values['Work_of_casuality'][0],
    'Fitness_of_casuality': unique_values['Fitness_of_casuality'][0]
}

# --- Prediction Logic ---
st.markdown("###")
if st.button("🔍 Predict Severity", type="primary", use_container_width=True):

    # 1. Gather all inputs into a dictionary
    input_data = {
        'Time': '00:00:00',  # Placeholder, not used by model directly as we use Hour
        'Day_of_week': day,
        'Age_band_of_driver': age_band,
        'Sex_of_driver': sex,
        'Educational_level': education,
        'Vehicle_driver_relation': default_inputs['Vehicle_driver_relation'],
        'Driving_experience': experience,
        'Type_of_vehicle': vehicle_type,
        'Owner_of_vehicle': owner,
        'Service_year_of_vehicle': service_year,
        'Defect_of_vehicle': defect,
        'Area_accident_occured': default_inputs['Area_accident_occured'],
        'Lanes_or_Medians': default_inputs['Lanes_or_Medians'],
        'Road_allignment': road_align,
        'Types_of_Junction': junction,
        'Road_surface_type': default_inputs['Road_surface_type'],
        'Road_surface_conditions': road_surface,
        'Light_conditions': light,
        'Weather_conditions': weather,
        'Type_of_collision': collision,
        'Number_of_vehicles_involved': vehicles,
        'Number_of_casualties': casualties,
        'Vehicle_movement': movement,
        'Casualty_class': default_inputs['Casualty_class'],
        'Sex_of_casualty': default_inputs['Sex_of_casualty'],
        'Age_band_of_casualty': default_inputs['Age_band_of_casualty'],
        'Casualty_severity': default_inputs['Casualty_severity'],
        'Work_of_casuality': default_inputs['Work_of_casuality'],
        'Fitness_of_casuality': default_inputs['Fitness_of_casuality'],
        'Pedestrian_movement': pedestrian,
        'Cause_of_accident': cause,
        'Hour_of_Day': hour
    }

    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. Predict
    # Get probabilities to handle the "Custom Threshold" we discussed
    prediction_proba = model.predict_proba(input_df)[0]

    # Custom Thresholds (Safety First!)
    threshold_fatal = 0.10
    threshold_serious = 0.20

    if prediction_proba[0] > threshold_fatal:
        final_pred = "Fatal Injury"
        severity_color = "red"
        icon = "🚨"
    elif prediction_proba[1] > threshold_serious:
        final_pred = "Serious Injury"
        severity_color = "orange"
        icon = "⚠️"
    else:
        final_pred = "Slight Injury"
        severity_color = "green"
        icon = "✅"

    # 4. Display Result
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color: {severity_color};">{icon} Prediction: {final_pred}</h2>
        <p>Confidence: {max(prediction_proba)*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Show detailed probabilities
    st.markdown("#### Detailed Probability Analysis")
    prob_df = pd.DataFrame(prediction_proba, index=[
                           'Fatal', 'Serious', 'Slight'], columns=['Probability'])
    st.bar_chart(prob_df)
