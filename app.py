import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Page Config ---
st.set_page_config(
    page_title="SafeRoute Pro",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    /* --- GLOBAL SETTINGS --- */
    .stApp {
        background-color: #2e2e2e; /* CHANGED TO DARK GREY */
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 6rem;
        padding-bottom: 6rem;
    }

    /* --- GLASS HEADER --- */
    .glass-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 5rem;
        background: rgba(46, 46, 46, 0.7); /* Semi-transparent Dark Grey */
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    .header-text {
        font-size: 24px;
        font-weight: 700;
        color: #f0f0f0;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* --- RESPONSIVE INPUT BODY --- */
    .input-section {
        background-color: #383838; /* Slightly lighter grey for cards */
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        border: 1px solid #4d4d4d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .input-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        border-color: #00d2ff;
    }

    .section-title {
        color: #00d2ff;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 5px;
    }

    /* Input Text Labels */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #cccccc !important;
    }

    /* --- GLASS FOOTER --- */
    .glass-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3rem;
        background: rgba(46, 46, 46, 0.7);
        backdrop-filter: blur(12px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #888888;
        font-size: 12px;
    }

    /* --- PREDICT BUTTON --- */
    div.stButton > button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
        width: 100%;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.6);
    }
    
    /* --- RESULT BOX --- */
    .result-box {
        background: rgba(30, 30, 30, 0.8);
        border-left: 5px solid #00d2ff;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.8s;
    }
    @keyframes fadeIn {
      from {opacity: 0; transform: translateY(10px);}
      to {opacity: 1; transform: translateY(0);}
    }

</style>
""", unsafe_allow_html=True)

# --- 3. HEADER INJECTION ---
st.markdown("""
    <div class="glass-header">
        <span class="header-text">🚦 Road Safety AI System</span>
    </div>
""", unsafe_allow_html=True)

# --- 4. LOAD DATA ---


@st.cache_resource
def load_data():
    model = joblib.load('rta_model_pipeline.joblib')
    unique_values = joblib.load('unique_values.joblib')
    return model, unique_values


try:
    model, unique_values = load_data()
except FileNotFoundError:
    st.error("Model files missing.")
    st.stop()

# --- 5. INTRODUCTORY SECTION (NEW!) ---
st.title("Accident Severity Risk Assessment")
st.markdown("""
<div style="background-color: #383838; padding: 15px; border-radius: 10px; border-left: 5px solid #00d2ff; margin-bottom: 25px;">
    <p style="font-size: 16px; margin: 0; color: #e0e0e0; line-height: 1.6;">
        <b>Welcome to the Predictive Analytics Dashboard.</b><br>
        This tool utilizes advanced machine learning algorithms to analyze environmental conditions, vehicle characteristics, 
        and driver demographics. By processing these variables, the system predicts the potential severity of a road accident 
        (Slight, Serious, or Fatal). Adjust the parameters below to simulate different scenarios and assess risk levels.
    </p>
</div>
""", unsafe_allow_html=True)

# --- 6. MAIN BODY ---

# We use columns to create a grid layout
c1, c2 = st.columns([1, 1])

with c1:
    # --- SECTION A: ENVIRONMENT ---
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌍 Environmental Factors</div>',
                unsafe_allow_html=True)

    hour = st.slider("Time of Day (24hr)", 0, 23, 18)
    day = st.selectbox("Day of Week", unique_values['Day_of_week'])
    weather = st.selectbox("Weather Conditions",
                           unique_values['Weather_conditions'])
    light = st.selectbox("Light Conditions", unique_values['Light_conditions'])

    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    # --- SECTION B: INCIDENT ---
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💥 Incident Specifics</div>',
                unsafe_allow_html=True)

    vehicles = st.number_input("Vehicles Involved", 1, 10, 2)
    casualties = st.number_input("Number of Casualties", 1, 10, 1)
    road_surface = st.selectbox(
        "Road Surface Condition", unique_values['Road_surface_conditions'])
    cause = st.selectbox("Primary Cause", unique_values['Cause_of_accident'])

    st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION C: DRIVER (Full Width) ---
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🚙 Driver & Vehicle Profile</div>',
            unsafe_allow_html=True)

dc1, dc2, dc3 = st.columns(3)
with dc1:
    age = st.selectbox("Driver Age Band", unique_values['Age_band_of_driver'])
    sex = st.selectbox("Driver Gender", unique_values['Sex_of_driver'])
with dc2:
    exp = st.selectbox("Driving Experience",
                       unique_values['Driving_experience'])
    v_type = st.selectbox("Vehicle Type", unique_values['Type_of_vehicle'])
with dc3:
    col_type = st.selectbox(
        "Collision Type", unique_values['Type_of_collision'])
    junction = st.selectbox(
        "Junction Type", unique_values['Types_of_Junction'])

# Hidden defaults
defaults = {
    'Educational_level': unique_values['Educational_level'][0],
    'Vehicle_driver_relation': unique_values['Vehicle_driver_relation'][0],
    'Owner_of_vehicle': unique_values['Owner_of_vehicle'][0],
    'Service_year_of_vehicle': unique_values['Service_year_of_vehicle'][0],
    'Defect_of_vehicle': unique_values['Defect_of_vehicle'][0],
    'Area_accident_occured': unique_values['Area_accident_occured'][0],
    'Lanes_or_Medians': unique_values['Lanes_or_Medians'][0],
    'Road_allignment': unique_values['Road_allignment'][0],
    'Road_surface_type': unique_values['Road_surface_type'][0],
    'Vehicle_movement': unique_values['Vehicle_movement'][0],
    'Casualty_class': unique_values['Casualty_class'][0],
    'Sex_of_casualty': unique_values['Sex_of_casualty'][0],
    'Age_band_of_casualty': unique_values['Age_band_of_casualty'][0],
    'Casualty_severity': unique_values['Casualty_severity'][0],
    'Work_of_casuality': unique_values['Work_of_casuality'][0],
    'Fitness_of_casuality': unique_values['Fitness_of_casuality'][0],
    'Pedestrian_movement': unique_values['Pedestrian_movement'][0]
}

st.markdown('</div>', unsafe_allow_html=True)

# --- 7. ACTION & RESULT ---
if st.button("RUN RISK ANALYSIS"):

    # Map Inputs
    input_data = {
        'Time': '00:00:00',
        'Day_of_week': day,
        'Age_band_of_driver': age,
        'Sex_of_driver': sex,
        'Educational_level': defaults['Educational_level'],
        'Vehicle_driver_relation': defaults['Vehicle_driver_relation'],
        'Driving_experience': exp,
        'Type_of_vehicle': v_type,
        'Owner_of_vehicle': defaults['Owner_of_vehicle'],
        'Service_year_of_vehicle': defaults['Service_year_of_vehicle'],
        'Defect_of_vehicle': defaults['Defect_of_vehicle'],
        'Area_accident_occured': defaults['Area_accident_occured'],
        'Lanes_or_Medians': defaults['Lanes_or_Medians'],
        'Road_allignment': defaults['Road_allignment'],
        'Types_of_Junction': junction,
        'Road_surface_type': defaults['Road_surface_type'],
        'Road_surface_conditions': road_surface,
        'Light_conditions': light,
        'Weather_conditions': weather,
        'Type_of_collision': col_type,
        'Number_of_vehicles_involved': vehicles,
        'Number_of_casualties': casualties,
        'Vehicle_movement': defaults['Vehicle_movement'],
        'Casualty_class': defaults['Casualty_class'],
        'Sex_of_casualty': defaults['Sex_of_casualty'],
        'Age_band_of_casualty': defaults['Age_band_of_casualty'],
        'Casualty_severity': defaults['Casualty_severity'],
        'Work_of_casuality': defaults['Work_of_casuality'],
        'Fitness_of_casuality': defaults['Fitness_of_casuality'],
        'Pedestrian_movement': defaults['Pedestrian_movement'],
        'Cause_of_accident': cause,
        'Hour_of_Day': hour  # Keep as 'Hour' or 'Hour_of_Day' based on what worked!
    }

    df_in = pd.DataFrame([input_data])

    try:
        probs = model.predict_proba(df_in)[0]

        # Thresholds
        p_fatal, p_serious, p_slight = probs[0], probs[1], probs[2]

        if p_fatal > 0.10:
            status = "CRITICAL RISK: FATAL"
            color = "#ff4b4b"
            msg = "Warning: The selected conditions indicate a high probability of fatal outcomes. Immediate safety intervention recommended."
        elif p_serious > 0.20:
            status = "HIGH RISK: SERIOUS INJURY"
            color = "#ffa500"
            msg = "Caution: Conditions are favorable for serious injuries. Drive with extreme care."
        else:
            status = "LOW RISK: SLIGHT INJURY"
            color = "#00d2ff"
            msg = "Risk Assessment: Accidents under these conditions are likely to be minor, but safety protocols should still be followed."

        st.markdown(f"""
        <div class="result-box" style="border-left-color: {color};">
            <h2 style="color: {color}; margin:0;">{status}</h2>
            <p style="color: #e0e0e0; margin-top: 10px; font-size: 16px;">{msg}</p>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
            <p style="font-size: 14px; margin-top: 5px; color: #aaa;">AI Confidence Score: <b>{max(probs)*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# --- 8. FOOTER INJECTION ---
st.markdown("""
    <div class="glass-footer">
        <span>Group 6 Capstone Project • Developed for Road Safety Analysis</span>
    </div>
""", unsafe_allow_html=True)
