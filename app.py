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

# --- 2. CUSTOM CSS (The "responsive interface" logic) ---
st.markdown("""
<style>
    /* --- GLOBAL SETTINGS --- */
    .stApp {
        background-color: #1a1a1d; /* SOLID DARK BODY */
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Remove default padding to allow header/footer to touch edges */
    .block-container {
        padding-top: 6rem;
        padding-bottom: 6rem;
    }

    /* --- PART 1: THE BLURRED HEADER --- */
    /* We create a fixed container at the top */
    .glass-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 5rem;
        background: rgba(220, 230, 240, 0.15); /* Soft Light Color */
        backdrop-filter: blur(12px); /* THE BLUR EFFECT */
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    .header-text {
        font-size: 24px;
        font-weight: 700;
        color: #e6e6e6;
        letter-spacing: 1px;
    }

    /* --- PART 2: THE RESPONSIVE INPUT BODY --- */
    /* Styling the container for inputs */
    .input-section {
        background-color: #2d2d30; /* Slightly lighter than body */
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        border: 1px solid #3e3e42;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1); /* SMOOTH TRANSITION */
    }

    /* HOVER EFFECT: Lift and Glow */
    .input-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        border-color: #00d2ff; /* Cyan accent on hover */
    }

    .section-title {
        color: #00d2ff;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 5px;
    }

    /* Style Streamlit Inputs to fit the dark theme */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #b0b0b0 !important;
    }

    /* --- PART 3: THE BLURRED FOOTER --- */
    .glass-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 4rem;
        background: rgba(220, 230, 240, 0.15); /* Soft Light Color */
        backdrop-filter: blur(12px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #cccccc;
        font-size: 14px;
    }

    /* PREDICT BUTTON STYLING */
    div.stButton > button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.6);
    }
    
    /* RESULT CARD */
    .result-box {
        background: rgba(0,0,0,0.5);
        border-left: 5px solid #00d2ff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 1s;
    }
    @keyframes fadeIn {
      from {opacity: 0;}
      to {opacity: 1;}
    }

</style>
""", unsafe_allow_html=True)

# --- 3. HTML HEADER INJECTION ---
st.markdown("""
    <div class="glass-header">
        <span class="header-text">🚦 Road Safety AI • Predictive Interface</span>
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

# --- 5. THE BODY (Responsive Inputs) ---

# We use columns to create a grid layout
c1, c2 = st.columns([1, 1])

with c1:
    # --- SECTION A: ENVIRONMENT ---
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌍 Environment</div>',
                unsafe_allow_html=True)

    hour = st.slider("Time of Day", 0, 23, 18)
    day = st.selectbox("Day of Week", unique_values['Day_of_week'])
    weather = st.selectbox("Weather", unique_values['Weather_conditions'])
    light = st.selectbox("Light Conditions", unique_values['Light_conditions'])

    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    # --- SECTION B: INCIDENT ---
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💥 Incident Details</div>',
                unsafe_allow_html=True)

    vehicles = st.number_input("Vehicles Involved", 1, 10, 2)
    casualties = st.number_input("Casualties", 1, 10, 1)
    road_surface = st.selectbox(
        "Road Surface", unique_values['Road_surface_conditions'])
    cause = st.selectbox("Cause", unique_values['Cause_of_accident'])

    st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION C: DRIVER (Full Width) ---
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🚙 Driver & Vehicle</div>',
            unsafe_allow_html=True)

dc1, dc2, dc3 = st.columns(3)
with dc1:
    age = st.selectbox("Driver Age", unique_values['Age_band_of_driver'])
    sex = st.selectbox("Driver Sex", unique_values['Sex_of_driver'])
with dc2:
    exp = st.selectbox("Experience", unique_values['Driving_experience'])
    v_type = st.selectbox("Vehicle Type", unique_values['Type_of_vehicle'])
with dc3:
    col_type = st.selectbox(
        "Collision Type", unique_values['Type_of_collision'])
    junction = st.selectbox("Junction", unique_values['Types_of_Junction'])

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

# --- 6. ACTION & RESULT ---
if st.button("RUN ANALYSIS"):

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
        'Hour_of_Day': hour
    }

    df_in = pd.DataFrame([input_data])

    try:
        probs = model.predict_proba(df_in)[0]

        # Thresholds
        p_fatal, p_serious, p_slight = probs[0], probs[1], probs[2]

        if p_fatal > 0.10:
            status = "HIGH RISK: FATAL"
            color = "#ff4b4b"
            msg = "Conditions suggest high likelihood of fatal injuries."
        elif p_serious > 0.20:
            status = "MODERATE RISK: SERIOUS"
            color = "#ffa500"
            msg = "Serious injuries are probable."
        else:
            status = "LOW RISK: SLIGHT"
            color = "#00d2ff"
            msg = "Likely minor injuries only."

        st.markdown(f"""
        <div class="result-box" style="border-left-color: {color};">
            <h2 style="color: {color}; margin:0;">{status}</h2>
            <p style="color: #ccc; margin-top: 10px;">{msg}</p>
            <p style="font-size: 14px; margin-top: 5px;">Confidence: {max(probs)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# --- 7. HTML FOOTER INJECTION ---
st.markdown("""
    <div class="glass-footer">
        <span>Group 6 Capstone • AI for Road Safety</span>
    </div>
""", unsafe_allow_html=True)
