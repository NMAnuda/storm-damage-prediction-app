import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Assess Storm Damage Risk", layout="wide")

# Custom CSS for professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.2);
        color: #fff;
        text-align: center;
    }

    
    .stForm {
        background: black;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2);
    }
    
    .prediction-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        color: #10b981;
        margin: 0.5rem 0;
    }
    
    .assess-btn {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.9rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .assess-btn:hover {
        background: linear-gradient(135deg, #3730a3 0%, #5b21b6 100%);
        transform: scale(1.02);
    }
    
    .info-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4f46e5;
    }
    
    .sidebar .sidebar-content {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    </style>
""", unsafe_allow_html=True)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'damage_model_pipeline.pkl')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'damage_model_pipeline.pkl' is in the correct location.")
    st.stop()

# Title and Header
st.markdown(
    """
    <div class="main-card">
        <h1 style="margin-bottom:0.2em; font-weight:700;">⚡ Assess Storm Damage Risk</h1>
        <h3 style="font-weight:400; opacity:0.95;">Enter storm event details to receive an AI-powered property damage prediction</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for additional info
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("ℹ️ Quick Guide")
    st.write("""
    **How it works:**  
    Provide storm parameters like location, type, and duration. Our Gradient Boosting model analyzes historical data to predict potential property damage in USD.
    
    **Key Features:**  
    - Real-time predictions  
    - Based on NOAA storm events data  
    - Handles various storm types and magnitudes  
    
    **Tips:**  
    - Use realistic coordinates (e.g., lat: 39.74, lon: -104.99 for Denver)  
    - Ensure end time is after begin time  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Input form with better layout to prevent overflow
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.subheader("Enter Storm Event Details")

# You may want to load these from your CSV for consistency
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
event_types = ["Flood", "Hurricane", "Tornado", "Thunderstorm", "Winter Storm"]
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
magnitude_types = ['EG', 'MG', 'None']

with st.form("risk_form", clear_on_submit=False):
    # Row 1: State, Event Type, Month
    col1_row1, col2_row1, col3_row1 = st.columns(3)
    with col1_row1:
        state = st.selectbox("State", options=states, help="Select the affected state")
    with col2_row1:
        event_type = st.selectbox("Event Type", options=event_types, help="Type of storm event")
    with col3_row1:
        month_name = st.selectbox("Month", options=months, help="Month of the event")
    
    # Row 2: Magnitude Type, Magnitude
    col1_row2, col2_row2 = st.columns(2)
    with col1_row2:
        magnitude_type = st.selectbox("Magnitude Type", options=magnitude_types, help="Type of magnitude measurement")
    with col2_row2:
        magnitude = st.number_input("Magnitude (mph)", min_value=0.0, max_value=300.0, value=50.0, step=0.1, help="Storm intensity in miles per hour")
    
    # Row 3: Lat/Lon
    col1_row3, col2_row3 = st.columns(2)
    with col1_row3:
        begin_lat = st.number_input("Begin Latitude", min_value=-90.0, max_value=90.0, value=39.74, step=0.01, help="Starting latitude")
    with col2_row3:
        begin_lon = st.number_input("Begin Longitude", min_value=-180.0, max_value=180.0, value=-104.99, step=0.01, help="Starting longitude")
    
    # Row 4: Begin Date/Time
    col1_row4, col2_row4 = st.columns(2)
    with col1_row4:
        begin_date = st.date_input("Begin Date", value=datetime(2025, 9, 30), help="Start date of the event")
    with col2_row4:
        begin_time = st.time_input("Begin Time", value=datetime(2025, 9, 30, 10, 41).time(), help="Start time of the event")
    
    # Row 5: End Date/Time
    col1_row5, col2_row5 = st.columns(2)
    with col1_row5:
        end_date = st.date_input("End Date", value=datetime(2025, 9, 30), help="End date of the event")
    with col2_row5:
        end_time = st.time_input("End Time", value=datetime(2025, 9, 30, 12, 41).time(), help="End time of the event")

    # Submit button
    submitted = st.form_submit_button("🚀 Assess Risk", use_container_width=True, type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Process submission
if submitted:
    with st.spinner("Analyzing storm risk..."):
        begin_datetime = datetime.combine(begin_date, begin_time)
        end_datetime = datetime.combine(end_date, end_time)
        duration_hours = (end_datetime - begin_datetime).total_seconds() / 3600.0
        
        if duration_hours < 0:
            st.error("❌ End date/time must be after begin date/time.")
        else:
            input_data = pd.DataFrame([{
                "STATE": state,
                "EVENT_TYPE": event_type,
                "MONTH_NAME": month_name,
                "MAGNITUDE": magnitude,
                "MAGNITUDE_TYPE": magnitude_type,
                "BEGIN_LAT": begin_lat,
                "BEGIN_LON": begin_lon,
                "DURATION_HOURS": duration_hours
            }])
            
            try:
                pred_log = model.predict(input_data)
                pred_log = np.clip(pred_log, 0, None)
                prediction = np.expm1(pred_log)[0]
                
                # Display prediction in a professional card with two decimal places
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <h2 style="color:#1f2937; margin-bottom:1rem;">✅ Prediction Results</h2>
                        <div class="prediction-value">${prediction:,.2f}K USD</div>
                        <p style="color:#6b7280; font-size:1.1rem;">Estimated Property Damage</p>
                        <p style="color:#4f46e5; font-weight:500;">Based on provided storm parameters</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Optional: Show input summary
                with st.expander("📋 Input Summary", expanded=False):
                    st.json({
                        "State": state,
                        "Event Type": event_type,
                        "Month": month_name,
                        "Magnitude": f"{magnitude} mph ({magnitude_type})",
                        "Location": f"Lat: {begin_lat}, Lon: {begin_lon}",
                        "Duration": f"{duration_hours:.1f} hours"
                    })
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# Footer
st.markdown("---")
col_footer1, col_footer2 = st.columns([3, 1])
with col_footer1:
    st.markdown(
        '<div class="info-section">'
        '<p style="margin:0; color:#6b7280;">This tool uses a trained Gradient Boosting Regressor on historical NOAA storm data for accurate risk assessment. '
        'Predictions are estimates and should be used for planning purposes only.</p>'
        '</div>',
        unsafe_allow_html=True
    )
with col_footer2:
    st.markdown("[Back to Home](/)")