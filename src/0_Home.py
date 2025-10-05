# app.py - Main entry point for the Streamlit app with top navigation
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit.components.v1 as components

# Custom CSS for modern, consistent styling across the app
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Card Styles */
    .main-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.2);
        color: #fff;
        text-align: center;
    }
    
    .feature-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        margin: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        color: #1f2937;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .feature-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 1rem;
        cursor: pointer;
        transition: all 0.2s;
        text-decoration: none;
        display: inline-block;
    }
    
    .feature-btn:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        transform: scale(1.05);
    }
    
    .get-started-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        color: #1f2937;
        text-align: center;
    }
    
    .get-started-btn {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.9rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
        text-decoration: none;
        display: inline-block;
    }
    
    .get-started-btn:hover {
        background: linear-gradient(135deg, #3730a3 0%, #5b21b6 100%);
        transform: scale(1.05);
    }
    
    /* Form Styling */
    .stForm {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Metrics and Success Messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    
    /* Sidebar Replacement - Inputs in Main for Assess Risk */
    .input-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Visualization Containers */
    .viz-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Home Content
st.markdown(
    """
    <div class="main-card">
        <h1 style="margin-bottom:0.2em; font-weight:700;">🌪️ Storm Damage Prediction App</h1>
        <h3 style="font-weight:400; opacity:0.95;">Advanced ML-powered tool for accurate storm property damage risk assessment</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Feature cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="feature-card">
            <h3 style="color:#4f46e5; margin-bottom:1rem;">⚡ Risk Assessment</h3>
            <p style="color:#6b7280;">Get instant storm damage predictions based on event and location parameters using a robust ML pipeline.</p>
            <a href="/Assess_Risk" target="_self"><button class="feature-btn">Start Assessment</button></a>

        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div class="feature-card">
            <h3 style="color:#4f46e5; margin-bottom:1rem;">📈 Data Visualization</h3>
            <p style="color:#6b7280;">Explore interactive charts and statistics showing patterns in storm damage data across the US.</p>
            <a href="/Statistics" target="_self"><button class="feature-btn">View Statistics</button></a>

        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        """
        <div class="feature-card">
            <h3 style="color:#4f46e5; margin-bottom:1rem;">🔬 About the Model</h3>
            <p style="color:#6b7280;">Understand how the ML model works, feature importance, and the science behind storm damage predictions.</p>
            <a href="/About" target="_self"><button class="feature-btn">Learn More</button></a>       
        </div>
        """,
        unsafe_allow_html=True
    )
# redeploy fix
