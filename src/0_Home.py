# app.py - Professional Storm Damage Prediction App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Storm Damage Prediction Platform",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS with modern design principles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(to bottom, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem 1rem;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #1e40af 0%, #7c3aed 50%, #db2777 100%);
        border-radius: 24px;
        padding: 4rem 3rem;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(30, 64, 175, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        color: white;
        text-align: center;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        line-height: 1.2;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        font-weight: 400;
        opacity: 0.95;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 2rem;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        display: block;
        margin-bottom: 0.3rem;
    }
    
    .stat-label {
        font-size: 0.95rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Feature Cards */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: black;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        display: block;
    }
    
    .feature-title {
        color: black;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        line-height: 1.3;
    }
    
    .feature-description {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }
    
    /* Buttons */
    .primary-btn {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        text-align: center;
        width: 100%;
    }
    
    .primary-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
    }
    
    .secondary-btn {
        background: white;
        color: #4f46e5;
        border: 2px solid #4f46e5;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        text-align: center;
    }
    
    .secondary-btn:hover {
        background: #4f46e5;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Info Section */
    .info-section {
        background: black;
        border-radius: 20px;
        padding: 3rem;
        margin: 3rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-left: 6px solid #4f46e5;
    }
    
     .info-title {
        color: #1e293b;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    .info-text {
        color: #475569;
        font-size: 1.1rem;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    
    /* Trust Indicators */
    .trust-section {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 20px;
        padding: 3rem;
        margin: 3rem 0;
        text-align: center;
    }
    
    .trust-title {
        color: #1e293b;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    .trust-items {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 2rem;
    }
    
    .trust-item {
        flex: 1;
        min-width: 200px;
        padding: 1.5rem;
    }
    
    .trust-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .trust-text {
        color: #475569;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* CTA Section */
    .cta-section {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 24px;
        padding: 4rem 3rem;
        margin: 4rem 0 2rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.4);
    }
    
    .cta-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .cta-text {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 2rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .cta-buttons {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .hero-stats {
            gap: 1.5rem;
        }
        
        .stat-value {
            font-size: 2rem;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .cta-title {
            font-size: 2rem;
        }
        
        .cta-buttons {
            flex-direction: column;
            padding: 0 1rem;
        }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">🌪️ Storm Damage Prediction Platform</h1>
            <p class="hero-subtitle">Enterprise-grade machine learning solution for accurate property damage risk assessment and storm impact analysis</p>
            
        
    </div>
""", unsafe_allow_html=True)

# Feature Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <h3 class="feature-title">Risk Assessment</h3>
            <p class="feature-description">Get instant, data-driven storm damage predictions powered by advanced machine learning algorithms trained on historical weather data.</p>
            <a href="/Assess_Risk" target="_self">
                <button class="primary-btn">Start Assessment →</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <h3 class="feature-title">Data Analytics</h3>
            <p class="feature-description">Explore comprehensive visualizations and statistical insights revealing storm damage patterns and trends across the United States.</p>
            <a href="/Statistics" target="_self">
                <button class="primary-btn">View Analytics →</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🔬</span>
            <h3 class="feature-title">Model Insights</h3>
            <p class="feature-description">Discover the science behind our predictions, including model architecture, feature importance analysis, and validation metrics.</p>
            <a href="/About" target="_self">
                <button class="primary-btn">Learn More →</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# Information Section
st.markdown("""
    <div class="info-section">
        <h2 class="info-title">Why Storm Damage Prediction Matters</h2>
        <p class="info-text">
            Severe weather events cause billions of dollars in property damage annually. Our platform leverages cutting-edge machine learning 
            to help property owners, insurers, and emergency management teams make informed decisions before storms strike.
        </p>
        <p class="info-text">
            By analyzing historical storm data, geographical features, and weather patterns, our model provides actionable insights 
            that enable proactive risk mitigation and resource allocation.
        </p>
    </div>
""", unsafe_allow_html=True)



# Call to Action Section
st.markdown("""
    <div class="cta-section">
        <h2 class="cta-title">Ready to Assess Your Storm Risk?</h2>
        <p class="cta-text">
            Start using our platform today to protect your property and make data-driven decisions about storm preparedness.
        </p>
        <div class="cta-buttons">
            <a href="/Assess_Risk" target="_self" style="text-decoration: none; flex: 1; max-width: 300px;">
                <button class="primary-btn">Get Started Now</button>
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #64748b; font-size: 0.9rem;">
        <p>© 2025 Storm Damage Prediction Platform. Powered by Advanced Machine Learning.</p>
    </div>
""", unsafe_allow_html=True)