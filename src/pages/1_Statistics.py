import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the page config for better layout
st.set_page_config(page_title="Storm Damage Prediction Statistics", layout="wide")

# Set a beautiful seaborn style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12

# Set the title of the page
st.title("🌪️ Storm Damage Prediction Statistics")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('StormEvents_cleaned.csv')
    # Convert DAMAGE_PROPERTY to numeric (assuming format like '10.00K' meaning 10,000 USD)
    if 'DAMAGE_PROPERTY' in df.columns:
        df['DAMAGE_PROPERTY'] = pd.to_numeric(
            df['DAMAGE_PROPERTY'].astype(str).str.replace('K', '').str.replace('$', '').str.strip(), 
            errors='coerce'
        ) * 1000
    return df

df = load_data()

# Display basic statistics
st.header("📊 Basic Statistics")
st.write("Here are some basic statistics about the storm damage dataset:")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Events", f"{len(df):,}")
    if 'DAMAGE_PROPERTY' in df.columns:
        st.metric("Avg Property Damage", f"${df['DAMAGE_PROPERTY'].mean()/1000 :,.0f}K")
with col2:
    if 'DAMAGE_PROPERTY' in df.columns:
        st.metric("Max Damage", f"${df['DAMAGE_PROPERTY'].max()/1000 :,.0f}K")
    st.metric("Unique States", df['STATE'].nunique() if 'STATE' in df.columns else 0)
numeric_cols = df.select_dtypes(include=[np.number])
if not numeric_cols.empty:
    st.dataframe(numeric_cols.describe().style.format("{:.2f}").background_gradient(cmap='viridis'), use_container_width=True)
else:
    st.warning("No numeric columns found for description.")

# Visualizations
st.header("📈 Visualizations")


# Event Type Count - Limit to top 10 for better visualization
if 'EVENT_TYPE' in df.columns:
    st.subheader("Top 10 Storm Event Types")
    event_counts = df['EVENT_TYPE'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("viridis", len(event_counts))
    sns.barplot(y=event_counts.index, x=event_counts.values, ax=ax, palette=colors)
    ax.set_title("Top 10 Storm Event Types by Count", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Count", fontsize=14)
    ax.set_ylabel("Event Type", fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('gray')
    ax.spines['left'].set_linewidth(0.5)
    # Add value labels on bars
    for i, v in enumerate(event_counts.values):
        ax.text(v + max(event_counts.values) * 0.01, i, str(v), va='center', fontsize=11)
    st.pyplot(fig)

# Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty and len(numeric_df.columns) > 1:
    st.subheader("Feature Correlation Heatmap")
    correlation_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, 
                square=True, ax=ax, cbar_kws={'shrink': 0.8}, mask=mask,
                annot_kws={"fontsize": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
else:
    st.warning("Insufficient numeric columns for correlation heatmap.")

# Conclusion
st.header("Insights")
st.write("The statistics and visualizations above provide valuable insights into the storm damage dataset, highlighting key patterns in event types, damage distributions, and feature relationships to better understand factors affecting storm damage predictions.")