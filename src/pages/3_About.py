import streamlit as st

# Set the page title and layout
st.set_page_config(page_title="About the Storm Damage Prediction Model", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("About the Storm Damage Prediction Model")

# Introduction
st.header("Introduction")
st.write(
    "The Storm Damage Prediction Model is designed to estimate potential property damage "
    "caused by storms based on various input parameters. This application leverages machine "
    "learning techniques to provide accurate predictions, helping communities prepare for and "
    "mitigate the impacts of severe weather events."
)

# Methodology
st.header("Methodology")
st.write(
    "The model was developed using historical storm data, which includes features such as "
    "storm type, magnitude, duration, and geographical information. The following steps were "
    "taken in the development of the model:\n"
    "1. Data Collection: Gathering relevant storm event data.\n"
    "2. Data Preprocessing: Cleaning and transforming the data for analysis.\n"
    "3. Feature Selection: Identifying key features that influence damage.\n"
    "4. Model Training: Using Gradient Boosting Regressor to train the model on the dataset.\n"
    "5. Evaluation: Assessing model performance using cross-validation and various metrics."
)

# Acknowledgments
st.header("Acknowledgments")
st.write(
    "We would like to acknowledge the contributions of various data sources and the open-source "
    "community for providing tools and libraries that made this project possible. Special thanks "
    "to the developers of Streamlit for creating an intuitive platform for building web applications."
)

# Diagram
st.header("Model Architecture")
st.info("Model Architecture Diagram would be displayed here. (Image: diagrams/model_architecture.png)")
# st.image("diagrams/model_architecture.png", caption="Model Architecture Diagram", use_column_width=True)

