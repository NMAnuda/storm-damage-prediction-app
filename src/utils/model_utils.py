# storm-damage-prediction-app/storm-damage-prediction-app/src/utils/model_utils.py

import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    """Load the trained model from the specified path."""
    model = joblib.load(model_path)
    return model

def preprocess_input(data):
    """Preprocess the input data for prediction."""
    # Assuming data is a dictionary with the required features
    df = pd.DataFrame([data])
    # Perform any necessary preprocessing steps here
    return df

def make_prediction(model, input_data):
    """Make a prediction using the loaded model and preprocessed input data."""
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return prediction

def inverse_transform(prediction):
    """Inverse transform the prediction if necessary (e.g., if log-transformed)."""
    return np.expm1(prediction)  # Assuming the prediction was log-transformed

def get_feature_importance(model, feature_names):
    """Get feature importance from the model."""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    return feature_importance.sort_values(by='Importance', ascending=False)