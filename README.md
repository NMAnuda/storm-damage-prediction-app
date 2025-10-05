# Storm Damage Prediction Application

This project is a Streamlit application designed to predict storm damage based on various input parameters. It utilizes a trained machine learning model to assess the risk of damage and provides users with insightful statistics and visualizations related to storm events.

## Project Structure

The project is organized as follows:

```
storm-damage-prediction-app
├── src
│   ├── app.py                # Main entry point of the Streamlit application
│   ├── pages                 # Contains different pages of the application
│   │   ├── 1_Statistics.py    # Displays statistics and visualizations
│   │   ├── 2_Assess_Risk.py   # Allows users to assess storm damage risk
│   │   └── 3_About.py         # Provides information about the application
│   ├── utils                 # Utility functions for the application
│   │   ├── __init__.py       # Initialization file for utils package
│   │   └── model_utils.py     # Functions for model loading and predictions
│   ├── assets                # Contains assets like CSS files
│   │   └── custom_theme.css   # Custom CSS for application styling
│   └── diagrams              # Contains diagrams related to the model
│       └── model_architecture.drawio # Diagram of the model architecture
├── requirements.txt          # Lists project dependencies
└── README.md                 # Documentation for the project
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd storm-damage-prediction-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application, execute the following command in your terminal:
```
streamlit run src/app.py
```

Once the application is running, you can navigate through the different pages:

- **Statistics**: View various statistics and visualizations related to storm damage predictions.
- **Assess Risk**: Input parameters to assess the risk of storm damage based on the trained model.
- **About**: Learn more about the application, its purpose, and methodology.

## Acknowledgments

This project utilizes machine learning techniques for storm damage prediction and is built using Streamlit for an interactive user experience. Special thanks to the contributors and libraries that made this project possible.

## License

This project is licensed under the MIT License - see the LICENSE file for details.