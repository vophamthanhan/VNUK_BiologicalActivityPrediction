import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        text-align: center;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox>div {
        background-color: #e8f5e9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load pre-trained models and scaler
scaler = joblib.load('C:/Users/OS/Desktop/BiologicalActivityPrediction_20040007_VoPhamThanhAn/models/scaler.pkl')
gnb_model = joblib.load('C:/Users/OS/Desktop/BiologicalActivityPrediction_20040007_VoPhamThanhAn/models/gaussian_nb.pkl')
xgb_model = joblib.load('C:/Users/OS/Desktop/BiologicalActivityPrediction_20040007_VoPhamThanhAn/models/xgboost.pkl')
ann_model = load_model('C:/Users/OS/Desktop/BiologicalActivityPrediction_20040007_VoPhamThanhAn/models/ann.h5', compile=False)
ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Set up Streamlit app title with style
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Biological Activity Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict activities based on compound data</h4>", unsafe_allow_html=True)
st.write("---")

# File uploader for user to upload data
st.markdown("<h3 style='color: #2e7d32;'>Upload Your Dataset</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file with compound data:", type=["csv"])

if uploaded_file:
    # Read and display uploaded data
    input_data = pd.read_csv(uploaded_file)
    st.markdown("<h3 style='color: #2e7d32;'>Uploaded Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(input_data)

    try:
        # Ensure required columns are present
        required_columns = ['Compounds', 'Type', 'Tax ID', 'Organism']
        missing_columns = [col for col in required_columns if col not in input_data.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Preprocess input data
            input_data['Type'] = input_data['Type'].astype('category').cat.codes
            input_data['Organism'] = input_data['Organism'].astype('category').cat.codes

            # Handle missing values by replacing with column mean
            if input_data.isnull().values.any():
                st.warning("Dataset contains NaN values. They will be replaced with column means.")
                input_data = input_data.fillna(input_data.mean())

            # Normalize the input data
            X_input = scaler.transform(input_data[['Compounds', 'Type', 'Tax ID', 'Organism']])

            # Model selection with improved UI
            st.markdown("<h3 style='color: #2e7d32;'>Select Prediction Model</h3>", unsafe_allow_html=True)
            model_option = st.selectbox("Choose a model:", ["ANN", "GaussianNB", "XGBoost"])
            predictions = None
            if model_option == "ANN":
                predictions = ann_model.predict(X_input).flatten()
            elif model_option == "GaussianNB":
                predictions = gnb_model.predict(X_input)
            else:
                predictions = xgb_model.predict(X_input)

            # Add prediction probability (for ANN and XGBoost models)
            if model_option in ["ANN", "XGBoost"]:
                probas = np.clip(predictions / predictions.sum() * 100, 0, 100)
                input_data['Prediction (%)'] = probas
            else:
                input_data['Prediction (%)'] = None

            # Display predictions
            st.markdown("<h3 style='color: #2e7d32;'>Predictions</h3>", unsafe_allow_html=True)
            input_data['Predicted Activity'] = predictions

            # Generate conclusion for each prediction
            def generate_conclusion(row):
                activity = row['Predicted Activity']
                if activity >= 80:
                    return "High Activity"
                elif activity >= 50:
                    return "Moderate Activity"
                else:
                    return "Low Activity"

            input_data['Conclusion'] = input_data.apply(generate_conclusion, axis=1)
            st.dataframe(input_data[['Compounds', 'Predicted Activity', 'Prediction (%)', 'Conclusion']])

            # Visualization: Distribution of predictions
            st.markdown("<h3 style='color: #2e7d32;'>Prediction Distribution</h3>", unsafe_allow_html=True)
            plt.figure(figsize=(8, 4))
            sns.histplot(predictions, kde=True, bins=20, color="green")
            plt.title("Prediction Distribution")
            plt.xlabel("Predicted Activity")
            plt.ylabel("Frequency")
            st.pyplot(plt)

            # Additional stats for prediction percentages
            if model_option in ["ANN", "XGBoost"]:
                st.markdown("<h3 style='color: #2e7d32;'>Statistics for Prediction Percentages</h3>", unsafe_allow_html=True)
                st.write(input_data[['Prediction (%)']].describe())

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Sidebar for additional options
st.sidebar.markdown("<h3 style='color: #4CAF50;'>About</h3>", unsafe_allow_html=True)
st.sidebar.write("""
This app predicts biological activity for compounds based on provided features:
- **Compounds**: Quantitative measure of compound data
- **Type**: Encoded as categorical
- **Tax ID**: Taxonomy ID of the organism
- **Organism**: Encoded as categorical
""")
st.sidebar.write("Created with Streamlit and Machine Learning models.")