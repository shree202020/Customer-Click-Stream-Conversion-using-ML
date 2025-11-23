import pandas as pd
import pickle
import numpy as np
import streamlit as st
from PIL import Image
import time 

# **✅ Move set_page_config() to the very first line**
st.set_page_config(
    page_title="E-Commerce Clickstream Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Load the models and scalers
from joblib import load

@st.cache_resource
def load_models():
    try:
        reg_model = load('model_reg.pkl')
        clf_model = load('logistic_reg_clss.pkl')
        scaler_reg = load('scalerfit_regression.pkl')
        scaler_clf = load('scalerfit_classification.pkl')
        return clf_model, reg_model, scaler_reg, scaler_clf
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        st.info("Please ensure all required files are in the same directory as the script")
        st.stop()


# Preprocess data function
def preprocess_data(df, scaler=None, features=None):
    if 'page2_clothing_model' in df.columns:
        df['page2_clothing_model'] = df['page2_clothing_model'].astype(str).str.extract(r'(\d+)').astype(float)
    
    if scaler is not None and features is not None:
        scaled_features = scaler.transform(df[features])
        df_scaled = pd.DataFrame(scaled_features, columns=features)
        return df_scaled
    return df

# Load models and scalers
clf_model, reg_model, scaler_reg, scaler_clf = load_models()

# Mappings
MAIN_CATEGORY_MAPPING = {"Trousers": 1, "Skirts": 2, "Blouses": 3, "Sale": 4}
PHOTO_MAPPING = {'En Face': 1, 'Profile': 2}
COLOR_MAPPING = {
    'Beige': 1, 'Black': 2, 'Blue': 3, 'Brown': 4, 'Burgundy': 5,
    'Gray': 6, 'Green': 7, 'Navy Blue': 8, 'Many Colors': 9,
    'Olive': 10, 'Pink': 11, 'Red': 12, 'Violet': 13, 'White': 14
}


# Title and description
st.title("🛍️ E-Commerce Clickstream Prediction")
st.markdown("### Predicts customer behavior using machine learning")

# Sidebar for model selection
model_type = st.sidebar.radio("Choose Model Type", ["Classification", "Regression"], help="Select the type of prediction you want to make")

# Feature definitions - Updated to include colour for regression
reg_features = ['page2_clothing_model', 'page1_main_category', 'model_photography', 'colour']
clf_features = ['page1_main_category', 'price', 'page2_clothing_model', 'colour', 'model_photography']

# Single Prediction Tab
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
with tab1:
    st.header("Individual Prediction")
    
    if model_type == "Classification":
        col1, col2 = st.columns(2)
        with col1:
            main_category = st.selectbox("Main Category", list(MAIN_CATEGORY_MAPPING.keys()), key='clf_category')
            clothing_model = st.number_input("Clothing Model ID", min_value=1, max_value=50, value=10, key='clf_model_id')
        with col2:
            price = st.number_input("Price", min_value=0, max_value=1000, value=100)
            colour = st.selectbox("Colour", list(COLOR_MAPPING.keys()), key='clf_colour')
            photography = st.selectbox("Photography Style", list(PHOTO_MAPPING.keys()), key='clf_photo')
        if st.button("🎯Predict Category", key="single_predict_clf"):
            input_data = pd.DataFrame({
                'page1_main_category': [MAIN_CATEGORY_MAPPING[main_category]],
                'page2_clothing_model': [clothing_model],
                'price': [price],
                'colour': [COLOR_MAPPING[colour]],
                'model_photography': [PHOTO_MAPPING[photography]]
            })
            input_data_scaled = preprocess_data(input_data, scaler_clf, clf_features)
            with st.spinner('Calculating prediction...'):
                prediction = clf_model.predict(input_data_scaled)[0]
                probability = clf_model.predict_proba(input_data_scaled)
                prediction_mapping = {1: "Going To Buy", 2: "Not Going To Buy"}
                prediction_text = prediction_mapping.get(prediction, "Unknown" )           
                st.success(f"Predicted Category: {prediction_text}")
                st.info(f"Prediction Confidence: {max(probability[0]) * 100:.2f}%")
    else:
        col1, col2 = st.columns(2)
        with col1:
            main_category = st.selectbox("Main Category", list(MAIN_CATEGORY_MAPPING.keys()), key='reg_category')
            clothing_model = st.number_input("Clothing Model ID", min_value=1, max_value=50, value=10, key='reg_model_id')
        with col2:
            photography = st.selectbox("Photography Style", list(PHOTO_MAPPING.keys()), key='reg_photo')
            colour = st.selectbox("Colour", list(COLOR_MAPPING.keys()), key='reg_colour')
        if st.button("🎯Predict Price", key="single_predict_reg"):
            input_data = pd.DataFrame({
                'page2_clothing_model': [clothing_model],
                'page1_main_category': [MAIN_CATEGORY_MAPPING[main_category]],
                'model_photography': [PHOTO_MAPPING[photography]],
                'colour': [COLOR_MAPPING[colour]]
            })
            input_data_scaled = preprocess_data(input_data, scaler_reg, reg_features)
            with st.spinner('Calculating prediction...'):
                prediction = reg_model.predict(input_data_scaled)
                st.success(f"Predicted Price: ${prediction[0]:.2f}")

# Batch Prediction Tab
with tab2:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)
        if model_type == "Classification":
            df_scaled = preprocess_data(df[clf_features], scaler_clf, clf_features)
            predictions = clf_model.predict(df_scaled)
            probabilities = clf_model.predict_proba(df_scaled)
            df['Predicted_Category'] = predictions
            df['Confidence'] = np.max(probabilities, axis=1)
        else:
            # Ensure all required features are present for regression
            for feature in reg_features:
                if feature not in df.columns:
                    st.error(f"Missing required feature: {feature}")
                    st.stop()
            df_scaled = preprocess_data(df[reg_features], scaler_reg, reg_features)
            df['Predicted_Price'] = reg_model.predict(df_scaled)
        st.subheader("🎯Predictions")
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button("Download predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
# Sidebar additional information
with st.sidebar:
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This app Predicts customer behavior using machine learning models.
    - Classification: Predicts price category
    - Regression: Predicts exact price
    """)
    st.markdown("---")
    st.markdown("### Model Features")
    if model_type == "Classification":
        st.markdown("""
        Required features:
        - Main Category
        - Price
        - Clothing Model ID
        - Color
        - Photography Style
        """)
    else:
        st.markdown("""
        Required features:
        - Clothing Model ID
        - Main Category
        - Photography Style
        - Color
        """)