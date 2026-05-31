import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Rainfall Predictor Engine", layout="centered")

st.title("🌧️ Rainfall Prediction Dashboard")
st.write("Enter local parameters below. The model trains live on the server to ensure version stability.")

# ==========================================
# PRODUCTION BACKEND: LIVE MODEL BUILDER
# ==========================================
@st.cache_resource
def train_and_cache_model():
    # Load the original weather dataset straight from a reliable web address
    url = "https://raw.githubusercontent.com/rattle-package/weatherAUS/master/data/weatherAUS.csv"
    df = pd.read_csv(url)
    
    # 1. Clean data using your exact original notebook rules
    if df['RainToday'].isnull().any():
        df['RainToday'] = df['RainToday'].fillna(df['RainToday'].mode()[0])
    if df['RainTomorrow'].isnull().any():
        df['RainTomorrow'] = df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
        
    features_list = [
        'MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed',
        'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm',
        'Temp9am','Temp3pm'
    ]
    
    X = df[features_list].copy()
    X['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
    
    # Fit the imputer on the real server environment
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    y = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
    
    # Fit the model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_imputed, y)
    
    return model, imputer

# Initialize the models quietly in the background
with st.spinner("Initializing ML pipeline components on the server..."):
    model, imputer = train_and_cache_model()

# ==========================================
# FRONTEND UI LAYOUT: INTERACTIVE SLIDERS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    min_temp = st.slider("Minimum Temperature (°C)", -5.0, 45.0, 15.0)
    max_temp = st.slider("Maximum Temperature (°C)", 0.0, 50.0, 26.0)
    rainfall = st.slider("Rainfall Today (mm)", 0.0, 100.0, 0.0)
    evap = st.slider("Evaporation (mm)", 0.0, 20.0, 5.0)
    sunshine = st.slider("Sunshine Hours", 0.0, 15.0, 8.0)
    wind_speed = st.slider("Wind Gust Speed (km/h)", 0.0, 130.0, 40.0)
    rain_today_raw = st.selectbox("Did it rain today?", ["No", "Yes"])

with col2:
    hum_9am = st.slider("Humidity at 9 AM (%)", 0.0, 100.0, 60.0)
    hum_3pm = st.slider("Humidity at 3 PM (%)", 0.0, 100.0, 50.0)
    press_9am = st.slider("Pressure at 9 AM (hPa)", 980.0, 1040.0, 1015.0)
    press_3pm = st.slider("Pressure at 3 PM (hPa)", 980.0, 1040.0, 1012.0)
    cloud_9am = st.slider("Cloud Cover at 9 AM (oktas)", 0.0, 9.0, 4.0)
    cloud_3pm = st.slider("Cloud Cover at 3 PM (oktas)", 0.0, 9.0, 4.0)
    temp_9am = st.slider("Temperature at 9 AM (°C)", -5.0, 45.0, 18.0)
    temp_3pm = st.slider("Temperature at 3 PM (°C)", -5.0, 45.0, 24.0)

rain_today_mapped = 1 if rain_today_raw == "Yes" else 0

input_data = np.array([[
    min_temp, max_temp, rainfall, evap, sunshine, wind_speed,
    hum_9am, hum_3pm, press_9am, press_3pm, cloud_9am, cloud_3pm,
    temp_9am, temp_3pm, rain_today_mapped
]])

st.markdown("---")

if st.button("Evaluate Forecasting Model", type="primary"):
    clean_input = imputer.transform(input_data)
    prediction = model.predict(clean_input)
    prediction_proba = model.predict_proba(clean_input)[0][1]
    
    if prediction[0] == 1:
        st.error(f"🌧️ **Rain Expected Tomorrow!** (Confidence: {prediction_proba * 100:.1f}%)")
    else:
        st.success(f"☀️ **No Rain Expected Tomorrow.** (Confidence: {(1 - prediction_proba) * 100:.1f}%)")
