
import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
from utils import load_data, preprocess_for_catboost, prepare_xy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bike Sharing Prediction", layout="wide")
st.title("🚴 Bike Sharing Rental Predictor")

@st.cache_resource
def train_all_models():
    """Run preprocessing and training once, cache the results"""
    with st.spinner("🔄 Loading and preprocessing data..."):
        df = load_data()
        df_p, scaler = preprocess_for_catboost(df, drop_atemp=True, scale_numeric=True, add_cyclic=True)
        X, y = prepare_xy(df_p)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    with st.spinner("Training models in background (this may take a minute)..."):
        models = {}
        
        # Linear Regression
        models['Linear'] = LinearRegression().fit(X_train, y_train)
        
        # Random Forest
        models['RandomForest'] = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
        
        # Gradient Boosting
        models['GradientBoosting'] = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42).fit(X_train, y_train)
        
        # CatBoost
        cat_features = ['season','mnth','hr','weekday','weathersit','holiday','workingday','yr']
        cat_features = [c for c in cat_features if c in X_train.columns]
        cat_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8, loss_function='RMSE', random_state=42, verbose=0)
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        cat_model.fit(train_pool)
        models['CatBoost'] = cat_model
    
    return models, scaler, X.columns.tolist()

# Load models and scaler
models, scaler, feature_cols = train_all_models()


st.markdown("---")
st.subheader("📊 Make a Prediction")

df_sample = load_data()
default = df_sample.iloc[0].to_dict()

# User input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        hr = st.selectbox("⏰ Hour (0-23)", options=list(range(24)), index=int(default.get('hr',12)))
        season = st.selectbox("🌡️ Season", options=[('Winter', 1), ('Spring', 2), ('Summer', 3), ('Fall', 4)], format_func=lambda x: x[0])
        mnth = st.selectbox("📅 Month (1-12)", options=list(range(1,13)), index=int(default.get('mnth',1))-1)
        weathersit = st.selectbox("☁️ Weather", options=[('Clear', 1), ('Mist', 2), ('Rain', 3), ('Heavy Rain', 4)], format_func=lambda x: x[0])
        temp = st.slider("🌡️ Temperature (normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    with col2:
        holiday = st.selectbox("🎉 Holiday", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
        weekday = st.selectbox("📆 Weekday (0=Sun, 6=Sat)", options=list(range(7)), index=int(default.get('weekday',0)))
        workingday = st.selectbox("💼 Working Day", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
        hum = st.slider("💧 Humidity (normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        windspeed = st.slider("💨 Windspeed (normalized)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    
    model_choice = st.selectbox(" Select Model", ["CatBoost", "RandomForest", "GradientBoosting", "Linear"], help="CatBoost typically performs best")
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    try:
        # Prepare input
        row = {
            'dteday': pd.to_datetime(default['dteday']),
            'season': season[1], 'yr': int(default['yr']), 'mnth': mnth, 'hr': hr,
            'holiday': holiday[0], 'weekday': weekday, 'workingday': workingday[0],
            'weathersit': weathersit[1], 'temp': temp, 'hum': hum, 'windspeed': windspeed,
            'casual': 0, 'registered': 0, 'cnt': 0
        }
        
        input_df = pd.DataFrame([row])
        df_p, _ = preprocess_for_catboost(input_df, drop_atemp=True, scale_numeric=True, add_cyclic=True)
        X_input, _ = prepare_xy(df_p, target='cnt', drop_cols_extra=['casual','registered'])
        
        # Ensure feature consistency
        for col in feature_cols:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[feature_cols]
        
        # Make prediction
        model = models[model_choice]
        pred = float(model.predict(X_input)[0])
        
        # Display result
        st.markdown("---")
        st.success("✅ Prediction Complete!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Used", model_choice)
        with col2:
            st.metric("Predicted Rentals", f"{pred:.0f}")
        with col3:
            st.metric("Status", "✓ Ready" if pred > 0 else "⚠ Check input")
        
        # Confidence indicator
        if pred > 1000:
            st.info(f"📈 High demand expected: {pred:.0f} rentals")
        elif pred > 500:
            st.info(f"📊 Moderate demand expected: {pred:.0f} rentals")
        else:
            st.info(f"📉 Low demand expected: {pred:.0f} rentals")
            
    except Exception as e:
        st.error(f" Prediction failed: {str(e)}")
        st.write("Please check your inputs and try again.")



st.markdown("---")
st.caption("Tip: Use CatBoost for best predictions. All models trained on historical bike sharing data.")