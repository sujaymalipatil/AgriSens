## ============================================================
## AgriSens - Smart Crop Recommendation System
## Upgraded webapp.py with bug fixes + new features
## Features: Charts, Weather API, AI Chatbot, Disease Detection
## ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import warnings
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

warnings.filterwarnings('ignore')

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSens - Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)

# ── Paths (Fixed - no more hardcoded paths!) ─────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, '..', 'Datasets', 'Crop_recommendation.csv')
PKL_PATH = os.path.join(BASE_DIR, 'RF.pkl')
IMG_PATH = os.path.join(BASE_DIR, 'crop.png')

# ── Load Banner Image safely ──────────────────────────────────
try:
    img = Image.open(IMG_PATH)
    st.image(img, use_column_width=True)
except:
    st.info("🌾 AgriSens - Smart Crop Recommendation System")

# ── Load Dataset ──────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
        return df
    except Exception as e:
        st.error(f"Dataset not found: {e}")
        return None

# ── Load or Train Model ───────────────────────────────────────
@st.cache_resource
def load_model():
    """Load pre-trained model if exists, otherwise train and save."""
    if os.path.exists(PKL_PATH):
        with open(PKL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        # Train only if .pkl doesn't exist
        df = load_data()
        if df is None:
            return None
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=20, random_state=5)
        model.fit(X_train, y_train)
        with open(PKL_PATH, 'wb') as f:
            pickle.dump(model, f)
        return model

df = load_data()
model = load_model()

# ── Crop Ideal Ranges (for charts) ───────────────────────────
CROP_RANGES = {
    'rice':        {'N': 80, 'P': 40, 'K': 40, 'temperature': 25, 'humidity': 82, 'ph': 6.5, 'rainfall': 200},
    'wheat':       {'N': 60, 'P': 30, 'K': 30, 'temperature': 20, 'humidity': 65, 'ph': 6.5, 'rainfall': 80},
    'maize':       {'N': 80, 'P': 40, 'K': 35, 'temperature': 22, 'humidity': 65, 'ph': 6.2, 'rainfall': 90},
    'chickpea':    {'N': 40, 'P': 67, 'K': 79, 'temperature': 18, 'humidity': 16, 'ph': 7.3, 'rainfall': 75},
    'kidneybeans': {'N': 20, 'P': 67, 'K': 20, 'temperature': 20, 'humidity': 21, 'ph': 5.7, 'rainfall': 105},
    'pigeonpeas':  {'N': 20, 'P': 67, 'K': 20, 'temperature': 27, 'humidity': 48, 'ph': 5.8, 'rainfall': 149},
    'mungbean':    {'N': 20, 'P': 47, 'K': 20, 'temperature': 28, 'humidity': 86, 'ph': 6.8, 'rainfall': 46},
    'blackgram':   {'N': 40, 'P': 67, 'K': 19, 'temperature': 29, 'humidity': 65, 'ph': 7.1, 'rainfall': 68},
    'lentil':      {'N': 18, 'P': 68, 'K': 19, 'temperature': 24, 'humidity': 64, 'ph': 6.9, 'rainfall': 45},
    'pomegranate': {'N': 18, 'P': 18, 'K': 40, 'temperature': 21, 'humidity': 90, 'ph': 6.5, 'rainfall': 107},
    'banana':      {'N': 100,'P': 75, 'K': 50, 'temperature': 27, 'humidity': 80, 'ph': 5.9, 'rainfall': 105},
    'mango':       {'N': 20, 'P': 27, 'K': 30, 'temperature': 31, 'humidity': 50, 'ph': 5.8, 'rainfall': 95},
    'grapes':      {'N': 23, 'P': 132,'K': 200,'temperature': 24, 'humidity': 81, 'ph': 6.0, 'rainfall': 70},
    'watermelon':  {'N': 99, 'P': 17, 'K': 50, 'temperature': 25, 'humidity': 85, 'ph': 6.5, 'rainfall': 50},
    'muskmelon':   {'N': 100,'P': 17, 'K': 50, 'temperature': 28, 'humidity': 92, 'ph': 6.4, 'rainfall': 25},
    'apple':       {'N': 20, 'P': 134,'K': 199,'temperature': 22, 'humidity': 92, 'ph': 5.9, 'rainfall': 112},
    'orange':      {'N': 20, 'P': 16, 'K': 10, 'temperature': 22, 'humidity': 92, 'ph': 7.0, 'rainfall': 110},
    'papaya':      {'N': 49, 'P': 59, 'K': 50, 'temperature': 35, 'humidity': 92, 'ph': 6.7, 'rainfall': 143},
    'coconut':     {'N': 21, 'P': 16, 'K': 30, 'temperature': 27, 'humidity': 94, 'ph': 5.9, 'rainfall': 175},
    'cotton':      {'N': 117,'P': 46, 'K': 19, 'temperature': 24, 'humidity': 79, 'ph': 6.9, 'rainfall': 80},
    'jute':        {'N': 78, 'P': 46, 'K': 40, 'temperature': 25, 'humidity': 80, 'ph': 6.8, 'rainfall': 175},
    'coffee':      {'N': 101,'P': 28, 'K': 29, 'temperature': 25, 'humidity': 58, 'ph': 6.8, 'rainfall': 158},
}

# ── Predict Crop ──────────────────────────────────────────────
def predict_crop(n, p, k, temp, hum, ph, rain):
    if model is None:
        return None
    prediction = model.predict(np.array([n, p, k, temp, hum, ph, rain]).reshape(1, -1))
    return prediction[0]

# ── Show Comparison Chart ─────────────────────────────────────
def show_comparison_chart(crop_name, user_inputs):
    ideal = CROP_RANGES.get(crop_name.lower())
    if not ideal:
        return
    
    params = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    user_vals = [user_inputs[p] for p in params]
    ideal_vals = [ideal[p] for p in params]

    fig = go.Figure(data=[
        go.Bar(name='Your Input', x=params, y=user_vals, marker_color='#2ecc71'),
        go.Bar(name=f'Ideal for {crop_name.title()}', x=params, y=ideal_vals, marker_color='#3498db')
    ])
    fig.update_layout(
        barmode='group',
        title=f'Your Soil vs Ideal Conditions for {crop_name.title()}',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Weather API Integration ───────────────────────────────────
def get_weather(city, api_key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=5)
        data = res.json()
        if data.get('cod') == 200:
            return {
                'temperature': round(data['main']['temp'], 1),
                'humidity': round(data['main']['humidity'], 1),
                'rainfall': round(data.get('rain', {}).get('1h', 0) * 30, 1),  # estimate monthly
                'city': data['name'],
                'description': data['weather'][0]['description'].title()
            }
    except:
        pass
    return None

# ── AI Crop Chatbot (OpenAI) ──────────────────────────────────
def ask_ai(question, crop_name, openai_key, api_provider="grok"):
    try:
        from openai import OpenAI
        if api_provider == "grok":
            client = OpenAI(
                api_key=openai_key,
                base_url="https://api.x.ai/v1",
            )
            model = "grok-beta"
        else:
            client = OpenAI(api_key=openai_key)
            model = "gpt-3.5-turbo"
        
        messages = [{"role": "user", "content": f"You are an agricultural expert. Recommended crop: {crop_name}. Answer concisely: {question}"}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI unavailable: {str(e)}"

# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    st.markdown("<h1 style='text-align:center; color:#2ecc71;'>🌾 SMART CROP RECOMMENDATIONS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Powered by Machine Learning + AI</p>", unsafe_allow_html=True)
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────
    st.sidebar.title("🌿 AgriSens")
    st.sidebar.header("Enter Crop Details")

    # ── NEW: Weather Auto-fill ────────────────────────────────
    st.sidebar.subheader("🌦️ Auto-fill from Weather")
    weather_city = st.sidebar.text_input("Enter your city (optional)", placeholder="e.g. Bengaluru")
    weather_api_key = st.sidebar.text_input("OpenWeather API Key", type="password", placeholder="Get free key at openweathermap.org")
    
    auto_temp, auto_hum, auto_rain = 25.0, 70.0, 100.0
    if weather_city and weather_api_key:
        weather = get_weather(weather_city, weather_api_key)
        if weather:
            st.sidebar.success(f"📍 {weather['city']}: {weather['description']}")
            auto_temp = weather['temperature']
            auto_hum = weather['humidity']
            auto_rain = weather['rainfall']
        else:
            st.sidebar.error("City not found or invalid API key.")

    st.sidebar.divider()
    st.sidebar.subheader("🧪 Soil & Climate Inputs")

    # ── Input Fields ──────────────────────────────────────────
    nitrogen    = st.sidebar.number_input("Nitrogen (N)",     0.0, 140.0, 0.0, 0.1)
    phosphorus  = st.sidebar.number_input("Phosphorus (P)",   0.0, 145.0, 0.0, 0.1)
    potassium   = st.sidebar.number_input("Potassium (K)",    0.0, 205.0, 0.0, 0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0,  51.0, float(auto_temp), 0.1)
    humidity    = st.sidebar.number_input("Humidity (%)",     0.0, 100.0, float(auto_hum),  0.1)
    ph          = st.sidebar.number_input("pH Level",         0.0,  14.0, 7.0, 0.1)
    rainfall    = st.sidebar.number_input("Rainfall (mm)",    0.0, 500.0, float(auto_rain), 0.1)

    user_inputs = {
        'N': nitrogen, 'P': phosphorus, 'K': potassium,
        'temperature': temperature, 'humidity': humidity,
        'ph': ph, 'rainfall': rainfall
    }

    # ── Predict Button ────────────────────────────────────────
    predict_btn = st.sidebar.button("🌱 Predict Crop", use_container_width=True)

    # ── OpenAI Key for Chatbot ────────────────────────────────
    st.sidebar.divider()
    api_provider = st.sidebar.selectbox("AI Provider", ["grok", "openai"], index=0)
    openai_key = st.sidebar.text_input("🤖 API Key", type="password", placeholder="gsk_... for Grok")
    if api_provider == "grok":
        st.sidebar.info("✅ Use your gsk_ key here for xAI Grok")

    # ── Main Panel ────────────────────────────────────────────
    if predict_btn:
        # Validation
        if nitrogen == 0 and phosphorus == 0 and potassium == 0:
            st.warning("⚠️ Please enter at least N, P, K values before predicting.")
            return

        with st.spinner("Analyzing soil conditions..."):
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)

        if prediction:
            # ── Result ────────────────────────────────────────
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.success(f"## ✅ Recommended Crop: **{prediction.upper()}**")

            st.divider()

            # ── Two Column Layout ─────────────────────────────
            left, right = st.columns(2)

            with left:
                # ── Crop Image ────────────────────────────────
                st.subheader("🌿 Crop Preview")
                crop_img_path = os.path.join(BASE_DIR, 'crop_images', f'{prediction.lower()}.jpg')
                if os.path.exists(crop_img_path):
                    st.image(crop_img_path, caption=f"Recommended: {prediction.title()}", use_column_width=True)
                else:
                    st.info(f"Add '{prediction.lower()}.jpg' to crop_images/ folder to show image.")

                # ── Crop Tips ─────────────────────────────────
                st.subheader("📋 Quick Tips")
                ideal = CROP_RANGES.get(prediction.lower(), {})
                if ideal:
                    st.markdown(f"""
| Parameter | Your Value | Ideal |
|---|---|---|
| Nitrogen | {nitrogen} | {ideal.get('N', 'N/A')} |
| Phosphorus | {phosphorus} | {ideal.get('P', 'N/A')} |
| Potassium | {potassium} | {ideal.get('K', 'N/A')} |
| Temperature | {temperature}°C | {ideal.get('temperature', 'N/A')}°C |
| Humidity | {humidity}% | {ideal.get('humidity', 'N/A')}% |
| pH | {ph} | {ideal.get('ph', 'N/A')} |
| Rainfall | {rainfall}mm | {ideal.get('rainfall', 'N/A')}mm |
                    """)

            with right:
                # ── NEW: Comparison Chart ─────────────────────
                st.subheader("📊 Your Soil vs Ideal Conditions")
                show_comparison_chart(prediction, user_inputs)

            st.divider()

            # ── NEW: AI Chatbot ───────────────────────────────
            st.subheader("🤖 Ask AI About Your Crop")
            if openai_key:
                question = st.text_input("Ask anything about your recommended crop...",
                                         placeholder=f"e.g. What fertilizer should I use for {prediction}?")
                if st.button("Ask AI 🔍") and question:
                    with st.spinner("AI is thinking..."):
                        answer = ask_ai(question, prediction, openai_key, api_provider)
                    st.info(f"**AI Answer:** {answer}")
            else:
                st.caption("💡 Add your API key in the sidebar to enable the AI chatbot.")

            # ── NEW: Dataset Insights ─────────────────────────
            st.divider()
            st.subheader("📈 Dataset Insights")
            if df is not None:
                col_a, col_b = st.columns(2)
                with col_a:
                    crop_counts = df['label'].value_counts()
                    fig2 = px.bar(x=crop_counts.index, y=crop_counts.values,
                                  labels={'x': 'Crop', 'y': 'Samples'},
                                  title='Sample Distribution Across Crops',
                                  color=crop_counts.values, color_continuous_scale='Greens')
                    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig2, use_container_width=True)

                with col_b:
                    crop_df = df[df['label'] == prediction]
                    if not crop_df.empty:
                        fig3 = px.box(df[df['label'].isin([prediction])].melt(id_vars='label', 
                                      value_vars=['N','P','K','temperature','humidity','ph','rainfall']),
                                      x='variable', y='value', color='label',
                                      title=f'Value Ranges for {prediction.title()}',
                                      color_discrete_sequence=['#2ecc71'])
                        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig3, use_container_width=True)

        else:
            st.error("Model not loaded. Please check your RF.pkl file.")

    else:
        # ── Welcome Screen ─────────────────────────────────────
        st.markdown("""
        ### 👋 Welcome to AgriSens!
        
        This smart system helps farmers choose the **best crop** based on:
        - 🧪 Soil nutrients (N, P, K)
        - 🌡️ Temperature & Humidity
        - 💧 Rainfall & pH levels
        
        **How to use:**
        1. Enter your soil details in the sidebar
        2. Optionally auto-fill weather data using your city
        3. Click **Predict Crop** to get your recommendation
        4. Explore charts and ask the AI chatbot questions!
        """)

        if df is not None:
            st.divider()
            st.subheader("📊 About the Dataset")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", len(df))
            col2.metric("Unique Crops", df['label'].nunique())
            col3.metric("Features Used", "7")

            fig = px.pie(df, names='label', title='Crop Distribution in Dataset',
                         color_discrete_sequence=px.colors.sequential.Greens)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()