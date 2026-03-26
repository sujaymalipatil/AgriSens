## ============================================================
## AgriSens - PREMIUM DASHBOARD VERSION 🚀
## Features:
## ✅ ML Crop Prediction
## ✅ Groq AI Insights
## ✅ Auto Weather Detection (No API)
## ✅ Analytics Dashboard (Charts)
## ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import warnings
import re
import json
import requests
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# ── ENV ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── PAGE CONFIG ─────────────────────────────────────
st.set_page_config(page_title="AgriSens", layout="wide")

# ── UI STYLE ────────────────────────────────────────
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:white;}
.hero {padding:30px;text-align:center;background:linear-gradient(45deg,#2ecc71,#27ae60);border-radius:15px;}
.card {background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;margin-bottom:20px;}
.section {font-size:22px;color:#2ecc71;}
</style>
""", unsafe_allow_html=True)

# ── PATHS ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, '..', 'Datasets', 'Crop_recommendation.csv')
PKL_PATH = os.path.join(BASE_DIR, 'RF.pkl')

# ── DATA ────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

df = load_data()

# ── MODEL ───────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(PKL_PATH):
        return pickle.load(open(PKL_PATH, 'rb'))

    X = df[['N','P','K','temperature','humidity','ph','rainfall']]
    y = df['label']
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    pickle.dump(model, open(PKL_PATH, 'wb'))
    return model

model = load_model()

# ── AUTO WEATHER (NO API) ───────────────────────────
def get_weather_auto():
    try:
        res = requests.get("https://ipapi.co/json/").json()
        city = res.get("city", "Unknown")

        weather = requests.get(f"https://wttr.in/{city}?format=j1").json()

        return {
            "city": city,
            "temp": float(weather['current_condition'][0]['temp_C']),
            "humidity": float(weather['current_condition'][0]['humidity']),
            "rainfall": float(weather['current_condition'][0]['precipMM']) * 30
        }
    except:
        return {"city": "Unknown", "temp": 25, "humidity": 70, "rainfall": 100}

# ── PREDICT ─────────────────────────────────────────
def predict_crop(n,p,k,t,h,ph,r):
    return model.predict([[n,p,k,t,h,ph,r]])[0]

# ── GROQ AI ─────────────────────────────────────────
def get_crop_info(crop, inputs):
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""
Return JSON only.

Crop: {crop}

Soil:
N={inputs['N']}, P={inputs['P']}, K={inputs['K']}
Temp={inputs['temperature']}, Humidity={inputs['humidity']}
pH={inputs['ph']}, Rainfall={inputs['rainfall']}

{{
"description":"",
"fertilizers":[{{"name":"","npk":"","dosage":"","time":""}}],
"advice":""
}}
"""

        res = client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model="llama-3.1-8b-instant"
        )

        raw = re.sub(r"```json|```","",res.choices[0].message.content)
        match = re.search(r"\{.*\}", raw, re.S)

        return json.loads(match.group(0)) if match else {"error":"AI error"}

    except Exception as e:
        return {"error":str(e)}

# ── MAIN UI ─────────────────────────────────────────
def main():

    # HERO
    st.markdown("""
    <div class="hero">
        <h1>🌾 AgriSens AI Dashboard</h1>
        <p>Smart Farming with AI + Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # WEATHER AUTO
    weather = get_weather_auto()

    col1, col2, col3 = st.columns(3)
    col1.metric("📍 Location", weather["city"])
    col2.metric("🌡️ Temp", f"{weather['temp']}°C")
    col3.metric("💧 Humidity", f"{weather['humidity']}%")

    st.divider()

    # SIDEBAR INPUT
    st.sidebar.header("🌿 Soil Inputs")

    n = st.sidebar.slider("Nitrogen", 0, 140, 50)
    p = st.sidebar.slider("Phosphorus", 0, 145, 40)
    k = st.sidebar.slider("Potassium", 0, 205, 40)
    ph = st.sidebar.slider("pH", 0.0, 14.0, 6.5)

    # Auto-filled
    t = weather["temp"]
    h = weather["humidity"]
    r = weather["rainfall"]

    if st.sidebar.button("🚀 Predict Crop"):

        crop = predict_crop(n,p,k,t,h,ph,r)

        # RESULT
        st.markdown(f"""
        <div class="card">
            <h2>🌱 Recommended Crop</h2>
            <h1 style="text-align:center;">{crop.upper()}</h1>
        </div>
        """, unsafe_allow_html=True)

        inputs = {"N":n,"P":p,"K":k,"temperature":t,"humidity":h,"ph":ph,"rainfall":r}

        # ── AI ───────────────────────────────────────
        if GROQ_API_KEY:
            data = get_crop_info(crop, inputs)

            if "error" not in data:

                st.markdown('<div class="card"><div class="section">🌿 Description</div>', unsafe_allow_html=True)
                st.write(data["description"])
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="card"><div class="section">🧪 Fertilizers</div>', unsafe_allow_html=True)
                for f in data["fertilizers"]:
                    st.write(f"**{f['name']}** | NPK: {f['npk']} | {f['dosage']}")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="card"><div class="section">💡 Advice</div>', unsafe_allow_html=True)
                st.write(data["advice"])
                st.markdown("</div>", unsafe_allow_html=True)

        # ── ANALYTICS DASHBOARD ───────────────────────
        st.subheader("📊 Crop Analytics Dashboard")

        colA, colB = st.columns(2)

        with colA:
            fig1 = px.histogram(df, x="label", title="Crop Distribution")
            st.plotly_chart(fig1, use_container_width=True)

        with colB:
            fig2 = px.box(df, x="label", y="temperature", title="Temperature vs Crop")
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        fig3 = px.scatter(
            df, x="rainfall", y="humidity",
            color="label",
            title="Rainfall vs Humidity"
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── RUN ────────────────────────────────────────────
if __name__ == "__main__":
    main()