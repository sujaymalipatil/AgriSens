## ============================================================
## AgriSens — Enhanced Interactive Dashboard (Corrected)
## Install: pip install streamlit scikit-learn pandas plotly requests groq
## Optional: pip install python-dotenv   (only if you use a .env file)
## Run:     streamlit run agrisens_app.py
## ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, re, json, warnings, requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ── GROQ API KEY ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
# GROQ_API_KEY = "gsk_your_key_here"

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSens AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL STYLES ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Bricolage+Grotesque:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Bricolage Grotesque', sans-serif; }
.stApp { background: #0d1f0f; color: #f5f0e8; }
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(circle at 20% 20%, rgba(46,125,50,0.10) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(30,77,43,0.08) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
}

[data-testid="stSidebar"] {
    background: rgba(10,26,12,0.97) !important;
    border-right: 1px solid rgba(102,187,106,0.15) !important;
}
[data-testid="stSidebar"] * { color: #f5f0e8 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: rgba(102,187,106,0.3) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #66bb6a !important; }

.stButton > button {
    background: #2e7d32 !important;
    color: #f5f0e8 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 14px 28px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover { background: #388e3c !important; }

[data-testid="stMetric"] {
    background: rgba(102,187,106,0.06);
    border: 1px solid rgba(102,187,106,0.15);
    border-radius: 16px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    color: rgba(245,240,232,0.45) !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #f5f0e8 !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 28px !important;
}

hr { border-color: rgba(102,187,106,0.12) !important; }

.agri-card {
    background: rgba(245,240,232,0.04);
    border: 1px solid rgba(102,187,106,0.15);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #66bb6a;
    margin-bottom: 10px;
}
.card-body { font-size: 15px; line-height: 1.7; color: rgba(245,240,232,0.80); font-weight: 300; }

.crop-hero {
    font-family: 'DM Serif Display', serif;
    font-size: 72px;
    line-height: 1;
    color: #f5f0e8;
    letter-spacing: -1px;
}
.crop-hero span { color: #f5c842; font-style: italic; }

.conf-wrap { display:flex; align-items:center; gap:12px; margin-top:12px; }
.conf-track { flex:1; height:5px; background:rgba(245,240,232,0.1); border-radius:3px; overflow:hidden; }
.conf-fill  { height:100%; background:linear-gradient(90deg,#2e7d32,#f5c842); border-radius:3px; }
.conf-pct   { font-family:'DM Mono',monospace; font-size:13px; color:#f5c842; min-width:38px; }

.fert-row {
    display:flex; align-items:flex-start; gap:12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(245,240,232,0.06);
}
.fert-row:last-child { border-bottom: none; }
.fert-dot   { width:7px; height:7px; border-radius:50%; background:#66bb6a; margin-top:6px; flex-shrink:0; }
.fert-name  { font-size:14px; font-weight:500; color:#f5f0e8; }
.fert-detail{ font-size:12px; color:rgba(245,240,232,0.4); font-family:'DM Mono',monospace; margin-top:2px; }

.hero-eyebrow { font-family:'DM Mono',monospace; font-size:11px; letter-spacing:3px; text-transform:uppercase; color:#66bb6a; margin-bottom:10px; }
.hero-title   { font-family:'DM Serif Display',serif; font-size:clamp(36px,5vw,60px); line-height:1.05; color:#f5f0e8; margin-bottom:14px; }
.hero-title em{ color:#66bb6a; font-style:italic; }
.hero-sub     { font-size:15px; color:rgba(245,240,232,0.5); font-weight:300; line-height:1.65; max-width:500px; }

.badge-row  { display:flex; flex-wrap:wrap; gap:10px; margin-top:18px; }
.badge      { padding:5px 14px; border-radius:100px; font-family:'DM Mono',monospace; font-size:11px; }
.badge-green{ background:rgba(102,187,106,0.12); border:1px solid rgba(102,187,106,0.25); color:#a5d6a7; }
.badge-amber{ background:rgba(212,146,10,0.12);  border:1px solid rgba(212,146,10,0.28);  color:#f5c842; }

.section-title{ font-family:'DM Serif Display',serif; font-size:26px; color:#f5f0e8; margin-bottom:6px; }
.section-sub  { font-size:13px; color:rgba(245,240,232,0.4); font-family:'DM Mono',monospace; letter-spacing:1px; }

@keyframes pulse{ 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.7)} }
</style>
""", unsafe_allow_html=True)

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, '..', 'Datasets', 'Crop_recommendation.csv')
PKL_PATH = os.path.join(BASE_DIR, 'RF.pkl')

# ── CROP INFO DATABASE ────────────────────────────────────────
CROP_DB = {
    "rice":        {"desc":"Thrives in waterlogged warm conditions. Ideal for clay-heavy, slightly acidic soils.",            "advice":"Maintain 5–10 cm standing water. Apply urea in split doses.",                          "ferts":[{"n":"Urea",             "d":"50 kg N/ha at transplanting"},    {"n":"DAP",           "d":"25 kg P₂O₅/ha basal dose"}]},
    "maize":       {"desc":"Demands high nitrogen and moderate water. Best in deep, well-drained loam soils.",                "advice":"Apply zinc sulfate if pH > 7. Ridge planting aids drainage.",                         "ferts":[{"n":"Urea",             "d":"120 kg/ha split 3×"},             {"n":"MOP",           "d":"60 kg K₂O/ha basal"}]},
    "chickpea":    {"desc":"Nitrogen-fixing legume. Suits well-drained, neutral to slightly alkaline soils.",                "advice":"Inoculate seeds with Rhizobium. Avoid waterlogging at all stages.",                  "ferts":[{"n":"SSP",              "d":"40 kg P₂O₅/ha at sowing"},        {"n":"Gypsum",        "d":"250 kg/ha for sulfur-deficient soils"}]},
    "kidneybeans": {"desc":"Warm-season legume preferring loose, fertile soil with good drainage.",                          "advice":"Mulch to retain moisture. Excess nitrogen inhibits N-fixation.",                       "ferts":[{"n":"DAP",              "d":"30 kg/ha basal"},                  {"n":"Borax",         "d":"1 kg/ha for boron-deficient soils"}]},
    "pigeonpeas":  {"desc":"Drought-tolerant legume suited to shallow, infertile soils.",                                    "advice":"Intercrop with cereals. Deep taproot means minimal irrigation needed.",                "ferts":[{"n":"SSP",              "d":"50 kg/ha at sowing"},             {"n":"Rhizobium",     "d":"25 g/kg seed as inoculant"}]},
    "mothbeans":   {"desc":"Extreme drought tolerance; grows in sandy, arid soils with very low rainfall.",                  "advice":"No irrigation in arid zones. Harvest before monsoon ends.",                           "ferts":[{"n":"Urea",             "d":"20 kg N/ha starter dose only"}]},
    "mungbean":    {"desc":"Short-duration legume improving soil structure. Moderate water needs.",                          "advice":"60–70 day cycle. Excellent green manure option post-harvest.",                        "ferts":[{"n":"DAP",              "d":"25 kg/ha at planting"},           {"n":"MOP",           "d":"20 kg/ha"}]},
    "blackgram":   {"desc":"Grows in various soils; prefers well-drained fertile loam.",                                     "advice":"Avoid saline conditions. Grow post-kharif for double cropping.",                     "ferts":[{"n":"SSP",              "d":"40 kg/ha"},                        {"n":"Urea",          "d":"20 kg N/ha starter"}]},
    "lentil":      {"desc":"Cool-season legume needing moderate, well-distributed rainfall.",                                "advice":"Excellent for rotation post-cereal. Sensitive to frost.",                            "ferts":[{"n":"DAP",              "d":"40 kg/ha basal"},                  {"n":"Zinc Sulfate",  "d":"25 kg/ha if deficient"}]},
    "pomegranate": {"desc":"Drought and salinity tolerant fruit crop. Thrives in semi-arid climates.",                       "advice":"Avoid waterlogging. Prune for open-center canopy to boost yield.",                   "ferts":[{"n":"Vermicompost",     "d":"10 kg/plant annually"},           {"n":"NPK 10:10:10",  "d":"500 g/plant in two splits"}]},
    "banana":      {"desc":"High potassium demand; thrives in deep, rich, well-drained loam with high humidity.",            "advice":"Drip irrigation preferred. Remove lateral shoots to focus energy.",                   "ferts":[{"n":"Urea",             "d":"200 g/plant in 4 splits"},        {"n":"MOP",           "d":"300 g/plant — high K requirement"}]},
    "mango":       {"desc":"Tropical tree crop adaptable to deep, well-drained sandy loam.",                                 "advice":"Prune after harvest. Apply micronutrients yearly for sustained yield.",               "ferts":[{"n":"NPK 12:32:16",     "d":"500 g/tree at flowering"},        {"n":"Urea",          "d":"1 kg/tree post-harvest"}]},
    "grapes":      {"desc":"Prefers dry climate, deep soil, and excellent drainage. pH 5.5–7.0.",                            "advice":"Train on trellis. Potassium critical for sugar accumulation.",                        "ferts":[{"n":"Potassium Nitrate","d":"20 g/vine fortnightly"},          {"n":"Calcium Nitrate","d":"15 g/vine bi-weekly"}]},
    "watermelon":  {"desc":"Warm-season crop needing sandy loam, good drainage, and ample sunshine.",                        "advice":"Plant on raised beds. Regular irrigation critical during fruit set.",                  "ferts":[{"n":"DAP",              "d":"40 kg/ha basal"},                  {"n":"MOP",           "d":"60 kg/ha in 2 splits"}]},
    "muskmelon":   {"desc":"Thrives in warm, arid zones with well-drained sandy soil. Frost-sensitive.",                     "advice":"Avoid overhead irrigation — promotes fungal disease.",                                "ferts":[{"n":"Urea",             "d":"50 kg N/ha in splits"},           {"n":"SSP",           "d":"40 kg P/ha basal"}]},
    "apple":       {"desc":"Requires cool winters for dormancy break. Prefers deep fertile well-drained loam.",              "advice":"Thin fruit to one per cluster. Apply lime if pH < 6.",                               "ferts":[{"n":"Urea",             "d":"500 g/tree in spring"},           {"n":"Borax",         "d":"0.3% foliar spray at bloom"}]},
    "orange":      {"desc":"Subtropical citrus needing mild winters and consistent moisture.",                               "advice":"Mulch tree basins. Monitor closely for citrus greening disease.",                    "ferts":[{"n":"NPK 15:15:15",     "d":"500 g/tree 4× per year"},         {"n":"Zinc Sulfate",  "d":"0.5% foliar 2× per year"}]},
    "papaya":      {"desc":"Fast-growing tropical fruit with very high nutrient demand. Dislikes cold.",                     "advice":"High N in early stages; shift to K at fruiting stage.",                              "ferts":[{"n":"Urea",             "d":"200 g/plant monthly"},            {"n":"MOP",           "d":"150 g/plant at fruiting"}]},
    "coconut":     {"desc":"Coastal crop adapted to high humidity, sandy loam, and saline tolerance.",                      "advice":"Apply green manure. Potassium lifts yield significantly.",                            "ferts":[{"n":"NPK 12:5:21",      "d":"1 kg/palm biannually"},           {"n":"Common Salt",   "d":"2 kg/palm as saline buffer"}]},
    "cotton":      {"desc":"High water and potassium consumer. Best in deep black cotton (Vertisol) soils.",                 "advice":"Bollworm IPM essential. Avoid excess N — promotes vegetative growth.",               "ferts":[{"n":"Urea",             "d":"80 kg N/ha split 3×"},            {"n":"MOP",           "d":"60 kg K₂O/ha basal"}]},
    "jute":        {"desc":"Thrives in humid tropical climate with loamy alluvial soil and moderate flooding.",              "advice":"Retting in slow-moving water for 2–3 weeks post-harvest.",                          "ferts":[{"n":"Urea",             "d":"60 kg N/ha"},                      {"n":"SSP",           "d":"40 kg P/ha basal"}]},
    "coffee":      {"desc":"Shade-loving; thrives in volcanic, well-drained, slightly acidic soil.",                        "advice":"Maintain 50% shade cover. Mulch heavily. pH 5.5–6.5 is critical.",                  "ferts":[{"n":"NPK 17:17:17",     "d":"250 g/plant 2× yearly"},          {"n":"Borax",         "d":"0.2% foliar for berry fill"}]},
}

# ── DATA ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    np.random.seed(42)
    rows = []
    for crop in CROP_DB.keys():
        for _ in range(100):
            rows.append({
                "N":           np.random.randint(0, 140),
                "P":           np.random.randint(0, 145),
                "K":           np.random.randint(0, 205),
                "temperature": round(np.random.uniform(8,  44),  1),
                "humidity":    round(np.random.uniform(14, 100), 1),
                "ph":          round(np.random.uniform(3.5, 9.5),1),
                "rainfall":    round(np.random.uniform(20, 300), 1),
                "label":       crop
            })
    return pd.DataFrame(rows)


# ── MODEL ────────────────────────────────────────────────────
@st.cache_resource
def load_model(_df):
    if os.path.exists(PKL_PATH):
        try:
            with open(PKL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ RF.pkl is incompatible with your scikit-learn version. Retraining now… ({e})")
            os.remove(PKL_PATH)

    X = _df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = _df['label']
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X, y)
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(m, f)
    st.success("✅ Model retrained and saved successfully.")
    return m


df    = load_data()
model = load_model(df)

# ── WEATHER ───────────────────────────────────────────────────
@st.cache_data(ttl=900)
def get_weather(city):
    try:
        wx  = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5).json()
        cur = wx['current_condition'][0]
        return {
            "city":     city,
            "temp":     float(cur['temp_C']),
            "humidity": float(cur['humidity']),
            "rainfall": float(cur['precipMM']) * 30,
            "desc":     cur['weatherDesc'][0]['value']
        }
    except Exception as e:
        st.sidebar.warning(f"⚠️ Weather fetch failed: {e}")
        return {"city": city, "temp": 28.0, "humidity": 65.0,
                "rainfall": 100.0, "desc": "Unavailable"}


# ── GROQ AI ───────────────────────────────────────────────────
def get_ai_insights(crop, inputs):
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""Return ONLY valid JSON — no extra text, no markdown fences.

Crop: {crop}
N={inputs['N']}, P={inputs['P']}, K={inputs['K']}
Temp={inputs['temperature']}, Humidity={inputs['humidity']}
pH={inputs['ph']}, Rainfall={inputs['rainfall']}

{{
  "description": "2-sentence agronomic description tailored to these exact soil values",
  "fertilizers": [{{"name":"","npk":"","dosage":"","time":""}}],
  "advice": "3-4 specific actionable tips for this exact soil profile"
}}"""
        res   = client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model="llama-3.1-8b-instant"
        )
        raw   = re.sub(r"```json|```", "", res.choices[0].message.content)
        match = re.search(r"\{.*\}", raw, re.S)
        return json.loads(match.group(0)) if match else None
    except Exception:
        return None


# ── PLOTLY SHARED THEME ───────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Bricolage Grotesque", color="rgba(245,240,232,0.7)", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#66bb6a","#f5c842","#4db6ac","#ef5350","#ce93d8","#80cbc4","#ffcc02","#a5d6a7"],
    transition=dict(duration=800, easing="cubic-in-out"),
)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 20px;">
          <div style="font-family:'DM Serif Display',serif;font-size:22px;color:#f5f0e8;
                      display:flex;align-items:center;gap:8px;">
            <span style="width:10px;height:10px;border-radius:50%;background:#66bb6a;
                         display:inline-block;animation:pulse 2.5s ease-in-out infinite;"></span>
            AgriSens
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
                      color:rgba(245,240,232,0.3);margin-top:3px;">AI CROP INTELLIGENCE</div>
        </div>
        """, unsafe_allow_html=True)

        city_input = st.text_input("📍 Your City", value="Bangalore")
        weather    = get_weather(city_input)

        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:10px;'
                    'letter-spacing:2px;color:#66bb6a;margin-bottom:12px;">MACRONUTRIENTS</div>',
                    unsafe_allow_html=True)
        n  = st.slider("Nitrogen (N)   · kg/ha", 0,   140, 50)
        p  = st.slider("Phosphorus (P) · kg/ha", 0,   145, 40)
        k  = st.slider("Potassium (K)  · kg/ha", 0,   205, 40)

        st.markdown('<div style="margin:16px 0 12px;height:1px;background:rgba(102,187,106,0.12);"></div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:10px;'
                    'letter-spacing:2px;color:#66bb6a;margin-bottom:12px;">SOIL CHEMISTRY</div>',
                    unsafe_allow_html=True)
        ph = st.slider("pH Level", 0.0, 14.0, 6.5, step=0.1)

        health  = min(100, int((n/140)*25 + (p/145)*25 + (k/205)*25 + (1 - abs(ph-6.5)/7.5)*25))
        h_color = "#66bb6a" if health >= 65 else ("#ffb74d" if health >= 40 else "#ef5350")
        h_label = "Excellent" if health >= 75 else ("Good" if health >= 55 else ("Fair" if health >= 35 else "Poor"))

        st.markdown(f"""
        <div style="background:rgba(245,240,232,0.03);border:1px solid rgba(102,187,106,0.1);
                    border-radius:12px;padding:14px 16px;margin:16px 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
                        color:rgba(245,240,232,0.35);">SOIL HEALTH</div>
            <div style="font-family:'DM Mono',monospace;font-size:13px;color:{h_color};">
              {health}% · {h_label}
            </div>
          </div>
          <div style="height:5px;background:rgba(245,240,232,0.1);border-radius:3px;overflow:hidden;">
            <div style="width:{health}%;height:100%;background:{h_color};border-radius:3px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin:4px 0 12px;height:1px;background:rgba(102,187,106,0.12);"></div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(245,240,232,0.03);border:1px solid rgba(102,187,106,0.1);
                    border-radius:12px;padding:14px 16px;margin-bottom:20px;">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:12px;">
            <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
                        color:rgba(245,240,232,0.35);">AUTO-DETECTED</div>
            <div style="display:inline-flex;align-items:center;gap:4px;font-family:'DM Mono',monospace;
                        font-size:9px;color:rgba(245,240,232,0.3);padding:2px 7px;border-radius:100px;
                        border:1px solid rgba(245,240,232,0.1);">
              <div style="width:5px;height:5px;border-radius:50%;background:#d4920a;"></div>live
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center;">
            <div>
              <div style="font-family:'DM Mono',monospace;font-size:15px;color:#f5f0e8;">{weather['temp']:.0f}°C</div>
              <div style="font-size:10px;color:rgba(245,240,232,0.35);margin-top:2px;">Temp</div>
            </div>
            <div>
              <div style="font-family:'DM Mono',monospace;font-size:15px;color:#f5f0e8;">{weather['humidity']:.0f}%</div>
              <div style="font-size:10px;color:rgba(245,240,232,0.35);margin-top:2px;">Humidity</div>
            </div>
            <div>
              <div style="font-family:'DM Mono',monospace;font-size:15px;color:#f5f0e8;">{weather['rainfall']:.0f}mm</div>
              <div style="font-size:10px;color:rgba(245,240,232,0.35);margin-top:2px;">Rainfall</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        predict_clicked = st.button("🌱 Analyze & Predict Crop")

    # ── HERO ─────────────────────────────────────────────────
    col_hero, col_wx = st.columns([3, 1])

    with col_hero:
        st.markdown(f"""
        <div style="padding:20px 0 30px;">
          <div class="hero-eyebrow">AI-powered crop intelligence</div>
          <div class="hero-title">Know what your<br><em>soil</em> wants to grow</div>
          <div class="hero-sub">Input soil nutrient levels and let our ML model recommend
            the ideal crop — trained on 2,200+ agronomic data points across 22 crop types.</div>
          <div class="badge-row">
            <div class="badge badge-green">22 Crop Types</div>
            <div class="badge badge-green">Random Forest Model</div>
            <div class="badge badge-amber">📍 {weather['city']}</div>
            <div class="badge badge-amber">{weather['desc']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_wx:
        st.markdown(f"""
        <div style="background:rgba(245,240,232,0.04);border:1px solid rgba(102,187,106,0.18);
                    border-radius:18px;padding:22px 24px;margin-top:20px;text-align:right;">
          <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
                      text-transform:uppercase;color:rgba(245,240,232,0.35);margin-bottom:6px;">
            {weather['city'].upper()}
          </div>
          <div style="font-family:'DM Serif Display',serif;font-size:52px;color:#f5f0e8;line-height:1;">
            {weather['temp']:.0f}°
          </div>
          <div style="font-size:12px;color:#a5d6a7;margin-top:6px;">
            {weather['desc']} · {weather['humidity']:.0f}% humidity
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    # ── STATS STRIP ──────────────────────────────────────────
    cols   = st.columns(7)
    params = [
        ("Nitrogen",    n,                   "kg/ha"),
        ("Phosphorus",  p,                   "kg/ha"),
        ("Potassium",   k,                   "kg/ha"),
        ("pH",          ph,                  ""),
        ("Temperature", weather['temp'],     "°C"),
        ("Humidity",    weather['humidity'], "%"),
        ("Rainfall",    weather['rainfall'], "mm"),
    ]
    for col, (lbl, val, unit) in zip(cols, params):
        display = f"{val:.1f}{unit}" if isinstance(val, float) else f"{val}{unit}"
        col.metric(lbl, display)

    st.markdown("<div style='margin-bottom:32px'></div>", unsafe_allow_html=True)

    # ── PREDICTION RESULT ─────────────────────────────────────
    if predict_clicked:
        features   = [[n, p, k, weather['temp'], weather['humidity'], ph, weather['rainfall']]]
        crop       = model.predict(features)[0]
        probs      = model.predict_proba(features)[0]
        confidence = int(max(probs) * 100)
        info       = CROP_DB.get(crop.lower(), CROP_DB["rice"])
        name       = crop.capitalize()

        ai = None
        if GROQ_API_KEY:
            with st.spinner("Getting AI insights…"):
                ai = get_ai_insights(crop, {
                    "N": n, "P": p, "K": k,
                    "temperature": weather['temp'],
                    "humidity":    weather['humidity'],
                    "ph":          ph,
                    "rainfall":    weather['rainfall']
                })

        desc   = ai["description"] if ai else info["desc"]
        advice = ai["advice"]      if ai else info["advice"]
        ferts  = ai["fertilizers"] if ai else [
            {"name": f["n"], "npk": "", "dosage": f["d"], "time": ""} for f in info["ferts"]
        ]

        # Result card
        st.markdown(f"""
        <div class="agri-card" style="border-color:rgba(102,187,106,0.3);">
          <div class="card-label">Recommended Crop</div>
          <div class="crop-hero"><span>{name}</span></div>
          <div class="conf-wrap">
            <div style="font-family:'DM Mono',monospace;font-size:11px;
                        color:rgba(245,240,232,0.35);letter-spacing:1px;">CONFIDENCE</div>
            <div class="conf-track">
              <div class="conf-fill" style="width:{confidence}%;"></div>
            </div>
            <div class="conf-pct">{confidence}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Description + Advice
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"""
            <div class="agri-card">
              <div class="card-label">About This Crop</div>
              <div class="card-body">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_r:
            st.markdown(f"""
            <div class="agri-card">
              <div class="card-label">Soil Advice</div>
              <div class="card-body">{advice}</div>
            </div>
            """, unsafe_allow_html=True)

        # Fertilizer table
        fert_rows = "".join([f"""
        <div class="fert-row">
          <div class="fert-dot"></div>
          <div>
            <div class="fert-name">{f.get('name','—')}{"  ·  NPK: " + f['npk'] if f.get('npk') else ""}</div>
            <div class="fert-detail">{f.get('dosage','—')}{"  ·  " + f['time'] if f.get('time') else ""}</div>
          </div>
        </div>""" for f in ferts])

        st.markdown(f"""
        <div class="agri-card">
          <div class="card-label">Fertilizer Recommendations</div>
          {fert_rows}
        </div>
        """, unsafe_allow_html=True)

        # Top-5 alternatives bar chart
        top5_idx   = np.argsort(probs)[::-1][:5]
        top5_crops = [model.classes_[i].capitalize() for i in top5_idx]
        top5_probs = [round(float(probs[i]) * 100, 1) for i in top5_idx]

        fig_alt = go.Figure(go.Bar(
            x=top5_probs,
            y=top5_crops,
            orientation='h',
            marker=dict(
                color=top5_probs,
                colorscale=[[0, "rgba(102,187,106,0.25)"], [1, "#66bb6a"]],
            ),
            text=[f"{v}%" for v in top5_probs],
            textfont=dict(family="DM Mono", color="rgba(245,240,232,0.8)"),
            textposition='outside'
        ))

        layout = PLOTLY_LAYOUT.copy()
        layout["title"] = dict(
            text="Top 5 Crop Alternatives",
            font=dict(family="DM Serif Display", size=18, color="#f5f0e8")
        )
        layout["height"] = 260
        layout["xaxis"] = dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.08)",
            range=[0, 115],
            title=None
        )
        layout["yaxis"] = dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.08)",
            autorange="reversed",
            title=None
        )
        fig_alt.update_layout(**layout)
        st.plotly_chart(fig_alt, use_container_width=True)

    # ── TABS ─────────────────────────────────────────────────
    st.markdown('<hr>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊  Distribution", "🌡️  Conditions", "⚗️  Nutrients"])

    with tab1:
        c1, c2 = st.columns(2)

        counts = df['label'].value_counts().reset_index()
        counts.columns = ['Crop', 'Count']
        fig1 = px.bar(counts, x='Crop', y='Count', color='Count',
                      color_continuous_scale=["rgba(102,187,106,0.3)", "#66bb6a"])
        fig1.update_traces(marker_line_width=0,
                           selector=dict(type='bar'))
        fig1.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, height=320,
                           title=dict(text="Records per Crop",
                                      font=dict(family="DM Serif Display", size=17, color="#f5f0e8")))
        fig1.update_xaxes(tickangle=45)
        c1.plotly_chart(fig1, use_container_width=True)

        avg_ph = df.groupby('label')['ph'].mean().reset_index().rename(columns={'ph': 'Avg pH'})
        fig2   = px.bar(avg_ph, x='label', y='Avg pH', color='Avg pH',
                        color_continuous_scale=["#ef5350", "#66bb6a", "#f5c842"])
        fig2.update_traces(marker_line_width=0,
                           selector=dict(type='bar'))
        fig2.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, height=320,
                           title=dict(text="Average pH by Crop",
                                      font=dict(family="DM Serif Display", size=17, color="#f5f0e8")))
        fig2.update_xaxes(tickangle=45)
        c2.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)

        fig3 = px.box(df, x='label', y='temperature', color='label',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_traces(marker=dict(size=3, opacity=0.5))
        fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=340,
                           title=dict(text="Temperature Range by Crop",
                                      font=dict(family="DM Serif Display", size=17, color="#f5f0e8")))
        fig3.update_xaxes(tickangle=45)
        c1.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter(df, x='rainfall', y='humidity', color='label', opacity=0.6,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig4.update_traces(marker=dict(size=6, line=dict(width=0.5, color='rgba(255,255,255,0.2)')))
        fig4.update_layout(**PLOTLY_LAYOUT, height=340,
                           title=dict(text="Rainfall vs Humidity",
                                      font=dict(family="DM Serif Display", size=17, color="#f5f0e8")),
                           legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
        c2.plotly_chart(fig4, use_container_width=True)

    with tab3:
        avg_npk = df.groupby('label')[['N','P','K']].mean().reset_index()
        melted  = avg_npk.melt(id_vars='label', var_name='Nutrient', value_name='kg/ha')
        fig5 = px.bar(melted, x='label', y='kg/ha', color='Nutrient', barmode='group',
                      color_discrete_map={"N":"#66bb6a","P":"#f5c842","K":"#4db6ac"})
        fig5.update_traces(marker_line_width=0,
                           selector=dict(type='bar'))
        fig5.update_layout(**PLOTLY_LAYOUT, height=360,
                           title=dict(text="Avg NPK Profile per Crop",
                                      font=dict(family="DM Serif Display", size=17, color="#f5f0e8")),
                           legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"))
        fig5.update_xaxes(tickangle=45)
        st.plotly_chart(fig5, use_container_width=True)

        c1, c2 = st.columns(2)

        fig6 = px.violin(df, x='label', y='ph', color='label', box=True,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig6.update_traces(meanline_visible=True,
                           points='outliers',
                           jitter=0.05)
        fig6.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=320,
                           title=dict(text="pH Distribution by Crop",
                                      font=dict(family="DM Serif Display", size=17, color="#f5f0e8")))
        fig6.update_xaxes(tickangle=45)
        c1.plotly_chart(fig6, use_container_width=True)

        corr = df[['N','P','K','temperature','humidity','ph','rainfall']].corr(numeric_only=True)
        fig7 = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[[0,"#0d1f0f"],[0.5,"#2e7d32"],[1,"#f5c842"]],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont=dict(size=11),
            hoverongaps=False,
        ))
        fig7.update_layout(**PLOTLY_LAYOUT, height=320,
                           title=dict(text="Feature Correlation Heatmap",
                                      font=dict(family="DM Serif Display", size=17, color="#f5f0e8")))
        c2.plotly_chart(fig7, use_container_width=True)

    # ── FOOTER ───────────────────────────────────────────────
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:20px 0;font-family:'DM Mono',monospace;
                font-size:11px;color:rgba(245,240,232,0.2);letter-spacing:1px;">
      AGRISENS · ML CROP INTELLIGENCE · STREAMLIT + SKLEARN + GROQ
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()