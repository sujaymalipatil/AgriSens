## ============================================================
## AgriSens — Biopunk Editorial Dashboard
## Install: pip install streamlit scikit-learn pandas plotly requests streamlit-echarts
## Optional: pip install python-dotenv groq
## Run:      streamlit run webapp.py
## ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, re, json, warnings, requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_echarts import st_echarts, JsCode

warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq as _Groq
        _groq_client = _Groq(api_key=GROQ_API_KEY)
    except ImportError:
        st.warning("⚠️ `groq` package not installed — AI insights disabled. Run: pip install groq")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSens AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for Radar chart and animation delay
if 'predicted_crop' not in st.session_state:
    st.session_state.predicted_crop = 'rice'
if 'anim_delay' not in st.session_state:
    st.session_state.anim_delay = 100

# ── GLOBAL STYLES ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Bricolage+Grotesque:opsz,wght@12..96,300;12..96,400;12..96,500;12..96,600&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Bricolage Grotesque', sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: #080f09;
    color: #f0ebe1;
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    background-size: 200px 200px;
    pointer-events: none;
    opacity: 0.6;
}

.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 70% 50% at 10% 15%, rgba(34,97,38,0.13) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 90% 85%, rgba(16,58,24,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 30% 30% at 50% 50%, rgba(245,200,66,0.03) 0%, transparent 70%);
    pointer-events: none;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d07 0%, #0a1a0c 100%) !important;
    border-right: 1px solid rgba(74,160,79,0.12) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] * { color: #e8e0d0 !important; }
[data-testid="stSidebar"] .stTextInput input, [data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(74,160,79,0.2) !important;
    border-radius: 10px !important;
    color: #e8e0d0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] .stTextInput input:focus, [data-testid="stSidebar"] .stNumberInput input:focus {
    border-color: rgba(74,160,79,0.5) !important;
    box-shadow: 0 0 0 3px rgba(74,160,79,0.08) !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: rgba(74,160,79,0.25) !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(135deg, #4aa04f, #8bc34a) !important;
    box-shadow: 0 0 12px rgba(74,160,79,0.5) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1e5c22 0%, #2e7d32 50%, #388e3c 100%) !important;
    color: #f0ebe1 !important;
    border: 1px solid rgba(74,160,79,0.3) !important;
    border-radius: 14px !important;
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 16px 32px !important;
    width: 100% !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 4px 24px rgba(46,125,50,0.25), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}
.stButton > button::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, transparent 100%);
    pointer-events: none;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 32px rgba(46,125,50,0.40), 0 0 0 1px rgba(74,160,79,0.4), inset 0 1px 0 rgba(255,255,255,0.1) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.99) !important; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(74,160,79,0.1);
    border-radius: 14px;
    padding: 14px 18px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(74,160,79,0.3), transparent);
}
[data-testid="stMetric"]:hover {
    border-color: rgba(74,160,79,0.22);
    background: rgba(255,255,255,0.04);
}
[data-testid="stMetricLabel"] {
    color: rgba(240,235,225,0.38) !important;
    font-size: 10px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #f0ebe1 !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 24px !important;
    line-height: 1.2 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(74,160,79,0.1) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: rgba(240,235,225,0.45) !important;
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(74,160,79,0.15) !important;
    color: #a5d6a7 !important;
    border: 1px solid rgba(74,160,79,0.25) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

hr { border: none !important; border-top: 1px solid rgba(74,160,79,0.08) !important; margin: 28px 0 !important; }

/* ── KEYFRAMES ── */
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 rgba(74,160,79,0.4); }
    50%       { opacity: 0.7; transform: scale(0.85); box-shadow: 0 0 0 6px rgba(74,160,79,0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes float-up {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 30px rgba(74,160,79,0.2), 0 0 60px rgba(74,160,79,0.08); }
    50%       { box-shadow: 0 0 50px rgba(74,160,79,0.35), 0 0 100px rgba(74,160,79,0.15); }
}
@keyframes sweep-in {
    from { width: 0%; }
    to   { width: var(--conf-w); }
}
@keyframes badge-pop {
    0%   { transform: scale(0.7) translateY(4px); opacity: 0; }
    70%  { transform: scale(1.05); }
    100% { transform: scale(1) translateY(0); opacity: 1; }
}

/* ── CHART ANIMATIONS ── */
@keyframes chart-reveal {
    0%   { opacity: 0; transform: translateY(30px) scale(0.95); filter: blur(8px); }
    100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0px); }
}

[data-testid="stPlotlyChart"] {
    animation: chart-reveal 0.9s cubic-bezier(0.2, 0.8, 0.2, 1) backwards;
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), filter 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    will-change: transform, filter, opacity;
}

/* Stagger delay for side-by-side charts */
[data-testid="column"]:nth-child(1) [data-testid="stPlotlyChart"] { animation-delay: 0.1s; }
[data-testid="column"]:nth-child(2) [data-testid="stPlotlyChart"] { animation-delay: 0.35s; }

/* Alpha-aware hover glow. This shadows the data INSIDE the transparent iframe! */
[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-6px) scale(1.015);
    filter: drop-shadow(0px 12px 24px rgba(74, 160, 79, 0.3)) brightness(1.15) !important;
    z-index: 10;
}

/* ── CUSTOM COMPONENTS ── */
.brand-wrap {
    padding: 4px 0 24px;
    border-bottom: 1px solid rgba(74,160,79,0.1);
    margin-bottom: 22px;
}
.brand-name {
    font-family: 'DM Serif Display', serif;
    font-size: 20px;
    color: #f0ebe1;
    display: flex;
    align-items: center;
    gap: 10px;
}
.brand-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #4aa04f;
    animation: pulse-dot 2.5s ease-in-out infinite;
    flex-shrink: 0;
}
.brand-sub {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    color: rgba(240,235,225,0.25);
    margin-top: 4px;
    text-transform: uppercase;
}

.sb-section {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(74,160,79,0.7);
    margin: 20px 0 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sb-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(74,160,79,0.15);
}

.gauge-wrap {
    display: flex;
    align-items: center;
    gap: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(74,160,79,0.12);
    border-radius: 14px;
    padding: 14px 18px;
    margin: 16px 0;
}
.gauge-label-block { flex: 1; }
.gauge-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 2.5px;
    color: rgba(240,235,225,0.3);
    text-transform: uppercase;
    margin-bottom: 4px;
}
.gauge-value {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #f0ebe1;
}

.wx-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(74,160,79,0.12);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 22px;
}
.wx-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}
.wx-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 2.5px;
    color: rgba(240,235,225,0.3);
    text-transform: uppercase;
    flex: 1;
}
.live-pip {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'DM Mono', monospace;
    font-size: 9px; color: rgba(240,235,225,0.3);
    padding: 3px 9px;
    border-radius: 100px;
    border: 1px solid rgba(240,235,225,0.1);
}
.live-pip-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #e8a020;
    animation: pulse-dot 2s ease-in-out infinite;
}
.wx-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    text-align: center;
}
.wx-cell-val {
    font-family: 'DM Mono', monospace;
    font-size: 16px;
    color: #f0ebe1;
}
.wx-cell-lbl {
    font-size: 10px;
    color: rgba(240,235,225,0.3);
    margin-top: 2px;
}

.hero-wrap { padding: 24px 0 36px; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #4aa04f;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 28px; height: 1px;
    background: #4aa04f;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(38px, 5vw, 68px);
    line-height: 1.03;
    color: #f0ebe1;
    margin-bottom: 18px;
    letter-spacing: -0.5px;
}
.hero-title em {
    font-style: italic;
    background: linear-gradient(135deg, #4aa04f 0%, #8bc34a 50%, #cddc39 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 15px;
    color: rgba(240,235,225,0.45);
    font-weight: 300;
    line-height: 1.7;
    max-width: 520px;
}
.badge-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
.badge {
    padding: 6px 15px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.5px;
    animation: badge-pop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) backwards;
}
.badge:nth-child(1) { animation-delay: 0.05s; }
.badge:nth-child(2) { animation-delay: 0.1s;  }
.badge:nth-child(3) { animation-delay: 0.15s; }
.badge:nth-child(4) { animation-delay: 0.2s;  }
.badge-green {
    background: rgba(74,160,79,0.1);
    border: 1px solid rgba(74,160,79,0.22);
    color: #a5d6a7;
}
.badge-amber {
    background: rgba(212,155,20,0.1);
    border: 1px solid rgba(212,155,20,0.25);
    color: #f5c842;
}

.wx-hero {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(74,160,79,0.15);
    border-radius: 20px;
    padding: 28px 30px;
    margin-top: 24px;
    text-align: right;
    position: relative;
    overflow: hidden;
}
.wx-hero::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(74,160,79,0.12) 0%, transparent 70%);
}
.wx-hero-city {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: rgba(240,235,225,0.3);
    margin-bottom: 8px;
    text-transform: uppercase;
}
.wx-hero-temp {
    font-family: 'DM Serif Display', serif;
    font-size: 64px;
    color: #f0ebe1;
    line-height: 1;
}
.wx-hero-temp sup {
    font-size: 24px;
    vertical-align: top;
    margin-top: 10px;
    display: inline-block;
    color: rgba(240,235,225,0.6);
}
.wx-hero-desc {
    font-size: 12px;
    color: #a5d6a7;
    margin-top: 8px;
}

.result-card {
    background: linear-gradient(135deg, rgba(14,32,16,0.9) 0%, rgba(8,20,10,0.95) 100%);
    border: 1px solid rgba(74,160,79,0.35);
    border-radius: 24px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    animation: float-up 0.5s cubic-bezier(0.34, 1.2, 0.64, 1) backwards, glow-pulse 4s ease-in-out infinite;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(74,160,79,0.6), transparent);
}
.result-card::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139,195,74,0.08) 0%, transparent 70%);
}
.result-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #4aa04f;
    margin-bottom: 12px;
}
.result-crop-name {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(56px, 8vw, 96px);
    line-height: 0.95;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #f0ebe1 30%, rgba(240,235,225,0.6) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-crop-name span {
    background: linear-gradient(135deg, #f5c842 0%, #ff9800 50%, #f5c842 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
}

.conf-row {
    display: flex; align-items: center; gap: 16px;
    margin-top: 20px;
}
.conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: rgba(240,235,225,0.3);
    text-transform: uppercase;
    white-space: nowrap;
}
.conf-track {
    flex: 1; height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #2e7d32, #66bb6a, #f5c842);
    border-radius: 4px;
    animation: sweep-in 1s cubic-bezier(0.34, 1.1, 0.64, 1) forwards 0.3s;
    width: 0%;   /* start at zero; animation drives to var(--conf-w) */
}
.conf-pct {
    font-family: 'DM Mono', monospace;
    font-size: 18px;
    font-weight: 500;
    color: #f5c842;
    min-width: 50px;
    text-align: right;
}

.info-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(74,160,79,0.12);
    border-radius: 20px;
    padding: 26px 28px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.info-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(74,160,79,0.2), transparent);
}
.info-card:hover { border-color: rgba(74,160,79,0.22); }
.info-card-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(74,160,79,0.8);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.info-card-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(74,160,79,0.1);
}
.info-card-body {
    font-size: 14px;
    line-height: 1.75;
    color: rgba(240,235,225,0.72);
    font-weight: 300;
}

.fert-list { display: flex; flex-direction: column; gap: 0; }
.fert-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 13px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.fert-item:last-child { border-bottom: none; padding-bottom: 0; }
.fert-num {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: rgba(74,160,79,0.12);
    border: 1px solid rgba(74,160,79,0.2);
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #a5d6a7;
    flex-shrink: 0;
    margin-top: 1px;
}
.fert-name { font-size: 14px; font-weight: 500; color: #f0ebe1; }
.fert-detail {
    font-size: 11px;
    color: rgba(240,235,225,0.35);
    font-family: 'DM Mono', monospace;
    margin-top: 3px;
}

.section-hd { margin-bottom: 20px; }
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #f0ebe1;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 12px;
    color: rgba(240,235,225,0.35);
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
}

.stat-strip-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 32px; }
.stat-pill {
    flex: 1; min-width: 90px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(74,160,79,0.1);
    border-radius: 14px;
    padding: 13px 16px;
    position: relative;
    overflow: hidden;
    transition: all 0.2s;
}
.stat-pill::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(74,160,79,0.3), transparent);
}
.stat-pill.wx-source::before {
    background: linear-gradient(90deg, transparent, rgba(232,160,32,0.4), transparent);
}
.stat-pill:hover {
    border-color: rgba(74,160,79,0.2);
    background: rgba(255,255,255,0.04);
    transform: translateY(-1px);
}
.stat-pill-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(240,235,225,0.3);
    margin-bottom: 5px;
}
.stat-pill-val {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #f0ebe1;
    line-height: 1;
}
.stat-pill-unit {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: rgba(240,235,225,0.35);
    margin-left: 3px;
}
.wx-badge {
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 1px;
    color: rgba(232,160,32,0.7);
    margin-top: 4px;
}

.footer {
    text-align: center;
    padding: 28px 0 16px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: rgba(240,235,225,0.15);
    text-transform: uppercase;
}
.footer-dot { display: inline-block; margin: 0 10px; opacity: 0.4; }
</style>
""", unsafe_allow_html=True)

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_default_csv = os.path.normpath(os.path.join(BASE_DIR, '..', 'Datasets', 'Crop_recommendation.csv'))
CSV_PATH = os.getenv("AGRISENS_CSV_PATH", _default_csv)
PKL_PATH = os.path.join(BASE_DIR, 'RF.pkl')

# ── CROP DATABASE ─────────────────────────────────────────────
CROP_DB = {
    "rice":        {"desc": "Thrives in waterlogged warm conditions. Ideal for clay-heavy, slightly acidic soils.",              "advice": "Maintain 5–10 cm standing water. Apply urea in split doses.",                           "ferts": [{"n": "Urea",              "d": "50 kg N/ha at transplanting"},    {"n": "DAP",            "d": "25 kg P₂O₅/ha basal dose"}]},
    "maize":       {"desc": "Demands high nitrogen and moderate water. Best in deep, well-drained loam soils.",                  "advice": "Apply zinc sulfate if pH > 7. Ridge planting aids drainage.",                           "ferts": [{"n": "Urea",              "d": "120 kg/ha split 3×"},              {"n": "MOP",            "d": "60 kg K₂O/ha basal"}]},
    "chickpea":    {"desc": "Nitrogen-fixing legume. Suits well-drained, neutral to slightly alkaline soils.",                   "advice": "Inoculate seeds with Rhizobium. Avoid waterlogging at all stages.",                   "ferts": [{"n": "SSP",               "d": "40 kg P₂O₅/ha at sowing"},        {"n": "Gypsum",         "d": "250 kg/ha for sulfur-deficient soils"}]},
    "kidneybeans": {"desc": "Warm-season legume preferring loose, fertile soil with good drainage.",                             "advice": "Mulch to retain moisture. Excess nitrogen inhibits N-fixation.",                        "ferts": [{"n": "DAP",               "d": "30 kg/ha basal"},                 {"n": "Borax",          "d": "1 kg/ha for boron-deficient soils"}]},
    "pigeonpeas":  {"desc": "Drought-tolerant legume suited to shallow, infertile soils.",                                       "advice": "Intercrop with cereals. Deep taproot means minimal irrigation needed.",                 "ferts": [{"n": "SSP",               "d": "50 kg/ha at sowing"},              {"n": "Rhizobium",      "d": "25 g/kg seed as inoculant"}]},
    "mothbeans":   {"desc": "Extreme drought tolerance; grows in sandy, arid soils with very low rainfall.",                    "advice": "No irrigation in arid zones. Harvest before monsoon ends.",                             "ferts": [{"n": "Urea",              "d": "20 kg N/ha starter dose only"}]},
    "mungbean":    {"desc": "Short-duration legume improving soil structure. Moderate water needs.",                             "advice": "60–70 day cycle. Excellent green manure option post-harvest.",                         "ferts": [{"n": "DAP",               "d": "25 kg/ha at planting"},            {"n": "MOP",            "d": "20 kg/ha"}]},
    "blackgram":   {"desc": "Grows in various soils; prefers well-drained fertile loam.",                                       "advice": "Avoid saline conditions. Grow post-kharif for double cropping.",                      "ferts": [{"n": "SSP",               "d": "40 kg/ha"},                        {"n": "Urea",           "d": "20 kg N/ha starter"}]},
    "lentil":      {"desc": "Cool-season legume needing moderate, well-distributed rainfall.",                                  "advice": "Excellent for rotation post-cereal. Sensitive to frost.",                               "ferts": [{"n": "DAP",               "d": "40 kg/ha basal"},                  {"n": "Zinc Sulfate",   "d": "25 kg/ha if deficient"}]},
    "pomegranate": {"desc": "Drought and salinity tolerant fruit crop. Thrives in semi-arid climates.",                         "advice": "Avoid waterlogging. Prune for open-center canopy to boost yield.",                    "ferts": [{"n": "Vermicompost",      "d": "10 kg/plant annually"},            {"n": "NPK 10:10:10",   "d": "500 g/plant in two splits"}]},
    "banana":      {"desc": "High potassium demand; thrives in deep, rich, well-drained loam with high humidity.",              "advice": "Drip irrigation preferred. Remove lateral shoots to focus energy.",                   "ferts": [{"n": "Urea",              "d": "200 g/plant in 4 splits"},         {"n": "MOP",            "d": "300 g/plant — high K requirement"}]},
    "mango":       {"desc": "Tropical tree crop adaptable to deep, well-drained sandy loam.",                                   "advice": "Prune after harvest. Apply micronutrients yearly for sustained yield.",                "ferts": [{"n": "NPK 12:32:16",      "d": "500 g/tree at flowering"},         {"n": "Urea",           "d": "1 kg/tree post-harvest"}]},
    "grapes":      {"desc": "Prefers dry climate, deep soil, and excellent drainage. pH 5.5–7.0.",                              "advice": "Train on trellis. Potassium critical for sugar accumulation.",                         "ferts": [{"n": "Potassium Nitrate", "d": "20 g/vine fortnightly"},          {"n": "Calcium Nitrate","d": "15 g/vine bi-weekly"}]},
    "watermelon":  {"desc": "Warm-season crop needing sandy loam, good drainage, and ample sunshine.",                          "advice": "Plant on raised beds. Regular irrigation critical during fruit set.",                  "ferts": [{"n": "DAP",               "d": "40 kg/ha basal"},                  {"n": "MOP",            "d": "60 kg/ha in 2 splits"}]},
    "muskmelon":   {"desc": "Thrives in warm, arid zones with well-drained sandy soil. Frost-sensitive.",                       "advice": "Avoid overhead irrigation — promotes fungal disease.",                                 "ferts": [{"n": "Urea",              "d": "50 kg N/ha in splits"},            {"n": "SSP",            "d": "40 kg P/ha basal"}]},
    "apple":       {"desc": "Requires cool winters for dormancy break. Prefers deep fertile well-drained loam.",                "advice": "Thin fruit to one per cluster. Apply lime if pH < 6.",                                "ferts": [{"n": "Urea",              "d": "500 g/tree in spring"},            {"n": "Borax",          "d": "0.3% foliar spray at bloom"}]},
    "orange":      {"desc": "Subtropical citrus needing mild winters and consistent moisture.",                                 "advice": "Mulch tree basins. Monitor closely for citrus greening disease.",                     "ferts": [{"n": "NPK 15:15:15",      "d": "500 g/tree 4× per year"},          {"n": "Zinc Sulfate",   "d": "0.5% foliar 2× per year"}]},
    "papaya":      {"desc": "Fast-growing tropical fruit with very high nutrient demand. Dislikes cold.",                       "advice": "High N in early stages; shift to K at fruiting stage.",                               "ferts": [{"n": "Urea",              "d": "200 g/plant monthly"},             {"n": "MOP",            "d": "150 g/plant at fruiting"}]},
    "coconut":     {"desc": "Coastal crop adapted to high humidity, sandy loam, and saline tolerance.",                         "advice": "Apply green manure. Potassium lifts yield significantly.",                              "ferts": [{"n": "NPK 12:5:21",       "d": "1 kg/palm biannually"},            {"n": "Common Salt",    "d": "2 kg/palm as saline buffer"}]},
    "cotton":      {"desc": "High water and potassium consumer. Best in deep black cotton (Vertisol) soils.",                   "advice": "Bollworm IPM essential. Avoid excess N — promotes vegetative growth.",                "ferts": [{"n": "Urea",              "d": "80 kg N/ha split 3×"},             {"n": "MOP",            "d": "60 kg K₂O/ha basal"}]},
    "jute":        {"desc": "Thrives in humid tropical climate with loamy alluvial soil and moderate flooding.",                "advice": "Retting in slow-moving water for 2–3 weeks post-harvest.",                           "ferts": [{"n": "Urea",              "d": "60 kg N/ha"},                      {"n": "SSP",            "d": "40 kg P/ha basal"}]},
    "coffee":      {"desc": "Shade-loving; thrives in volcanic, well-drained, slightly acidic soil.",                           "advice": "Maintain 50% shade cover. Mulch heavily. pH 5.5–6.5 is critical.",                   "ferts": [{"n": "NPK 17:17:17",      "d": "250 g/plant 2× yearly"},           {"n": "Borax",          "d": "0.2% foliar for berry fill"}]},
}

# ── DATA ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH), False   
    st.warning(
        f"⚠️ Dataset not found at `{CSV_PATH}`. "
        "Set the AGRISENS_CSV_PATH env var or place the file at the default path. "
        "Running on synthetic data — predictions may be less accurate."
    )
    np.random.seed(42)
    rows = []
    for crop in CROP_DB.keys():
        for _ in range(100):
            rows.append({
                "N": np.random.randint(0, 140), "P": np.random.randint(0, 145),
                "K": np.random.randint(0, 205),
                "temperature": round(np.random.uniform(8, 44), 1),
                "humidity":    round(np.random.uniform(14, 100), 1),
                "ph":          round(np.random.uniform(3.5, 9.5), 1),
                "rainfall":    round(np.random.uniform(20, 300), 1),
                "label":       crop
            })
    return pd.DataFrame(rows), True

@st.cache_resource
def load_model(_df, _version_token: str):
    if os.path.exists(PKL_PATH):
        try:
            with open(PKL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ RF.pkl incompatible — retraining. ({e})")
            os.remove(PKL_PATH)
    X = _df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = _df['label']
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X, y)
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(m, f)
    st.success("✅ Model trained and cached.")
    return m

df, is_synthetic = load_data()
_df_token = str(pd.util.hash_pandas_object(df).sum())
model = load_model(df, _df_token)

# ── WEATHER ───────────────────────────────────────────────────
RAINFALL_FALLBACK_MM = 100.0

@st.cache_data(ttl=900)
def get_weather(city: str) -> dict:
    try:
        wx  = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5).json()
        cur = wx['current_condition'][0]
        temp     = float(cur['temp_C'])
        humidity = float(cur['humidity'])

        daily_precip = sum(
            float(h['precipMM'])
            for w in wx.get('weather', [])[:1]   
            for h in w.get('hourly', [])
        )
        rainfall_mm    = round(daily_precip, 1)
        rainfall_label = "Today mm"

        desc = cur['weatherDesc'][0]['value']
        return {
            "city":           city,
            "temp":           temp,
            "humidity":       humidity,
            "rainfall":       rainfall_mm,
            "rainfall_label": rainfall_label,
            "desc":           desc,
        }
    except Exception as e:
        st.sidebar.warning(f"⚠️ Weather fetch failed: {e}")
        return {
            "city":           city,
            "temp":           28.0,
            "humidity":       65.0,
            "rainfall":       RAINFALL_FALLBACK_MM,
            "rainfall_label": "Rain mm",
            "desc":           "Unavailable",
        }

# ── GROQ AI ───────────────────────────────────────────────────
def get_ai_insights(crop: str, inputs: dict):
    if not _groq_client:
        return None
    try:
        prompt = (
            f"Return ONLY valid JSON — no extra text, no markdown fences.\n"
            f"Crop: {crop}\n"
            f"N={inputs['N']}, P={inputs['P']}, K={inputs['K']}\n"
            f"Temp={inputs['temperature']}, Humidity={inputs['humidity']}\n"
            f"pH={inputs['ph']}, Rainfall={inputs['rainfall']}\n\n"
            '{"description":"2-sentence agronomic description",'
            '"fertilizers":[{"name":"","npk":"","dosage":"","time":""}],'
            '"advice":"3-4 specific actionable tips"}'
        )
        res = _groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL
        )
        raw   = re.sub(r"```json|```", "", res.choices[0].message.content).strip()
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            st.warning("⚠️ AI returned no JSON object — using built-in crop data.")
            return None
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            st.warning("⚠️ AI returned unexpected JSON structure — using built-in crop data.")
            return None
        return parsed
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ AI JSON parse error: {e} — using built-in crop data.")
        return None
    except Exception as e:
        st.warning(f"⚠️ AI call failed: {e} — using built-in crop data.")
        return None

# ── PLOTLY THEME ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Bricolage Grotesque", color="rgba(240,235,225,0.6)", size=12),
    colorway=["#4aa04f", "#f5c842", "#4db6ac", "#8bc34a", "#ff9800", "#2e7d32", "#cddc39", "#00897b"],
    hoverlabel=dict(
        bgcolor="rgba(8, 15, 9, 0.95)",
        bordercolor="rgba(74, 160, 79, 0.3)",
        font=dict(family="DM Mono", size=12, color="#f0ebe1")
    ),
    xaxis=dict(showgrid=False, zeroline=False, linecolor="rgba(255,255,255,0.06)", tickfont=dict(size=11), automargin=True),
    yaxis=dict(gridcolor="rgba(255,255,255,0.03)", zeroline=False, linecolor="rgba(255,255,255,0.0)", tickfont=dict(size=11), automargin=True),
    margin=dict(l=20, r=20, t=60, b=20),
)

chart_title_font = dict(family="DM Serif Display", size=18, color="#f0ebe1")

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown("""
        <div class="brand-wrap">
          <div class="brand-name"><div class="brand-dot"></div>AgriSens</div>
          <div class="brand-sub">AI Crop Intelligence · v2.2</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Location</div>', unsafe_allow_html=True)
        city_input = st.text_input("City name", value="Bangalore", placeholder="Enter city…", label_visibility="collapsed")
        weather = get_weather(city_input)

        st.markdown(f"""
        <div class="wx-card">
          <div class="wx-header">
            <div class="wx-label">Auto-detected</div>
            <div class="live-pip"><div class="live-pip-dot"></div>live</div>
          </div>
          <div class="wx-grid">
            <div><div class="wx-cell-val">{weather['temp']:.0f}°</div><div class="wx-cell-lbl">Temp °C</div></div>
            <div><div class="wx-cell-val">{weather['humidity']:.0f}%</div><div class="wx-cell-lbl">Humidity</div></div>
            <div><div class="wx-cell-val">{weather['rainfall']:.1f}</div><div class="wx-cell-lbl">{weather['rainfall_label']}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Macronutrients</div>', unsafe_allow_html=True)
        n = st.slider("Nitrogen (N) · kg/ha",  0,  140, 50)
        p = st.slider("Phosphorus (P) · kg/ha", 0,  145, 40)
        k = st.slider("Potassium (K) · kg/ha",  0,  205, 40)

        st.markdown('<div class="sb-section">Soil Chemistry</div>', unsafe_allow_html=True)
        ph = st.slider("pH Level", 0.0, 14.0, 6.5, step=0.1)

        st.markdown('<div class="sb-section">Climate Inputs</div>', unsafe_allow_html=True)
        st.caption("Pre-filled from live weather — adjust if needed.")
        temp_input     = st.slider("Temperature · °C",  0.0,  50.0, float(weather['temp']),     step=0.5)
        humidity_input = st.slider("Humidity · %",      0.0, 100.0, float(weather['humidity']), step=1.0)
        rainfall_input = st.slider("Rainfall · mm",     0.0, 300.0, float(weather['rainfall']), step=1.0)

        health  = min(100, int((n/140)*25 + (p/145)*25 + (k/205)*25 + (1 - abs(ph-6.5)/7.5)*25))
        h_color = "#4aa04f" if health >= 65 else ("#ff9800" if health >= 40 else "#ef5350")
        h_label = "Excellent" if health >= 75 else ("Good" if health >= 55 else ("Fair" if health >= 35 else "Poor"))

        radius = 28
        circ   = 2 * 3.14159 * radius
        dash   = (health / 100) * circ
        st.markdown(f"""
        <div class="gauge-wrap">
          <svg width="64" height="64" viewBox="0 0 64 64">
            <circle cx="32" cy="32" r="{radius}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="5"/>
            <circle cx="32" cy="32" r="{radius}" fill="none" stroke="{h_color}" stroke-width="5" stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-dashoffset="{circ/4:.1f}" stroke-linecap="round" style="filter:drop-shadow(0 0 4px {h_color}88)"/>
            <text x="32" y="35" text-anchor="middle" font-family="DM Mono,monospace" font-size="11" fill="{h_color}" font-weight="500">{health}%</text>
          </svg>
          <div class="gauge-label-block">
            <div class="gauge-label">Soil Health</div>
            <div class="gauge-value">{h_label}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("🌱  Analyze & Predict Crop")

        # --- ANIMATION CONTROLS ---
        st.markdown('<div class="sb-section">Animation Speed</div>', unsafe_allow_html=True)
        st.caption("Adjust chart stagger delay (ms)")
        
        def sync_from_num():
            st.session_state.anim_delay = st.session_state.delay_num_input

        def sync_from_slider():
            st.session_state.anim_delay = st.session_state.delay_slider_input

        col_slider, col_input = st.columns([2, 1])
        with col_slider:
            st.slider("Delay Slider", 0, 500, 
                      value=st.session_state.anim_delay, 
                      key="delay_slider_input", 
                      on_change=sync_from_slider, 
                      label_visibility="collapsed")
        with col_input:
            st.number_input("Delay Input", 0, 500, 
                            value=st.session_state.anim_delay, 
                            key="delay_num_input", 
                            on_change=sync_from_num, 
                            label_visibility="collapsed")
        
        delay_ms = st.session_state.anim_delay

    col_hero, col_wx = st.columns([3, 1])

    with col_hero:
        data_note = " · synthetic data" if is_synthetic else ""
        st.markdown(f"""
        <div class="hero-wrap">
          <div class="hero-eyebrow">AI-powered crop intelligence</div>
          <div class="hero-title">Know what your<br><em>soil</em> wants to grow</div>
          <div class="hero-sub">Input soil nutrient levels and let our ML model surface the ideal crop — trained on 2,200+ agronomic data points across 22 crop types.</div>
          <div class="badge-row">
            <div class="badge badge-green">22 Crop Types</div>
            <div class="badge badge-green">Random Forest{data_note}</div>
            <div class="badge badge-amber">📍 {weather['city']}</div>
            <div class="badge badge-amber">{weather['desc']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_wx:
        st.markdown(f"""
        <div class="wx-hero">
          <div class="wx-hero-city">{weather['city'].upper()}</div>
          <div class="wx-hero-temp">{weather['temp']:.0f}<sup>°C</sup></div>
          <div class="wx-hero-desc">{weather['desc']}<br>{weather['humidity']:.0f}% humidity</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    params = [
        ("Nitrogen",    n,              "kg/ha", False),
        ("Phosphorus",  p,              "kg/ha", False),
        ("Potassium",   k,              "kg/ha", False),
        ("pH",          ph,             "",      False),
        ("Temperature", temp_input,     "°C",    True),
        ("Humidity",    humidity_input, "%",     True),
        ("Rainfall",    rainfall_input, "mm",    True),
    ]
    
    pills_html = '<div class="stat-strip-row">'
    for lbl, val, unit, from_wx in params:
        v_str      = f"{val:.1f}" if isinstance(val, float) else str(val)
        wx_cls     = " wx-source" if from_wx else ""
        wx_sub     = '<div class="wx-badge">⬡ weather</div>' if from_wx else ""
        pills_html += f'<div class="stat-pill{wx_cls}"><div class="stat-pill-lbl">{lbl}</div><div class="stat-pill-val">{v_str}<span class="stat-pill-unit">{unit}</span></div>{wx_sub}</div>'
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    if predict_clicked:
        features   = [[n, p, k, temp_input, humidity_input, ph, rainfall_input]]
        crop       = model.predict(features)[0]
        st.session_state.predicted_crop = crop.lower()
        probs      = model.predict_proba(features)[0]
        confidence = int(max(probs) * 100)
        info       = CROP_DB.get(crop.lower(), CROP_DB["rice"])
        name       = crop.capitalize()

        ai = None
        if _groq_client:
            with st.spinner("Consulting AI…"):
                ai = get_ai_insights(crop, {
                    "N": n, "P": p, "K": k, "temperature": temp_input, "humidity": humidity_input, "ph": ph, "rainfall": rainfall_input,
                })

        desc   = ai.get("description", info["desc"])  if isinstance(ai, dict) else info["desc"]
        advice = ai.get("advice",      info["advice"]) if isinstance(ai, dict) else info["advice"]
        ferts  = ai.get("fertilizers", [])             if isinstance(ai, dict) else [
            {"name": f["n"], "npk": "", "dosage": f["d"], "time": ""} for f in info["ferts"]
        ]

        st.markdown(f"""
        <div class="result-card">
          <div class="result-eyebrow">Recommended Crop</div>
          <div class="result-crop-name"><span>{name}</span></div>
          <div class="conf-row">
            <div class="conf-label">Confidence</div>
            <div class="conf-track"><div class="conf-fill" style="--conf-w: {confidence}%;"></div></div>
            <div class="conf-pct">{confidence}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"""
            <div class="info-card"><div class="info-card-label">About This Crop</div><div class="info-card-body">{desc}</div></div>
            """, unsafe_allow_html=True)
        with col_r:
            st.markdown(f"""
            <div class="info-card"><div class="info-card-label">Soil Advice</div><div class="info-card-body">{advice}</div></div>
            """, unsafe_allow_html=True)

        fert_items = ""
        for i, f in enumerate(ferts, 1):
            npk_part  = f"  ·  {f['npk']}"  if f.get("npk")  else ""
            time_part = f"  ·  {f['time']}" if f.get("time") else ""
            fert_items += f"""
            <div class="fert-item">
              <div class="fert-num">{i:02d}</div>
              <div><div class="fert-name">{f.get('name', '—')}{npk_part}</div><div class="fert-detail">{f.get('dosage', '—')}{time_part}</div></div>
            </div>"""

        st.markdown(f"""
        <div class="info-card"><div class="info-card-label">Fertilizer Recommendations</div><div class="fert-list">{fert_items}</div></div>
        """, unsafe_allow_html=True)

        top5_idx   = np.argsort(probs)[::-1][:5]
        top5_crops = [model.classes_[i].capitalize() for i in top5_idx][::-1]
        top5_probs = [round(float(probs[i]) * 100, 1) for i in top5_idx][::-1]

        options = {
            "backgroundColor": "transparent",
            "animationDuration": 1800, 
            "animationEasing": "elasticOut",
            "animationDelay": JsCode(f"function(idx) {{ return idx * {delay_ms}; }}").js_code,
            "grid": {"left": "3%", "right": "12%", "bottom": "3%", "top": "10%", "containLabel": True},
            "xAxis": {"type": "value", "show": False, "max": 100},
            "yAxis": {"type": "category", "data": top5_crops, "axisLine": {"show": False}, "axisTick": {"show": False}, "axisLabel": {"color": "rgba(240,235,225,0.7)", "fontFamily": "DM Mono", "fontSize": 12}},
            "series": [{"data": top5_probs, "type": "bar", "itemStyle": {"color": "#4aa04f", "borderRadius": [0, 4, 4, 0]}, "label": {"show": True, "position": "right", "formatter": "{c}%", "color": "#f0ebe1", "fontFamily": "DM Mono"}}]
        }

        st.markdown('<div style="font-family: \'DM Serif Display\', serif; font-size: 18px; color: #f0ebe1; margin-bottom: -10px; margin-left: 10px;">Top 5 Alternatives</div>', unsafe_allow_html=True)
        st_echarts(options=options, height="240px")


    # ─────────────────────────────────────────────────────────
    # DATA TABS
    # ─────────────────────────────────────────────────────────
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-hd">
      <div class="section-title">Data Explorer</div>
      <div class="section-sub">Agronomic insights across 22 crop types</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  📊  Distribution  ", "  🌡️  Conditions  ", "  ⚗️  Nutrients  "])

    with tab1:
        c1, c2 = st.columns(2)
        
        # --- 1. Records per Crop (ECharts) ---
        counts = df['label'].value_counts().reset_index()
        counts.columns = ['Crop', 'Count']
        
        options_fig1 = {
            "backgroundColor": "transparent",
            "animationDuration": 2000, 
            "animationEasing": "cubicInOut",
            "animationDelay": JsCode(f"function(idx) {{ return idx * {delay_ms}; }}").js_code,
            "tooltip": {
                "trigger": "axis", "backgroundColor": "rgba(8, 15, 9, 0.95)", "borderColor": "rgba(74, 160, 79, 0.3)",
                "textStyle": {"color": "#f0ebe1", "fontFamily": "DM Mono", "fontSize": 12}
            },
            "grid": {"left": "3%", "right": "3%", "bottom": "5%", "top": "10%", "containLabel": True},
            "xAxis": {
                "type": "category", "data": counts['Crop'].tolist(),
                "axisLabel": {"rotate": 45, "color": "rgba(240,235,225,0.6)", "fontFamily": "Bricolage Grotesque", "fontSize": 11},
                "axisTick": {"show": False}, "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.06)"}}
            },
            "yAxis": {
                "type": "value", "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.03)"}},
                "axisLabel": {"color": "rgba(240,235,225,0.6)", "fontFamily": "Bricolage Grotesque", "fontSize": 11}
            },
            "series": [{"data": counts['Count'].tolist(), "type": "bar", "itemStyle": {"color": "#4aa04f", "borderRadius": [4, 4, 0, 0]}}]
        }
        
        c1.markdown('<div style="font-family: \'DM Serif Display\', serif; font-size: 18px; color: #f0ebe1; margin-bottom: 5px;">Records per Crop</div>', unsafe_allow_html=True)
        with c1:
            st_echarts(options=options_fig1, height="320px")

        # --- 2. Average pH by Crop (Distributed Column Chart) ---
        avg_ph = df.groupby('label')['ph'].mean().reset_index().rename(columns={'ph': 'Avg pH'})
        
        # Custom JS color function for the Distributed look
        colors_js = JsCode("""
        function(params) {
            var colorList = ['#4aa04f', '#f5c842', '#4db6ac', '#8bc34a', '#ff9800', '#2e7d32', '#cddc39', '#00897b', '#e53935', '#8e24aa', '#3949ab', '#039be5', '#00acc1', '#43a047', '#7cb342', '#c0ca33', '#fdd835', '#ffb300', '#fb8c00', '#f4511e', '#6d4c41', '#757575'];
            return colorList[params.dataIndex % colorList.length];
        }
        """).js_code

        options_fig2 = {
            "backgroundColor": "transparent",
            "animationDuration": 2000, 
            "animationEasing": "cubicOut",
            # Zig-zag pattern utilizing the dynamic delay input
            "animationDelay": JsCode(f"""
                function(idx) {{ 
                    return (idx % 2 === 0 ? 0 : 500) + (idx * {int(delay_ms * 0.2)}); 
                }}
            """).js_code,
            "tooltip": {
                "trigger": "axis", "backgroundColor": "rgba(8, 15, 9, 0.95)", "borderColor": "rgba(74, 160, 79, 0.3)",
                "textStyle": {"color": "#f0ebe1", "fontFamily": "DM Mono", "fontSize": 12}
            },
            "grid": {"left": "3%", "right": "3%", "bottom": "5%", "top": "10%", "containLabel": True},
            "xAxis": {
                "type": "category", "data": avg_ph['label'].tolist(),
                "axisLabel": {"rotate": 45, "color": "rgba(240,235,225,0.6)", "fontFamily": "Bricolage Grotesque", "fontSize": 11},
                "axisTick": {"show": False}, "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.06)"}}
            },
            "yAxis": {
                "type": "value", "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.03)"}},
                "axisLabel": {"color": "rgba(240,235,225,0.6)", "fontFamily": "Bricolage Grotesque", "fontSize": 11}
            },
            "series": [{
                "data": [round(val, 2) for val in avg_ph['Avg pH'].tolist()],
                "type": "bar",
                "itemStyle": {"color": colors_js, "borderRadius": [4, 4, 0, 0]}
            }]
        }
        
        c2.markdown('<div style="font-family: \'DM Serif Display\', serif; font-size: 18px; color: #f0ebe1; margin-bottom: 5px;">Average pH by Crop</div>', unsafe_allow_html=True)
        with c2:
            st_echarts(options=options_fig2, height="320px")


    with tab2:
        # --- 3. Temperature Range per Crop (Range Column Chart) ---
        temp_range = df.groupby('label')['temperature'].agg(min_temp='min', max_temp='max').reset_index()
        bases = temp_range['min_temp'].round(1).tolist()
        ranges = (temp_range['max_temp'] - temp_range['min_temp']).round(1).tolist()
        labels = temp_range['label'].tolist()

        options_fig3 = {
            "backgroundColor": "transparent",
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "backgroundColor": "rgba(8, 15, 9, 0.95)",
                "borderColor": "rgba(74, 160, 79, 0.3)",
                "textStyle": {"color": "#f0ebe1", "fontFamily": "DM Mono", "fontSize": 12},
                "formatter": JsCode("""
                    function(params) {
                        var min = params[0].value;
                        var range = params[1].value;
                        var max = (min + range).toFixed(1);
                        return params[0].name + '<br/>Min: ' + min + ' °C<br/>Max: ' + max + ' °C';
                    }
                """).js_code
            },
            "grid": {"left": "2%", "right": "2%", "bottom": "5%", "top": "15%", "containLabel": True},
            "xAxis": {
                "type": "category", "data": labels,
                "axisLabel": {"rotate": 45, "color": "rgba(240,235,225,0.6)", "fontFamily": "Bricolage Grotesque", "fontSize": 11},
                "axisTick": {"show": False}, "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.06)"}}
            },
            "yAxis": {
                "type": "value", "name": "°C",
                "nameTextStyle": {"color": "rgba(240,235,225,0.4)", "fontFamily": "DM Mono", "padding": [0, 0, 0, 10]},
                "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.03)"}},
                "axisLabel": {"color": "rgba(240,235,225,0.6)", "fontFamily": "Bricolage Grotesque", "fontSize": 11}
            },
            "series": [
                {
                    "name": "Base", "type": "bar", "stack": "Temp",
                    "itemStyle": {"borderColor": "transparent", "color": "transparent"},
                    "emphasis": {"itemStyle": {"borderColor": "transparent", "color": "transparent"}},
                    "data": bases
                },
                {
                    "name": "Range", "type": "bar", "stack": "Temp",
                    "itemStyle": {"color": "#ff9800", "borderRadius": 4},
                    "animationDuration": 2000, 
                    "animationEasing": "cubicInOut",
                    "animationDelay": JsCode(f"function(idx) {{ return idx * {delay_ms}; }}").js_code,
                    "data": ranges
                }
            ]
        }

        st.markdown('<div style="font-family: \'DM Serif Display\', serif; font-size: 18px; color: #f0ebe1; margin-bottom: 5px;">Temperature Range by Crop</div>', unsafe_allow_html=True)
        st_echarts(options=options_fig3, height="340px")


    with tab3:
        # --- 4. Average NPK Profile per Crop (Stacked Column Chart) ---
        avg_npk = df.groupby('label')[['N', 'P', 'K']].mean().reset_index()
        
        options_fig5 = {
            "backgroundColor": "transparent",
            "tooltip": {
                "trigger": "axis", "axisPointer": {"type": "shadow"},
                "backgroundColor": "rgba(8, 15, 9, 0.95)", "borderColor": "rgba(74, 160, 79, 0.3)",
                "textStyle": {"color": "#f0ebe1", "fontFamily": "DM Mono", "fontSize": 12}
            },
            "legend": {
                "data": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
                "textStyle": {"color": "rgba(240,235,225,0.6)", "fontFamily": "Bricolage Grotesque", "fontSize": 12},
                "top": "0%", "right": "1%"
            },
            "grid": {"left": "2%", "right": "2%", "bottom": "5%", "top": "15%", "containLabel": True},
            "xAxis": {
                "type": "category", "data": avg_npk['label'].tolist(),
                "axisLabel": {"rotate": 45, "color": "rgba(240,235,225,0.6)", "fontFamily": "Bricolage Grotesque", "fontSize": 11},
                "axisTick": {"show": False}, "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.06)"}}
            },
            "yAxis": {
                "type": "value", "name": "kg/ha",
                "nameTextStyle": {"color": "rgba(240,235,225,0.4)", "fontFamily": "DM Mono", "padding": [0, 0, 0, 10]},
                "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.03)"}},
                "axisLabel": {"color": "rgba(240,235,225,0.6)", "fontFamily": "Bricolage Grotesque", "fontSize": 11}
            },
            "series": [
                {
                    "name": "Nitrogen (N)", "type": "bar", "stack": "total", "data": [round(x) for x in avg_npk['N'].tolist()],
                    "itemStyle": {"color": "#4aa04f"},
                    "animationDuration": 2000, "animationEasing": "cubicOut",
                    "animationDelay": JsCode(f"""
                        function(idx) {{ 
                            return (idx % 2 === 0 ? 0 : 500) + (idx * {int(delay_ms * 0.2)}); 
                        }}
                    """).js_code
                },
                {
                    "name": "Phosphorus (P)", "type": "bar", "stack": "total", "data": [round(x) for x in avg_npk['P'].tolist()],
                    "itemStyle": {"color": "#f5c842"},
                    "animationDuration": 2000, "animationEasing": "cubicOut",
                    "animationDelay": JsCode(f"""
                        function(idx) {{ 
                            return (idx % 2 === 0 ? 0 : 500) + (idx * {int(delay_ms * 0.2)}) + 150; 
                        }}
                    """).js_code 
                },
                {
                    "name": "Potassium (K)", "type": "bar", "stack": "total", "data": [round(x) for x in avg_npk['K'].tolist()],
                    "itemStyle": {"color": "#4db6ac", "borderRadius": [4, 4, 0, 0]},
                    "animationDuration": 2000, "animationEasing": "cubicOut",
                    "animationDelay": JsCode(f"""
                        function(idx) {{ 
                            return (idx % 2 === 0 ? 0 : 500) + (idx * {int(delay_ms * 0.2)}) + 300; 
                        }}
                    """).js_code 
                }
            ]
        }
        
        st.markdown('<div style="font-family: \'DM Serif Display\', serif; font-size: 18px; color: #f0ebe1; margin-bottom: 5px;">Average NPK Profile per Crop</div>', unsafe_allow_html=True)
        st_echarts(options=options_fig5, height="340px")

        # Keeping the Correlation Heatmap and Violin Plot
        c1, c2 = st.columns(2)
        fig6 = px.violin(df, x='label', y='ph', color='label', box=True)
        fig6.update_traces(meanline_visible=True, points='outliers', jitter=0.05, line=dict(width=1.5), opacity=0.7)
        fig6.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=320, title=dict(text="pH Distribution by Crop", font=chart_title_font, x=0.01))
        fig6.update_xaxes(tickangle=45, tickfont=dict(size=11))
        c1.plotly_chart(fig6, use_container_width=True, theme=None)

        corr = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].corr(numeric_only=True)
        fig7 = go.Figure(go.Heatmap(
            z=corr.values.tolist(), x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale=[[0, "#080f09"], [0.5, "#1e5c22"], [1, "#f5c842"]],
            zmin=-1, zmax=1, texttemplate="%{z:.2f}",
            textfont=dict(size=11, family="DM Mono", color="#f0ebe1"), hoverongaps=False,
        ))
        fig7.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Feature Correlation Heatmap", font=chart_title_font, x=0.01))
        c2.plotly_chart(fig7, use_container_width=True, theme=None)

    # ─────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
      AgriSens
      <span class="footer-dot">·</span>
      ML Crop Intelligence
      <span class="footer-dot">·</span>
      Streamlit + scikit-learn + Groq
      <span class="footer-dot">·</span>
      v2.2
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()