# 🌾 AgriSens — Smart Farming Assistant

AgriSens is a full-stack smart farming platform that combines a static web landing page with a Streamlit-powered ML dashboard for crop recommendation. It helps farmers make data-driven decisions using soil parameters, real-time weather, and an optional AI-powered insight engine.

---

## ✨ Features

- **Crop Recommendation** — Predicts the best crop to grow based on soil nutrient levels (N, P, K), pH, temperature, humidity, and rainfall using a trained Random Forest model.
- **AI Insights (optional)** — Integrates with Groq's LLM API to generate natural language farming insights for the recommended crop.
- **Live Weather** — Fetches real-time weather data for the user's location, displayed in the sidebar.
- **Data Explorer** — Interactive charts (via Plotly & Apache ECharts) visualizing NPK profiles, pH distributions, temperature ranges, and feature correlation heatmaps across all crops.
- **Web Landing Page** — A standalone HTML/CSS/JS site with navigation to the Streamlit app, weather forecast, farming guide, and fertilizer recommendation (in development).

---

## 🗂️ Project Structure

```
AgriSens/
│
├── AgriSens-web-app/          # Static landing page
│   ├── index.html
│   ├── css/
│   ├── js/
│   ├── weather-forecast/
│   ├── guide/
│   ├── explore/
│   └── developing-phase/
│
└── CROP-RECOMMENDATION/       # Streamlit ML app
    ├── webapp.py              # Main Streamlit dashboard
    ├── Crop_reccom(final).ipynb  # Model training notebook
    ├── Crop_recommendation.csv   # Training dataset
    ├── RF.pkl                 # Trained Random Forest model
    ├── requirements.txt
    └── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/agrisens.git
   cd agrisens/CROP-RECOMMENDATION
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install streamlit-echarts groq python-dotenv   # additional packages
   ```

4. **Set up environment variables** *(optional — required for AI insights)*

   Create a `.env` file in the `CROP-RECOMMENDATION/` directory:

   ```env
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```

   > ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

5. **Run the app**

   ```bash
   streamlit run webapp.py
   ```

   The app will open at `http://localhost:8501`.

---

## 🌐 Web App

The landing page is a static site — simply open `AgriSens-web-app/index.html` in your browser, or host it on any static hosting service (GitHub Pages, Netlify, etc.).

The **Explore Now** button links to the live Streamlit deployment at:  
👉 https://crop-recomm.streamlit.app/

---

## 🤖 ML Model

The crop recommendation model is a **Random Forest Classifier** trained on the `Crop_recommendation.csv` dataset. It takes the following inputs:

| Parameter | Unit |
|---|---|
| Nitrogen (N) | kg/ha |
| Phosphorus (P) | kg/ha |
| Potassium (K) | kg/ha |
| Temperature | °C |
| Humidity | % |
| pH | 0–14 |
| Rainfall | mm |

The trained model is serialized as `RF.pkl` and loaded directly by `webapp.py`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard framework |
| `scikit-learn` | Random Forest model |
| `pandas` / `numpy` | Data processing |
| `plotly` | Interactive charts |
| `streamlit-echarts` | Apache ECharts integration |
| `groq` *(optional)* | LLM-powered crop insights |
| `python-dotenv` *(optional)* | Secure API key loading |

---

## 🔮 Roadmap

- [x] Crop recommendation with Random Forest
- [x] Real-time weather integration
- [x] AI-powered insights via Groq
- [ ] Secure `.env`-based API key management
- [ ] Fertilizer recommendation module
- [ ] Community forum & newsletter

---

## 📄 License

© Team AgriSens. All rights reserved.
