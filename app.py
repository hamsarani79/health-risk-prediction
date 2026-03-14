import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

from symptoms import match_symptoms, get_all_symptoms, DISEASE_SYMPTOMS
from recommend import get_tips, get_lifestyle_tips

st.set_page_config(
    page_title="HealthSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #f7f9fc;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #0f2942 0%, #1a4a7a 60%, #2980b9 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(15,41,66,0.18);
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-subtitle {
    color: #a8cbea;
    font-size: 1rem;
    margin-top: 0.3rem;
    font-weight: 400;
}
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin-right: 6px;
    margin-top: 10px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Section headers */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f2942;
    padding: 0.5rem 0 0.4rem 0.8rem;
    border-left: 4px solid #2980b9;
    margin: 1.5rem 0 1rem 0;
    background: linear-gradient(90deg, #eef4fb, transparent);
    border-radius: 0 8px 8px 0;
}

/* Disease cards */
.disease-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem;
    margin: 0.6rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-top: 4px solid #2980b9;
    transition: transform 0.2s;
}
.disease-card:hover { transform: translateY(-2px); }
.disease-card.high  { border-top-color: #e74c3c; }
.disease-card.med   { border-top-color: #f39c12; }
.disease-card.low   { border-top-color: #27ae60; }

.risk-label-high { color: #e74c3c; font-weight: 700; font-size: 1rem; }
.risk-label-med  { color: #f39c12; font-weight: 700; font-size: 1rem; }
.risk-label-low  { color: #27ae60; font-weight: 700; font-size: 1rem; }

/* Symptom chips */
.symptom-chip {
    display: inline-block;
    background: #eef4fb;
    color: #1a4a7a;
    border: 1px solid #c5ddf4;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.78rem;
    margin: 3px 2px;
}
.symptom-chip.matched {
    background: #fff3cd;
    border-color: #f39c12;
    color: #7d5a00;
    font-weight: 600;
}

/* Tip box */
.tip-box {
    background: #f0f7ff;
    border-left: 3px solid #2980b9;
    padding: 8px 14px;
    border-radius: 0 8px 8px 0;
    margin: 5px 0;
    font-size: 0.9rem;
    color: #1a3a5c;
}

/* Result summary */
.result-header {
    background: linear-gradient(135deg, #0f2942, #1a4a7a);
    color: white;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    text-align: center;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(90deg, #0f2942, #2980b9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.3px;
}

div[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.stProgress > div > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

DISEASE_CONFIG = {
    "Heart Disease":       {"icon": "🫀", "color": "#e74c3c"},
    "Diabetes":            {"icon": "🩸", "color": "#8e44ad"},
    "Stroke":              {"icon": "🧠", "color": "#e67e22"},
    "High Blood Pressure": {"icon": "🩺", "color": "#2980b9"},
    "High Cholesterol":    {"icon": "🧬", "color": "#27ae60"},
}

@st.cache_resource
def load_all_models():
    models = {}
    for disease in DISEASE_CONFIG.keys():
        fname = f"model_{disease.lower().replace(' ', '_')}.pkl"
        if os.path.exists(fname):
            models[disease] = joblib.load(fname)
    if not models:
        models = train_synthetic_models()
    return models

def train_synthetic_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    n = 6000
    features = ["BMI","Smoking","AlcoholDrinking","PhysicalActivity",
                "DiffWalking","Sex","AgeCategory","Education",
                "HighBP","HighChol","Fruits","Veggies"]

    bmi      = np.random.normal(27,6,n).clip(12,60)
    smoking  = np.random.binomial(1,0.44,n)
    alcohol  = np.random.binomial(1,0.07,n)
    phys     = np.random.binomial(1,0.75,n)
    diffwalk = np.random.binomial(1,0.17,n)
    sex      = np.random.binomial(1,0.5,n)
    age      = np.random.randint(1,14,n)
    edu      = np.random.randint(1,7,n)
    highbp   = np.random.binomial(1,0.43,n)
    highchol = np.random.binomial(1,0.42,n)
    fruits   = np.random.binomial(1,0.6,n)
    veggies  = np.random.binomial(1,0.7,n)

    X = np.column_stack([bmi,smoking,alcohol,phys,diffwalk,sex,age,edu,highbp,highchol,fruits,veggies])

    targets = {
        "Heart Disease":       (0.25*highbp + 0.20*highchol + 0.15*smoking + 0.06*(age/13) - 0.10*phys + np.random.normal(0,0.05,n)) > 0.30,
        "Diabetes":            (0.20*(bmi/60) + 0.15*highbp + 0.10*smoking + 0.08*(age/13) - 0.08*phys - 0.05*fruits + np.random.normal(0,0.05,n)) > 0.18,
        "Stroke":              (0.20*highbp + 0.15*smoking + 0.10*alcohol + 0.08*(age/13) + 0.05*diffwalk + np.random.normal(0,0.05,n)) > 0.22,
        "High Blood Pressure": (0.20*(bmi/60) + 0.15*smoking + 0.10*alcohol + 0.10*(age/13) - 0.08*phys + np.random.normal(0,0.05,n)) > 0.25,
        "High Cholesterol":    (0.20*highbp + 0.15*(bmi/60) + 0.10*smoking - 0.08*phys - 0.05*veggies + np.random.normal(0,0.05,n)) > 0.20,
    }

    models = {}
    for disease, y_bool in targets.items():
        y = y_bool.astype(int)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestClassifier(n_estimators=100, random_state=42)
        m.fit(X_train, y_train)
        fname = f"model_{disease.lower().replace(' ', '_')}.pkl"
        joblib.dump((m, features), fname)
        models[disease] = (m, features)
    return models

def make_gauge(risk_pct, disease, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={'suffix': "%", 'font': {'size': 22, 'color': color}},
        title={'text': disease, 'font': {'size': 12, 'color': '#555'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 9}},
            'bar': {'color': color},
            'bgcolor': 'white',
            'steps': [
                {'range': [0, 30],   'color': '#eafaf1'},
                {'range': [30, 60],  'color': '#fef9e7'},
                {'range': [60, 100], 'color': '#fdedec'},
            ],
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def risk_label(pct):
    if pct >= 60:   return "HIGH",   "high",  "risk-label-high"
    elif pct >= 30: return "MEDIUM", "med",   "risk-label-med"
    else:           return "LOW",    "low",   "risk-label-low"

if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "user_input" not in st.session_state:
    st.session_state.user_input = None
# UI STARTS HERE

# Header
st.markdown("""
<div class="app-header">
    <div class="app-title">🫀 HealthSense AI</div>
    <div class="app-subtitle">Personalized Health Risk Prediction Platform — BRFSS 2015 Dataset</div>
    <span class="badge">Random Forest ML</span>
    <span class="badge">5 Disease Predictions</span>
    <span class="badge">Symptom Checker</span>
    <span class="badge">What-If Simulator</span>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs([" Health Form & Prediction", " Symptom Checker"])

# ══════════════════════════════════════════════════════════
# TAB 1 — Health Form
# ══════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-title">Enter Your Health Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("** Personal Details**")
        age_options = {
            "18–24":1,"25–29":2,"30–34":3,"35–39":4,"40–44":5,
            "45–49":6,"50–54":7,"55–59":8,"60–64":9,"65–69":10,
            "70–74":11,"75–79":12,"80+":13
        }
        age_label = st.selectbox("Age Group", list(age_options.keys()))
        age_val   = age_options[age_label]

        sex_input = st.radio("Sex", ["Female","Male"], horizontal=True)
        sex_val   = 1 if sex_input == "Male" else 0

        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                              help="Use a BMI calculator if you don't know yours")

        edu_map = {
            "Never attended school":1, "Elementary school":2,
            "Some high school":3,      "High school graduate":4,
            "Some college":5,          "College graduate":6
        }
        edu_val = edu_map[st.selectbox("Education Level", list(edu_map.keys()))]

    with col2:
        st.markdown("** Medical History**")
        highbp_input   = st.radio("High Blood Pressure?",    ["No","Yes"], horizontal=True)
        highbp_val     = 1 if highbp_input   == "Yes" else 0

        highchol_input = st.radio("High Cholesterol?",       ["No","Yes"], horizontal=True)
        highchol_val   = 1 if highchol_input == "Yes" else 0

        stroke_input   = st.radio("Ever had a Stroke?",      ["No","Yes"], horizontal=True)
        stroke_val     = 1 if stroke_input   == "Yes" else 0

        diffwalk_input = st.radio("Difficulty Walking?",     ["No","Yes"], horizontal=True)
        diffwalk_val   = 1 if diffwalk_input == "Yes" else 0

    with col3:
        st.markdown("** Lifestyle Habits**")
        smoking_input = st.radio("Smoker?",                        ["No","Yes"], horizontal=True)
        smoking_val   = 1 if smoking_input == "Yes" else 0

        alcohol_input = st.radio("Heavy Alcohol Drinker?",         ["No","Yes"], horizontal=True)
        alcohol_val   = 1 if alcohol_input == "Yes" else 0

        phys_input    = st.radio("Physically Active (last 30 days)?", ["No","Yes"], horizontal=True)
        phys_val      = 1 if phys_input    == "Yes" else 0

        fruits_input  = st.radio("Eat Fruits daily?",              ["No","Yes"], horizontal=True)
        fruits_val    = 1 if fruits_input  == "Yes" else 0

        veggies_input = st.radio("Eat Vegetables daily?",          ["No","Yes"], horizontal=True)
        veggies_val   = 1 if veggies_input == "Yes" else 0

    st.markdown("---")
    _, mid, _ = st.columns([1,2,1])
    with mid:
        predict_btn = st.button(" Predict My Health Risks")

    if predict_btn:
        models = load_all_models()

        user_input = {
            "BMI": bmi, "Smoking": smoking_val, "AlcoholDrinking": alcohol_val,

            "PhysicalActivity": phys_val, "DiffWalking": diffwalk_val,
            "Sex": sex_val, "AgeCategory": age_val, "Education": edu_val,
            "HighBP": highbp_val, "HighChol": highchol_val,
            "Fruits": fruits_val, "Veggies": veggies_val,
            "Stroke": stroke_val,
        }


        predictions = {}
        for disease, config in DISEASE_CONFIG.items():
            if disease not in models:
                continue
            model_obj, feat_list = models[disease]
            input_vals = [user_input.get(f, 0) for f in feat_list]
            input_df   = pd.DataFrame([input_vals], columns=feat_list)
            prob       = model_obj.predict_proba(input_df)[0][1]
            predictions[disease] = round(prob * 100, 1)

        st.session_state.predictions = predictions
        st.session_state.user_input  = user_input

    if st.session_state.predictions is not None:
        predictions = st.session_state.predictions
        user_input  = st.session_state.user_input
        models      = load_all_models()

        st.markdown('<div class="section-title">📊 Your Health Risk Dashboard</div>', unsafe_allow_html=True)

        g_cols = st.columns(5)
        for i, (disease, pct) in enumerate(predictions.items()):
            color = DISEASE_CONFIG[disease]["color"]
            icon  = DISEASE_CONFIG[disease]["icon"]
            with g_cols[i]:
                st.plotly_chart(make_gauge(pct, f"{icon} {disease}", color),
                                use_container_width=True, key=f"gauge_{disease}")

        st.markdown('<div class="section-title"> Detailed Risk Report</div>', unsafe_allow_html=True)

        card_cols = st.columns(2)
        for i, (disease, pct) in enumerate(predictions.items()):
            level, css_class, label_class = risk_label(pct)
            icon  = DISEASE_CONFIG[disease]["icon"]
            color = DISEASE_CONFIG[disease]["color"]
            tips  = get_tips(disease)

            with card_cols[i % 2]:
                st.markdown(f"""
                <div class="disease-card {css_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.1rem; font-weight:700; color:#0f2942;">{icon} {disease}</span>
                        <span class="{label_class}">{level} — {pct}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(int(pct))

                with st.expander(" Tips & Recommendations"):
                    for tip in tips:
                        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

        # Lifestyle tips
        st.markdown('<div class="section-title">🌿 Your Lifestyle Summary</div>', unsafe_allow_html=True)
        lifestyle_tips = get_lifestyle_tips(user_input)
        lt_cols = st.columns(2)
        for i, tip in enumerate(lifestyle_tips):
            with lt_cols[i % 2]:
                st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption(" This platform is for educational purposes only. It is NOT a medical diagnosis. Please consult a qualified healthcare professional.")


with tab2:
    st.markdown('<div class="section-title"> Select Your Symptoms</div>', unsafe_allow_html=True)
    st.markdown("Select **one or more symptoms** you are currently experiencing:")

    all_symptoms = get_all_symptoms()

    selected = st.multiselect(
        "Choose symptoms (you can select multiple):",
        options=all_symptoms,
        placeholder="Start typing or scroll to find symptoms..."
    )

    st.markdown('<div class="section-title">Or Browse by Disease Category</div>', unsafe_allow_html=True)

    browse_cols = st.columns(len(DISEASE_SYMPTOMS))
    for i, (disease, syms) in enumerate(DISEASE_SYMPTOMS.items()):
        icon = DISEASE_CONFIG[disease]["icon"]
        with browse_cols[i]:
            st.markdown(f"**{icon} {disease}**")
            for s in syms:
                chip_class = "symptom-chip matched" if s in selected else "symptom-chip"
                st.markdown(f'<span class="{chip_class}">{s}</span>', unsafe_allow_html=True)

    if selected:
        st.markdown('<div class="section-title"> Symptom Match Results</div>', unsafe_allow_html=True)
        st.markdown(f"You selected **{len(selected)} symptom(s)**. Here's which diseases they may indicate:")

        results = match_symptoms(selected)

        for disease, data in results.items():
            pct   = data["percentage"]
            count = data["matched"]
            total = data["total"]
            icon  = DISEASE_CONFIG[disease]["icon"]
            color = DISEASE_CONFIG[disease]["color"]
            level, css_class, _ = risk_label(pct)

            st.markdown(f"""
            <div class="disease-card {css_class}" style="border-top-color:{color}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-size:1rem; font-weight:700; color:#0f2942;">{icon} {disease}</span>
                    <span style="font-weight:700; color:{color};">{count}/{total} symptoms matched — {pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(pct))

            if data["symptoms"]:
                st.markdown("Matched symptoms: " + 
                    " ".join([f'<span class="symptom-chip matched">{s}</span>' for s in data["symptoms"]]),
                    unsafe_allow_html=True)

            st.markdown("")

        st.warning("Symptom matching is not a diagnosis. Multiple diseases can share symptoms. Always consult a doctor.")

    else:
        st.info(" Please select at least one symptom to see results.")