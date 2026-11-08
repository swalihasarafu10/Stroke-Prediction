import math

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="StrokeSense | Risk Predictor",
    page_icon="🫀",
    layout="wide",
)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "best_stroke_model.pkl"
PREP_PATH = BASE / "preprocessing.pkl"


# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

:root { color-scheme: dark; }
html, body, .stApp, .stMarkdown, label, p, span, div { font-family: 'Poppins', sans-serif; }

.stApp {
    background:
        radial-gradient(900px 500px at 8% -5%, rgba(99,102,241,.22), transparent 60%),
        radial-gradient(800px 500px at 100% 10%, rgba(34,211,238,.12), transparent 55%),
        linear-gradient(160deg, #080c1a 0%, #0d1328 55%, #130f2a 100%);
    color: #e2e8f0;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], footer, #MainMenu { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1180px; }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden; color: #fff;
    background: linear-gradient(120deg, #3730a3 0%, #6d28d9 48%, #0e7490 100%);
    border: 1px solid rgba(165,180,252,.25);
    border-radius: 26px; padding: 2.4rem 2.8rem; margin-bottom: 1.6rem;
    box-shadow: 0 24px 50px rgba(0,0,0,.45), 0 0 60px rgba(99,102,241,.25);
}
.hero::before, .hero::after {
    content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,.07);
}
.hero::before { width: 280px; height: 280px; right: -70px; top: -90px; }
.hero::after  { width: 160px; height: 160px; right: 170px; bottom: -80px; }
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 700; letter-spacing: -.5px; color: #fff; }
.hero p  { margin: .5rem 0 0; font-size: 1.02rem; opacity: .92; max-width: 640px; color: #e0e7ff; }
.hero .pill {
    display: inline-block; margin-top: 1rem; margin-right: .5rem; padding: .3rem .85rem;
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px; font-size: .78rem; font-weight: 500; color: #fff;
}

/* ---------- Cards (st.container with key="card_...") ---------- */
div[class*="st-key-card"] {
    background: rgba(19,26,50,.78);
    border: 1px solid rgba(129,140,248,.18);
    border-radius: 22px; padding: 1.3rem 1.5rem 1.1rem;
    box-shadow: 0 12px 32px rgba(0,0,0,.35);
    backdrop-filter: blur(8px);
}
.sec-title {
    display: flex; align-items: center; gap: .6rem; margin-bottom: .6rem;
    font-size: 1.05rem; font-weight: 600; color: #f1f5f9;
}
.sec-title .ico {
    width: 34px; height: 34px; border-radius: 11px; display: grid; place-items: center;
    background: rgba(99,102,241,.2); border: 1px solid rgba(129,140,248,.25); font-size: 1.05rem;
}

/* ---------- Readable text on dark ---------- */
[data-testid="stWidgetLabel"] p { color: #cbd5e1 !important; font-weight: 500; font-size: .9rem; }
[data-testid="stCaptionContainer"] p { color: #94a3b8 !important; }
[data-testid="stToggle"] p, [data-testid="stCheckbox"] p { color: #cbd5e1 !important; }
[data-testid="stSliderThumbValue"] { color: #c7d2fe !important; font-weight: 600; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: #94a3b8 !important; }

/* ---------- Select boxes ---------- */
div[data-baseweb="select"] > div {
    border-radius: 12px; background: #151d3b !important; border: 1px solid #2f3a66 !important;
}
div[data-baseweb="select"] * { color: #e2e8f0 !important; }
div[data-baseweb="select"] svg { fill: #94a3b8 !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [role="listbox"] {
    background: #151d3b !important;
}
div[data-baseweb="popover"] li { color: #e2e8f0 !important; background: transparent !important; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] {
    background: rgba(99,102,241,.25) !important;
}

/* ---------- Segmented choice (gender / residence) ---------- */
[data-testid="stButtonGroup"] button {
    border-radius: 12px !important; border: 1px solid #343f6e !important;
    background: #151d3b !important; padding: .45rem 1.1rem !important;
}
[data-testid="stButtonGroup"] button p { color: #cbd5e1 !important; font-weight: 500; }
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"],
[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    border-color: transparent !important; box-shadow: 0 6px 18px rgba(99,102,241,.45);
}
[data-testid="stButtonGroup"] button[aria-checked="true"] p,
[data-testid="stButtonGroup"] button[aria-pressed="true"] p,
[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] p {
    color: #ffffff !important; font-weight: 600;
}

/* ---------- Fallback if st.radio is used on older Streamlit ---------- */
div[role="radiogroup"] { gap: .5rem; }
div[role="radiogroup"] label {
    padding: .4rem 1rem; border-radius: 12px; border: 1px solid #343f6e; background: #151d3b;
}
div[role="radiogroup"] label > div:first-child { display: none; }
div[role="radiogroup"] label p { color: #cbd5e1 !important; font-weight: 500; }
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #4f46e5, #7c3aed); border-color: transparent;
}
div[role="radiogroup"] label:has(input:checked) p { color: #fff !important; font-weight: 600; }

/* ---------- Button ---------- */
.stFormSubmitButton button, div.stButton button {
    width: 100%; border: none; color: #fff; font-weight: 600; font-size: 1.05rem;
    padding: .85rem 1.6rem; border-radius: 16px;
    background: linear-gradient(90deg, #4f46e5, #7c3aed 55%, #0891b2);
    box-shadow: 0 12px 30px rgba(99,102,241,.45);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stFormSubmitButton button:hover, div.stButton button:hover {
    transform: translateY(-2px); box-shadow: 0 16px 38px rgba(99,102,241,.6); color: #fff;
}
.stFormSubmitButton button p, div.stButton button p { color: #fff !important; }
[data-testid="stForm"] { border: none; padding: 0; background: transparent; }

/* ---------- Results ---------- */
.gauge-wrap { text-align: center; }
.risk-badge {
    display: inline-block; padding: .4rem 1.2rem; border-radius: 999px;
    font-weight: 600; font-size: .95rem; margin-top: .2rem;
}
.risk-msg { color: #94a3b8; font-size: .92rem; margin-top: .8rem; line-height: 1.55; }

.vital {
    background: rgba(24,32,62,.85); border: 1px solid rgba(129,140,248,.16);
    border-radius: 18px; padding: 1rem 1.1rem;
    box-shadow: 0 8px 22px rgba(0,0,0,.3); height: 100%;
}
.vital .k { font-size: .76rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .6px; font-weight: 500; }
.vital .v { font-size: 1.55rem; font-weight: 700; color: #f1f5f9; margin: .15rem 0 .35rem; }
.vital .s { display: inline-block; font-size: .74rem; font-weight: 600; padding: .15rem .6rem; border-radius: 999px; }

.chip {
    display: inline-block; margin: .22rem .3rem .22rem 0; padding: .38rem .85rem;
    border-radius: 999px; font-size: .82rem; font-weight: 500;
}
.chip.bad  { background: rgba(239,68,68,.13);  color: #fca5a5; border: 1px solid rgba(248,113,113,.35); }
.chip.good { background: rgba(16,185,129,.13); color: #6ee7b7; border: 1px solid rgba(52,211,153,.35); }

.note {
    margin-top: 1.4rem; padding: 1rem 1.3rem; border-radius: 16px; font-size: .84rem;
    background: rgba(245,158,11,.10); border: 1px solid rgba(251,191,36,.3);
    color: #fcd34d; line-height: 1.55;
}
.foot { text-align: center; color: #8391ab; font-size: .8rem; margin-top: 2rem; }

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: rgba(19,26,50,.78); border: 1px solid rgba(129,140,248,.18); border-radius: 16px;
}
[data-testid="stExpander"] summary p { color: #e2e8f0 !important; font-weight: 500; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p { color: #cbd5e1; }
[data-testid="stAlert"] p { color: #e2e8f0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s: str) -> str:
    """Strip indentation so markdown never turns the HTML into a code block."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def section(icon: str, title: str):
    st.markdown(
        html(f'<div class="sec-title"><span class="ico">{icon}</span>{title}</div>'),
        unsafe_allow_html=True,
    )


def choice(label, options, default):
    """Highlighted single choice. Falls back to st.radio on older Streamlit."""
    if hasattr(st, "segmented_control"):
        value = st.segmented_control(
            label, options, default=default, selection_mode="single", key=f"choice_{label}"
        )
        return value if value is not None else default  # if user un-clicks, keep default
    return st.radio(label, options, index=options.index(default), horizontal=True)


# ------------------------------------------------------------------
# Load model + preprocessing
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(PREP_PATH)


if not (MODEL_PATH.exists() and PREP_PATH.exists()):
    st.error(
        "Could not find `best_stroke_model.pkl` and `preprocessing.pkl`. "
        "Put them in the same folder as `app.py`."
    )
    st.stop()

model, prep = load_artifacts()


def prepare(raw: dict) -> pd.DataFrame:
    """Apply the exact same steps used during training."""
    row = pd.DataFrame([raw])

    row["bmi"] = row["bmi"].fillna(prep["bmi_median"])
    row["cholesterol"] = row["cholesterol"].fillna(prep["chol_median"])

    for col, (lo, hi) in prep["caps"].items():
        row[col] = row[col].clip(lo, hi)

    row["high_glucose"] = (row["avg_glucose_level"] > 126).astype(int)

    row = pd.get_dummies(row, dtype=int)
    row = row.reindex(columns=prep["columns"], fill_value=0).astype(float)
    row[prep["num_cols"]] = prep["scaler"].transform(row[prep["num_cols"]])
    return row


# ------------------------------------------------------------------
# Helpers for the results panel
# ------------------------------------------------------------------
def risk_level(p: float):
    if p < 0.35:
        return "Low risk", "#34d399", "rgba(16,185,129,.16)", "#6ee7b7", (
            "The profile entered resembles patients with a lower likelihood of stroke. "
            "Keep up healthy habits and routine check-ups."
        )
    if p < 0.65:
        return "Moderate risk", "#fbbf24", "rgba(245,158,11,.16)", "#fcd34d", (
            "Several factors point towards an elevated likelihood. Consider discussing "
            "blood pressure, glucose and lifestyle with a doctor."
        )
    return "High risk", "#f87171", "rgba(239,68,68,.16)", "#fca5a5", (
        "The profile entered strongly resembles patients who had a stroke. "
        "A medical consultation is advised to review the risk factors below."
    )


def gauge_svg(p: float, color: str) -> str:
    """Semicircle gauge drawn as a plain arc to the exact end point (no dash/animation tricks)."""
    p = min(max(p, 0.0), 1.0)
    cx, cy, r = 100, 100, 80
    ang = p * math.pi                      # 0 = far left, pi = far right
    ex = cx - r * math.cos(ang)
    ey = cy - r * math.sin(ang)
    if p >= 0.999:                         # full gauge: end exactly on the right edge
        ex, ey = cx + r, cy
    fill_path = (
        f'<path d="M{cx - r} {cy} A{r} {r} 0 0 1 {ex:.2f} {ey:.2f}" fill="none" '
        f'stroke="{color}" stroke-width="16" stroke-linecap="round"/>'
        if p > 0.004 else ""
    )
    return html(f"""
    <svg viewBox="0 0 200 118" width="100%" style="max-width:340px">
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#263056"
            stroke-width="16" stroke-linecap="round"/>
      {fill_path}
      <text x="100" y="90" text-anchor="middle" font-size="30" font-weight="700"
            fill="#f1f5f9" font-family="Poppins, sans-serif">{p * 100:.0f}%</text>
      <text x="100" y="109" text-anchor="middle" font-size="8.5" fill="#94a3b8"
            font-family="Poppins, sans-serif">estimated stroke probability</text>
    </svg>
    """)


def vital_card(label, value, status, bg, fg):
    return html(f"""
    <div class="vital">
      <div class="k">{label}</div>
      <div class="v">{value}</div>
      <span class="s" style="background:{bg};color:{fg}">{status}</span>
    </div>
    """)


GOOD = ("rgba(16,185,129,.16)", "#6ee7b7")
WARN = ("rgba(245,158,11,.16)", "#fcd34d")
BAD = ("rgba(239,68,68,.16)", "#fca5a5")


def bp_status(s, d):
    if s >= 140 or d >= 90:
        return ("High",) + BAD
    if s >= 120 or d >= 80:
        return ("Elevated",) + WARN
    return ("Normal",) + GOOD


def glucose_status(g):
    if g > 126:
        return ("High",) + BAD
    if g >= 100:
        return ("Elevated",) + WARN
    return ("Normal",) + GOOD


def bmi_status(b):
    if b >= 30:
        return ("Obese",) + BAD
    if b >= 25:
        return ("Overweight",) + WARN
    if b >= 18.5:
        return ("Healthy",) + GOOD
    return ("Underweight",) + WARN


def chol_status(c):
    if c >= 240:
        return ("High",) + BAD
    if c >= 200:
        return ("Borderline",) + WARN
    return ("Desirable",) + GOOD


# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
st.markdown(
    html("""
    <div class="hero">
      <h1>🫀 StrokeSense</h1>
      <p>Enter a patient profile and get an instant, model-based estimate of stroke
         likelihood, along with the key risk factors behind it.</p>
      <span class="pill">Machine Learning</span>
      <span class="pill">17 clinical inputs</span>
      <span class="pill">Instant results</span>
    </div>
    """),
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("patient_form"):
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        with st.container(key="card_personal"):
            section("👤", "Personal details")
            age = st.slider("Age", 18, 95, 55)
            gender = choice("Gender", ["Female", "Male"], "Female")
            work_type = st.selectbox(
                "Work type",
                ["Private", "Self-employed", "Govt_job", "Retired", "Student", "Unemployed"],
                format_func=lambda x: x.replace("_", " "),
            )
            residence_type = choice("Residence", ["Urban", "Rural"], "Urban")

        with st.container(key="card_lifestyle"):
            section("🏃", "Lifestyle")
            smoking_status = st.selectbox("Smoking", ["Never", "Former", "Current"])
            alcohol_consumption = st.selectbox("Alcohol", ["Non-drinker", "Moderate", "Heavy"])
            physical_activity = st.selectbox("Physical activity", ["High", "Moderate", "Low"])

    with c2:
        with st.container(key="card_vitals"):
            section("🩺", "Vitals and lab values")
            systolic_bp = st.slider("Systolic BP (mmHg)", 90, 200, 120)
            diastolic_bp = st.slider("Diastolic BP (mmHg)", 55, 125, 80)
            avg_glucose_level = st.slider("Avg glucose (mg/dL)", 60, 320, 100)
            bmi = st.slider("BMI", 15.0, 50.0, 25.0, step=0.1)
            cholesterol = st.slider("Cholesterol (mg/dL)", 110, 340, 190)

    with c3:
        with st.container(key="card_history"):
            section("📋", "Medical history")
            hypertension = st.toggle("Diagnosed hypertension")
            diabetes = st.toggle("Diabetes")
            heart_disease = st.toggle("Heart disease")
            atrial_fibrillation = st.toggle("Atrial fibrillation")
            family_history_stroke = st.toggle("Family history of stroke")
            st.caption("Switch on everything that applies to the patient.")

    submitted = st.form_submit_button("Analyze stroke risk")

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if submitted:
    raw = {
        "age": age,
        "gender": gender,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "hypertension": int(hypertension),
        "avg_glucose_level": float(avg_glucose_level),
        "diabetes": int(diabetes),
        "bmi": float(bmi),
        "cholesterol": float(cholesterol),
        "heart_disease": int(heart_disease),
        "atrial_fibrillation": int(atrial_fibrillation),
        "smoking_status": smoking_status,
        "alcohol_consumption": alcohol_consumption,
        "physical_activity": physical_activity,
        "family_history_stroke": int(family_history_stroke),
        "work_type": work_type,
        "residence_type": residence_type,
    }

    proba = float(model.predict_proba(prepare(raw))[0, 1])
    level, color, badge_bg, badge_fg, message = risk_level(proba)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        with st.container(key="card_result"):
            section("📊", "Risk assessment")
            st.markdown(
                html(f"""
                <div class="gauge-wrap">
                  {gauge_svg(proba, color)}
                  <div><span class="risk-badge" style="background:{badge_bg};color:{badge_fg}">{level}</span></div>
                  <div class="risk-msg">{message}</div>
                </div>
                """),
                unsafe_allow_html=True,
            )

    with right:
        # Vital summary cards
        bp = bp_status(systolic_bp, diastolic_bp)
        gl = glucose_status(avg_glucose_level)
        bm = bmi_status(bmi)
        ch = chol_status(cholesterol)

        v1, v2, v3, v4 = st.columns(4, gap="small")
        v1.markdown(vital_card("Blood pressure", f"{systolic_bp}/{diastolic_bp}", *bp), unsafe_allow_html=True)
        v2.markdown(vital_card("Glucose", f"{avg_glucose_level}", *gl), unsafe_allow_html=True)
        v3.markdown(vital_card("BMI", f"{bmi:.1f}", *bm), unsafe_allow_html=True)
        v4.markdown(vital_card("Cholesterol", f"{cholesterol}", *ch), unsafe_allow_html=True)

        # Risk factors and healthy signs
        risks, goods = [], []

        def add(cond_bad, bad_text, cond_good=None, good_text=None):
            if cond_bad:
                risks.append(bad_text)
            elif cond_good and good_text:
                goods.append(good_text)

        add(age >= 60, "Age 60 or above")
        add(systolic_bp >= 140 or diastolic_bp >= 90, "High blood pressure",
            systolic_bp < 120 and diastolic_bp < 80, "Healthy blood pressure")
        add(avg_glucose_level > 126, "High glucose",
            avg_glucose_level < 100, "Normal glucose")
        add(bmi >= 30, "Obesity (BMI 30+)", 18.5 <= bmi < 25, "Healthy weight")
        add(cholesterol >= 240, "High cholesterol", cholesterol < 200, "Desirable cholesterol")
        add(smoking_status == "Current", "Current smoker", smoking_status == "Never", "Never smoked")
        add(alcohol_consumption == "Heavy", "Heavy alcohol use",
            alcohol_consumption == "Non-drinker", "No alcohol use")
        add(physical_activity == "Low", "Low physical activity",
            physical_activity in ("High", "Moderate"), "Physically active")
        add(hypertension, "Diagnosed hypertension")
        add(diabetes, "Diabetes")
        add(heart_disease, "Heart disease")
        add(atrial_fibrillation, "Atrial fibrillation")
        add(family_history_stroke, "Family history of stroke")
        if not (hypertension or diabetes or heart_disease or atrial_fibrillation):
            goods.append("No known cardiovascular conditions")

        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        with st.container(key="card_factors"):
            section("⚠️", "Risk factors detected")
            if risks:
                st.markdown(
                    "".join(f'<span class="chip bad">{r}</span>' for r in risks),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("No major risk factors detected.")

            section("✅", "Protective signs")
            if goods:
                st.markdown(
                    "".join(f'<span class="chip good">{g}</span>' for g in goods),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("No protective signs identified.")

    st.markdown(
        html("""
        <div class="note">
          <b>Please note:</b> this tool is an educational project trained on
          <b>synthetic data</b>. It is not a medical device and must not be used for
          diagnosis or treatment decisions. Always consult a qualified healthcare professional.
        </div>
        """),
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# About
# ------------------------------------------------------------------
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
with st.expander("About the model"):
    st.write(f"**Algorithm:** {type(model).__name__}")
    st.write(f"**Input features after encoding:** {len(prep['columns'])}")
    st.write(
        "The inputs are cleaned, capped at training percentiles, one-hot encoded and "
        "scaled with the same steps used during training, then passed to the saved model."
    )

st.markdown('<div class="foot">Built with Streamlit · scikit-learn</div>', unsafe_allow_html=True)