import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Intelligent Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  /* ── reset & base ── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

  /* ── palette ── */
  :root {
    --bg:       #0D0F14;
    --surface:  #13161E;
    --border:   #1F2535;
    --accent:   #4F8EF7;
    --accent2:  #7C5CFC;
    --success:  #22C55E;
    --danger:   #EF4444;
    --warn:     #F59E0B;
    --text:     #E2E8F0;
    --muted:    #64748B;
  }

  /* force dark bg */
  .stApp { background: var(--bg); color: var(--text); }

  /* ── hero header ── */
  .hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 60%, #0D0F14 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(79,142,247,.18) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #4F8EF7, #7C5CFC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 .4rem;
  }
  .hero-sub { color: var(--muted); font-size: .95rem; margin: 0; }

  /* ── section headers ── */
  .section-label {
    font-size: .72rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: var(--accent);
    margin: 2rem 0 .75rem;
    display: flex; align-items: center; gap: .5rem;
  }
  .section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
  }

  /* ── card ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }
  .card-sm {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
  }
  .card-sm .cs-label { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing:.08em; }
  .card-sm .cs-value { font-size: 1.35rem; font-weight: 700; color: var(--text); margin-top: .25rem; }
  .card-sm .cs-sub   { font-size: .8rem;  color: var(--muted); margin-top: .15rem; }

  /* ── result boxes ── */
  .result-churn {
    background: linear-gradient(135deg, rgba(239,68,68,.12), rgba(239,68,68,.04));
    border: 1px solid rgba(239,68,68,.45);
    border-radius: 16px; padding: 2rem; text-align: center;
  }
  .result-safe {
    background: linear-gradient(135deg, rgba(34,197,94,.12), rgba(34,197,94,.04));
    border: 1px solid rgba(34,197,94,.45);
    border-radius: 16px; padding: 2rem; text-align: center;
  }
  .result-title { font-size: 1.8rem; font-weight: 700; margin-bottom: .35rem; }
  .result-sub   { font-size: .95rem; color: var(--muted); }

  /* ── risk alert boxes ── */
  .alert-high {
    background: rgba(239,68,68,.08);
    border-left: 4px solid var(--danger);
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem; margin-top: 1.5rem;
  }
  .alert-low {
    background: rgba(34,197,94,.08);
    border-left: 4px solid var(--success);
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem; margin-top: 1.5rem;
  }
  .alert-title { font-weight: 600; font-size: 1rem; margin-bottom: .4rem; }
  .alert-body  { font-size: .88rem; color: var(--muted); line-height: 1.6; }

  /* ── risk meter ── */
  .meter-wrap { padding: 1.5rem 0 .5rem; }
  .meter-bar-bg {
    height: 10px; border-radius: 999px;
    background: linear-gradient(90deg, #22C55E 0%, #F59E0B 50%, #EF4444 100%);
    position: relative; margin: .6rem 0;
  }
  .meter-needle {
    position: absolute; top: 50%; transform: translate(-50%, -50%);
    width: 18px; height: 18px; border-radius: 50%;
    background: #fff; border: 3px solid #0D0F14;
    box-shadow: 0 0 0 2px var(--accent);
  }
  .meter-labels {
    display: flex; justify-content: space-between;
    font-size: .72rem; color: var(--muted); margin-top: .3rem;
  }

  /* ── probability number ── */
  .prob-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.5rem; font-weight: 600; line-height: 1;
  }
  .prob-label { font-size: .8rem; color: var(--muted); text-transform: uppercase; letter-spacing:.1em; margin-top:.3rem; }

  /* ── sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  .sb-logo { font-size: 1.4rem; font-weight: 700; color: var(--text); margin-bottom: .15rem; }
  .sb-tag  { font-size: .72rem; color: var(--accent); text-transform: uppercase; letter-spacing:.1em; }
  .sb-divider { border: none; border-top: 1px solid var(--border); margin: 1.25rem 0; }
  .sb-section  { font-size: .72rem; font-weight: 600; color: var(--muted);
                 text-transform: uppercase; letter-spacing:.1em; margin-bottom:.6rem; }
  .sb-body { font-size: .85rem; color: var(--text); line-height: 1.65; }
  .sb-chip {
    display: inline-block; background: rgba(79,142,247,.12);
    color: var(--accent); border-radius: 999px;
    padding: .2rem .7rem; font-size: .75rem; font-weight: 500; margin: .2rem .1rem;
  }
  .sb-dev-card {
    background: rgba(79,142,247,.06); border: 1px solid var(--border);
    border-radius: 10px; padding: .9rem 1rem; margin-top: .5rem;
  }
  .sb-dev-name { font-weight: 600; font-size: .9rem; }
  .sb-dev-role { font-size: .78rem; color: var(--muted); }

  /* ── footer ── */
  .footer {
    text-align: center; padding: 2rem 0 .5rem;
    font-size: .8rem; color: var(--muted); border-top: 1px solid var(--border);
    margin-top: 3rem;
  }
  .footer span { color: var(--accent); font-weight: 600; }

  /* ── Streamlit widget tweaks ── */
  div[data-baseweb="select"] > div,
  div[data-baseweb="input"]  > div {
    background: #1A1E2A !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
  }
  label { color: var(--muted) !important; font-size: .82rem !important; }
  .stButton > button {
    background: linear-gradient(135deg, #4F8EF7, #7C5CFC);
    color: #fff; border: none; border-radius: 10px;
    padding: .75rem 2.5rem; font-size: 1rem; font-weight: 600;
    letter-spacing: .01em; cursor: pointer; width: 100%;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity: .88; }
  div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #4F8EF7, #7C5CFC) !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.sav", "rb") as f:
        return pickle.load(f)


model = load_model()

FEATURE_ORDER = [
    "SeniorCitizen", "MonthlyCharges", "TotalCharges",
    "gender_Female", "gender_Male",
    "Partner_No", "Partner_Yes",
    "Dependents_No", "Dependents_Yes",
    "PhoneService_No", "PhoneService_Yes",
    "MultipleLines_No", "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No", "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No", "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No", "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No", "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No", "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaperlessBilling_No", "PaperlessBilling_Yes",
    "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "tenure_group_1 - 12", "tenure_group_13 - 24", "tenure_group_25 - 36",
    "tenure_group_37 - 48", "tenure_group_49 - 60", "tenure_group_61 - 72",
]


def build_features(raw: dict) -> pd.DataFrame:
    row = {f: 0 for f in FEATURE_ORDER}
    row["SeniorCitizen"]   = raw["SeniorCitizen"]
    row["MonthlyCharges"]  = raw["MonthlyCharges"]
    row["TotalCharges"]    = raw["TotalCharges"]

    for col in [f"gender_{raw['gender']}",
                f"Partner_{raw['Partner']}",
                f"Dependents_{raw['Dependents']}",
                f"PhoneService_{raw['PhoneService']}",
                f"MultipleLines_{raw['MultipleLines']}",
                f"InternetService_{raw['InternetService']}",
                f"OnlineSecurity_{raw['OnlineSecurity']}",
                f"OnlineBackup_{raw['OnlineBackup']}",
                f"DeviceProtection_{raw['DeviceProtection']}",
                f"TechSupport_{raw['TechSupport']}",
                f"StreamingTV_{raw['StreamingTV']}",
                f"StreamingMovies_{raw['StreamingMovies']}",
                f"Contract_{raw['Contract']}",
                f"PaperlessBilling_{raw['PaperlessBilling']}",
                f"PaymentMethod_{raw['PaymentMethod']}",
                f"tenure_group_{raw['tenure_group']}"]:
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[FEATURE_ORDER]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">📡 ChurnGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">Telecom Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-body">
      ChurnGuard uses machine learning to predict whether a telecom customer
      is likely to cancel their subscription — enabling proactive retention
      strategies before it's too late.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Model Details</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:.5rem;">
      <span class="sb-chip">Scikit-Learn</span>
      <span class="sb-chip">50 Features</span>
      <span class="sb-chip">One-Hot Encoded</span>
    </div>
    <div class="sb-body">
      • Binary classification (Churn / No Churn)<br>
      • <code>predict_proba()</code> confidence scoring<br>
      • Trained on Telco Customer dataset<br>
      • Features: demographics, services, billing
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Developer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-dev-card">
      <div class="sb-dev-name">Gaurav </div>
      <div class="sb-dev-role">ML Engineer · Data Scientist</div>
      <div style="margin-top:.5rem;font-size:.78rem;color:#64748B;">
        📧 gaurav200b@gmail.com<br>
        🔗 github.com/Gaurav215b
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-body" style="font-size:.75rem;color:#475569;">
      Model file: <code>model.sav</code><br>
      Version: 1.0.0 &nbsp;|&nbsp; Framework: Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">Customer Churn Prediction</p>
  <p class="hero-sub">
    Fill in the customer profile below and click <strong>Analyze Customer</strong>
    to get an instant churn risk assessment powered by machine learning.
  </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════════

with st.form("prediction_form"):

    # ── 1. Customer Information ──────────────────────────────────────────────
    st.markdown('<div class="section-label">👤 Customer Information</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender       = st.selectbox("Gender", ["Male", "Female"])
            senior       = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c2:
            partner      = st.selectbox("Has Partner", ["Yes", "No"])
            dependents   = st.selectbox("Has Dependents", ["No", "Yes"])
        with c3:
            tenure_group = st.selectbox(
                "Tenure Group",
                ["1 - 12", "13 - 24", "25 - 36", "37 - 48", "49 - 60", "61 - 72"],
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 2. Services ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📞 Services</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        with c2:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )
        with c3:
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 3. Internet Services ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">🌐 Internet Add-ons</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        _inet_opts = ["Yes", "No", "No internet service"]
        with c1:
            online_security   = st.selectbox("Online Security",   _inet_opts)
            online_backup     = st.selectbox("Online Backup",     _inet_opts)
        with c2:
            device_protection = st.selectbox("Device Protection", _inet_opts)
            tech_support      = st.selectbox("Tech Support",      _inet_opts)
        with c3:
            st.markdown("""
            <div style="padding:.5rem 0;font-size:.82rem;color:#64748B;line-height:1.7;">
              ℹ️ Select <em>No internet service</em> if the customer has no
              internet subscription to ensure accurate one-hot encoding.
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 4. Entertainment Services ─────────────────────────────────────────────
    st.markdown('<div class="section-label">🎬 Entertainment Services</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            streaming_tv     = st.selectbox("Streaming TV",     _inet_opts)
        with c2:
            streaming_movies = st.selectbox("Streaming Movies", _inet_opts)
        with c3:
            st.markdown(" ")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 5. Billing Information ────────────────────────────────────────────────
    st.markdown('<div class="section-label">💳 Billing Information</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            contract    = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless   = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c2:
            payment     = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
            )
        with c3:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0,
                                              max_value=500.0, value=65.0, step=0.5)
            total_charges   = st.number_input("Total Charges ($)",   min_value=0.0,
                                              max_value=10000.0, value=1200.0, step=10.0)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Analyze Customer")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

if submitted:
    raw = dict(
        SeniorCitizen   = 1 if senior == "Yes" else 0,
        MonthlyCharges  = monthly_charges,
        TotalCharges    = total_charges,
        gender          = gender,
        Partner         = partner,
        Dependents      = dependents,
        PhoneService    = phone_service,
        MultipleLines   = multiple_lines,
        InternetService = internet_service,
        OnlineSecurity  = online_security,
        OnlineBackup    = online_backup,
        DeviceProtection= device_protection,
        TechSupport     = tech_support,
        StreamingTV     = streaming_tv,
        StreamingMovies = streaming_movies,
        Contract        = contract,
        PaperlessBilling= paperless,
        PaymentMethod   = payment,
        tenure_group    = tenure_group,
    )

    X        = build_features(raw)
    pred     = model.predict(X)[0]
    proba    = model.predict_proba(X)[0]
    churn_p  = round(proba[1] * 100, 1)
    safe_p   = round(proba[0] * 100, 1)
    churned  = (pred == 1)

    st.markdown("---")
    st.markdown('<div class="section-label">📊 Prediction Results</div>', unsafe_allow_html=True)

    # ── Top metrics row ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="card-sm">
          <div class="cs-label">💰 Monthly Charges</div>
          <div class="cs-value">${monthly_charges:.2f}</div>
          <div class="cs-sub">per month</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="card-sm">
          <div class="cs-label">🧾 Total Charges</div>
          <div class="cs-value">${total_charges:,.2f}</div>
          <div class="cs-sub">lifetime</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="card-sm">
          <div class="cs-label">📄 Contract</div>
          <div class="cs-value" style="font-size:1rem">{contract}</div>
          <div class="cs-sub">billing cycle</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="card-sm">
          <div class="cs-label">⏱ Tenure Group</div>
          <div class="cs-value" style="font-size:1.1rem">{tenure_group}</div>
          <div class="cs-sub">months</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main result + risk meter ─────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        if churned:
            st.markdown(f"""
            <div class="result-churn">
              <div class="result-title" style="color:#EF4444;">⚠️ Likely to Churn</div>
              <div class="result-sub">This customer shows high churn indicators.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
              <div class="result-title" style="color:#22C55E;">✅ Likely to Stay</div>
              <div class="result-sub">This customer shows low churn risk.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bar
        bar_col = "#EF4444" if churn_p > 70 else ("#F59E0B" if churn_p > 30 else "#22C55E")
        needle_left = f"{churn_p}%"

        st.markdown(f"""
        <div class="meter-wrap">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;">
            <span style="font-size:.78rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em;">
              Churn Risk Meter
            </span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                         font-weight:600;color:{bar_col};">{churn_p}%</span>
          </div>
          <div class="meter-bar-bg">
            <div class="meter-needle" style="left:{needle_left};"></div>
          </div>
          <div class="meter-labels">
            <span>Low Risk</span><span>Medium</span><span>High Risk</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(churn_p))

    with right:
        churn_color = "#EF4444" if churned else "#22C55E"
        safe_color  = "#22C55E" if not churned else "#EF4444"

        st.markdown(f"""
        <div class="card" style="text-align:center;padding:2rem;">
          <div style="font-size:.75rem;color:#64748B;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:.5rem;">Churn Probability</div>
          <div class="prob-number" style="color:{churn_color};">{churn_p}%</div>
          <div class="prob-label">confidence score</div>
          <div style="margin-top:1.5rem;display:flex;gap:1rem;justify-content:center;">
            <div style="flex:1;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);
                        border-radius:10px;padding:.75rem;">
              <div style="font-size:.7rem;color:#64748B;margin-bottom:.2rem;">Churn</div>
              <div style="font-family:'JetBrains Mono',monospace;font-weight:600;
                          color:#EF4444;">{churn_p}%</div>
            </div>
            <div style="flex:1;background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);
                        border-radius:10px;padding:.75rem;">
              <div style="font-size:.7rem;color:#64748B;margin-bottom:.2rem;">Retain</div>
              <div style="font-family:'JetBrains Mono',monospace;font-weight:600;
                          color:#22C55E;">{safe_p}%</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Risk alert sections ──────────────────────────────────────────────────
    if churn_p > 70:
        st.markdown(f"""
        <div class="alert-high">
          <div class="alert-title" style="color:#EF4444;">🚨 High-Risk Customer — Immediate Attention Required</div>
          <div class="alert-body">
            This customer has a <strong>{churn_p}%</strong> probability of churning.
            Consider offering a targeted retention package: discounted plan upgrade,
            loyalty rewards, or a dedicated account manager outreach within 48 hours.
            Month-to-month contracts with electronic check payments historically show
            the highest churn correlation — review their contract type and payment friction.
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif churn_p < 30:
        st.markdown(f"""
        <div class="alert-low">
          <div class="alert-title" style="color:#22C55E;">🎉 Low-Risk Customer — Engagement Opportunity</div>
          <div class="alert-body">
            This customer has only a <strong>{churn_p}%</strong> churn probability —
            excellent retention profile. Consider cross-selling premium add-ons,
            offering a loyalty reward, or requesting a referral. Customers in this
            segment are strong candidates for upsell campaigns.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built with <span>Streamlit</span> and <span>Scikit-Learn</span> &nbsp;·&nbsp;
  ChurnGuard Telecom Intelligence &nbsp;·&nbsp; v1.0.0
</div>
""", unsafe_allow_html=True)