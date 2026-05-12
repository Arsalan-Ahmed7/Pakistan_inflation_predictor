import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="🇵🇰 Pakistan Savings Predictor",
    page_icon="🇵🇰", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0f0; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    [data-testid="stSidebar"] { background-color: #1a1d2e; border-right: 1px solid #2a2d3e; }
    [data-testid="stSidebar"] label,[data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color:#e0e0f0 !important; }
    [data-testid="stMetric"] { background:#1a1d2e; border:1px solid #2a2d3e;
        border-radius:10px; padding:12px 16px; }
    [data-testid="stMetricLabel"] { color:#a0a8d0 !important; font-size:11px !important; letter-spacing:1px; }
    [data-testid="stMetricValue"] { color:#6ea8fe !important; font-family:monospace; }
    .stTabs [data-baseweb="tab-list"] { background:#1a1d2e; border-radius:10px; gap:4px; }
    .stTabs [data-baseweb="tab"] { color:#a0a8d0; font-family:monospace; font-size:13px; }
    .stTabs [aria-selected="true"] { color:#e0e0f0 !important; background:#0f1117 !important; border-radius:8px; }
    h1,h2,h3 { color:#e0e0f0 !important; font-family:monospace; }
    .badge { display:inline-block; background:#1a1d2e; border:1px solid #3d4166;
        border-radius:20px; padding:2px 10px; font-size:10px; color:#6070a0;
        margin:2px; font-family:monospace; }
    .vcard { border-radius:10px; padding:16px; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VERIFIED DATA — Sources cited per row
# ══════════════════════════════════════════════════════════════════════════════

# Monthly 2026 data — PBS official press releases
MONTHLY_2026 = {
    "Jan": 5.80, "Feb": 7.00, "Mar": 7.30, "Apr": 10.89
}
LATEST_CPI        = 10.89   # Apr 2026 — PBS press release May 1 2026
LATEST_RATE       = 11.50   # Apr 27 2026 — SBP MPC (Dawn, Business Recorder, Geo)
LATEST_REAL_RATE  = round(LATEST_RATE - LATEST_CPI, 2)   # +0.61%
LATEST_CORE       = 8.25    # Apr 2026 urban 8.0%, rural 8.5% → avg
YTD_2026_AVG      = 7.74    # Jan–Apr 2026, finhisaab/PBS
GDP_H1_FY26       = 3.8     # H1-FY26 YoY — SBP Apr 27 MPC statement
FX_RESERVES_B     = 15.8    # Apr 24 2026 — SBP MPC statement

@st.cache_data
def load_data():
    """
    Annual data 2000-2025.
    CPI  : PBS annual averages — finhisaab.com/PBS; World Bank FP.CPI.TOTL.ZG
    Rate : SBP end-of-year — CEIC, FocusEconomics, SBP MPC statements
    GDP  : World Bank NY.GDP.MKTP.KD.ZG / IMF WEO
    PKR  : SBP / CEIC annual averages
    """
    df = pd.DataFrame({
        "year": list(range(2000, 2026)),

        # PBS annual average CPI inflation (%)
        "inflation": [
            4.37, 3.15, 3.29, 2.50, 6.97, 9.12, 7.92, 7.58,   # 2000-07
            19.78,12.13,12.90,12.42,10.26, 7.28, 7.19, 2.55,   # 2008-15
            3.75,  4.25, 5.28, 9.40, 9.47, 9.50,19.87,30.77,   # 2016-23
            12.65, 3.53,                                          # 2024-25
        ],

        # SBP policy rate end-of-year (%)
        # Key events:
        #  Jun 2023: hiked to 22% (all-time high) — Business Recorder
        #  Jun 2024: first cut → 20.5% (start of easing cycle)
        #  Dec 2024: ended at 13.0% — FocusEconomics confirmed
        #  May 2025: cut to 11.0%; Dec 16 2025: cut to 10.5% — SBP MPC Dec 2025
        "policy_rate": [
            13.00,13.00, 9.50, 7.50, 8.00, 9.00, 9.50,10.00,   # 2000-07
            15.00,14.00,13.00,12.00,10.50, 9.50,10.00, 6.00,   # 2008-15
             5.75, 6.00, 8.50,13.25, 7.00, 7.00,16.00,22.00,   # 2016-23
            13.00,10.50,                                          # 2024-25
        ],

        # World Bank / IMF GDP growth (%)
        "gdp_growth": [
            4.3, 2.0, 3.1, 4.8, 7.5, 9.0, 5.8, 4.8,
            1.7, 4.1, 2.6, 3.6, 4.4, 3.7, 4.1, 5.1,
            4.6, 5.2, 5.5, 3.3,-0.5, 5.7, 6.0,-0.2,
            2.4, 2.7,
        ],

        # SBP / CEIC annual average PKR/USD
        "exchange_rate": [
             53.65, 61.50, 59.72, 57.75, 57.57, 59.72, 60.04, 60.63,
             70.40, 81.71, 85.19, 86.34, 93.40,101.63,102.86,102.11,
            104.77,105.45,121.82,158.44,160.88,177.00,204.54,284.00,
            278.50,281.27,
        ],
    })
    df["real_rate"]           = df["policy_rate"] - df["inflation"]
    df["inflation_lag1"]      = df["inflation"].shift(1)
    df["inflation_lag2"]      = df["inflation"].shift(2)
    df["exchange_rate_chg"]   = df["exchange_rate"].pct_change() * 100
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def train_model(_df):
    features = ["inflation_lag1","inflation_lag2","policy_rate",
                "gdp_growth","exchange_rate_chg"]
    train = _df[_df.year <= 2022]
    test  = _df[_df.year >  2022]
    sc    = StandardScaler()
    m     = LinearRegression()
    m.fit(sc.fit_transform(train[features]), train["inflation"])
    r2  = r2_score(train["inflation"], m.predict(sc.transform(train[features])))
    mae = mean_absolute_error(test["inflation"], m.predict(sc.transform(test[features])))
    val = test[["year","inflation"]].copy()
    val["predicted"] = m.predict(sc.transform(test[features])).round(2)
    val["error_pp"]  = (val["predicted"] - val["inflation"]).round(2)
    return m, sc, features, round(r2,3), round(mae,2), val

def get_forecast(model, sc, features, df, scenario):
    """
    Recursive forecast 2026-2031.
    Anchors 2026 = blend of model output (50%) + YTD Jan-Apr PBS avg 7.74% (50%).
    Policy rate anchored at 11.5% (SBP Apr 27 2026 hike); gradual easing assumed.
    SBP MPC Apr 2026 stated inflation "likely above target range for most FY27".
    """
    mult = {"🟢 Optimistic": 0.82, "🟡 Base (SBP)": 1.00, "🔴 Pessimistic": 1.20}[scenario]
    macro = {
        #        policy   gdp   fx_chg
        2026: (11.50,  3.8,   2.5),
        2027: (10.50,  4.5,   2.0),
        2028: ( 9.50,  5.0,   1.5),
        2029: ( 9.00,  5.0,   1.5),
        2030: ( 8.50,  5.2,   1.0),
        2031: ( 8.00,  5.5,   1.0),
    }
    # Verified baselines: 2024=12.65%, 2025=3.53% (PBS annual)
    prev2 = df[df.year==2024]["inflation"].values[0]   # 12.65
    prev1 = df[df.year==2025]["inflation"].values[0]   # 3.53
    rows  = []
    for yr, (pr, gdp, fx) in macro.items():
        row  = [prev1, prev2, pr, gdp, fx]
        pred = float(model.predict(sc.transform([row]))[0])
        if yr == 2026:
            # Blend model with known Jan-Apr 2026 PBS avg (7.74%)
            pred = pred * 0.50 + YTD_2026_AVG * 0.50
        pred = round(max(3.0, min(pred * mult, 32.0)), 2)
        rows.append({"year": yr, "inflation": pred,
                     "policy_rate": pr, "real_rate": round(pr - pred, 2)})
        prev2, prev1 = prev1, pred
    return pd.DataFrame(rows)

def savings_trajectory(principal, rate, fcast):
    rows = [{"year": 2026, "nominal": float(principal), "real": float(principal),
             "infl": fcast.iloc[0]["inflation"]}]
    nom, real = float(principal), float(principal)
    for _, r in fcast.iloc[1:].iterrows():
        nom  = nom  * (1 + rate/100)
        real = real * ((1 + rate/100) / (1 + r["inflation"]/100))
        rows.append({"year": int(r["year"]), "nominal": round(nom), "real": round(real),
                     "infl": r["inflation"]})
    return pd.DataFrame(rows)

def verdict(rr):
    if rr > 3:   return "💰 STRONGLY SAVE",  "#39d98a", "#0d2e1a"
    if rr > 0:   return "✅ SAVE",            "#6fcf97", "#0a2018"
    if rr > -3:  return "⚖️ NEUTRAL",         "#f2c94c", "#2e2500"
    return              "⚠️ SPEND / INVEST",   "#ff5c7a", "#2d0a12"

def pkr(v):
    v = float(v)
    if v >= 1e7: return f"₨{v/1e7:.2f}Cr"
    if v >= 1e5: return f"₨{v/1e5:.2f}L"
    if v >= 1e3: return f"₨{v/1e3:.0f}K"
    return f"₨{v:.0f}"

BG, SBG, GR, TC = "#0f1117", "#1a1d2e", "#2a2d3e", "#a0a8d0"

def fig0(title=""):
    f = go.Figure()
    f.update_layout(
        title=dict(text=title, font=dict(color="#e0e0f0", size=13, family="monospace")),
        plot_bgcolor=SBG, paper_bgcolor=BG,
        font=dict(color=TC, family="monospace", size=11),
        legend=dict(bgcolor=SBG, bordercolor=GR, borderwidth=1, font=dict(size=10)),
        margin=dict(l=50,r=20,t=45,b=40), hovermode="x unified",
        xaxis=dict(gridcolor=GR, tickcolor="#6070a0", linecolor=GR),
        yaxis=dict(gridcolor=GR, tickcolor="#6070a0", linecolor=GR),
    )
    return f

# ─── load ────────────────────────────────────────────────────────────────────
df = load_data()
model, sc, features, tr2, tmae, val_df = train_model(df)

# ─── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    principal = st.number_input("💵 Principal (PKR)", 10_000, 500_000_000,
                                1_000_000, 50_000, format="%d")
    st.caption(f"**{pkr(principal)}**")
    dep_rate  = st.slider("🏦 Deposit Rate (%)", 5.0, 22.0, 11.5, 0.25)

    st.markdown("**Quick Presets**")
    c1,c2 = st.columns(2)
    if c1.button("Savings\n8%",   use_container_width=True): dep_rate = 8.0
    if c2.button("Fixed Dep\n11.5%", use_container_width=True): dep_rate = 11.5
    if c1.button("NSS\n15%",     use_container_width=True): dep_rate = 15.0
    if c2.button("Sukuk\n17%",   use_container_width=True): dep_rate = 17.0

    st.markdown("---")
    scenario = st.selectbox("📊 Scenario", ["🟢 Optimistic","🟡 Base (SBP)","🔴 Pessimistic"], 1)

    st.markdown("---")
    st.markdown("**📋 Model**")
    st.markdown(f"- R²: `{tr2}` (2002–2022)")
    st.markdown(f"- MAE: `{tmae}%` (2023–2025)")
    st.markdown("---")
    st.markdown("**🔗 Sources**")
    for s in ["PBS (CPI)","SBP (Policy Rate)","World Bank (GDP)","CEIC (PKR/USD)"]:
        st.markdown(f"<span class='badge'>{s}</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Dawn · Business Recorder · Geo</span>", unsafe_allow_html=True)

# ─── compute ─────────────────────────────────────────────────────────────────
fcast = get_forecast(model, sc, features, df, scenario)
sdf   = savings_trajectory(principal, dep_rate, fcast)
final_real = sdf.iloc[-1]["real"]
gain       = final_real - principal
lbl_now, col_now, _ = verdict(LATEST_REAL_RATE)

# ─── header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#111424,#1a1d2e,#0d1a2e);
            border:1px solid #2a2d3e;border-radius:12px;padding:18px 28px;margin-bottom:16px;'>
  <div style='display:flex;align-items:center;gap:14px;'>
    <span style='font-size:38px;'>🇵🇰</span>
    <div>
      <div style='font-size:20px;font-weight:bold;letter-spacing:3px;color:#6ea8fe;font-family:monospace;'>
        PAKISTAN SAVINGS PREDICTOR
      </div>
      <div style='font-size:10px;color:#6070a0;letter-spacing:1.5px;font-family:monospace;margin-top:4px;'>
        RATIONAL EXPECTATIONS MODEL &nbsp;·&nbsp;
        PBS / SBP / WORLD BANK DATA &nbsp;·&nbsp;
        LAST UPDATED: MAY 2026
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── live KPI row ─────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("📍 CPI Apr 2026 (PBS)",  f"{LATEST_CPI}%",   "↑ from 7.3% Mar")
k2.metric("🏦 SBP Rate (Live)",    f"{LATEST_RATE}%",   "+100bps Apr 27")
k3.metric("📊 Real Rate (Live)",   f"{LATEST_REAL_RATE:+.2f}%", lbl_now.split()[1])
k4.metric("📅 2025 CPI (PBS avg)", "3.53%",            "↓ 9.1pp vs 2024")
k5.metric("💹 GDP H1-FY26",        f"{GDP_H1_FY26}%",  "vs 1.9% prior yr")
k6.metric("🏛️ FX Reserves",        f"${FX_RESERVES_B}B", "Apr 24 SBP")

st.markdown("<br>", unsafe_allow_html=True)

# ─── monthly 2026 bar ────────────────────────────────────────────────────────
with st.expander("📊 2026 Monthly CPI Inflation — PBS Official Data", expanded=True):
    months = list(MONTHLY_2026.keys())
    values = list(MONTHLY_2026.values())
    colors = ["#ff5c7a" if v > 9 else "#ff8c42" if v > 7 else "#f2c94c" if v > 5 else "#39d98a"
              for v in values]
    fm = fig0("CPI Inflation YoY — 2026 Month by Month (PBS Press Releases)")
    fm.add_trace(go.Bar(x=months, y=values, marker_color=colors,
                        text=[f"{v:.2f}%" for v in values], textposition="outside",
                        hovertemplate="%{x} 2026: %{y:.2f}%<extra>PBS</extra>"))
    fm.add_hline(y=7.0, line_dash="dot", line_color="#6ea8fe", line_width=1,
                 annotation_text="SBP upper target 7%", annotation_font_color="#6ea8fe",
                 annotation_font_size=10)
    fm.update_yaxes(title_text="YoY (%)", range=[0, 14])
    st.plotly_chart(fm, use_container_width=True)
    st.caption(
        "**Sources:** PBS monthly CPI press releases — "
        "Jan 5.8%, Feb 7.0% (TradingEconomics/PBS), "
        "Mar 7.3% (PBS/Dawn Apr 1 2026), "
        "Apr 10.89% (PBS press release May 1 2026, confirmed Business Recorder & Geo). "
        "SBP hiked policy rate +100bps to 11.5% on Apr 27 2026 in response."
    )

# ─── tabs ─────────────────────────────────────────────────────────────────────
t1,t2,t3,t4 = st.tabs(["📊 Dashboard","📈 Inflation Model","💰 Savings Value","🧠 Decision Engine"])

# ══════════════════════════════════════════════════════════════════════════════
with t1:
    c1,c2 = st.columns(2)

    with c1:
        hist = df[df.year >= 2010]
        f1   = fig0("CPI Inflation — PBS Historical (2010-2025) + RE Forecast (2026-2031)")
        f1.add_trace(go.Scatter(x=hist["year"], y=hist["inflation"],
            name="Historical CPI (PBS)", line=dict(color="#6ea8fe",width=2.5),
            fill="tozeroy", fillcolor="rgba(110,168,254,0.08)"))
        f1.add_trace(go.Scatter(x=hist["year"], y=hist["policy_rate"],
            name="Policy Rate (SBP)", line=dict(color="#39d98a",width=1.5,dash="dash")))
        f1.add_trace(go.Scatter(x=fcast["year"], y=fcast["inflation"],
            name="RE Forecast", line=dict(color="#ff8c42",width=2.5,dash="dot"),
            mode="lines+markers", marker=dict(size=7,color="#ff8c42")))
        # Plot actual 2026 monthly dots
        f1.add_trace(go.Scatter(
            x=[2026,2026,2026,2026],
            y=[MONTHLY_2026["Jan"],MONTHLY_2026["Feb"],MONTHLY_2026["Mar"],MONTHLY_2026["Apr"]],
            name="2026 Monthly (PBS)", mode="markers",
            marker=dict(size=9, color="#ffd166", symbol="circle")))
        f1.add_vline(x=2025.5, line_dash="dot", line_color="#3d4166",
                     annotation_text="▶ Forecast", annotation_font_color="#6070a0", annotation_font_size=10)
        f1.update_yaxes(title_text="%")
        st.plotly_chart(f1, use_container_width=True)

    with c2:
        f2 = fig0(f"Savings — {pkr(principal)} @ {dep_rate:.2f}% | {scenario}")
        f2.add_trace(go.Scatter(x=sdf["year"], y=sdf["nominal"],
            name="Nominal", line=dict(color="#39d98a",width=2.5),
            fill="tozeroy", fillcolor="rgba(57,217,138,0.08)",
            hovertemplate="%{x}: ₨%{y:,.0f}<extra>Nominal</extra>"))
        f2.add_trace(go.Scatter(x=sdf["year"], y=sdf["real"],
            name="Real (2026 PKR)", line=dict(color="#ff5c7a",width=2.5),
            fill="tozeroy", fillcolor="rgba(255,92,122,0.08)",
            hovertemplate="%{x}: ₨%{y:,.0f}<extra>Real</extra>"))
        f2.add_hline(y=principal, line_dash="dot", line_color="white",
                     line_width=0.8, annotation_text="Principal",
                     annotation_font_color="#a0a8d0", annotation_font_size=10)
        gc = "#39d98a" if gain >= 0 else "#ff5c7a"
        f2.add_annotation(x=2031, y=float(sdf.iloc[-1]["real"]),
            text=f"Real 2031: {pkr(final_real)}", font=dict(color=gc,size=11),
            showarrow=False, xanchor="right")
        f2.update_yaxes(tickformat="₨,.0f")
        st.plotly_chart(f2, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        bc = ["#39d98a" if v>0 else "#ff5c7a" for v in fcast["real_rate"]]
        f3 = fig0("Real Interest Rate Forecast = Policy Rate − CPI")
        f3.add_trace(go.Bar(x=fcast["year"], y=fcast["real_rate"], marker_color=bc,
            text=[f"{v:+.1f}%" for v in fcast["real_rate"]], textposition="outside",
            hovertemplate="%{x}: %{y:+.2f}%<extra>Real Rate</extra>"))
        f3.add_hline(y=0, line_color="white", line_width=0.8, line_dash="dot")
        # Mark live real rate
        f3.add_annotation(x=2026, y=LATEST_REAL_RATE+0.8,
            text=f"Live: {LATEST_REAL_RATE:+.2f}%", font=dict(color="#ffd166",size=10), showarrow=False)
        f3.update_yaxes(title_text="%")
        st.plotly_chart(f3, use_container_width=True)

    with c4:
        f4 = fig0("SBP Policy Rate Full History — Verified (2000-2026)")
        f4.add_trace(go.Scatter(x=df["year"], y=df["policy_rate"],
            name="Policy Rate", line=dict(color="#6ea8fe",width=2),
            fill="tozeroy", fillcolor="rgba(110,168,254,0.08)"))
        f4.add_trace(go.Scatter(x=df["year"], y=df["inflation"],
            name="CPI (PBS)", line=dict(color="#ff8c42",width=1.5,dash="dash")))
        # Mark current live rate
        f4.add_hline(y=LATEST_RATE, line_color="#ffd166", line_width=1, line_dash="dash",
                     annotation_text=f"Current: {LATEST_RATE}% (May 2026)",
                     annotation_font_color="#ffd166", annotation_font_size=10)
        f4.add_annotation(x=2023, y=22.5, text="22% Peak\n(Jun 2023)",
            font=dict(color="#ff5c7a",size=10), showarrow=True, arrowcolor="#ff5c7a", ax=30, ay=-25)
        f4.update_yaxes(title_text="%")
        st.plotly_chart(f4, use_container_width=True)

    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2a2d3e;border-radius:10px;padding:14px 20px;'>
      <span style='color:#6ea8fe;font-family:monospace;font-size:12px;font-weight:bold;'>✅ DATA VERIFICATION</span>
      <div style='margin-top:8px;'>
        <span class='badge'>✅ CPI 2000–2025 — PBS/finhisaab annual avgs</span>
        <span class='badge'>✅ 2023: 30.77% — World Bank confirmed</span>
        <span class='badge'>✅ 2024: 12.65% — PBS confirmed</span>
        <span class='badge'>✅ 2025: 3.53% — PBS confirmed</span>
        <span class='badge'>✅ Jan 2026: 5.8% — PBS/TradingEconomics</span>
        <span class='badge'>✅ Feb 2026: 7.0% — PBS/TradingEconomics</span>
        <span class='badge'>✅ Mar 2026: 7.3% — PBS press release (Dawn Apr 1)</span>
        <span class='badge'>✅ Apr 2026: 10.89% — PBS press release (May 1 2026)</span>
        <span class='badge'>✅ SBP rate 11.5% — SBP MPC Apr 27 2026 (Dawn/Geo/BRecorder)</span>
        <span class='badge'>✅ GDP H1-FY26: 3.8% — SBP MPC statement Apr 27</span>
        <span class='badge'>✅ FX Reserves $15.8B — SBP Apr 24 2026</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
with t2:
    l,r = st.columns([1,1])
    with l:
        st.markdown("### 📐 Rational Expectations Model")
        st.markdown("""
        Based on **Muth (1961)** and **Lucas (1972)**:
        agents forecast inflation using all available information efficiently.
        """)
        st.latex(r"\pi_t = \alpha + \beta_1\pi_{t-1} + \beta_2\pi_{t-2} + \beta_3 r_t + \beta_4 g_t + \beta_5 \Delta e_t + \varepsilon_t")
        st.markdown("""
        | Symbol | Variable | Source |
        |--------|----------|--------|
        | π(t−1,t−2) | Lagged CPI | PBS |
        | r | SBP Policy Rate | SBP |
        | g | GDP Growth | World Bank |
        | Δe | PKR/USD change | SBP/CEIC |
        """)
        m1,m2,m3 = st.columns(3)
        m1.metric("R² (Train)", str(tr2), "2002–2022")
        m2.metric("MAE (Test)", f"{tmae}%", "2023–2025 PBS")
        m3.metric("Test obs", "3", "All real PBS")

        st.markdown("### 🧪 Validation — Real PBS Data")
        vd = val_df.copy()
        vd.columns = ["Year","Actual PBS (%)","Predicted (%)","Error (pp)"]
        st.dataframe(vd.set_index("Year"), use_container_width=True)

    with r:
        coef_df = pd.DataFrame({
            "Feature": ["π(t−1)","π(t−2)","SBP Rate","GDP Growth","PKR Change"],
            "Coef": model.coef_.round(3)
        }).sort_values("Coef")
        fc = fig0("Model Coefficients (Standardized)")
        fc.add_trace(go.Bar(x=coef_df["Coef"], y=coef_df["Feature"], orientation="h",
            marker_color=["#39d98a" if v>=0 else "#ff5c7a" for v in coef_df["Coef"]],
            text=[f"{v:+.3f}" for v in coef_df["Coef"]], textposition="outside"))
        st.plotly_chart(fc, use_container_width=True)

        td = df[df.year<=2022]
        ff = fig0("In-Sample Fit (2002–2022) vs PBS")
        ff.add_trace(go.Scatter(x=td["year"], y=td["inflation"],
            name="PBS Actual", line=dict(color="#6ea8fe",width=2.5)))
        ff.add_trace(go.Scatter(x=td["year"],
            y=model.predict(sc.transform(td[features])).round(2),
            name="Model Fit", line=dict(color="#39d98a",width=1.5,dash="dash")))
        ff.update_yaxes(title_text="%")
        st.plotly_chart(ff, use_container_width=True)

    st.markdown("### 📅 Forecast Table 2026–2031")
    fd = fcast.copy()
    fd["verdict"] = fd["real_rate"].apply(lambda x: verdict(x)[0])
    fd.columns = ["Year","Inflation (%)","Policy Rate (%)","Real Rate (%)","Verdict"]
    st.dataframe(fd.set_index("Year"), use_container_width=True)
    st.caption(
        f"Baseline: 2025 CPI=3.53% (PBS). 2026 blended with Jan–Apr 2026 PBS avg (7.74%). "
        f"Policy rate anchored at 11.5% (SBP Apr 27 2026 MPC). Scenario: **{scenario}**. "
        f"SBP MPC Apr 2026 stated inflation 'likely above 7% target for most FY27'."
    )

# ══════════════════════════════════════════════════════════════════════════════
with t3:
    fs = fig0(f"Real vs Nominal Savings — {pkr(principal)} @ {dep_rate:.2f}% | {scenario}")
    fs.add_trace(go.Scatter(x=sdf["year"], y=sdf["nominal"],
        name="Nominal (PKR)", line=dict(color="#39d98a",width=3),
        fill="tozeroy", fillcolor="rgba(57,217,138,0.10)",
        hovertemplate="%{x}: ₨%{y:,.0f}<extra>Nominal</extra>"))
    fs.add_trace(go.Scatter(x=sdf["year"], y=sdf["real"],
        name="Real (2026 PKR)", line=dict(color="#ff5c7a",width=3),
        fill="tozeroy", fillcolor="rgba(255,92,122,0.10)",
        hovertemplate="%{x}: ₨%{y:,.0f}<extra>Real</extra>"))
    fs.add_hline(y=principal, line_dash="dot", line_color="white", line_width=0.8,
                 annotation_text="Principal", annotation_font_color="#a0a8d0", annotation_font_size=10)
    fs.update_yaxes(tickformat="₨,.0f")
    st.plotly_chart(fs, use_container_width=True)

    s1,s2,s3,s4 = st.columns(4)
    pp_pct = (final_real/principal*100)
    s1.metric("Start (2026)",   pkr(principal))
    s2.metric("Nominal (2031)", pkr(sdf.iloc[-1]["nominal"]))
    s3.metric("Real (2031)",    pkr(final_real), f"{'+' if gain>=0 else ''}{pkr(gain)}")
    s4.metric("PP Retained",    f"{pp_pct:.1f}%", f"{'Gained' if pp_pct>100 else 'Lost'} {abs(pp_pct-100):.1f}pp")

    st.markdown("### 📊 Instrument Comparison")
    fsc = fig0(f"Real Value of {pkr(principal)} by Instrument (2026 PKR, {scenario})")
    for lbl, rate, col in [("Savings Acct 8%",8,"#ff5c7a"),
                            ("Fixed Dep 11.5%",11.5,"#6ea8fe"),
                            ("NSS Cert 15%",15,"#39d98a"),
                            ("Sukuk Bond 17%",17,"#ffd166")]:
        sc_df = savings_trajectory(principal, rate, fcast)
        fsc.add_trace(go.Scatter(x=sc_df["year"], y=sc_df["real"], name=lbl,
            line=dict(color=col,width=2), mode="lines+markers", marker=dict(size=5),
            hovertemplate=f"{lbl}<br>%{{x}}: ₨%{{y:,.0f}}<extra></extra>"))
    fsc.add_hline(y=principal, line_dash="dot", line_color="white", line_width=0.8,
                  annotation_text="Principal", annotation_font_color="#a0a8d0", annotation_font_size=10)
    fsc.update_yaxes(tickformat="₨,.0f")
    st.plotly_chart(fsc, use_container_width=True)

    st.markdown("### 📋 Year-by-Year Breakdown")
    tbl = sdf.copy()
    tbl["PP Retained"]          = (tbl["real"]/principal*100).round(1).astype(str)+"%"
    tbl["Nominal"]              = tbl["nominal"].apply(pkr)
    tbl["Real (2026 PKR)"]      = tbl["real"].apply(pkr)
    tbl["Inflation (PBS/RE)"]   = tbl["infl"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(tbl[["year","Inflation (PBS/RE)","Nominal","Real (2026 PKR)","PP Retained"]]
                 .set_index("year").rename_axis("Year"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("### 🧠 Save vs Spend Decision — 2026–2031")

    # Show live current verdict first
    lbl_live, col_live, bg_live = verdict(LATEST_REAL_RATE)
    st.markdown(f"""
    <div class='vcard' style='background:{bg_live};border:2px solid {col_live}66;'>
      <div style='font-size:11px;color:#6070a0;font-family:monospace;'>
        🔴 LIVE — MAY 2026 (PBS Apr + SBP Rate)
      </div>
      <div style='font-size:28px;margin:6px 0;'>{lbl_live.split()[0]}</div>
      <div style='color:{col_live};font-weight:bold;font-size:16px;margin-bottom:8px;'>
        {' '.join(lbl_live.split()[1:])}
      </div>
      <div style='color:#a0a8d0;font-size:13px;'>
        CPI (Apr 2026 PBS): <b style='color:#ff8c42;'>10.89%</b> &nbsp;|&nbsp;
        SBP Rate: <b style='color:#6ea8fe;'>11.50%</b> &nbsp;|&nbsp;
        Real Rate: <b style='color:{col_live};'>{LATEST_REAL_RATE:+.2f}%</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Forecast Years")
    cols = st.columns(3)
    for i, (_,r) in enumerate(fcast.iterrows()):
        lbl,col,bg = verdict(r["real_rate"])
        bw = min(100, max(5, (r["real_rate"]+12)*5))
        with cols[i%3]:
            st.markdown(f"""
            <div class='vcard' style='background:{bg};border:1px solid {col}44;'>
              <div style='font-size:10px;color:#6070a0;font-family:monospace;'>YEAR {int(r["year"])}</div>
              <div style='font-size:22px;margin:5px 0;'>{lbl.split()[0]}</div>
              <div style='color:{col};font-weight:bold;font-size:13px;margin-bottom:6px;'>
                {' '.join(lbl.split()[1:])}
              </div>
              <div style='color:#a0a8d0;font-size:12px;'>
                CPI: <b style='color:#ff8c42;'>{r["inflation"]:.2f}%</b>&nbsp;
                Rate: <b style='color:#6ea8fe;'>{r["policy_rate"]:.1f}%</b>
              </div>
              <div style='color:#a0a8d0;font-size:12px;margin-top:2px;'>
                Real Rate: <b style='color:{col};'>{r["real_rate"]:+.2f}%</b>
              </div>
              <div style='margin-top:8px;height:4px;background:#0f1117;border-radius:4px;'>
                <div style='height:100%;border-radius:4px;width:{bw:.0f}%;background:{col};'></div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 Fisher's Real Rate Decision Framework")
    for ico,rule,lbl,detail,col in [
        ("💰","Real Rate > 3%","STRONGLY SAVE",
         "Bank fixed deposits and NSS certificates clearly beat inflation. Your money grows in real terms.","#39d98a"),
        ("✅","0% < Real Rate ≤ 3%","SAVE (Marginally)",
         "Savings slightly outpace inflation. NSS certificates (~15%) offer better protection than plain savings accounts.","#6fcf97"),
        ("⚖️","-3% < Real Rate ≤ 0%","NEUTRAL — Diversify",
         "Inflation nearly cancels interest income. Mix savings with gold (tola), KSE-100, or real estate.","#f2c94c"),
        ("⚠️","Real Rate < -3%","SPEND / INVEST IN REAL ASSETS",
         "Idle PKR savings are losing purchasing power rapidly. Property, gold, USD holdings, or equities are better.","#ff5c7a"),
    ]:
        st.markdown(f"""
        <div style='background:#0f1117;border:1px solid {col}22;border-radius:8px;
                    padding:14px 18px;margin-bottom:8px;display:flex;gap:14px;align-items:flex-start;'>
          <span style='font-size:24px;flex-shrink:0;'>{ico}</span>
          <div>
            <span style='color:{col};font-weight:bold;font-size:13px;font-family:monospace;'>{lbl}</span>
            <span style='color:#3d4166;font-size:11px;margin-left:12px;'>when {rule}</span>
            <div style='color:#a0a8d0;font-size:12px;margin-top:5px;line-height:1.6;'>{detail}</div>
          </div>
        </div>""", unsafe_allow_html=True)

# ─── footer ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#3d4166;font-size:10px;font-family:monospace;
            border-top:1px solid #2a2d3e;padding-top:14px;'>
  CPI: Pakistan Bureau of Statistics (PBS) — pbs.gov.pk &nbsp;·&nbsp;
  Policy Rate: State Bank of Pakistan (SBP) MPC Statements — sbp.org.pk &nbsp;·&nbsp;
  GDP: World Bank / IMF WEO &nbsp;·&nbsp; PKR/USD: SBP / CEIC<br>
  News verification: Dawn · Business Recorder · Geo.tv &nbsp;·&nbsp;
  Model: Rational Expectations OLS (Muth 1961, Lucas 1972) &nbsp;·&nbsp;
  Not financial advice. Last data: May 2026.
</div>
""", unsafe_allow_html=True)
