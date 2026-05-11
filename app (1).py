import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇵🇰 Pakistan Savings Predictor",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme */
    .stApp { background-color: #0f1117; color: #e0e0f0; }
    .block-container { padding-top: 1.5rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1d2e; border-right: 1px solid #2a2d3e; }
    [data-testid="stSidebar"] * { color: #e0e0f0 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #a0a8d0 !important; font-size: 11px !important; }
    [data-testid="stMetricValue"] { color: #6ea8fe !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #1a1d2e; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { color: #a0a8d0; }
    .stTabs [aria-selected="true"] { color: #e0e0f0 !important; background: #0f1117 !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #2a2d3e; border-radius: 8px; }

    /* Headers */
    h1, h2, h3 { color: #e0e0f0 !important; }
    h1 { font-family: monospace; letter-spacing: 2px; }

    /* Info/Success/Warning boxes */
    .verdict-save    { background:#0d2e1a; border:1px solid #39d98a55; border-radius:10px; padding:16px; }
    .verdict-neutral { background:#2e2500; border:1px solid #f2c94c55; border-radius:10px; padding:16px; }
    .verdict-spend   { background:#2d0a12; border:1px solid #ff5c7a55; border-radius:10px; padding:16px; }

    /* Slider label */
    .stSlider label { color: #a0a8d0 !important; }

    /* Input */
    .stNumberInput label { color: #a0a8d0 !important; }
    .stSelectbox label   { color: #a0a8d0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Verified Data ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.DataFrame({
        'year': list(range(2000, 2026)),
        # PBS Annual Average CPI (verified via finhisaab.com / PBS)
        'inflation': [
            4.37, 3.15, 3.29, 2.50, 6.97, 9.12, 7.92, 7.58,
            19.78,12.13,12.90,12.42,10.26, 7.28, 7.19, 2.55,
            3.75, 4.25, 5.28, 9.40, 9.47, 9.50,19.87,30.76,
            12.65, 3.53
        ],
        # SBP end-of-year policy rate (verified via CEIC / SBP)
        'policy_rate': [
            13.0, 13.0, 9.5, 7.5, 8.0, 9.0, 9.5,10.0,
            15.0, 14.0,13.0,12.0,10.5, 9.5,10.0, 6.0,
            5.75,  6.0, 8.5,13.25,7.0, 8.75,16.0,22.0,
            13.0, 11.5
        ],
        # GDP growth % — World Bank / IMF
        'gdp_growth': [
            4.3, 2.0, 3.1, 4.8, 7.5, 9.0, 5.8, 4.8,
            1.7, 4.1, 2.6, 3.6, 4.4, 3.7, 4.1, 5.1,
            4.6, 5.2, 5.5, 3.3,-0.5, 5.7, 6.0,-0.2,
            2.4, 2.7
        ],
        # PKR/USD annual avg — SBP
        'exchange_rate': [
            53.7, 61.5, 59.7, 57.8, 57.6, 59.7, 60.0, 60.6,
            70.4, 81.7, 85.5, 86.3, 93.4,101.6,102.9,102.1,
           104.8,105.5,121.8,158.4,160.9,177.0,204.5,284.0,
           278.5,281.3
        ],
    })
    df['real_interest_rate']   = df['policy_rate'] - df['inflation']
    df['inflation_lag1']       = df['inflation'].shift(1)
    df['inflation_lag2']       = df['inflation'].shift(2)
    df['exchange_rate_change'] = df['exchange_rate'].pct_change() * 100
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def train_model(df):
    features = ['inflation_lag1','inflation_lag2','policy_rate',
                'gdp_growth','exchange_rate_change']
    train = df[df.year <= 2022]
    test  = df[df.year >  2022]

    scaler   = StandardScaler()
    Xs_train = scaler.fit_transform(train[features])
    Xs_test  = scaler.transform(test[features])

    model = LinearRegression()
    model.fit(Xs_train, train['inflation'])

    train_r2 = r2_score(train['inflation'], model.predict(Xs_train))
    test_mae = mean_absolute_error(test['inflation'], model.predict(Xs_test))

    return model, scaler, features, round(train_r2,3), round(test_mae,2)

def get_forecast(model, scaler, features, df, deposit_rate, scenario):
    multipliers = {'🟢 Optimistic': 0.78, '🟡 Base (SBP)': 1.0, '🔴 Pessimistic': 1.22}
    m = multipliers[scenario]

    future_macros = {
        2026: {'policy_rate': 11.5, 'gdp_growth': 4.0, 'exchange_rate_change': 3.0},
        2027: {'policy_rate': 10.5, 'gdp_growth': 4.5, 'exchange_rate_change': 2.5},
        2028: {'policy_rate':  9.5, 'gdp_growth': 5.0, 'exchange_rate_change': 2.0},
        2029: {'policy_rate':  9.0, 'gdp_growth': 5.0, 'exchange_rate_change': 2.0},
        2030: {'policy_rate':  8.5, 'gdp_growth': 5.2, 'exchange_rate_change': 1.5},
        2031: {'policy_rate':  8.0, 'gdp_growth': 5.5, 'exchange_rate_change': 1.5},
    }

    prev2 = df[df.year==2024]['inflation'].values[0]
    prev1 = df[df.year==2025]['inflation'].values[0]
    rows  = []

    for yr, macro in future_macros.items():
        row  = [prev1, prev2, macro['policy_rate'],
                macro['gdp_growth'], macro['exchange_rate_change']]
        pred = float(model.predict(scaler.transform([row]))[0])
        if yr == 2026:
            pred = pred * 0.6 + 10.89 * 0.4
        pred = round(max(3.0, min(pred * m, 30.0)), 2)
        rir  = round(macro['policy_rate'] - pred, 2)
        rows.append({'year': yr, 'inflation': pred,
                     'policy_rate': macro['policy_rate'], 'real_rate': rir})
        prev2, prev1 = prev1, pred

    return pd.DataFrame(rows)

def compute_savings(principal, deposit_rate, forecast_df):
    rows = [{'year': 2026, 'nominal': principal, 'real': principal,
             'inflation': forecast_df.iloc[0]['inflation']}]
    nominal = principal
    real    = principal
    for _, r in forecast_df.iloc[1:].iterrows():
        nominal = nominal * (1 + deposit_rate/100)
        real    = real    * ((1 + deposit_rate/100) / (1 + r['inflation']/100))
        rows.append({'year': int(r['year']), 'nominal': round(nominal,2),
                     'real': round(real,2), 'inflation': r['inflation']})
    return pd.DataFrame(rows)

def verdict(real_rate):
    if real_rate > 3:   return "💰 STRONGLY SAVE",  "#39d98a", "save"
    if real_rate > 0:   return "✅ SAVE",             "#6fcf97", "save"
    if real_rate > -3:  return "⚖️ NEUTRAL",          "#f2c94c", "neutral"
    return "⚠️ SPEND / INVEST",                        "#ff5c7a", "spend"

def fmt_pkr(v):
    if v >= 1e6: return f"₨{v/1e6:.2f}M"
    if v >= 1e3: return f"₨{v/1e3:.0f}K"
    return f"₨{v:.0f}"

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOT_BG   = "#1a1d2e"
PAPER_BG  = "#0f1117"
GRID_COL  = "#2a2d3e"
TEXT_COL  = "#a0a8d0"
TICK_COL  = "#6070a0"

def dark_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e0f0", size=13)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COL, family="monospace"),
        legend=dict(bgcolor="#1a1d2e", bordercolor="#3d4166", borderwidth=1),
        margin=dict(l=40, r=20, t=45, b=40),
        xaxis=dict(gridcolor=GRID_COL, tickcolor=TICK_COL, linecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, tickcolor=TICK_COL, linecolor=GRID_COL),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()
model, scaler, features, train_r2, test_mae = train_model(df)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇵🇰 Controls")
    st.markdown("---")

    principal = st.number_input(
        "Principal (PKR)", min_value=10_000, max_value=100_000_000,
        value=1_000_000, step=50_000, format="%d"
    )
    st.caption(f"➜ {fmt_pkr(principal)}")

    deposit_rate = st.slider(
        "Deposit / Savings Rate (%)", min_value=5.0, max_value=22.0,
        value=13.0, step=0.25
    )

    st.markdown("**Quick Presets**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Savings\nAcct 8%"):  deposit_rate = 8.0
        if st.button("Fixed\nDep 13%"):   deposit_rate = 13.0
    with col2:
        if st.button("NSS\n15%"):         deposit_rate = 15.0
        if st.button("Sukuk\n17%"):       deposit_rate = 17.0

    scenario = st.selectbox(
        "Inflation Scenario",
        ["🟢 Optimistic", "🟡 Base (SBP)", "🔴 Pessimistic"],
        index=1
    )

    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown(f"- Train R²: `{train_r2}`")
    st.markdown(f"- Test MAE: `{test_mae}%`")
    st.markdown(f"- Data: PBS · SBP · World Bank")
    st.markdown(f"- Training: 2002–2022")
    st.markdown(f"- Validation: 2023–2025")

# ─── COMPUTE ──────────────────────────────────────────────────────────────────
forecast_df = get_forecast(model, scaler, features, df, deposit_rate, scenario)
savings_df  = compute_savings(principal, deposit_rate, forecast_df)
final_real  = savings_df.iloc[-1]['real']
real_gain   = final_real - principal
mid_verdict, mid_color, mid_cls = verdict(forecast_df.iloc[2]['real_rate'])

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#111424,#1a1d2e,#0d1a2e);
            border:1px solid #2a2d3e; border-radius:12px; padding:20px 28px; margin-bottom:20px;'>
  <div style='display:flex; align-items:center; gap:14px;'>
    <span style='font-size:40px;'>🇵🇰</span>
    <div>
      <h1 style='margin:0; font-size:22px; letter-spacing:3px; color:#6ea8fe;'>
        PAKISTAN SAVINGS PREDICTOR
      </h1>
      <p style='margin:4px 0 0; color:#6070a0; font-size:11px; letter-spacing:1.5px;'>
        RATIONAL EXPECTATIONS MODEL · PBS/SBP/WORLD BANK DATA · FORECAST 2026–2031
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI METRICS ──────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("2025 Inflation (PBS)", "3.53%",  "-27.2% vs 2023")
k2.metric("2026 Forecast",  f"{forecast_df.iloc[0]['inflation']:.1f}%", "RE Model")
k3.metric("Policy Rate Now", "11.5%", "SBP Apr 2026")
k4.metric("Real Rate 2028",  f"{forecast_df.iloc[2]['real_rate']:+.1f}%",
          "Positive ✅" if forecast_df.iloc[2]['real_rate'] > 0 else "Negative ⚠️")
k5.metric("Savings Verdict", mid_verdict.split()[1] if len(mid_verdict.split())>1 else mid_verdict)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "📈 Inflation Model", "💰 Savings Value", "🧠 Decision Engine"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)

    # ── Chart 1: Historical + Forecast Inflation ──────────────────────────────
    with col_a:
        hist_recent = df[df.year >= 2010]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_recent['year'], y=hist_recent['inflation'],
            name='Historical CPI (PBS)', line=dict(color='#6ea8fe', width=2),
            fill='tozeroy', fillcolor='rgba(110,168,254,0.1)',
            hovertemplate='%{x}: %{y:.2f}%<extra>Historical</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['year'], y=forecast_df['inflation'],
            name='RE Forecast', line=dict(color='#ff8c42', width=2.5, dash='dot'),
            mode='lines+markers', marker=dict(size=7, color='#ff8c42'),
            hovertemplate='%{x}: %{y:.2f}%<extra>Forecast</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=hist_recent['year'], y=hist_recent['policy_rate'],
            name='Policy Rate (SBP)', line=dict(color='#39d98a', width=1.5, dash='dash'),
            hovertemplate='%{x}: %{y:.2f}%<extra>Policy Rate</extra>'
        ))
        fig.add_vline(x=2025.5, line_dash="dot", line_color="#3d4166",
                      annotation_text="Forecast →", annotation_font_color="#6070a0")
        dark_layout(fig, "CPI Inflation — Historical (PBS) + RE Forecast")
        fig.update_yaxes(title_text="(%)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2: Savings Nominal vs Real ─────────────────────────────────────
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=savings_df['year'], y=savings_df['nominal'],
            name='Nominal Value', line=dict(color='#39d98a', width=2.5),
            fill='tozeroy', fillcolor='rgba(57,217,138,0.08)',
            hovertemplate='%{x}: ₨%{y:,.0f}<extra>Nominal</extra>'
        ))
        fig2.add_trace(go.Scatter(
            x=savings_df['year'], y=savings_df['real'],
            name='Real Value (2026 PKR)', line=dict(color='#ff5c7a', width=2.5),
            fill='tozeroy', fillcolor='rgba(255,92,122,0.08)',
            hovertemplate='%{x}: ₨%{y:,.0f}<extra>Real</extra>'
        ))
        fig2.add_hline(y=principal, line_dash="dot", line_color="white",
                       line_width=0.8, annotation_text="Principal",
                       annotation_font_color="#6070a0")
        dark_layout(fig2, f"Savings: {fmt_pkr(principal)} @ {deposit_rate:.1f}%")
        fig2.update_yaxes(tickformat="₨,.0f")
        gain_col = "#39d98a" if real_gain >= 0 else "#ff5c7a"
        fig2.add_annotation(
            x=2031, y=final_real,
            text=f"Real gain: {fmt_pkr(real_gain)}",
            font=dict(color=gain_col, size=11), showarrow=False, xanchor="right"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Real Interest Rate Bars ─────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        bar_colors = ['#39d98a' if v > 0 else '#ff5c7a' for v in forecast_df['real_rate']]
        fig3 = go.Figure(go.Bar(
            x=forecast_df['year'], y=forecast_df['real_rate'],
            marker_color=bar_colors,
            text=[f"{v:+.1f}%" for v in forecast_df['real_rate']],
            textposition='outside',
            hovertemplate='%{x}: %{y:+.2f}%<extra>Real Rate</extra>'
        ))
        fig3.add_hline(y=0, line_color="white", line_width=0.8, line_dash="dot")
        dark_layout(fig3, "Real Interest Rate Forecast (Policy Rate − CPI)")
        fig3.update_yaxes(title_text="%")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: Exchange Rate History ───────────────────────────────────────
    with col_d:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df['year'], y=df['exchange_rate'],
            name='PKR/USD', line=dict(color='#ff5c7a', width=2),
            fill='tozeroy', fillcolor='rgba(255,92,122,0.1)',
            hovertemplate='%{x}: ₨%{y:.1f} per USD<extra></extra>'
        ))
        dark_layout(fig4, "PKR / USD Exchange Rate (SBP Annual Average)")
        fig4.update_yaxes(title_text="PKR per USD")
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INFLATION MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📐 Model Theory")
        st.markdown("""
        Based on **Muth (1961)** and **Lucas (1972)** Rational Expectations:

        > Agents form forecasts using **all available information** efficiently.
        > Their predictions are unbiased and incorporate macro signals.

        **Equation:**
        """)
        st.latex(r"\pi_t = \alpha + \beta_1\pi_{t-1} + \beta_2\pi_{t-2} + \beta_3 r_t + \beta_4 g_t + \beta_5 \Delta e_t + \varepsilon_t")
        st.markdown("""
        | Symbol | Variable | Source |
        |--------|----------|--------|
        | π(t-1), π(t-2) | Lagged inflation | PBS |
        | r | SBP Policy Rate | SBP |
        | g | GDP Growth | World Bank |
        | Δe | PKR depreciation | SBP |
        """)

        st.markdown("### 📊 Model Performance")
        m1, m2 = st.columns(2)
        m1.metric("Training R²", f"{train_r2}", "2002–2022")
        m2.metric("Test MAE", f"{test_mae}%", "2023–2025 PBS data")

        st.markdown("### 🧪 Validation on Real Data")
        val_df = df[df.year > 2022][['year','inflation']].copy()
        feat_data = df[df.year > 2022][features]
        val_df['predicted'] = model.predict(scaler.transform(feat_data)).round(2)
        val_df['error'] = (val_df['predicted'] - val_df['inflation']).round(2)
        val_df.columns = ['Year','Actual (PBS)','Predicted','Error']
        st.dataframe(val_df.set_index('Year'), use_container_width=True)

    with col2:
        # Model coefficients chart
        coef_df = pd.DataFrame({
            'Feature': ['Inflation (t-1)','Inflation (t-2)','Policy Rate','GDP Growth','PKR Change'],
            'Coefficient': model.coef_.round(3)
        }).sort_values('Coefficient')

        colors = ['#39d98a' if v >= 0 else '#ff5c7a' for v in coef_df['Coefficient']]
        fig_coef = go.Figure(go.Bar(
            x=coef_df['Coefficient'], y=coef_df['Feature'],
            orientation='h', marker_color=colors,
            text=[f"{v:+.3f}" for v in coef_df['Coefficient']],
            textposition='outside'
        ))
        dark_layout(fig_coef, "Model Coefficients (Standardized)")
        fig_coef.update_xaxes(title_text="Coefficient Value")
        st.plotly_chart(fig_coef, use_container_width=True)

        # Fit chart
        train_data = df[df.year <= 2022]
        fitted = model.predict(scaler.transform(train_data[features]))
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            x=train_data['year'], y=train_data['inflation'],
            name='Actual (PBS)', line=dict(color='#6ea8fe', width=2)
        ))
        fig_fit.add_trace(go.Scatter(
            x=train_data['year'], y=fitted.round(2),
            name='Model Fit', line=dict(color='#39d98a', width=1.5, dash='dash')
        ))
        dark_layout(fig_fit, "Model Fit on Training Data (2002–2022)")
        fig_fit.update_yaxes(title_text="Inflation (%)")
        st.plotly_chart(fig_fit, use_container_width=True)

    # Forecast table
    st.markdown("### 📅 Forecast 2026–2031")
    fcast_display = forecast_df.copy()
    fcast_display['verdict'] = fcast_display['real_rate'].apply(
        lambda r: verdict(r)[0]
    )
    fcast_display.columns = ['Year','Inflation (%)','Policy Rate (%)','Real Rate (%)','Verdict']
    st.dataframe(fcast_display.set_index('Year'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SAVINGS VALUE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    # Area chart
    fig_sav = go.Figure()
    fig_sav.add_trace(go.Scatter(
        x=savings_df['year'], y=savings_df['nominal'],
        name='Nominal Value', line=dict(color='#39d98a', width=3),
        fill='tozeroy', fillcolor='rgba(57,217,138,0.12)',
        hovertemplate='%{x}: ₨%{y:,.0f}<extra>Nominal</extra>'
    ))
    fig_sav.add_trace(go.Scatter(
        x=savings_df['year'], y=savings_df['real'],
        name='Real Value (2026 PKR)', line=dict(color='#ff5c7a', width=3),
        fill='tozeroy', fillcolor='rgba(255,92,122,0.12)',
        hovertemplate='%{x}: ₨%{y:,.0f}<extra>Real</extra>'
    ))
    fig_sav.add_hline(y=principal, line_dash="dot", line_color="white",
                      line_width=0.8, annotation_text="Your Principal",
                      annotation_font_color="#a0a8d0")
    dark_layout(fig_sav, f"Savings Trajectory — {fmt_pkr(principal)} @ {deposit_rate:.1f}% Deposit Rate")
    fig_sav.update_yaxes(tickformat="₨,.0f")
    st.plotly_chart(fig_sav, use_container_width=True)

    # Scenario comparison
    st.markdown("### 📊 Scenario Comparison — Real Value by Instrument")
    fig_sc = go.Figure()
    scenarios_compare = [
        ('Savings Account (8%)',  8.0,  '#ff5c7a'),
        ('Fixed Deposit (13%)',  13.0,  '#6ea8fe'),
        ('NSS Certificate (15%)',15.0,  '#39d98a'),
        ('Sukuk Bond (17%)',     17.0,  '#ffd166'),
    ]
    for label, rate, color in scenarios_compare:
        sc_df = compute_savings(principal, rate, forecast_df)
        fig_sc.add_trace(go.Scatter(
            x=sc_df['year'], y=sc_df['real'],
            name=label, line=dict(color=color, width=2),
            mode='lines+markers', marker=dict(size=5),
            hovertemplate=f'{label}<br>%{{x}}: ₨%{{y:,.0f}}<extra></extra>'
        ))
    fig_sc.add_hline(y=principal, line_dash="dot", line_color="white",
                     line_width=0.8, annotation_text="Principal",
                     annotation_font_color="#a0a8d0")
    dark_layout(fig_sc, f"Real Value of {fmt_pkr(principal)} — By Instrument (2026 PKR)")
    fig_sc.update_yaxes(tickformat="₨,.0f")
    st.plotly_chart(fig_sc, use_container_width=True)

    # Detailed table
    st.markdown("### 📋 Year-by-Year Breakdown")
    table_df = savings_df.copy()
    table_df['pp_retained_%'] = ((table_df['real'] / principal) * 100).round(1)
    table_df['nominal'] = table_df['nominal'].apply(fmt_pkr)
    table_df['real']    = table_df['real'].apply(fmt_pkr)
    table_df['inflation'] = table_df['inflation'].apply(lambda x: f"{x:.1f}%")
    table_df['pp_retained_%'] = table_df['pp_retained_%'].apply(lambda x: f"{x:.1f}%")
    table_df.columns = ['Year','Inflation','Nominal Value','Real Value (2026 PKR)','PP Retained']
    st.dataframe(table_df.set_index('Year'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🧠 Fisher's Real Rate Decision — Year by Year")
    cols = st.columns(3)
    for i, (_, r) in enumerate(forecast_df.iterrows()):
        label, color, cls = verdict(r['real_rate'])
        with cols[i % 3]:
            st.markdown(f"""
            <div class='verdict-{cls}' style='margin-bottom:12px;'>
              <div style='font-size:11px; color:#6070a0; margin-bottom:4px;'>YEAR {int(r['year'])}</div>
              <div style='font-size:24px;'>{label.split()[0]}</div>
              <div style='color:{color}; font-weight:bold; font-size:14px; margin:6px 0 4px;'>
                {' '.join(label.split()[1:])}
              </div>
              <div style='color:#a0a8d0; font-size:12px;'>Inflation: <b style='color:#ff8c42;'>{r['inflation']:.1f}%</b></div>
              <div style='color:#a0a8d0; font-size:12px;'>Real Rate: <b style='color:{color};'>{r['real_rate']:+.1f}%</b></div>
              <div style='margin-top:8px; height:4px; background:#0f1117; border-radius:4px;'>
                <div style='height:100%; border-radius:4px; width:{min(100,max(5,(r["real_rate"]+12)*5))}%;
                            background:{color};'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 Decision Framework")

    rules = [
        ("💰", "Real Rate > 3%", "STRONGLY SAVE",
         "Bank deposits clearly beat inflation. Fixed deposits and NSS certificates are optimal.", "#39d98a"),
        ("✅", "0% < Real Rate ≤ 3%", "SAVE (Marginally)",
         "Savings beat inflation, but only slightly. Government savings certificates recommended.", "#6fcf97"),
        ("⚖️", "-3% < Real Rate ≤ 0%", "NEUTRAL — Diversify",
         "Inflation nearly cancels your interest. Consider gold (tola), KSE-100 equities, or property.", "#f2c94c"),
        ("⚠️", "Real Rate < -3%", "SPEND / INVEST IN REAL ASSETS",
         "Every day of idle cash you lose purchasing power. Property, gold, USD bonds, or KSE-100.", "#ff5c7a"),
    ]
    for icon, rule, label, detail, color in rules:
        st.markdown(f"""
        <div style='background:#0f1117; border:1px solid {color}22; border-radius:8px;
                    padding:14px; margin-bottom:8px; display:flex; gap:14px; align-items:flex-start;'>
          <span style='font-size:24px;'>{icon}</span>
          <div>
            <span style='color:{color}; font-weight:bold; font-size:13px;'>{label}</span>
            <span style='color:#3d4166; font-size:12px; margin-left:10px;'>when {rule}</span>
            <div style='color:#a0a8d0; font-size:12px; margin-top:4px;'>{detail}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏦 Pakistan Alternatives When Real Rates Are Negative")
    alts = ["Gold (Tola/10g)", "KSE-100 Index", "Real Estate", "Sukuk Bonds",
            "USD Cash", "Prize Bonds", "Roshan Digital Account", "Mutual Funds"]
    st.markdown(" ".join([
        f"<span style='background:#1a1d2e; border:1px solid #3d4166; border-radius:20px; "
        f"padding:4px 12px; font-size:12px; color:#a0a8d0; margin:4px; display:inline-block;'>{a}</span>"
        for a in alts
    ]), unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#3d4166; font-size:11px; border-top:1px solid #2a2d3e; padding-top:16px;'>
  Data: State Bank of Pakistan · Pakistan Bureau of Statistics · World Bank · IMF<br>
  Model: Rational Expectations OLS (Muth 1961, Lucas 1972) · Fisher's Real Rate Theorem<br>
  All forecasts are model estimates, not financial advice.
</div>
""", unsafe_allow_html=True)
