import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Framingham CHD · DSV Mini Project",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark medical theme */
.stApp {
    background-color: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2d4a;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #e63946, #f4a261);
}
.metric-card .label {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7a8aaa;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f0f4ff;
    line-height: 1;
}
.metric-card .delta {
    font-size: 12px;
    color: #64b5f6;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #e0e6ff;
    border-bottom: 2px solid #e63946;
    padding-bottom: 8px;
    margin: 32px 0 20px 0;
    display: inline-block;
}

/* Week badge */
.week-badge {
    display: inline-block;
    background: linear-gradient(135deg, #e63946, #c1121f);
    color: white;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 6px;
}

/* Insight box */
.insight-box {
    background: rgba(100, 181, 246, 0.08);
    border-left: 3px solid #64b5f6;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 16px 0;
    font-size: 0.9rem;
    color: #b0c4de;
}
.insight-box strong { color: #e8eaf0; }

/* Hero */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    line-height: 1.15;
    color: #f0f4ff;
    margin-bottom: 6px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #7a8aaa;
    letter-spacing: 0.04em;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #7a8aaa;
    font-weight: 500;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #e8eaf0 !important;
}

/* Plotly charts: transparent bg */
.js-plotly-plot .plotly { background: transparent !important; }

/* Divider */
hr { border-color: #1e2d4a; margin: 32px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    orig = pd.read_csv("../data/framingham.csv")
    ab   = pd.read_csv("../data/ab.csv", index_col=0)
    eng  = pd.read_csv("../data/week6_engineered.csv")
    sel  = pd.read_csv("../data/week7_selected.csv")
    pca_df = pd.read_csv("../data/week7_pca.csv")
    return orig, ab, eng, sel, pca_df

@st.cache_data
def compute_rf(sel):
    X = sel.drop("TenYearCHD", axis=1)
    y = sel["TenYearCHD"]
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, X.columns.tolist(), rf.feature_importances_

@st.cache_data
def compute_pca(sel):
    X = sel.drop("TenYearCHD", axis=1)
    pca = PCA(random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_

orig, ab, eng, sel, pca_df = load_data()
rf, feat_names, feat_imp = compute_rf(sel)
pca_coords, pca_evr = compute_pca(sel)

# ─── Colour palette ───────────────────────────────────────────────────────────
C_BG   = "#0a0e1a"
C_CARD = "#111827"
C_RED  = "#e63946"
C_BLUE = "#64b5f6"
C_ORG  = "#f4a261"
C_GRN  = "#52b788"
C_GRID = "#1e2d4a"
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b0c4de", family="DM Sans"),
    margin=dict(t=40, b=30, l=10, r=10),
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px'>
        <span style='font-size:2.5rem'>🫀</span>
        <div style='font-family: DM Serif Display, serif; font-size:1.2rem; color:#f0f4ff; margin-top:6px'>
            Framingham CHD
        </div>
        <div style='font-size:11px; color:#7a8aaa; letter-spacing:0.1em; text-transform:uppercase'>
            DSV Mini Project · Weeks 1–7
        </div>
    </div>
    <hr style='border-color:#1e2d4a; margin:12px 0'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 EDA & Cleaning", "⚙️ Feature Engineering",
         "📐 Dimensionality Reduction", "🤖 Risk Predictor"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e2d4a; margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#4a5a7a; padding: 0 4px'>
        <b style='color:#7a8aaa'>Dataset</b><br>Framingham Heart Study<br>
        4,238 patients · 15 features<br><br>
        <b style='color:#7a8aaa'>Target</b><br>10-Year CHD Risk<br>
        15.2% positive rate
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero-title'>What drives a heart<br><i>to fail in ten years?</i></div>
    <div class='hero-subtitle'>Framingham Heart Study — Predicting 10-Year Coronary Heart Disease Risk</div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # KPI row
    chd_pos = orig["TenYearCHD"].sum()
    chd_rate = orig["TenYearCHD"].mean() * 100
    avg_age_chd = orig[orig["TenYearCHD"]==1]["age"].mean()
    male_rate   = orig[orig["male"]==1]["TenYearCHD"].mean() * 100
    female_rate = orig[orig["male"]==0]["TenYearCHD"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, delta in [
        (c1, "Total Patients",    "4,238",        "Framingham cohort"),
        (c2, "CHD Positive",      f"{chd_pos}",   f"{chd_rate:.1f}% of cohort"),
        (c3, "Avg Age (CHD+)",    f"{avg_age_chd:.0f} yrs", "vs 48 yrs in CHD−"),
        (c4, "Male vs Female Risk", f"{male_rate:.0f}% / {female_rate:.0f}%", "male / female CHD rate"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='label'>{label}</div>
            <div class='value'>{value}</div>
            <div class='delta'>{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("<div class='section-header'>CHD Prevalence</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["No CHD", "CHD Risk"],
            values=[orig["TenYearCHD"].value_counts()[0], chd_pos],
            hole=0.65,
            marker_colors=[C_GRN, C_RED],
            textinfo="label+percent",
            textfont=dict(size=13, color="#e8eaf0"),
        ))
        fig.update_layout(**CHART_LAYOUT, height=280,
                          annotations=[dict(text=f"<b>{chd_rate:.1f}%</b><br>CHD",
                                            x=0.5, y=0.5, showarrow=False,
                                            font=dict(size=16, color="#f0f4ff"))])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-header'>CHD Rate by Age Group</div>", unsafe_allow_html=True)
        orig2 = orig.copy()
        orig2["age_group"] = pd.cut(orig2["age"], bins=[30,40,50,60,71],
                                    labels=["30–40","41–50","51–60","61–70"])
        age_chd = orig2.groupby("age_group", observed=True)["TenYearCHD"].mean().reset_index()
        age_chd.columns = ["Age Group", "CHD Rate"]
        age_chd["CHD Rate %"] = (age_chd["CHD Rate"]*100).round(1)
        fig2 = px.bar(age_chd, x="Age Group", y="CHD Rate %",
                      color="CHD Rate %", color_continuous_scale=["#52b788","#f4a261","#e63946"],
                      text="CHD Rate %")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(**CHART_LAYOUT, height=280,
                           coloraxis_showscale=False,
                           yaxis=dict(gridcolor=C_GRID, title="CHD Rate (%)"),
                           xaxis=dict(gridcolor=C_GRID))
        st.plotly_chart(fig2, use_container_width=True)

    # Insight
    st.markdown("""
    <div class='insight-box'>
        <strong>Key Finding:</strong> CHD risk increases dramatically with age — from 5.1% in the 30s to
        28.5% in the 60s. Men face a 52% higher relative risk than women (18.9% vs 12.4%).
        Patients with diabetes face more than double the risk (36.7% vs 14.6%).
    </div>
    """, unsafe_allow_html=True)

    # Risk factors heatmap overview
    st.markdown("<div class='section-header'>Risk Profile Comparison</div>", unsafe_allow_html=True)
    factors = {
        "Feature": ["Age (median)", "sysBP (median)", "Glucose (median)", "BMI (median)", "totChol (median)"],
        "No CHD":  [48, 127, 78, 25.2, 234],
        "CHD+":    [55, 139, 79, 26.2, 238],
    }
    factor_df = pd.DataFrame(factors)
    factor_df["Δ"] = (((factor_df["CHD+"] - factor_df["No CHD"]) / factor_df["No CHD"])*100).round(1)
    factor_df["Δ label"] = factor_df["Δ"].map(lambda x: f"+{x}%" if x>0 else f"{x}%")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="No CHD", x=factor_df["Feature"], y=factor_df["No CHD"],
                          marker_color=C_GRN, opacity=0.85))
    fig3.add_trace(go.Bar(name="CHD+",   x=factor_df["Feature"], y=factor_df["CHD+"],
                          marker_color=C_RED, opacity=0.85))
    fig3.update_layout(**CHART_LAYOUT, height=320, barmode="group",
                       legend=dict(orientation="h", y=1.1),
                       yaxis=dict(gridcolor=C_GRID),
                       xaxis=dict(gridcolor=C_GRID))
    st.plotly_chart(fig3, use_container_width=True)

    # Project journey timeline
    st.markdown("<div class='section-header'>Project Journey</div>", unsafe_allow_html=True)
    weeks = [
        ("Week 1–2", "Team Formation & Data Hunting", "Dataset selection, GitHub setup, first inspection"),
        ("Week 3",   "The Cleaning Sprint",            "Null imputation (mode/median), IQR outlier capping"),
        ("Week 4",   "EDA Deep Dive",                  "Univariate, bivariate & multivariate analysis"),
        ("Week 5",   "Mid-Review: Data Audit",         "Cleaned dataset presented, statistical findings"),
        ("Week 6",   "Feature Engineering",             "Log1p skew fix, ordinal encoding, RobustScaler"),
        ("Week 7",   "Dimensionality Reduction",        "Consensus feature selection + PCA (9 components, 95% var)"),
        ("Week 8",   "Dashboard Construction",          "Streamlit + Plotly narrative — you are here 🎯"),
    ]
    cols = st.columns(len(weeks))
    for col, (wk, title, desc) in zip(cols, weeks):
        is_now = wk == "Week 8"
        col.markdown(f"""
        <div style='background:{"linear-gradient(135deg,#1a2540,#2a1a30)" if is_now else "#111827"};
                    border:1px solid {"#e63946" if is_now else "#1e2d4a"};
                    border-radius:10px; padding:14px 12px; height:160px; overflow:hidden'>
            <div style='font-size:10px; font-weight:600; color:{"#e63946" if is_now else "#4a5a7a"};
                        letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px'>{wk}</div>
            <div style='font-size:12px; font-weight:600; color:#e8eaf0; margin-bottom:8px;
                        line-height:1.3'>{title}</div>
            <div style='font-size:11px; color:#7a8aaa; line-height:1.5'>{desc}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA & Cleaning":
    st.markdown("<div class='hero-title'>Exploratory Data Analysis<br><i>& Cleaning</i></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Weeks 2–4 · Know your data before you model it</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🧹 Missing Values", "🔗 Correlations"])

    with tab1:
        st.markdown("<div class='section-header'>Feature Distributions by CHD Status</div>", unsafe_allow_html=True)

        feat = st.selectbox("Select feature", ["age","sysBP","diaBP","BMI","glucose","totChol","heartRate","cigsPerDay"])
        fig = go.Figure()
        for cls, color, name in [(0, C_GRN, "No CHD"), (1, C_RED, "CHD+")]:
            vals = orig[orig["TenYearCHD"]==cls][feat].dropna()
            fig.add_trace(go.Violin(y=vals, name=name, box_visible=True, meanline_visible=True,
                                    fillcolor=color, opacity=0.7, line_color=color,
                                    marker=dict(color=color)))
        fig.update_layout(**CHART_LAYOUT, height=380,
                          yaxis=dict(gridcolor=C_GRID, title=feat),
                          violingap=0.3)
        st.plotly_chart(fig, use_container_width=True)

        # Skewness bar
        st.markdown("<div class='section-header'>Skewness of Continuous Features</div>", unsafe_allow_html=True)
        cont = ["age","cigsPerDay","totChol","sysBP","diaBP","BMI","heartRate","glucose"]
        sk = orig[cont].skew().sort_values(ascending=False)
        colors_sk = [C_RED if abs(v)>1 else C_ORG if abs(v)>0.5 else C_GRN for v in sk.values]
        fig_sk = go.Figure(go.Bar(x=sk.index, y=sk.values, marker_color=colors_sk,
                                  text=[f"{v:.2f}" for v in sk.values],
                                  textposition="outside"))
        fig_sk.add_hline(y=0.5, line_dash="dash", line_color=C_ORG, annotation_text="Moderate (0.5)")
        fig_sk.add_hline(y=1.0, line_dash="dash", line_color=C_RED, annotation_text="High (1.0)")
        fig_sk.update_layout(**CHART_LAYOUT, height=300,
                              yaxis=dict(gridcolor=C_GRID, title="Skewness"),
                              xaxis=dict(gridcolor=C_GRID))
        st.plotly_chart(fig_sk, use_container_width=True)
        st.markdown("""<div class='insight-box'><strong>cigsPerDay</strong> shows the highest skew (1.18)
        — most patients are non-smokers (0 cigs) with a long right tail of heavy smokers.
        This was fixed with a <strong>log1p transform</strong> in Week 6, reducing skew to 0.32.</div>""",
        unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-header'>Missing Values Before Cleaning</div>", unsafe_allow_html=True)
        nulls = orig.isnull().sum().sort_values(ascending=False)
        nulls = nulls[nulls > 0]
        null_pct = (nulls / len(orig) * 100).round(2)

        fig_null = go.Figure()
        fig_null.add_trace(go.Bar(x=nulls.index, y=nulls.values, name="Missing Count",
                                  marker_color=C_RED, opacity=0.85,
                                  text=null_pct.values, texttemplate="%{text:.1f}%",
                                  textposition="outside"))
        fig_null.update_layout(**CHART_LAYOUT, height=320,
                               yaxis=dict(gridcolor=C_GRID, title="Missing Count"),
                               xaxis=dict(gridcolor=C_GRID))
        st.plotly_chart(fig_null, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.markdown("""<div class='insight-box'>
        <strong>glucose</strong> had 388 missing values (9.2%) — the highest.
        Imputed with <strong>median</strong> to avoid influence from extreme values.</div>""",
        unsafe_allow_html=True)
        c2.markdown("""<div class='insight-box'>
        <strong>Categorical nulls</strong> (education, BPMeds) imputed with
        <strong>mode</strong> — the most frequent value is the safest assumption
        for binary/ordinal features.</div>""", unsafe_allow_html=True)

        # Before / after comparison
        st.markdown("<div class='section-header'>Outlier Treatment — Before vs After IQR Capping</div>",
                    unsafe_allow_html=True)
        feat_iqr = st.selectbox("Select feature", ["glucose","sysBP","cigsPerDay","BMI","totChol"])
        col_a, col_b = st.columns(2)
        with col_a:
            fig_b = px.box(orig, y=feat_iqr, color_discrete_sequence=[C_RED],
                           title=f"{feat_iqr} — Before (Raw)")
            fig_b.update_layout(**CHART_LAYOUT, height=300)
            st.plotly_chart(fig_b, use_container_width=True)
        with col_b:
            fig_a = px.box(ab, y=feat_iqr, color_discrete_sequence=[C_GRN],
                           title=f"{feat_iqr} — After (IQR Capped)")
            fig_a.update_layout(**CHART_LAYOUT, height=300)
            st.plotly_chart(fig_a, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-header'>Correlation Heatmap — All Features</div>", unsafe_allow_html=True)
        corr = orig.corr().round(2)
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu_r", zmid=0,
            text=corr.values, texttemplate="%{text:.2f}",
            colorbar=dict(tickfont=dict(color="#b0c4de")),
        ))
        fig_heat.update_layout(**CHART_LAYOUT, height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""<div class='insight-box'>
        <strong>sysBP ↔ diaBP</strong> are strongly correlated (r ≈ 0.79) — both measure
        blood pressure and carry overlapping information. PCA in Week 7 merges these into
        a single "blood pressure" component, eliminating this multicollinearity.</div>""",
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Feature Engineering":
    st.markdown("<div class='hero-title'>Feature Engineering</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Week 6 · Transforming raw data into model-ready signals</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Skew Correction", "🔢 Scaling", "🏷️ Encoding"])

    with tab1:
        st.markdown("<div class='section-header'>Log1p Transform — cigsPerDay</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(orig, x="cigsPerDay", nbins=40, color_discrete_sequence=[C_RED],
                               title=f"Before Log1p  (skew = {orig['cigsPerDay'].skew():.2f})")
            fig.update_layout(**CHART_LAYOUT, height=300,
                              xaxis=dict(gridcolor=C_GRID), yaxis=dict(gridcolor=C_GRID))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            log_vals = np.log1p(orig["cigsPerDay"].dropna())
            fig2 = px.histogram(x=log_vals, nbins=40, color_discrete_sequence=[C_GRN],
                                title=f"After Log1p  (skew = {log_vals.skew():.2f})")
            fig2.update_layout(**CHART_LAYOUT, height=300,
                               xaxis=dict(title="log1p(cigsPerDay)", gridcolor=C_GRID),
                               yaxis=dict(gridcolor=C_GRID))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""<div class='insight-box'>
        <strong>Why Log1p, not Box-Cox?</strong> — cigsPerDay contains zeros (non-smokers).
        Box-Cox requires strictly positive values and fails on zero. Log1p = log(1 + x) is
        safe for zeros and achieves the best skew reduction (1.18 → 0.32).</div>""",
        unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-header'>RobustScaler vs Alternatives</div>", unsafe_allow_html=True)
        from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

        feat_s = st.selectbox("Feature", ["sysBP","glucose","BMI","totChol","age"])
        vals = ab[feat_s].dropna().values.reshape(-1,1)

        comparisons = {
            "Original"     : vals.flatten(),
            "StandardScaler": StandardScaler().fit_transform(vals).flatten(),
            "MinMaxScaler"  : MinMaxScaler().fit_transform(vals).flatten(),
            "RobustScaler ✅": RobustScaler().fit_transform(vals).flatten(),
        }
        fig_comp = make_subplots(rows=1, cols=4,
                                 subplot_titles=list(comparisons.keys()))
        clrs = [C_BLUE, C_ORG, "#a78bfa", C_GRN]
        for i, (name, vals_t) in enumerate(comparisons.items(), 1):
            fig_comp.add_trace(
                go.Violin(y=vals_t, fillcolor=clrs[i-1], opacity=0.75,
                          line_color=clrs[i-1], box_visible=True, showlegend=False,
                          meanline_visible=True, name=name),
                row=1, col=i
            )
        fig_comp.update_layout(**CHART_LAYOUT, height=360)
        fig_comp.update_yaxes(gridcolor=C_GRID)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown(f"""<div class='insight-box'>
        <strong>RobustScaler</strong> uses median + IQR — not mean/std.
        Even after IQR capping, clinical features like <em>{feat_s}</em> have extreme boundary values.
        StandardScaler's mean is pulled by these; MinMaxScaler's range is set by them.
        RobustScaler is immune to both — the right choice for medical data.</div>""",
        unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='section-header'>Education — Ordinal Encoding</div>", unsafe_allow_html=True)
        edu_counts = orig["education"].value_counts().sort_index()
        edu_chd    = orig.groupby("education")["TenYearCHD"].mean() * 100
        edu_labels = {1:"Some HS", 2:"HS Grad", 3:"Some College", 4:"College+"}

        col_a, col_b = st.columns(2)
        with col_a:
            fig_e = px.bar(x=[edu_labels[k] for k in edu_counts.index],
                           y=edu_counts.values, color_discrete_sequence=[C_BLUE],
                           title="Education Distribution")
            fig_e.update_layout(**CHART_LAYOUT, height=300,
                                yaxis=dict(gridcolor=C_GRID, title="Count"),
                                xaxis=dict(gridcolor=C_GRID))
            st.plotly_chart(fig_e, use_container_width=True)
        with col_b:
            fig_e2 = px.bar(x=[edu_labels[k] for k in edu_chd.index],
                            y=edu_chd.values, color_discrete_sequence=[C_ORG],
                            title="CHD Rate by Education Level",
                            text=[f"{v:.1f}%" for v in edu_chd.values])
            fig_e2.update_traces(textposition="outside")
            fig_e2.update_layout(**CHART_LAYOUT, height=300,
                                 yaxis=dict(gridcolor=C_GRID, title="CHD Rate (%)"),
                                 xaxis=dict(gridcolor=C_GRID))
            st.plotly_chart(fig_e2, use_container_width=True)
        st.markdown("""<div class='insight-box'>
        <strong>Why Ordinal Encoding, not One-Hot?</strong> Education has a natural order
        (Some HS &lt; HS Grad &lt; Some College &lt; College+). One-Hot encoding would destroy this order.
        Higher education correlates with <em>lower</em> CHD risk — a trend that ordinal encoding preserves.</div>""",
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DIMENSIONALITY REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Dimensionality Reduction":
    st.markdown("<div class='hero-title'>Dimensionality Reduction</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Week 7 · Feature Selection & PCA</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🏆 Feature Selection", "🔭 PCA"])

    with tab1:
        st.markdown("<div class='section-header'>Random Forest Feature Importance</div>", unsafe_allow_html=True)
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": feat_imp})
        imp_df = imp_df.sort_values("Importance")
        colors_imp = [C_RED if v >= 0.10 else C_ORG if v >= 0.05 else "#4a5a7a"
                      for v in imp_df["Importance"]]
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
            marker_color=colors_imp,
            text=[f"{v:.3f}" for v in imp_df["Importance"]],
            textposition="outside"
        ))
        fig_imp.add_vline(x=0.10, line_dash="dash", line_color=C_RED)
        fig_imp.add_vline(x=0.05, line_dash="dash", line_color=C_ORG)
        fig_imp.update_layout(**CHART_LAYOUT, height=420,
                              xaxis=dict(gridcolor=C_GRID, title="Importance (Gini)"),
                              yaxis=dict(gridcolor=C_GRID))
        st.plotly_chart(fig_imp, use_container_width=True)

        # CHD rate by sysBP group
        st.markdown("<div class='section-header'>Top Feature Deep-Dive — sysBP vs CHD Risk</div>", unsafe_allow_html=True)
        orig2 = orig.copy()
        orig2["bp_group"] = pd.cut(orig2["sysBP"],
                                   bins=[80,120,130,140,160,300],
                                   labels=["Normal\n(<120)","Elevated\n(120–130)",
                                           "Stage 1\n(130–140)","Stage 2\n(140–160)","Crisis\n(>160)"])
        bp_chd = orig2.groupby("bp_group", observed=True)["TenYearCHD"].mean().reset_index()
        bp_chd["CHD %"] = (bp_chd["TenYearCHD"]*100).round(1)
        fig_bp = px.bar(bp_chd, x="bp_group", y="CHD %",
                        color="CHD %", color_continuous_scale=["#52b788","#f4a261","#e63946"],
                        text="CHD %")
        fig_bp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bp.update_layout(**CHART_LAYOUT, height=300,
                             coloraxis_showscale=False,
                             yaxis=dict(gridcolor=C_GRID, title="CHD Rate (%)"),
                             xaxis=dict(gridcolor=C_GRID, title="Blood Pressure Category"))
        st.plotly_chart(fig_bp, use_container_width=True)
        st.markdown("""<div class='insight-box'>
        CHD risk escalates sharply with blood pressure — from 8.6% in normal range to
        <strong>32.4%</strong> in hypertensive crisis. sysBP ranked as the top feature
        across all three selection methods (correlation, mutual information, RF importance).</div>""",
        unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-header'>PCA Explained Variance</div>", unsafe_allow_html=True)
        cum_var = np.cumsum(pca_evr)
        pca_labels = [f"PC{i+1}" for i in range(len(pca_evr))]

        fig_pca = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pca.add_trace(go.Bar(x=pca_labels, y=pca_evr*100,
                                 name="Individual %", marker_color=C_BLUE, opacity=0.8))
        fig_pca.add_trace(go.Scatter(x=pca_labels, y=cum_var*100,
                                     name="Cumulative %", mode="lines+markers",
                                     line=dict(color=C_RED, width=2.5),
                                     marker=dict(size=7)), secondary_y=True)
        fig_pca.add_hline(y=80, line_dash="dot", line_color=C_ORG,
                          annotation_text="80%", secondary_y=True)
        fig_pca.add_hline(y=95, line_dash="dot", line_color=C_RED,
                          annotation_text="95%", secondary_y=True)
        fig_pca.update_layout(**CHART_LAYOUT, height=360,
                              legend=dict(orientation="h", y=1.1),
                              yaxis=dict(gridcolor=C_GRID, title="Individual Variance (%)"),
                              yaxis2=dict(title="Cumulative Variance (%)"))
        st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown("<div class='section-header'>PCA 2D Patient Map</div>", unsafe_allow_html=True)
        pca_plot_df = pd.DataFrame({
            "PC1": pca_coords[:,0], "PC2": pca_coords[:,1],
            "CHD": sel["TenYearCHD"].map({0:"No CHD",1:"CHD+"})
        })
        fig_scatter = px.scatter(pca_plot_df, x="PC1", y="PC2", color="CHD",
                                 color_discrete_map={"No CHD": C_GRN, "CHD+": C_RED},
                                 opacity=0.45, size_max=6,
                                 title=f"PC1 ({pca_evr[0]*100:.1f}%) vs PC2 ({pca_evr[1]*100:.1f}%)")
        fig_scatter.update_traces(marker=dict(size=4))
        fig_scatter.update_layout(**CHART_LAYOUT, height=420,
                                  xaxis=dict(gridcolor=C_GRID),
                                  yaxis=dict(gridcolor=C_GRID),
                                  legend=dict(title=""))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("""<div class='insight-box'>
        <strong>PC1</strong> (29.7% variance) captures the <em>blood pressure burden</em>
        (sysBP, diaBP, prevalentHyp). <strong>PC2</strong> (19.7%) captures <em>metabolic aging</em>
        (age, glucose). CHD+ patients cluster toward the upper-right — higher on both dimensions.
        9 components preserve ≥95% of total variance.</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Risk Predictor":
    st.markdown("<div class='hero-title'>Interactive<br><i>Risk Predictor</i></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Enter patient values to estimate 10-Year CHD risk using the Random Forest model</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    X_train = sel.drop("TenYearCHD", axis=1)
    y_train = sel["TenYearCHD"]

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### Patient Parameters")
        age_v     = st.slider("Age (years)",      32, 70, 50)
        sysBP_v   = st.slider("Systolic BP (mmHg)", 83, 295, 130)
        diaBP_v   = st.slider("Diastolic BP (mmHg)", 48, 143, 82)
        glucose_v = st.slider("Glucose (mg/dL)",   40, 400, 78)
        BMI_v     = st.slider("BMI",               15.0, 57.0, 25.4, step=0.1)
        totChol_v = st.slider("Total Cholesterol (mg/dL)", 107, 700, 235)
        male_v    = st.selectbox("Sex", ["Female", "Male"])
        hyp_v     = st.selectbox("Hypertension", ["No", "Yes"])
        diab_v    = st.selectbox("Diabetes",     ["No", "Yes"])
        edu_v     = st.selectbox("Education", ["1 - Some High School","2 - HS Graduate","3 - Some College","4 - College+"])

    with col_result:
        from sklearn.preprocessing import RobustScaler

        # Build the same scaler on training data
        scaler_pred = RobustScaler()
        scaler_pred.fit(X_train[["sysBP","age","diaBP","BMI","glucose","totChol"]])

        male_int = 1 if male_v == "Male" else 0
        hyp_int  = 1 if hyp_v  == "Yes"  else 0
        diab_int = 1 if diab_v == "Yes"  else 0
        edu_int  = int(edu_v[0])

        raw_cont = np.array([[sysBP_v, age_v, diaBP_v, BMI_v, glucose_v, totChol_v]])
        scaled_cont = scaler_pred.transform(raw_cont)[0]

        patient = {
            "sysBP": scaled_cont[0], "age": scaled_cont[1],
            "diaBP": scaled_cont[2], "BMI": scaled_cont[3],
            "glucose": scaled_cont[4], "totChol": scaled_cont[5],
            "male": male_int, "prevalentHyp": hyp_int,
            "diabetes": diab_int, "education": edu_int
        }
        patient_df = pd.DataFrame([patient])[X_train.columns]
        prob = rf.predict_proba(patient_df)[0][1]

        # Gauge chart
        risk_pct = prob * 100
        if risk_pct < 10:
            risk_label, risk_color = "LOW", C_GRN
        elif risk_pct < 20:
            risk_label, risk_color = "MODERATE", C_ORG
        else:
            risk_label, risk_color = "HIGH", C_RED

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            number={"suffix": "%", "font": {"color": risk_color, "size": 48, "family": "DM Serif Display"}},
            delta={"reference": 15.2, "valueformat": ".1f",
                   "increasing": {"color": C_RED}, "decreasing": {"color": C_GRN}},
            gauge={
                "axis": {"range": [0, 60], "tickfont": {"color": "#7a8aaa"}},
                "bar": {"color": risk_color, "thickness": 0.3},
                "bgcolor": "#1e2d4a",
                "bordercolor": "#1e2d4a",
                "steps": [
                    {"range": [0, 10],  "color": "rgba(82,183,136,0.15)"},
                    {"range": [10, 20], "color": "rgba(244,162,97,0.15)"},
                    {"range": [20, 60], "color": "rgba(230,57,70,0.15)"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 2},
                              "thickness": 0.8, "value": 15.2}
            },
            title={"text": f"10-Year CHD Risk<br><span style='font-size:14px;color:{risk_color}'>"
                           f"● {risk_label} RISK</span>",
                   "font": {"color": "#e8eaf0", "size": 16}}
        ))
        fig_gauge.update_layout(**CHART_LAYOUT, height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature contribution
        st.markdown("#### Feature Contribution to This Prediction")
        contrib_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": rf.feature_importances_,
            "Value": patient_df.values[0]
        }).sort_values("Importance", ascending=False)

        fig_contrib = go.Figure(go.Bar(
            x=contrib_df["Importance"], y=contrib_df["Feature"],
            orientation="h",
            marker_color=[C_RED if v > contrib_df["Value"].median() else C_BLUE
                          for v in contrib_df["Value"]],
            text=[f"{v:.3f}" for v in contrib_df["Importance"]],
            textposition="outside"
        ))
        fig_contrib.update_layout(**CHART_LAYOUT, height=320,
                                  xaxis=dict(gridcolor=C_GRID, title="RF Importance"),
                                  yaxis=dict(gridcolor=C_GRID))
        st.plotly_chart(fig_contrib, use_container_width=True)

        st.markdown(f"""<div class='insight-box'>
        <strong>Model:</strong> Random Forest (100 trees, trained on 10 selected features)<br>
        <strong>Cohort baseline:</strong> 15.2% CHD rate (white line on gauge)<br>
        <strong>This patient:</strong> {risk_pct:.1f}% estimated 10-year CHD probability
        </div>""", unsafe_allow_html=True)

        st.caption("⚠️ For educational purposes only — not a clinical diagnostic tool.")
