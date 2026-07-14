"""
app.py
──────
Streamlit Dashboard — Student Performance Prediction
Premium dark-themed UI. No emojis. All text fully visible.

Run with:
  streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title            = "Student Performance Predictor",
    page_icon             = "S",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Variables ── */
:root {
    --bg          : #080812;
    --bg2         : #0e0e1c;
    --card        : #13132a;
    --card2       : #18183a;
    --border      : #252545;
    --border2     : #30306a;
    --violet      : #7c6dfa;
    --violet-soft : #9d92fb;
    --rose        : #f05f7a;
    --teal        : #3ecfa0;
    --amber       : #f6b73c;
    --text        : #ddddf5;
    --text-muted  : #7878a8;
    --text-dim    : #4a4a7a;
    --shadow-v    : 0 4px 32px rgba(124,109,250,0.18);
    --shadow-r    : 0 4px 24px rgba(240,95,122,0.15);
}

/* ── Reset ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0b1e 0%, #0e0e24 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Sidebar sliders — give them room so label+value don't collide ── */
[data-testid="stSidebar"] [data-testid="stSlider"] {
    margin-top: 2px !important;
    margin-bottom: 12px !important;
    padding-left: 18px !important;
    padding-right: 18px !important;
}

/* Stack label text and value on separate lines — prevents overlap */
[data-testid="stSidebar"] [data-testid="stSlider"] label {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] label > div {
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-between !important;
    width: 100% !important;
}

/* ── Sidebar sliders ── */
[data-testid="stSlider"] > div > div { padding: 0 !important; }
[data-testid="stSlider"] label {
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px !important;
    line-height: 1.6 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 8px !important;
    gap: 4px !important;
}
[data-testid="stTabs"] button {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: none !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 22px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.18s ease !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: var(--card) !important;
    color: var(--violet-soft) !important;
    border-bottom: 2px solid var(--violet) !important;
    font-weight: 700 !important;
}
[data-testid="stTabs"] button:hover {
    background: var(--card2) !important;
    color: var(--text) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid var(--violet) !important;
    color: var(--violet-soft) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--violet) !important;
    color: #fff !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--violet) 0%, var(--teal) 100%) !important;
    border-radius: 4px !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--card2) !important;
    border-radius: 4px !important;
    height: 8px !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; opacity: 1 !important; }

/* ────────────────────────────
   CUSTOM COMPONENT STYLES
   ──────────────────────────── */

/* Hero */
.hero-wrap {
    padding: 36px 0 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.hero-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--violet-soft);
    margin-bottom: 10px;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    line-height: 1.15;
    color: var(--text);
    margin-bottom: 10px;
    letter-spacing: -0.03em;
}
.hero-title span {
    background: linear-gradient(120deg, #7c6dfa 0%, #f05f7a 55%, #3ecfa0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--text-muted);
    line-height: 1.6;
}
.hero-tags {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 16px;
}
.hero-tag {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid var(--border2);
    color: var(--text-muted);
    background: var(--card);
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-v);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.violet::before { background: var(--violet); }
.kpi-card.rose::before   { background: var(--rose);   }
.kpi-card.teal::before   { background: var(--teal);   }
.kpi-card.amber::before  { background: var(--amber);  }
.kpi-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1;
    white-space: nowrap;
    overflow: visible;
}
.kpi-card.violet .kpi-value { color: var(--violet-soft); }
.kpi-card.rose   .kpi-value { color: var(--rose);        }
.kpi-card.teal   .kpi-value { color: var(--teal);        }
.kpi-card.amber  .kpi-value { color: var(--amber);       }
.kpi-sub {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 6px;
}

/* Section header */
.sec-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
    padding-left: 14px;
    border-left: 3px solid var(--violet);
    margin-bottom: 18px;
}

/* Grade result card */
.result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 32px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-card::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 18px;
    padding: 1px;
    background: linear-gradient(135deg, var(--violet), var(--rose));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}
.result-label {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 16px;
}
.result-grade {
    font-size: 7rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.06em;
    background: linear-gradient(135deg, #7c6dfa, #f05f7a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0);  }
    50%       { transform: translateY(-6px); }
}
.result-marks {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text);
    margin-top: 12px;
    letter-spacing: -0.03em;
}
.result-marks span { font-size: 1rem; font-weight: 400; color: var(--text-muted); }
.result-model {
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: 10px;
    font-style: italic;
}

/* Input summary card */
.summary-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
}
.input-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
}
.input-row:last-child { border-bottom: none; }
.input-key {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.input-val {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
}

/* Grade scale */
.grade-scale {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-top: 16px;
}
.grade-chip {
    border-radius: 10px;
    padding: 10px 6px;
    text-align: center;
    font-size: 0.8rem;
    font-weight: 700;
    line-height: 1.4;
}
.grade-chip .grade-letter {
    font-size: 1.3rem;
    font-weight: 900;
    display: block;
}
.grade-chip .grade-range {
    font-size: 0.68rem;
    font-weight: 500;
    opacity: 0.8;
    display: block;
}

/* Model card row */
.model-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid var(--border);
    background: var(--card);
    transition: border-color 0.18s ease;
}
.model-row:hover { border-color: var(--border2); }
.model-row.best { border-color: var(--violet); background: #1a1840; }
.model-rank {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-dim);
    width: 24px;
}
.model-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text);
    flex: 1;
    padding-left: 10px;
}
.model-row.best .model-name { color: var(--violet-soft); }
.model-badge {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 3px 9px;
    border-radius: 10px;
    background: var(--violet);
    color: #fff;
    margin-left: 8px;
}
.model-metrics {
    display: flex;
    gap: 16px;
}
.metric-chip {
    text-align: right;
}
.metric-chip .mc-label {
    font-size: 0.6rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: block;
}
.metric-chip .mc-val {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text);
}

/* Chart wrapper */
.chart-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
}
.chart-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 14px;
    padding-left: 10px;
    border-left: 2px solid var(--violet);
}

/* Sidebar sections */
.sb-section {
    padding: 20px 18px;
    border-bottom: 1px solid var(--border);
}
.sb-title {
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--violet-soft);
    margin-bottom: 14px;
}
.sb-stat {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 0.8rem;
    border-bottom: 1px solid var(--border);
}
.sb-stat:last-child { border-bottom: none; }
.sb-stat-key { color: var(--text-muted); }
.sb-stat-val { font-weight: 700; color: var(--text); }

/* Footer */
.footer {
    text-align: center;
    padding: 24px 0 12px;
    font-size: 0.75rem;
    color: var(--text-dim);
    border-top: 1px solid var(--border);
    margin-top: 32px;
    letter-spacing: 0.04em;
}
.footer strong { color: var(--text-muted); font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Matplotlib theme
# ═══════════════════════════════════════════════════════════════════════════════
BG      = "#080812"
SURFACE = "#13132a"
TXTC    = "#ddddf5"
PALETTE = ["#7c6dfa", "#f05f7a", "#3ecfa0", "#f6b73c"]

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : SURFACE,
    "axes.edgecolor"   : "#252545",
    "axes.labelcolor"  : TXTC,
    "xtick.color"      : TXTC,
    "ytick.color"      : TXTC,
    "text.color"       : TXTC,
    "grid.color"       : "#1e1e38",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "axes.titlepad"    : 14,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
})


def grade_color(g: str) -> str:
    return {"A": "#3ecfa0", "B": "#7c6dfa", "C": "#f6b73c", "D": "#f05f7a", "F": "#e03050"}.get(g, "#fff")


def grade_bg(g: str) -> str:
    return {"A": "#0d2e22", "B": "#16154a", "C": "#2e2210", "D": "#2e100e", "F": "#2a0a12"}.get(g, "#1a1a2e")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline (cached)
# ═══════════════════════════════════════════════════════════════════════════════

# Absolute root of the project (same directory as app.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner="Initialising ML pipeline — please wait…")
def load_pipeline():
    from data_generator import generate_dataset
    from preprocessing  import load_and_preprocess
    from train_models   import train_and_evaluate
    from evaluate       import build_comparison_table

    csv_path        = os.path.join(ROOT_DIR, "data", "student_data.csv")
    model_dir       = os.path.join(ROOT_DIR, "models")
    best_model_path = os.path.join(model_dir, "best_model.pkl")
    all_models_path = os.path.join(model_dir, "all_models.pkl")
    scaler_path     = os.path.join(model_dir, "scaler.pkl")
    imputer_path    = os.path.join(model_dir, "imputer.pkl")

    # ── Ensure dataset exists ─────────────────────────────────────────────────
    if not os.path.exists(csv_path):
        generate_dataset(save_path=csv_path)

    # ── Preprocess (always needed for df_clean, y_test, feature_names) ────────
    X_train, X_test, y_train, y_test, feature_names, scaler, df_clean = \
        load_and_preprocess(
            csv_path=csv_path,
            model_dir=model_dir,
        )

    # ── Load pre-trained models from disk (fast) — train only if missing ──────
    if os.path.exists(all_models_path) and os.path.exists(best_model_path):
        import joblib
        results    = joblib.load(all_models_path)
        best_model = joblib.load(best_model_path)
        # Determine best model name by R²
        best_name  = max(results, key=lambda k: results[k]["R2"])
    else:
        results, best_name, best_model = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            model_dir=model_dir,
        )

    comparison_df = build_comparison_table(results)

    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                feature_names=feature_names, scaler=scaler, df_clean=df_clean,
                results=results, best_name=best_name, best_model=best_model,
                comparison_df=comparison_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Chart factories
# ═══════════════════════════════════════════════════════════════════════════════
def fig_heatmap(df_clean):
    cols = ["attendance_pct", "study_hours", "assignment_score", "previous_marks", "final_grade"]
    labels = ["Attendance %", "Study Hours", "Assignment Score", "Previous Marks", "Final Grade"]
    corr = df_clean[cols].corr()
    corr.index   = labels
    corr.columns = labels
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(250, 10, s=90, l=40, as_cmap=True),
                linewidths=0.6, linecolor=BG, ax=ax,
                annot_kws={"size": 9.5, "weight": "bold", "color": TXTC},
                cbar_kws={"shrink": 0.75})
    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold", color=TXTC, pad=14)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    ax.tick_params(length=0)
    plt.tight_layout(pad=1.5)
    return fig


def fig_importance(results, feature_names):
    preferred  = ["Gradient Boosting", "Random Forest", "Decision Tree"]
    model_name = next((m for m in preferred if m in results), list(results.keys())[0])
    model      = results[model_name]["model"]
    labels_map = {
        "attendance_pct": "Attendance %", "study_hours": "Study Hours",
        "assignment_score": "Assignment Score", "previous_marks": "Previous Marks",
    }
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    else:
        imps = np.abs(model.coef_); imps = imps / imps.sum()

    feat_df = (pd.DataFrame({"Feature": feature_names, "Importance": imps})
               .sort_values("Importance", ascending=True))
    feat_df["Feature"] = feat_df["Feature"].map(labels_map)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    colors = [PALETTE[i % 4] for i in range(len(feat_df))]
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
                   color=colors, edgecolor="none", height=0.48)
    for bar, val in zip(bars, feat_df["Importance"]):
        ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9.5,
                color=TXTC, fontweight="bold")
    ax.set_xlim(0, max(feat_df["Importance"]) * 1.3)
    ax.set_title(f"Feature Importance   ({model_name})", fontsize=12, fontweight="bold", color=TXTC)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout(pad=1.4)
    return fig


def fig_actual_vs_pred(results, y_test, best_name):
    y_pred = results[best_name]["y_pred"]
    r2     = results[best_name]["R2"]
    fig, ax = plt.subplots(figsize=(6, 5.4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    ax.scatter(y_test, y_pred, c=PALETTE[0], alpha=0.55, edgecolors="none", s=36, zorder=3)
    lo = min(y_test.min(), y_pred.min()) - 2
    hi = max(y_test.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], color=PALETTE[1], lw=2, linestyle="--",
            label="Perfect Fit Line", zorder=2)
    ax.set_xlabel("Actual Marks", fontsize=10)
    ax.set_ylabel("Predicted Marks", fontsize=10)
    ax.set_title(f"Actual vs Predicted Marks   (R\u00b2 = {r2:.4f})", fontsize=12,
                 fontweight="bold", color=TXTC)
    ax.legend(framealpha=0.15, facecolor=SURFACE, edgecolor="#252545", fontsize=9)
    ax.grid(alpha=0.35); ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout(pad=1.4)
    return fig


def fig_model_comparison(comparison_df):
    models  = comparison_df["Model"].tolist()
    x       = np.arange(len(models))
    width   = 0.22
    metrics = ["MAE", "RMSE", "R\u00b2"]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)

    for i, (metric, color) in enumerate(zip(metrics, PALETTE)):
        col_key = "R\u00b2" if metric == "R\u00b2" else metric
        vals    = comparison_df[col_key].values
        offset  = (i - 1) * width
        bars    = ax.bar(x + offset, vals, width, label=metric,
                         color=color, alpha=0.88, edgecolor="none")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8, color=TXTC, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylabel("Score / Error Value", fontsize=10)
    ax.set_title("ML Model Comparison   (MAE  |  RMSE  |  R\u00b2)", fontsize=12,
                 fontweight="bold", color=TXTC)
    ax.legend(framealpha=0.15, facecolor=SURFACE, edgecolor="#252545", fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout(pad=1.4)
    return fig


def fig_distributions(df_clean):
    feature_info = {
        "attendance_pct"  : ("Attendance %",      PALETTE[0]),
        "study_hours"     : ("Study Hours / Day", PALETTE[1]),
        "assignment_score": ("Assignment Score",  PALETTE[2]),
        "previous_marks"  : ("Previous Marks",    PALETTE[3]),
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    fig.patch.set_facecolor(BG)
    for ax, (col, (label, color)) in zip(axes, feature_info.items()):
        ax.set_facecolor(SURFACE)
        data = df_clean[col].dropna()
        ax.hist(data, bins=26, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(data.mean(), color="#fff", lw=1.4, linestyle="--", alpha=0.7)
        ax.set_title(label, fontsize=10.5, fontweight="bold", color=TXTC)
        ax.set_xlabel("Value", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3); ax.tick_params(length=0)
    plt.suptitle("Feature Distributions  (dashed line = mean)",
                 fontsize=12, fontweight="bold", color=TXTC, y=1.04)
    plt.tight_layout(pad=1.2)
    return fig


def fig_grade_dist(df_clean):
    counts = df_clean["letter_grade"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    colors = [grade_color(g) for g in counts.index]
    bars   = ax.bar(counts.index, counts.values, color=colors,
                    edgecolor="none", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=11,
                color=TXTC, fontweight="bold")
    ax.set_title("Grade Distribution", fontsize=12, fontweight="bold", color=TXTC)
    ax.set_xlabel("Letter Grade", fontsize=10)
    ax.set_ylabel("Number of Students", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3); ax.tick_params(length=0)
    plt.tight_layout(pad=1.4)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
data          = load_pipeline()
scaler        = data["scaler"]
results       = data["results"]
best_name     = data["best_name"]
best_model    = data["best_model"]
comparison_df = data["comparison_df"]
df_clean      = data["df_clean"]
y_test        = data["y_test"]
feature_names = data["feature_names"]

from predict import score_to_grade

best_r2   = float(comparison_df["R\u00b2"].max())
best_rmse = float(comparison_df.loc[comparison_df["Model"] == best_name, "RMSE"].values[0])
best_mae  = float(comparison_df.loc[comparison_df["Model"] == best_name, "MAE"].values[0])
n_students = len(df_clean)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="sb-section" style="background:linear-gradient(160deg,#16154a,#0e0e1c);
         border-bottom:1px solid #252545; padding:28px 18px;">
        <div style="font-size:0.62rem; font-weight:800; letter-spacing:0.18em;
                    text-transform:uppercase; color:#7c6dfa; margin-bottom:10px;">
            Student Performance
        </div>
        <div style="font-size:1.35rem; font-weight:800; color:#ddddf5;
                    letter-spacing:-0.02em; line-height:1.2;">
            Grade Predictor
        </div>
        <div style="font-size:0.75rem; color:#7878a8; margin-top:6px;">
            Machine Learning — Powered Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 20px 18px 4px 18px; border-bottom: 1px solid #252545;">
        <div style="font-size:0.65rem; font-weight:800; letter-spacing:0.18em;
                    text-transform:uppercase; color:#9d92fb; margin-bottom:14px;">
            Input Parameters
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    st.markdown("""<div style='font-size:0.72rem;font-weight:700;color:#7878a8;
        text-transform:uppercase;letter-spacing:0.08em;
        margin:14px 0 4px;'>Attendance %</div>""", unsafe_allow_html=True)
    attendance = st.slider("Attendance %", min_value=40.0, max_value=100.0,
                           value=85.0, step=0.5, format="%.1f%%",
                           label_visibility="collapsed")

    st.markdown("""<div style='font-size:0.72rem;font-weight:700;color:#7878a8;
        text-transform:uppercase;letter-spacing:0.08em;
        margin:14px 0 4px;'>Study Hours / Day</div>""", unsafe_allow_html=True)
    study_hrs  = st.slider("Study Hours / Day", min_value=0.5, max_value=10.0,
                           value=4.0, step=0.5, format="%.1f hrs",
                           label_visibility="collapsed")

    st.markdown("""<div style='font-size:0.72rem;font-weight:700;color:#7878a8;
        text-transform:uppercase;letter-spacing:0.08em;
        margin:14px 0 4px;'>Assignment Score</div>""", unsafe_allow_html=True)
    assign_sc  = st.slider("Assignment Score", min_value=0.0, max_value=100.0,
                           value=80.0, step=1.0, format="%.0f",
                           label_visibility="collapsed")

    st.markdown("""<div style='font-size:0.72rem;font-weight:700;color:#7878a8;
        text-transform:uppercase;letter-spacing:0.08em;
        margin:14px 0 4px;'>Previous Marks</div>""", unsafe_allow_html=True)
    prev_marks = st.slider("Previous Marks", min_value=20.0, max_value=100.0,
                           value=75.0, step=1.0, format="%.0f",
                           label_visibility="collapsed")

    st.markdown("""<div style="height:12px;"></div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-section">
        <div class="sb-title">System Information</div>
        <div class="sb-stat">
            <span class="sb-stat-key">Best Model</span>
            <span class="sb-stat-val">{best_name}</span>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-key">R&#178; Score</span>
            <span class="sb-stat-val">{best_r2:.4f}</span>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-key">Dataset Size</span>
            <span class="sb-stat-val">{n_students} Students</span>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-key">Features Used</span>
            <span class="sb-stat-val">4 Factors</span>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-key">Models Trained</span>
            <span class="sb-stat-val">4 Algorithms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">Predictive Analytics Platform</div>
    <div class="hero-title">Student Performance <span>Prediction</span></div>
    <div class="hero-sub">
        A machine learning system that forecasts student academic outcomes
        using attendance, study patterns, assignment performance, and academic history.
    </div>
    <div class="hero-tags">
        <span class="hero-tag">Linear Regression</span>
        <span class="hero-tag">Decision Tree</span>
        <span class="hero-tag">Random Forest</span>
        <span class="hero-tag">Gradient Boosting</span>
        <span class="hero-tag">Scikit-learn</span>
        <span class="hero-tag">Python</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# KPI ROW — custom cards (no st.metric truncation)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card violet">
        <div class="kpi-label">Best Performing Model</div>
        <div class="kpi-value">{best_name}</div>
        <div class="kpi-sub">Ranked by R&#178; score across all models</div>
    </div>
    <div class="kpi-card teal">
        <div class="kpi-label">R&#178; Score</div>
        <div class="kpi-value">{best_r2:.4f}</div>
        <div class="kpi-sub">Coefficient of determination</div>
    </div>
    <div class="kpi-card rose">
        <div class="kpi-label">Root Mean Squared Error</div>
        <div class="kpi-value">{best_rmse:.3f}</div>
        <div class="kpi-sub">Average prediction deviation</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-label">Training Dataset</div>
        <div class="kpi-value">{n_students} Students</div>
        <div class="kpi-sub">80% train — 20% test split</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "  Grade Prediction  ",
    "  Model Comparison  ",
    "  Visualizations    ",
    "  Dataset Explorer  ",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    X_raw      = np.array([[attendance, study_hrs, assign_sc, prev_marks]])
    X_scaled   = scaler.transform(X_raw)
    pred_marks = float(np.clip(best_model.predict(X_scaled)[0], 0, 100))
    pred_grade = score_to_grade(pred_marks)
    gc         = grade_color(pred_grade)
    gbg        = grade_bg(pred_grade)

    col_result, col_inputs = st.columns([1, 1], gap="large")

    with col_result:
        st.markdown("<div class='sec-header'>Prediction Result</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Grade</div>
            <div class="result-grade" style="
                background: linear-gradient(135deg, {gc}, {gc}99);
                -webkit-background-clip: text;
                background-clip: text;">
                {pred_grade}
            </div>
            <div class="result-marks">
                {pred_marks:.1f} <span>out of 100</span>
            </div>
            <div class="result-model">Predicted using {best_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-bottom:6px; font-size:0.78rem; font-weight:700;
                    letter-spacing:0.1em; text-transform:uppercase; color:#7878a8;">
            Performance Score &nbsp;—&nbsp; {pred_marks:.1f} / 100
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(pred_marks))

        # Grade scale
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.12em;
                    text-transform:uppercase; color:#7878a8; margin-bottom:10px;">
            Grade Scale Reference
        </div>
        """, unsafe_allow_html=True)
        grade_scale = [
            ("A", "85 - 100", "#3ecfa0", "#0d2e22"),
            ("B", "70 - 84",  "#7c6dfa", "#16154a"),
            ("C", "55 - 69",  "#f6b73c", "#2e2210"),
            ("D", "40 - 54",  "#f05f7a", "#2e100e"),
            ("F", "Below 40", "#e03050", "#2a0a12"),
        ]
        g_cols = st.columns(5)
        for col, (letter, rng, color, bg) in zip(g_cols, grade_scale):
            col.markdown(f"""
            <div class="grade-chip" style="background:{bg}; border:1px solid {color}44;">
                <span class="grade-letter" style="color:{color};">{letter}</span>
                <span class="grade-range" style="color:{color}cc;">{rng}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_inputs:
        st.markdown("<div class='sec-header'>Input Summary</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-card">
            <div class="input-row">
                <span class="input-key">Attendance Percentage</span>
                <span class="input-val">{attendance:.1f}%</span>
            </div>
            <div class="input-row">
                <span class="input-key">Daily Study Hours</span>
                <span class="input-val">{study_hrs:.1f} hours</span>
            </div>
            <div class="input-row">
                <span class="input-key">Assignment Score</span>
                <span class="input-val">{assign_sc:.0f} / 100</span>
            </div>
            <div class="input-row">
                <span class="input-key">Previous Semester Marks</span>
                <span class="input-val">{prev_marks:.0f} / 100</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sec-header'>How the Prediction Works</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-card" style="font-size:0.85rem; line-height:1.7;
             color:#7878a8;">
            <p style="margin:0 0 10px;">
                Your input values are scaled using a fitted
                <strong style="color:#ddddf5;">StandardScaler</strong> and fed into
                the best-performing model selected after comparing all four algorithms
                on a held-out 20% test set.
            </p>
            <p style="margin:0 0 10px;">
                The best model for this dataset is
                <strong style="color:#9d92fb;">{best_name}</strong>, which achieved
                an R&#178; score of <strong style="color:#3ecfa0;">{best_r2:.4f}</strong>
                and an RMSE of <strong style="color:#f05f7a;">{best_rmse:.3f}</strong>
                marks on the test set.
            </p>
            <p style="margin:0;">
                Features used: Attendance %, Study Hours, Assignment Score,
                and Previous Semester Marks — all four contribute significantly
                to the final prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-header'>All Models — Performance Metrics</div>", unsafe_allow_html=True)

    # Model cards
    ranks = ["1st", "2nd", "3rd", "4th"]
    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        is_best  = (row["Model"] == best_name)
        best_cls = "best" if is_best else ""
        badge    = '<span class="model-badge">Best Model</span>' if is_best else ""
        st.markdown(f"""
        <div class="model-row {best_cls}">
            <div class="model-rank">{ranks[idx]}</div>
            <div class="model-name">{row["Model"]}{badge}</div>
            <div class="model-metrics">
                <div class="metric-chip">
                    <span class="mc-label">MAE</span>
                    <span class="mc-val">{row["MAE"]:.4f}</span>
                </div>
                <div class="metric-chip">
                    <span class="mc-label">MSE</span>
                    <span class="mc-val">{row["MSE"]:.4f}</span>
                </div>
                <div class="metric-chip">
                    <span class="mc-label">RMSE</span>
                    <span class="mc-val">{row["RMSE"]:.4f}</span>
                </div>
                <div class="metric-chip">
                    <span class="mc-label">R&#178; Score</span>
                    <span class="mc-val" style="color:{'#3ecfa0' if is_best else '#ddddf5'};">
                        {row["R\u00b2"]:.4f}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-header'>Visual Comparison Chart</div>", unsafe_allow_html=True)
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    f = fig_model_comparison(comparison_df)
    st.pyplot(f, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # Metric explanation
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    exp_c1, exp_c2, exp_c3 = st.columns(3)
    for col, (metric, desc, color) in zip(
        [exp_c1, exp_c2, exp_c3],
        [
            ("MAE — Mean Absolute Error",
             "Average absolute difference between predicted and actual marks. Lower is better.",
             "#f05f7a"),
            ("RMSE — Root Mean Squared Error",
             "Square root of the average squared differences. Penalises large errors more. Lower is better.",
             "#f6b73c"),
            ("R\u00b2 — Coefficient of Determination",
             "Proportion of variance explained by the model. Ranges 0 to 1. Higher is better.",
             "#3ecfa0"),
        ]
    ):
        col.markdown(f"""
        <div class="summary-card" style="font-size:0.82rem;">
            <div style="font-size:0.75rem; font-weight:700; color:{color};
                        margin-bottom:6px; text-transform:uppercase; letter-spacing:0.06em;">
                {metric}
            </div>
            <div style="color:#7878a8; line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2, gap="large")
    with r1c1:
        st.markdown('<div class="chart-wrap"><div class="chart-title">Feature Correlation Matrix</div>',
                    unsafe_allow_html=True)
        f = fig_heatmap(df_clean)
        st.pyplot(f, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="chart-wrap"><div class="chart-title">Feature Importance</div>',
                    unsafe_allow_html=True)
        f = fig_importance(results, feature_names)
        st.pyplot(f, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2, gap="large")
    with r2c1:
        st.markdown('<div class="chart-wrap"><div class="chart-title">Actual vs Predicted Grades</div>',
                    unsafe_allow_html=True)
        f = fig_actual_vs_pred(results, y_test, best_name)
        st.pyplot(f, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="chart-wrap"><div class="chart-title">Grade Distribution</div>',
                    unsafe_allow_html=True)
        f = fig_grade_dist(df_clean)
        st.pyplot(f, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-wrap"><div class="chart-title">Feature Distributions</div>',
                unsafe_allow_html=True)
    f = fig_distributions(df_clean)
    st.pyplot(f, use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DATASET
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-header'>Student Dataset Explorer</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="summary-card" style="margin-bottom:16px; display:flex;
         justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div style="font-size:0.85rem; color:#7878a8;">
            Showing records from the synthetic student dataset used for model training and evaluation.
        </div>
        <div style="font-size:0.82rem; color:#ddddf5; font-weight:700;">
            {n_students} total records &nbsp;|&nbsp; 7 columns
        </div>
    </div>
    """, unsafe_allow_html=True)

    grades  = ["All Grades"] + sorted(df_clean["letter_grade"].unique().tolist())
    sel_gr  = st.selectbox("Filter by Letter Grade:", grades)
    df_show = df_clean if sel_gr == "All Grades" else df_clean[df_clean["letter_grade"] == sel_gr]

    col_rename = {
        "student_id"      : "Student ID",
        "attendance_pct"  : "Attendance %",
        "study_hours"     : "Study Hours",
        "assignment_score": "Assignment Score",
        "previous_marks"  : "Previous Marks",
        "final_grade"     : "Final Grade",
        "letter_grade"    : "Letter Grade",
    }
    st.dataframe(
        df_show[list(col_rename.keys())].rename(columns=col_rename),
        use_container_width=True,
        height=360,
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-header'>Descriptive Statistics</div>", unsafe_allow_html=True)
    desc = (df_clean[["attendance_pct", "study_hours", "assignment_score",
                       "previous_marks", "final_grade"]]
            .rename(columns={
                "attendance_pct": "Attendance %", "study_hours": "Study Hours",
                "assignment_score": "Assignment Score", "previous_marks": "Previous Marks",
                "final_grade": "Final Grade"})
            .describe().round(2))
    st.dataframe(desc, use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    csv_bytes = df_clean.to_csv(index=False).encode()
    st.download_button(
        label     = "Download Full Dataset as CSV",
        data      = csv_bytes,
        file_name = "student_performance_dataset.csv",
        mime      = "text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <strong>Student Performance Prediction System</strong> &nbsp;|&nbsp;
    Built with Python, Scikit-learn, and Streamlit &nbsp;|&nbsp;
    Models: Linear Regression &middot; Decision Tree &middot; Random Forest &middot; Gradient Boosting
    &nbsp;|&nbsp; Best Model: {best_name} (R&#178; = {best_r2:.4f})
</div>
""", unsafe_allow_html=True)
