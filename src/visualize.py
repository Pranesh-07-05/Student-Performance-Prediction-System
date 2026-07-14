"""
visualize.py
────────────
Generates and saves four publication-quality charts to outputs/.

Charts
------
1. Correlation Heatmap
2. Feature Importance (Random Forest / Gradient Boosting)
3. Actual vs Predicted Scatter Plot
4. Model Comparison Bar Chart
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
PALETTE      = ["#6C63FF", "#FF6584", "#43C59E", "#F9A825"]
BACKGROUND   = "#0F0F1A"
SURFACE      = "#1C1C2E"
TEXT_COLOR   = "#E8E8F0"
ACCENT       = "#6C63FF"

plt.rcParams.update({
    "figure.facecolor" : BACKGROUND,
    "axes.facecolor"   : SURFACE,
    "axes.edgecolor"   : "#3A3A5C",
    "axes.labelcolor"  : TEXT_COLOR,
    "xtick.color"      : TEXT_COLOR,
    "ytick.color"      : TEXT_COLOR,
    "text.color"       : TEXT_COLOR,
    "grid.color"       : "#2A2A44",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "font.family"      : "DejaVu Sans",
    "axes.titlepad"    : 12,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df_clean: pd.DataFrame, save: bool = True) -> str:
    """
    Seaborn heatmap of pairwise Pearson correlations among numeric features.
    """
    numeric_cols = ["attendance_pct", "study_hours", "assignment_score",
                    "previous_marks", "final_grade"]
    corr = df_clean[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle only
    sns.heatmap(
        corr, mask=False,
        annot=True, fmt=".2f",
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        linewidths=0.5, linecolor="#0F0F1A",
        ax=ax,
        annot_kws={"size": 10, "color": TEXT_COLOR},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
        print(f"[visualize] Saved → {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(results: dict, feature_names: list, save: bool = True) -> str:
    """
    Horizontal bar chart of feature importances from tree-based models.
    Falls back to Linear Regression coefficients if no tree model is present.
    """
    preferred = ["Gradient Boosting", "Random Forest", "Decision Tree"]
    model_name = next((m for m in preferred if m in results), list(results.keys())[0])
    model = results[model_name]["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        kind = "Feature Importance"
    else:
        importances = np.abs(model.coef_)
        importances /= importances.sum()
        kind = "Abs. Coefficient (Normalized)"

    feat_df = pd.DataFrame({
        "Feature"    : feature_names,
        "Importance" : importances,
    }).sort_values("Importance")

    labels = {
        "attendance_pct"  : "Attendance %",
        "study_hours"     : "Study Hours",
        "assignment_score": "Assignment Score",
        "previous_marks"  : "Previous Marks",
    }
    feat_df["Feature"] = feat_df["Feature"].map(labels).fillna(feat_df["Feature"])

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(feat_df))]
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
                   color=colors, edgecolor="none", height=0.55)

    # value labels
    for bar, val in zip(bars, feat_df["Importance"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=10, color=TEXT_COLOR)

    ax.set_xlabel(kind)
    ax.set_title(f"Feature Importance  ({model_name})", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR)
    ax.set_xlim(0, max(feat_df["Importance"]) * 1.25)
    ax.grid(axis="x", alpha=0.4)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
        print(f"[visualize] Saved → {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Actual vs Predicted Scatter
# ─────────────────────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(results: dict, y_test: np.ndarray,
                              best_name: str, save: bool = True) -> str:
    """
    Scatter plot of actual vs predicted grades for the best model.
    """
    y_pred = results[best_name]["y_pred"]
    r2     = results[best_name]["R2"]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    # scatter
    ax.scatter(y_test, y_pred, c=ACCENT, alpha=0.55, edgecolors="none", s=40,
               label="Predictions")

    # perfect-fit line
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#FF6584", lw=2, linestyle="--",
            label="Perfect Fit")

    ax.set_xlabel("Actual Grade (Marks)", fontsize=11)
    ax.set_ylabel("Predicted Grade (Marks)", fontsize=11)
    ax.set_title(f"Actual vs Predicted  |  R2 = {r2:.4f}  ({best_name})",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.legend(framealpha=0.2, facecolor=SURFACE, edgecolor="#3A3A5C")
    ax.grid(alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
        print(f"[visualize] Saved → {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Comparison Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(comparison_df: pd.DataFrame, save: bool = True) -> str:
    """
    Grouped bar chart comparing MAE, RMSE, and R² across all models.
    """
    models  = comparison_df["Model"].tolist()
    metrics = ["MAE", "RMSE", "R²"]
    x       = np.arange(len(models))
    width   = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    for i, (metric, color) in enumerate(zip(metrics, PALETTE)):
        vals = comparison_df[metric].tolist() if metric in comparison_df.columns else \
               comparison_df["R²"].tolist()
        offset = (i - 1) * width
        bars = ax.bar(x + offset, comparison_df[metric if metric in comparison_df.columns else "R²"],
                      width, label=metric, color=color, alpha=0.85, edgecolor="none")
        # value labels on top
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=10)
    ax.set_ylabel("Score / Error")
    ax.set_title("Model Comparison  (MAE  |  RMSE  |  R2)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax.legend(framealpha=0.2, facecolor=SURFACE, edgecolor="#3A3A5C")
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
        print(f"[visualize] Saved → {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate ALL charts at once
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_charts(df_clean, results, feature_names, y_test,
                         best_name, comparison_df) -> dict:
    return {
        "heatmap"           : plot_correlation_heatmap(df_clean),
        "feature_importance": plot_feature_importance(results, feature_names),
        "actual_vs_pred"    : plot_actual_vs_predicted(results, y_test, best_name),
        "model_comparison"  : plot_model_comparison(comparison_df),
    }
