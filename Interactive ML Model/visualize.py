"""
visualize.py
------------
Generates all figures from the research paper's data and model predictions.

Figures produced
----------------
  fig1_arctic_sie_trend.png       — Arctic SIE decline 1979–2024 (Table 2)
  fig2_sie_ismr_correlation.png   — Arctic SIE vs ISMR anomaly scatter
  fig3_slr_projections.png        — SLR by SSP scenario (Table 6)
  fig4_crop_yield_heatmap.png     — Crop yield loss heatmap (Table 5)
  fig5_india_warming_scenarios.png— India warming trajectories (Table 6)
  fig6_feature_importance.png     — Model feature importances

Run:  python visualize.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from data import (ARCTIC_SIE, ISMR_ANOMALY_SERIES, SEA_LEVEL_OBS,
                  CROP_YIELD, SSP_SCENARIOS, generate_training_data,
                  FEATURE_COLS, TARGET_COLS, TARGET_LABELS)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0a0e1a",
    "axes.facecolor":    "#111827",
    "axes.edgecolor":    "#2d3748",
    "axes.labelcolor":   "#a0aec0",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#718096",
    "ytick.color":       "#718096",
    "grid.color":        "#1a2235",
    "grid.linewidth":    0.6,
    "text.color":        "#e2e8f0",
    "font.family":       "monospace",
    "figure.dpi":        130,
})

ACCENT  = "#63b3ed"
AMBER   = "#f6ad55"
GREEN   = "#68d391"
RED     = "#fc8181"
MUTED   = "#4a5568"

SSP_COLORS = {1: GREEN, 2: ACCENT, 3: AMBER, 4: RED}
SSP_NAMES  = {1: "SSP1-2.6", 2: "SSP2-4.5", 3: "SSP3-7.0", 4: "SSP5-8.5"}

def save(fig, name):
    fig.savefig(name, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {name}")

# ---------------------------------------------------------------------------
# Fig 1 — Arctic SIE Trend
# ---------------------------------------------------------------------------
def fig1_arctic_sie():
    df = ARCTIC_SIE.copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df.year, df.sie_mkm2, color=ACCENT, lw=1.8, zorder=3)
    ax.scatter(df.year, df.sie_mkm2, color=ACCENT, s=40, zorder=4)
    ax.scatter([2023], [3.74], color=RED, s=80, zorder=5, label="2023 record low")
    # Trend line
    z = np.polyfit(df.year, df.sie_mkm2, 1)
    p = np.poly1d(z)
    ax.plot(df.year, p(df.year), "--", color=AMBER, lw=1, alpha=0.7, label=f"Trend: {z[0]*10:.3f} Mkm²/decade")
    ax.fill_between(df.year, df.sie_mkm2, 2.5, alpha=0.08, color=ACCENT)
    ax.axhline(5.76, color=MUTED, lw=0.8, ls=":", label="1981–2010 climatological avg.")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sept. SIE (Mkm²)")
    ax.set_title("Arctic September Sea Ice Extent — 1979 to 2024\n"
                 "Source: NSIDC Sea Ice Index v3.0", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.15, labelcolor="white")
    ax.grid(True, axis="y")
    ax.set_xlim(1977, 2026)
    fig.tight_layout()
    save(fig, "fig1_arctic_sie_trend.png")

# ---------------------------------------------------------------------------
# Fig 2 — SIE vs ISMR Correlation
# ---------------------------------------------------------------------------
def fig2_sie_ismr():
    sie_df  = ARCTIC_SIE.set_index("year")
    ismr_df = ISMR_ANOMALY_SERIES.set_index("year")
    merged  = sie_df.join(ismr_df).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: time series dual axis
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(merged.index, merged.sie_mkm2, color=ACCENT, lw=1.8, label="SIE (left)")
    ax2.plot(merged.index, merged.ismr_anom, color=AMBER, lw=1.5, ls="--", label="ISMR anom. (right)")
    ax.set_xlabel("Year")
    ax.set_ylabel("SIE (Mkm²)", color=ACCENT)
    ax2.set_ylabel("ISMR anomaly (%)", color=AMBER)
    ax.tick_params(axis="y", colors=ACCENT)
    ax2.tick_params(axis="y", colors=AMBER)
    ax.set_title("SIE vs ISMR — Observed\n1979–2024", fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8, framealpha=0.15, labelcolor="white")
    ax.grid(True, axis="y")

    # Right: scatter + regression
    ax = axes[1]
    r = np.corrcoef(merged.sie_mkm2, merged.ismr_anom)[0, 1]
    ax.scatter(merged.sie_mkm2, merged.ismr_anom, color=ACCENT, s=55, alpha=0.85, zorder=3)
    z = np.polyfit(merged.sie_mkm2, merged.ismr_anom, 1)
    xs = np.linspace(merged.sie_mkm2.min()-0.2, merged.sie_mkm2.max()+0.2, 100)
    ax.plot(xs, np.poly1d(z)(xs), color=RED, lw=1.5, ls="--")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Arctic SIE (Mkm²)")
    ax.set_ylabel("ISMR anomaly (%)")
    ax.set_title(f"Scatter: r = {r:.3f}\n(Kulkarni & Agarwal 2024)", fontsize=10)
    ax.grid(True)
    ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=11, color=RED, fontweight="bold")

    fig.suptitle("Arctic Sea Ice ↔ Indian Summer Monsoon Teleconnection",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_sie_ismr_correlation.png")

# ---------------------------------------------------------------------------
# Fig 3 — SLR Projections
# ---------------------------------------------------------------------------
def fig3_slr():
    years = np.linspace(2024, 2100, 80)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    slr_rates = {1: 0.30, 2: 0.45, 3: 0.65, 4: 0.90}
    for ssp, slr_end in slr_rates.items():
        vals = slr_end * (years - 2024) / 76 * 100  # cm
        ax.plot(years, vals, color=SSP_COLORS[ssp], lw=2, label=SSP_NAMES[ssp])
        ax.fill_between(years, vals * 0.75, vals * 1.35,
                        alpha=0.07, color=SSP_COLORS[ssp])
    # Observed rate context
    ax.axvline(2024, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea level rise (cm) vs 2024")
    ax.set_title("Indian Ocean Sea Level Rise Projections by SSP Scenario\n"
                 "Source: IPCC AR6 — Table 6; Sadai & Karmalkar (2025)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.15, labelcolor="white")
    ax.grid(True)
    fig.tight_layout()
    save(fig, "fig3_slr_projections.png")

# ---------------------------------------------------------------------------
# Fig 4 — Crop Yield Heatmap
# ---------------------------------------------------------------------------
def fig4_crop_yield():
    df = CROP_YIELD.set_index("crop")[["yield_1p5C", "yield_2C", "yield_4C"]]
    df.columns = ["+1.5°C", "+2°C", "+4°C"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(
        df, annot=True, fmt=".1f", ax=ax,
        cmap="RdYlGn", center=0, vmin=-35, vmax=5,
        linewidths=0.5, linecolor="#0a0e1a",
        annot_kws={"size": 10, "color": "white", "weight": "bold"},
        cbar_kws={"label": "Yield change (%)"},
    )
    ax.set_title("Projected Crop Yield Changes Under Warming Scenarios (%)\n"
                 "Source: NICRA/ICAR — Table 5; IPCC AR6; World Bank (2013)", fontsize=11)
    ax.set_xlabel("Warming scenario")
    ax.set_ylabel("")
    ax.tick_params(axis="x", colors="#e2e8f0")
    ax.tick_params(axis="y", colors="#e2e8f0", rotation=0)
    fig.tight_layout()
    save(fig, "fig4_crop_yield_heatmap.png")

# ---------------------------------------------------------------------------
# Fig 5 — India Warming Trajectories
# ---------------------------------------------------------------------------
def fig5_warming():
    years = np.linspace(2024, 2100, 80)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    warm_end = {1: 1.50, 2: 2.25, 3: 3.40, 4: 4.25}
    for ssp, wend in warm_end.items():
        vals = wend * (years - 2024) / 76
        ax.plot(years, vals, color=SSP_COLORS[ssp], lw=2, label=SSP_NAMES[ssp])
        ax.fill_between(years, vals * 0.82, vals * 1.20,
                        alpha=0.07, color=SSP_COLORS[ssp])
    ax.axhline(1.5, color="white", lw=0.6, ls="--", alpha=0.4, label="Paris 1.5°C target")
    ax.axhline(2.0, color=AMBER,   lw=0.6, ls="--", alpha=0.4, label="Paris 2.0°C target")
    ax.set_xlabel("Year")
    ax.set_ylabel("India mean warming (°C vs 1995–2014)")
    ax.set_title("India Temperature Projections — CMIP6 Multi-model Ensemble\n"
                 "Source: IPCC AR6 Table 6; MoES Assessment (2020/2025)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.15, labelcolor="white")
    ax.grid(True)
    fig.tight_layout()
    save(fig, "fig5_india_warming_scenarios.png")

# ---------------------------------------------------------------------------
# Fig 6 — Feature Importances
# ---------------------------------------------------------------------------
def fig6_feature_importance():
    with open("india_climate_model.pkl", "rb") as f:
        pipeline = pickle.load(f)

    importances = np.mean(
        [est.feature_importances_
         for est in pipeline.named_steps["model"].estimators_],
        axis=0,
    )
    feat_labels = {
        "sie":       "Arctic SIE",
        "ant_melt":  "Antarctic melt",
        "sst_anom":  "Indian Ocean SST",
        "ao_phase":  "AO phase",
        "year":      "Target year",
        "ssp":       "SSP scenario",
    }
    labels = [feat_labels[f] for f in FEATURE_COLS]
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh([labels[i] for i in idx], importances[idx],
                   color=ACCENT, height=0.6)
    for bar, val in zip(bars, importances[idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color="#a0aec0")
    ax.set_xlabel("Mean feature importance (averaged across targets)")
    ax.set_title("Model Feature Importances\n"
                 "GradientBoostingRegressor — MultiOutputRegressor", fontsize=11)
    ax.set_xlim(0, importances.max() * 1.18)
    ax.grid(True, axis="x")
    fig.tight_layout()
    save(fig, "fig6_feature_importance.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    fig1_arctic_sie()
    fig2_sie_ismr()
    fig3_slr()
    fig4_crop_yield()
    fig5_warming()
    fig6_feature_importance()
    print("\nAll figures saved.")
