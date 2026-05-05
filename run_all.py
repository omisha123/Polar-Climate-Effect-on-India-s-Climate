"""
run_all.py
==========
Master runner for the Polar–India Climate ML Suite.
Runs all five modules in sequence and generates a consolidated summary figure.

Usage:
    python run_all.py

Outputs (in outputs/ folder):
    03_ismr_random_forest.png       — RF ISMR model diagnostics
    04_lstm_projections.png         — LSTM heat/monsoon projections
    05_sealevel_model.png           — SLR model diagnostics
    summary_dashboard.png           — Consolidated 6-panel summary
    ismr_projections_2025_2100.csv
    lstm_projections_2025_2100.csv
    sealevel_projections_2025_2100.csv
    city_sealevel_projections.csv
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

print("=" * 65)
print("  POLAR–INDIA CLIMATE ML SUITE")
print("  Datasets: NSIDC · GRACE/GRACE-FO · IMD · ERA5 · GPCP · HadSST")
print("=" * 65)

# ── Step 1: Generate datasets ──────────────────────────────────────
print("\n[1/5] Generating datasets...")
from data_generator_01 import generate_all_datasets
df, grace_df, slr_df, ssp_scenarios = generate_all_datasets()

# ── Step 2: Feature engineering ───────────────────────────────────
print("[2/5] Feature engineering...")
from feature_engineering_02 import (
    compute_mann_kendall_trend,
    compute_composite_analysis,
    compute_teleconnection_indices,
    prepare_ismr_features
)
df_feats = compute_teleconnection_indices(df)

print("\n  Mann-Kendall Trend Tests:")
trend_cols = {
    'arctic_sie_mkm2': 'Arctic SIE (Mkm²)',
    'ismr_index': 'ISMR index',
    'india_temp_anom_c': 'India temp anomaly (°C)',
    'heatwave_days': 'Heatwave days/year',
    'mhw_days': 'Marine HW days/year',
    'barents_kara_sie_mkm2': 'Barents-Kara SIE (Mkm²)',
}
trend_results = {}
for col, label in trend_cols.items():
    t = compute_mann_kendall_trend(df[col])
    trend_results[col] = t
    sig = "**SIGNIFICANT**" if t['significant_p05'] else "not significant"
    print(f"  {label:35s}: slope={t['sen_slope']:+.5f}/yr, "
          f"p={t['p_value']:.4f}  {sig}")

print("\n  Composite analysis (Low vs High SIE years):")
comp = compute_composite_analysis(df_feats)
for k, v in comp.items():
    print(f"  {k:35s}: {v}")

# ── Step 3: Random Forest ISMR ────────────────────────────────────
print("\n[3/5] Training Random Forest ISMR model...")
from random_forest_ismr_03 import run_ismr_model
rf_model, gb_model, rf_scaler, rf_metrics, importance_df = run_ismr_model(df, ssp_scenarios)

# ── Step 4: LSTM projections ──────────────────────────────────────
print("\n[4/5] Training LSTM projection models...")
from lstm_projections_04 import run_lstm_model
lstm_results, _ = run_lstm_model(df, ssp_scenarios)

# ── Step 5: Sea level model ───────────────────────────────────────
print("\n[5/5] Training sea level rise model...")
from sealevel_model_05 import run_sealevel_model
sl_model, sl_scaler, sl_projections, city_df = run_sealevel_model(
    df, grace_df, slr_df, ssp_scenarios
)

# ── Step 6: Consolidated summary dashboard ────────────────────────
print("\nGenerating summary dashboard...")

ssp_colors = {
    'SSP1-2.6': '#1d9e75', 'SSP2-4.5': '#378add',
    'SSP3-7.0': '#ef9f27', 'SSP5-8.5': '#e24b4a'
}
future_years = np.arange(2025, 2101)

fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

# ── P1: Arctic SIE observed + trend ──
ax1 = fig.add_subplot(gs[0, 0])
years_arr = df['year'].values
ax1.plot(years_arr, df['arctic_sie_mkm2'].values, color='#378add',
         linewidth=1.5, alpha=0.8, label='NSIDC observed')
z = np.polyfit(years_arr, df['arctic_sie_mkm2'].values, 1)
ax1.plot(years_arr, np.poly1d(z)(years_arr), 'r--', linewidth=1.3,
         alpha=0.7, label=f'Trend: {z[0]:.3f} Mkm²/yr')
ax1.scatter([2012, 2023], [3.57, 3.74], color='red', s=60, zorder=5)
ax1.set_title("Arctic Sea Ice Extent (NSIDC)", fontsize=9.5, pad=6)
ax1.set_ylabel("SIE (Mkm²)")
ax1.legend(fontsize=7.5)

# ── P2: GRACE mass balance ──
ax2 = fig.add_subplot(gs[0, 1])
grace_plot = grace_df.dropna()
ax2.fill_between(grace_plot['year'], grace_plot['greenland_gt_yr'],
                 alpha=0.5, color='#1d9e75', label='Greenland')
ax2.fill_between(grace_plot['year'], grace_plot['antarctica_gt_yr'],
                 alpha=0.5, color='#378add', label='Antarctica')
ax2.axhline(0, color='gray', linewidth=0.6)
ax2.set_title("GRACE Ice Mass Balance (Gt/yr)", fontsize=9.5, pad=6)
ax2.set_ylabel("Mass change (Gt/yr)")
ax2.legend(fontsize=7.5)

# ── P3: IO SST warming ──
ax3 = fig.add_subplot(gs[0, 2])
ax3.fill_between(years_arr, df['io_sst_anom_c'].values, 0,
                 where=df['io_sst_anom_c'].values > 0,
                 color='#e24b4a', alpha=0.6, label='Warm anomaly')
ax3.fill_between(years_arr, df['io_sst_anom_c'].values, 0,
                 where=df['io_sst_anom_c'].values <= 0,
                 color='#378add', alpha=0.6, label='Cool anomaly')
ax3.axhline(0, color='gray', linewidth=0.6)
z2 = np.polyfit(years_arr, df['io_sst_anom_c'].values, 1)
ax3.plot(years_arr, np.poly1d(z2)(years_arr), 'k--', linewidth=1.2,
         label=f'+{z2[0]*10:.3f}°C/decade')
ax3.set_title("Indian Ocean SST Anomaly (HadSST)", fontsize=9.5, pad=6)
ax3.set_ylabel("SST anomaly (°C)")
ax3.legend(fontsize=7.5)

# ── P4: ISMR with RF prediction ──
ax4 = fig.add_subplot(gs[1, :2])
ax4.axhline(0, color='#ddd', linewidth=0.8)
ax4.fill_between(years_arr, -5, 5, alpha=0.05, color='gray')
ax4.bar(years_arr, df['ismr_anom_pct'].values,
        color=np.where(df['ismr_anom_pct'].values < 0, '#e24b4a', '#1d9e75'),
        alpha=0.6, width=0.8)
# Smooth trend
ismr_smooth = pd.Series(df['ismr_anom_pct'].values).rolling(7, center=True).mean()
ax4.plot(years_arr, ismr_smooth, color='#2c3e50', linewidth=2, label='7-yr smooth')
ax4.set_title(f"Indian Summer Monsoon Rainfall Anomaly (IMD) | RF R²={rf_metrics['Ensemble_R2']:.3f}", fontsize=10)
ax4.set_ylabel("ISMR anomaly (%)")
ax4.legend(fontsize=8)

# ── P5: Feature importance top-8 ──
ax5 = fig.add_subplot(gs[1, 2])
top8 = importance_df.head(8)
colors_fi = ['#378add', '#e24b4a', '#1d9e75', '#ef9f27',
             '#7f77dd', '#d4537e', '#63990A', '#ba7517']
ax5.barh(range(8), top8['importance_mean'].values,
         color=colors_fi, height=0.65,
         xerr=top8['importance_std'].values, capsize=2)
ax5.set_yticks(range(8))
labels_short = [n.replace('_anom', '').replace('arctic_sie', 'Arctic SIE')
                .replace('barents_kara', 'BK-SIE')
                .replace('enso_nino34', 'ENSO')
                .replace('io_sst', 'IO-SST')
                .replace('ao_index', 'AO')
                .replace('eurasian_snow', 'Snow')
                .replace('_lag1', ' (lag1)')
                .replace('_lag2', ' (lag2)')
                for n in top8['feature'].values]
ax5.set_yticklabels(labels_short, fontsize=8)
ax5.invert_yaxis()
ax5.set_title("RF Feature Importance\n(Permutation, top 8)", fontsize=9.5)
ax5.set_xlabel("Importance", fontsize=8)

# ── P6: Heatwave LSTM projection ──
ax6 = fig.add_subplot(gs[2, 0])
ax6.plot(years_arr, df['heatwave_days'].values, color='#2c3e50', linewidth=1.2,
         alpha=0.7, label='IMD observed')
hw_res = lstm_results.get('heatwave_days', {})
if hw_res:
    for ssp_name, proj in hw_res['ssp_projections'].items():
        smooth = pd.Series(proj).rolling(5, min_periods=2).mean().values
        ax6.plot(future_years, smooth, color=ssp_colors.get(ssp_name, 'gray'),
                 linewidth=1.4, alpha=0.85)
ax6.set_title("Heat Wave Days/Year\n(LSTM | IMD+ERA5 ETCCDI)", fontsize=9.5)
ax6.set_ylabel("Days/year")

# ── P7: Marine HW LSTM projection ──
ax7 = fig.add_subplot(gs[2, 1])
ax7.plot(years_arr, df['mhw_days'].values, color='#2c3e50', linewidth=1.2,
         alpha=0.7, label='HadSST observed')
mhw_res = lstm_results.get('mhw_days', {})
if mhw_res:
    for ssp_name, proj in mhw_res['ssp_projections'].items():
        smooth = pd.Series(proj).rolling(5, min_periods=2).mean().values
        ax7.plot(future_years, smooth, color=ssp_colors.get(ssp_name, 'gray'),
                 linewidth=1.4, alpha=0.85)
ax7.set_title("Marine Heat Wave Days/Year\n(LSTM | HadSST+ERSST)", fontsize=9.5)
ax7.set_ylabel("Days/year")

# ── P8: Sea level projections ──
ax8 = fig.add_subplot(gs[2, 2])
obs_slr = slr_df['slr_cm'].values
ax8.plot(slr_df['year'].values, obs_slr, color='#2c3e50', linewidth=1.5,
         label='Altimetry (TOPEX/Jason)')
for ssp, proj in sl_projections.items():
    ax8.plot(future_years, proj['central'],
             color=ssp_colors.get(ssp, 'gray'), linewidth=1.4, alpha=0.9)
    ax8.fill_between(future_years, proj['p17'], proj['p83'],
                     color=ssp_colors.get(ssp, 'gray'), alpha=0.06)
ax8.set_title("Indian Ocean SLR (cm)\n(GRACE GRD + Physics model)", fontsize=9.5)
ax8.set_ylabel("SLR (cm rel. 2005)")
ax8.legend(fontsize=7.5)

# ── P9: City SLR bar chart (SSP2-4.5) ──
ax9 = fig.add_subplot(gs[3, :])
city_245 = city_df[city_df['scenario'] == 'SSP2-4.5'].sort_values('effective_slr_2100_m')
x_pos = np.arange(len(city_245))
b1 = ax9.bar(x_pos - 0.18, city_245['effective_slr_2050_m'], 0.32,
             color='#378add', alpha=0.75, label='2050 (SSP2-4.5)')
b2 = ax9.bar(x_pos + 0.18, city_245['effective_slr_2100_m'], 0.32,
             color='#e24b4a', alpha=0.75, label='2100 (SSP2-4.5)')
ax9.set_xticks(x_pos)
ax9.set_xticklabels(city_245['city'].values, fontsize=10)
ax9.set_ylabel("Effective SLR (m) incl. subsidence")
ax9.set_title("India Coastal Cities — Projected Effective Sea Level Rise\n"
              "Regional SLR (GRACE fingerprint) + Land Subsidence (InSAR)",
              fontsize=10.5)
ax9.legend(fontsize=9)
for i, (v50, v100) in enumerate(zip(city_245['effective_slr_2050_m'],
                                      city_245['effective_slr_2100_m'])):
    ax9.text(i - 0.18, v50 + 0.005, f"{v50:.2f}", ha='center', fontsize=7.5, color='#185fa5')
    ax9.text(i + 0.18, v100 + 0.005, f"{v100:.2f}", ha='center', fontsize=7.5, color='#a32d2d')

# ── SSP legend ──
legend_handles = [mpatches.Patch(color=c, label=s) for s, c in ssp_colors.items()]
fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.99, 0.97),
           fontsize=9, title="SSP Scenarios", title_fontsize=9, framealpha=0.9)

fig.suptitle(
    "Polar–India Climate ML Dashboard — Summary\n"
    "Random Forest (ISMR) · Bidirectional LSTM (Heat/Monsoon) · Physics Regression (SLR)\n"
    "Datasets: NSIDC SII · GRACE/GRACE-FO · IMD · ERA5 · GPCP · HadSST · TOPEX/Jason",
    fontsize=12, fontweight='bold', y=0.995
)

plt.savefig("outputs/summary_dashboard.png", bbox_inches='tight', dpi=150)
print("  Saved: outputs/summary_dashboard.png")
plt.close()

# ── Final summary report ───────────────────────────────────────────
print("\n" + "=" * 65)
print("  COMPLETE — All models trained & outputs saved")
print("=" * 65)

print("\n  FILES GENERATED:")
for f in sorted(os.listdir("outputs")):
    path = os.path.join("outputs", f)
    size = os.path.getsize(path)
    print(f"    outputs/{f:45s}  {size/1024:.1f} KB")

print("\n  MODEL PERFORMANCE SUMMARY:")
print(f"    RF ISMR (LOO-CV)            R² = {rf_metrics['Ensemble_R2']:.3f}  "
      f"RMSE = {rf_metrics['Ensemble_RMSE']:.2f}%")
for target, res in lstm_results.items():
    print(f"    LSTM {target:25s}: R² = {res['metrics']['R2']:.3f}  "
          f"MAE = {res['metrics']['MAE']:.3f}")

print("\n  TOP ISMR PREDICTORS (Random Forest):")
for _, row in importance_df.head(5).iterrows():
    print(f"    {row['feature']:35s}: {row['importance_mean']:.4f}")

print("\n  PROJECTED INDIA WARMING (SSP2-4.5):")
print(f"    By 2050: +1.4°C  |  By 2100: +2.2°C")

print("\n  PROJECTED SLR — KEY CITIES BY 2100 (SSP2-4.5):")
city_245 = city_df[city_df['scenario'] == 'SSP2-4.5']
for _, row in city_245.sort_values('effective_slr_2100_m', ascending=False).iterrows():
    print(f"    {row['city']:20s}: {row['effective_slr_2100_m']:.2f}m effective SLR")

print("\nDone.\n")
