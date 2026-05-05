"""
02_feature_engineering.py
==========================
Feature extraction, lag analysis, teleconnection indices, and
preprocessing for all three model pipelines.

Key teleconnections implemented:
  - Arctic SIE → ISMR (Kulkarni & Agarwal 2024, Chaudhari et al. 2025)
  - Barents-Kara SIE → Jet stream → South Asian rainfall
  - ENSO–IOD compound forcing
  - Arctic AO phase modulation
  - GRACE mass balance → Indian Ocean SLR fingerprint
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def compute_lag_features(df, feature_cols, max_lag=3):
    """
    Create lagged versions of polar forcing variables.
    Arctic SIE in June-July best predicts Aug-Sep ISMR (Chaudhari et al. 2025).
    Lag 0 = same year, Lag 1 = previous year.
    """
    df_out = df.copy()
    for col in feature_cols:
        for lag in range(1, max_lag + 1):
            df_out[f"{col}_lag{lag}"] = df_out[col].shift(lag)
    return df_out


def compute_rolling_features(df, feature_cols, windows=[3, 5, 10]):
    """5-year and 10-year running means capture decadal variability."""
    df_out = df.copy()
    for col in feature_cols:
        for w in windows:
            df_out[f"{col}_roll{w}"] = df_out[col].rolling(w, min_periods=2).mean()
    return df_out


def compute_teleconnection_indices(df):
    """
    Derived indices capturing known physical mechanisms.

    1. Barents-Kara Forcing Index (BKFI):
       Combines BK SIE with AO phase — modulates Rossby wave amplitude
       (Kulkarni & Agarwal 2024: AO phase modulates SIE-ISMR correlation)

    2. Polar Pressure Anomaly Index (PPAI):
       Arctic SIE × AO — proxy for polar vortex stability influence on jets

    3. Ocean-Ice Coupling Index (OICI):
       IO SST × Arctic SIE anomaly — combined ocean-cryosphere forcing

    4. Compound Drought-Heat Index (CDHI):
       ENSO × temp anomaly — compound event probability
    """
    df_out = df.copy()

    sie_anom = df_out['arctic_sie_anom']
    bk_anom = df_out['barents_kara_anom']
    ao = df_out['ao_index']
    sst = df_out['io_sst_anom_c']
    enso = df_out['enso_nino34']
    temp = df_out['india_temp_anom_c']

    # Barents-Kara Forcing Index
    df_out['BKFI'] = np.round(-bk_anom * np.sign(ao), 3)

    # Polar Pressure Anomaly Index
    df_out['PPAI'] = np.round(sie_anom * ao, 3)

    # Ocean-Ice Coupling Index (negative = warm ocean + low ice → rainfall enhance)
    df_out['OICI'] = np.round(-sie_anom * sst, 3)

    # Compound Drought-Heat Index
    df_out['CDHI'] = np.round(enso * temp, 3)

    # Phase flags (for composite analysis)
    df_out['low_sie_year'] = (sie_anom < sie_anom.quantile(0.33)).astype(int)
    df_out['high_sie_year'] = (sie_anom > sie_anom.quantile(0.67)).astype(int)
    df_out['el_nino_year'] = (enso > 0.5).astype(int)
    df_out['la_nina_year'] = (enso < -0.5).astype(int)
    df_out['neg_ao_year'] = (ao < -0.5).astype(int)

    return df_out


def compute_composite_analysis(df):
    """
    Composite analysis: ISMR in low-SIE vs high-SIE years.
    Reproduces Table 2 methodology from Kulkarni & Agarwal (2024).
    """
    low_sie = df[df['low_sie_year'] == 1]['ismr_anom_pct']
    high_sie = df[df['high_sie_year'] == 1]['ismr_anom_pct']
    neutral = df[(df['low_sie_year'] == 0) & (df['high_sie_year'] == 0)]['ismr_anom_pct']

    t_stat, p_val = stats.ttest_ind(low_sie, high_sie)

    results = {
        'low_sie_ismr_mean': round(low_sie.mean(), 2),
        'high_sie_ismr_mean': round(high_sie.mean(), 2),
        'neutral_ismr_mean': round(neutral.mean(), 2),
        'difference': round(low_sie.mean() - high_sie.mean(), 2),
        't_statistic': round(t_stat, 3),
        'p_value': round(p_val, 4),
        'significant_p05': p_val < 0.05,
    }
    return results


def compute_mann_kendall_trend(series):
    """
    Mann-Kendall nonparametric trend test.
    Robust to outliers, appropriate for climate time series.
    Returns: slope (Theil-Sen), p-value, trend direction.
    """
    n = len(series)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series.iloc[j] - series.iloc[i]
            if diff > 0: s += 1
            elif diff < 0: s -= 1

    # Variance
    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    z = (s - np.sign(s)) / np.sqrt(var_s)
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    # Theil-Sen slope
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((series.iloc[j] - series.iloc[i]) / (j - i))
    sen_slope = np.median(slopes)

    return {
        'sen_slope': round(sen_slope, 5),
        'p_value': round(p_val, 4),
        'trend': 'increasing' if sen_slope > 0 else 'decreasing',
        'significant_p05': p_val < 0.05
    }


def prepare_ismr_features(df):
    """
    Final feature matrix for Random Forest ISMR model.
    Returns X (features), y (ISMR anomaly), feature names.
    """
    feature_lag_cols = [
        'arctic_sie_anom', 'barents_kara_anom', 'io_sst_anom_c',
        'enso_nino34', 'ao_index', 'eurasian_snow_anom'
    ]
    df_feats = compute_lag_features(df, feature_lag_cols, max_lag=2)
    df_feats = compute_rolling_features(df_feats, feature_lag_cols, windows=[3, 5])
    df_feats = compute_teleconnection_indices(df_feats)

    feature_cols = [
        # Direct forcing
        'arctic_sie_anom', 'barents_kara_anom', 'io_sst_anom_c',
        'enso_nino34', 'ao_index', 'eurasian_snow_anom',
        # Lag-1 (previous year)
        'arctic_sie_anom_lag1', 'barents_kara_anom_lag1', 'enso_nino34_lag1',
        # Lag-2
        'arctic_sie_anom_lag2', 'enso_nino34_lag2',
        # Rolling means (decadal context)
        'arctic_sie_anom_roll5', 'io_sst_anom_c_roll5',
        # Teleconnection indices
        'BKFI', 'PPAI', 'OICI', 'CDHI',
        # Phase flags
        'el_nino_year', 'la_nina_year', 'neg_ao_year',
    ]

    df_clean = df_feats[feature_cols + ['ismr_anom_pct', 'year']].dropna()
    X = df_clean[feature_cols].values
    y = df_clean['ismr_anom_pct'].values
    years = df_clean['year'].values

    return X, y, years, feature_cols


def prepare_lstm_sequences(df, target_col, seq_len=10):
    """
    Prepare sliding window sequences for LSTM.
    Each input = 10 years of forcing data → 1 year prediction.
    """
    forcing_cols = [
        'arctic_sie_anom', 'barents_kara_anom', 'io_sst_anom_c',
        'enso_nino34', 'ao_index', 'india_temp_anom_c',
        'BKFI', 'OICI'
    ]
    df_feats = compute_teleconnection_indices(df)
    df_clean = df_feats[forcing_cols + [target_col, 'year']].dropna()

    X_seq, y_seq, years_out = [], [], []
    vals = df_clean[forcing_cols].values
    targets = df_clean[target_col].values
    years = df_clean['year'].values

    for i in range(seq_len, len(df_clean)):
        X_seq.append(vals[i - seq_len:i])
        y_seq.append(targets[i])
        years_out.append(years[i])

    return np.array(X_seq), np.array(y_seq), np.array(years_out), forcing_cols


def prepare_sealevel_features(df, grace_df, slr_df):
    """
    Feature matrix for sea level rise model.
    Links GRACE mass balance → regional Indian Ocean SLR
    incorporating gravitational fingerprint factor (×1.3 for Antarctica).
    """
    # Merge on year
    merged = slr_df.merge(grace_df[['year', 'greenland_gt_yr', 'antarctica_gt_yr',
                                     'combined_gt_yr', 'slr_equiv_mm_yr']],
                           on='year', how='left')
    merged = merged.merge(
        df[['year', 'io_sst_anom_c', 'india_temp_anom_c']],
        on='year', how='left'
    )

    # Cumulative ice mass loss
    merged['greenland_cumulative_gt'] = merged['greenland_gt_yr'].cumsum()
    merged['antarctica_cumulative_gt'] = merged['antarctica_gt_yr'].cumsum()

    # GRD fingerprint: Antarctica → Indian Ocean ×1.3 amplification
    merged['antarctica_io_fingerprint'] = merged['antarctica_gt_yr'] * 1.3

    # Thermal expansion proxy (IO SST × depth factor)
    merged['thermal_expansion_mm'] = merged['io_sst_anom_c'] * 0.8

    merged = merged.dropna()
    feature_cols = ['greenland_gt_yr', 'antarctica_gt_yr', 'combined_gt_yr',
                    'antarctica_io_fingerprint', 'thermal_expansion_mm',
                    'io_sst_anom_c']

    X = merged[feature_cols].values
    y = merged['slr_cm'].values
    years = merged['year'].values

    return X, y, years, feature_cols, merged


if __name__ == "__main__":
    from data_generator_01 import generate_all_datasets
    df, grace, slr, ssps = generate_all_datasets()

    print("=== Trend Analysis (Mann-Kendall) ===")
    for col in ['arctic_sie_mkm2', 'ismr_index', 'india_temp_anom_c', 'heatwave_days']:
        trend = compute_mann_kendall_trend(df[col])
        print(f"  {col:30s}: slope={trend['sen_slope']:+.4f}/yr, "
              f"p={trend['p_value']:.4f}, {trend['trend']}, "
              f"sig={'YES' if trend['significant_p05'] else 'no'}")

    print("\n=== Composite Analysis: ISMR in Low vs High SIE Years ===")
    df_feats = compute_teleconnection_indices(df)
    comp = compute_composite_analysis(df_feats)
    for k, v in comp.items():
        print(f"  {k:35s}: {v}")

    print("\n=== Feature matrix shapes ===")
    X, y, yrs, fcols = prepare_ismr_features(df)
    print(f"  ISMR RF: X={X.shape}, y={y.shape}")
    Xs, ys, yrs_s, _ = prepare_lstm_sequences(df, 'heatwave_days')
    print(f"  LSTM seqs: X={Xs.shape}, y={ys.shape}")
