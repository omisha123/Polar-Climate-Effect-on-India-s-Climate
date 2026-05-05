"""
01_data_generator.py
====================
Generates realistic synthetic datasets matching the statistical properties,
trends, and variability of the real observational sources used in the paper:

  - NSIDC Sea Ice Index v3.0    (Arctic SIE, 1979-2024)
  - GRACE / GRACE-FO            (Ice mass balance, 2002-2024)
  - GPCP v3.2                   (Global precipitation, 1979-2024)
  - CRU TS v4.09                (India temperature/precip, 1951-2024)
  - ERA5 Reanalysis              (ENSO, AO, Barents-Kara SIE, SST, 1979-2024)
  - IMD Gridded Dataset          (India rainfall & temperature, 1951-2024)
  - HadSST 4.1.1                 (Indian Ocean SST, 1850-2024)
  - TOPEX/Jason Altimetry        (Sea level anomaly, 1993-2024)
  - IMBIE Antarctica             (Ice mass, 1992-2020)

NOTE: Replace generate_*() calls with real API/file loaders in production.
      Documented with exact URLs and dataset access methods for each source.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────
# REAL DATA ACCESS NOTES (replace generate_ fns)
# ─────────────────────────────────────────────
# NSIDC SII:   https://nsidc.org/data/G02135  (CSV download)
# GRACE:       https://podaac.jpl.nasa.gov/GRACE  (NetCDF via Earthdata)
# GPCP:        https://psl.noaa.gov/data/gridded/data.gpcp.html
# CRU TS:      https://crudata.uea.ac.uk/cru/data/hrg/
# ERA5:        https://cds.climate.copernicus.eu/api/v2  (cdsapi Python package)
# IMD:         https://www.imdpune.gov.in/library/public/Gridded_Data_Download.html
# HadSST:      https://www.metoffice.gov.uk/hadobs/hadsst4/
# Altimetry:   https://sealevel.nasa.gov/data/sealevel-data/sealevel/  (CSIRO)
# IMBIE:       https://imbie.org/data-downloads/


def generate_arctic_sie(years):
    """
    NSIDC Sea Ice Index v3.0 — September Arctic SIE (Mkm²)
    Trend: -0.77 Mkm²/decade (p<0.001)
    Observed range: 7.2 (1979) → 3.74 (2023 record low)
    """
    n = len(years)
    trend = 7.5 - (years - 1979) * 0.077
    # interannual variability (std ~0.6 Mkm²) + regime shift post-2007
    variability = np.random.normal(0, 0.45, n)
    regime = np.where(years > 2007, -0.35, 0)
    sie = np.clip(trend + variability + regime, 2.8, 8.5)
    # pin known record years
    sie[years == 2012] = 3.57
    sie[years == 2023] = 3.74
    return np.round(sie, 3)


def generate_barents_kara_sie(years):
    """
    Barents-Kara Sea regional SIE — critical for ISMR teleconnection
    Trend: ~-50% since 1979
    """
    n = len(years)
    trend = 1.2 - (years - 1979) * 0.012
    variability = np.random.normal(0, 0.15, n)
    return np.clip(np.round(trend + variability, 3), 0.05, 1.5)


def generate_grace_mass_balance(years):
    """
    GRACE/GRACE-FO ice mass balance (Gt/year)
    Greenland: accelerating from ~-150 Gt/yr (2002) to ~-280 Gt/yr (2024)
    Antarctica: accelerating from ~-80 Gt/yr (2002) to ~-150 Gt/yr (2024)
    """
    mask = years >= 2002
    y_grace = years[mask]
    n = len(y_grace)

    greenland = -150 - (y_grace - 2002) * 5.8 + np.random.normal(0, 22, n)
    antarctica = -80 - (y_grace - 2002) * 3.5 + np.random.normal(0, 14, n)

    df = pd.DataFrame({
        'year': y_grace,
        'greenland_gt_yr': np.round(greenland, 1),
        'antarctica_gt_yr': np.round(antarctica, 1),
        'combined_gt_yr': np.round(greenland + antarctica, 1),
        # Sea level equivalent (mm): 1 Gt ≈ 1/360 mm SLR
        'slr_equiv_mm_yr': np.round(np.abs(greenland + antarctica) / 360, 3)
    })
    return df


def generate_enso_index(years):
    """
    ERA5 / CPC Niño3.4 SST anomaly index
    Significant El Niño years: 1983, 1987, 1992, 1997-98, 2002, 2009, 2015-16, 2023-24
    """
    n = len(years)
    # base quasi-periodic ENSO signal (3-7 year cycle)
    t = np.linspace(0, 4 * np.pi, n)
    base = 0.6 * np.sin(t) + 0.3 * np.sin(2.3 * t)
    noise = np.random.normal(0, 0.4, n)
    enso = base + noise
    # major El Niños
    for yr, val in [(1983, 2.1), (1988, -1.5), (1998, 2.5), (2010, -1.8),
                    (2016, 2.3), (2020, -1.2), (2024, 1.8)]:
        if yr in years:
            enso[years == yr] = val
    return np.round(np.clip(enso, -3, 3), 2)


def generate_ao_index(years):
    """
    Arctic Oscillation (AO) index from ERA5
    Modulates Arctic-ISMR teleconnection strength
    """
    n = len(years)
    return np.round(np.random.normal(0, 1.2, n), 2)


def generate_indian_ocean_sst(years):
    """
    HadSST 4.1.1 — Indian Ocean basin-mean SST anomaly
    Trend: +0.12°C/decade (faster than global mean)
    Total warming: ~1°C since 1951
    """
    n = len(years)
    trend = (years - 1979) * 0.013
    variability = np.random.normal(0, 0.18, n)
    # step warming post 1998 El Niño
    step = np.where(years >= 1998, 0.22, 0)
    return np.round(trend + variability + step, 3)


def generate_imd_rainfall(years, arctic_sie, enso, io_sst, bk_sie, ao):
    """
    IMD Gridded Rainfall — All-India Summer Monsoon Rainfall index
    Base: ~89 cm (long-period average 1961-2010 = 100%)
    Key drivers: ENSO (-), IO SST (+), Arctic SIE (-/+), Barents-Kara (-)
    Correlation SIE-ISMR: -0.4 to -0.6 (Kulkarni & Agarwal 2024)
    """
    n = len(years)
    # Multi-linear RF-inspired relationship
    sie_anom = arctic_sie - np.mean(arctic_sie)
    bk_anom = bk_sie - np.mean(bk_sie)

    signal = (100
              - 3.2 * sie_anom          # Arctic SIE: negative → more rain NW India
              - 4.5 * bk_anom           # Barents-Kara: key teleconnection
              + 2.8 * io_sst            # Warm IO → enhanced moisture
              - 3.8 * enso              # El Niño → deficit
              + 1.2 * ao                # Negative AO → strengthened teleconnection
              + (years - 1979) * 0.08   # Long-term trend
              + np.random.normal(0, 4.5, n))  # residual

    # Drought years: 2002, 2009, 2014, 2015 (El Niño)
    for yr in [2002, 2009, 2014, 2015]:
        if yr in years:
            signal[years == yr] -= 12
    # Excess years: 1994, 2019
    for yr in [1994, 2019]:
        if yr in years:
            signal[years == yr] += 10

    return np.round(np.clip(signal, 70, 130), 2)


def generate_india_temperature(years):
    """
    CRU TS / IMD — India mean annual temperature anomaly
    Total warming 1901-2024: ~0.89°C (Pillai et al. 2025, PLOS Climate)
    Acceleration post-2000
    """
    n = len(years)
    trend = (years - 1979) * 0.022
    variability = np.random.normal(0, 0.15, n)
    accel = np.where(years > 2000, (years - 2000) * 0.01, 0)
    return np.round(trend + variability + accel, 3)


def generate_heatwave_days(years, temp_anom):
    """
    ETCCDI index — heat wave days per year (India, IMD+ERA5)
    Baseline ~8 days/yr (1979-1990), rising to ~22+ days by 2024
    """
    n = len(years)
    base = 8 + (years - 1979) * 0.38
    temp_driver = temp_anom * 4.2
    noise = np.random.normal(0, 2.5, n)
    return np.round(np.clip(base + temp_driver + noise, 3, 60), 1)


def generate_sea_level(years):
    """
    TOPEX/Jason/Sentinel-6 altimetry — Indian Ocean SLR
    Rate: 3.3 mm/yr (global), ~3.7 mm/yr (Indian Ocean)
    Acceleration from ~2.1 mm/yr (1990s) to ~4.2 mm/yr (2020s)
    """
    mask = years >= 1993
    y_alt = years[mask]
    n = len(y_alt)
    t = y_alt - 1993

    # Accelerating trend
    rate = 2.1 + t * 0.065   # mm/yr increasing
    cumulative = np.cumsum(rate) / 10  # cm
    # Normalize so 1993 = 0
    cumulative = cumulative - cumulative[0]
    noise = np.random.normal(0, 0.4, n)
    seasonal = 0.5 * np.sin(2 * np.pi * t)

    return pd.DataFrame({
        'year': y_alt,
        'slr_cm': np.round(cumulative + noise, 2),
        'rate_mm_yr': np.round(rate + np.random.normal(0, 0.3, n), 2)
    })


def generate_marine_heatwave_days(years, io_sst):
    """
    HadSST + ERSST v5 — Indian Ocean marine heat wave days/year
    Historical: ~20 days/yr (1979-2000), projected ~200 days/yr by 2050
    (PLOS Climate 2025, Pillai et al.)
    """
    n = len(years)
    trend = 20 + (years - 1979) * 2.9
    sst_driver = io_sst * 12
    noise = np.random.normal(0, 8, n)
    return np.round(np.clip(trend + sst_driver + noise, 5, 365), 1)


def generate_all_datasets():
    """Master function — generates all datasets and saves to CSV"""

    years_full = np.arange(1951, 2025)
    years_79 = np.arange(1979, 2025)

    print("Generating datasets...")

    # --- Polar datasets ---
    arctic_sie = generate_arctic_sie(years_79)
    bk_sie = generate_barents_kara_sie(years_79)
    enso = generate_enso_index(years_79)
    ao = generate_ao_index(years_79)
    io_sst = generate_indian_ocean_sst(years_79)
    india_temp = generate_india_temperature(years_79)
    ismr = generate_imd_rainfall(years_79, arctic_sie, enso, io_sst, bk_sie, ao)
    heatwave = generate_heatwave_days(years_79, india_temp)
    mhw = generate_marine_heatwave_days(years_79, io_sst)
    grace = generate_grace_mass_balance(years_79)
    slr = generate_sea_level(years_79)

    # --- Main feature matrix (1979-2024) ---
    df_main = pd.DataFrame({
        'year': years_79,
        # POLAR FORCING (NSIDC + ERA5)
        'arctic_sie_mkm2': arctic_sie,
        'arctic_sie_anom': arctic_sie - arctic_sie.mean(),
        'barents_kara_sie_mkm2': bk_sie,
        'barents_kara_anom': bk_sie - bk_sie.mean(),
        # OCEANIC (HadSST + ERSST)
        'io_sst_anom_c': io_sst,
        'enso_nino34': enso,
        # ATMOSPHERIC (ERA5)
        'ao_index': ao,
        # INDIA OUTCOMES (IMD + ERA5)
        'ismr_index': ismr,          # 100 = long-period average
        'ismr_anom_pct': ismr - 100,
        'india_temp_anom_c': india_temp,
        'heatwave_days': heatwave,
        'mhw_days': mhw,
        # DERIVED
        'sie_trend_5yr': pd.Series(arctic_sie).rolling(5).mean().values,
        'ismr_deficit': (ismr < 94).astype(int),   # drought year flag
        'ismr_excess': (ismr > 106).astype(int),   # excess year flag
    })

    # Sea level ratio NW India / IGP (westward shift proxy, GPCP)
    nw_ratio = 0.85 + (years_79 - 1979) * 0.005 + np.random.normal(0, 0.04, len(years_79))
    df_main['nw_igp_ratio'] = np.round(nw_ratio, 3)

    # Eurasian snow cover anomaly (ERA5, Oct-Nov) — known ISMR precursor
    df_main['eurasian_snow_anom'] = np.round(np.random.normal(0, 1, len(years_79)), 2)

    df_main.to_csv("data/main_features_1979_2024.csv", index=False)
    print(f"  Saved: data/main_features_1979_2024.csv  ({len(df_main)} rows, {len(df_main.columns)} features)")

    grace.to_csv("data/grace_mass_balance_2002_2024.csv", index=False)
    print(f"  Saved: data/grace_mass_balance_2002_2024.csv  ({len(grace)} rows)")

    slr.to_csv("data/sealevel_altimetry_1993_2024.csv", index=False)
    print(f"  Saved: data/sealevel_altimetry_1993_2024.csv  ({len(slr)} rows)")

    # --- SSP Scenario parameters (IPCC AR6 Table) ---
    ssp_scenarios = pd.DataFrame({
        'scenario': ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'],
        'code': [126, 245, 370, 585],
        'warming_2050_c': [1.1, 1.4, 1.8, 2.3],
        'warming_2100_c': [1.5, 2.2, 3.3, 4.2],
        'slr_central_2100_m': [0.32, 0.52, 0.72, 1.05],
        'slr_high_2100_m': [0.55, 0.85, 1.15, 1.68],
        'hw_multiplier_2100': [1.8, 2.5, 3.5, 5.0],
        'monsoon_intensity_change_pct': [0.5, 1.2, 2.0, 3.5],
    })
    ssp_scenarios.to_csv("data/ssp_scenarios_ar6.csv", index=False)
    print(f"  Saved: data/ssp_scenarios_ar6.csv")

    print("\nAll datasets ready.\n")
    return df_main, grace, slr, ssp_scenarios


if __name__ == "__main__":
    df, grace, slr, ssps = generate_all_datasets()
    print("Dataset summary:")
    print(df.describe().round(3))
    print("\nKey correlations with ISMR:")
    corr_cols = ['arctic_sie_anom', 'barents_kara_anom', 'io_sst_anom_c',
                 'enso_nino34', 'ao_index', 'eurasian_snow_anom']
    for col in corr_cols:
        r = df['ismr_anom_pct'].corr(df[col])
        print(f"  ISMR vs {col:30s}: r = {r:+.3f}")
