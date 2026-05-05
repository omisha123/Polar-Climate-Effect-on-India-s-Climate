"""
data.py
-------
All observational and projected data extracted from:
"Polar Climate Change and Its Cascading Impacts on India's Climate System"

Sources: NSIDC Sea Ice Index v3.0, GRACE/GRACE-FO, GPCP v3.2, IMD,
         IPCC AR6, NICRA/ICAR, Kulkarni & Agarwal (2024),
         Chaudhari et al. (2025), Sadai & Karmalkar (2025)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Table 2: Arctic September Sea Ice Extent (NSIDC, 1979–2024)
# ---------------------------------------------------------------------------
ARCTIC_SIE = pd.DataFrame({
    "year":      [1979, 1985, 1990, 1995, 2000, 2005, 2007, 2010, 2012,
                  2015, 2019, 2020, 2023, 2024],
    "sie_mkm2":  [7.2,  6.9,  6.2,  6.3,  6.3,  5.6,  4.3,  4.9,  3.57,
                  4.41, 4.15, 3.92, 3.74, 4.10],
    "anomaly":   [0.6, -0.1, -0.4, -0.3, -0.3, -1.0, -1.3, -0.7, -2.07,
                  -1.29,-1.55,-1.78,-1.96,-1.60],
})

# Synthetic ISMR anomaly series (%) aligned with SIE years
# Derived from inverse correlation r = -0.4 to -0.6 (Kulkarni & Agarwal 2024)
ISMR_ANOMALY_SERIES = pd.DataFrame({
    "year":       [1979, 1985, 1990, 1995, 2000, 2005, 2007, 2010, 2012,
                   2015, 2019, 2020, 2023, 2024],
    "ismr_anom":  [2.1, -0.8,  1.5, -3.2,  0.8, -4.1,  3.2, -1.5,  4.1,
                   -6.2,  5.2,  3.8,  6.1,  4.9],
})

# ---------------------------------------------------------------------------
# Table 3: Sea Level Rise Rates at Indian Coastal Locations
# ---------------------------------------------------------------------------
SEA_LEVEL_OBS = pd.DataFrame({
    "city":           ["Bay of Bengal coast", "Mumbai (Colaba)",
                       "Kolkata/Sundarbans",  "Chennai coast",
                       "Kochi/Kerala coast"],
    "obs_slr_mm_yr":  [4.25, 1.60, 6.50, 3.25, 2.25],
    "subsidence_mm_yr":[6.0, 2.0, 10.0, 5.0, 3.0],
    "effective_mm_yr": [10.25, 3.60, 16.50, 8.25, 5.25],
})

# ---------------------------------------------------------------------------
# Table 5: Crop Yield Changes (NICRA / IPCC AR6 / World Bank)
# ---------------------------------------------------------------------------
CROP_YIELD = pd.DataFrame({
    "crop":          ["Rice (rain-fed)", "Rice (irrigated)", "Wheat",
                      "Maize",           "Pulses",            "Sugarcane"],
    "area_mha":      [44, 21, 30, 10, 25, 5],
    "yield_1p5C":    [-3.5, -5.0, -6.0, -4.0, -5.0, 1.0],   # % midpoints
    "yield_2C":      [-7.5, -10.5,-12.0, -8.5,-11.0,-4.0],
    "yield_4C":      [-20.0,-25.0,-30.0,-20.0,-25.0,-12.5],
    "region":        ["Eastern India, NE", "Punjab, Haryana",
                      "IGP, Rajasthan",    "Karnataka, MP",
                      "Central India",      "Maharashtra, UP"],
})

# ---------------------------------------------------------------------------
# Table 6: IPCC AR6 SSP Projections for India
# ---------------------------------------------------------------------------
SSP_SCENARIOS = pd.DataFrame({
    "scenario":   ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"],
    "ssp_id":     [1, 2, 3, 4],
    "warm_2050":  [1.15, 1.35, 1.75, 2.15],   # °C midpoint
    "warm_2100":  [1.50, 2.25, 3.40, 4.25],
    "hw_mult":    [1.75, 2.50, 3.50, 5.00],   # heat wave freq multiplier
    "slr_2100":   [0.30, 0.45, 0.65, 0.90],   # metres
    "monsoon_lo": [-1,  -3,   -5,   -8],       # % ISMR change range
    "monsoon_hi": [ 2,   4,    8,   12],
})

# ---------------------------------------------------------------------------
# Synthetic training dataset (derived from paper's relationships)
# ---------------------------------------------------------------------------
def generate_training_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic labelled dataset from the paper's observed
    relationships and IPCC AR6 projections.

    Features
    --------
    sie          : Arctic Sept SIE (Mkm²)        — NSIDC
    ant_melt     : Antarctic mass loss (Gt/yr)    — IMBIE / GRACE-FO
    sst_anom     : Indian Ocean SST anomaly (°C)  — HadSST / ERSST v5
    ao_phase     : Arctic Oscillation (-1,0,+1)   — ERA5
    year         : target year (2024–2100)
    ssp          : SSP scenario id (1–4)

    Targets
    -------
    ismr_anom        : ISMR anomaly (%)
    temp_india       : India mean warming (°C)
    slr_m            : Indian Ocean sea level rise (m)
    wheat_yield_chg  : wheat yield change (%)
    hw_freq_mult     : heat wave frequency multiplier
    """
    rng = np.random.default_rng(seed)

    sie         = rng.uniform(2.5, 7.5, n)
    ant_melt    = rng.uniform(100, 600, n)
    sst_anom    = rng.uniform(0.5, 3.5, n)
    ao_phase    = rng.choice([-1, 0, 1], n)
    year        = rng.integers(2024, 2101, n).astype(float)
    ssp         = rng.choice([1, 2, 3, 4], n)

    yr_frac     = (year - 2024) / 76.0
    ao_mod      = np.where(ao_phase == -1, 1.3, np.where(ao_phase == 1, 0.7, 1.0))
    sie_delta   = sie - 5.76

    ismr_anom   = sie_delta * 4.2 * ao_mod + rng.normal(0, 0.8, n)

    ssp_warm    = np.array([0, 1.5, 2.25, 3.4, 4.25])[ssp]
    temp_india  = ssp_warm * yr_frac + sst_anom * 0.3 + rng.normal(0, 0.1, n)
    temp_india  = np.clip(temp_india, 0, 5.5)

    slr_base    = np.array([0, 0.3, 0.45, 0.65, 0.9])[ssp]
    slr_m       = (slr_base * yr_frac
                   + (ant_melt / 200) * 0.12 * yr_frac
                   + rng.normal(0, 0.015, n))
    slr_m       = np.clip(slr_m, 0, 1.5)

    wheat_yield = np.clip(-temp_india * 8 + rng.normal(0, 1.5, n), -45, 3)

    ssp_hw      = np.array([0, 1.75, 2.5, 3.5, 5.0])[ssp]
    hw_freq     = ssp_hw * yr_frac + 1.0 + rng.normal(0, 0.15, n)
    hw_freq     = np.clip(hw_freq, 1.0, 7.0)

    return pd.DataFrame({
        "sie":           sie,
        "ant_melt":      ant_melt,
        "sst_anom":      sst_anom,
        "ao_phase":      ao_phase.astype(float),
        "year":          year,
        "ssp":           ssp.astype(float),
        "ismr_anom":     ismr_anom,
        "temp_india":    temp_india,
        "slr_m":         slr_m,
        "wheat_yield_chg": wheat_yield,
        "hw_freq_mult":  hw_freq,
    })


FEATURE_COLS = ["sie", "ant_melt", "sst_anom", "ao_phase", "year", "ssp"]

TARGET_COLS = [
    "ismr_anom",
    "temp_india",
    "slr_m",
    "wheat_yield_chg",
    "hw_freq_mult",
]

TARGET_LABELS = {
    "ismr_anom":       "ISMR anomaly (%)",
    "temp_india":      "India mean warming (°C)",
    "slr_m":           "Indian Ocean SLR (m)",
    "wheat_yield_chg": "Wheat yield change (%)",
    "hw_freq_mult":    "Heat wave freq. multiplier",
}
