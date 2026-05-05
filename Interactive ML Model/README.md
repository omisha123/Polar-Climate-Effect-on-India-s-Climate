# India Climate Impact Predictor — ML Model

Multi-output regression model trained on relationships from:
**"Polar Climate Change and Its Cascading Impacts on India's Climate System"**

---

## Files

| File | Purpose |
|---|---|
| `data.py` | All observational data from the paper (Tables 2–6) + training data generator |
| `model.py` | Trains the ML model and saves `india_climate_model.pkl` |
| `predict.py` | Interactive CLI for user predictions |
| `visualize.py` | Generates all figures from the paper's data |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Train the model
```bash
python model.py
```
Prints test-set MAE and R² per target, feature importances, saves `india_climate_model.pkl`.

### Step 2 — Predict interactively
```bash
python predict.py
```
Prompts you for each input and prints a formatted prediction table.

### Quick prediction (CLI flags)
```bash
python predict.py --sie 3.74 --ssp 3 --year 2070 --sst_anom 2.1
```

### Batch mode — all SSP scenarios
```bash
python predict.py --batch
```

### Generate figures
```bash
python visualize.py
```
Produces 6 PNG figures.

---

## Model Inputs

| Input | Description | Range |
|---|---|---|
| `sie` | Arctic September sea ice extent (Mkm²) | 2.5 – 7.5 |
| `ant_melt` | Antarctic mass loss rate (Gt/yr) | 100 – 600 |
| `sst_anom` | Indian Ocean SST anomaly °C above 1951 baseline | 0.5 – 3.5 |
| `ao_phase` | Arctic Oscillation phase | negative / neutral / positive |
| `year` | Target year | 2030 – 2100 |
| `ssp` | SSP emissions scenario | 1 / 2 / 3 / 4 |

## Model Outputs

| Output | Description | Source relationship |
|---|---|---|
| `ismr_anom` | Indian Summer Monsoon Rainfall anomaly (%) | Kulkarni & Agarwal (2024), r = −0.4 to −0.6 |
| `temp_india` | India mean warming (°C vs 1995–2014) | IPCC AR6 Table 6; MoES (2020) |
| `slr_m` | Indian Ocean sea level rise (m) | Sadai & Karmalkar (2025); IPCC AR6 |
| `wheat_yield_chg` | Wheat yield change (%) | NICRA/ICAR; IPCC AR6 Table 5 |
| `hw_freq_mult` | Heat wave frequency multiplier | MoES projection; IPCC AR6 |

---

## Data Sources

- NSIDC Sea Ice Index v3.0 (Arctic SIE 1979–2024)
- GRACE/GRACE-FO (Antarctic mass balance)
- IPCC AR6 SSP scenario projections
- GPCP v3.2 / IMD Gridded Data (precipitation)
- NICRA/ICAR crop yield projections
- Kulkarni & Agarwal (2024) — *Int. J. Climatology*
- Chaudhari et al. (2025) — *Ocean-Land-Atmosphere Research*
- Sadai & Karmalkar (2025) — SLR gravitational fingerprints
