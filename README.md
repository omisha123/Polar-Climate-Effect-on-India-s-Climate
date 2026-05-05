# Polar–India Climate ML Suite
## Dataset-driven predictive models for ISMR, Sea Level Rise & Heat Extremes

### Members
Omisha Iyer
Niharika Vuchintala
Archie Patel

### Files
| File | Purpose |
|------|---------|
| `01_data_generator.py` | Simulates/downloads all datasets (NSIDC, GRACE, IMD, ERA5, GPCP, HadSST) |
| `02_feature_engineering.py` | Feature extraction, lag analysis, teleconnection indices |
| `03_random_forest_ismr.py` | Random Forest model for Indian Summer Monsoon Rainfall |
| `04_lstm_projections.py` | LSTM model for heat waves, MHW days & monsoon shift index |
| `05_sealevel_model.py` | Physics-constrained regression for Indian Ocean SLR |
| `run_all.py` | Master runner — trains all models & generates outputs |

### Quick Start
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn scipy joblib
python run_all.py
```

### Outputs
All charts saved to `outputs/` folder. Models saved to `models/` folder.
