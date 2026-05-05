"""
model.py
--------
Trains a multi-output gradient-boosted regression model on the synthetic
dataset generated from the research paper's observed relationships.

Model: MultiOutputRegressor wrapping GradientBoostingRegressor
       (one sub-estimator per target variable)

Run:   python model.py
Saves: india_climate_model.pkl
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from data import generate_training_data, FEATURE_COLS, TARGET_COLS, TARGET_LABELS

# ---------------------------------------------------------------------------
# 1. Generate data
# ---------------------------------------------------------------------------
print("Generating training data from paper relationships...")
df = generate_training_data(n=3000, seed=42)

X = df[FEATURE_COLS].values
y = df[TARGET_COLS].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 2. Build pipeline
# ---------------------------------------------------------------------------
base_gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    min_samples_leaf=5,
    random_state=42,
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  MultiOutputRegressor(base_gbr, n_jobs=-1)),
])

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
print("Training model...")
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

print("\n── Model performance on held-out test set ──")
print(f"{'Target':<28} {'MAE':>10} {'R²':>8}")
print("─" * 50)
for i, (col, label) in enumerate(TARGET_LABELS.items()):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    r2  = r2_score(y_test[:, i], y_pred[:, i])
    print(f"{label:<28} {mae:>10.4f} {r2:>8.4f}")

# ---------------------------------------------------------------------------
# 5. Feature importance (averaged across sub-estimators)
# ---------------------------------------------------------------------------
print("\n── Feature importances (mean across targets) ──")
importances = np.mean(
    [est.feature_importances_ for est in pipeline.named_steps["model"].estimators_],
    axis=0,
)
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<12} {imp:.4f}  {bar}")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
MODEL_PATH = "india_climate_model.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)
print(f"\nModel saved → {MODEL_PATH}")
