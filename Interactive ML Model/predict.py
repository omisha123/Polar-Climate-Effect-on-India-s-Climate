"""
predict.py
----------
Interactive CLI to query the trained India climate impact model.

Usage:
    python predict.py                      # interactive prompt
    python predict.py --batch              # run all SSP scenarios for 2050
    python predict.py --sie 3.74 --ssp 3   # single quick prediction
"""

import argparse
import pickle
import textwrap
import numpy as np

from data import FEATURE_COLS, TARGET_COLS, TARGET_LABELS

MODEL_PATH = "india_climate_model.pkl"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def predict_single(sie, ant_melt, sst_anom, ao_phase, year, ssp):
    """Return a dict of predictions for one set of inputs."""
    X = np.array([[sie, ant_melt, sst_anom, ao_phase, year, ssp]])
    y = MODEL.predict(X)[0]
    return dict(zip(TARGET_COLS, y))

def ao_encode(label: str) -> float:
    mapping = {"negative": -1.0, "neutral": 0.0, "positive": 1.0,
               "neg": -1.0, "neu": 0.0, "pos": 1.0,
               "-1": -1.0, "0": 0.0, "1": 1.0}
    return mapping.get(label.lower(), 0.0)

def print_results(preds: dict, inputs: dict):
    """Pretty-print prediction results."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         India Climate Impact — Model Prediction          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Arctic SIE      : {inputs['sie']:.2f} Mkm²"
          f"{'':>28}║")
    print(f"║  Antarctic melt  : {inputs['ant_melt']:.0f} Gt/yr"
          f"{'':>29}║")
    print(f"║  Indian Ocean SST: +{inputs['sst_anom']:.1f}°C above 1951 baseline"
          f"{'':>22}║")
    ao_labels = {-1.0: "Negative", 0.0: "Neutral", 1.0: "Positive"}
    print(f"║  AO phase        : {ao_labels[inputs['ao_phase']]}"
          f"{'':>37}║")
    ssp_names = {1:"SSP1-2.6", 2:"SSP2-4.5", 3:"SSP3-7.0", 4:"SSP5-8.5"}
    print(f"║  Scenario        : {ssp_names[int(inputs['ssp'])]}"
          f"{'':>38}║")
    print(f"║  Target year     : {int(inputs['year'])}"
          f"{'':>38}║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  PREDICTIONS                                             ║")
    print("╠══════════════════════════════════════════════════════════╣")

    icons = {
        "ismr_anom":       "🌧",
        "temp_india":      "🌡",
        "slr_m":           "🌊",
        "wheat_yield_chg": "🌾",
        "hw_freq_mult":    "🔥",
    }
    for col, label in TARGET_LABELS.items():
        val = preds[col]
        if col == "ismr_anom":
            fmt = f"{val:+.2f} %"
            note = "above-normal monsoon" if val > 0 else "below-normal monsoon"
        elif col == "temp_india":
            fmt = f"+{val:.2f} °C"
            note = "vs 1995–2014 baseline"
        elif col == "slr_m":
            fmt = f"{val*100:.1f} cm"
            note = "Indian Ocean median"
        elif col == "wheat_yield_chg":
            fmt = f"{val:+.1f} %"
            note = "IGP / Rajasthan"
        elif col == "hw_freq_mult":
            fmt = f"{val:.2f}×"
            note = "more frequent vs 1976–2005"
        line = f"  {label:<28} {fmt:<14}  {note}"
        print(f"║{line:<58}║")

    print("╠══════════════════════════════════════════════════════════╣")

    # Risk interpretation
    ismr = preds["ismr_anom"]
    slr  = preds["slr_m"]
    hw   = preds["hw_freq_mult"]

    risks = []
    if abs(ismr) > 5:
        risks.append(f"Severe monsoon {'surplus' if ismr>0 else 'deficit'} — NW India flood or IGP drought risk high")
    if slr > 0.4:
        risks.append(f"SLR > {slr*100:.0f} cm — Mumbai/Kolkata coastal inundation risk elevated")
    if hw > 3.0:
        risks.append(f"Heat waves {hw:.1f}× more frequent — compound heat-drought events likely")
    if preds["wheat_yield_chg"] < -15:
        risks.append(f"Wheat yield −{abs(preds['wheat_yield_chg']):.0f}% — food security threat in IGP")

    if risks:
        print("║  ⚠  KEY RISKS                                            ║")
        for r in risks:
            for line in textwrap.wrap(r, 56):
                print(f"║    {line:<56}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------
def interactive():
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  India Climate Impact Predictor — Interactive Mode")
    print("  Source: Polar Climate Change & India's Climate System (2025)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Enter values or press Enter to use defaults.\n")

    defaults = {
        "sie":       ("Arctic Sept. SIE (Mkm²) [2.5–7.5]", 4.10),
        "ant_melt":  ("Antarctic mass loss (Gt/yr) [100–600]", 200.0),
        "sst_anom":  ("Indian Ocean SST anomaly °C [0.5–3.5]", 1.2),
        "ao_phase":  ("AO phase [negative/neutral/positive]", "neutral"),
        "year":      ("Target year [2030–2100]", 2050),
        "ssp":       ("SSP scenario [1=SSP1-2.6 / 2=SSP2-4.5 / 3=SSP3-7.0 / 4=SSP5-8.5]", 2),
    }

    inputs = {}
    for key, (prompt, default) in defaults.items():
        raw = input(f"  {prompt} (default {default}): ").strip()
        if not raw:
            inputs[key] = default
        elif key == "ao_phase":
            inputs[key] = ao_encode(raw)
        else:
            inputs[key] = float(raw)

    if isinstance(inputs["ao_phase"], str):
        inputs["ao_phase"] = ao_encode(inputs["ao_phase"])

    preds = predict_single(**inputs)
    print_results(preds, inputs)

# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------
def batch():
    print("\n── Batch: All SSP scenarios × 2050 & 2100 ──\n")
    base = dict(sie=4.10, ant_melt=200, sst_anom=1.2, ao_phase=0.0)
    header = f"{'Scenario':<12} {'Year':>6} {'ISMR%':>8} {'Temp°C':>8} {'SLR cm':>8} {'Wheat%':>8} {'HW mult':>8}"
    print(header)
    print("─" * len(header))
    ssp_names = {1:"SSP1-2.6", 2:"SSP2-4.5", 3:"SSP3-7.0", 4:"SSP5-8.5"}
    for ssp in [1, 2, 3, 4]:
        for year in [2050, 2100]:
            p = predict_single(**base, year=year, ssp=ssp)
            print(
                f"{ssp_names[ssp]:<12} {year:>6}"
                f" {p['ismr_anom']:>+8.2f}"
                f" {p['temp_india']:>8.2f}"
                f" {p['slr_m']*100:>8.1f}"
                f" {p['wheat_yield_chg']:>+8.1f}"
                f" {p['hw_freq_mult']:>8.2f}"
            )
    print()

# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="India Climate Impact Predictor")
    parser.add_argument("--batch",    action="store_true", help="Run batch scenario table")
    parser.add_argument("--sie",      type=float, default=None)
    parser.add_argument("--ant_melt", type=float, default=200)
    parser.add_argument("--sst_anom", type=float, default=1.2)
    parser.add_argument("--ao",       type=str,   default="neutral")
    parser.add_argument("--year",     type=int,   default=2050)
    parser.add_argument("--ssp",      type=int,   default=2)
    args = parser.parse_args()

    if args.batch:
        batch()
    elif args.sie is not None:
        inputs = dict(
            sie=args.sie, ant_melt=args.ant_melt, sst_anom=args.sst_anom,
            ao_phase=ao_encode(args.ao), year=float(args.year), ssp=float(args.ssp)
        )
        preds = predict_single(**inputs)
        print_results(preds, inputs)
    else:
        interactive()
