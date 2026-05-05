import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 120
})

tf.random.set_seed(42)
np.random.seed(42)

def build_lstm_model(seq_len, n_features):
    inp = keras.Input(shape=(seq_len, n_features))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model

def train_lstm_target(X_seq, y_seq, years_seq, target_name, epochs=120):
    split = int(len(X_seq) * 0.80)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # FIX: Use ONE scaler for all features. 
    # LSTMs expect (Samples, TimeSteps, Features). We flatten to 2D to scale.
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    
    n_feat = X_train.shape[2]
    X_train_reshaped = X_train.reshape(-1, n_feat)
    X_test_reshaped = X_test.reshape(-1, n_feat)
    
    X_train_s = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_s = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

    # Target scaler
    y_sc = MinMaxScaler(feature_range=(0, 1))
    y_train_s = y_sc.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = build_lstm_model(X_seq.shape[1], X_seq.shape[2])

    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)
    ]

    model.fit(
        X_train_s, y_train_s,
        epochs=epochs,
        batch_size=8,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )

    y_pred_s = model.predict(X_test_s, verbose=0).ravel()
    y_pred = y_sc.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    metrics = {
        'R2': round(r2_score(y_test, y_pred), 3),
        'MAE': round(mean_absolute_error(y_test, y_pred), 3),
        'test_years': years_seq[split:]
    }

    print(f"    {target_name:25s}: R²={metrics['R2']:.3f}  MAE={metrics['MAE']:.3f}")

    return model, scaler_X, y_sc, y_pred, metrics

def project_future_lstm(model, scaler_X, y_sc, last_window, n_years=76,
                        warming_rate=0.028, sie_decline=0.01):
    window = last_window.copy()
    preds = []

    for i in range(n_years):
        # FIX: Scale the whole window at once. Scaler now expects 10 columns.
        window_s = scaler_X.transform(window)
        
        pred_s = model.predict(window_s[np.newaxis], verbose=0)[0, 0]
        pred = y_sc.inverse_transform([[pred_s]])[0, 0]
        preds.append(float(pred))

        # Advance window
        new_row = window[-1].copy()
        new_row[0] += sie_decline
        new_row[1] += sie_decline * 0.8
        new_row[2] += warming_rate * 0.6
        new_row[3] = np.random.normal(0, 0.9)
        new_row[4] = np.random.normal(-0.15, 1.1)
        new_row[5] += warming_rate
        new_row[6] = -new_row[1] * np.sign(new_row[4])
        new_row[7] = -new_row[0] * new_row[2]
        window = np.vstack([window[1:], new_row])

    return np.array(preds)

def run_all_lstm_targets(df, ssp_scenarios):
    from feature_engineering_02 import prepare_lstm_sequences
    
    targets = {
        'heatwave_days': ('Heat wave days/year (ETCCDI, IMD+ERA5)', '#e24b4a'),
        'mhw_days': ('Marine heat wave days (HadSST+ERSST)', '#378add'),
        'nw_igp_ratio': ('NW/IGP monsoon shift index (GPCP)', '#1d9e75'),
    }

    results = {}
    future_years = np.arange(2025, 2101)

    for target_col, (target_label, color) in targets.items():
        print(f"\n Training LSTM → {target_col}")
        X_seq, y_seq, years_seq, forcing_cols = prepare_lstm_sequences(df, target_col, seq_len=10)
        
        model, scaler_X, y_sc, y_pred_test, metrics = train_lstm_target(
            X_seq, y_seq, years_seq, target_label
        )

        last_window = X_seq[-1]

        ssp_projections = {}
        for _, ssp_row in ssp_scenarios.iterrows():
            warming = ssp_row['warming_2100_c']
            sie_rate = -0.010 - warming * 0.003
            proj = project_future_lstm(
                model, scaler_X, y_sc, last_window,
                n_years=76, warming_rate=warming * 0.013, sie_decline=sie_rate
            )
            ssp_projections[ssp_row['scenario']] = np.round(proj, 2)

        results[target_col] = {
            'model': model, 'scaler_X': scaler_X, 'y_sc': y_sc,
            'y_pred_test': y_pred_test, 'metrics': metrics,
            'years_seq': years_seq, 'y_seq': y_seq,
            'ssp_projections': ssp_projections,
            'label': target_label, 'color': color,
            'forcing_cols': forcing_cols
        }

        if not os.path.exists('models'): os.makedirs('models')
        model.save(f"models/lstm_{target_col}.keras")

    return results, future_years

def plot_lstm_results(df, results, future_years, ssp_scenarios):
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    ssp_colors = {'SSP1-2.6': '#1d9e75', 'SSP2-4.5': '#378add', 'SSP3-7.0': '#ef9f27', 'SSP5-8.5': '#e24b4a'}
    target_list = ['heatwave_days', 'mhw_days', 'nw_igp_ratio']
    ylabels = ['Heat wave days/year', 'MHW days/year', 'NW/IGP rainfall ratio']

    for row_idx, (target_col, ylabel) in enumerate(zip(target_list, ylabels)):
        res = results[target_col]
        obs = df.set_index('year')[target_col]

        ax_l = fig.add_subplot(gs[row_idx, 0])
        ax_l.plot(obs.index, obs.values, color='#2c3e50', linewidth=1.5, label='Observed')
        ax_l.plot(res['metrics']['test_years'], res['y_pred_test'], color=res['color'], linestyle='--', label='LSTM test set')
        ax_l.set_ylabel(ylabel)
        ax_l.set_title(f"LSTM Hindcast: {res['label']}")
        ax_l.legend()

        ax_r = fig.add_subplot(gs[row_idx, 1])
        ax_r.plot(obs.index[-20:], obs.values[-20:], color='#2c3e50', alpha=0.6, label='Observed (Recent)')

        for ssp_name, proj in res['ssp_projections'].items():
            smooth = pd.Series(proj).rolling(5, center=True, min_periods=2).mean().values
            ax_r.plot(future_years, smooth, color=ssp_colors.get(ssp_name, 'gray'), label=ssp_name)

        ax_r.set_ylabel(ylabel)
        ax_r.set_title(f"SSP Projections 2025–2100")
        ax_r.legend(loc='upper left')

    if not os.path.exists('outputs'): os.makedirs('outputs')
    plt.savefig("outputs/04_lstm_projections.png", bbox_inches='tight', dpi=150)
    plt.close()

def run_lstm_model(df, ssp_scenarios):
    print("\n" + "="*60)
    print("MODULE 4: LSTM Projection Models")
    print("="*60)
    results, future_years = run_all_lstm_targets(df, ssp_scenarios)
    plot_lstm_results(df, results, future_years, ssp_scenarios)

    # Save LSTM projections CSV
    rows = []
    for target_col, res in results.items():
        for ssp_name, proj in res['ssp_projections'].items():
            for i, yr in enumerate(future_years):
                rows.append({'year': yr, 'target': target_col,
                             'scenario': ssp_name, 'value': proj[i]})
    pd.DataFrame(rows).to_csv("outputs/lstm_projections_2025_2100.csv", index=False)
    print("  Saved: outputs/lstm_projections_2025_2100.csv")

    return results, future_years

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from data_generator_01 import generate_all_datasets
    df, grace, slr, ssps = generate_all_datasets()
    run_lstm_model(df, ssps)