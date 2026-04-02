"""
Static vs Live Model Comparison
================================
Trains separate LSTM + XGBoost models on:
  1. STATIC data (NASA OMNI2 historical)
  2. LIVE data (NOAA DSCOVR real-time)

Uses the full data fusion pipeline with engineered features.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_fusion_pipeline import create_advanced_dataset

LOOKBACK_HOURS = 12
FORECAST_HOURS = 1

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_feature_cols(df):
    """Get feature columns (exclude raw source columns and target)."""
    exclude = ['kp_index', 'target',
               'l1_speed', 'l1_density', 'l1_thermal_speed',
               'dscovr_speed', 'dscovr_density', 'dscovr_temp']
    return [c for c in df.columns if c not in exclude]


def prepare_sequences(df, target_col='kp_index'):
    df = df.copy()
    df['target'] = df[target_col].shift(-FORECAST_HOURS)
    df.dropna(inplace=True)

    features = get_feature_cols(df)
    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df[features])
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(df[['target']])

    X, y = [], []
    for i in range(len(X_scaled) - LOOKBACK_HOURS):
        X.append(X_scaled[i : i + LOOKBACK_HOURS])
        y.append(y_scaled[i + LOOKBACK_HOURS])

    return np.array(X), np.array(y), scaler_y, features


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_xgboost_model(X_train, X_test, y_train, y_test):
    """Train XGBoost on flattened sequences."""
    from xgboost import XGBRegressor
    
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_te_flat = X_test.reshape(X_test.shape[0], -1)
    
    model = XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
        early_stopping_rounds=20, eval_metric='rmse'
    )
    model.fit(X_tr_flat, y_train.ravel(),
              eval_set=[(X_te_flat, y_test.ravel())], verbose=False)
    
    y_pred = model.predict(X_te_flat)
    return y_pred


def train_and_eval(df, name):
    print(f"\n{'='*60}\nTraining {name} Models (LSTM + XGBoost)\n{'='*60}")
    
    X, y, scaler_y, features = prepare_sequences(df)
    
    if len(X) < 100:
        print(f"Not enough data for {name} ({len(X)} sequences).")
        return None
    
    print(f"  Features: {len(features)}, Sequences: {len(X)}")
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # LSTM
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(
        X_train, y_train, epochs=50, batch_size=32,
        validation_split=0.1, verbose=0,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=8, factor=0.5)
        ]
    )
    
    y_pred_lstm = model.predict(X_test, verbose=0)
    y_pred_lstm_actual = scaler_y.inverse_transform(y_pred_lstm)
    y_true = scaler_y.inverse_transform(y_test)
    
    lstm_mae = mean_absolute_error(y_true, y_pred_lstm_actual)
    lstm_r2 = r2_score(y_true, y_pred_lstm_actual)
    lstm_acc = np.mean(np.abs(y_true - y_pred_lstm_actual) <= 1.0) * 100
    
    print(f"\n  {name} LSTM Results:")
    print(f"    MAE: {lstm_mae:.4f} | R²: {lstm_r2:.4f} | Accuracy ±1: {lstm_acc:.2f}%")
    
    # XGBoost
    y_pred_xgb = train_xgboost_model(X_train, X_test, y_train, y_test)
    y_pred_xgb_actual = scaler_y.inverse_transform(y_pred_xgb.reshape(-1, 1))
    
    xgb_mae = mean_absolute_error(y_true, y_pred_xgb_actual)
    xgb_r2 = r2_score(y_true, y_pred_xgb_actual)
    xgb_acc = np.mean(np.abs(y_true - y_pred_xgb_actual) <= 1.0) * 100
    
    print(f"\n  {name} XGBoost Results:")
    print(f"    MAE: {xgb_mae:.4f} | R²: {xgb_r2:.4f} | Accuracy ±1: {xgb_acc:.2f}%")
    
    return {
        'lstm': {
            'mae': lstm_mae, 'r2': lstm_r2, 'accuracy': lstm_acc,
            'y_true': y_true, 'y_pred': y_pred_lstm_actual
        },
        'xgboost': {
            'mae': xgb_mae, 'r2': xgb_r2, 'accuracy': xgb_acc,
            'y_true': y_true, 'y_pred': y_pred_xgb_actual
        },
        'history': history
    }


def main():
    print("\n" + "=" * 60)
    print("  STATIC vs LIVE MODEL COMPARISON")
    print("=" * 60)
    
    # 1. Static (NASA OMNI2)
    print("\nLoading STATIC Historical Data (NASA OMNI2)...")
    df_static = create_advanced_dataset(source_filter='NASA_OMNI2', verbose=False)
    
    # 2. Live (NOAA DSCOVR)
    print("Loading LIVE Operational Data (NOAA DSCOVR)...")
    df_live = create_advanced_dataset(source_filter='NOAA_DSCOVR', verbose=False)
    
    results = {}
    
    if not df_static.empty and len(df_static) > 200:
        results['STATIC'] = train_and_eval(df_static, "Static (OMNI2)")
    else:
        print(f"Static Data insufficient ({len(df_static)} rows).")
        
    if not df_live.empty and len(df_live) > 200:
        results['LIVE'] = train_and_eval(df_live, "Live (NOAA L1)")
    else:
        print(f"Live Data insufficient ({len(df_live)} rows).")
    
    # Generate comparison plot
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Static vs Live Data Models: Kp Forecast Accuracy", fontsize=14, fontweight="bold")
        
        # Bar chart comparison
        ax = axes[0]
        model_types = []
        accs = []
        colors = []
        for source, res in results.items():
            for mtype in ['lstm', 'xgboost']:
                if mtype in res:
                    model_types.append(f"{source}\n{mtype.upper()}")
                    accs.append(res[mtype]['accuracy'])
                    colors.append('tab:blue' if source == 'STATIC' else 'tab:red')
        
        bars = ax.bar(model_types, accs, color=colors, alpha=0.7)
        ax.set_ylabel("Accuracy (±1 Kp) %")
        ax.set_title("Accuracy Comparison")
        ax.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="90% target")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        
        # Prediction plot (best model from each source)
        ax = axes[1]
        for source, res in results.items():
            best_model = max(['lstm', 'xgboost'], key=lambda m: res[m]['accuracy'] if m in res else 0)
            y_true = res[best_model]['y_true'][:100]
            y_pred = res[best_model]['y_pred'][:100]
            
            if source == 'STATIC':
                ax.plot(y_true, 'k--', alpha=0.5, label='Actual Kp', linewidth=1)
            ax.plot(y_pred, label=f"{source} ({best_model.upper()})",
                    color='tab:blue' if source == 'STATIC' else 'tab:red',
                    alpha=0.7, linewidth=1.5)
        
        ax.set_title("Predictions (first 100 steps)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Kp Index")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "live_vs_static_plot.png"), dpi=150)
        plt.close()
        print(f"\nSaved comparison plot to {OUTPUT_DIR}/live_vs_static_plot.png")
        
        # Save summary
        summary = {}
        for source, res in results.items():
            summary[source] = {
                mtype: {'mae': float(res[mtype]['mae']), 'r2': float(res[mtype]['r2']),
                         'accuracy_within_1': float(res[mtype]['accuracy'])}
                for mtype in ['lstm', 'xgboost'] if mtype in res
            }
        with open(os.path.join(OUTPUT_DIR, "live_vs_static_results.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {OUTPUT_DIR}/live_vs_static_results.json")


if __name__ == "__main__":
    main()
