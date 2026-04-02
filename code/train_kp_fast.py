"""
Dashboard Model Training — Kp 6-Hour Forecast
================================================
Trains the model used by the web dashboard for real-time 1h-6h predictions.
Uses the data fusion pipeline for better features and more epochs for accuracy.
"""

import os
import json
import pickle
import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 24
N_FORECAST = 6

print("=" * 60)
print("  DASHBOARD MODEL TRAINING (Kp 6-Hour Forecast)")
print("=" * 60)

# Load data using the fused pipeline approach
print("\n[1/5] Loading data...")
conn = psycopg2.connect(DB_URI)
query = """
    SELECT 
        date_trunc('hour', s.observation_time) as obs_hour,
        AVG(s.proton_speed) as speed, 
        AVG(s.proton_density) as density,
        AVG(s.proton_thermal_speed) as thermal_speed,
        AVG(k.kp_index) as kp
    FROM swis_moments s
    LEFT JOIN geomagnetic_indices k 
       ON date_trunc('hour', s.observation_time) = date_trunc('hour', k.timestamp)
    GROUP BY obs_hour
    ORDER BY obs_hour ASC
"""
df = pd.read_sql(query, conn)
conn.close()

df.set_index('obs_hour', inplace=True)
df = df.interpolate(method='linear', limit=4).ffill(limit=12).bfill()
df = df.fillna(df.median())

# Engineer features for the dashboard model
print("[2/5] Engineering features...")
df['dynamic_pressure'] = df['density'] * (df['speed'] ** 2) * 1.6726e-6
df['speed_gradient'] = df['speed'].diff().fillna(0)
df['density_gradient'] = df['density'].diff().fillna(0)
df['speed_variance_6h'] = df['speed'].rolling(6, min_periods=1).var()
df['density_variance_6h'] = df['density'].rolling(6, min_periods=1).var()
df['speed_ema'] = df['speed'].ewm(span=6, min_periods=1).mean()
df['kp_lag_1h'] = df['kp'].shift(1).fillna(method='bfill')
df['kp_lag_3h'] = df['kp'].shift(3).fillna(method='bfill')
df['kp_rolling_mean'] = df['kp'].rolling(6, min_periods=1).mean()
df['speed_x_density'] = df['speed'] * df['density']
df['momentum_flux'] = df['density'] * df['speed']

df.fillna(0, inplace=True)

print(f"  Total rows: {len(df)}")
print(f"  Features: {df.columns.tolist()}")

# Create multi-step targets
for i in range(1, N_FORECAST + 1):
    df[f'kp+{i}h'] = df['kp'].shift(-i)

df.dropna(inplace=True)

if len(df) == 0:
    print("No data available after dropna!")
    exit()

# Feature columns (exclude kp and targets)
feature_cols = [c for c in df.columns if not c.startswith('kp')]
feature_cols.append('kp')  # Include current kp as a feature
target_cols = [f'kp+{i}h' for i in range(1, N_FORECAST + 1)]

print(f"\n[3/5] Preparing sequences...")
print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

features = df[feature_cols].values
targets = df[target_cols].values

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
scaler_Y = MinMaxScaler()
y_scaled = scaler_Y.fit_transform(targets)

X, Y = [], []
for i in range(len(X_scaled) - SEQ_LEN):
    X.append(X_scaled[i:i+SEQ_LEN])
    Y.append(y_scaled[i+SEQ_LEN])

X = np.array(X)
Y = np.array(Y)

# Train/test split
split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print(f"  Training shape: {X_train.shape}, Test shape: {X_test.shape}")

# Build and train model
print(f"\n[4/5] Training model (up to 80 epochs with early stopping)...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(feature_cols))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(N_FORECAST)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, Y_train,
    epochs=80,
    batch_size=32,
    validation_data=(X_test, Y_test),
    verbose=1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5, min_lr=1e-5)
    ]
)

# Evaluate
print(f"\n[5/5] Evaluating...")
Y_pred_scaled = model.predict(X_test, verbose=0)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_true = scaler_Y.inverse_transform(Y_test)

# Per-hour metrics
print(f"\n  {'Forecast':>10} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Acc±1':>8}")
print(f"  {'-'*46}")
metrics = {}
for i in range(N_FORECAST):
    mae = mean_absolute_error(Y_true[:, i], Y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(Y_true[:, i], Y_pred[:, i]))
    r2 = r2_score(Y_true[:, i], Y_pred[:, i])
    acc = np.mean(np.abs(Y_true[:, i] - Y_pred[:, i]) <= 1.0) * 100
    
    print(f"  {'Kp+' + str(i+1) + 'h':>10} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {acc:>7.1f}%")
    metrics[f'kp+{i+1}h'] = {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2), 'accuracy': float(acc)}

# Overall 1h accuracy (most important)
overall_acc_1h = metrics['kp+1h']['accuracy']
print(f"\n  ★ 1-Hour Ahead Accuracy (±1 Kp): {overall_acc_1h:.1f}%")

# Save model and scalers
model.save(os.path.join(BASE_DIR, 'kp_model.keras'))
with open(os.path.join(BASE_DIR, 'kp_scaler_X.pkl'), 'wb') as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(BASE_DIR, 'kp_scaler_Y.pkl'), 'wb') as f:
    pickle.dump(scaler_Y, f)

# Save dashboard model metrics
with open(os.path.join(OUTPUT_DIR, 'dashboard_model_metrics.json'), 'w') as f:
    json.dump({
        'model_type': 'LSTM',
        'sequence_length': SEQ_LEN,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'per_hour_metrics': metrics,
        'best_1h_accuracy': overall_acc_1h
    }, f, indent=2)

print(f"\n  Model saved: kp_model.keras")
print(f"  Scalers saved: kp_scaler_X.pkl, kp_scaler_Y.pkl")
print(f"  Metrics saved: {OUTPUT_DIR}/dashboard_model_metrics.json")
print("  Done!")
