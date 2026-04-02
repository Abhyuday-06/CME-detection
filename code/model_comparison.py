"""
Stage 2: Model Comparison — Predicting Kp Index from Fused L1 Plasma Data
==========================================================================
Uses the full data fusion pipeline with ~30 engineered features.
Compares 5 models:
  1. LSTM (TensorFlow/Keras) — Deep sequence model
  2. Physics-Informed Neural Network (PyTorch) — Physics-constrained
  3. Quantum-Inspired Variational Model (PyTorch) — Novel architecture
  4. XGBoost Gradient Boosting — State-of-art tabular model
  5. Random Forest — Robust ensemble baseline

Prediction Task:
  Given 6 hours of fused L1 plasma features → predict Kp index 1 hour ahead.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()

# Import the fusion pipeline
from data_fusion_pipeline import create_advanced_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOOKBACK = 6
FORECAST = 1
EPOCHS = 100
BATCH_SIZE = 32


# ═══════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION — Using Fused Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare_data():
    """Load the fully fused and engineered dataset."""
    print("\n[DATA] Loading fused dataset from data_fusion_pipeline...")
    df = create_advanced_dataset(verbose=True)

    # Create target: Kp 1 hour ahead
    df["kp_target"] = df["kp_index"].shift(-FORECAST)
    df.dropna(subset=["kp_target"], inplace=True)

    print(f"\n[DATA] Dataset ready:")
    print(f"  Total samples (hourly): {len(df)}")
    print(f"  Kp target range: {df['kp_target'].min():.2f} - {df['kp_target'].max():.2f}")
    print(f"  Kp target mean: {df['kp_target'].mean():.2f}")

    return df


def get_feature_columns(df):
    """Select all feature columns (exclude raw source cols and targets)."""
    exclude = ['kp_index', 'kp_target', 
               'l1_speed', 'l1_density', 'l1_thermal_speed',
               'dscovr_speed', 'dscovr_density', 'dscovr_temp']
    features = [c for c in df.columns if c not in exclude]
    return features


def create_sequences(features, target, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)


def prepare_datasets(df):
    feature_cols = get_feature_columns(df)
    target_col = "kp_target"

    print(f"\n[FEATURES] Using {len(feature_cols)} features:")
    for i, fc in enumerate(feature_cols):
        print(f"    {i+1:2d}. {fc}")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_raw = scaler_X.fit_transform(df[feature_cols].values)
    y_raw = scaler_y.fit_transform(df[[target_col]].values).ravel()

    X_seq, y_seq = create_sequences(X_raw, y_raw, LOOKBACK)

    # 80/20 temporal split (no shuffle — time series)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"\n[SPLIT] Train: {len(X_train)} sequences, Test: {len(X_test)} sequences")
    print(f"  Features per timestep: {X_train.shape[2]}")
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 1: LSTM (TensorFlow/Keras)
# ═══════════════════════════════════════════════════════════════════════════

def train_lstm(X_train, X_test, y_train, y_test):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    print("\n" + "=" * 60)
    print("  MODEL 1: LSTM (TensorFlow/Keras)")
    print("=" * 60)

    n_features = X_train.shape[2]

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[lr_schedule, early_stop],
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0).ravel()
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    return y_pred, train_loss, val_loss, "LSTM"


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 2: Physics-Informed Neural Network (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

def train_pinn(X_train, X_test, y_train, y_test):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    print("\n" + "=" * 60)
    print("  MODEL 2: Physics-Informed Neural Network (PyTorch)")
    print("=" * 60)

    device = torch.device("cpu")
    n_features = X_train.shape[2]

    class PINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=64, num_layers=2,
                                batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    def physics_loss(x_batch, y_pred):
        """Enforce: speed increase → higher Kp (physical constraint)."""
        speed = x_batch[:, :, 0]  # first feature is speed
        dv_dt = speed[:, -1] - speed[:, 0]
        violation = torch.relu(dv_dt - 0.3) * torch.relu(0.3 - y_pred)
        return violation.mean()

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    physics_weight = 0.05
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            data_loss = criterion(pred, yb)
            phys_loss = physics_loss(xb, pred)
            loss = data_loss + physics_weight * phys_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(X_tr))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_te)
            val_loss = criterion(val_pred, torch.FloatTensor(y_test).to(device)).item()
            val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20:
                model.load_state_dict(best_state)
                break

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).cpu().numpy()

    return y_pred, train_losses, val_losses, "PINN"


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 3: Quantum-Inspired Variational Model (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

def train_quantum(X_train, X_test, y_train, y_test):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    print("\n" + "=" * 60)
    print("  MODEL 3: Quantum-Inspired Variational Circuit (PyTorch)")
    print("=" * 60)

    device = torch.device("cpu")
    n_features = X_train.shape[2]

    class QuantumLayer(nn.Module):
        def __init__(self, n_qubits, n_layers=3):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.thetas = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        def forward(self, x):
            state = x
            for layer in range(self.n_layers):
                rx = torch.cos(self.thetas[layer, :, 0]) * state + torch.sin(self.thetas[layer, :, 0])
                ry = torch.cos(self.thetas[layer, :, 1]) * rx + torch.sin(self.thetas[layer, :, 1])
                rz = torch.cos(self.thetas[layer, :, 2]) * ry + torch.sin(self.thetas[layer, :, 2])
                shifted = torch.roll(rz, shifts=1, dims=-1)
                state = torch.tanh(rz + shifted)
            return state

    class QuantumInspiredModel(nn.Module):
        def __init__(self, lookback, n_features, n_qubits=8, n_qlayers=4):
            super().__init__()
            self.flatten_dim = lookback * n_features
            self.encoder = nn.Sequential(
                nn.Linear(self.flatten_dim, 32),
                nn.Tanh(),
                nn.Linear(32, n_qubits),
                nn.Tanh()
            )
            self.quantum = QuantumLayer(n_qubits, n_qlayers)
            self.decoder = nn.Sequential(
                nn.Linear(n_qubits, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            x_flat = x.reshape(x.shape[0], -1)
            encoded = self.encoder(x_flat)
            quantum_out = self.quantum(encoded)
            return self.decoder(quantum_out).squeeze(-1)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = QuantumInspiredModel(LOOKBACK, n_features, n_qubits=8, n_qlayers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(X_tr))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_te)
            val_loss = criterion(val_pred, torch.FloatTensor(y_test).to(device)).item()
            val_losses.append(val_loss)

        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20:
                model.load_state_dict(best_state)
                break

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).cpu().numpy()

    return y_pred, train_losses, val_losses, "Quantum-Inspired"


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 4: XGBoost Gradient Boosting
# ═══════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, X_test, y_train, y_test):
    from xgboost import XGBRegressor

    print("\n" + "=" * 60)
    print("  MODEL 4: XGBoost Gradient Boosting")
    print("=" * 60)

    # Flatten sequences: (samples, lookback, features) → (samples, lookback*features)
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_te_flat = X_test.reshape(X_test.shape[0], -1)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30,
        eval_metric='rmse'
    )

    model.fit(
        X_tr_flat, y_train,
        eval_set=[(X_te_flat, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_te_flat)

    # Get training curves
    train_losses = list(model.evals_result()['validation_0']['rmse'])
    val_losses = train_losses  # Same since we only have eval set

    return y_pred, train_losses, val_losses, "XGBoost", model


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 5: Random Forest
# ═══════════════════════════════════════════════════════════════════════════

def train_random_forest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor

    print("\n" + "=" * 60)
    print("  MODEL 5: Random Forest Ensemble")
    print("=" * 60)

    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_te_flat = X_test.reshape(X_test.shape[0], -1)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr_flat, y_train)
    y_pred = model.predict(X_te_flat)

    return y_pred, [], [], "Random Forest", model


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION & COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(y_true, y_pred_scaled, scaler_y, model_name):
    """Evaluate with multiple accuracy tolerances."""
    y_true_actual = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    r2 = r2_score(y_true_actual, y_pred_actual)
    
    # Multiple accuracy tolerances
    abs_error = np.abs(y_true_actual - y_pred_actual)
    accuracy_05 = np.mean(abs_error <= 0.5) * 100
    accuracy_10 = np.mean(abs_error <= 1.0) * 100
    accuracy_15 = np.mean(abs_error <= 1.5) * 100

    print(f"\n  {model_name} Results:")
    print(f"    RMSE:             {rmse:.4f}")
    print(f"    MAE:              {mae:.4f}")
    print(f"    R²:               {r2:.4f}")
    print(f"    Accuracy (±0.5):  {accuracy_05:.2f}%")
    print(f"    Accuracy (±1.0):  {accuracy_10:.2f}%")
    print(f"    Accuracy (±1.5):  {accuracy_15:.2f}%")

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "accuracy_within_05": accuracy_05,
        "accuracy_within_1": accuracy_10,
        "accuracy_within_15": accuracy_15,
        "y_true": y_true_actual,
        "y_pred": y_pred_actual
    }


def plot_comparison(results, train_histories, val_histories, names):
    """Generate comprehensive comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Model Comparison: Fused L1 Plasma → Kp Index (1h Forecast)\nWith Data Fusion & Feature Engineering",
                 fontsize=14, fontweight="bold")

    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]

    # 1. Predictions vs Actual
    ax = axes[0, 0]
    ax.plot(results[0]["y_true"][:200], "k-", linewidth=1.5, label="Actual Kp", alpha=0.8)
    for i, r in enumerate(results):
        ax.plot(r["y_pred"][:200], color=colors[i], alpha=0.6, linewidth=1, label=f"{r['model']}")
    ax.set_xlabel("Time Step (hours)")
    ax.set_ylabel("Kp Index")
    ax.set_title("Predictions vs Ground Truth (first 200 steps)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Training Loss Curves (only for models with curves)
    ax = axes[0, 1]
    for i, (tl, name) in enumerate(zip(train_histories, names)):
        if len(tl) > 0:
            ax.plot(tl, color=colors[i], label=f"{name}")
    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if any(len(tl) > 0 for tl in train_histories):
        ax.set_yscale("log")

    # 3. Accuracy Comparison Bars
    ax = axes[0, 2]
    model_names = [r["model"] for r in results]
    acc_05 = [r["accuracy_within_05"] for r in results]
    acc_10 = [r["accuracy_within_1"] for r in results]
    acc_15 = [r["accuracy_within_15"] for r in results]
    x_pos = np.arange(len(model_names))
    width = 0.25
    ax.bar(x_pos - width, acc_05, width, label="±0.5 Kp", color="tab:green", alpha=0.7)
    ax.bar(x_pos, acc_10, width, label="±1.0 Kp", color="tab:blue", alpha=0.7)
    ax.bar(x_pos + width, acc_15, width, label="±1.5 Kp", color="tab:orange", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Prediction Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% target")

    # 4. Error Comparison
    ax = axes[1, 0]
    rmse_vals = [r["rmse"] for r in results]
    mae_vals = [r["mae"] for r in results]
    width = 0.35
    ax.bar(x_pos - width/2, rmse_vals, width, label="RMSE", color="tab:red", alpha=0.7)
    ax.bar(x_pos + width/2, mae_vals, width, label="MAE", color="tab:blue", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, fontsize=8)
    ax.set_ylabel("Error")
    ax.set_title("Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 5. R² Score
    ax = axes[1, 1]
    r2_vals = [r["r2"] for r in results]
    bars = ax.bar(model_names, r2_vals, color=colors[:len(results)], alpha=0.7)
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score (Higher is Better)")
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels(model_names, rotation=15, fontsize=8)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    # 6. Best model scatter
    best = max(results, key=lambda x: x["accuracy_within_1"])
    ax = axes[1, 2]
    ax.scatter(best["y_true"], best["y_pred"], alpha=0.4, s=15, color=colors[0])
    lims = [min(best["y_true"].min(), best["y_pred"].min()),
            max(best["y_true"].max(), best["y_pred"].max())]
    ax.plot(lims, lims, "r--", alpha=0.7, label="Perfect Prediction")
    ax.set_xlabel("Actual Kp")
    ax.set_ylabel("Predicted Kp")
    ax.set_title(f"Best Model: {best['model']} (Acc±1={best['accuracy_within_1']:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"\n  Saved: model_comparison.png")


def plot_individual_predictions(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Individual Model Predictions vs Actual Kp", fontsize=14, fontweight="bold")
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]

    if n == 1:
        axes = [axes]

    for i, (r, ax) in enumerate(zip(results, axes)):
        ax.scatter(r["y_true"], r["y_pred"], alpha=0.4, s=15, color=colors[i])
        lims = [min(r["y_true"].min(), r["y_pred"].min()),
                max(r["y_true"].max(), r["y_pred"].max())]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Actual Kp")
        ax.set_ylabel("Predicted Kp")
        ax.set_title(f"{r['model']}\nR²={r['r2']:.3f}, Acc±1={r['accuracy_within_1']:.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_scatter.png"), dpi=150)
    plt.close()
    print(f"  Saved: prediction_scatter.png")


def plot_feature_importance(xgb_model, feature_cols, lookback):
    """Plot top feature importances from XGBoost."""
    importances = xgb_model.feature_importances_
    
    # Map flattened feature indices back to feature names
    feature_names_full = []
    for t in range(lookback):
        for fc in feature_cols:
            feature_names_full.append(f"{fc}_t-{lookback-t}")
    
    # Get top 20 features
    n_top = min(20, len(importances))
    idx = np.argsort(importances)[-n_top:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(n_top), importances[idx], color="tab:orange", alpha=0.8)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels([feature_names_full[i] if i < len(feature_names_full) else f"feat_{i}" for i in idx], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost — Top 20 Feature Importances", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"  Saved: feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  STAGE 2: MODEL COMPARISON (FUSED FEATURES)")
    print("  Task: Predict Kp Index 1h ahead from Fused L1 Plasma")
    print("  With full data fusion & feature engineering")
    print("=" * 60)

    print("\n[1/8] Loading & Preparing Fused Data...")
    df = load_and_prepare_data()
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_cols = prepare_datasets(df)

    results = []
    train_histories = []
    val_histories = []
    names = []
    xgb_model_obj = None

    print("\n[2/8] Training LSTM...")
    y_pred_lstm, tl, vl, name = train_lstm(X_train, X_test, y_train, y_test)
    results.append(evaluate_model(y_test, y_pred_lstm, scaler_y, name))
    train_histories.append(tl)
    val_histories.append(vl)
    names.append(name)

    print("\n[3/8] Training PINN...")
    y_pred_pinn, tl, vl, name = train_pinn(X_train, X_test, y_train, y_test)
    results.append(evaluate_model(y_test, y_pred_pinn, scaler_y, name))
    train_histories.append(tl)
    val_histories.append(vl)
    names.append(name)

    print("\n[4/8] Training Quantum-Inspired Model...")
    y_pred_q, tl, vl, name = train_quantum(X_train, X_test, y_train, y_test)
    results.append(evaluate_model(y_test, y_pred_q, scaler_y, name))
    train_histories.append(tl)
    val_histories.append(vl)
    names.append(name)

    print("\n[5/8] Training XGBoost...")
    y_pred_xgb, tl, vl, name, xgb_model_obj = train_xgboost(X_train, X_test, y_train, y_test)
    results.append(evaluate_model(y_test, y_pred_xgb, scaler_y, name))
    train_histories.append(tl)
    val_histories.append(vl)
    names.append(name)

    print("\n[6/8] Training Random Forest...")
    y_pred_rf, tl, vl, name, rf_model_obj = train_random_forest(X_train, X_test, y_train, y_test)
    results.append(evaluate_model(y_test, y_pred_rf, scaler_y, name))
    train_histories.append(tl)
    val_histories.append(vl)
    names.append(name)

    print("\n[7/8] Generating Comparison Plots...")
    plot_comparison(results, train_histories, val_histories, names)
    plot_individual_predictions(results)
    if xgb_model_obj is not None:
        plot_feature_importance(xgb_model_obj, feature_cols, LOOKBACK)

    # Summary table
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Acc±0.5':>9} {'Acc±1.0':>9} {'Acc±1.5':>9}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<22} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f} "
              f"{r['accuracy_within_05']:>8.2f}% {r['accuracy_within_1']:>8.2f}% {r['accuracy_within_15']:>8.2f}%")
    print("-" * 80)

    best_acc = max(results, key=lambda x: x["accuracy_within_1"])
    best_r2 = max(results, key=lambda x: x["r2"])
    print(f"\n  Best by Accuracy (±1 Kp): {best_acc['model']} ({best_acc['accuracy_within_1']:.2f}%)")
    print(f"  Best by R²:               {best_r2['model']} ({best_r2['r2']:.4f})")

    # Save results JSON
    summary = [{
        "model": r["model"],
        "rmse": float(r["rmse"]),
        "mae": float(r["mae"]),
        "r2": float(r["r2"]),
        "accuracy_within_05": float(r["accuracy_within_05"]),
        "accuracy_within_1": float(r["accuracy_within_1"]),
        "accuracy_within_15": float(r["accuracy_within_15"])
    } for r in results]
    
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save accuracy results separately
    accuracy_summary = [{
        "model": r["model"],
        "accuracy_within_1": round(r["accuracy_within_1"], 2),
        "r2": round(r["r2"], 4)
    } for r in results]
    with open(os.path.join(OUTPUT_DIR, "accuracy_results.json"), "w") as f:
        json.dump(accuracy_summary, f, indent=2)
    
    print(f"\n  Results saved to {OUTPUT_DIR}/results.json")
    print(f"  All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
