"""
Stage 2: Model Comparison — Predicting Kp Index from L1 Plasma Data
====================================================================
Without feature extraction. Compare:
  1. LSTM (TensorFlow/Keras)
  2. Physics-Informed Neural Network (PyTorch)
  3. Quantum-Inspired Variational Model (PyTorch)

Prediction Task:
  Given 6 hours of L1 plasma (speed, density) → predict Kp index 1 hour ahead.
"""

import os
import json
import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOOKBACK = 6
FORECAST = 1
EPOCHS = 80
BATCH_SIZE = 16


# ═══════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare_data():
    conn = psycopg2.connect(DB_URI)

    df_dscovr = pd.read_sql("""
        SELECT observation_time, proton_speed, proton_density
        FROM dscovr_plasma
        WHERE proton_speed IS NOT NULL AND proton_density IS NOT NULL
        ORDER BY observation_time ASC
    """, conn, parse_dates=["observation_time"])

    df_kp = pd.read_sql("""
        SELECT timestamp AS observation_time, kp_index
        FROM geomagnetic_indices
        ORDER BY timestamp ASC
    """, conn, parse_dates=["observation_time"])

    conn.close()

    # Find overlap period
    start = max(df_dscovr["observation_time"].min(), df_kp["observation_time"].min())
    end = min(df_dscovr["observation_time"].max(), df_kp["observation_time"].max())
    print(f"Overlap period: {start} to {end}")

    # Resample DSCOVR to hourly
    df_dscovr = df_dscovr[(df_dscovr["observation_time"] >= start) & (df_dscovr["observation_time"] <= end)]
    df_dscovr = df_dscovr.set_index("observation_time").resample("1h").mean().dropna()

    # Resample Kp to hourly (forward-fill since Kp is 3-hourly)
    df_kp = df_kp[(df_kp["observation_time"] >= start) & (df_kp["observation_time"] <= end)]
    df_kp = df_kp.drop_duplicates(subset="observation_time").set_index("observation_time")
    df_kp = df_kp.resample("1h").ffill().dropna()

    # Merge
    df = df_dscovr.join(df_kp, how="inner")
    df.dropna(inplace=True)

    # Create target: Kp 1 hour ahead
    df["kp_target"] = df["kp_index"].shift(-FORECAST)
    df.dropna(inplace=True)

    print(f"Training samples (hourly): {len(df)}")
    print(f"Features: proton_speed, proton_density")
    print(f"Target: kp_index (1h ahead)")
    print(f"Speed range: {df['proton_speed'].min():.1f} - {df['proton_speed'].max():.1f} km/s")
    print(f"Kp range: {df['kp_target'].min():.2f} - {df['kp_target'].max():.2f}")

    return df


def create_sequences(features, target, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)


def prepare_datasets(df):
    feature_cols = ["proton_speed", "proton_density"]
    target_col = "kp_target"

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_raw = scaler_X.fit_transform(df[feature_cols].values)
    y_raw = scaler_y.fit_transform(df[[target_col]].values).ravel()

    X_seq, y_seq = create_sequences(X_raw, y_raw, LOOKBACK)

    # 80/20 split
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"Train: {len(X_train)} sequences, Test: {len(X_test)} sequences")
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 1: LSTM (TensorFlow/Keras)
# ═══════════════════════════════════════════════════════════════════════════

def train_lstm(X_train, X_test, y_train, y_test):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    print("\n" + "=" * 60)
    print("  MODEL 1: LSTM (TensorFlow/Keras)")
    print("=" * 60)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(LOOKBACK, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
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

    class PINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    # Physics loss: Kp should increase when speed increases rapidly
    # dKp/dt ~ f(dV/dt) — enforce correlation between speed change rate and Kp change
    def physics_loss(x_batch, y_pred):
        speed = x_batch[:, :, 0]  # (batch, lookback)
        dv_dt = speed[:, -1] - speed[:, 0]  # speed change over window
        # Higher speed increase → higher predicted Kp (soft constraint)
        # Penalize when speed increases but predicted Kp is low
        violation = torch.relu(dv_dt - 0.3) * torch.relu(0.3 - y_pred)
        return violation.mean()

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    physics_weight = 0.1

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

    class QuantumLayer(nn.Module):
        """
        Simulates a variational quantum circuit with parameterized rotation gates.
        Uses Rx, Ry, Rz rotation unitary matrices applied to a classical state
        vector, emulating qubit rotations without a quantum backend.
        """
        def __init__(self, n_qubits, n_layers=3):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            # Trainable rotation angles (Rx, Ry, Rz per qubit per layer)
            self.thetas = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        def forward(self, x):
            # x: (batch, n_qubits) — encoded input features
            batch_size = x.shape[0]
            state = x

            for layer in range(self.n_layers):
                # Apply parameterized rotations
                rx = torch.cos(self.thetas[layer, :, 0]) * state + torch.sin(self.thetas[layer, :, 0])
                ry = torch.cos(self.thetas[layer, :, 1]) * rx + torch.sin(self.thetas[layer, :, 1])
                rz = torch.cos(self.thetas[layer, :, 2]) * ry + torch.sin(self.thetas[layer, :, 2])

                # Entanglement: circular shift + combine (simulates CNOT-like coupling)
                shifted = torch.roll(rz, shifts=1, dims=-1)
                state = torch.tanh(rz + shifted)

            return state

    class QuantumInspiredModel(nn.Module):
        def __init__(self, lookback, n_features, n_qubits=4, n_qlayers=3):
            super().__init__()
            self.flatten_dim = lookback * n_features
            # Classical encoder: compress sequence to qubit-count dims
            self.encoder = nn.Sequential(
                nn.Linear(self.flatten_dim, 16),
                nn.Tanh(),
                nn.Linear(16, n_qubits),
                nn.Tanh()
            )
            self.quantum = QuantumLayer(n_qubits, n_qlayers)
            # Classical decoder
            self.decoder = nn.Sequential(
                nn.Linear(n_qubits, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )

        def forward(self, x):
            # x: (batch, lookback, features)
            x_flat = x.reshape(x.shape[0], -1)
            encoded = self.encoder(x_flat)
            quantum_out = self.quantum(encoded)
            return self.decoder(quantum_out).squeeze(-1)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = QuantumInspiredModel(LOOKBACK, 2, n_qubits=4, n_qlayers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

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

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).cpu().numpy()

    return y_pred, train_losses, val_losses, "Quantum-Inspired"


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION & COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(y_true, y_pred_scaled, scaler_y, model_name):
    # Inverse transform
    y_true_actual = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    r2 = r2_score(y_true_actual, y_pred_actual)
    # Accuracy: % of predictions within ±1 Kp unit (standard space weather tolerance)
    accuracy_within_1 = np.mean(np.abs(y_true_actual - y_pred_actual) <= 1.0) * 100

    print(f"\n  {model_name} Results:")
    print(f"    RMSE:           {rmse:.4f}")
    print(f"    MAE:            {mae:.4f}")
    print(f"    R²:             {r2:.4f}")
    print(f"    Accuracy (±1):  {accuracy_within_1:.2f}%")

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "accuracy_within_1": accuracy_within_1,
        "y_true": y_true_actual,
        "y_pred": y_pred_actual
    }


def plot_comparison(results, train_histories, val_histories, names):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Comparison: L1 Plasma → Kp Index (1h Forecast)\nWithout Feature Extraction",
                 fontsize=14, fontweight="bold")

    # 1. Predictions vs Actual
    ax = axes[0, 0]
    ax.plot(results[0]["y_true"], "k-", linewidth=1.5, label="Actual Kp")
    colors = ["tab:blue", "tab:red", "tab:green"]
    for i, r in enumerate(results):
        ax.plot(r["y_pred"], color=colors[i], alpha=0.7, linewidth=1, label=f"{r['model']} Predicted")
    ax.set_xlabel("Time Step (hours)")
    ax.set_ylabel("Kp Index")
    ax.set_title("Predictions vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Training Loss Curves
    ax = axes[0, 1]
    for i, (tl, name) in enumerate(zip(train_histories, names)):
        ax.plot(tl, color=colors[i], label=f"{name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title("Training Loss Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # 3. Error Bars
    ax = axes[1, 0]
    model_names = [r["model"] for r in results]
    rmse_vals = [r["rmse"] for r in results]
    mae_vals = [r["mae"] for r in results]
    x_pos = np.arange(len(model_names))
    width = 0.35
    ax.bar(x_pos - width/2, rmse_vals, width, label="RMSE", color="tab:red", alpha=0.7)
    ax.bar(x_pos + width/2, mae_vals, width, label="MAE", color="tab:blue", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Error")
    ax.set_title("Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4. R² Score comparison
    ax = axes[1, 1]
    r2_vals = [r["r2"] for r in results]
    bars = ax.bar(model_names, r2_vals, color=colors[:len(results)], alpha=0.7)
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score (Higher is Better)")
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"\n  Saved: model_comparison.png")


def plot_individual_predictions(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Individual Model Predictions vs Actual Kp", fontsize=14, fontweight="bold")
    colors = ["tab:blue", "tab:red", "tab:green"]

    for i, (r, ax) in enumerate(zip(results, axes)):
        ax.scatter(r["y_true"], r["y_pred"], alpha=0.5, s=20, color=colors[i])
        lims = [
            min(r["y_true"].min(), r["y_pred"].min()),
            max(r["y_true"].max(), r["y_pred"].max())
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect Prediction")
        ax.set_xlabel("Actual Kp")
        ax.set_ylabel("Predicted Kp")
        ax.set_title(f"{r['model']} (R²={r['r2']:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_scatter.png"), dpi=150)
    plt.close()
    print(f"  Saved: prediction_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  STAGE 2: MODEL COMPARISON")
    print("  Task: Predict Kp Index 1h ahead from L1 Plasma Data")
    print("  No feature extraction applied")
    print("=" * 60)

    print("\n[1/5] Loading & Preparing Data...")
    df = load_and_prepare_data()
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_datasets(df)

    results = []
    train_histories = []
    val_histories = []
    names = []

    print("\n[2/5] Training LSTM...")
    y_pred_lstm, tl, vl, name = train_lstm(X_train, X_test, y_train, y_test)
    results.append(evaluate_model(y_test, y_pred_lstm, scaler_y, name))
    train_histories.append(tl)
    val_histories.append(vl)
    names.append(name)

    print("\n[3/5] Training PINN...")
    y_pred_pinn, tl, vl, name = train_pinn(X_train, X_test, y_train, y_test)
    results.append(evaluate_model(y_test, y_pred_pinn, scaler_y, name))
    train_histories.append(tl)
    val_histories.append(vl)
    names.append(name)

    print("\n[4/5] Training Quantum-Inspired Model...")
    y_pred_q, tl, vl, name = train_quantum(X_train, X_test, y_train, y_test)
    results.append(evaluate_model(y_test, y_pred_q, scaler_y, name))
    train_histories.append(tl)
    val_histories.append(vl)
    names.append(name)

    print("\n[5/5] Generating Comparison Plots...")
    plot_comparison(results, train_histories, val_histories, names)
    plot_individual_predictions(results)

    # Summary table
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Acc±1':>9}")
    print("-" * 62)
    for r in results:
        print(f"{r['model']:<25} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f} {r['accuracy_within_1']:>8.2f}%")
    print("-" * 62)

    best = min(results, key=lambda x: x["rmse"])
    print(f"\nBest model by RMSE: {best['model']}")

    # Save results JSON
    summary = [{
        "model": r["model"],
        "rmse": float(r["rmse"]),
        "mae": float(r["mae"]),
        "r2": float(r["r2"]),
        "accuracy_within_1": float(r["accuracy_within_1"])
    } for r in results]
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR}/results.json")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
