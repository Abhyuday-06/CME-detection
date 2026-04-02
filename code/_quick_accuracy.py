"""
Quick accuracy check: trains all 3 models with reduced epochs
to compute accuracy-within-1-Kp for the resume.
"""
import os, json
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')

LOOKBACK = 6
FORECAST = 1
EPOCHS   = 25   # enough to converge for reporting purposes
BATCH    = 64

# ── DATA ──────────────────────────────────────────────────────────────────
def load_data():
    conn = psycopg2.connect(DB_URI)
    df_d = pd.read_sql("SELECT observation_time, proton_speed, proton_density FROM dscovr_plasma WHERE proton_speed IS NOT NULL AND proton_density IS NOT NULL ORDER BY observation_time", conn, parse_dates=["observation_time"])
    df_k = pd.read_sql("SELECT timestamp AS observation_time, kp_index FROM geomagnetic_indices ORDER BY timestamp", conn, parse_dates=["observation_time"])
    conn.close()

    start = max(df_d["observation_time"].min(), df_k["observation_time"].min())
    end   = min(df_d["observation_time"].max(), df_k["observation_time"].max())

    df_d = df_d[(df_d["observation_time"]>=start)&(df_d["observation_time"]<=end)].set_index("observation_time").resample("1h").mean().dropna()
    df_k = df_k[(df_k["observation_time"]>=start)&(df_k["observation_time"]<=end)].drop_duplicates("observation_time").set_index("observation_time").resample("1h").ffill().dropna()

    df = df_d.join(df_k, how="inner")
    df["kp_target"] = df["kp_index"].shift(-FORECAST)
    df.dropna(inplace=True)
    print(f"Samples: {len(df)}  |  Kp range: {df['kp_target'].min():.2f} – {df['kp_target'].max():.2f}")
    return df

def make_seqs(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys)

def prepare(df):
    sX, sy = MinMaxScaler(), MinMaxScaler()
    X = sX.fit_transform(df[["proton_speed","proton_density"]].values)
    y = sy.fit_transform(df[["kp_target"]].values).ravel()
    Xs, ys = make_seqs(X, y, LOOKBACK)
    split = int(len(Xs)*0.8)
    return Xs[:split], Xs[split:], ys[:split], ys[split:], sy

def accuracy(y_true, y_pred, tol=1.0):
    return np.mean(np.abs(y_true - y_pred) <= tol) * 100

# ── LSTM ──────────────────────────────────────────────────────────────────
def run_lstm(Xtr, Xte, ytr, yte, sy):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    m = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(LOOKBACK,2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH, validation_data=(Xte,yte), verbose=0)
    yp = sy.inverse_transform(m.predict(Xte, verbose=0)).ravel()
    yt = sy.inverse_transform(yte.reshape(-1,1)).ravel()
    return yt, yp, "LSTM"

# ── PINN ──────────────────────────────────────────────────────────────────
def run_pinn(Xtr, Xte, ytr, yte, sy):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    class PINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(2, 32, batch_first=True)
            self.fc   = nn.Sequential(nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:,-1,:]).squeeze(-1)

    def phys(xb, yp):
        dv = xb[:,:,0][:,-1] - xb[:,:,0][:,0]
        return (torch.relu(dv-0.3)*torch.relu(0.3-yp)).mean()

    Xt = torch.FloatTensor(Xtr); yt_ = torch.FloatTensor(ytr)
    Xe = torch.FloatTensor(Xte)
    ds = TensorDataset(Xt, yt_); dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model = PINN(); opt = torch.optim.Adam(model.parameters(), lr=0.001); crit = nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            p = model(xb)
            (crit(p, yb) + 0.1*phys(xb, p)).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        yp = model(Xe).numpy()
    yp = sy.inverse_transform(yp.reshape(-1,1)).ravel()
    yt = sy.inverse_transform(yte.reshape(-1,1)).ravel()
    return yt, yp, "PINN"

# ── QUANTUM ───────────────────────────────────────────────────────────────
def run_quantum(Xtr, Xte, ytr, yte, sy):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    class QL(nn.Module):
        def __init__(self, nq, nl=3):
            super().__init__()
            self.th = nn.Parameter(torch.randn(nl,nq,3)*0.1)
        def forward(self, x):
            s = x
            for l in range(self.th.shape[0]):
                rx = torch.cos(self.th[l,:,0])*s + torch.sin(self.th[l,:,0])
                ry = torch.cos(self.th[l,:,1])*rx + torch.sin(self.th[l,:,1])
                rz = torch.cos(self.th[l,:,2])*ry + torch.sin(self.th[l,:,2])
                s  = torch.tanh(rz + torch.roll(rz,1,-1))
            return s

    class QModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(LOOKBACK*2,16), nn.Tanh(), nn.Linear(16,4), nn.Tanh())
            self.q   = QL(4,3)
            self.dec = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,1))
        def forward(self, x):
            return self.dec(self.q(self.enc(x.reshape(x.shape[0],-1)))).squeeze(-1)

    Xt = torch.FloatTensor(Xtr); yt_ = torch.FloatTensor(ytr)
    Xe = torch.FloatTensor(Xte)
    ds = TensorDataset(Xt, yt_); dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model = QModel(); opt = torch.optim.Adam(model.parameters(), lr=0.001); crit = nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        yp = model(Xe).numpy()
    yp = sy.inverse_transform(yp.reshape(-1,1)).ravel()
    yt = sy.inverse_transform(yte.reshape(-1,1)).ravel()
    return yt, yp, "Quantum-Inspired"

# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    Xtr, Xte, ytr, yte, sy = prepare(df)

    results = []
    for fn in [run_lstm, run_pinn, run_quantum]:
        name = fn.__name__.replace("run_","").upper()
        print(f"\nTraining {name}...")
        yt, yp, label = fn(Xtr, Xte, ytr, yte, sy)
        acc = accuracy(yt, yp)
        r2  = r2_score(yt, yp)
        print(f"  Accuracy (±1 Kp): {acc:.2f}%   R²: {r2:.4f}")
        results.append({"model": label, "accuracy_within_1": round(acc,2), "r2": round(r2,4)})

    print("\n" + "="*50)
    print(f"  {'Model':<22} {'Acc ±1 Kp':>12} {'R²':>8}")
    print("-"*44)
    for r in results:
        print(f"  {r['model']:<22} {r['accuracy_within_1']:>11.2f}% {r['r2']:>8.4f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output", "accuracy_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
