import psycopg2
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')

conn = psycopg2.connect(DB_URI)
query = """
    SELECT 
        date_trunc('hour', s.observation_time) as obs_hour,
        AVG(s.proton_speed) as speed, 
        AVG(s.proton_density) as density, 
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
df = df.interpolate(method='linear').ffill().bfill()
print("Data head:\n", df.head())

for i in range(1, 7):
    df[f'kp+{i}h'] = df['kp'].shift(-i)

df.dropna(inplace=True)

if len(df) == 0:
    print("No data available after dropna!!!")
    exit()

features = df[['speed', 'density', 'kp']].values
targets = df[[f'kp+{i}h' for i in range(1, 7)]].values

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
scaler_Y = MinMaxScaler()
y_scaled = scaler_Y.fit_transform(targets)

SEQ_LEN = 24
X, Y = [], []
for i in range(len(X_scaled) - SEQ_LEN):
    X.append(X_scaled[i:i+SEQ_LEN])
    Y.append(y_scaled[i+SEQ_LEN])

X = np.array(X)
Y = np.array(Y)

print("Training shape:", X.shape, Y.shape)

model = Sequential([
    LSTM(32, activation='relu', input_shape=(SEQ_LEN, 3)),
    Dropout(0.2),
    Dense(6)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, Y, epochs=5, batch_size=32, verbose=1)

base = os.path.dirname(os.path.abspath(__file__))
model.save(os.path.join(base, 'kp_model.keras'))
with open(os.path.join(base, 'kp_scaler_X.pkl'), 'wb') as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(base, 'kp_scaler_Y.pkl'), 'wb') as f:
    pickle.dump(scaler_Y, f)

print("Kp Model and Scalers saved successfully!")
