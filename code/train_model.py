import pandas as pd
import numpy as np
import psycopg2
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from dotenv import load_dotenv

load_dotenv()

# DATABASE CONFIGURATION
DB_URI = os.getenv('DB_URI')

# HYPERPARAMETERS
LOOKBACK_HOURS = 24  # Use past 24 hours to predict
FORECAST_HORIZON = 1 # Predict next 1 hour
EPOCHS = 20
BATCH_SIZE = 32

def get_db_connection():
    return psycopg2.connect(DB_URI)

def fetch_training_data():
    print("Connecting to Database...")
    conn = get_db_connection()
    
    query = """
        SELECT observation_time, proton_speed, proton_density, proton_thermal_speed, alpha_density
        FROM swis_moments
        ORDER BY observation_time ASC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} raw data points.")
    return df

def preprocess_data(df):
    print("Preprocessing Data...")
    
    df['observation_time'] = pd.to_datetime(df['observation_time'])
    df.set_index('observation_time', inplace=True)
    
    # Resample to Hourly Averages
    df_resampled = df.resample('h').mean()
    df_resampled = df_resampled.interpolate(method='linear')
    df_resampled.dropna(inplace=True)
    
    print(f"Resampled to {len(df_resampled)} hourly points.")
    return df_resampled

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback + horizon - 1, 0]) 
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    df = fetch_training_data()
    if df.empty:
        print("No data found. Aborting.")
        return

    df_processed = preprocess_data(df)
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_processed.values)
    
    # Determine absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    model_path = os.path.join(script_dir, 'cme_prediction_model.keras')

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    X, y = create_sequences(data_scaled, LOOKBACK_HOURS, FORECAST_HORIZON)
    print(f"Created {len(X)} training sequences.")
    
    if len(X) < 100:
        print("Warning: Very small dataset. Model will likely overfit.")

    # Split Train/Test (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Training LSTM Model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
