import pandas as pd
import numpy as np
import tensorflow as tf
import psycopg2
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

load_dotenv()

# SETTINGS
MODEL_PATH = 'cme_prediction_model.keras'
SCALER_PATH = 'scaler.pkl'
LOOKBACK_HOURS = 24

# DB CONFIG
DB_URI = os.getenv('DB_URI')

def get_data():
    conn = psycopg2.connect(DB_URI)
    query = """
        SELECT observation_time, proton_speed, proton_density, proton_thermal_speed, alpha_density
        FROM swis_moments
        ORDER BY observation_time ASC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def test_random_predictions(n=5):
    # 1. Load Resources
    print("Loading Model & Scaler...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Prepare Data
    print("Fetching historical data...")
    df = get_data()
    df['observation_time'] = pd.to_datetime(df['observation_time'])
    df.set_index('observation_time', inplace=True)
    
    # Resample to match training frequency (Hourly)
    df_resampled = df.resample('h').mean().interpolate(method='linear')
    data_values = df_resampled.values

    print(f"\n--- RUNNING {n} RANDOM BACKTESTS ---")
    print(f"{'Time (T)':<25} | {'Actual (T+1h)':<15} | {'Predicted (T+1h)':<15} | {'Error':<10}")
    print("-" * 75)

    # 3. Run Tests
    # We strictly test on the LAST 20% of data (The "Future" relative to training)
    # This ensures we are testing generalization, not memorization (overfitting).
    split_index = int(len(df_resampled) * 0.8)
    max_idx = len(df_resampled) - 2 

    print(f"Total Data Points: {len(df_resampled)}")
    print(f"Training Range: 0 to {split_index}")
    print(f"Testing Range (Unseen): {split_index} to {max_idx}")
    
    total_error = 0
    
    for _ in range(n):
        # Pick a random index ONLY from the test set
        idx = np.random.randint(split_index, max_idx)
        
        input_seq = data_values[idx-LOOKBACK_HOURS+1 : idx+1] 
        target_val = data_values[idx+1][0]
        target_time = df_resampled.index[idx+1]

        # Scale Input
        input_scaled = scaler.transform(input_seq)
        input_reshaped = input_scaled.reshape(1, LOOKBACK_HOURS, 4)
        
        # Predict
        prediction_scaled = model.predict(input_reshaped, verbose=0)
        
        # Inverse Scale (We only care about the first column: speed)
        dummy_row = np.zeros((1, 4))
        dummy_row[0, 0] = prediction_scaled[0][0]
        prediction_actual = scaler.inverse_transform(dummy_row)[0][0]
        
        # Calculate Error
        error_pct = abs(target_val - prediction_actual) / target_val * 100
        total_error += error_pct
        
        print(f"{str(target_time):<25} | {target_val:<15.2f} | {prediction_actual:<15.2f} | {error_pct:<5.2f}%")

    print("-" * 75)
    print(f"Average Error on UNSEEN Data: {total_error/n:.2f}%")

if __name__ == '__main__':
    test_random_predictions(n=10)