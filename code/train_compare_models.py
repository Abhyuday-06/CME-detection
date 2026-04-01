import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import matplotlib.pyplot as plt

# Import the fusion pipeline we just built
from data_fusion_pipeline import create_advanced_dataset

LOOKBACK_HOURS = 12
FORECAST_HOURS = 1

def prepare_sequences(df, target_col='kp_index'):
    """
    Transforms tabular dataframe into sequence tensors for LSTM.
    """
    # Shift target column for forecasting
    df['target'] = df[target_col].shift(-FORECAST_HOURS)
    df.dropna(inplace=True)

    features = [c for c in df.columns if c not in [target_col, 'target']]
    
    # Scale Features
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df[features])
    
    # Scale Target
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(df[['target']])

    X, y = [], []
    for i in range(len(X_scaled) - LOOKBACK_HOURS):
        X.append(X_scaled[i : i + LOOKBACK_HOURS])
        y.append(y_scaled[i + LOOKBACK_HOURS])

    return np.array(X), np.array(y), scaler_y

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

def train_and_eval(df, name):
    print(f"\n{'='*50}\nTraining {name} Model\n{'='*50}")
    
    X, y, scaler_y = prepare_sequences(df)
    
    if len(X) < 100:
        print(f"Not enough data for {name} ({len(X)} sequences).")
        return None
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    
    # Evaluate
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{name} Results:")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    
    return {
        'model': model,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'history': history,
        'y_true': y_true,
        'y_pred': y_pred
    }

def main():
    # 1. Fetch Static/Historical Model Data ('NASA_OMNI2')
    print("Loading STATIC Historical Data (NASA OMNI2)...")
    df_static = create_advanced_dataset(source_filter='NASA_OMNI2')
    
    # 2. Fetch Live/Realtime Model Data ('NOAA_DSCOVR')
    print("Loading LIVE Operational Data (NOAA DSCOVR)...")
    df_live = create_advanced_dataset(source_filter='NOAA_DSCOVR')
    
    results = {}
    
    if not df_static.empty:
        results['STATIC'] = train_and_eval(df_static, "Static (OMNI2)")
    else:
        print("Static Data is empty.")
        
    if not df_live.empty:
        results['LIVE'] = train_and_eval(df_live, "Live (NOAA L1)")
    else:
        print("Live Data is empty.")
        
    # Plot Comparison
    if 'STATIC' in results and 'LIVE' in results:
        plt.figure(figsize=(10,5))
        
        plt.plot(results['STATIC']['y_true'][:100], label='Actual Target (Kp)', color='black', alpha=0.5, linestyle='--')
        plt.plot(results['STATIC']['y_pred'][:100], label='Static Model Prediction', color='blue')
        
        # Adjust indices if they differ in test set size, but this gives a qualitative look
        min_len = min(len(results['STATIC']['y_pred']), len(results['LIVE']['y_pred']), 100)
        plt.plot(results['LIVE']['y_pred'][:min_len], label='Live Model Prediction', color='red', alpha=0.7)
        
        plt.title("Static vs Live Data Models: Kp Forecast Accuracy")
        plt.legend()
        plt.grid()
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, "live_vs_static_plot.png")
        
        plt.savefig(img_path)
        print(f"\nSaved comparison plot to {img_path}")

if __name__ == "__main__":
    main()
