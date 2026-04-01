import os
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')

def get_db_connection():
    return psycopg2.connect(DB_URI)

def load_data(source_filter=None):
    """
    Loads data from Aditya-L1, DSCOVR/OMNI (filtered by source if needed), and Kp Index.
    source_filter can be 'NASA_OMNI2' (static/historical) or 'NOAA_DSCOVR' (live).
    Returns raw dataframes.
    """
    conn = get_db_connection()

    # 1. Aditya-L1 (SWIS)
    query_aditya = """
        SELECT observation_time, proton_speed AS l1_speed, proton_density AS l1_density, 
               proton_thermal_speed AS l1_thermal_speed
        FROM swis_moments
        WHERE proton_speed > 0
        ORDER BY observation_time ASC
    """
    df_aditya = pd.read_sql(query_aditya, conn, parse_dates=["observation_time"])
    
    # Clean CDF fill values
    for col in ['l1_density', 'l1_thermal_speed']:
        df_aditya.loc[df_aditya[col] < -1e10, col] = np.nan

    df_aditya.set_index('observation_time', inplace=True)

    # 2. DSCOVR / OMNI (L1 or L2)
    where_clause = f"WHERE source = '{source_filter}'" if source_filter else ""
    query_dscovr = f"""
        SELECT observation_time, proton_speed AS dscovr_speed, proton_density AS dscovr_density, 
               proton_temperature AS dscovr_temp
        FROM dscovr_plasma
        {where_clause}
        ORDER BY observation_time ASC
    """
    df_dscovr = pd.read_sql(query_dscovr, conn, parse_dates=["observation_time"])
    df_dscovr.set_index('observation_time', inplace=True)

    # 3. Kp Index (Target)
    query_kp = """
        SELECT timestamp AS observation_time, kp_index
        FROM geomagnetic_indices
        ORDER BY timestamp ASC
    """
    df_kp = pd.read_sql(query_kp, conn, parse_dates=["observation_time"])
    df_kp.set_index('observation_time', inplace=True)

    conn.close()
    
    return df_aditya, df_dscovr, df_kp

def map_and_fuse_data(df_aditya, df_dscovr, df_kp, freq='1h'):
    """
    Solves Point 2: Map Timestamps & Point 4: Data Fusion.
    Resamples everything to a common time grid (freq) and outer joins.
    """
    # Resample all to numeric mean mappings
    df_aditya_res = df_aditya.resample(freq).mean()
    df_dscovr_res = df_dscovr.resample(freq).mean()
    
    # For Kp Index, it's a 3-hourly step value, so we forward-fill to our new grid
    # but first take the mean in the rare case multiple fall into the same bin.
    df_kp_res = df_kp.resample(freq).mean().ffill(limit=3) # max 3 hours missing interpolation

    # FUSION - Outer Join them
    df_fused = df_dscovr_res.join(df_aditya_res, how='outer')
    df_fused = df_fused.join(df_kp_res, how='outer')

    return df_fused

def handle_missing_data(df):
    """
    Solves Point 1: Handling Missing Data.
    We'll use a combination of interpolation for short gaps, 
    and median/rolling imputation for longer gaps.
    """
    # 1. Short gaps (up to 4 hours): Linear Interpolation
    df = df.interpolate(method='linear', limit=4)
    
    # 2. Medium gaps: Forward fill (up to 12 hours) - assumes steady state wind
    df = df.ffill(limit=12)

    # 3. For any remaining gaps at the beginning/end, or very long gaps, 
    # fill with the overall column median to maintain the tensor shapes for training
    df = df.fillna(df.median())

    return df

def engineer_features(df):
    """
    Solves Point 3: Feature Engineering / Hidden Patterns.
    """
    # Ensure physics features exist by treating fusion columns (favoring DSCOVR first, then Aditya)
    speed = df['dscovr_speed'].fillna(df['l1_speed'])
    density = df['dscovr_density'].fillna(df['l1_density'])
    
    # Dynamic Pressure (Proportional to Density * Speed^2)
    # Mass of proton ~1.67e-27 kg. Using proxy scaling here:
    df['dynamic_pressure'] = density * (speed ** 2) * 1.6726e-6  # Proxy unit conversion

    # Create Rate of Change (Derivatives)
    df['speed_gradient'] = speed.diff()
    df['density_gradient'] = density.diff()

    # Rolling Statistics (Turbulence / Variance in past 6 hours)
    df['speed_variance_6h'] = speed.rolling(window=6).var()
    df['density_variance_6h'] = density.rolling(window=6).var()
    df['pressure_mean_6h'] = df['dynamic_pressure'].rolling(window=6).mean()

    # Create an explicit missingness indicator (helps the model learn fusion reliability)
    df['is_aditya_missing'] = df['l1_speed'].isna().astype(int)
    df['is_dscovr_missing'] = df['dscovr_speed'].isna().astype(int)

    # Fill NaN caused by diff/rolling with 0 or bfill
    df.bfill(inplace=True)
    df.fillna(0, inplace=True) # Final fallback

    return df

def create_advanced_dataset(source_filter=None):
    """
    Execute the entire pipeline
    """
    df_aditya, df_dscovr, df_kp = load_data(source_filter=source_filter)
    
    df_fused = map_and_fuse_data(df_aditya, df_dscovr, df_kp)
    df_fused = handle_missing_data(df_fused)
    df_fused = engineer_features(df_fused)

    # Drop target nulls purely at the very end
    df_fused.dropna(subset=['kp_index'], inplace=True)

    return df_fused

if __name__ == "__main__":
    df = create_advanced_dataset()
    print("Advanced Fused Dataset generated!")
    print("Columns:", df.columns.tolist())
    print("Total Rows:", len(df))
    print(df.head())
