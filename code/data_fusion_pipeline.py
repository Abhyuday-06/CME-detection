"""
Data Fusion Pipeline — Multi-Source Space Weather Data Integration
==================================================================
Fuses three heterogeneous data sources into a unified ML-ready dataset:
  1. Aditya-L1 ASPEX (SWIS) — Indian L1 solar wind plasma
  2. NOAA DSCOVR / NASA OMNI2 — L1 plasma from US missions
  3. NOAA Planetary Kp Index — Geomagnetic activity target

Pipeline Steps:
  1. Load & clean raw data from PostgreSQL
  2. Resample to common 1-hour time grid (timestamp mapping)
  3. Outer-join fusion across all three sources
  4. Multi-strategy missing data imputation
  5. Feature engineering (physics-based hidden patterns)
  6. Generate formal fusion & data quality report
"""

import os
import json
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_db_connection():
    return psycopg2.connect(DB_URI)


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

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
    
    # Clean CDF fill values (~-1e31)
    for col in ['l1_density', 'l1_thermal_speed']:
        if col in df_aditya.columns:
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


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: TIMESTAMP MAPPING & FUSION (Outer Join)
# ═══════════════════════════════════════════════════════════════════════════

def map_and_fuse_data(df_aditya, df_dscovr, df_kp, freq='1h'):
    """
    Resamples all sources to a common time grid (freq) and outer-joins them.
    This ensures every hour has a potential data point from all sources.
    """
    df_aditya_res = df_aditya.resample(freq).mean()
    df_dscovr_res = df_dscovr.resample(freq).mean()
    
    # Kp Index is 3-hourly, forward-fill to hourly but limit to 3h
    df_kp_res = df_kp.resample(freq).mean().ffill(limit=3)

    # FUSION — Outer Join
    df_fused = df_dscovr_res.join(df_aditya_res, how='outer')
    df_fused = df_fused.join(df_kp_res, how='outer')

    return df_fused


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: MULTI-STRATEGY MISSING DATA IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def handle_missing_data(df, verbose=True):
    """
    Multi-strategy imputation:
      Strategy 1: Short gaps (≤4h) — Linear interpolation
      Strategy 2: Medium gaps (≤12h) — Forward fill (steady-state assumption)
      Strategy 3: Remaining — Column median fill
    
    Returns imputed dataframe and a report dict with before/after stats.
    """
    # Record BEFORE stats
    before_stats = {}
    for col in df.columns:
        total = len(df)
        missing = df[col].isna().sum()
        before_stats[col] = {
            'total_rows': int(total),
            'missing_before': int(missing),
            'pct_missing_before': round(missing / total * 100, 2) if total > 0 else 0
        }

    # Strategy 1: Short gaps (up to 4 hours) — Linear Interpolation
    df = df.interpolate(method='linear', limit=4)
    
    # Strategy 2: Medium gaps — Forward fill up to 12 hours
    df = df.ffill(limit=12)

    # Strategy 3: Remaining gaps — Column median
    df = df.fillna(df.median())

    # Record AFTER stats
    for col in df.columns:
        missing_after = df[col].isna().sum()
        before_stats[col]['missing_after'] = int(missing_after)
        before_stats[col]['pct_missing_after'] = round(
            missing_after / before_stats[col]['total_rows'] * 100, 2
        ) if before_stats[col]['total_rows'] > 0 else 0
        before_stats[col]['filled'] = before_stats[col]['missing_before'] - int(missing_after)
    
    if verbose:
        print("\n  ┌─ MISSING DATA IMPUTATION REPORT ─────────────────────────┐")
        print(f"  │ {'Column':<25} {'Before':>8} {'After':>8} {'Filled':>8} │")
        print(f"  ├{'─'*55}┤")
        for col, stats in before_stats.items():
            pct = stats['pct_missing_before']
            print(f"  │ {col:<25} {stats['missing_before']:>7}  {stats['missing_after']:>7}  {stats['filled']:>7}  │")
        print(f"  └{'─'*55}┘")

    return df, before_stats


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: FEATURE ENGINEERING (Hidden Pattern Discovery)
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """
    Creates physics-informed features that reveal hidden patterns in solar wind data.
    
    Feature Categories:
      A) Cross-source consensus (data fusion features)
      B) Physics-derived quantities (dynamic pressure, coupling)
      C) Temporal derivatives (rate of change)
      D) Rolling statistics (turbulence & variability)
      E) Lag features (autoregressive Kp memory)
      F) Interaction features (non-linear combinations)
      G) Source availability indicators (fusion reliability)
    """
    # ─── A) Cross-Source Consensus Features ─────────────────────────────
    # Prefer DSCOVR data, fallback to Aditya-L1 (multi-mission fusion)
    df['speed'] = df['dscovr_speed'].fillna(df['l1_speed'])
    df['density'] = df['dscovr_density'].fillna(df['l1_density'])
    
    speed = df['speed']
    density = df['density']
    
    # ─── B) Physics-Derived Features ────────────────────────────────────
    
    # Dynamic Pressure: Pdyn = m_p * n * v^2 (nPa proxy)
    # Key CME indicator — pressure spikes drive geomagnetic storms
    df['dynamic_pressure'] = density * (speed ** 2) * 1.6726e-6
    
    # Kinetic Energy Density proxy: ½ * n * v^2
    df['kinetic_energy'] = 0.5 * density * speed ** 2

    # Momentum flux: n * v — measures total particle momentum hitting Earth
    df['momentum_flux'] = density * speed

    # ─── C) Temporal Derivatives (Rate of Change) ───────────────────────
    # Sudden changes in solar wind parameters are CME signatures
    
    df['speed_gradient_1h'] = speed.diff(1)       # 1-hour rate of change
    df['speed_gradient_3h'] = speed.diff(3)       # 3-hour rate of change
    df['density_gradient_1h'] = density.diff(1)
    df['density_gradient_3h'] = density.diff(3)
    df['pressure_gradient'] = df['dynamic_pressure'].diff(1)

    # ─── D) Rolling Statistics (Turbulence & Variability) ───────────────
    # Elevated variance indicates disturbed solar wind conditions
    
    df['speed_mean_6h'] = speed.rolling(window=6, min_periods=1).mean()
    df['speed_std_6h'] = speed.rolling(window=6, min_periods=1).std()
    df['speed_variance_6h'] = speed.rolling(window=6, min_periods=1).var()
    df['density_variance_6h'] = density.rolling(window=6, min_periods=1).var()
    df['pressure_mean_6h'] = df['dynamic_pressure'].rolling(window=6, min_periods=1).mean()
    
    # Exponential Moving Average — gives more weight to recent values
    df['speed_ema_6h'] = speed.ewm(span=6, min_periods=1).mean()
    df['density_ema_6h'] = density.ewm(span=6, min_periods=1).mean()

    # ─── E) Lag Features (Kp Autoregressive Memory) ─────────────────────
    # Kp index has strong autocorrelation — past Kp predicts future Kp
    if 'kp_index' in df.columns:
        df['kp_lag_1h'] = df['kp_index'].shift(1)
        df['kp_lag_3h'] = df['kp_index'].shift(3)
        df['kp_lag_6h'] = df['kp_index'].shift(6)
        df['kp_rolling_mean_6h'] = df['kp_index'].rolling(window=6, min_periods=1).mean()

    # ─── F) Interaction / Non-Linear Features ───────────────────────────
    # Non-linear combinations capture complex physics relationships
    
    df['speed_x_density'] = speed * density          # Mass flux proxy
    df['speed_squared'] = speed ** 2                  # Energy proxy
    df['log_density'] = np.log1p(density.clip(lower=0))  # Log-transform skewed density

    # ─── G) Source Availability Indicators ──────────────────────────────
    # Tells the model which sources contributed to each row
    df['is_aditya_available'] = df['l1_speed'].notna().astype(int)
    df['is_dscovr_available'] = df['dscovr_speed'].notna().astype(int)
    df['source_count'] = df['is_aditya_available'] + df['is_dscovr_available']

    # Fill NaN caused by diff/rolling/lag operations
    df = df.bfill()
    df = df.fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5: FUSION REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_fusion_report(df_aditya, df_dscovr, df_kp, df_fused, missing_stats):
    """
    Generates a comprehensive fusion & data quality report.
    Returns a report dict and prints a formatted summary.
    """
    # Compute overlap statistics
    aditya_hours = set(df_aditya.index.floor('h'))
    dscovr_hours = set(df_dscovr.index.floor('h'))
    kp_hours = set(df_kp.index.floor('h'))
    
    overlap_ad = len(aditya_hours & dscovr_hours)
    overlap_ak = len(aditya_hours & kp_hours)
    overlap_dk = len(dscovr_hours & kp_hours)
    overlap_all = len(aditya_hours & dscovr_hours & kp_hours)

    report = {
        'source_records': {
            'aditya_l1': len(df_aditya),
            'dscovr_omni': len(df_dscovr),
            'kp_index': len(df_kp)
        },
        'source_date_ranges': {
            'aditya_l1': [str(df_aditya.index.min()), str(df_aditya.index.max())] if len(df_aditya) > 0 else ['N/A', 'N/A'],
            'dscovr_omni': [str(df_dscovr.index.min()), str(df_dscovr.index.max())] if len(df_dscovr) > 0 else ['N/A', 'N/A'],
            'kp_index': [str(df_kp.index.min()), str(df_kp.index.max())] if len(df_kp) > 0 else ['N/A', 'N/A']
        },
        'unique_hours': {
            'aditya_l1': len(aditya_hours),
            'dscovr_omni': len(dscovr_hours),
            'kp_index': len(kp_hours)
        },
        'overlaps': {
            'aditya_dscovr': overlap_ad,
            'aditya_kp': overlap_ak,
            'dscovr_kp': overlap_dk,
            'all_three': overlap_all
        },
        'fused_dataset': {
            'total_rows': len(df_fused),
            'total_features': len(df_fused.columns),
            'features': df_fused.columns.tolist()
        },
        'missing_data_report': missing_stats
    }

    print("\n" + "=" * 60)
    print("  DATA FUSION REPORT")
    print("=" * 60)
    
    print(f"\n  Source Records:")
    print(f"    Aditya-L1:       {len(df_aditya):>8,} records")
    print(f"    DSCOVR/OMNI:     {len(df_dscovr):>8,} records")
    print(f"    Kp Index:        {len(df_kp):>8,} records")
    
    print(f"\n  Timestamp Overlap (hourly):")
    print(f"    Aditya ∩ DSCOVR: {overlap_ad:>8,} hours")
    print(f"    Aditya ∩ Kp:     {overlap_ak:>8,} hours")
    print(f"    DSCOVR ∩ Kp:     {overlap_dk:>8,} hours")
    print(f"    All Three:       {overlap_all:>8,} hours")
    
    print(f"\n  Fused Dataset:")
    print(f"    Total rows:      {len(df_fused):>8,}")
    print(f"    Total features:  {len(df_fused.columns):>8}")
    
    # Fusion success rate
    total = len(df_fused)
    if total > 0:
        aditya_coverage = (df_fused['is_aditya_available'].sum() / total * 100) if 'is_aditya_available' in df_fused.columns else 0
        dscovr_coverage = (df_fused['is_dscovr_available'].sum() / total * 100) if 'is_dscovr_available' in df_fused.columns else 0
        print(f"\n  Fusion Coverage (post-imputation):")
        print(f"    Aditya-L1:       {aditya_coverage:>7.1f}%")
        print(f"    DSCOVR/OMNI:     {dscovr_coverage:>7.1f}%")
    
    print("=" * 60)
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "fusion_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to: {report_path}")
    
    return report


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def create_advanced_dataset(source_filter=None, verbose=True):
    """
    Executes the full pipeline:
      1. Load raw data
      2. Resample & fuse timestamps
      3. Impute missing data
      4. Engineer features
      5. Generate report
    
    Returns the final fused, engineered dataframe.
    """
    if verbose:
        print("\n[PIPELINE] Loading raw data...")
    df_aditya, df_dscovr, df_kp = load_data(source_filter=source_filter)
    
    if verbose:
        print(f"  Aditya-L1: {len(df_aditya)} records")
        print(f"  DSCOVR/OMNI: {len(df_dscovr)} records")
        print(f"  Kp Index: {len(df_kp)} records")
    
    if verbose:
        print("\n[PIPELINE] Mapping timestamps & fusing data...")
    df_fused = map_and_fuse_data(df_aditya, df_dscovr, df_kp)
    
    if verbose:
        print(f"  Fused rows (before imputation): {len(df_fused)}")
    
    if verbose:
        print("\n[PIPELINE] Imputing missing data...")
    df_fused, missing_stats = handle_missing_data(df_fused, verbose=verbose)
    
    if verbose:
        print("\n[PIPELINE] Engineering features...")
    df_fused = engineer_features(df_fused)

    # Drop rows where target is still null
    df_fused.dropna(subset=['kp_index'], inplace=True)
    
    if verbose:
        print(f"  Final dataset: {len(df_fused)} rows x {len(df_fused.columns)} features")
        print(f"  Features: {df_fused.columns.tolist()}")
    
    if verbose:
        generate_fusion_report(df_aditya, df_dscovr, df_kp, df_fused, missing_stats)

    return df_fused


if __name__ == "__main__":
    df = create_advanced_dataset()
    print(f"\nAdvanced Fused Dataset generated!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample (first 3 rows):")
    print(df.head(3))
