import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv

load_dotenv()

# DATABASE CONNECTION
DB_URI = os.getenv('DB_URI')

def detect_cme_events(start_date, end_date):
    conn = psycopg2.connect(DB_URI)
    
    print(f" Analyzing data from {start_date} to {end_date}...")

    # 1. FETCH RAW DATA
    # We fetch Alphas and Protons to calculate the ratio
    query = f"""
        SELECT record_id, observation_time, 
               proton_speed, proton_density, 
               alpha_density
        FROM swis_moments 
        WHERE observation_time BETWEEN '{start_date}' AND '{end_date}'
        AND proton_speed > 0 AND proton_speed < 3000
        ORDER BY observation_time ASC;
    """
    df = pd.read_sql(query, conn)
    
    if df.empty:
        print(" No data found for analysis.")
        return

    # 2. FEATURE ENGINEERING (The Data Science Part)
    # Calculate Rolling Mean & Std Dev (6-hour window)
    # Assuming data is roughly hourly or minute-cadence, we use a time-based window
    df.set_index('observation_time', inplace=True)
    
    # Calculate Alpha/Proton Ratio (Handle division by zero)
    df['alpha_ratio'] = df['alpha_density'] / df['proton_density'].replace(0, np.nan)
    
    # Calculate Rolling Statistics for Speed
    window_size = '6h' 
    df['rolling_mean_v'] = df['proton_speed'].rolling(window=window_size).mean()
    df['rolling_std_v'] = df['proton_speed'].rolling(window=window_size).std()

    # 3. APPLY DETECTION LOGIC (EVENT GROUPING WITH COOLDOWN)
    alerts = []
    
    # State Variables
    in_event = False
    event_start_time = None
    last_danger_time = None
    max_speed_in_event = 0
    
    # Cooldown: How long to wait before declaring an event "over" (e.g., 4 hours)
    COOLDOWN_PERIOD = pd.Timedelta(hours=4)
    
    for time, row in df.iterrows():
        
        # --- CONDITIONS ---
        # 1. High Speed (Storm) - Lowered threshold to catch slower CMEs
        is_fast = row['proton_speed'] > 450
        
        # 2. Shock (Sudden Jump)
        threshold = row['rolling_mean_v'] + (3 * row['rolling_std_v'])
        is_shock = row['proton_speed'] > threshold if not pd.isna(threshold) else False
        
        # 3. Alpha Ratio (Driver Gas)
        is_enriched = row['alpha_ratio'] > 0.04
        
        # TRIGGER LOGIC: Is this a dangerous moment?
        # Relaxed trigger: Speed > 450 AND (Shock OR Enriched) OR Speed > 600 (Absolute Storm)
        is_danger = (is_fast and is_shock) or (is_fast and is_enriched) or (row['proton_speed'] > 600)

        # --- STATE MACHINE WITH COOLDOWN ---
        
        if is_danger:
            # Update the last time we saw danger
            last_danger_time = time
            
            if not in_event:
                # START of a new event
                in_event = True
                event_start_time = time
                max_speed_in_event = row['proton_speed']
            else:
                # CONTINUATION: Update max speed
                if row['proton_speed'] > max_speed_in_event:
                    max_speed_in_event = row['proton_speed']
        
        elif in_event:
            # Danger is False, but we are in an event.
            # Check if enough time has passed since the last danger signal to close it.
            time_since_last_danger = time - last_danger_time
            
            if time_since_last_danger > COOLDOWN_PERIOD:
                # END of the event (Cooldown expired)
                in_event = False
                
                # Calculate duration based on when the danger actually stopped
                duration = last_danger_time - event_start_time
                
                # Filter out tiny blips (e.g., must last at least 30 mins)
                if duration.total_seconds() > 1800: 
                    severity = 'HIGH' if max_speed_in_event > 800 else 'MEDIUM'
                    
                    alerts.append({
                        'generated_at': event_start_time,
                        'severity': severity,
                        'message': f"CME Event Detected. Duration: {duration}. Peak Speed: {int(max_speed_in_event)} km/s",
                        'swis_record_id': row['record_id'] # Link to the end record for reference
                    })

    # Handle case where event is still active at the end of the dataset
    if in_event:
        duration = last_danger_time - event_start_time
        if duration.total_seconds() > 1800:
            severity = 'HIGH' if max_speed_in_event > 800 else 'MEDIUM'
            alerts.append({
                'generated_at': event_start_time,
                'severity': severity,
                'message': f"CME Event Detected (Ongoing). Duration: {duration}. Peak Speed: {int(max_speed_in_event)} km/s",
                'swis_record_id': 0 # Placeholder
            })

    print(f"⚡ Analysis Complete. Condensed {len(df)} points into {len(alerts)} Discrete Events.")

    # 4. SAVE ALERTS TO DATABASE (The DBMS Part)
    if alerts:
        cursor = conn.cursor()
        
        # Prepare data for batch insert
        insert_query = """
            INSERT INTO alerts (generated_at, severity, message, swis_record_id)
            VALUES %s
        """
        # Convert numpy types to native Python types for psycopg2
        data_tuples = [
            (
                x['generated_at'].to_pydatetime() if hasattr(x['generated_at'], 'to_pydatetime') else x['generated_at'], 
                x['severity'], 
                x['message'], 
                int(x['swis_record_id'])
            ) 
            for x in alerts
        ]
        
        execute_values(cursor, insert_query, data_tuples)
        conn.commit()
        print(" Alerts saved to database.")
    else:
        print(" No threats detected (Quiet Sun).")

    conn.close()

# --- RUN ANALYSIS ---
def run_full_analysis():
    conn = psycopg2.connect(DB_URI)
    cursor = conn.cursor()
    
    # 1. Clear existing alerts to prevent duplicates
    print("Clearing old alerts...")
    cursor.execute("TRUNCATE TABLE alerts RESTART IDENTITY CASCADE;")
    conn.commit()
    
    # 2. Find the full data range
    print("Detecting data range...")
    cursor.execute("SELECT MIN(observation_time), MAX(observation_time) FROM swis_moments")
    min_date, max_date = cursor.fetchone()
    conn.close()
    
    if min_date and max_date:
        print(f"Running detection from {min_date} to {max_date}")
        detect_cme_events(str(min_date), str(max_date))
    else:
        print("No data in swis_moments to analyze.")

if __name__ == "__main__":
    run_full_analysis()