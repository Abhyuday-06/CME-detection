import requests
import psycopg2
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DB_URI = os.getenv('DB_URI')

# NOAA SWPC Real-time Kp Index Endpoint
# Detailed format: List of lists. First list is header.
# Headers: [time_tag, Kp, a_running, station_count]
NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

def get_db_connection():
    try:
        conn = psycopg2.connect(DB_URI)
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def fetch_noaa_data():
    print(f"Fetching data from {NOAA_KP_URL}...")
    try:
        response = requests.get(NOAA_KP_URL)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def store_kp_indices():
    raw_data = fetch_noaa_data()
    if not raw_data:
        return

    conn = get_db_connection()
    if not conn:
        return

    cursor = conn.cursor()

    # 1. Parse Data (Skip header row)
    header = raw_data[0]
    rows = raw_data[1:]
    
    print(f"Found {len(rows)} records from NOAA.")

    # 2. Get latest timestamp in DB to avoid duplicates
    # Only insert records newer than what we have
    cursor.execute("SELECT MAX(timestamp) FROM geomagnetic_indices WHERE source = 'NOAA_SWPC'")
    result = cursor.fetchone()
    last_db_time = result[0]
    
    if last_db_time:
        print(f"Latest data in DB: {last_db_time}")
    else:
        print("No existing NOAA data in DB. Importing all...")

    new_records_count = 0
    
    for row in rows:
        # Format: ["2024-10-10 00:00:00", "3.33", "12", "4"]
        time_str = row[0]
        kp_val = row[1]
        
        try:
            # Parse timestamp (Handle milliseconds if present)
            # Example: "2026-02-05 00:00:00.000"
            try:
                obs_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            except ValueError:
                obs_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            
            # Skip if older than what we have (simple deduplication)
            if last_db_time and obs_time <= last_db_time:
                continue

            # Insert
            cursor.execute("""
                INSERT INTO geomagnetic_indices (timestamp, kp_index, source)
                VALUES (%s, %s, 'NOAA_SWPC')
            """, (obs_time, float(kp_val)))
            
            new_records_count += 1
            
        except ValueError as e:
            print(f"Skipping row {row}: {e}")
            continue

    conn.commit()
    print(f"Successfully inserted {new_records_count} new Kp index records.")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    store_kp_indices()
