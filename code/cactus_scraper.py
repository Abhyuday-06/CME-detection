import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

def save_to_db(df):
    if df is None or df.empty:
        print(" No data to save.")
        return

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()

        data_tuples = [tuple(x) for x in df.to_numpy()]
        
        sql = """
            INSERT INTO cme_events (event_id, start_time, velocity, angular_width, is_halo)
            VALUES %s
            ON CONFLICT (event_id) DO NOTHING;
        """
        
        execute_values(cur, sql, data_tuples)
        
        conn.commit()
        print(f" Successfully saved {len(df)} events to Database.")
        
    except Exception as e:
        print(f" Database Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur: cur.close()
        if conn: conn.close()

def scrape_cactus(year, month):
    url = f"https://www.sidc.be/cactus/catalog/LASCO/2_5_0/qkl/{year}/{month:02d}/cmecat.txt"
    
    print(f" Fetching data from: {url}")
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f" Failed to fetch. Status: {response.status_code}")
            return None
    except Exception as e:
        print(f" Network Error: {e}")
        return None

    lines = response.text.split('\n')
    data_lines = [line for line in lines if line and not line.startswith('#')]
    
    cme_data = []
    for line in data_lines:
        parts = line.split('|')
        if len(parts) >= 10:
            # ORIGINAL ID from file (e.g., "001")
            raw_id = parts[0].strip()
            
            # Format: YYYYMM-ID (e.g., "202408-001")
            unique_id = f"{year}{month:02d}-{raw_id}"
            
            # Halo Level: 'II', 'III', 'IV'. 
            # We convert to Boolean for our SQL table (is_halo)
            halo_code = parts[9].strip()
            is_halo = True if halo_code in ['II', 'III', 'IV'] else False

            cme_data.append({
                'event_id': unique_id,         
                'start_time': parts[1].strip(),
                'angular_width': float(parts[4].strip()),
                'velocity': int(parts[5].strip()),
                'is_halo': is_halo
            })
            
    return pd.DataFrame(cme_data)

if __name__ == "__main__":
    for year in [2024, 2025, 2026]:
        for month in range(1, 13):
            print(f"\n--- Scraping CME Data for {year}-{month:02d} ---")
            df_cme = scrape_cactus(year, month)
            
            if df_cme is not None and not df_cme.empty:
                print(df_cme.head())
                save_to_db(df_cme)
            else:
                print(f" No CME data found for this {year} - {month:02d}.")
    print("\nScraping completed.")