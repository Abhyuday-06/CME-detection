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
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Scrape CACTUS CME data.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM)", default=None)
    parser.add_argument("--end", type=str, help="End date (YYYY-MM)", default=None)
    
    args = parser.parse_args()

    # Default range if not provided: 2024-01 to Present + 1 Month (to be safe)
    start_year, start_month = 2024, 1
    now = datetime.now()
    end_year, end_month = now.year, now.month

    if args.start:
        try:
            d_start = datetime.strptime(args.start, "%Y-%m")
            start_year, start_month = d_start.year, d_start.month
        except ValueError:
            print("Invalid start date format. Use YYYY-MM")
            exit(1)

    if args.end:
        try:
            d_end = datetime.strptime(args.end, "%Y-%m")
            end_year, end_month = d_end.year, d_end.month
        except ValueError:
            print("Invalid end date format. Use YYYY-MM")
            exit(1)

    print(f"Scraping range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")

    # Iterate through months
    curr_y, curr_m = start_year, start_month
    while (curr_y < end_year) or (curr_y == end_year and curr_m <= end_month):
        print(f"\n--- Scraping CME Data for {curr_y}-{curr_m:02d} ---")
        df_cme = scrape_cactus(curr_y, curr_m)
        
        if df_cme is not None and not df_cme.empty:
            print(df_cme.head())
            save_to_db(df_cme)
        else:
            print(f" No CME data found for {curr_y}-{curr_m:02d}.")

        # Increment month
        curr_m += 1
        if curr_m > 12:
            curr_m = 1
            curr_y += 1
            
    print("\nScraping completed.")