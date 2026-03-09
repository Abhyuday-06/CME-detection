import requests
import psycopg2
from psycopg2.extras import execute_values
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')

# GFZ Potsdam Kp archive (2024 onwards)
GFZ_KP_URL = "https://kp.gfz-potsdam.de/app/json/?start=2024-05-01&end=2026-03-06"

# NOAA 1-month Kp
NOAA_KP_1M_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"


def get_db_connection():
    return psycopg2.connect(DB_URI)


def fetch_gfz_kp():
    print(f"Fetching historical Kp from GFZ Potsdam...")
    response = requests.get(GFZ_KP_URL, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_noaa_1m_kp():
    print(f"Fetching 1-month Kp from NOAA...")
    response = requests.get(NOAA_KP_1M_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def store_gfz_kp():
    data = fetch_gfz_kp()
    print(f"Fetched {len(data)} records from GFZ.")

    conn = get_db_connection()
    cursor = conn.cursor()

    batch = []
    for entry in data:
        ts_str = entry.get("datetime")
        kp = entry.get("Kp")
        if ts_str is None or kp is None:
            continue

        obs_time = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        batch.append((obs_time, float(kp), "GFZ_Potsdam"))

    if batch:
        execute_values(cursor, """
            INSERT INTO geomagnetic_indices (timestamp, kp_index, source)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, batch, page_size=500)
        conn.commit()
        print(f"Inserted up to {len(batch)} GFZ Kp records.")
    else:
        print("No records parsed from GFZ.")

    cursor.close()
    conn.close()


def store_noaa_1m_kp():
    data = fetch_noaa_1m_kp()
    print(f"Fetched {len(data)} records from NOAA 1-month.")

    conn = get_db_connection()
    cursor = conn.cursor()

    batch = []
    for entry in data:
        ts_str = entry.get("time_tag")
        kp = entry.get("kp_index")
        if ts_str is None or kp is None:
            continue

        try:
            obs_time = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        batch.append((obs_time, float(kp), "NOAA_SWPC"))

    if batch:
        execute_values(cursor, """
            INSERT INTO geomagnetic_indices (timestamp, kp_index, source)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, batch, page_size=500)
        conn.commit()
        print(f"Inserted up to {len(batch)} NOAA 1-month Kp records.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    try:
        store_gfz_kp()
    except Exception as e:
        print(f"GFZ fetch failed: {e}")

    try:
        store_noaa_1m_kp()
    except Exception as e:
        print(f"NOAA 1-month fetch failed: {e}")
