import requests
import psycopg2
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')

# NOAA DSCOVR real-time plasma endpoints
PLASMA_7DAY_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
PLASMA_2HR_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"


def get_db_connection():
    return psycopg2.connect(DB_URI)


def fetch_dscovr_plasma(url=PLASMA_7DAY_URL):
    print(f"Fetching DSCOVR plasma from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    # First row is header: ["time_tag","density","speed","temperature"]
    return data[1:]


def store_dscovr_plasma(use_7day=True):
    from psycopg2.extras import execute_values

    url = PLASMA_7DAY_URL if use_7day else PLASMA_2HR_URL
    rows = fetch_dscovr_plasma(url)
    print(f"Fetched {len(rows)} records.")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(observation_time) FROM dscovr_plasma")
    last_time = cursor.fetchone()[0]
    if last_time:
        print(f"Latest DB record: {last_time}")

    batch = []
    for row in rows:
        time_str, density, speed, temperature = row[0], row[1], row[2], row[3]

        if density is None and speed is None:
            continue

        try:
            obs_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            obs_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

        if last_time and obs_time <= last_time:
            continue

        batch.append((
            obs_time,
            float(speed) if speed else None,
            float(density) if density else None,
            float(temperature) if temperature else None,
        ))

    if batch:
        execute_values(cursor, """
            INSERT INTO dscovr_plasma (observation_time, proton_speed, proton_density, proton_temperature)
            VALUES %s
            ON CONFLICT (observation_time) DO NOTHING
        """, batch, page_size=500)
        conn.commit()
        print(f"Inserted {len(batch)} new DSCOVR plasma records.")
    else:
        print("No new records to insert.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    store_dscovr_plasma(use_7day=True)
