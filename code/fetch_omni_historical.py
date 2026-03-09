"""
Fetch historical hourly Kp Index + Solar Wind data from NASA OMNI2
covering the Aditya-L1 observation period (May 2024 - Jan 2026).
"""

import requests
import psycopg2
from psycopg2.extras import execute_values
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')

OMNI_CGI = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"


def get_db_connection():
    return psycopg2.connect(DB_URI)


def fetch_omni_data(start="20240501", end="20260106"):
    """Fetch hourly solar wind speed, density, Kp from NASA OMNI2."""
    print(f"Fetching OMNI2 hourly data from {start} to {end}...")

    # OMNI2 variable codes (from OMNIWeb CGI):
    #  23 = SW Proton Density (N/cm^3)
    #  24 = SW Plasma Speed (km/s)
    #  38 = Kp*10
    data = (
        f"activity=retrieve&"
        f"res=hour&"
        f"spacecraft=omni2&"
        f"start_date={start}&"
        f"end_date={end}&"
        f"vars=23&vars=24&vars=38"
    )

    r = requests.post(
        OMNI_CGI, data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=120
    )
    r.raise_for_status()

    # Parse HTML response — data is in <pre> block, each line: YEAR DOY HR val1 val2 val3
    lines = r.text.strip().split("\n")
    records = []

    for line in lines:
        stripped = line.strip()
        parts = stripped.split()
        if len(parts) < 6:
            continue
        try:
            year = int(parts[0])
            doy = int(parts[1])
            hour = int(parts[2])
        except ValueError:
            continue

        if year < 2024:
            continue

        try:
            dt = datetime(year, 1, 1, hour, 0, 0, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        except ValueError:
            continue

        # Response columns: density(var23), speed(var24), Kp*10(var38)
        density_str = parts[3]
        speed_str = parts[4]
        kp_str = parts[5]

        # OMNI fill values: 999.9 for density, 9999. for speed, 99 for Kp*10
        density = float(density_str) if float(density_str) < 900 else None
        speed = float(speed_str) if float(speed_str) < 9000 else None
        kp_raw = int(kp_str) if int(kp_str) < 90 else None
        kp = kp_raw / 10.0 if kp_raw is not None else None

        records.append((dt, speed, density, kp))

    print(f"Parsed {len(records)} hourly records.")
    return records


def store_omni_data(records):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Store Kp data
    kp_batch = [(r[0], r[3], "NASA_OMNI2") for r in records if r[3] is not None]
    if kp_batch:
        # Delete existing OMNI Kp to avoid duplicates, then bulk insert
        cursor.execute("DELETE FROM geomagnetic_indices WHERE source = 'NASA_OMNI2'")
        execute_values(cursor, """
            INSERT INTO geomagnetic_indices (timestamp, kp_index, source)
            VALUES %s
        """, kp_batch, page_size=1000)
        conn.commit()
        print(f"Inserted {len(kp_batch)} Kp records (source=NASA_OMNI2)")

    # Store DSCOVR-equivalent plasma in dscovr_plasma (OMNI2 uses ACE/DSCOVR/Wind merged data)
    plasma_batch = [(r[0], r[1], r[2], None, "NASA_OMNI2") for r in records
                    if r[1] is not None or r[2] is not None]
    if plasma_batch:
        cursor.execute("DELETE FROM dscovr_plasma WHERE source = 'NASA_OMNI2'")
        execute_values(cursor, """
            INSERT INTO dscovr_plasma (observation_time, proton_speed, proton_density, proton_temperature, source)
            VALUES %s
            ON CONFLICT (observation_time) DO NOTHING
        """, plasma_batch, page_size=1000)
        conn.commit()
        print(f"Inserted {len(plasma_batch)} OMNI2 plasma records into dscovr_plasma")

    # Summary
    cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM geomagnetic_indices")
    r = cursor.fetchone()
    print(f"\nTotal Kp records now: {r[0]}  ({r[1]} → {r[2]})")

    cursor.execute("SELECT COUNT(*), MIN(observation_time), MAX(observation_time) FROM dscovr_plasma")
    r = cursor.fetchone()
    print(f"Total DSCOVR/OMNI plasma: {r[0]}  ({r[1]} → {r[2]})")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    records = fetch_omni_data()
    store_omni_data(records)
