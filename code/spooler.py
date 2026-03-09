import os
import time
import cdflib
import psycopg2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')


def get_db_connection():
    return psycopg2.connect(DB_URI)


def spool_cdf_file(filepath, delay=0.0):
    """
    Reads a CDF file and inserts rows one-by-one into swis_moments,
    simulating a real-time data feed from Aditya-L1.
    Set delay=0 for bulk historical backfill, or delay>0 for real-time simulation.
    """
    print(f"Spooling: {filepath}")
    cdf = cdflib.CDF(filepath)

    epochs = cdf.varget("epoch_for_cdf_mod")
    p_density = cdf.varget("proton_density")
    p_speed = cdf.varget("proton_bulk_speed")
    p_temp = cdf.varget("proton_thermal")
    a_density = cdf.varget("alpha_density")
    a_speed = cdf.varget("alpha_bulk_speed")
    a_temp = cdf.varget("alpha_thermal")
    sc_x = cdf.varget("spacecraft_xpos")
    sc_y = cdf.varget("spacecraft_ypos")
    sc_z = cdf.varget("spacecraft_zpos")

    timestamps = cdflib.cdfepoch.to_datetime(epochs)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT instrument_id FROM instruments WHERE name = 'ASPEX'")
    res = cursor.fetchone()
    instrument_id = res[0] if res else None

    inserted = 0
    skipped = 0

    for i in range(len(timestamps)):
        if p_speed[i] < 0 or p_speed[i] > 2000:
            skipped += 1
            continue

        ts = timestamps[i]
        if hasattr(ts, 'item'):
            ts = ts.item()
        if isinstance(ts, int):
            ts = datetime.utcfromtimestamp(ts / 1e9)

        cursor.execute("""
            INSERT INTO swis_moments
            (observation_time, proton_density, proton_speed, proton_thermal_speed,
             alpha_density, alpha_speed, alpha_thermal_speed, sc_x, sc_y, sc_z, instrument_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (
            ts,
            float(p_density[i]), float(p_speed[i]), float(p_temp[i]),
            float(a_density[i]), float(a_speed[i]), float(a_temp[i]),
            float(sc_x[i]), float(sc_y[i]), float(sc_z[i]),
            instrument_id
        ))

        inserted += 1

        if delay > 0:
            conn.commit()
            time.sleep(delay)

        if inserted % 500 == 0:
            conn.commit()
            print(f"  {inserted} rows inserted...")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"  Done: {inserted} inserted, {skipped} skipped (bad data).")


def spool_directory(data_dir, delay=0.0):
    """Spool all CDF files in a directory."""
    cdf_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.cdf')
    ])
    print(f"Found {len(cdf_files)} CDF files in {data_dir}")

    for filepath in cdf_files:
        spool_cdf_file(filepath, delay=delay)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aditya-L1 Data Spooler")
    parser.add_argument("--dir", type=str, help="Directory with CDF files")
    parser.add_argument("--file", type=str, help="Single CDF file to spool")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between rows in seconds (0=bulk, >0=realtime sim)")
    args = parser.parse_args()

    if args.file:
        spool_cdf_file(args.file, delay=args.delay)
    elif args.dir:
        spool_directory(args.dir, delay=args.delay)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pos_dir = os.path.join(base, "data", "SWIS-ISSDC", "positive")
        neg_dir = os.path.join(base, "data", "SWIS-ISSDC", "negative")
        for d in [pos_dir, neg_dir]:
            if os.path.isdir(d):
                spool_directory(d, delay=args.delay)
