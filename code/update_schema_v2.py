import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')


def apply_schema_updates():
    conn = psycopg2.connect(DB_URI)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connected to database.")

    # 1. Create DSCOVR plasma table
    print("\n[1/3] Creating 'dscovr_plasma' table...")
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dscovr_plasma (
                record_id BIGSERIAL PRIMARY KEY,
                observation_time TIMESTAMPTZ NOT NULL,
                proton_speed DOUBLE PRECISION,
                proton_density DOUBLE PRECISION,
                proton_temperature DOUBLE PRECISION,
                source VARCHAR(50) DEFAULT 'NOAA_DSCOVR',
                CONSTRAINT unique_dscovr_obs_time UNIQUE (observation_time)
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dscovr_obs_time
            ON dscovr_plasma (observation_time);
        """)
        print("  - dscovr_plasma table ready.")
    except Exception as e:
        print(f"  - Error: {e}")

    # 2. Add index on geomagnetic_indices.timestamp for fast joins
    print("\n[2/3] Adding index on geomagnetic_indices...")
    try:
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_kp_timestamp
            ON geomagnetic_indices (timestamp);
        """)
        print("  - Index on geomagnetic_indices.timestamp ready.")
    except Exception as e:
        print(f"  - Error: {e}")

    # 3. Add index on swis_moments.observation_time
    print("\n[3/3] Adding index on swis_moments...")
    try:
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_swis_obs_time
            ON swis_moments (observation_time);
        """)
        print("  - Index on swis_moments.observation_time ready.")
    except Exception as e:
        print(f"  - Error: {e}")

    cursor.close()
    conn.close()
    print("\nSchema updates complete.")


if __name__ == "__main__":
    apply_schema_updates()
