import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DB_URI'))
cur = conn.cursor()

print("=== DATA RANGES ===")
cur.execute("SELECT COUNT(*), MIN(observation_time), MAX(observation_time) FROM swis_moments")
r = cur.fetchone()
print(f"Aditya-L1:  {r[0]:>8} rows  |  {r[1]}  ->  {r[2]}")

cur.execute("SELECT COUNT(*), MIN(observation_time), MAX(observation_time) FROM dscovr_plasma")
r = cur.fetchone()
print(f"DSCOVR:     {r[0]:>8} rows  |  {r[1]}  ->  {r[2]}")

cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM geomagnetic_indices")
r = cur.fetchone()
print(f"Kp Index:   {r[0]:>8} rows  |  {r[1]}  ->  {r[2]}")

print("\n=== OVERLAP ANALYSIS ===")
# Aditya-L1 vs Kp
cur.execute("""
    SELECT COUNT(*) FROM swis_moments s
    WHERE EXISTS (
        SELECT 1 FROM geomagnetic_indices g
        WHERE DATE_TRUNC('hour', g.timestamp) = DATE_TRUNC('hour', s.observation_time)
    )
""")
print(f"Aditya-L1 rows with Kp match (hourly): {cur.fetchone()[0]}")

# DSCOVR vs Kp
cur.execute("""
    SELECT COUNT(*) FROM dscovr_plasma d
    WHERE EXISTS (
        SELECT 1 FROM geomagnetic_indices g
        WHERE DATE_TRUNC('hour', g.timestamp) = DATE_TRUNC('hour', d.observation_time)
    )
""")
print(f"DSCOVR rows with Kp match (hourly): {cur.fetchone()[0]}")

# Aditya-L1 vs DSCOVR
cur.execute("""
    SELECT COUNT(*) FROM swis_moments s
    WHERE EXISTS (
        SELECT 1 FROM dscovr_plasma d
        WHERE DATE_TRUNC('hour', d.observation_time) = DATE_TRUNC('hour', s.observation_time)
    )
""")
print(f"Aditya-L1 rows with DSCOVR match (hourly): {cur.fetchone()[0]}")

conn.close()
