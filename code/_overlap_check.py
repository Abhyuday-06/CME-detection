"""Quick check: hourly overlap across all 3 data sources."""
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DB_URI'))
cur = conn.cursor()

cur.execute("""
WITH
  aditya_hours AS (SELECT DISTINCT date_trunc('hour', observation_time) AS hr FROM swis_moments WHERE proton_speed > 0),
  dscovr_hours AS (SELECT DISTINCT date_trunc('hour', observation_time) AS hr FROM dscovr_plasma WHERE proton_speed IS NOT NULL),
  kp_hours AS (SELECT DISTINCT date_trunc('hour', timestamp) AS hr FROM geomagnetic_indices)
SELECT
  (SELECT COUNT(*) FROM aditya_hours),
  (SELECT COUNT(*) FROM dscovr_hours),
  (SELECT COUNT(*) FROM kp_hours),
  (SELECT COUNT(*) FROM aditya_hours a JOIN dscovr_hours d ON a.hr = d.hr),
  (SELECT COUNT(*) FROM aditya_hours a JOIN kp_hours k ON a.hr = k.hr),
  (SELECT COUNT(*) FROM dscovr_hours d JOIN kp_hours k ON d.hr = k.hr),
  (SELECT COUNT(*) FROM aditya_hours a JOIN dscovr_hours d ON a.hr = d.hr JOIN kp_hours k ON a.hr = k.hr)
""")
r = cur.fetchone()
print(f"Aditya unique hours:   {r[0]}")
print(f"DSCOVR unique hours:   {r[1]}")
print(f"Kp unique hours:       {r[2]}")
print(f"Aditya-DSCOVR overlap: {r[3]}")
print(f"Aditya-Kp overlap:     {r[4]}")
print(f"DSCOVR-Kp overlap:     {r[5]}")
print(f"ALL THREE overlap:     {r[6]}")

# Also check NaN/NULL counts for each table
print("\n=== NULL/NaN ANALYSIS ===")
for table, cols in [
    ("swis_moments", ["proton_speed", "proton_density", "proton_thermal_speed", "alpha_density", "alpha_speed"]),
    ("dscovr_plasma", ["proton_speed", "proton_density", "proton_temperature"]),
    ("geomagnetic_indices", ["kp_index"]),
]:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    total = cur.fetchone()[0]
    print(f"\n{table} (total: {total})")
    for col in cols:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL")
        nulls = cur.fetchone()[0]
        # Check CDF fill values for swis_moments
        fill_count = 0
        if table == "swis_moments" and col in ("proton_thermal_speed", "alpha_density", "alpha_speed"):
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} < -1e10")
            fill_count = cur.fetchone()[0]
        cur.execute(f"""SELECT MIN({col}), MAX({col}), AVG({col}),
                       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}),
                       PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}),
                       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}),
                       STDDEV({col})
                    FROM {table} WHERE {col} IS NOT NULL""")
        s = cur.fetchone()
        fill_str = f", fill_values={fill_count}" if fill_count > 0 else ""
        effective_null = nulls + fill_count
        print(f"  {col}: null={nulls}{fill_str} (effective_missing={effective_null}/{total} = {effective_null/total*100:.1f}%)")
        print(f"    min={s[0]:.4f}, Q1={s[4]:.4f}, median={s[3]:.4f}, Q3={s[5]:.4f}, max={s[1]:.4f}")
        print(f"    mean={s[2]:.4f}, std={s[6]:.4f}")
        # IQR outlier count
        iqr = float(s[5]) - float(s[4])
        lo = float(s[4]) - 1.5 * iqr
        hi = float(s[5]) + 1.5 * iqr
        if table == "swis_moments" and col in ("proton_thermal_speed", "alpha_density", "alpha_speed"):
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NOT NULL AND {col} > -1e10 AND ({col} < {lo} OR {col} > {hi})")
        else:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NOT NULL AND ({col} < {lo} OR {col} > {hi})")
        outliers = cur.fetchone()[0]
        print(f"    IQR outliers (< {lo:.2f} or > {hi:.2f}): {outliers} ({outliers/total*100:.1f}%)")

conn.close()
