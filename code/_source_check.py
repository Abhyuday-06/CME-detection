"""Check DSCOVR source-level stats and Aditya coverage."""
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DB_URI'))
cur = conn.cursor()

# DSCOVR stats by source
for col in ['proton_speed', 'proton_density']:
    for src in ['NASA_OMNI2', 'NOAA_DSCOVR']:
        cur.execute(f"""SELECT COUNT(*), AVG({col}), MIN({col}), MAX({col}),
                       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col})
                    FROM dscovr_plasma WHERE source = %s AND {col} IS NOT NULL""", (src,))
        r = cur.fetchone()
        print(f'{src} {col}: n={r[0]}, avg={r[1]:.2f}, min={r[2]:.2f}, max={r[3]:.2f}, median={r[4]:.2f}')

# proton_density fill values in swis_moments
cur.execute('SELECT COUNT(*) FROM swis_moments WHERE proton_density < -1e10')
print(f'\nswis_moments proton_density fill values: {cur.fetchone()[0]}')

# Aditya actual day coverage
cur.execute("""SELECT COUNT(DISTINCT date_trunc('day', observation_time)) FROM swis_moments WHERE proton_speed > 0""")
print(f'Aditya total unique days: {cur.fetchone()[0]}')

cur.execute("""SELECT date_trunc('month', observation_time) as mo, COUNT(*), COUNT(DISTINCT date_trunc('day', observation_time)) as days
FROM swis_moments WHERE proton_speed > 0
GROUP BY mo ORDER BY mo""")
print('\nAditya monthly breakdown:')
for r in cur.fetchall():
    print(f'  {r[0].strftime("%Y-%m")}: {r[1]:>7} records, {r[2]:>3} days')

conn.close()
