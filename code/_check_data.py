import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DB_URI'))
cur = conn.cursor()

cur.execute('SELECT COUNT(*) FROM swis_moments WHERE proton_thermal_speed < -1e10')
print('thermal_speed fill values:', cur.fetchone()[0])

cur.execute('SELECT COUNT(*) FROM swis_moments WHERE alpha_density < -1e10')
print('alpha_density fill values:', cur.fetchone()[0])

cur.execute("""SELECT COUNT(*), AVG(alpha_density), MIN(alpha_density), MAX(alpha_density)
FROM swis_moments WHERE alpha_density > -1e10 AND alpha_density IS NOT NULL""")
r = cur.fetchone()
print(f'alpha_density valid: count={r[0]}, avg={r[1]:.2f}, min={r[2]:.2f}, max={r[3]:.2f}')

cur.execute("""SELECT COUNT(*), AVG(proton_thermal_speed), MIN(proton_thermal_speed), MAX(proton_thermal_speed)
FROM swis_moments WHERE proton_thermal_speed > -1e10 AND proton_thermal_speed IS NOT NULL""")
r = cur.fetchone()
print(f'thermal_speed valid: count={r[0]}, avg={r[1]:.2f}, min={r[2]:.2f}, max={r[3]:.2f}')

cur.execute('SELECT COUNT(*), COUNT(proton_temperature) FROM dscovr_plasma')
r = cur.fetchone()
print(f'DSCOVR total: {r[0]}, non-null temperature: {r[1]}')

cur.execute('SELECT COUNT(*) FROM dscovr_plasma WHERE proton_temperature IS NULL OR proton_temperature = 0')
print(f'DSCOVR null/zero temperature: {cur.fetchone()[0]}')

cur.execute("""SELECT source, COUNT(*), AVG(proton_temperature), MIN(proton_temperature), MAX(proton_temperature)
FROM dscovr_plasma GROUP BY source""")
for r in cur.fetchall():
    print(f'Source {r[0]}: count={r[1]}, avg_temp={r[2]}, min={r[3]}, max={r[4]}')

# Check Kp bar plot data
cur.execute("""SELECT COUNT(*), MIN(kp_index), MAX(kp_index), AVG(kp_index)
FROM geomagnetic_indices""")
r = cur.fetchone()
print(f'Kp: count={r[0]}, min={r[1]:.2f}, max={r[2]:.2f}, avg={r[3]:.2f}')

conn.close()
