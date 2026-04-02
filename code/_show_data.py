"""Print columns and sample rows from all 3 data sources."""
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DB_URI'))
cur = conn.cursor()

print('=' * 90)
print('  DATA SOURCE 1: Aditya-L1 ASPEX (swis_moments)')
print('=' * 90)
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='swis_moments' ORDER BY ordinal_position")
print('Columns:', [f'{r[0]} ({r[1]})' for r in cur.fetchall()])
cur.execute("SELECT observation_time, proton_speed, proton_density, proton_thermal_speed, alpha_density, alpha_speed FROM swis_moments WHERE proton_speed > 0 ORDER BY observation_time LIMIT 5")
rows = cur.fetchall()
header = f"{'observation_time':>32} {'proton_speed':>14} {'proton_density':>16} {'thermal_speed':>15} {'alpha_density':>15} {'alpha_speed':>13}"
print(f"\n{header}")
print('-' * len(header))
for r in rows:
    t = str(r[0])
    d = f'{r[2]:.4f}' if r[2] and r[2] > -1e10 else 'NaN'
    ts = f'{r[3]:.4f}' if r[3] and r[3] > -1e10 else 'NaN'
    ad = f'{r[4]:.4f}' if r[4] and r[4] > -1e10 else 'NaN'
    print(f'{t:>32} {r[1]:>14.4f} {d:>16} {ts:>15} {ad:>15} {r[5]:>13.4f}')

print(f'\n{"=" * 90}')
print('  DATA SOURCE 2: DSCOVR / OMNI2 Plasma (dscovr_plasma)')
print('=' * 90)
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='dscovr_plasma' ORDER BY ordinal_position")
print('Columns:', [f'{r[0]} ({r[1]})' for r in cur.fetchall()])

# OMNI2 rows
cur.execute("SELECT observation_time, proton_speed, proton_density, proton_temperature, source FROM dscovr_plasma WHERE source='NASA_OMNI2' AND proton_speed IS NOT NULL ORDER BY observation_time LIMIT 3")
rows = cur.fetchall()
header = f"{'observation_time':>32} {'proton_speed':>14} {'proton_density':>16} {'proton_temp':>13} {'source':>16}"
print(f"\n{header}")
print('-' * len(header))
for r in rows:
    temp = f'{r[3]:.1f}' if r[3] else 'NULL'
    print(f'{str(r[0]):>32} {r[1]:>14.2f} {r[2]:>16.2f} {temp:>13} {r[4]:>16}')

# NOAA_DSCOVR rows
cur.execute("SELECT observation_time, proton_speed, proton_density, proton_temperature, source FROM dscovr_plasma WHERE source='NOAA_DSCOVR' ORDER BY observation_time LIMIT 3")
for r in cur.fetchall():
    temp = f'{r[3]:.1f}' if r[3] else 'NULL'
    print(f'{str(r[0]):>32} {r[1]:>14.2f} {r[2]:>16.2f} {temp:>13} {r[4]:>16}')

print(f'\n{"=" * 90}')
print('  DATA SOURCE 3: Kp Index (geomagnetic_indices)')
print('=' * 90)
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='geomagnetic_indices' ORDER BY ordinal_position")
print('Columns:', [f'{r[0]} ({r[1]})' for r in cur.fetchall()])

cur.execute("SELECT timestamp, kp_index, source FROM geomagnetic_indices ORDER BY timestamp LIMIT 3")
rows = cur.fetchall()
header = f"{'timestamp':>32} {'kp_index':>10} {'source':>16}"
print(f"\n{header}")
print('-' * len(header))
for r in rows:
    print(f'{str(r[0]):>32} {r[1]:>10.2f} {r[2]:>16}')
cur.execute("SELECT timestamp, kp_index, source FROM geomagnetic_indices WHERE source='NOAA_SWPC' ORDER BY timestamp LIMIT 3")
for r in cur.fetchall():
    print(f'{str(r[0]):>32} {r[1]:>10.2f} {r[2]:>16}')

print(f'\n{"=" * 90}')
print('  SUMMARY')
print('=' * 90)
for table, name in [('swis_moments', 'Aditya-L1'), ('dscovr_plasma', 'DSCOVR/OMNI2'), ('geomagnetic_indices', 'Kp Index')]:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    cnt = cur.fetchone()[0]
    cur.execute(f"SELECT MIN(observation_time), MAX(observation_time) FROM {table}" if table != 'geomagnetic_indices'
                else f"SELECT MIN(timestamp), MAX(timestamp) FROM {table}")
    mn, mx = cur.fetchone()
    print(f'  {name:<20} {cnt:>10,} records   {str(mn.date()):>12} to {str(mx.date()):>12}')

conn.close()
