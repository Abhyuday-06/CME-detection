import psycopg2
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

def get_db_connection():
    return psycopg2.connect(DB_URI)

def seed_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("Seeding Database...")

    # 1. SATELLITES
    print("   - Satellites")
    satellites = [
        ('Aditya-L1', 'ISRO', '2023-09-02'),
        ('SOHO', 'ESA/NASA', '1995-12-02'),
        ('ACE', 'NASA', '1997-08-25'),
        ('DSCOVR', 'NOAA', '2015-02-11')
    ]
    
    sat_ids = {}
    for name, agency, date in satellites:
        cursor.execute("""
            INSERT INTO satellites (name, agency, launch_date) 
            VALUES (%s, %s, %s) 
            ON CONFLICT (name) DO UPDATE SET agency = EXCLUDED.agency
            RETURNING satellite_id;
        """, (name, agency, date))
        sat_id = cursor.fetchone()[0]
        sat_ids[name] = sat_id

    # 2. INSTRUMENTS
    print("   - Instruments")
    instruments = [
        (sat_ids['Aditya-L1'], 'ASPEX', 'Particle Analyzer'),
        (sat_ids['Aditya-L1'], 'PAPA', 'Plasma Analyser'),
        (sat_ids['Aditya-L1'], 'MAG', 'Magnetometer'),
        (sat_ids['Aditya-L1'], 'SUIT', 'UV Telescope'),
        (sat_ids['Aditya-L1'], 'VELC', 'Coronagraph'),
        (sat_ids['SOHO'], 'LASCO', 'Coronagraph'),
        (sat_ids['ACE'], 'SWEPAM', 'Plasma Monitor'),
        (sat_ids['DSCOVR'], 'PlasMag', 'Plasma Magnetometer')
    ]
    
    for sat_id, name, type_ in instruments:
        cursor.execute("""
            INSERT INTO instruments (satellite_id, name, type) 
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (sat_id, name, type_))

    # 3. PROCESSING LEVELS
    print("   - Processing Levels")
    levels = [
        ('L0', 'Raw Telemetry Data'),
        ('L1', 'Calibrated Data in Physical Units'),
        ('L2', 'Derived Scientific Products (Moments)')
    ]
    for code, desc in levels:
        cursor.execute("""
            INSERT INTO processing_levels (code, description) 
            VALUES (%s, %s)
            ON CONFLICT (code) DO NOTHING;
        """, (code, desc))

    # 4. ROLES
    print("   - Roles")
    roles = ['Admin', 'Scientist', 'Viewer']
    role_ids = {}
    for r in roles:
        cursor.execute("""
            INSERT INTO roles (role_name) VALUES (%s)
            ON CONFLICT (role_name) DO UPDATE SET role_name = EXCLUDED.role_name
            RETURNING role_id;
        """, (r,))
        role_ids[r] = cursor.fetchone()[0]

    # 5. USERS (Default Admin)
    print("   - Users")
    # Password: 'admin_password' (hashed)
    # In a real app, use bcrypt/argon2. Here using SHA256 for simplicity of the demo.
    admin_pass = os.getenv('ADMIN_PASSWORD', 'admin123')
    pass_hash = hashlib.sha256(admin_pass.encode()).hexdigest()
    
    cursor.execute("""
        INSERT INTO users (username, password_hash, email, role_id, created_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (username) DO NOTHING;
    """, ('admin', pass_hash, 'admin@isro.gov.in', role_ids['Admin'], datetime.now()))

    conn.commit()
    conn.close()
    print("Database Seeded Successfully!")

if __name__ == "__main__":
    try:
        seed_database()
    except Exception as e:
        print(f"Error: {e}")
