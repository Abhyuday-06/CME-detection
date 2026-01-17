import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

try:
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    
    cur.execute("SELECT MIN(observation_time), MAX(observation_time), COUNT(*) FROM swis_moments;")
    min_t, max_t, count = cur.fetchone()
    
    print(f"Total Records: {count}")
    print(f"Time Range: {min_t} to {max_t}")
    
    conn.close()
except Exception as e:
    print(e)