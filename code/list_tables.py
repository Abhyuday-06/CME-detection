import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

try:
    conn = psycopg2.connect(DB_URI)
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    
    tables = cursor.fetchall()
    print("Tables in database:")
    for table in tables:
        print(f"- {table[0]}")
        
    conn.close()

except Exception as e:
    print(f"Error: {e}")
