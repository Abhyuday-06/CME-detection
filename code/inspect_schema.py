import psycopg2
from psycopg2.extras import execute_values
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

def get_db_connection():
    return psycopg2.connect(DB_URI)

def seed_satellites(conn):
    print("Seeding Satellites...")
    cursor = conn.cursor()
    satellites = [
        ('Aditya-L1', 'ISRO', 'L1 Point', '2023-09-02'),
        ('SOHO', 'ESA/NASA', 'L1 Point', '1995-12-02'),
        ('ACE', 'NASA', 'L1 Point', '1997-08-25'),
        ('DSCOVR', 'NOAA', 'L1 Point', '2015-02-11')
    ]
    # Assuming table structure: name, agency, orbit, launch_date
    # I need to check the schema to be sure, but I'll try a generic INSERT and catch errors if columns mismatch.
    # Actually, better to check columns first or use a safe insert if I knew the schema.
    # Since I don't have the schema, I'll assume standard columns based on table names.
    # If it fails, I'll inspect the table columns.
    
    # Let's inspect columns first to be safe.
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'satellites'")
    columns = [row[0] for row in cursor.fetchall()]
    print(f"  Satellites Columns: {columns}")
    
    # Based on columns, I will construct the INSERT.
    # For now, I'll just print columns and return, so I can adjust the script in the next step.
    return

def inspect_tables(conn):
    cursor = conn.cursor()
    tables = ['satellites', 'instruments', 'processing_levels', 'roles', 'users']
    schema_info = {}
    
    for table in tables:
        cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")
        columns = [row[0] for row in cursor.fetchall()]
        schema_info[table] = columns
        print(f"Table '{table}': {columns}")
        
    return schema_info

if __name__ == "__main__":
    try:
        conn = get_db_connection()
        inspect_tables(conn)
        conn.close()
    except Exception as e:
        print(f"Error: {e}")
