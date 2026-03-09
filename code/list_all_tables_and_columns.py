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
    
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"Found {len(tables)} tables in database:\n")
    
    for table in tables:
        print(f"Table: {table}")
        # Get columns for this table
        cursor.execute(f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """)
        columns = cursor.fetchall()
        print("  Columns:")
        for col in columns:
            print(f"    - {col[0]} ({col[1]}, nullable: {col[2]})")
        print("")
        
    conn.close()

except Exception as e:
    print(f"Error: {e}")
