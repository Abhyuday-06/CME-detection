import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

def update_schema():
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        print("Connected to database.")

        # Check if full_name column exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='users' AND column_name='full_name';
        """)
        
        if not cur.fetchone():
            print("Adding 'full_name' column to users table...")
            cur.execute("ALTER TABLE users ADD COLUMN full_name VARCHAR(100);")
            conn.commit()
            print("Column added successfully.")
        else:
            print("'full_name' column already exists.")

        conn.close()

    except Exception as e:
        print(f"Error updating schema: {e}")

if __name__ == "__main__":
    update_schema()
