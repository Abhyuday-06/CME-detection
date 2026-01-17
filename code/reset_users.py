import psycopg2
from werkzeug.security import generate_password_hash
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

def reset_users():
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        
        # Ensure Roles Exist
        cur.execute("INSERT INTO roles (role_name) VALUES ('scientist'), ('viewer') ON CONFLICT (role_name) DO NOTHING")
        
        # Get Role IDs
        cur.execute("SELECT role_id FROM roles WHERE role_name = 'viewer'")
        viewer_id = cur.fetchone()[0]
        
        cur.execute("SELECT role_id FROM roles WHERE role_name = 'scientist'")
        scientist_id = cur.fetchone()[0]
        
        admin_pass = os.getenv('ADMIN_PASSWORD', 'admin123')
        guest_pass = os.getenv('GUEST_PASSWORD', 'guest123')

        users = [
            ("guest", guest_pass, viewer_id, "guest@solarwatch.com", "Guest User"),
            ("admin", admin_pass, scientist_id, "admin@solarwatch.com", "System Admin")
        ]
        
        for username, pwd, role_id, email, fullname in users:
            pwd_hash = generate_password_hash(pwd)
            print(f"Resetting {username}...")
            
            # Check if user exists
            cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
            res = cur.fetchone()
            
            if res:
                # Update
                cur.execute("""
                    UPDATE users 
                    SET password_hash = %s, role_id = %s, email = %s, full_name = %s
                    WHERE username = %s
                """, (pwd_hash, role_id, email, fullname, username))
            else:
                # Insert
                cur.execute("""
                    INSERT INTO users (username, password_hash, role_id, email, full_name)
                    VALUES (%s, %s, %s, %s, %s)
                """, (username, pwd_hash, role_id, email, fullname))
                
        conn.commit()
        conn.close()
        print("Users reset successfully.")
    except Exception as e:
        print(f"Error resetting users: {e}")

if __name__ == "__main__":
    reset_users()
