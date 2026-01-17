import psycopg2
from werkzeug.security import generate_password_hash
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

def setup_database():
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        print("Connected to database.")

        # 1. Create Roles Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                role_id SERIAL PRIMARY KEY,
                role_name VARCHAR(50) UNIQUE NOT NULL
            );
        """)
        
        # 2. Create Users Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email VARCHAR(100),
                role_id INT REFERENCES roles(role_id),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # 3. Create Model Feedback Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_feedback (
                feedback_id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(user_id),
                observation_time TIMESTAMPTZ,
                ai_predicted_speed DOUBLE PRECISION,
                human_corrected_speed DOUBLE PRECISION,
                comments TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # 4. Create Action Logs Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS action_logs (
                log_id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(user_id),
                action_type VARCHAR(50),
                details TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        print("Tables created successfully.")

        # 5. Insert Roles
        roles = ['scientist', 'viewer']
        for role in roles:
            cur.execute("INSERT INTO roles (role_name) VALUES (%s) ON CONFLICT (role_name) DO NOTHING;", (role,))
        
        # 6. Get Role IDs
        cur.execute("SELECT role_id, role_name FROM roles;")
        role_map = {name: r_id for r_id, name in cur.fetchall()}
        
        # 7. Insert Default Users
        admin_pass = os.getenv('ADMIN_PASSWORD', 'admin123')
        guest_pass = os.getenv('GUEST_PASSWORD', 'guest123')
        
        users = [
            ('admin', admin_pass, 'admin@solarwatch.org', 'scientist'),
            ('guest', guest_pass, 'guest@solarwatch.org', 'viewer')
        ]

        for username, password, email, role_name in users:
            role_id = role_map.get(role_name)
            if role_id:
                password_hash = generate_password_hash(password)
                # Check if user exists
                cur.execute("SELECT user_id FROM users WHERE username = %s;", (username,))
                if not cur.fetchone():
                    cur.execute("""
                        INSERT INTO users (username, password_hash, email, role_id)
                        VALUES (%s, %s, %s, %s);
                    """, (username, password_hash, email, role_id))
                    print(f"Created user: {username} ({role_name})")
                else:
                    print(f"User {username} already exists.")
            else:
                print(f"Role {role_name} not found!")

        conn.commit()
        cur.close()
        conn.close()
        print("Database setup complete.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    setup_database()
