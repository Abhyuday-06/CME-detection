import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

def apply_fixes():
    try:
        conn = psycopg2.connect(DB_URI)
        conn.autocommit = True  # Enable autocommit for structural changes
        cursor = conn.cursor()
        print("Connected to database...")

        # FIX 1: CONNECT ALERTS TO ANALYSIS RUNS
        print("\n[1/3] Updating 'alerts' table...")
        try:
            # Check if column exists first
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='alerts' AND column_name='run_id';")
            if not cursor.fetchone():
                cursor.execute("""
                    ALTER TABLE alerts 
                    ADD COLUMN run_id INTEGER;

                    ALTER TABLE alerts
                    ADD CONSTRAINT fk_alerts_run
                    FOREIGN KEY (run_id) 
                    REFERENCES analysis_runs(run_id);
                """)
                print(" - Added 'run_id' column and Foreign Key constraint.")
            else:
                print(" - 'run_id' column already exists.")
        except Exception as e:
            print(f" - Error: {e}")

        # FIX 2: CONNECT DATA TO PROCESSING LEVELS
        print("\n[2/3] Updating 'swis_moments' table (Processing Levels)...")
        try:
            # Check if column exists first
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='swis_moments' AND column_name='level_id';")
            if not cursor.fetchone():
                cursor.execute("""
                    ALTER TABLE swis_moments 
                    ADD COLUMN level_id INTEGER;

                    ALTER TABLE swis_moments
                    ADD CONSTRAINT fk_swis_level
                    FOREIGN KEY (level_id) 
                    REFERENCES processing_levels(level_id);
                """)
                print(" - Added 'level_id' column and Foreign Key constraint.")
            else:
                print(" - 'level_id' column already exists.")
        except Exception as e:
            print(f" - Error: {e}")

        # FIX 3: COMPOSITE UNIQUE CONSTRAINT
        print("\n[3/3] Updating 'swis_moments' Unique Constraint...")
        try:
            # Drop old single-column constraint if it exists (captured from feeder.py logic)
            cursor.execute("""
                ALTER TABLE swis_moments DROP CONSTRAINT IF EXISTS unique_observation_time;
            """)
            
            # Add new composite constraint
            # Note: This might fail if you already have duplicate data for the same time/instrument.
            cursor.execute("""
                ALTER TABLE swis_moments 
                ADD CONSTRAINT unique_instrument_observation 
                UNIQUE (instrument_id, observation_time);
            """)
            print(" - Success: Added composite unique constraint (instrument_id + observation_time).")
        except psycopg2.errors.UniqueViolation:
             print(" - Critical Error: Cannot add constraint because duplicate data already exists in the table.")
             print(" - Tip: Run 'DELETE FROM swis_moments a USING swis_moments b WHERE a.record_id < b.record_id AND a.observation_time = b.observation_time;' to clean it.")
        except Exception as e:
             # Check if constraint already exists
             if "already exists" in str(e):
                 print(" - Constraint already exists.")
             else:
                 print(f" - Error: {e}")

    except Exception as e:
        print(f"Fatal Database Error: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    apply_fixes()
