import os
import cdflib
import psycopg2
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

try:
    conn = psycopg2.connect(DB_URI)
    cursor = conn.cursor()
    print("Connected to PostgreSQL successfully.")
    
    # ENSURE UNIQUE CONSTRAINT EXISTS
    # This prevents duplicates and allows us to use "ON CONFLICT DO NOTHING"
    try:
        cursor.execute("""
            ALTER TABLE swis_moments 
            ADD CONSTRAINT unique_observation_time UNIQUE (observation_time);
        """)
        conn.commit()
        print("Constraint added: 'observation_time' is now unique.")
    except psycopg2.errors.DuplicateTable:
        # Constraint already exists, all good
        conn.rollback()
        pass
    except psycopg2.errors.InvalidJsonText:
        conn.rollback()
    except Exception as e:
        # If table setup is different, just warn
        # If duplicate keys exist, we can't create the constraint.
        print(f"Notice: Could not add unique constraint. Reason: {e}")
        conn.rollback()
        
except Exception as e:
    print(f"Database connection failed: {e}")
    exit()

def process_cdf_file(filepath):
    print(f"  Processing: {filepath}...")
    
    try:
        cdf = cdflib.CDF(filepath)
        
        # Use correct CDF variable names
        epochs = cdf.varget("epoch_for_cdf_mod") 
        p_density = cdf.varget("proton_density")
        p_speed = cdf.varget("proton_bulk_speed")
        p_temp = cdf.varget("proton_thermal")
        
        a_density = cdf.varget("alpha_density")
        a_speed = cdf.varget("alpha_bulk_speed")
        a_temp = cdf.varget("alpha_thermal")
        
        sc_x = cdf.varget("spacecraft_xpos")
        sc_y = cdf.varget("spacecraft_ypos")
        sc_z = cdf.varget("spacecraft_zpos")

        # Convert CDF Epoch to Python Datetime objects
        timestamps = cdflib.cdfepoch.to_datetime(epochs)
        
        records_added = 0
        
        for i in range(len(timestamps)):
            
            # Skip rows with bad data (Fill Values are often -1.0E31)
            # Also check for valid speed range
            if p_speed[i] < 0 or p_speed[i] > 2000: 
                continue

            sql = """
                INSERT INTO swis_moments 
                (observation_time, proton_density, proton_speed, proton_thermal_speed, alpha_density, alpha_speed, alpha_thermal_speed, sc_x, sc_y, sc_z)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """
            
            # Explicitly convert numpy types to Python float/int for SQL compatibility
            # Convert numpy.datetime64 to python datetime
            ts = timestamps[i]
            if hasattr(ts, 'item'):
                ts = ts.item()
                
            # If it's still an integer (nanoseconds), convert to datetime
            if isinstance(ts, int):
                # Assuming nanoseconds if it's huge
                if ts > 1e18: 
                    ts = datetime.fromtimestamp(ts / 1e9)
            
            data = (
                ts,
                float(p_density[i]),
                float(p_speed[i]),
                float(p_temp[i]),
                float(a_density[i]),
                float(a_speed[i]),
                float(a_temp[i]),
                float(sc_x[i]),
                float(sc_y[i]),
                float(sc_z[i])
            )
            
            cursor.execute(sql, data)
            records_added += 1

        conn.commit()
        print(f"   Success! Inserted {records_added} rows.")

    except Exception as e:
        print(f"   Error processing file: {e}")
        conn.rollback()

def main():
    base_folder = "E:\Programming\CME-Detection\data\SWIS-ISSDC"
    subfolders = ["positive", "negative"]

    for sub in subfolders:
        folder_path = os.path.join(base_folder, sub)
        
        if not os.path.exists(folder_path):
            print(f" Warning: Folder not found: {folder_path}")
            continue
            
        print(f"\n Scanning folder: {folder_path}")
        files = sorted(os.listdir(folder_path))
        
        for filename in files:
            if filename.endswith(".cdf"):
                
                # If V01 and V02 both exist, skip V01
                if "V01" in filename:
                    v2_name = filename.replace("V01", "V02")
                    if v2_name in files:
                        print(f"  Skipping {filename} (Newer V02 exists)")
                        continue
                
                full_path = os.path.join(folder_path, filename)
                process_cdf_file(full_path)

    cursor.close()
    conn.close()
    print("\n All files processed. Database is populated!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory containing CDF files to process")
    args = parser.parse_args()

    if args.dir:
        # Custom directory mode (for web uploads)
        if os.path.exists(args.dir):
            print(f"Scanning provided directory: {args.dir}")
            for filename in sorted(os.listdir(args.dir)):
                if filename.endswith(".cdf"):
                     process_cdf_file(os.path.join(args.dir, filename))
            cursor.close()
            conn.close()
        else:
            print(f"Directory not found: {args.dir}")
    else:
        # Default recursive scan mode
        main()
