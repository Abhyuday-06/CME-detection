import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import matplotlib.dates as mdates
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')

def fetch_data(start_date, end_date):
    conn = psycopg2.connect(DB_URI)
    
    # Filter out bad data (Speed > 2000 or < 0 is noise)
    swis_query = f"""
        SELECT observation_time, proton_speed, proton_density 
        FROM swis_moments 
        WHERE observation_time BETWEEN '{start_date}' AND '{end_date}'
        AND proton_speed > 0 AND proton_speed < 2000
        ORDER BY observation_time ASC;
    """
    df_swis = pd.read_sql(swis_query, conn)
    
    # Get Ground Truth Events (The Validation)
    cme_query = f"""
        SELECT start_time, velocity, is_halo 
        FROM cme_events 
        WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY start_time ASC;
    """
    df_cme = pd.read_sql(cme_query, conn)
    
    conn.close()
    return df_swis, df_cme

def plot_space_weather(df_swis, df_cme, title_date):
    if df_swis.empty:
        print(" No SWIS data found for this range! Did you run 'feeder.py' for these dates?")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- PLOT 1: PROTON SPEED (The Shock) ---
    color = 'tab:red'
    ax1.set_xlabel('Time (UTC)')
    ax1.set_ylabel('Proton Speed (km/s)', color=color)
    ax1.plot(df_swis['observation_time'], df_swis['proton_speed'], color=color, linewidth=1, label='Speed')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- PLOT 2: PROTON DENSITY (The Cloud) ---
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Proton Density (cm⁻³)', color=color)
    ax2.plot(df_swis['observation_time'], df_swis['proton_density'], color=color, alpha=0.3, label='Density')
    ax2.tick_params(axis='y', labelcolor=color)

    # --- OVERLAY: CME EVENTS (The Truth) ---
    # Draw vertical lines where CACTUS says a CME happened
    for index, row in df_cme.iterrows():
        line_color = 'purple' if row['is_halo'] else 'green'
        label = "HALO CME" if row['is_halo'] else "CME"
        
        plt.axvline(x=row['start_time'], color=line_color, linestyle='--', linewidth=2)
        plt.text(row['start_time'], ax1.get_ylim()[1]*0.9, label, color=line_color, rotation=90)

    plt.title(f'Aditya-L1 Space Weather: {title_date}')
    plt.tight_layout()
    plt.show()

# --- RUN IT ---
# Change these dates to match the files you actually downloaded!
START = "2024-10-05 00:00:00"
END   = "2024-10-15 00:00:00"

print(f"Generating plot for {START} to {END}...")
swis, cmes = fetch_data(START, END)
plot_space_weather(swis, cmes, "August Quiet Period")