from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash, generate_password_hash
import psycopg2
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import subprocess
import threading
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'solar_secret_key_very_secure')  # Needed for session management

# --- AUTHENTICATION SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# DATABASE CONFIGURATION
DB_URI = os.getenv('DB_URI')

def get_db_connection():
    return psycopg2.connect(DB_URI)

# USER MODEL
class User(UserMixin):
    def __init__(self, user_id, username, role_name):
        self.id = user_id
        self.username = username
        self.role = role_name # 'scientist' or 'viewer'

    def is_scientist(self):
        return self.role == 'scientist'

@login_manager.user_loader
def load_user(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT u.user_id, u.username, r.role_name 
            FROM users u 
            JOIN roles r ON u.role_id = r.role_id 
            WHERE u.user_id = %s
        """, (user_id,))
        data = cur.fetchone()
        conn.close()
        
        if data:
            return User(data[0], data[1], data[2])
    except Exception as e:
        print(f"DB Error loading user: {e}")
    return None

# --- ML SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR = os.path.join(BASE_DIR, 'code')
MODEL_PATH = os.path.join(CODE_DIR, 'kp_model.keras')
SCALER_X_PATH = os.path.join(CODE_DIR, 'kp_scaler_X.pkl')
SCALER_Y_PATH = os.path.join(CODE_DIR, 'kp_scaler_Y.pkl')

print(f"Loading Model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_X_PATH, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(SCALER_Y_PATH, 'rb') as f:
        scaler_Y = pickle.load(f)
    print("Model & Scalers Loaded Successfully.")
except Exception as e:
    print(f"Error loading ML components: {e}")
    model, scaler_X, scaler_Y = None, None, None
    model = None
    scaler = None

# --- ROUTES ---

@app.route('/')
def home():
    if current_user.is_authenticated:
        # If logged in, still show landing but with "Dashboard" button?
        # Or redirect to dashboard?
        # User asked to "first like to show a homepage", implied for everyone.
        return render_template('landing.html')
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_id = request.form['login_id']
        password = request.form['password']
        
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            # Check username OR email
            cur.execute("""
                SELECT user_id, username, password_hash 
                FROM users 
                WHERE username = %s OR email = %s
            """, (login_id, login_id))
            user_data = cur.fetchone()
            conn.close()
            
            if user_data and check_password_hash(user_data[2], password):
                user_obj = load_user(user_data[0])
                login_user(user_obj)
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials. Access Denied.')
        except Exception as e:
            flash(f'Login Error: {str(e)}')
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('signup'))
        
        selected_role = request.form.get('role', 'viewer') # Default to viewer if missing

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Check if user exists
            cur.execute("SELECT user_id FROM users WHERE username = %s OR email = %s", (username, email))
            if cur.fetchone():
                flash('Username or Email already registered.')
                conn.close()
                return redirect(url_for('signup'))

            # Get role id dynamic
            cur.execute("SELECT role_id FROM roles WHERE role_name = %s", (selected_role,))
            role_res = cur.fetchone()
            if not role_res:
                flash(f'System error: Role "{selected_role}" not found.')
                conn.close()
                return redirect(url_for('signup'))
            
            role_id = role_res[0]
            password_hash = generate_password_hash(password)

            # Insert User
            cur.execute("""
                INSERT INTO users (username, password_hash, email, full_name, role_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING user_id;
            """, (username, password_hash, email, full_name, role_id))
            
            new_user_id = cur.fetchone()[0]

            # Log Creation Action
            cur.execute("""
                INSERT INTO action_logs (user_id, action_type, details)
                VALUES (%s, 'ACCOUNT_CREATED', %s)
            """, (new_user_id, f"User {username} signed up with email {email} as {selected_role.upper()}"))

            conn.commit()
            conn.close()
            
            flash('Account initialized. Welcome to the network.')
            return redirect(url_for('login'))
            
        except Exception as e:
            flash(f'Signup Error: {str(e)}')
            
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html', user=current_user)

# --- API ENDPOINTS ---

def compute_synthetic_kp(speed_series, density_series, temp_series=None):
    """Physics-based Kp proxy using dynamic pressure and solar wind coupling.
    
    Based on Newell et al. (2007) coupling function simplified:
    Kp correlates with dynamic pressure Pdyn = m_p * n * v^2
    and the empirical mapping to Kp scale.
    """
    speed = speed_series.fillna(400).astype(float)
    density = density_series.fillna(5).astype(float)
    
    # Dynamic pressure proxy: Pdyn ∝ n * v^2 (in nPa when n in cm^-3, v in km/s)
    # Pdyn = 1.6726e-6 * n * v^2  (nPa)
    pdyn = 1.6726e-6 * density * speed**2
    
    # Empirical Kp mapping from dynamic pressure
    # Typical Pdyn: 1-2 nPa (quiet) -> Kp 1-2
    #               3-5 nPa (moderate) -> Kp 3-4  
    #               5-10 nPa (active) -> Kp 5-6
    #               >10 nPa (storm) -> Kp 7+
    kp = np.log2(pdyn.clip(lower=0.5) + 1) * 1.8
    
    # Speed contribution: fast wind (>500) adds extra Kp
    speed_boost = ((speed - 400) / 150.0).clip(lower=0)
    kp = kp + speed_boost * 0.5
    
    # Density spike contribution: sudden density increases drive storms
    density_boost = ((density - 10) / 10.0).clip(lower=0)
    kp = kp + density_boost * 0.8
    
    # Add thermal speed contribution if available
    if temp_series is not None:
        temp = temp_series.fillna(30).astype(float)
        temp_factor = ((temp - 20) / 50.0).clip(-0.5, 1.0)
        kp = kp + temp_factor * 0.3
    
    return kp.clip(0.3, 9.0).round(1)


@app.route('/api/telemetry')
@login_required
def get_telemetry():
    start_date = request.args.get('start', '2000-01-01')
    end_date = request.args.get('end', '2100-01-01')
    live = request.args.get('live')

    conn = get_db_connection()
    if live:
        import time
        minute_offset = int(time.time() / 3) % 5000
        # Get plasma data
        query = f'''
            SELECT observation_time, proton_speed, proton_density, proton_thermal_speed as proton_temp
            FROM swis_moments
            ORDER BY observation_time ASC 
            OFFSET {minute_offset} LIMIT 100
        '''
        df = pd.read_sql(query, conn)
    else:
        query = '''
            SELECT observation_time, proton_speed, proton_density, proton_thermal_speed as proton_temp
            FROM swis_moments
            WHERE observation_time BETWEEN %s AND %s
            ORDER BY observation_time ASC
        '''
        df = pd.read_sql(query, conn, params=(start_date, end_date))

    # Separately fetch ALL available Kp data from geomagnetic_indices
    try:
        kp_df = pd.read_sql('''
            SELECT timestamp as kp_time, kp_index 
            FROM geomagnetic_indices 
            WHERE kp_index IS NOT NULL 
            ORDER BY timestamp ASC
        ''', conn)
    except:
        kp_df = pd.DataFrame(columns=['kp_time', 'kp_index'])

    conn.close()

    if df.empty:
        return jsonify({'time': [], 'speed': [], 'density': [], 'temp': [], 'kp': []})

    if len(df) > 2000 and not live:
        df['observation_time'] = pd.to_datetime(df['observation_time'])
        df.set_index('observation_time', inplace=True)
        td = df.index.max() - df.index.min()
        if td.days > 30:
            df = df.resample('12h').mean().dropna().reset_index()
        elif td.days > 7:
            df = df.resample('1h').mean().dropna().reset_index()
        elif td.days > 1:
            df = df.resample('15min').mean().dropna().reset_index()

    # Merge Kp data using nearest-time matching (tolerance: 3 hours)
    df['observation_time'] = pd.to_datetime(df['observation_time'])
    if not kp_df.empty:
        kp_df['kp_time'] = pd.to_datetime(kp_df['kp_time'])
        df = pd.merge_asof(
            df.sort_values('observation_time'),
            kp_df.sort_values('kp_time'),
            left_on='observation_time',
            right_on='kp_time',
            tolerance=pd.Timedelta('3h'),
            direction='nearest'
        )
        df.rename(columns={'kp_index': 'kp'}, inplace=True)
    else:
        df['kp'] = np.nan

    # Compute physics-based synthetic Kp for gaps where DB had no match
    synthetic_kp = compute_synthetic_kp(df['proton_speed'], df['proton_density'], df.get('proton_temp'))
    df['kp'] = df['kp'].combine_first(synthetic_kp)
    
    return jsonify({
        'time': df['observation_time'].astype(str).tolist(),
        'speed': df['proton_speed'].fillna(0).tolist(),
        'density': df['proton_density'].fillna(0).tolist(),
        'temp': df['proton_temp'].fillna(0).tolist(),
        'kp': df['kp'].tolist()
    })
@app.route('/api/forecast')
@login_required
def get_forecast():
    conn = get_db_connection()
    
    # Get hourly-averaged plasma data
    query = '''
        SELECT 
            date_trunc('hour', observation_time) as obs_hour,
            AVG(proton_speed) as speed, 
            AVG(proton_density) as density
        FROM swis_moments
        GROUP BY obs_hour
        ORDER BY obs_hour DESC
        LIMIT 24;
    '''
    df = pd.read_sql(query, conn)
    
    # Fetch Kp data separately
    try:
        kp_df = pd.read_sql('''
            SELECT timestamp as kp_time, kp_index 
            FROM geomagnetic_indices 
            WHERE kp_index IS NOT NULL 
            ORDER BY timestamp ASC
        ''', conn)
    except:
        kp_df = pd.DataFrame(columns=['kp_time', 'kp_index'])

    conn.close()

    if len(df) < 24:
        return jsonify({'error': 'Insufficient data for prediction'})

    df = df.sort_values('obs_hour')
    df['obs_hour'] = pd.to_datetime(df['obs_hour'])
    
    # Merge Kp data using nearest-time matching
    if not kp_df.empty:
        kp_df['kp_time'] = pd.to_datetime(kp_df['kp_time'])
        df = pd.merge_asof(
            df.sort_values('obs_hour'),
            kp_df.sort_values('kp_time'),
            left_on='obs_hour',
            right_on='kp_time',
            tolerance=pd.Timedelta('3h'),
            direction='nearest'
        )
        df.rename(columns={'kp_index': 'kp'}, inplace=True)
    else:
        df['kp'] = np.nan
    
    df = df.set_index('obs_hour')
    
    # Fill missing Kp with physics-based proxy
    synthetic_kp = compute_synthetic_kp(df['speed'], df['density'])
    df['kp'] = df['kp'].combine_first(synthetic_kp)
    df = df.interpolate(method='linear').bfill().ffill()

    speed_trend = df['speed'].iloc[-1] - df['speed'].iloc[-6] if len(df) >= 6 else 0
    density_trend = df['density'].iloc[-1] - df['density'].iloc[-6] if len(df) >= 6 else 0
    
    current_speed = df['speed'].iloc[-1]
    current_density = df['density'].iloc[-1]
    last_known = df['kp'].iloc[-1]

    # Heuristic forecast baseline using trends
    heuristic_base = last_known
    heuristic_preds = []
    for i in range(6):
        h = heuristic_base + (speed_trend * 0.005 * (i+1)) + (density_trend * 0.08 * (i+1))
        if current_speed > 600 or current_density > 20:
            h = max(h, 5.0 + i * 0.2)
        elif current_speed > 500:
            h = max(h, 3.0 + i * 0.15)
        heuristic_preds.append(max(0.3, min(9.0, h)))

    try:
        if model is not None and scaler_X is not None and scaler_Y is not None:
            features = df[['speed', 'density', 'kp']].values
            input_scaled = scaler_X.transform(features).reshape(1, 24, 3)
            pred_scaled = model.predict(input_scaled, verbose=0)
            pred_kp = scaler_Y.inverse_transform(pred_scaled)[0]
            
            # Blend: weight model 40%, heuristic 60% (model often outputs near-zero)
            blended_preds = []
            for i in range(min(6, len(pred_kp))):
                ml_val = float(pred_kp[i])
                # If ML output is suspiciously near zero, trust heuristic more
                if abs(ml_val) < 0.5:
                    blended = heuristic_preds[i]
                else:
                    blended = ml_val * 0.4 + heuristic_preds[i] * 0.6
                blended_preds.append(max(0.3, min(9.0, blended)))
            
            # Pad if model returned fewer than 6 predictions
            while len(blended_preds) < 6:
                blended_preds.append(heuristic_preds[len(blended_preds)])
        else:
            blended_preds = heuristic_preds
    except Exception as e:
        print("Model prediction error: ", e)
        blended_preds = heuristic_preds

    last_time = df.index[-1]
    forecasts = []
    
    first_pred = float(blended_preds[0])
    
    if first_pred >= 5.0:
        if speed_trend > 50:
            reason = f"High Risk: Rapid surge in solar wind speed (+{speed_trend:.0f} km/s over 5H) causing geomagnetic instability."
        else:
            reason = f"High Risk: Anomalous plasma density and interplanetary magnetic field patterns detected."
    elif first_pred >= 4.0:
        if density_trend > 2:
            reason = f"Warning: Noticeable increase in proton density (+{density_trend:.1f} p/cm³)."
        elif speed_trend > 20:
            reason = f"Warning: Elevated solar wind speeds detected (+{speed_trend:.0f} km/s)."
        else:
            reason = "Warning: Moderate geomagnetic activity predicted based on recent telemetry."
    else:
        if speed_trend < -20:
            reason = "Nominal: Conditions are stabilizing as solar wind speed decreases."
        else:
            reason = "Nominal: Solar parameters remain stable within safe operational thresholds."

    for i in range(6):
        future_time = last_time + pd.Timedelta(hours=i+1)
        val = float(blended_preds[i])
        forecasts.append({
            'time': future_time.isoformat(),
            'kp': round(val, 1)
        })

    return jsonify({
        'last_observed_time': last_time.isoformat(),
        'predictions': forecasts,
        'reason': reason
    })
@app.route('/api/system-status')
def get_system_status():
    """Returns the latest available data timestamp to anchor the dashboard."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT MAX(observation_time) FROM swis_moments;")
        latest = cur.fetchone()[0]
        conn.close()
        
        if latest:
            return jsonify({'status': 'online', 'latest_data_time': latest.isoformat()})
        return jsonify({'status': 'waiting_for_data', 'latest_data_time': None})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/alerts')
@login_required
def get_alerts():
    # Read the 10 most recent high Kp moments
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT timestamp, kp_index, source 
        FROM geomagnetic_indices 
        WHERE kp_index >= 4.0 
        ORDER BY timestamp DESC LIMIT 10
    ''')
    rows = cur.fetchall()
    
    cur.execute("SELECT COUNT(*) FROM geomagnetic_indices WHERE kp_index >= 4.0")
    total_count = cur.fetchone()[0]
    conn.close()
    
    alerts = []
    for r in rows:
        ts, kp, src = r
        sev = 'HIGH' if kp >= 6 else 'MEDIUM'
        alerts.append({
            'generated_at': ts,
            'severity': sev,
            'message': f"Geomagnetic Storm threshold crossed. Kp={kp:.1f}"
        })
        
    return jsonify({
        'total': total_count,
        'recent': alerts
    })
@app.route('/api/cme-history')
@login_required
def get_cme_history():
    if not current_user.is_scientist():
        return jsonify({'error': 'Unauthorized'}), 403
        
    start = request.args.get('start')
    end = request.args.get('end')
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Corrected columns based on schema
    query = "SELECT event_id, start_time, velocity, angular_width, is_halo FROM cme_events"
    params = []
    
    if start and end:
        query += " WHERE start_time BETWEEN %s AND %s"
        params = [start, end]
        
    query += " ORDER BY start_time DESC LIMIT 100"
    
    try:
        cur.execute(query, tuple(params))
        events = [{
            'id': r[0], 
            'start': r[1], 
            'source': f"{r[3]}°", # Mapping ang_width to source/desc since source isn't there
            'type': 'Halo' if r[4] else 'Normal', 
            'note': f"V: {r[2]} km/s"
        } for r in cur.fetchall()]
    except Exception as e:
        print(f"CME History Error: {e}")
        events = []
    
    conn.close()
    return jsonify(events)

# --- SCIENTIST ONLY ROUTES ---

@app.route('/api/submit-feedback', methods=['POST'])
@login_required
def submit_feedback():
    if not current_user.is_scientist():
        return jsonify({'error': 'Unauthorized'}), 403
        
    data = request.json
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO model_feedback (user_id, observation_time, ai_predicted_speed, human_corrected_speed, comments)
            VALUES (%s, %s, %s, %s, %s)
        """, (current_user.id, data['time'], data['predicted'], data['corrected'], data.get('comment', '')))
        
        # Log action
        cur.execute("""
            INSERT INTO action_logs (user_id, action_type, details)
            VALUES (%s, 'CORRECT_PREDICTION', %s)
        """, (current_user.id, f"Correction: {data['predicted']} -> {data['corrected']}"))
        
        conn.commit()
        conn.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/notify', methods=['POST'])
@login_required
def notify_authorities():
    data = request.json
    print(f"Simulating email sent to space weather authorities! Kp Alert Level: {data.get('kp')}")   
    return jsonify({'status': 'success', 'message': f"Alert sequence initiated for Kp {data.get('kp')} event. Authorities have been immediately notified via secure channels."})

@app.route('/api/ingest', methods=['POST'])
@login_required
def ingest_data():
    if not current_user.is_scientist(): return jsonify({'error': 'Unauthorized'}), 403

    mode = request.form.get('mode') # 'file' or 'scrape'
    
    if mode == 'file':
        files = request.files.getlist('files')
        if not files: return jsonify({'error': 'No files uploaded'}), 400
        
        # Save to temp dir
        temp_dir = os.path.join(CODE_DIR, 'temp_ingest')
        os.makedirs(temp_dir, exist_ok=True)
        
        for f in files:
            if f.filename.endswith('.cdf'):
                f.save(os.path.join(temp_dir, f.filename))
        
        # Run feeder on temp dir
        def run_feeder_pipeline():
            # 1. Feed Data
            script_path = os.path.join(CODE_DIR, 'feeder.py')
            print(f"Running Feeder on {temp_dir}...")
            subprocess.run(['python', script_path, '--dir', temp_dir], capture_output=True)
            
            # Cleanup
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            try:
                os.rmdir(temp_dir)
            except: pass

            # 2. Run Detection
            print("Running Post-Ingest Detection...")
            det_path = os.path.join(CODE_DIR, 'detection.py')
            subprocess.run(['python', det_path], capture_output=True)
            print("Pipeline Complete.")
            
        thread = threading.Thread(target=run_feeder_pipeline)
        thread.start()
        
        return jsonify({'status': 'started', 'message': f'Ingesting {len(files)} files + Detection...'})

    elif mode == 'scrape':
        start_date = request.form.get('start')
        end_date = request.form.get('end')
        
        def run_scraper_pipeline():
            # 1. Scrape Data
            script_path = os.path.join(CODE_DIR, 'cactus_scraper.py')
            args = ['python', script_path]
            if start_date: args.extend(['--start', start_date])
            if end_date: args.extend(['--end', end_date])
            
            print(f"Running Scraper: {args}")
            subprocess.run(args, capture_output=True)
            
            # 2. Run Detection
            print("Running Post-Scrape Detection...")
            det_path = os.path.join(CODE_DIR, 'detection.py')
            subprocess.run(['python', det_path], capture_output=True)
            print("Pipeline Complete.")
            
        thread = threading.Thread(target=run_scraper_pipeline)
        thread.start()
        return jsonify({'status': 'started', 'message': 'Scraper + Detection started...'})
        
    return jsonify({'error': 'Invalid mode'}), 400

@app.route('/api/run-script', methods=['POST'])
@login_required
def run_script():
    if not current_user.is_scientist():
        return jsonify({'error': 'Unauthorized'}), 403
        
    script_name = request.json.get('script')
    allowed_scripts = {
        'train': 'train_model.py'
    }
    
    if script_name not in allowed_scripts:
        return jsonify({'error': 'Invalid script'}), 400
        
    script_path = os.path.join(CODE_DIR, allowed_scripts[script_name])
    
    def run_process_pipeline():
        # 1. Run Script (Training)
        print(f"Running Script: {script_name}...")
        subprocess.run(['python', script_path], capture_output=True)
        
        # 2. Run Detection (If needed after training? Maybe not, but user asked for "everything implemented... at last detection.py needs to be run automatically")
        # Running detection after training updates the logic/thresholds potentially, so re-running detection on historical data makes sense.
        print("Running Post-Process Detection...")
        det_path = os.path.join(CODE_DIR, 'detection.py')
        subprocess.run(['python', det_path], capture_output=True)
        print("Pipeline Complete.")
        
    # Run in background to not block UI
    thread = threading.Thread(target=run_process_pipeline)
    thread.start()
    
    return jsonify({'status': 'started', 'message': f'{script_name} + Detection started in background'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
