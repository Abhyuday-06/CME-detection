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
MODEL_PATH = os.path.join(CODE_DIR, 'cme_prediction_model.keras')
SCALER_PATH = os.path.join(CODE_DIR, 'scaler.pkl')

print(f"Loading Model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Model & Scaler Loaded Successfully.")
except Exception as e:
    print(f"Error loading ML components: {e}")
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

@app.route('/api/telemetry')
@login_required
def get_telemetry():
    # Limit default range for faster initial load
    start_date = request.args.get('start', '2024-05-10')
    end_date = request.args.get('end', '2024-10-15')
    
    conn = get_db_connection()
    # Downsample for big ranges using modulo if needed, but LIMIT is safer for now
    query = """
        SELECT observation_time, 
               proton_speed, proton_density, proton_thermal_speed, alpha_density,
               sc_x, sc_y, sc_z
        FROM swis_moments 
        WHERE observation_time BETWEEN %s AND %s
        ORDER BY observation_time ASC
        LIMIT 5000;
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    
    if df.empty:
        return jsonify({'time': [], 'speed': [], 'density': [], 'temperature': [], 'alpha_ratio': [], 'bx': [], 'by': [], 'bz': []})

    # Calculate Ratio
    df['alpha_ratio'] = (df['alpha_density'] / df['proton_density'].replace(0, np.nan)) * 100
    
    return jsonify({
        'time': df['observation_time'].astype(str).tolist(),
        'speed': df['proton_speed'].tolist(),
        'density': df['proton_density'].tolist(),
        'temperature': df['proton_thermal_speed'].tolist(),
        'alpha_ratio': df['alpha_ratio'].fillna(0).tolist(),
        'bx': df['sc_x'].fillna(0).tolist(),
        'by': df['sc_y'].fillna(0).tolist(),
        'bz': df['sc_z'].fillna(0).tolist()
    })

    # Calculate Ratio
    df['alpha_ratio'] = (df['alpha_density'] / df['proton_density'].replace(0, np.nan)) * 100
    
    return jsonify({
        'time': df['observation_time'].astype(str).tolist(),
        'speed': df['proton_speed'].tolist(),
        'density': df['proton_density'].tolist(),
        'temperature': df['proton_temperature'].tolist(),
        'alpha_ratio': df['alpha_ratio'].fillna(0).tolist()
    })

@app.route('/api/forecast')
@login_required
def get_forecast():
    if model is None:
        return jsonify({'error': 'Model not loaded'})
        
    conn = get_db_connection()
    # Get last 24h of data for prediction context
    query = """
        SELECT proton_speed, proton_density, proton_thermal_speed, alpha_density, observation_time
        FROM swis_moments
        ORDER BY observation_time DESC
        LIMIT 24;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    if len(df) < 24:
        return jsonify({'error': 'Insufficient data for prediction'})
    
    # Sort correctly as input needs to be chronological
    last_24h = df.sort_values('observation_time').set_index('observation_time')
    
    # --- PROGNOSTIC LOOP (Generate 6 Hours of Forecast) ---
    forecasts = []
    current_input_seq = last_24h.values # Shape (24, 4)
    last_known_time = last_24h.index[-1]
    
    for i in range(6):
        # Scale Input
        input_scaled = scaler.transform(current_input_seq)
        input_reshaped = input_scaled.reshape(1, 24, 4)
        
        # Predict Next Step
        prediction_scaled = model.predict(input_reshaped, verbose=0)
        predicted_speed = prediction_scaled[0][0] # Scaled result
        
        # Inverse Transform (using dummy array since scaler expects 4 features)
        dummy_row = np.zeros((1, 4))
        dummy_row[0, 0] = predicted_speed
        
        # For the next input step, we need to append this prediction.
        # However, our model is multivariate (4 features) but we only predict Speed.
        # A simple autoregressive strategy is to repeat the last known values for other features
        # or assume they stay constant. For this demo, we'll keep last known values constant.
        
        predicted_actual_speed = scaler.inverse_transform(dummy_row)[0][0]
        
        future_time = last_known_time + pd.Timedelta(hours=i+1)
        forecasts.append({
            'time': future_time.isoformat(),
            'speed': float(predicted_actual_speed)
        })
        
        # Update Sequence for next iteration:
        # 1. Shift array left
        # 2. Append new row [pred_speed, last_density, last_temp, last_alpha]
        new_row = current_input_seq[-1].copy()
        new_row[0] = predicted_actual_speed # Update speed with prediction
        
        current_input_seq = np.vstack([current_input_seq[1:], new_row])

    return jsonify({
        "predictions": forecasts,
        "last_observed_time": last_known_time.isoformat()
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
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get total count
    cur.execute("SELECT COUNT(*) FROM alerts")
    total_count = cur.fetchone()[0]
    
    # Get recent
    cur.execute("SELECT generated_at, severity, message FROM alerts ORDER BY generated_at DESC LIMIT 10")
    alerts = [{'generated_at': r[0], 'severity': r[1], 'message': r[2]} for r in cur.fetchall()]
    
    conn.close()
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
        # We start a thread to run the script
        def run_feeder():
            script_path = os.path.join(CODE_DIR, 'feeder.py')
            subprocess.run(['python', script_path, '--dir', temp_dir], capture_output=True)
            # Cleanup
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
            
        thread = threading.Thread(target=run_feeder)
        thread.start()
        
        return jsonify({'status': 'started', 'message': f'Ingesting {len(files)} files...'})

    elif mode == 'scrape':
        days = request.form.get('days', 7)
        # Assuming scraper script has args, or we just run it. 
        # Current scraper seems to do "latest". We'll just run it.
        def run_scraper():
            script_path = os.path.join(CODE_DIR, 'cactus_scraper.py')
            subprocess.run(['python', script_path], capture_output=True)
            
        thread = threading.Thread(target=run_scraper)
        thread.start()
        return jsonify({'status': 'started', 'message': 'Scraper started...'})
        
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
    
    def run_process():
        # In production this would use proper task queues like Celery
        subprocess.run(['python', script_path], capture_output=True)
        
    # Run in background to not block UI
    thread = threading.Thread(target=run_process)
    thread.start()
    
    return jsonify({'status': 'started', 'message': f'{script_name} started in background'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
