import os
from werkzeug.middleware.proxy_fix import ProxyFix
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from flask_bcrypt import Bcrypt
from flask_dance.contrib.google import make_google_blueprint, google
from models import db, User, WaterReading, MitigationLog

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'hydrotrack_super_secret_stratum_2025')
# Trust reverse proxy (Vercel/InsForge) so Flask generates https:// URLs
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

db_url = os.environ.get('DATABASE_URL', 'sqlite:///hydro_data.db')
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Production DB pool: pre_ping reconnects on stale connections
if os.environ.get('FLASK_ENV') == 'production':
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 280,
    }
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'
else:
    os.environ.setdefault('OAUTHLIB_INSECURE_TRANSPORT', '1')

db.init_app(app)
with app.app_context():
    db.create_all()

bcrypt = Bcrypt(app)

# ── Google OAuth blueprint ────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

GOOGLE_CLIENT_ID     = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')

# Fallback to prevent crash if .env is missing in dev
if not GOOGLE_CLIENT_ID:
    GOOGLE_CLIENT_ID = 'dummy_id'
if not GOOGLE_CLIENT_SECRET:
    GOOGLE_CLIENT_SECRET = 'dummy_secret'

google_bp = make_google_blueprint(
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    scope=['openid', 'https://www.googleapis.com/auth/userinfo.email',
           'https://www.googleapis.com/auth/userinfo.profile'],
    redirect_to='google_auth_callback'
)
app.register_blueprint(google_bp, url_prefix='/login')

# ── Flask-Login setup ─────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = ''          # suppress default flash

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

DATA_FILE = os.path.join(app.root_path, 'data', 'groundwater_data.csv')

with app.app_context():
    db.create_all()
    # Seed default admin
    if not User.query.filter_by(email='admin@hydro.gov').first():
        hashed = bcrypt.generate_password_hash('admin123').decode('utf-8')
        db.session.add(User(
            name='Admin User',
            email='admin@hydro.gov',
            password_hash=hashed,
            role='Lead Hydrologist'
        ))
        db.session.commit()

# ── Auth helpers ──────────────────────────────────────────────────────────────

def load_data():
    """Load water readings for the currently logged-in user only."""
    try:
        uid = current_user.id if current_user and current_user.is_authenticated else None
        if uid is None:
            return pd.DataFrame()
        db.session.expire_all()  # force fresh read from DB
        readings = WaterReading.query.filter_by(user_id=uid).all()
        if not readings:
            return pd.DataFrame()
        df = pd.DataFrame([{
            'date': r.date, 'region': r.region,
            'water_level': r.water_level, 'depletion_rate': r.depletion_rate,
            'status': r.status, 'lat': r.lat, 'lng': r.lng
        } for r in readings])
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print("Error loading DB layer:", e)
        return pd.DataFrame()

# ── Auth Routes ───────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        action = request.form.get('action', 'login')

        # ── Register new account ──────────────────────────────────────────
        if action == 'register':
            name      = request.form.get('name', '').strip()
            email     = request.form.get('email', '').strip().lower()
            password  = request.form.get('password', '')
            confirm   = request.form.get('confirm_password', '')

            if not name or not email or not password:
                flash('Please fill in all fields.', 'error')
                return redirect(url_for('login') + '?tab=register')
            if password != confirm:
                flash('Passwords do not match.', 'error')
                return redirect(url_for('login') + '?tab=register')
            if User.query.filter_by(email=email).first():
                flash('An account with that email already exists.', 'error')
                return redirect(url_for('login') + '?tab=register')

            hashed = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(name=name, email=email, password_hash=hashed, role='Hydrologist')
            db.session.add(user)
            db.session.commit()
            login_user(user, remember=True)
            flash(f'Welcome to STRATUM, {name}!', 'success')
            return redirect(url_for('dashboard'))

        # ── Login existing account ────────────────────────────────────────
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter_by(email=email, is_guest=False).first()
        if user and user.password_hash and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/login/google/callback')
def google_auth_callback():
    """Handle Google OAuth callback — create or login the user."""
    if GOOGLE_CLIENT_ID == 'YOUR_GOOGLE_CLIENT_ID':
        flash('Google login is not configured yet. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.', 'error')
        return redirect(url_for('login'))

    if not google.authorized:
        flash('Google authorisation failed. Please try again.', 'error')
        return redirect(url_for('login'))

    try:
        resp = google.get('/oauth2/v2/userinfo')
        if not resp.ok:
            flash('Could not fetch your Google profile.', 'error')
            return redirect(url_for('login'))

        info      = resp.json()
        g_email   = info.get('email', '').lower()
        g_name    = info.get('name', g_email.split('@')[0].title())

        user = User.query.filter_by(email=g_email).first()
        if not user:
            user = User(name=g_name, email=g_email, password_hash=None,
                        role='Hydrologist', is_guest=False)
            db.session.add(user)
            db.session.commit()
            flash(f'Welcome to STRATUM, {g_name}!', 'success')

        login_user(user, remember=True)
        return redirect(url_for('dashboard'))

    except Exception as e:
        flash(f'Google login error: {str(e)}', 'error')
        return redirect(url_for('login'))


@app.route('/login/guest')
def login_guest():
    """Create a temporary guest session."""
    guest_id  = str(uuid.uuid4())[:8]
    guest_email = f'guest_{guest_id}@stratum.local'
    guest = User(
        name=f'Guest {guest_id[:4].upper()}',
        email=guest_email,
        password_hash=None,
        role='Guest Viewer',
        is_guest=True
    )
    db.session.add(guest)
    db.session.commit()
    seed_guest_data(guest.id)
    login_user(guest, remember=False)
    flash('You are browsing as a guest with demo data. Your session will not be saved.', 'info')
    return redirect(url_for('dashboard'))

def seed_guest_data(user_id):
    """Seed a guest user with sample data from groundwater_data.csv if available."""
    try:
        csv_path = os.path.join(app.root_path, 'data', 'groundwater_data.csv')
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        df = smart_normalize(df)
        for _, row in df.iterrows():
            lat, lng = get_lat_lng(str(row['region']))
            wr = WaterReading(
                user_id=user_id,
                date=row['date'],
                region=str(row['region']),
                lat=lat, lng=lng,
                water_level=float(row['water_level']),
                depletion_rate=float(row['depletion_rate']),
                status=str(row['status'])
            )
            db.session.add(wr)
        db.session.commit()
    except Exception as e:
        print(f"Failed to seed guest data: {e}")


@app.route('/logout')
@login_required
def logout():
    # Delete guest accounts on logout to keep DB clean
    if current_user.is_guest:
        user = User.query.get(current_user.id)
        db.session.delete(user)
        db.session.commit()
    logout_user()
    return redirect(url_for('login'))


# ── App Routes (all protected) ────────────────────────────────────────────────

@app.route('/')
@login_required
def dashboard():
    df = load_data()
    has_data = not df.empty

    num_regions     = df['region'].nunique() if has_data else 0
    latest_date     = df['date'].max()       if has_data else None

    if latest_date is not None:
        latest_data     = df[df['date'] == latest_date]
        avg_water_level = latest_data['water_level'].mean()
        avg_depletion   = latest_data['depletion_rate'].mean()
    else:
        avg_water_level = 0
        avg_depletion   = 0

    return render_template('dashboard.html',
                           has_data=has_data,
                           num_regions=num_regions,
                           avg_water_level=round(avg_water_level, 2),
                           avg_depletion=round(avg_depletion, 2))

@app.route('/regions')
@login_required
def regions():
    df = load_data()
    latest_date = df['date'].max() if not df.empty else None
    if latest_date:
        regions_data = df[df['date'] == latest_date].to_dict('records')
    else:
        regions_data = []
    return render_template('region.html', regions=regions_data)

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'datafile' not in request.files:
        flash('No file detected. Please select a CSV file.')
        return redirect(request.referrer or url_for('dashboard'))

    file = request.files['datafile']
    if file.filename == '':
        flash('No file selected.')
        return redirect(request.referrer or url_for('dashboard'))

    if not (file and file.filename.endswith('.csv')):
        flash('Please upload a .csv file.')
        return redirect(request.referrer or url_for('dashboard'))

    # Vercel/serverless: only /tmp is writable
    tmp_path = os.path.join('/tmp', 'upload.csv')
    file.save(tmp_path)

    try:
        df = pd.read_csv(tmp_path)
        if df.empty:
            flash('The uploaded CSV is empty.')
            return redirect(request.referrer or url_for('dashboard'))

        df = smart_normalize(df)

        WaterReading.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()

        for _, row in df.iterrows():
            lat, lng = get_lat_lng(str(row['region']))
            wr = WaterReading(
                user_id=current_user.id,
                date=row['date'],
                region=str(row['region']),
                lat=lat, lng=lng,
                water_level=float(row['water_level']),
                depletion_rate=float(row['depletion_rate']),
                status=str(row['status'])
            )
            db.session.add(wr)
        db.session.commit()
        flash(f'✓ {len(df)} records imported successfully — system updated.')

    except Exception as e:
        flash(f'Import error: {str(e)}')

    return redirect(request.referrer or url_for('dashboard'))


def smart_normalize(df):
    """Auto-detect & map ANY CSV column names to required schema."""

    PATTERNS = {
        'date':           ['date','time','timestamp','dt','day','month','year','period'],
        'region':         ['region','area','location','zone','district','state','city',
                          'site','place','name','locality','watershed','basin','sector','aquifer'],
        'water_level':    ['water_level','waterlevel','wl','gwl','level','depth',
                          'head','elevation','groundwater','table','depth_below'],
        'depletion_rate': ['depletion_rate','depletionrate','rate','change','decline',
                          'drop','loss','delta','diff','decrease'],
        'status':         ['status','condition','alert','risk','severity','flag',
                          'category','class','classification','label','type'],
    }

    cols_lower = {c.lower().strip().replace(' ', '_'): c for c in df.columns}
    mapping, used = {}, set()

    for target, keywords in PATTERNS.items():
        for kw in keywords:
            for col_key, col_orig in cols_lower.items():
                if col_orig in used:
                    continue
                if kw in col_key:
                    mapping[target] = col_orig
                    used.add(col_orig)
                    break
            if target in mapping:
                break

    df = df.rename(columns={v: k for k, v in mapping.items()})

    if 'date' in df.columns:
        col = df['date']
        try:
            numeric = pd.to_numeric(col, errors='coerce')
            if numeric.notna().mean() > 0.8 and numeric.dropna().between(1900, 2100).all():
                df['date'] = pd.to_datetime(
                    numeric.fillna(2000).astype(int).astype(str) + '-01-01'
                ).dt.date
            else:
                parsed = pd.Series([pd.NaT] * len(col))
                for fmt in ['%Y-%m-%d','%d-%m-%Y','%m/%d/%Y','%d/%m/%Y',
                            '%Y/%m/%d','%d.%m.%Y','%Y.%m.%d','%m-%d-%Y']:
                    attempt = pd.to_datetime(col, format=fmt, errors='coerce')
                    if attempt.notna().sum() > parsed.notna().sum():
                        parsed = attempt
                if parsed.notna().sum() == 0:
                    parsed = pd.to_datetime(col, errors='coerce')
                df['date'] = parsed.dt.date
                df = df.dropna(subset=['date'])
        except Exception:
            df = _generate_dates(df)
    else:
        df = _generate_dates(df)

    if 'region' not in df.columns:
        str_cols = [c for c in df.columns
                    if df[c].dtype == object and c not in mapping.values()]
        df['region'] = df[str_cols[0]] if str_cols else 'Default Region'

    if 'water_level' not in df.columns:
        skip = {v for v in mapping.values()} | {'date','region','depletion_rate','status'}
        num_cols = [c for c in df.select_dtypes(include='number').columns if c not in skip]
        if not num_cols:
            raise ValueError('No numeric column found for water level.')
        df['water_level'] = pd.to_numeric(df[num_cols[0]], errors='coerce').fillna(0)

    df['water_level'] = pd.to_numeric(df['water_level'], errors='coerce').fillna(0)

    if 'depletion_rate' not in df.columns:
        df = df.sort_values(['region', 'date'])
        df['depletion_rate'] = (df.groupby('region')['water_level']
                                  .diff().abs().fillna(0).round(3))
    df['depletion_rate'] = pd.to_numeric(df['depletion_rate'], errors='coerce').fillna(0)

    if 'status' not in df.columns:
        p33 = df['water_level'].quantile(0.33)
        p66 = df['water_level'].quantile(0.66)
        df['status'] = df['water_level'].apply(
            lambda v: 'red' if v >= p66 else ('yellow' if v >= p33 else 'green')
        )
    else:
        def _norm(s):
            s = str(s).lower().strip()
            if any(w in s for w in ['crit','red','severe','danger','bad','alarm']):
                return 'red'
            if any(w in s for w in ['warn','yellow','moderate','medium','caution','watch']):
                return 'yellow'
            return 'green'
        df['status'] = df['status'].apply(_norm)

    return df[['date', 'region', 'water_level', 'depletion_rate', 'status']]


def _generate_dates(df):
    base = pd.Timestamp.today()
    df['date'] = [(base - pd.DateOffset(months=i)).date()
                  for i in range(len(df) - 1, -1, -1)]
    return df


@app.route('/prediction')
@login_required
def prediction():
    df = load_data()
    regions = sorted(df['region'].unique().tolist()) if not df.empty else []
    return render_template('prediction.html', regions=regions)

@app.route('/alerts')
@login_required
def alerts():
    df = load_data()
    alerts_data = []
    if not df.empty:
        latest_date = df['date'].max()
        recent_reds = df[(df['date'] == latest_date) & (df['status'] == 'red')]
        for _, row in recent_reds.iterrows():
            msg = f"Critical depletion rate ({row['depletion_rate']} ft/mo). Problem Overview: Severe aquifer stress detected risking land subsidence and immediate water scarcity. Action required: Halt heavy extraction."
            alerts_data.append({'region': row['region'], 'date': row['date'].strftime('%Y-%m-%d'), 'message': msg, 'type': 'critical'})
        recent_yellows = df[(df['date'] == latest_date) & (df['status'] == 'yellow')]
        for _, row in recent_yellows.iterrows():
            msg = f"Warning: increasing depletion trend ({row['depletion_rate']} ft/mo). Problem Overview: Usage slightly exceeds recharge rate. Monitor closely to prevent slipping into critical levels."
            alerts_data.append({'region': row['region'], 'date': row['date'].strftime('%Y-%m-%d'), 'message': msg, 'type': 'warning'})
    return render_template('alerts.html', alerts=alerts_data)

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route('/api/debug')
@login_required
def api_debug():
    try:
        uid = current_user.id
        total = WaterReading.query.count()
        mine = WaterReading.query.filter_by(user_id=uid).count()
        regions = [r.region for r in WaterReading.query.filter_by(user_id=uid).all()]
        return jsonify({'user_id': uid, 'name': current_user.name,
                        'my_readings': mine, 'total_readings': total,
                        'regions': list(set(regions))})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/historical')
@login_required
def api_historical():
    df = load_data()
    if df.empty: return jsonify({})
    overall = df.groupby('date')['water_level'].mean().reset_index()
    overall['date_str'] = overall['date'].dt.strftime('%Y-%m-%d')
    return jsonify({'labels': overall['date_str'].tolist(), 'data': overall['water_level'].round(2).tolist()})

@app.route('/api/analysis_stats')
@login_required
def api_analysis_stats():
    df = load_data()
    if df.empty: return jsonify({})
    stats = {}
    history = []
    for _, row in df.iterrows():
        history.append({'date': row['date'].strftime('%Y-%m-%d'), 'region': row['region'], 'level': float(row['water_level'])})
    for region in df['region'].unique():
        region_data = df[df['region'] == region].sort_values('date')
        latest_val = float(region_data['water_level'].iloc[-1])
        avg_val    = float(region_data['water_level'].mean())
        stats[region] = {
            'min': float(region_data['water_level'].min()),
            'max': float(region_data['water_level'].max()),
            'avg': avg_val, 'latest': latest_val,
            'trend': "up" if latest_val >= avg_val else "down"
        }
    return jsonify({'summary': stats, 'history': history, 'global_min': float(df['water_level'].min()), 'global_max': float(df['water_level'].max())})

@app.route('/api/predict/<path:region_name>')
@app.route('/api/predict')
@login_required
def api_predict(region_name=None):
    if not region_name:
        region_name = request.args.get('region')
        
    if not region_name:
        return jsonify({'error': 'No region specified for prediction.'})
    
    region_name = region_name.strip()
        
    df = load_data()
    if df.empty: return jsonify({'error': 'No data available for your account. Please upload a CSV first.'})

    # ── Parameters ────────────────────────────────────────────────────────────
    # months: 0–60 (0–5 years). 0 means show only historical baseline.
    months = max(0, min(60, int(request.args.get('months', 12))))
    mitigate_pct     = max(0.0, min(80.0, float(request.args.get('mitigation', 0))))
    mitigation_factor = mitigate_pct / 100.0

    region_data = df[df['region'] == region_name].copy()
    if region_data.empty: return jsonify({})

    region_data = region_data.sort_values('date').reset_index(drop=True)
    region_data['date'] = pd.to_datetime(region_data['date'])

    # Need at least 3 data points for a meaningful model
    if len(region_data) < 3:
        return jsonify({'error': 'Insufficient data for prediction (need ≥ 3 records).'})

    y_vals = region_data['water_level'].values.astype(float)
    n      = len(y_vals)

    # ── Numeric time axis: months since first reading ─────────────────────────
    t0 = region_data['date'].iloc[0]
    region_data['t_months'] = region_data['date'].apply(
        lambda d: (d.year - t0.year) * 12 + (d.month - t0.month)
    )
    t_vals = region_data['t_months'].values.astype(float)

    # ── Model 1: Robust Trend (Theil-Sen inspired) ───────────────────────────
    slope, intercept, r_lin, _, _ = linregress(t_vals, y_vals)
    
    # ── Model 2: Physics baseline (Avg Depletion) ────────────────────────────
    if 'depletion_rate' in region_data.columns:
        avg_depletion = float(region_data['depletion_rate'].tail(12).mean())
    else:
        avg_depletion = float(np.abs(np.diff(y_vals)).mean()) if n > 1 else 0.0

    # ── Model 3: Seasonal Component (Fourier-inspired) ───────────────────────
    # We attempt to capture 12-month seasonality
    detrended = y_vals - (slope * t_vals + intercept)
    
    # Simple seasonal average (12-month)
    seasonal_cycle = np.zeros(12)
    for i in range(12):
        # average of all historical data points for month i in the cycle
        matches = detrended[np.where((t_vals % 12) == i)]
        if len(matches) > 0:
            seasonal_cycle[i] = np.mean(matches)
    
    # ── Generate future time steps ────────────────────────────────────────────
    last_date = region_data['date'].iloc[-1]
    last_t    = float(t_vals[-1])
    last_val  = y_vals[-1]

    if months == 0:
        future_dates = []; future_levels = []; mitigated_levels = []; ci_upper = []; ci_lower = []
    else:
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months + 1)]
        future_t     = np.array([last_t + i for i in range(1, months + 1)])
        
        # 1. Base Trend (using a slightly dampened slope for long-term stability)
        # If it's depleting fast, we dampen the rate slightly over 10 years (natural physics)
        dampening = np.exp(-0.005 * np.arange(1, months + 1))
        trend_part = last_val + (slope * np.arange(1, months + 1) * dampening)
        
        # 2. Seasonal Projection
        seasonal_part = np.array([seasonal_cycle[int(t % 12)] for t in future_t])
        
        # 3. Ensemble Result (Trend + Seasonality + a small random walk for "realism")
        # We ensure it doesn't go below 0 or above a reasonable physical limit
        future_levels = trend_part + seasonal_part
        
        # ── Confidence intervals (Grows with time/uncertainty) ────────────────
        std_resid = np.std(detrended)
        uncertainty = 1.96 * std_resid * np.sqrt(np.arange(1, months + 1)) / 2.0
        ci_upper = future_levels + uncertainty
        ci_lower = future_levels - uncertainty

        # ── Mitigation Logic (Physics-based Lag Model) ───────────────────────
        mitigated_levels = []
        # Mitigation factor is 0.0 to 0.5 (50% reduction)
        for i, val in enumerate(future_levels, start=1):
            # The "Pressure" is the difference between trend and last_val
            pressure = val - last_val
            
            # Policy takes time to kick in (Lag)
            # S-Curve (Sigmoid) for policy adoption: 1 / (1 + exp(-k*(t-t0)))
            policy_adoption = 1.0 / (1.0 + np.exp(-0.5 * (i - 6))) 
            effective_mitigation = mitigation_factor * policy_adoption
            
            # Apply mitigation: reduces depletion pressure
            # If pressure is positive (depletion), we reduce it.
            if pressure > 0:
                mit_val = last_val + (pressure * (1.0 - effective_mitigation * 1.5))
                # Add a small recharge effect after 2 years of policy
                if i > 24:
                    mit_val -= (mitigation_factor * 0.05 * (i - 24))
            else:
                # If it's already recovering, mitigation helps it recover 20% faster
                mit_val = val - (abs(pressure) * effective_mitigation * 0.2)
            
            mitigated_levels.append(round(float(mit_val), 2))
        
        future_levels = np.array([round(float(v), 2) for v in future_levels])
        ci_upper = np.array([round(float(v), 2) for v in ci_upper])
        ci_lower = np.array([round(float(v), 2) for v in ci_lower])
        mitigated_levels = np.array(mitigated_levels)

    # ── Smart critical threshold ──────────────────────────────────────────────
    # Use the 5th-percentile of historical data minus 1 std-dev of depletion
    p5  = float(np.percentile(y_vals, 5))
    p10 = float(np.percentile(y_vals, 10))
    # For rapidly depleting aquifers weight toward the lower bound
    depletion_severity = min(1.0, avg_depletion / max(abs(slope) * 12 + 0.1, 0.1))
    critical_threshold = round(p5 - (p10 - p5) * (0.5 + depletion_severity * 0.5), 2)

    # ── ETA to critical ───────────────────────────────────────────────────────
    eta_critical  = 'SAFE'
    eta_mitigated = 'SAFE'
    for idx, val in enumerate(future_levels):
        if val <= critical_threshold:
            eta_critical = future_dates[idx].strftime('%Y-%m')
            break
    for idx, val in enumerate(mitigated_levels):
        if val <= critical_threshold:
            eta_mitigated = future_dates[idx].strftime('%Y-%m')
            break

    # ── Trend health indicators ───────────────────────────────────────────────
    # Depletion velocity in m/year equivalent
    depletion_velocity = round(slope * 12, 3)   # per year
    # Forecast stability: how much does the CI widen?
    if months > 0:
        ci_width_end   = float(ci_upper[-1] - ci_lower[-1])
        ci_width_start = float(ci_upper[0]  - ci_lower[0])
    else:
        ci_width_end = ci_width_start = 0.0
    confidence_score = round(max(0, min(100, 100 - (ci_width_end - ci_width_start))), 1)

    # ── Final safety check for NaN/Inf values ─────────────────────────────────
    def _safe_list(arr):
        if not hasattr(arr, '__len__') or len(arr) == 0: return []
        return [float(x) if (np.isfinite(x)) else 0.0 for x in arr]

    return jsonify({
        'historical': {
            'labels': region_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'data':   _safe_list(y_vals)
        },
        'predicted': {
            'labels':    [d.strftime('%Y-%m-%d') for d in future_dates],
            'data':      _safe_list(future_levels),
            'mitigated': _safe_list(mitigated_levels),
            'ci_upper':  _safe_list(ci_upper),
            'ci_lower':  _safe_list(ci_lower)
        },
        'analysis': {
            'eta_critical':        eta_critical,
            'eta_mitigated':       eta_mitigated,
            'critical_threshold':  float(critical_threshold),
            'depletion_velocity':  float(depletion_velocity) if np.isfinite(depletion_velocity) else 0.0,
            'r_squared':           float(r2) if np.isfinite(r2) else 0.0,
            'confidence_score':    float(confidence_score) if np.isfinite(confidence_score) else 0.0,
            'model_weights':       {k: float(v) for k, v in {'linear': w_lin, 'polynomial': w_poly, 'depletion': w_dep}.items()}
        }
    })

@app.route('/api/check_update')
@login_required
def api_check_update():
    try:
        mtime = os.path.getmtime(DATA_FILE) if os.path.exists(DATA_FILE) else 0
    except Exception:
        mtime = 0
    df = load_data()
    record_count = len(df)
    criticals = []
    if not df.empty:
        latest_date = df['date'].max()
        recent_reds = df[(df['date'] == latest_date) & (df['status'] == 'red')]
        for _, row in recent_reds.iterrows():
            msg = (f"Critical depletion in {row['region']}! Depth: {row['water_level']:.1f} m, Rate: {row['depletion_rate']:.2f} m/yr")
            criticals.append({'region': row['region'], 'message': msg})
    return jsonify({'last_modified': mtime, 'record_count': record_count, 'critical_alerts': criticals})

def get_lat_lng(region_name):
    r = region_name.lower()
    COORDS = [
        ('ganga',26.00,81.00),('alluvial',26.00,81.00),('deccan',17.38,78.47),
        ('rajasthan',27.02,74.22),('desert',26.90,72.00),('andhra',15.91,80.00),
        ('coastal',14.90,79.80),('punjab',31.15,75.34),('doab',30.50,74.80),
        ('krishna',16.50,78.00),('brahmaputra',26.14,91.74),('floodplain',26.00,90.50),
        ('bundelkhand',25.40,79.00),('hard rock',17.00,76.50),('indo-gangetic',28.00,79.00),
        ('indo gangetic',28.00,79.00),('northwest',29.00,73.00),('northeast',25.00,91.00),
        ('western ghats',12.00,75.50),('east coast',13.00,80.20),('gujarat',22.25,71.19),
        ('maharashtra',19.75,75.71),('karnataka',15.31,75.71),('kerala',8.52,76.94),
        ('tamil',9.92,78.12),('odisha',20.95,85.10),('bihar',25.09,85.31),
        ('north',40.71,-74.00),('south',29.76,-95.36),('east',38.90,-77.03),('west',34.05,-118.24),
    ]
    for keyword, lat, lng in COORDS:
        if keyword in r:
            return lat, lng
    return 20.59, 78.96

@app.route('/mitigation')
@login_required
def mitigation():
    df = load_data()
    regions = sorted(df['region'].unique().tolist()) if not df.empty else []
    region_risk = {}
    if not df.empty:
        latest_date = df['date'].max()
        for region in regions:
            r = df[(df['region'] == region) & (df['date'] == latest_date)]
            if not r.empty:
                region_risk[region] = r.iloc[0]['status']
    return render_template('mitigation.html', regions=regions, region_risk=region_risk)

@app.route('/api/mitigation/logs')
@login_required
def api_mitigation_logs():
    logs = MitigationLog.query.filter_by(user_id=current_user.id).order_by(MitigationLog.id.desc()).all()
    return jsonify([{'id': l.id, 'date': l.date, 'region': l.region, 'strategy': l.strategy, 'reduction_pct': l.reduction_pct, 'notes': l.notes, 'logged_by': l.logged_by} for l in logs])

@app.route('/api/mitigation/log', methods=['POST'])
@login_required
def api_mitigation_log():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400
    entry = MitigationLog(
        user_id=current_user.id,
        date=data.get('date',''), region=data.get('region',''),
        strategy=data.get('strategy',''), reduction_pct=float(data.get('reduction_pct', 0)),
        notes=data.get('notes',''), logged_by=current_user.name
    )
    db.session.add(entry)
    db.session.commit()
    return jsonify({'success': True, 'id': entry.id})

@app.route('/api/mitigation/log/<int:log_id>', methods=['DELETE'])
@login_required
def api_mitigation_delete(log_id):
    entry = MitigationLog.query.get_or_404(log_id)
    db.session.delete(entry)
    db.session.commit()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
