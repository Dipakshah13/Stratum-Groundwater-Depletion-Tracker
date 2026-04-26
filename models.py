from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False, default='User')
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)   # NULL for guests
    role          = db.Column(db.String(50),  default='Hydrologist')
    is_guest      = db.Column(db.Boolean,     default=False)

class WaterReading(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date          = db.Column(db.Date,    nullable=False)
    region        = db.Column(db.String(100), nullable=False)
    water_level   = db.Column(db.Float,   nullable=False)
    depletion_rate= db.Column(db.Float,   nullable=False)
    status        = db.Column(db.String(50),  nullable=False)
    lat           = db.Column(db.Float,   nullable=True)
    lng           = db.Column(db.Float,   nullable=True)

class MitigationLog(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date          = db.Column(db.String(20),  nullable=False)
    region        = db.Column(db.String(100), nullable=False)
    strategy      = db.Column(db.String(200), nullable=False)
    reduction_pct = db.Column(db.Float,       nullable=False)
    notes         = db.Column(db.Text,        nullable=True)
    logged_by     = db.Column(db.String(100), default='Admin')
