# STRATUM — Groundwater Depletion Tracking & Predictive Analytics Platform
### Complete Application Overview

---

## 🌐 What is STRATUM?

**STRATUM** is an advanced, data-driven web application built for **hydrologists, environmental scientists, and water policy makers** to:
- Monitor aquifer health across multiple geographic regions
- Visualize historical groundwater trends with precision
- Predict future depletion using a hybrid Ensemble ML + GenAI engine
- Simulate the impact of conservation strategies in real-time
- Log and track real-world intervention actions with full audit trails
- Generate professional PDF reports for regulatory submissions

> STRATUM transforms raw groundwater data into a live, AI-augmented decision-support command center.

---

## 🏗️ Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│                         BROWSER (Client)                      │
│   HTML5 Templates + Vanilla JS + Chart.js + Bootstrap 5      │
│   Markdown-to-HTML + html2pdf Reporting Engine               │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP / REST API
┌──────────────────────▼───────────────────────────────────────┐
│                    FLASK BACKEND (Python)                      │
│   app.py — 850+ lines, 25+ routes, ML + AI logic             │
│   Ensemble Regression (Scipy/Sklearn)                        │
│   GenAI Intelligence (Groq + Llama 3.3 70B)                  │
└──────────────────────┬───────────────────────────────────────┘
                       │ SQLAlchemy ORM
┌──────────────────────▼───────────────────────────────────────┐
│              DATABASE LAYER                                   │
│   SQLite (development) │ PostgreSQL (production)             │
│   Tables: User │ WaterReading │ MitigationLog                │
└──────────────────────────────────────────────────────────────┘
```

### Data Model (3 tables)

| Table | Key Fields | Purpose |
|---|---|---|
| `User` | id, name, email, role, is_guest | Auth & multi-user isolation |
| `WaterReading` | date, region, water_level, depletion_rate, status, lat, lng | Historical aquifer data |
| `MitigationLog` | region, strategy, reduction_pct, date, notes, logged_by | Intervention audit trail |

---

## 🔐 Security & Authentication
- **Multi-Pathway Login**: Email/Password, Google OAuth 2.0 (SSO), and Guest Demo Mode.
- **Session Hardening**: Persistent sessions with `remember me`, `SESSION_COOKIE_SECURE` in production, and aggressive `session.clear()` on logout.
- **Cache Control**: Global `no-store` headers to prevent stale data display after logout.
- **DB Verification**: Every login attempt verifies user existence against the DB to prevent stale session loops.

---

## 📋 Application Modules (6 Pages)

---

### 1. 📊 Dashboard (`/`)
*The mission control home screen.*

**KPI Summary Cards (3)**
- **Average Water Level** — System-wide average depth (ft)
- **Average Depletion Rate** — Rate of loss (ft/month)
- **Monitored Regions** — Count of active sensors/zones

**System-Wide Groundwater Trajectory Chart**
- Powered by Chart.js (line chart with gradient fill)
- Interactive tooltips showing exact levels per year.

**PDF Reporting**
- One-click "Export Report" button generates high-fidelity PDF summaries of the current view.

---

### 2. 📥 Data Import — Smart CSV Uploader
*Universal ingestion for any hydrological dataset.*

**Smart Column Normalization Engine:**
- Auto-detects 40+ column name variants across 5 fields (date, region, level, etc.)
- **Auto-generation**: Computes depletion rates and status categories (Red/Yellow/Green) if missing.
- **Geospatial Mapping**: Automatic lat/lng lookup for 30+ major hydrological regions.

---

### 3. 🔮 Predictive Forecasting (`/prediction`)
*AI-powered aquifer trajectory simulation with professional hydrological reporting.*

#### Scenario Builder & Prediction Engine:
- **Horizon Slider**: Forecast up to 10 years into the future.
- **Ensemble Model**: Combines Linear Trend, Seasonal Fourier cycles, and Physics-based depletion.
- **Confidence Intervals**: 95% shaded probability band grows with time horizon.

#### 🤖 AI Hydrological Analysis (Groq Llama 3.3):
- Every simulation triggers a Generative AI hydrological report.
- Interprets data trends, explains R² accuracy, and provides region-specific risk assessments.
- **Formatted Insights**: Scannable markdown-to-HTML output with bullet points and bolded key metrics.

#### Model Intelligence Grid (6 metrics):
| Metric | Description |
|---|---|
| Projected End Level | Final forecasted water level |
| Intervention Gain | ft saved by the selected policy |
| Critical Threshold | Dynamic floor based on 5th percentile |
| Depletion Velocity | Linear trend rate (ft/year) |
| Model Accuracy (R²) | Reliability score of the statistical fit |
| Forecast Confidence | Stability score of the CI band |

---

### 4. 📈 Analysis (`/analysis`)
*Deep statistical breakdown and multi-dimensional comparison.*

- **Geographic Heatmap**: Visualizes depletion across time and regions.
- **Radar Analysis**: Compares health relative to basement levels across the entire system.
- **Live Data Grid**: Sortable, searchable historical logs with status indicators.

---

### 5. 🛡️ Mitigation Planner (`/mitigation`)
*Strategy selection, impact preview, and AI-driven evaluation.*

- **Strategy Library**: 7 pre-built interventions (Drip Irrigation, Aquifer Recharge, etc.).
- **Live Impact Overlay**: Instantly view how a strategy "bends the curve" on the forecast.
- **🤖 AI Strategy Evaluation**: Groq AI evaluates the feasibility of your selected strategy for the target region.
- **Intervention Audit Log**: Persistent history of all saved conservation actions.

---

### 6. 🚨 Alerts (`/alerts`)
*Automated critical and warning notifications.*

- **Critical Alerts**: Triggered for `status = red` with recommended halt in extraction.
- **Visual Feedback**: Pulsing red indicators and scannable impact summaries.
- **Real-time Polling**: Background updates via `/api/check_update`.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | GET | Ensemble forecast + AI hydrological insight |
| `/api/ai_analysis` | POST | Triggers Groq Llama 3.3 analysis for reports |
| `/api/mitigation/log`| POST | Save a conservation intervention |
| `/api/historical` | GET | Global trajectory data |
| `/api/check_update` | GET | Real-time system monitoring |

---

## 🛠️ Technology Stack
- **Backend**: Flask 3.0.3, Python 3.12
- **Intelligence**: **Groq API (Llama 3.3 70B Versatile)**
- **Science**: Pandas, NumPy, Scikit-Learn, SciPy
- **Frontend**: Vanilla JS, Chart.js, Bootstrap 5, Markdown rendering
- **Reporting**: html2pdf.js, PDF generation engine
- **Auth**: Flask-Login, Flask-Bcrypt, Google OAuth 2.0
- **Database**: SQLite / PostgreSQL (SQLAlchemy)

---

## ✨ What Makes STRATUM Stand Out
1. **Hybrid Intelligence**: Combines rigid statistical regression with flexible Generative AI analysis.
2. **Real-World Physics**: Mitigation models include a sigmoid lag, simulating realistic policy adoption times.
3. **Zero-Configuration Ingestion**: Works with any CSV data from any source globally.
4. **Professional Reporting**: Integrated PDF generation and scannable AI reports for executive review.
5. **Multi-User Isolation**: Enterprise-grade auth ensures data privacy for individual hydrologists.
