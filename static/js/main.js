// ── Stratum Live Intelligence & Alert Engine ────────────────────────────────
let _lastMtime       = null;
let _lastRecordCount = null;
let _initialLoadDone = false;

// ── Web-Audio Alarm System ───────────────────────────────────────────────────
let _alarmCtx       = null;
let _alarmLoopTimer = null;
let _alarmActive    = false;

function _getAudioCtx() {
    if (!_alarmCtx || _alarmCtx.state === 'closed') {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return null;
        _alarmCtx = new AC();
    }
    return _alarmCtx;
}

// Plays a professional "Sonar Pulse" alert
function _playOnePulse(ctx) {
    const now = ctx.currentTime;
    const masterGain = ctx.createGain();
    masterGain.gain.setValueAtTime(0.4, now);
    masterGain.connect(ctx.destination);

    // Filter for a cleaner, high-end "electronic" feel
    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(1200, now);
    filter.Q.setValueAtTime(1, now);
    filter.connect(masterGain);

    // Two layered oscillators for a rich, scientific sound
    [1200, 2400].forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const env = ctx.createGain();
        
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, now);
        
        // Sonar "Ping" envelope
        env.gain.setValueAtTime(0, now);
        env.gain.linearRampToValueAtTime(0.8 - (idx * 0.3), now + 0.05);
        env.gain.exponentialRampToValueAtTime(0.001, now + 1.2);
        
        osc.connect(env);
        env.connect(filter);
        osc.start(now);
        osc.stop(now + 1.5);
    });
}

function stopGlobalAlarm() {
    _alarmActive = false;
    if (_alarmLoopTimer) { clearTimeout(_alarmLoopTimer); _alarmLoopTimer = null; }
    
    // Update UI
    const indicator = document.getElementById('alarmIndicator');
    if (indicator) indicator.classList.add('d-none');
    
    const dot = document.getElementById('systemStatusDot');
    if (dot) {
        dot.style.background = 'var(--status-green)';
        dot.style.boxShadow = '0 0 0 3px rgba(5, 150, 105, 0.1)';
    }
}

function playGlobalAlarm() {
    const toggle = document.getElementById('alertSystemToggle');
    if (toggle && !toggle.checked) return; // Muted
    if (_alarmActive) return;
    
    _alarmActive = true;
    const indicator = document.getElementById('alarmIndicator');
    if (indicator) indicator.classList.remove('d-none');
    
    const dot = document.getElementById('systemStatusDot');
    if (dot) {
        dot.style.background = '#dc2626';
        dot.style.boxShadow = '0 0 0 3px rgba(220, 38, 36, 0.2)';
    }

    const ctx = _getAudioCtx();
    if (!ctx) return;

    function _loop() {
        if (!_alarmActive) return;
        const audioCtx = _getAudioCtx();
        if (!audioCtx) return;

        const doPlay = () => {
            _playOnePulse(audioCtx);
            _alarmLoopTimer = setTimeout(_loop, 3500); // 3.5s interval is less "spammy"
        };

        if (audioCtx.state === 'suspended') {
            audioCtx.resume().then(doPlay);
        } else {
            doPlay();
        }
    }
    _loop();
}

// ── Notification UI ──────────────────────────────────────────────────────────
function updateAlertSummary(alerts) {
    const summary = document.getElementById('activeAlertsSummary');
    if (!summary) return;

    if (!alerts || alerts.length === 0) {
        summary.innerHTML = 'Monitoring active. <span class="text-success fw-bold">System stable.</span>';
        summary.classList.remove('text-danger');
        stopGlobalAlarm();
        return;
    }

    summary.innerHTML = `<span class="text-danger fw-bold">${alerts.length} Critical Alert${alerts.length > 1 ? 's' : ''}</span> detected across ${[...new Set(alerts.map(a => a.region))].length} region(s). Action required.`;
    summary.classList.add('text-danger');
}

function showGlobalToast(alerts) {
    if (!alerts || alerts.length === 0) return;
    
    let container = document.getElementById('global-toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'global-toast-container';
        container.style.cssText = 'position:fixed;top:80px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:12px;';
        document.body.appendChild(container);
    }
    
    // Clear old toasts to prevent spam
    container.innerHTML = '';

    const toast = document.createElement('div');
    toast.className = 'animate-critical shadow-lg';
    toast.style.cssText = 'width:380px;margin:0;padding:20px;background:white;border-radius:16px;border-left:6px solid #dc2626;display:flex;flex-direction:column;gap:12px;border:1px solid rgba(220,38,36,0.1);';

    const header = `
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
            <div style="width:10px;height:10px;background:#dc2626;border-radius:50%;" class="spinner-grow spinner-grow-sm"></div>
            <div style="font-weight:800;color:#dc2626;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;">Critical System Alert</div>
        </div>`;

    const content = alerts.slice(0, 2).map(a => `
        <div style="font-size:0.85rem;line-height:1.4;color:#1e293b;padding-left:20px;position:relative;">
            <div style="position:absolute;left:0;top:6px;width:6px;height:6px;background:#e2e8f0;border-radius:50%;"></div>
            <strong>${a.region}</strong>: ${a.message.split('!')[1] || a.message}
        </div>
    `).join('');

    const more = alerts.length > 2 ? `<div style="font-size:0.75rem;color:#64748b;padding-left:20px;">+ ${alerts.length - 2} more regions...</div>` : '';

    const footer = `
        <div style="display:flex;gap:8px;margin-top:4px;">
            <button id="toastSilence" style="flex:1;padding:8px;font-size:0.75rem;font-weight:700;background:#dc2626;color:#fff;border:none;border-radius:8px;cursor:pointer;">Silence Alarm</button>
            <button id="toastView" style="flex:1;padding:8px;font-size:0.75rem;font-weight:600;background:#f1f5f9;color:#334155;border:none;border-radius:8px;cursor:pointer;">Review All</button>
        </div>`;

    toast.innerHTML = header + content + more + footer;
    container.appendChild(toast);

    document.getElementById('toastSilence').onclick = () => { stopGlobalAlarm(); toast.remove(); };
    document.getElementById('toastView').onclick = () => { window.location.href = '/alerts'; };

    setTimeout(() => { if (toast.parentNode) toast.remove(); }, 15000);
}

// ── Data Refresh Logic ───────────────────────────────────────────────────────
async function _refreshKPIs() {
    try {
        const response = await fetch(window.location.href);
        const html     = await response.text();
        const parser   = new DOMParser();
        const doc      = parser.parseFromString(html, 'text/html');

        ['.kpi-card', '#regionTable tbody'].forEach(sel => {
            const incoming = doc.querySelector(sel);
            const current  = document.querySelector(sel);
            if (incoming && current) current.innerHTML = incoming.innerHTML;
        });

        if (window.location.pathname.includes('/alerts')) {
            const inc = doc.querySelector('.chart-card');
            const cur = document.querySelector('.chart-card');
            if (inc && cur) cur.innerHTML = inc.innerHTML;
        }
    } catch (e) {}
}

async function checkUpdates() {
    try {
        const res  = await fetch('/api/check_update');
        const data = await res.json();
        const { last_modified: mtime, record_count: count, critical_alerts: alerts } = data;

        if (!_initialLoadDone) {
            _initialLoadDone = true;
            _lastMtime       = mtime;
            _lastRecordCount = count;
            updateAlertSummary(alerts);
            return;
        }

        const changed = count !== _lastRecordCount || mtime > _lastMtime;
        if (changed) {
            _lastMtime = mtime;
            _lastRecordCount = count;
            
            _refreshKPIs();
            if (typeof window.refreshCharts === 'function') window.refreshCharts();

            updateAlertSummary(alerts);
            if (alerts && alerts.length > 0) {
                const storedKey = localStorage.getItem('alertPlayedForCount');
                if (storedKey !== String(count)) {
                    playGlobalAlarm();
                    showGlobalToast(alerts);
                    localStorage.setItem('alertPlayedForCount', String(count));
                }
            }
        }
    } catch (e) { console.error('Poll failed:', e); }
}

// Initialise toggle state from storage
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('alertSystemToggle');
    if (toggle) {
        toggle.checked = localStorage.getItem('alertSystemEnabled') !== 'false';
        toggle.onchange = () => {
            localStorage.setItem('alertSystemEnabled', toggle.checked);
            if (!toggle.checked) stopGlobalAlarm();
        };
    }
    
    // Start polling
    setInterval(checkUpdates, 4000); // 4s is more professional for production
    checkUpdates();
});
