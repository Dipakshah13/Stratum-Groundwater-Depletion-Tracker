// ── HydroTrack Live-Update Engine ───────────────────────────────────────────
let _lastMtime       = null;
let _lastRecordCount = null;
let _initialLoadDone = false;

// ── Web-Audio alarm ──────────────────────────────────────────────────────────
// Persistent AudioContext and loop state
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

// Plays one 2-second burst: a professional, non-irritating double chime
function _playOneBurst(ctx) {
    const now = ctx.currentTime;

    // Master gain — audible but not harsh (0.6)
    const masterGain = ctx.createGain();
    masterGain.gain.setValueAtTime(0.6, now);
    masterGain.connect(ctx.destination);

    // Compressor to keep the sound smooth
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.setValueAtTime(-12, now);
    comp.knee.setValueAtTime(40, now);
    comp.ratio.setValueAtTime(12, now);
    comp.attack.setValueAtTime(0, now);
    comp.release.setValueAtTime(0.25, now);
    comp.connect(masterGain);

    // Two soft chimes: a gentle "ding-ding" (minor third interval for an alert feel)
    // Chime 1: 0.0s (784 Hz - G5)  |  Chime 2: 0.3s (659 Hz - E5)
    [[0.0, 784], [0.3, 659]].forEach(([start, freq]) => {
        const osc  = ctx.createOscillator();
        const gEnv = ctx.createGain();

        // Sine wave is much softer on the ears than square/sawtooth
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, now + start);

        // Gentle envelope: smooth attack, long natural decay
        gEnv.gain.setValueAtTime(0, now + start);
        gEnv.gain.linearRampToValueAtTime(1.0, now + start + 0.03);  // 30ms soft attack
        gEnv.gain.exponentialRampToValueAtTime(0.01, now + start + 1.0); // 1-second long fade out
        
        osc.connect(gEnv);
        gEnv.connect(comp);
        
        osc.start(now + start);
        osc.stop(now + start + 1.5); // Ensure oscillator stops after decay
    });
}

function stopGlobalAlarm() {
    _alarmActive = false;
    if (_alarmLoopTimer) { clearTimeout(_alarmLoopTimer); _alarmLoopTimer = null; }
    if (_alarmCtx) {
        try { _alarmCtx.close(); } catch(e) {}
        _alarmCtx = null;
    }
}

function playGlobalAlarm() {
    if (_alarmActive) return;   // already running
    _alarmActive = true;

    const ctx = _getAudioCtx();
    if (!ctx) return;

    function _loop() {
        if (!_alarmActive) return;
        const audioCtx = _getAudioCtx();
        if (!audioCtx) return;

        const doPlay = () => {
            _playOneBurst(audioCtx);
            // Schedule next burst in exactly 2 seconds (seamless loop)
            _alarmLoopTimer = setTimeout(_loop, 2000);
        };

        if (audioCtx.state === 'suspended') {
            audioCtx.resume().then(doPlay);
        } else {
            doPlay();
        }
    }

    _loop();
}

// ── Toast notification ───────────────────────────────────────────────────────
function showGlobalToast(message) {
    let container = document.getElementById('global-toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'global-toast-container';
        container.style.cssText = 'position:fixed;top:80px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:12px;';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = 'alert-item critical animate-critical shadow-lg';
    toast.style.cssText = 'width:400px;margin:0;padding:18px 20px;background:white;border-radius:12px;border-left:6px solid #dc2626;display:flex;flex-direction:column;gap:10px;';

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:flex-start;gap:12px;';
    row.innerHTML = `
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2" style="flex-shrink:0;margin-top:2px"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        <div style="flex:1;">
            <div style="font-weight:700;color:#dc2626;font-size:0.82rem;letter-spacing:0.04em;margin-bottom:3px;">⚠ CRITICAL ALERT</div>
            <div style="font-size:0.84rem;line-height:1.45;color:#1e293b;">${message}</div>
        </div>`;

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:8px;';
    const stopBtn = document.createElement('button');
    stopBtn.textContent = '⏹ Stop Alarm';
    stopBtn.style.cssText = 'flex:1;padding:6px 0;font-size:0.78rem;font-weight:700;background:#dc2626;color:#fff;border:none;border-radius:7px;cursor:pointer;';
    const viewBtn = document.createElement('button');
    viewBtn.textContent = 'View Alerts →';
    viewBtn.style.cssText = 'flex:1;padding:6px 0;font-size:0.78rem;font-weight:600;background:#f1f5f9;color:#334155;border:none;border-radius:7px;cursor:pointer;';

    stopBtn.addEventListener('click', () => { stopGlobalAlarm(); toast.remove(); });
    viewBtn.addEventListener('click', () => { window.location.href = '/alerts'; });
    btnRow.appendChild(stopBtn);
    btnRow.appendChild(viewBtn);

    toast.appendChild(row);
    toast.appendChild(btnRow);
    container.appendChild(toast);

    // Auto-dismiss after 12 s; stop alarm when dismissed
    setTimeout(() => { if (toast.parentNode) { toast.remove(); stopGlobalAlarm(); } }, 12000);
}

// ── Safely call refreshCharts ────────────────────────────────────────────────
function _safeRefresh() {
    if (typeof window.refreshCharts === 'function') {
        try { window.refreshCharts(); } catch (e) { console.warn('refreshCharts error', e); }
    }
}

// ── Update KPI cards via DOM diff ────────────────────────────────────────────
async function _refreshKPIs() {
    try {
        const response = await fetch(window.location.href);
        const html     = await response.text();
        const parser   = new DOMParser();
        const doc      = parser.parseFromString(html, 'text/html');

        const incoming = doc.querySelectorAll('.kpi-card');
        const current  = document.querySelectorAll('.kpi-card');
        if (incoming.length === current.length) {
            current.forEach((el, i) => { el.innerHTML = incoming[i].innerHTML; });
        }

        const incomingTbody = doc.querySelector('#regionTable tbody');
        if (incomingTbody) {
            const cur = document.querySelector('#regionTable tbody');
            if (cur) cur.innerHTML = incomingTbody.innerHTML;
        }

        if (window.location.pathname.includes('/alerts')) {
            const incomingAlerts = doc.querySelector('.chart-card');
            const currentAlerts  = document.querySelector('.chart-card');
            if (incomingAlerts && currentAlerts) {
                currentAlerts.innerHTML = incomingAlerts.innerHTML;
                document.querySelectorAll('.alert-item.critical')
                        .forEach(a => a.classList.add('animate-critical'));
            }
        }
    } catch (e) { /* silently ignore */ }
}

// ── Main poll ────────────────────────────────────────────────────────────────
async function checkUpdates() {
    try {
        const res  = await fetch('/api/check_update');
        const data = await res.json();
        const { last_modified: mtime, record_count: count, critical_alerts: alerts } = data;

        // ── First call: initialise state ──────────────────────────────
        if (!_initialLoadDone) {
            _initialLoadDone = true;
            _lastMtime       = mtime;
            _lastRecordCount = count;

            // DOMContentLoaded already handles the initial chart render;
            // only play alarm if criticals exist on first load
            const storedKey = localStorage.getItem('alertPlayedForCount');
            if (alerts && alerts.length > 0 && storedKey !== String(count)) {
                playGlobalAlarm();
                alerts.forEach(a => showGlobalToast(a.message));
                localStorage.setItem('alertPlayedForCount', String(count));
            }
            return;
        }

        // ── Subsequent calls: detect changes ──────────────────────────────
        const countChanged = count !== _lastRecordCount;
        const mtimeChanged = mtime > _lastMtime;

        if (countChanged || mtimeChanged) {
            _lastMtime       = mtime;
            _lastRecordCount = count;

            // Refresh all charts
            _safeRefresh();

            // Refresh KPI cards
            _refreshKPIs();

            // Show LIVE badge
            const topbar = document.querySelector('.topbar-title');
            if (topbar && !topbar.innerHTML.includes('live-indicator')) {
                topbar.innerHTML += ` <span class="live-indicator ms-2 badge bg-success bg-opacity-10 text-success" style="font-size:0.6rem;vertical-align:middle;"><span class="spinner-grow spinner-grow-sm me-1" style="width:8px;height:8px;" role="status" aria-hidden="true"></span>LIVE</span>`;
            }

            // Alert sound
            if (alerts && alerts.length > 0) {
                const storedKey = localStorage.getItem('alertPlayedForCount');
                if (storedKey !== String(count)) {
                    playGlobalAlarm();
                    alerts.forEach(a => showGlobalToast(a.message));
                    localStorage.setItem('alertPlayedForCount', String(count));
                }
            }
        }
    } catch (e) {
        console.error('Live poll failed:', e);
    }
}

// Start polling every 3 seconds
setInterval(checkUpdates, 3000);
checkUpdates(); // run immediately on page load
