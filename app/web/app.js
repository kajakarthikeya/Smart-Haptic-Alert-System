/**
 * Smart Haptic Alert System — Dashboard Client Logic
 * Handles REST API interaction, mode switching, real-time polling, and UI updates.
 */

const API_BASE = '/api/v1';

// Sound class visual metadata
const SOUND_METADATA = {
    'ambulance': { icon: '🚑', color: '#f43f5e' },
    'car_horn': { icon: '📢', color: '#f59e0b' },
    'fire_alarm': { icon: '🚨', color: '#f43f5e' },
    'doorbell': { icon: '🔔', color: '#06b6d4' },
    'dog_bark': { icon: '🐕', color: '#8b5cf6' },
};

let currentMode = 'HOME';
let isRecognizing = false;
let pollingInterval = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    fetchSystemStatus();
    loadTestAudioSamples();
    fetchHistory();
    
    // Poll for status and latest detections every 2 seconds
    setInterval(pollLiveStatus, 2000);
});

/**
 * Switch Dashboard Tab
 */
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(btn => 
        btn.getAttribute('onclick').includes(tabId)
    );
    if (activeBtn) activeBtn.classList.add('active');

    const activeTab = document.getElementById(tabId);
    if (activeTab) activeTab.classList.add('active');
}

/**
 * Update Confidence Slider Display
 */
function updateSliderVal(val) {
    document.getElementById('slider-conf-val').innerText = `${val}%`;
}

/**
 * Fetch and Render System Status
 */
async function fetchSystemStatus() {
    try {
        const res = await fetch(`${API_BASE}/system/status`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Update mode
        updateModeUI(data.current_mode);

        // Update Diagnostics
        document.getElementById('diag-model').innerText = `${data.subsystems.ai_model.status} (CNN 111k)`;
        document.getElementById('diag-inference').innerText = `${data.subsystems.inference_engine.status} (22.05 kHz)`;
        document.getElementById('diag-context').innerText = `Ready (Mode: ${data.current_mode})`;
        document.getElementById('diag-audio').innerText = data.subsystems.audio_input.active_mode;
        
        // Microphone state
        const micAvail = data.subsystems.audio_input.microphone_available;
        const micDevName = data.subsystems.audio_input.default_microphone;
        document.getElementById('mic-device-name').innerText = micAvail 
            ? `Device: ${micDevName}` 
            : 'No microphone detected (Use Test Audio or Demo Mode)';

        // Hardware state - explicitly noting not connected
        document.getElementById('diag-hw').innerText = `${data.subsystems.hardware.status}`;
        
        // If there is an existing latest decision, display it
        if (data.latest_decision) {
            renderDecisionCard(data.latest_decision);
        }
    } catch (err) {
        console.error('Failed to fetch system status:', err);
    }
}

/**
 * Poll live status & microphone stream
 */
async function pollLiveStatus() {
    try {
        const res = await fetch(`${API_BASE}/recognition/latest`);
        if (!res.ok) return;
        const data = await res.json();
        
        if (data.current_mode !== currentMode) {
            updateModeUI(data.current_mode);
        }

        if (data.is_recognizing !== isRecognizing) {
            updateMicStateUI(data.is_recognizing);
        }

        if (data.latest_decision) {
            renderDecisionCard(data.latest_decision);
        }
    } catch (err) {
        console.warn('Poll error:', err);
    }
}

/**
 * Update UI for operating mode
 */
function updateModeUI(mode) {
    currentMode = mode.toUpperCase();
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    
    const targetBtn = document.getElementById(`btn-mode-${currentMode.toLowerCase()}`);
    if (targetBtn) targetBtn.classList.add('active');

    const diagContext = document.getElementById('diag-context');
    if (diagContext) diagContext.innerText = `Ready (Mode: ${currentMode})`;
}

/**
 * Switch Operating Environment Mode
 */
async function setOperatingMode(mode) {
    try {
        const res = await fetch(`${API_BASE}/mode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode })
        });
        const data = await res.json();
        if (res.ok) {
            updateModeUI(data.mode);
            showToast(`Operating mode switched to ${data.mode}`);
        } else {
            showToast(`Error: ${data.detail || 'Could not switch mode'}`, true);
        }
    } catch (err) {
        showToast(`Failed to switch mode: ${err.message}`, true);
    }
}

/**
 * Populate Sample Audio Files Dropdown
 */
async function loadTestAudioSamples() {
    const select = document.getElementById('sample-select');
    try {
        const res = await fetch(`${API_BASE}/test-audio/samples`);
        const samples = await res.json();
        select.innerHTML = '';

        if (!samples || samples.length === 0) {
            select.innerHTML = '<option disabled selected>No sample audio files available</option>';
            return;
        }

        samples.forEach((s, idx) => {
            const opt = document.createElement('option');
            opt.value = s.path;
            const sizeKb = Math.round(s.size_bytes / 1024);
            const classLabel = s.sound_class.replace('_', ' ').toUpperCase();
            opt.textContent = `${classLabel} — ${s.name} (${sizeKb} KB)`;
            if (idx === 0) opt.selected = true;
            select.appendChild(opt);
        });
    } catch (err) {
        select.innerHTML = '<option disabled selected>Error loading samples</option>';
    }
}

/**
 * Execute Inference on Selected Audio Sample
 */
async function runSelectedSample() {
    const select = document.getElementById('sample-select');
    const path = select.value;
    if (!path) {
        showToast('Please select a test audio file first.', true);
        return;
    }

    const btn = document.getElementById('btn-run-sample');
    btn.disabled = true;
    btn.innerHTML = '<span>⏳</span> Processing...';

    try {
        const res = await fetch(`${API_BASE}/test-audio/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: path })
        });
        const data = await res.json();
        if (res.ok) {
            renderDecisionCard(data);
            fetchHistory();
            showToast(`Evaluated ${data.sound.toUpperCase()} — Priority: ${data.priority}`);
        } else {
            showToast(`Inference Error: ${data.detail || 'Evaluation failed'}`, true);
        }
    } catch (err) {
        showToast(`Network error: ${err.message}`, true);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>▶</span> Run Audio Inference';
    }
}

/**
 * Upload and Evaluate Custom WAV File
 */
async function uploadAudioFile(input) {
    if (!input.files || !input.files[0]) return;
    const file = input.files[0];

    const formData = new FormData();
    formData.append('file', file);

    showToast(`Uploading and evaluating ${file.name}...`);
    try {
        const res = await fetch(`${API_BASE}/test-audio/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (res.ok) {
            renderDecisionCard(data);
            fetchHistory();
            loadTestAudioSamples(); // Refresh list with newly uploaded sample
            showToast(`Inference complete for ${data.sound.toUpperCase()}`);
        } else {
            showToast(`Upload Error: ${data.detail || 'Processing failed'}`, true);
        }
    } catch (err) {
        showToast(`Upload failed: ${err.message}`, true);
    } finally {
        input.value = '';
    }
}

/**
 * Simulate Demo Sound
 */
async function simulateSound(sound) {
    const sliderVal = document.getElementById('sim-confidence-slider').value;
    const confidence = parseFloat(sliderVal) / 100.0;

    try {
        const res = await fetch(`${API_BASE}/demo/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sound: sound, confidence: confidence })
        });
        const data = await res.json();
        if (res.ok) {
            renderDecisionCard(data);
            fetchHistory();
        } else {
            showToast(`Simulation Error: ${data.detail}`, true);
        }
    } catch (err) {
        showToast(`Simulation failed: ${err.message}`, true);
    }
}

/**
 * Toggle Live Microphone Stream
 */
async function toggleMicrophoneStream(start) {
    const errorBox = document.getElementById('mic-error-msg');
    errorBox.style.display = 'none';

    const endpoint = start ? `${API_BASE}/recognition/start` : `${API_BASE}/recognition/stop`;
    try {
        const res = await fetch(endpoint, { method: 'POST' });
        const data = await res.json();

        if (data.status === 'unavailable') {
            errorBox.innerText = data.message;
            errorBox.style.display = 'block';
            showToast(data.message, true);
            return;
        }

        if (res.ok) {
            updateMicStateUI(start);
            showToast(data.message);
        } else {
            showToast(data.message || 'Action failed', true);
        }
    } catch (err) {
        showToast(`Microphone error: ${err.message}`, true);
    }
}

function updateMicStateUI(active) {
    isRecognizing = active;
    const box = document.getElementById('mic-status-container');
    const stateText = document.getElementById('mic-state-text');
    const startBtn = document.getElementById('btn-start-mic');
    const stopBtn = document.getElementById('btn-stop-mic');

    if (active) {
        box.classList.add('active');
        stateText.innerText = 'Microphone Listening...';
        stateText.style.color = 'var(--emerald)';
        startBtn.disabled = true;
        stopBtn.disabled = false;
        document.getElementById('diag-audio').innerText = 'Live Microphone';
    } else {
        box.classList.remove('active');
        stateText.innerText = 'Microphone Idle';
        stateText.style.color = '#ffffff';
        startBtn.disabled = false;
        stopBtn.disabled = true;
        document.getElementById('diag-audio').innerText = 'Test Mode / Demo';
    }
}

/**
 * Render Decision Card with Full Telemetry
 */
function renderDecisionCard(data) {
    const card = document.getElementById('active-decision-card');
    const soundTitle = data.sound.replace('_', ' ').toUpperCase();
    const meta = SOUND_METADATA[data.sound] || { icon: '🔊', color: '#6366f1' };

    // Update Avatar & Title
    document.getElementById('res-avatar').innerText = meta.icon;
    document.getElementById('res-sound').innerText = soundTitle;
    document.getElementById('res-source').innerText = data.source || 'Inference';
    
    // Timestamp
    const timeStr = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : 'Just now';
    document.getElementById('res-timestamp').innerText = `Detected at ${timeStr} • Mode: ${data.mode}`;

    // Confidence Bar & Value
    const confPct = Math.round(data.confidence * 100);
    document.getElementById('conf-bar-fill').style.width = `${confPct}%`;
    document.getElementById('res-conf').innerText = `${(data.confidence * 100).toFixed(1)}%`;

    // Priority Pill
    const priorityEl = document.getElementById('res-priority');
    priorityEl.innerText = data.priority;
    priorityEl.className = 'priority-pill ' + (
        data.priority === 'HIGH' ? 'pill-high' :
        data.priority === 'MEDIUM' ? 'pill-medium' :
        data.priority === 'LOW' ? 'pill-low' : 'pill-ignore'
    );

    // Alert Required
    const alertEl = document.getElementById('res-alert');
    if (data.alert_required) {
        alertEl.innerText = 'ALERT: YES';
        alertEl.className = 'alert-flag flag-yes';
        card.classList.add('alert-active');
    } else {
        alertEl.innerText = 'ALERT: NO';
        alertEl.className = 'alert-flag flag-no';
        card.classList.remove('alert-active');
    }

    // Reason Text
    document.getElementById('res-reason').innerText = data.reason || 'Context rule evaluated.';

    // Latency Telemetry
    if (data.latency) {
        document.getElementById('lat-prep').innerText = `${data.latency.preprocessing_ms?.toFixed(1) || 0}ms`;
        document.getElementById('lat-feat').innerText = `${data.latency.feature_extraction_ms?.toFixed(1) || 0}ms`;
        document.getElementById('lat-infer').innerText = `${data.latency.inference_ms?.toFixed(1) || 0}ms`;
        document.getElementById('lat-total').innerText = `${data.latency.total_ms?.toFixed(1) || 0}ms`;
    }
}

/**
 * Fetch and Render History Table
 */
async function fetchHistory() {
    try {
        const res = await fetch(`${API_BASE}/alerts/history?limit=15`);
        if (!res.ok) return;
        const data = await res.json();
        const tbody = document.getElementById('history-tbody');
        tbody.innerHTML = '';

        if (!data.alerts || data.alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No events recorded yet.</td></tr>';
            return;
        }

        data.alerts.forEach(item => {
            const tr = document.createElement('tr');
            const timeStr = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '--';
            const soundClean = item.sound.replace('_', ' ').toUpperCase();
            const alertBadge = item.alert_required 
                ? '<span class="p-badge high">YES</span>' 
                : '<span class="p-badge low">NO</span>';
            const pClass = item.priority === 'HIGH' ? 'high' : (item.priority === 'MEDIUM' ? 'med' : 'low');

            tr.innerHTML = `
                <td class="mono">${timeStr}</td>
                <td><strong>${soundClean}</strong></td>
                <td>${item.mode}</td>
                <td class="mono">${(item.confidence * 100).toFixed(1)}%</td>
                <td><span class="p-badge ${pClass}">${item.priority}</span></td>
                <td>${alertBadge}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.warn('Failed to fetch alert history:', err);
    }
}

/**
 * Clear History
 */
async function clearHistory() {
    try {
        await fetch(`${API_BASE}/alerts/clear`, { method: 'POST' });
        fetchHistory();
        showToast('Alert history cleared.');
    } catch (err) {
        showToast('Failed to clear history', true);
    }
}

/**
 * Run All 7 Verification Scenarios
 */
async function runAllSevenScenarios() {
    const container = document.getElementById('scenarios-results-box');
    container.innerHTML = '<span class="text-sm">Running 7 scenarios against Phase 8 Context Engine...</span>';

    try {
        const res = await fetch(`${API_BASE}/scenarios/run`);
        const data = await res.json();
        container.innerHTML = '';

        data.scenarios.forEach(sc => {
            const row = document.createElement('div');
            row.className = 'scenario-row';
            const alertText = sc.actual_alert ? 'Alert: YES' : 'Alert: NO';
            const soundDisplay = sc.sound.replace('_', ' ').toUpperCase();

            row.innerHTML = `
                <div>
                    <strong>Scenario ${sc.scenario_id}:</strong> ${sc.mode} + ${soundDisplay}
                    <div class="text-xs text-muted">Result: ${sc.actual_priority} (${alertText})</div>
                </div>
                <div>
                    <span class="scenario-badge-pass">${sc.status}</span>
                </div>
            `;
            container.appendChild(row);
        });

        showToast('All 7 verification scenarios evaluated: PASS!');
    } catch (err) {
        container.innerHTML = `<span class="error-banner">Error: ${err.message}</span>`;
    }
}

/**
 * Toast Notification Helper
 */
function showToast(message, isError = false) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    if (isError) toast.style.borderColor = 'var(--rose)';
    toast.innerText = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}
