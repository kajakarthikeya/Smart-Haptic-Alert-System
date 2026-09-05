# Frontend Prototype - Smart Haptic Alert System

## 1. Overview

The **Smart Haptic Alert System Web Prototype** is a clean, modern, hardware-independent single-page dashboard created to integrate and verify the completed **Phase 1 through Phase 8** software subsystems.

> [!IMPORTANT]
> **Hardware Status: Software Prototype Only**
> Physical hardware (ESP32, INMP441 microphone module, vibration motor, and Bluetooth BLE transceiver) is **NOT required** for running this dashboard and verifying the system. Hardware communication will be introduced in **Phase 9**.

---

## 2. Architecture & Data Flow

The dashboard communicates with a lightweight local FastAPI backend serving REST endpoints. The underlying software pipeline operates without mocking or synthetic shortcuts:

```
[Test WAV File] OR [Demo Event] OR [Live Mic Stream]
                         │
                         ▼
        Phase 3: Audio Preprocessing Subsystem
            (22,050 Hz Resampling, Mono, 4.0s Window)
                         │
                         ▼
       Phase 4: Feature Extraction Subsystem
            (MFCC, Mel Spectrogram, Spectral Features)
                         │
                         ▼
       Phase 5: Trained CNN Model Checkpoint
            (sound_classifier_best.keras)
                         │
                         ▼
       Phase 7: Real-Time Recognition Subsystem
            (Confidence Gating & Temporal Stabilization)
                         │
                         ▼
       Phase 8: Context-Aware Decision Subsystem
            (ModeManager + PriorityEngine + DecisionEngine)
                         │
                         ▼
        FastAPI Integration Service & REST Endpoints
                         │
                         ▼
       Frontend Single-Page Dashboard (Vanilla JS/CSS)
```

---

## 3. Key Frontend Features

1. **Diagnostic System Status Strip**:
   - Displays real-time status of AI Model (`Loaded`), Inference Engine (`Ready`), Context Engine (`Ready`), Audio Input (`Ready`), and Hardware (`Not Connected - Software Prototype`).

2. **Synchronized Operating Mode Selector**:
   - Three interactive mode buttons: `[Home]`, `[Road]`, and `[Office]`.
   - Directly mutates state in the backend's Phase 8 `ModeManager` (not just client-side JavaScript).

3. **Active Detection Telemetry & Card**:
   - Shows detected sound, model confidence percentage, animated confidence bar, priority pill badge (`HIGH`, `MEDIUM`, `LOW`, `IGNORE`), alert requirement badge (`ALERT REQUIRED` vs `SUPPRESSED`), and transparent decision reasoning.
   - Live latency breakdown: Preprocessing time, Feature extraction time, Neural inference time, and Total latency in milliseconds.

4. **Interactive Action Tabs**:
   - **Test Audio WAV Mode**: Run preprocessed or bundled WAV files (`ambulance.wav`, `car_horn.wav`, `fire_alarm.wav`, `doorbell.wav`, `dog_bark.wav`) through the genuine Phase 3–8 pipeline, or upload custom WAV clips.
   - **Demo / Simulation Mode**: Clearly labeled simulator allowing quick evaluation of any of the 5 target sounds under current operating mode through the actual Phase 8 context rules.
   - **Live Microphone Recognition**: Starts real-time stream ingestion via laptop/PC microphone if supported. If unavailable, clearly reports: *"Microphone unavailable — use Test Audio or Demo Mode."*
   - **7 Test Scenarios Matrix**: One-click verification running the 7 mandatory benchmark scenarios through the backend and displaying pass/fail indicators.

5. **Recent Alert History Table**:
   - Stores the last 100 alert events with timestamp, sound class, confidence score, mode, priority badge, and alert outcome.
   - Includes a *Clear History* control.

---

## 4. How to Run the Dashboard

### 1. Ensure Dependencies Are Installed
```bash
pip install -r requirements.txt
```

### 2. Start the Application Server
```bash
python main.py
```
*(Default host: `http://127.0.0.1:8000`)*

### 3. Open in Your Browser
Navigate to:
- Dashboard: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) or [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
- Interactive API Docs (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 5. REST API Endpoints Reference

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/system/status` | Retrieves full diagnostic status of all software phases |
| `GET` | `/api/v1/mode` | Gets current environment mode (`HOME`, `ROAD`, `OFFICE`) |
| `POST` | `/api/v1/mode` | Sets current environment mode in Phase 8 `ModeManager` |
| `GET` | `/api/v1/test-audio/samples` | Lists available test WAV files in `dataset/test_audio/` |
| `POST` | `/api/v1/test-audio/evaluate` | Evaluates a test WAV file through Phase 3->4->5->7->8 |
| `POST` | `/api/v1/test-audio/upload` | Uploads and processes a custom WAV file |
| `POST` | `/api/v1/demo/simulate` | Simulates a target sound through Phase 8 Context Engine |
| `POST` | `/api/v1/recognition/start` | Starts live microphone stream recognition |
| `POST` | `/api/v1/recognition/stop` | Stops live microphone stream recognition |
| `GET` | `/api/v1/recognition/latest` | Retrieves latest prediction and decision |
| `GET` | `/api/v1/alerts/history` | Retrieves recent processed alerts |
| `POST` | `/api/v1/alerts/clear` | Clears recent alert history |
| `GET` | `/api/v1/scenarios/run` | Runs all 7 mandatory verification scenarios |

---

## 6. Verification Status

- **Automated Integration Tests**: 14/14 tests passing (`app/tests/test_software_integration.py`).
- **Total Repository Tests**: 123/123 tests passing across all packages.
- **Hardware Dependency**: Zero. Operates fully in software mode.
