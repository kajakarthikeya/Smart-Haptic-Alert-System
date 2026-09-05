# API Documentation - Smart Haptic Alert System

## 1. Overview

The Smart Haptic Alert System provides a modular REST API built with **FastAPI**. It bridges the web frontend (and future mobile applications or BLE gateways) with the underlying Phase 1–8 AI inference and context decision pipelines.

- **Base URL**: `http://127.0.0.1:8000`
- **Prefix**: `/api/v1`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **OpenAPI JSON**: `http://127.0.0.1:8000/openapi.json`

---

## 2. Endpoints Reference

### 2.1 System Diagnostics

#### `GET /api/v1/system/status`
Retrieves comprehensive diagnostic metrics across AI model state, inference subsystem, context engine, and hardware availability.

**Response `200 OK`**:
```json
{
  "system_status": "Ready",
  "components": {
    "ai_model": {
      "status": "Loaded",
      "model_path": "app/outputs/model_training/sound_classifier_best.keras",
      "target_classes": ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"],
      "num_classes": 5,
      "architecture": "2D CNN (111,237 params)"
    },
    "inference_engine": {
      "status": "Ready",
      "sample_rate": 22050,
      "window_duration_sec": 4.0,
      "confidence_threshold": 0.70
    },
    "context_engine": {
      "status": "Ready",
      "current_mode": "HOME",
      "supported_modes": ["HOME", "ROAD", "OFFICE"],
      "confidence_threshold": 0.70
    },
    "audio_input": {
      "status": "Ready",
      "microphone_available": true,
      "default_microphone": "Microphone Array (Realtek)",
      "active_mode": "Test Audio / Demo"
    },
    "hardware": {
      "status": "Not Connected",
      "detail": "Software Prototype (Hardware was NOT required for this verification)",
      "ble_connected": false
    }
  },
  "current_mode": "HOME",
  "supported_sounds": ["ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark"],
  "is_recognizing": false,
  "latest_decision": null
}
```

---

### 2.2 Environment Mode Management

#### `GET /api/v1/mode`
Retrieves the active operating mode.

**Response `200 OK`**:
```json
{
  "mode": "HOME",
  "supported_modes": ["HOME", "ROAD", "OFFICE"]
}
```

#### `POST /api/v1/mode`
Switches operating mode in the backend's Phase 8 `ModeManager`.

**Request Body**:
```json
{
  "mode": "ROAD"
}
```

**Response `200 OK`**:
```json
{
  "status": "success",
  "previous_mode": "HOME",
  "current_mode": "ROAD"
}
```

---

### 2.3 Test Audio File Evaluation

#### `GET /api/v1/test-audio/samples`
Returns a list of verified audio files available for test processing.

**Response `200 OK`**:
```json
[
  {
    "name": "ambulance.wav",
    "path": "dataset/test_audio/ambulance.wav",
    "sound_class": "ambulance",
    "size_bytes": 176444,
    "source": "test_audio"
  },
  {
    "name": "car_horn.wav",
    "path": "dataset/test_audio/car_horn.wav",
    "sound_class": "car_horn",
    "size_bytes": 176444,
    "source": "test_audio"
  }
]
```

#### `POST /api/v1/test-audio/evaluate`
Processes an audio WAV file through the full Phase 3 $\rightarrow$ 4 $\rightarrow$ 5 $\rightarrow$ 7 $\rightarrow$ 8 pipeline.

**Request Body**:
```json
{
  "file_path": "dataset/test_audio/car_horn.wav",
  "mode": "ROAD"
}
```

**Response `200 OK`**:
```json
{
  "sound": "car_horn",
  "confidence": 0.94,
  "mode": "ROAD",
  "priority": "HIGH",
  "alert_required": true,
  "reason": "Car Horn has HIGH priority in ROAD mode.",
  "timestamp": "2026-09-05T07:16:39.451068+00:00",
  "source": "file:car_horn.wav",
  "latency": {
    "preprocessing_ms": 17.86,
    "feature_extraction_ms": 1176.67,
    "inference_ms": 36.04,
    "total_ms": 1232.79
  },
  "top_probabilities": {
    "car_horn": 0.94,
    "ambulance": 0.03,
    "fire_alarm": 0.02
  },
  "prediction_status": "CONFIRMED"
}
```

#### `POST /api/v1/test-audio/upload`
Uploads a custom WAV file via `multipart/form-data` and executes the pipeline.

---

### 2.4 Demo / Simulation Mode

#### `POST /api/v1/demo/simulate`
Simulates a target sound with specified confidence under the current or overridden mode, passing through the actual Phase 8 `ContextDecisionEngine`.

**Request Body**:
```json
{
  "sound": "car_horn",
  "confidence": 0.92,
  "mode": "ROAD"
}
```

**Response `200 OK`**:
```json
{
  "sound": "car_horn",
  "confidence": 0.92,
  "mode": "ROAD",
  "priority": "HIGH",
  "alert_required": true,
  "reason": "Car Horn has HIGH priority in ROAD mode.",
  "timestamp": "2026-09-05T07:16:19.123456+00:00",
  "source": "demo_simulation"
}
```

---

### 2.5 Real-Time Recognition Control

#### `POST /api/v1/recognition/start`
Starts live microphone stream capture and periodic inference if microphone is accessible.

#### `POST /api/v1/recognition/stop`
Stops live microphone recognition.

#### `GET /api/v1/recognition/latest`
Polls the latest prediction and decision event.

---

### 2.6 Alert History & Scenarios

#### `GET /api/v1/alerts/history`
Returns recent alert history (up to 100 in-memory items).

#### `POST /api/v1/alerts/clear`
Clears in-memory alert history.

#### `GET /api/v1/scenarios/run`
Executes the 7 mandatory verification scenarios against the Phase 8 context engine and returns test results.
