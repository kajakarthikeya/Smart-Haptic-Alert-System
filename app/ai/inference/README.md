# Real-Time Sound Recognition Module (`app/ai/inference`)

## Overview

The `app/ai/inference` module provides an end-to-end, high-performance real-time sound recognition engine for the **Smart Haptic Alert System**. It converts raw streaming microphone audio or recorded WAV files into classified environmental sound alerts with low-latency execution, confidence gating, and multi-window prediction stability.

---

## Architecture & Subsystems

| Component | File | Responsibility |
| :--- | :--- | :--- |
| **AudioDeviceManager** | `audio_capture.py` | Enumerates host audio recording hardware devices and queries device specifications. |
| **MicrophoneAudioCapture** | `audio_capture.py` | Non-blocking thread-safe streaming audio capture with circular buffer windowing (88,200 samples). |
| **RealtimeFeaturePipeline** | `feature_pipeline.py` | Standardizes audio (Mono, 22050 Hz, [-0.95, 0.95]) and extracts composite 2D features `(184, 173, 1)`. |
| **InferenceModelLoader** | `model_loader.py` | Eagerly loads model weights once at startup, validates shapes and target class counts. |
| **PredictionStabilizer** | `prediction.py` | Multi-window consensus buffer ($N=3, K=2$) that filters out transient acoustic noise spikes. |
| **RealtimeSoundRecognizer** | `realtime_recognizer.py` | Master orchestrator coordinating capture, feature extraction, inference, and latency telemetry. |
| **CLI Dashboard** | `cli.py` | Live terminal user interface with real-time alerts, device listing, test-file mode, and session summary. |
| **Exceptions** | `exceptions.py` | Domain exception hierarchy inheriting from `InferenceError`. |

---

## Target Sound Classes

The engine recognizes exactly the 5 target classes from Phase 5 training:
1. **Ambulance** (ID: 0)
2. **Car Horn** (ID: 1)
3. **Fire Alarm** (ID: 2)
4. **Doorbell** (ID: 3)
5. **Dog Bark** (ID: 4)

---

## Real-Time Inference Workflow

```
[Microphone Stream / Test WAV]
        │ (22,050 Hz, Mono)
        ▼
[Circular Rolling Buffer] ────────> Sized to exactly 4.0 seconds (88,200 samples)
        │
        ▼
[Signal Preprocessing] ───────────> Mono Conversion -> Peak Normalization [-0.95, 0.95] -> Zero-Padding/Trimming
        │
        ▼
[Feature Extractor] ──────────────> Extract Composite Matrix (184, 173) -> Expand to (1, 184, 173, 1)
        │
        ▼
[CNNSoundClassifier] ─────────────> 2D CNN Softmax Inference -> Output Probabilities & Winning Class
        │
        ▼
[Confidence Gating] ──────────────> If confidence >= threshold (default 0.70):
        │                              Pass Class Label
        │                           Else:
        │                              Mark "Unknown / Low Confidence"
        ▼
[Prediction Stabilizer] ──────────> Require K agreement in last N windows:
        │                              Matches >= K -> CONFIRMED
        │                              Matches < K  -> TENTATIVE
        ▼
[PredictionResult Event] ─────────> Dispatch to CLI / Consumer Service (Phase 8 Context Manager)
```

---

## CLI Usage

### 1. List Available Audio Recording Devices
```powershell
python -m app.ai.inference.cli --list-devices
```

### 2. Run Test-File Mode (Offline WAV Verification)
```powershell
python -m app.ai.inference.cli --test-file dataset/processed/car_horn/sample_0.wav --threshold 0.30
```

### 3. Run Live Microphone Streaming
```powershell
python -m app.ai.inference.cli --live --threshold 0.70 --hop-sec 1.0
```

---

## Programmatic API Example

```python
from app.ai.inference.realtime_recognizer import RealtimeSoundRecognizer
from app.ai.inference.prediction import PredictionResult

# Initialize recognizer (loads model once)
recognizer = RealtimeSoundRecognizer(confidence_threshold=0.70)

# Offline single-file recognition
result = recognizer.recognize_file("dataset/processed/car_horn/sample_0.wav")
print(f"Detected: {result.predicted_class} ({result.confidence * 100:.1f}%) Status: {result.status.value}")

# Real-time microphone streaming
def on_alert(res: PredictionResult):
    if res.status.value == "CONFIRMED":
        print(f"CONFIRMED ALERT: {res.predicted_class} ({res.confidence*100:.1f}%)")

recognizer.start_streaming(callback=on_alert, hop_duration_sec=1.0)

# Stop streaming when done
recognizer.stop_streaming()
summary = recognizer.get_session_summary()
print(f"Average Total Latency: {summary['average_total_latency_ms']:.2f} ms")
```
