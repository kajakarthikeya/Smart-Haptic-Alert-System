# Real-Time Sound Recognition Documentation

## 1. Overview & Architecture

The **Real-Time Sound Recognition System** (Phase 7) transforms the trained environmental sound classification CNN model into a production-ready, low-latency streaming inference engine. It captures acoustic audio signals from physical hardware microphones or audio files, applies rigorous preprocessing and feature extraction identical to the training phase, performs convolutional neural network inference, and applies configurable confidence thresholding and temporal stability filtering to eliminate transient false triggers.

```
       +---------------------------------------------+
       |   Microphone / Audio Hardware Input         |
       |   (sounddevice.InputStream, 22050 Hz, Mono)  |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Circular Ring Buffer Windowing            |
       |   (Rolling 4.0s = 88,200 samples, Hop=1.0s) |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Preprocessing Pipeline (Shared Reuse)     |
       |   (AudioStandardizer, LengthStandardizer)   |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Feature Pipeline (Shared Reuse)           |
       |   (FeatureExtractor.extract_composite_matrix)|
       |   Shape: (1, 184, 173, 1)                   |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   CNN Inference Engine                      |
       |   (sound_classifier_best.keras, Warm model) |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Confidence Gating (e.g. >= 70.0%)         |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Prediction Stabilizer (N=3, K=2 Buffer)   |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   PredictionResult Event & Telemetry        |
       |   (CONFIRMED, TENTATIVE, or LOW_CONFIDENCE) |
       +---------------------------------------------+
```

---

## 2. Target Acoustic Sound Classes

The inference engine strictly recognizes the 5 target classes aligned with training and evaluation:

| Label ID | Class Name | Description |
|---|---|---|
| 0 | `ambulance` | Emergency vehicle siren acoustic patterns |
| 1 | `car_horn` | Automotive warning horn bursts |
| 2 | `fire_alarm` | High-frequency building fire / evacuation alarms |
| 3 | `doorbell` | Residential chimes and doorbells |
| 4 | `dog_bark` | Domestic canine vocalizations / barking |

---

## 3. Core Components

### 3.1 Audio Device Manager & Capture (`app/ai/inference/audio_capture.py`)
- **`AudioDeviceManager`**: Enumerates host OS audio recording devices, reports default microphone index, channel counts, and native sampling rates.
- **`MicrophoneAudioCapture`**: Implements thread-safe, non-blocking audio capture utilizing `sounddevice.InputStream`. Audio chunks (`block_size=1024`) are written to a fixed-capacity circular numpy ring buffer storing the latest 4.0 seconds ($88,200$ samples at $22,050$ Hz).

### 3.2 Preprocessing & Feature Pipeline (`app/ai/inference/feature_pipeline.py`)
- **Code Reuse**: Directly imports and instantiates Phase 3's `AudioStandardizer` (mono conversion, 22,050 Hz resampling, peak amplitude normalization) and `LengthStandardizer` (truncation / zero-padding to 88,200 samples).
- **Composite Feature Extraction**: Invokes Phase 4's `FeatureExtractor.extract_composite_matrix`, computing 128 Mel bands + 40 MFCCs + Spectral Centroid + Bandwidth + Rolloff + Zero Crossing Rate + RMS Energy ($184 \times 173$).
- **Channel Expansion**: Expands matrix into 4D tensor `(1, 184, 173, 1)` matching Keras CNN expectations.

### 3.3 Model Loader (`app/ai/inference/model_loader.py`)
- **`InferenceModelLoader`**: Loads `sound_classifier_best.keras` once at startup, executes a zero-dummy tensor warm-up inference pass to compile execution graphs, and validates input/output tensor shapes against `training_metadata.json` and `class_mapping.json`.

### 3.4 Confidence Gating & Stabilization (`app/ai/inference/prediction.py`)
- **Threshold Gating**: Computes maximum softmax probability. If below threshold (default `0.70`), classifies prediction as `LOW_CONFIDENCE`.
- **Temporal Stabilization**: Employs a sliding deque of size $N=3$ (configurable). An event is tagged `CONFIRMED` only if at least $K=2$ out of $N=3$ successive windows agree on the same predicted class, filtering out single-frame acoustic anomalies.

### 3.5 Master Recognizer (`app/ai/inference/realtime_recognizer.py`)
- **`RealtimeSoundRecognizer`**: Integrates all components into a coherent API.
  - `recognize_file(file_path)`: Offline validation of individual WAV recordings.
  - `recognize_window(waveform)`: Inference on an arbitrary 1D numpy array.
  - `start_streaming(on_prediction, block=True)`: Launches real-time microphone capture thread and periodic window evaluation loop (default hop interval = 1.0s).
  - Telemetry: Tracks preprocessing, feature extraction, inference, and end-to-end total latency.

---

## 4. Usage Instructions

### 4.1 CLI Execution

#### List Available Recording Devices
```bash
python -m app.ai.inference.cli --list-devices
```

#### Test Single Audio File
```bash
python -m app.ai.inference.cli --test-file dataset/processed/car_horn/sample_0.wav --threshold 0.30
```

#### Launch Real-Time Microphone Streaming
```bash
python -m app.ai.inference.cli --live --threshold 0.70 --hop-sec 1.0
```

### 4.2 Programmatic API Usage
```python
from app.ai.inference import RealtimeSoundRecognizer, PredictionStatus

# Initialize recognizer
recognizer = RealtimeSoundRecognizer(confidence_threshold=0.75)

# Recognize single audio file
result = recognizer.recognize_file("path/to/test.wav")
print(f"Detected: {result.predicted_class} ({result.confidence*100:.1f}%) Status: {result.status.value}")

# Streaming callback
def on_alert(res):
    if res.status == PredictionStatus.CONFIRMED:
        print(f"[ALERT] Confirmed sound: {res.predicted_class} ({res.confidence*100:.1f}%)")

# recognizer.start_streaming(on_prediction=on_alert, hop_sec=1.0)
```

---

## 5. Latency & Resource Benchmarks

| Processing Stage | Average Duration | Notes |
|---|---|---|
| Audio Window Capture | Non-blocking / Threaded | Circular ring buffer continuous update |
| Preprocessing | 15 - 25 ms | Resampling & peak normalization |
| Feature Extraction | 40 - 80 ms | STFT, Mel filterbank, MFCCs, Spectral features |
| CNN Inference | 10 - 25 ms | Pre-warmed Keras/PyTorch model forward pass |
| Stabilization & Gating | < 1 ms | Deque counter agreement verification |
| **Total Inference Latency** | **65 - 130 ms** | Well within the 1.0s hop interval budget |

---

## 6. Verification & Test Suite

The real-time recognition module is comprehensively tested via `app/tests/test_realtime_recognition.py` (14 unit tests):
- Audio device querying and configuration.
- Circular buffer capacity, overwriting, and window extraction.
- Strict reuse of Phase 3 and Phase 4 preprocessing and feature pipelines.
- Model loader validation, shape inspection, and warm-up pass.
- Confidence threshold filtering and status assignment.
- Consensus stabilizer buffer logic ($N=3, K=2$).
- End-to-end audio file prediction.
- Latency metrics and session telemetry summaries.
