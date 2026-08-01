# Smart Haptic Alert System - Architecture Specification

## Architectural Overview

The **Smart Haptic Alert System** is engineered using **Clean Architecture** and **SOLID Principles** to provide a scalable, maintainable AI assistance platform for hearing-impaired users.

```
+-----------------------------------------------------------------------+
|                           Delivery Layer                              |
|   +--------------------------+     +-------------------------------+  |
|   |  FastAPI Router (api/)   |     |  Mobile App Companion Interface|  |
|   +------------+-------------+     +---------------+---------------+  |
+----------------|-----------------------------------|------------------+
                 |                                   |
                 v                                   v
+-----------------------------------------------------------------------+
|                         Controller Layer                              |
|   +--------------------------+     +-------------------------------+  |
|   |  AlertController        |     |  ModeController               |  |
|   +------------+-------------+     +---------------+---------------+  |
+----------------|-----------------------------------|------------------+
                 |                                   |
                 v                                   v
+-----------------------------------------------------------------------+
|                          Service Layer                                |
|   +--------------------------+     +-------------------------------+  |
|   |  AlertService            |     |  AudioService                 |  |
|   +------------+-------------+     +---------------+---------------+  |
+----------------|-----------------------------------|------------------+
                 |                                   |
                 v                                   v
+-----------------------------------------------------------------------+
|                      Context Engine Layer                             |
|   +----------------------------------------------------------------+  |
|   |  ContextManager (Home, Road, Office Mode Priority Evaluation) |  |
|   +----------------------------------------------------------------+  |
+------------------------------------+----------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
+-----------------------------+             +-----------------------------+
|    AI Pipeline Subsystem    |             |    Bluetooth LE Subsystem   |
|  - AudioPreprocessor        |             |  - HapticPacketSerializer   |
|  - SpectrogramExtractor     |             |  - ESP32BLEManager          |
|  - ModelFactory / Classifier|             |                             |
+-----------------------------+             +--------------+--------------+
                                                           |
                                                           v
                                            +-----------------------------+
                                            |   ESP32 Wearable Device     |
                                            |   (Haptic Vibration Motors) |
                                            +-----------------------------+
```

## System Workflow & Signal Path

1. **Audio Intake**: Microphone captures real-time 16 kHz audio PCM buffers via `AudioService`.
2. **AI Inference Pipeline**:
   - `AudioPreprocessor`: Normalizes peak amplitude and formats to 1.0s window.
   - `SpectrogramExtractor`: Generates 64-band Log-Mel Spectrogram representation.
   - `BaseSoundClassifier` / `ModelFactory`: Evaluates spectrogram, outputting sound label and confidence score.
3. **Context Evaluation**:
   - `AlertService` passes sound event to `ContextManager`.
   - `ContextManager` evaluates active `EnvironmentMode` (`HOME`, `ROAD`, `OFFICE`).
   - If priority exceeds threshold (e.g. Baby Crying in Home mode), alert is authorized.
4. **Haptic Hardware Dispatch**:
   - `HapticPacketSerializer` encodes alert ID and priority into compact 6-byte binary payload.
   - `ESP32BLEManager` writes payload over BLE GATT characteristic to trigger vibration motors on user's wearable wristband.
