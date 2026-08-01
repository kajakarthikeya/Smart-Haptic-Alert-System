# Software Architecture Specification - Smart Haptic Alert System

## 1. Executive Architectural Overview

The **Smart Haptic Alert System** is an AI-powered assistance platform designed for hearing-impaired users. The system captures real-time environmental sound streams, classifies target acoustic events using deep neural networks, prioritizes alerts according to user context (**Home**, **Road**, **Office**), and dispatches haptic vibration patterns to an **ESP32 wearable device**.

The architecture adheres strictly to **Clean Architecture** principles, maintaining clear boundaries between domain business logic, AI pipelines, hardware transport protocols, and delivery layers.

```
+-----------------------------------------------------------------------------------+
|                                 DELIVERY LAYER                                    |
|   +-------------------------------------+     +-------------------------------+   |
|   | FastAPI REST Routers (app/api/)     |     | Mobile App Companion          |   |
|   +------------------+------------------+     +---------------+---------------+   |
+----------------------|----------------------------------------|-------------------+
                       v                                        v
+-----------------------------------------------------------------------------------+
|                               CONTROLLER LAYER                                    |
|   +-------------------------------------+     +-------------------------------+   |
|   | AlertController (app/controllers/)  |     | ModeController                |   |
|   +------------------+------------------+     +---------------+---------------+   |
+----------------------|----------------------------------------|-------------------+
                       v                                        v
+-----------------------------------------------------------------------------------+
|                                SERVICE LAYER                                      |
|   +-------------------------------------+     +-------------------------------+   |
|   | AlertService (app/services/)        |     | AudioService                  |   |
|   +------------------+------------------+     +---------------+---------------+   |
+----------------------|----------------------------------------|-------------------+
                       v                                        v
+-----------------------------------------------------------------------------------+
|                            CONTEXT ENGINE LAYER                                   |
|   +---------------------------------------------------------------------------+   |
|   | ContextManager (Home, Road, Office Mode Priority Matrix & Thresholds)     |   |
|   +------------------------------------+--------------------------------------+   |
+----------------------------------------|------------------------------------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
+-----------------------------------------+     +-----------------------------------+
|          AI PIPELINE SUBSYSTEM          |     |    BLUETOOTH HARDWARE SUBSYSTEM   |
|  - Dataset Management (app/ai/dataset/) |     |  - HapticPacketSerializer         |
|  - Preprocessing (app/ai/preprocessing/)|     |  - ESP32BLEManager                |
|  - Feature Extraction                   |     +-----------------+-----------------+
|  - BaseSoundClassifier / ModelFactory   |                       |
+-----------------------------------------+                       v
                                                +-----------------------------------+
                                                |     ESP32 Wearable Device         |
                                                |     (Haptic Vibration Motors)     |
                                                +-----------------------------------+
```

---

## 2. Core Subsystems

### 2.1 AI Subsystem (`app/ai/`)
Structured into decoupled pipeline phases:
- **`dataset/`**: Directory management, multi-format loading (`.wav`, `.mp3`, `.flac`), validation, statistics, and exploration. Target classes: `ambulance`, `car_horn`, `fire_alarm`, `doorbell`, `dog_bark`.
- **`preprocessing/`**: Resampling (16 kHz), peak normalization, and fixed 1.0-second framing.
- **`feature_extraction/`**: 64-band Log-Mel Spectrogram extraction.
- **`models/`**: `BaseSoundClassifier` contract, `ModelFactory`, and `StarterMockClassifier`.
- **`training/`**: Training pipeline loop and TFLite quantization exporter.
- **`inference/`**: End-to-end `SoundInferenceEngine` integrating preprocessor, feature extractor, and classifier.

### 2.2 Environmental Context Subsystem (`app/context/`)
- `ContextManager`: Evaluates incoming sound events against active `EnvironmentMode` (`HOME`, `ROAD`, `OFFICE`).
- Sound priority levels (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `IGNORE`) and confidence threshold filters.

### 2.3 Bluetooth Hardware Subsystem (`app/bluetooth/`)
- `HapticPacketSerializer`: Encodes alert ID and priority into a 6-byte binary payload.
- `ESP32BLEManager`: Manages BLE GATT characteristic writes to the ESP32 wristband.

---

## 3. SOLID Principles Enforcement

- **Single Responsibility Principle (SRP)**: Each class handles a single concern (e.g. `DatasetValidator` only validates; `DatasetStatisticsCalculator` only computes metrics).
- **Open/Closed Principle (OCP)**: New sound models or data loaders can be added by implementing base abstractions without modifying existing caller code.
- **Liskov Substitution Principle (LSP)**: All classifier implementations fulfill the `BaseSoundClassifier` interface.
- **Interface Segregation Principle (ISP)**: Interfaces are lean and focused (`BaseDatasetLoader`, `BasePreprocessor`, `BaseFeatureExtractor`).
- **Dependency Inversion Principle (DIP)**: Higher-level services (`AlertService`, `SoundInferenceEngine`) depend on abstract contracts rather than concrete implementations.
