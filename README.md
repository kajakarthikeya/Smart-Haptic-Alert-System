# Smart Haptic Alert System

An AI-powered wearable assistance system for hearing-impaired users that detects important environmental sounds, prioritizes them according to the user's selected mode (**Home**, **Road**, **Office**), and communicates haptic alert signals to an **ESP32 wearable device**.

---

## Architecture & Principles

Built following **Clean Architecture**, **SOLID principles**, and **Python 3.11** best practices.

```
Smart-Haptic-Alert-System/
├── README.md                           # Project overview & documentation
├── requirements.txt                    # Production dependencies
├── main.py                             # System bootstrap entrypoint
├── config.py                           # Central configuration manager
├── .env.example                        # Environment variables template
├── .gitignore                          # Git exclusion rules
│
├── app/
│   ├── ai/                             # Machine Learning Subsystem
│   │   ├── dataset/                    # Data ingestion & class indexing
│   │   ├── preprocessing/              # Audio normalization & framing
│   │   ├── feature_extraction/        # Log-Mel Spectrogram extraction
│   │   ├── training/                   # Trainer pipeline & model exporter
│   │   ├── inference/                  # Real-time sound inference engine
│   │   ├── models/                     # Abstract Base Model & ModelFactory
│   │   └── utils/                      # Metrics & confusion matrix helpers
│   │
│   ├── context/                        # Context Prioritization Engine
│   │   ├── config/                     # Home, Road, Office Mode profiles
│   │   └── context_manager.py          # Sound priority evaluator
│   │
│   ├── bluetooth/                      # Hardware Communication Subsystem
│   │   ├── ble_manager.py              # ESP32 BLE client interface
│   │   └── protocol.py                 # Haptic binary packet serializer
│   │
│   ├── api/                            # FastAPI REST Delivery Layer
│   ├── controllers/                    # Request & command dispatchers
│   ├── services/                       # Core business logic services
│   ├── utils/                          # Structured logger & helper utilities
│   ├── tests/                          # Automated pytest test suites
│   ├── outputs/                        # Model binaries & visual artifacts
│   ├── logs/                           # System execution log storage
│   └── docs/                           # Architecture specifications
│
└── mobile_app/                         # Mobile Companion App integration guide
```

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.11+
- Virtual Environment (`venv`)

### 2. Environment Setup
Clone the repository and prepare the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

### 3. Run Bootstrap Entrypoint

```bash
python main.py
```

### 4. Run Test Suite

```bash
pytest
```

---

## Environmental Operating Modes

- **Home Mode**: Prioritizes doorbells, baby crying, fire alarms, glass shattering; suppresses traffic noise.
- **Road Mode**: Prioritizes vehicle horns, emergency sirens, vehicle engines; suppresses doorbells.
- **Office Mode**: Prioritizes door knocks, speech/calling name, phone ringers; suppresses keyboard typing and road sounds.

---

## Hardware Integration (ESP32 Wearable)

Alerts are sent to an ESP32 wearable wristband via Bluetooth LE GATT characteristic payload format:
- **0x00 - 0x01**: uint16 Alert Hash
- **0x02**: uint8 Priority Level
- **0x03**: uint8 Haptic Pattern Preset
- **0x04 - 0x05**: uint16 Vibration Duration (ms)
