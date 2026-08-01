# App Package Core (`app/`)

## Purpose
The `app/` directory houses the core application logic for the **Smart Haptic Alert System** formatted according to **Clean Architecture** principles.

## Package Architecture

```
app/
├── ai/             # AI Subsystem (Dataset, Preprocessing, Extraction, Models, Training, Inference)
├── context/        # Environmental Context Engine & Mode Priority Configurations (Home, Road, Office)
├── bluetooth/      # ESP32 BLE Hardware Communication & Signal Protocol
├── api/            # API Delivery Endpoints (FastAPI Routers)
├── controllers/    # Request & Business Command Dispatchers
├── services/       # Core Business Logic Services (Alert Service, Audio Stream Manager)
├── utils/          # Cross-cutting Utilities (Structured Logger, Helper functions)
├── tests/          # Unit & Integration Test Suites
├── outputs/        # Output Artifacts Storage (TFLite Models, Exported Spectrograms)
├── logs/           # Application Execution Log Storage
└── docs/           # Architecture diagrams & extended technical specifications
```

## Key Principles
1. **Separation of Concerns**: Each package handles a single operational domain.
2. **Dependency Rule**: Inner layers (Context, AI Interfaces) do not depend on outer layers (FastAPI, Hardware Drivers).
3. **Testability**: All business logic and priority handlers are testable in isolation.
