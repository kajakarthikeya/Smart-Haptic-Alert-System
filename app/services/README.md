# Application Services (`app/services/`)

## Purpose
Enforces core business logic rules, orchestrating interactions between AI inference outputs, context prioritization rules, and hardware communication interfaces.

## Key Services
- **`alert_service.py`**: Accepts detected environmental sound events, queries `ContextManager` for urgency authorization, formats alert records, and dispatches haptic instructions via `ESP32BLEManager`.
- **`audio_service.py`**: Manages real-time audio intake buffers and forwards raw PCM chunks to `SoundInferenceEngine`.
