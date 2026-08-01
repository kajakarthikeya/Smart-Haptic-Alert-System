# Mobile Application Companion (`mobile_app/`)

## Purpose
Reserved directory for the cross-platform (Flutter / React Native / Native Mobile) mobile companion application.

## Planned Features (Future Phases)
1. **Mode Switcher UI**: Quick toggles for switching between **Home**, **Road**, and **Office** modes.
2. **Wearable BLE Pairing**: Device discovery and pairing interface for the ESP32 wristband.
3. **Alert Dashboard & History**: Real-time notification log and visual indicators for detected sounds.
4. **Sensitivity Customization**: Adjusting threshold sliders per sound category.

## Integration Endpoint
Communicates with `app.api.routes` FastAPI endpoints (`http://<host>:8000/api/v1`).
