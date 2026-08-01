# API Delivery Subsystem (`app/api/`)

## Purpose
Exposes RESTful FastAPI endpoints allowing external clients (e.g. Mobile Application, Wearable companion app, Web dashboard) to inspect system status, change environment modes, and view alert history.

## API Endpoint Matrix

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | System status check |
| `GET` | `/api/v1/mode` | Fetch active operating mode profile |
| `POST` | `/api/v1/mode` | Switch mode (`HOME`, `ROAD`, `OFFICE`) |
| `POST` | `/api/v1/alerts/trigger` | Manually evaluate & dispatch a sound alert |
| `GET` | `/api/v1/alerts/history` | Retrieve alert history records |
