# Application Controllers (`app/controllers/`)

## Purpose
Acts as the intermediary between delivery interfaces (FastAPI API endpoints, CLI, mobile app requests) and internal application services.

## Key Controllers
- **`alert_controller.py`**: Manages manual alert trigger requests and history retrieval operations.
- **`mode_controller.py`**: Handles environment mode inspection (`HOME`, `ROAD`, `OFFICE`) and user mode switching commands.
