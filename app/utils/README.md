# Application Utilities (`app/utils/`)

## Purpose
Provides shared, reusable utility modules across the system without introducing circular dependencies.

## Key Modules
- **`logger.py`**: Centralized structured logging implementation supporting console output and rotating log files under `app/logs/`.
- **`helpers.py`**: Cross-cutting helper routines including ISO 8601 UTC timestamp generation, unique UUID alert tracking, and audio file validation.

## Usage Example

```python
from app.utils.logger import get_logger
from app.utils.helpers import generate_alert_id, format_timestamp

logger = get_logger(__name__)
alert_id = generate_alert_id()
logger.info(f"Generated alert {alert_id} at {format_timestamp()}")
```
