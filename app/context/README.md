# Context Engine Package (`app/context/`)

## Purpose
The Context Engine evaluates detected environmental sounds against user-selected operating profiles (**Home**, **Road**, **Office**) to determine alert urgency and prevent alert fatigue.

## Key Components
- **`context_manager.py`**: State container and decision module evaluating sound labels and confidence scores against active profiles.
- **`config/mode_profiles.py`**: Prioritization matrices, thresholds, and enumerations (`EnvironmentMode`, `SoundPriority`).

## Usage Example

```python
from app.context import ContextManager, EnvironmentMode, SoundPriority

context_mgr = ContextManager(initial_mode=EnvironmentMode.HOME)

# Evaluate a detected sound event
should_alert, priority = context_mgr.evaluate_sound(sound_label="baby_crying", confidence=0.88)
if should_alert:
    print(f"Triggering haptic alert! Priority: {priority.name}")

# Change operating mode
context_mgr.set_mode(EnvironmentMode.ROAD)
```
