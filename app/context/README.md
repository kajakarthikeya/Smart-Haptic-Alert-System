# Context-Aware Decision Module (`app/context/`)

## 1. Overview & Conceptual Flow

The **Context-Aware Decision Module** evaluates detected environmental sounds produced by the AI Recognition Subsystem (Phase 7), contextualizes them according to the user's active environment mode (**Home**, **Road**, **Office**), applies confidence gating, and outputs a structured alert decision specifying priority level, boolean alert requirement, and human-readable reasoning.

```
+-------------------------------------------------------------+
| Phase 7 Real-Time Inference (PredictionResult)              |
| (sound="car_horn", confidence=0.93, timestamp="...")        |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| SoundPrediction Data Model                                  |
| (sound: str, confidence: float, timestamp: str)             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| ContextDecisionEngine                                       |
|                                                             |
|   1. Confidence Gating: [0.0, 1.0]                          |
|      If confidence < threshold (0.70) -> IGNORE, Alert=False|
|                                                             |
|   2. Query ModeManager: Active Mode (HOME / ROAD / OFFICE)  |
|                                                             |
|   3. Query PriorityEngine:                                  |
|      - Validate sound in target_classes                     |
|      - Validate mode in supported_modes                     |
|      - O(1) Matrix Lookup                                   |
|                                                             |
|   4. Evaluate Alert Decision Policy:                        |
|      HIGH / MEDIUM -> Alert=True                            |
|      LOW / IGNORE   -> Alert=False                          |
|                                                             |
|   5. Synthesize Explanation Reason                          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| DecisionResult                                              |
| {                                                           |
|   "sound": "car_horn",                                      |
|   "confidence": 0.93,                                       |
|   "mode": "ROAD",                                           |
|   "priority": "HIGH",                                       |
|   "alert_required": true,                                   |
|   "reason": "Car Horn has HIGH priority in ROAD mode."      |
| }                                                           |
+-------------------------------------------------------------+
```

---

## 2. Supported Operating Modes

- **`HOME`**: Domestic residential setting (sensitive to doorbells, smoke/fire alarms, domestic emergencies).
- **`ROAD`**: Outdoor and transit setting (sensitive to automotive horns, emergency vehicle sirens, traffic threats).
- **`OFFICE`**: Professional workplace (filters mundane noise; prioritizes building alarms and emergency evacuations).

---

## 3. Supported Sounds & Priority Matrix

The engine targets the 5 AI environmental sound classes from Phases 4–7:

| Sound Label | HOME Mode | ROAD Mode | OFFICE Mode |
|---|---|---|---|
| `ambulance` | **HIGH** | **HIGH** | **HIGH** |
| `car_horn` | **MEDIUM** | **HIGH** | **LOW** |
| `fire_alarm` | **HIGH** | **HIGH** | **HIGH** |
| `doorbell` | **HIGH** | **LOW** | **LOW** |
| `dog_bark` | **MEDIUM** | **LOW** | **LOW** |

### Priority Levels & Default Alert Policy:
- **`HIGH`** $\rightarrow$ Immediate Alert (`True`)
- **`MEDIUM`** $\rightarrow$ Alert (`True`)
- **`LOW`** $\rightarrow$ Informational / No Immediate Alert (`False`)
- **`IGNORE`** $\rightarrow$ Filtered / No Alert (`False`)

---

## 4. Confidence Gating & Safety Rules

- Reuses the Phase 7 confidence threshold (default: `0.70`).
- If an AI prediction has confidence $< 0.70$, the decision engine suppresses the alert, setting priority to `IGNORE`, `alert_required=False`, with reason `"Prediction confidence (X.XX%) is below the configured threshold (70.00%)."`.
- Confidence values $< 0.0$, $> 1.0$, `NaN`, or `Inf` raise `InvalidConfidenceError`.
- Unknown sound labels raise `UnknownSoundError`.
- Unsupported modes raise `InvalidModeError`.

---

## 5. Usage Example

```python
from app.context import (
    ContextDecisionEngine,
    EnvironmentMode,
    ModeManager,
    SoundPrediction,
)

# 1. Initialize engine
mode_mgr = ModeManager(default_mode=EnvironmentMode.ROAD)
engine = ContextDecisionEngine(mode_manager=mode_mgr, confidence_threshold=0.70)

# 2. Evaluate sound prediction
prediction = SoundPrediction(sound="car_horn", confidence=0.92)
decision = engine.evaluate(prediction)

print(decision)
# DecisionResult(sound='car_horn', confidence=92.00%, mode=ROAD, priority=HIGH, alert=True, reason='Car Horn has HIGH priority in ROAD mode.')
print(decision.to_dict())
```
