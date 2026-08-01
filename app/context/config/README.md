# Mode Profiles & Priority Definitions (`app/context/config/`)

## Purpose
Defines environment mode enumeration (`EnvironmentMode`), alert urgency scale (`SoundPriority`), and priority profiles (`ModeProfile`) for **Home**, **Road**, and **Office** contexts.

## Default Priority Profiles Matrix

| Sound Event | Home Mode | Road Mode | Office Mode |
| :--- | :---: | :---: | :---: |
| Fire Alarm / Smoke Detector | CRITICAL | CRITICAL | CRITICAL |
| Siren | HIGH | CRITICAL | MEDIUM |
| Doorbell | HIGH | IGNORE | LOW |
| Door Knock | MEDIUM | IGNORE | HIGH |
| Car Horn | LOW | HIGH | IGNORE |
| Baby Crying | HIGH | LOW | IGNORE |
| Speech / Name Call | LOW | MEDIUM | HIGH |

## Customization
Additional profiles can be registered dynamically at runtime in the `ContextManager` without modifying core logic.
