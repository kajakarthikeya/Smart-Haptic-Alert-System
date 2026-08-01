# Bluetooth Subsystem (`app/bluetooth/`)

## Purpose
The Bluetooth package manages Bluetooth Low Energy (BLE) communication between the central AI system host and the remote **ESP32 Wearable Device**.

## Key Components
- **`protocol.py`**: `HapticPacketSerializer` serializing alert IDs and priority urgency levels into compact 6-byte binary packages.
- **`ble_manager.py`**: Abstract `BaseBLEManager` and starter `ESP32BLEManager` for managing connection state and GATT characteristic writes.

## Binary Protocol Specification

| Offset | Type | Field | Description |
| :--- | :--- | :--- | :--- |
| `0x00 - 0x01` | uint16 | `alert_hash` | Hash value identifying unique alert event |
| `0x02` | uint8 | `priority` | Priority level integer (1=Low, 2=Medium, 3=High, 4=Critical) |
| `0x03` | uint8 | `pattern_id` | ESP32 vibration pattern preset (1 to 5) |
| `0x04 - 0x05` | uint16 | `duration_ms` | Vibration burst duration in milliseconds |

## Hardware Target
- Target Microcontroller: ESP32 / ESP32-S3
- Protocol: BLE GATT Characteristic Write
