"""Configuration Management Module for Smart Haptic Alert System.

Provides environment variable loading, type casting, validation, and central
access to system settings across all packages.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional
BASE_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    def load_dotenv(path: Optional[Path] = None, override: bool = False) -> None:
        target = path or (BASE_DIR / ".env")
        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        key = k.strip()
                        val = v.strip().strip('"').strip("'")
                        if override or key not in os.environ:
                            os.environ[key] = val

    load_dotenv()


@dataclass(frozen=True)
class SystemConfig:
    """System identification and operational configuration."""
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "Smart Haptic Alert System"))
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    initial_mode: str = field(default_factory=lambda: os.getenv("INITIAL_MODE", "HOME").upper())


@dataclass(frozen=True)
class AudioConfig:
    """Audio sampling and feature extraction settings."""
    sample_rate: int = field(default_factory=lambda: int(os.getenv("SAMPLE_RATE", "16000")))
    channels: int = field(default_factory=lambda: int(os.getenv("AUDIO_CHANNELS", "1")))
    frame_duration_sec: float = field(default_factory=lambda: float(os.getenv("FRAME_DURATION_SEC", "1.0")))
    n_fft: int = field(default_factory=lambda: int(os.getenv("N_FFT", "512")))
    hop_length: int = field(default_factory=lambda: int(os.getenv("HOP_LENGTH", "160")))
    n_mels: int = field(default_factory=lambda: int(os.getenv("N_MELS", "64")))


@dataclass(frozen=True)
class BLEConfig:
    """Bluetooth LE ESP32 Wearable target configuration."""
    device_name: str = field(default_factory=lambda: os.getenv("BLE_DEVICE_NAME", "ESP32-Haptic-Alert"))
    device_mac: str = field(default_factory=lambda: os.getenv("BLE_DEVICE_MAC", "00:11:22:33:44:55"))
    alert_service_uuid: str = field(
        default_factory=lambda: os.getenv("BLE_ALERT_SERVICE_UUID", "0000180a-0000-1000-8000-00805f9b34fb")
    )
    alert_char_uuid: str = field(
        default_factory=lambda: os.getenv("BLE_ALERT_CHAR_UUID", "00002a29-0000-1000-8000-00805f9b34fb")
    )
    reconnect_attempts: int = field(default_factory=lambda: int(os.getenv("BLE_RECONNECT_ATTEMPTS", "5")))


@dataclass(frozen=True)
class APIConfig:
    """FastAPI delivery layer configuration."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset storage paths and target sound class configurations."""
    root_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("DATASET_ROOT_DIR", "dataset"))
    raw_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("DATASET_RAW_DIR", "dataset/raw"))
    processed_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("DATASET_PROCESSED_DIR", "dataset/processed")
    )
    test_audio_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("DATASET_TEST_DIR", "dataset/test_audio")
    )
    target_classes: tuple[str, ...] = ("ambulance", "car_horn", "fire_alarm", "doorbell", "dog_bark")
    supported_extensions: tuple[str, ...] = (".wav", ".mp3", ".flac")


@dataclass(frozen=True)
class PathConfig:
    """File system paths for model artifacts, logs, and outputs."""
    base_dir: Path = BASE_DIR
    model_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("MODEL_PATH", "app/outputs/sound_classifier.tflite")
    )
    log_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("LOG_DIR", "app/logs"))
    output_dir: Path = BASE_DIR / "app" / "outputs"


class AppSettings:
    """Central settings manager singleton aggregating all configuration domains."""

    def __init__(self) -> None:
        self.system = SystemConfig()
        self.audio = AudioConfig()
        self.ble = BLEConfig()
        self.api = APIConfig()
        self.paths = PathConfig()
        self.dataset = DatasetConfig()

        # Ensure directories exist
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

    def reload(self) -> None:
        """Reload environment settings dynamically."""
        load_dotenv(BASE_DIR / ".env", override=True)
        self.__init__()


# Global singleton instance for settings access
settings = AppSettings()
