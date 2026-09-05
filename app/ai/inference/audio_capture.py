"""
Microphone Audio Capture and Rolling Buffer Subsystem.

Provides:
1. AudioDeviceManager: Enumerates and validates audio input hardware devices.
2. MicrophoneAudioCapture: Non-blocking, thread-safe streaming audio capture using sounddevice
   with circular buffer windowing, graceful error handling, and clean resource release.
"""

from dataclasses import dataclass
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None
    _SOUNDDEVICE_AVAILABLE = False

from config import Config, InferenceConfig, settings
from app.ai.inference.exceptions import (
    AudioCaptureError,
    DeviceNotFoundError,
    MicrophoneInitializationError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Metadata describing an audio hardware input device."""
    device_id: int
    name: str
    max_input_channels: int
    default_samplerate: float
    is_default: bool


class AudioDeviceManager:
    """Discovers and inspects available audio input hardware devices."""

    @staticmethod
    def is_sounddevice_available() -> bool:
        """Returns True if the sounddevice library is successfully imported."""
        return _SOUNDDEVICE_AVAILABLE

    @classmethod
    def list_input_devices(cls) -> List[AudioDeviceInfo]:
        """Queries the host operating system for all active audio recording devices.

        Returns:
            List of AudioDeviceInfo objects for devices supporting >= 1 input channel.
        """
        if not cls.is_sounddevice_available():
            logger.warning("sounddevice library not available; returning empty device list.")
            return []

        try:
            devices = sd.query_devices()
            default_input_id = sd.default.device[0] if sd.default.device is not None else -1

            input_devices: List[AudioDeviceInfo] = []
            for dev_id, dev in enumerate(devices):
                in_channels = dev.get("max_input_channels", 0)
                if in_channels > 0:
                    info = AudioDeviceInfo(
                        device_id=dev_id,
                        name=dev.get("name", f"Device {dev_id}"),
                        max_input_channels=in_channels,
                        default_samplerate=float(dev.get("default_samplerate", 44100.0)),
                        is_default=(dev_id == default_input_id),
                    )
                    input_devices.append(info)

            logger.info("Discovered %d audio input devices.", len(input_devices))
            return input_devices

        except Exception as exc:
            logger.error("Failed to query host audio devices: %s", exc)
            return []

    @classmethod
    def get_device_info(cls, device_id: Optional[int] = None) -> AudioDeviceInfo:
        """Retrieves hardware specifications for a specific or default input device."""
        devices = cls.list_input_devices()
        if not devices:
            raise DeviceNotFoundError("No audio recording devices found on this system.")

        if device_id is None:
            # Prefer marked default, else first available input
            for d in devices:
                if d.is_default:
                    return d
            return devices[0]

        for d in devices:
            if d.device_id == device_id:
                return d

        available_ids = [d.device_id for d in devices]
        raise DeviceNotFoundError(
            f"Requested audio device ID {device_id} not found. Available input IDs: {available_ids}"
        )


class MicrophoneAudioCapture:
    """
    Manages non-blocking streaming microphone capture into a circular rolling buffer.

    Maintains a rolling audio buffer sized to the exact window duration (default 4.0s = 88,200 samples)
    for seamless real-time inference without sample dropouts.
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        block_size: Optional[int] = None,
        window_duration_sec: Optional[float] = None,
        device_id: Optional[int] = None,
    ) -> None:
        self.config = config or getattr(settings, "inference", None)

        self.sample_rate = int(
            sample_rate
            or (self.config.sample_rate if self.config else 22050)
        )
        self.channels = int(
            channels
            or (self.config.channels if self.config else 1)
        )
        self.block_size = int(
            block_size
            or (self.config.block_size if self.config else 1024)
        )
        self.window_duration_sec = float(
            window_duration_sec
            or (self.config.window_duration_sec if self.config else 4.0)
        )
        self.window_samples = int(self.sample_rate * self.window_duration_sec)

        self.device_id = (
            device_id
            if device_id is not None
            else (self.config.input_device_id if self.config else None)
        )

        # Threading and streaming state
        self._is_recording = False
        self._stream: Optional[Any] = None
        self._lock = threading.Lock()
        self._audio_queue: queue.Queue = queue.Queue()

        # Rolling circular buffer: 1D float32 array of shape (window_samples,)
        self._rolling_buffer = np.zeros(self.window_samples, dtype=np.float32)
        self._total_samples_captured = 0

    @property
    def is_recording(self) -> bool:
        """Returns True if the microphone stream is actively recording."""
        return self._is_recording

    @property
    def total_samples_captured(self) -> int:
        """Returns cumulative count of captured audio samples."""
        return self._total_samples_captured

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """Internal sounddevice callback executed from audio thread."""
        if status:
            logger.warning("Microphone stream status flag: %s", status)

        # Mono conversion if recorded in multi-channel
        if indata.ndim > 1 and indata.shape[1] > 1:
            chunk = np.mean(indata, axis=1, dtype=np.float32)
        else:
            chunk = indata.flatten().astype(np.float32)

        # Update circular rolling buffer under lock
        with self._lock:
            num_new = len(chunk)
            if num_new >= self.window_samples:
                self._rolling_buffer[:] = chunk[-self.window_samples:]
            else:
                self._rolling_buffer[:-num_new] = self._rolling_buffer[num_new:]
                self._rolling_buffer[-num_new:] = chunk
            self._total_samples_captured += num_new

        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def start(self) -> None:
        """Initializes and opens the microphone audio input stream."""
        if not _SOUNDDEVICE_AVAILABLE:
            raise MicrophoneInitializationError(
                "sounddevice library is not available. Ensure it is installed and host audio drivers are active."
            )

        if self._is_recording:
            logger.warning("Microphone capture stream is already running.")
            return

        try:
            logger.info(
                "Initializing microphone capture: device_id=%s, sr=%d Hz, channels=%d, block_size=%d",
                self.device_id,
                self.sample_rate,
                self.channels,
                self.block_size,
            )

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                device=self.device_id,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording = True
            logger.info("Microphone stream started successfully.")

        except Exception as exc:
            self._is_recording = False
            self._stream = None
            raise MicrophoneInitializationError(
                f"Failed to open microphone input stream on device {self.device_id}: {exc}"
            ) from exc

    def stop(self) -> None:
        """Stops and closes the audio stream gracefully."""
        if not self._is_recording:
            return

        self._is_recording = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            logger.info("Microphone capture stopped and resources released.")
        except Exception as exc:
            logger.warning("Error encountered while closing audio stream: %s", exc)
            self._stream = None

    def get_latest_window(self) -> np.ndarray:
        """Retrieves a snapshot of the latest 4.0-second (88,200 samples) rolling audio buffer.

        Returns:
            1D numpy array of shape (window_samples,) with float32 values.
        """
        with self._lock:
            return np.copy(self._rolling_buffer)

    def read_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Pulls the next available audio chunk from the queue (blocking with timeout)."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_buffer(self) -> None:
        """Resets the rolling buffer to zeros."""
        with self._lock:
            self._rolling_buffer.fill(0.0)
            self._total_samples_captured = 0
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

    def __enter__(self) -> "MicrophoneAudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
