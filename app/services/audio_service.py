"""Audio Stream Capture & Ingestion Manager Service."""

from typing import Any, Callable, Optional
from app.ai.inference.inference_engine import SoundInferenceEngine
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class AudioService:
    """Manages audio capture queue and forwards frame buffers to inference engine."""

    def __init__(self, inference_engine: Optional[SoundInferenceEngine] = None) -> None:
        """Initializes audio stream service.

        Args:
            inference_engine: SoundInferenceEngine pipeline instance.
        """
        self._inference_engine = inference_engine or SoundInferenceEngine()
        self._is_streaming: bool = False
        self._sample_rate = settings.audio.sample_rate
        logger.info(f"AudioService initialized (target sample rate: {self._sample_rate} Hz)")

    def start_listening(self, callback: Optional[Callable[[str, float], None]] = None) -> None:
        """Starts real-time microphone capture or stream ingestion thread.

        Args:
            callback: Optional callback function triggered when sound is recognized.
        """
        if self._is_streaming:
            logger.warning("AudioService is already streaming.")
            return

        self._is_streaming = True
        logger.info("Audio capture stream started.")

    def stop_listening(self) -> None:
        """Stops live audio stream capture."""
        if not self._is_streaming:
            return

        self._is_streaming = False
        logger.info("Audio capture stream stopped.")

    def process_frame(self, raw_audio_pcm: Any) -> Optional[tuple[str, float]]:
        """Processes an incoming raw audio frame buffer synchronously.

        Args:
            raw_audio_pcm: Raw PCM audio array chunk.

        Returns:
            Tuple of (detected_sound_label, confidence_score) or None.
        """
        if not self._is_streaming:
            logger.debug("Received audio frame while service is idle. Processing frame directly.")

        label, confidence, _ = self._inference_engine.run_inference(raw_audio_pcm)
        return label, confidence

    @property
    def is_listening(self) -> bool:
        """Returns active listening status."""
        return self._is_streaming
