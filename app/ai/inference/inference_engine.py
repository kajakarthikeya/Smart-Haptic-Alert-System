"""Real-Time Inference Engine connecting Preprocessor, Feature Extractor, and Sound Classifier."""

from typing import Any, List, Optional, Tuple
from app.ai.models.base_model import BaseSoundClassifier
from app.ai.models.model_factory import ModelFactory
from app.ai.preprocessing.audio_preprocessor import AudioPreprocessor
from app.ai.feature_extraction.spectrogram_extractor import SpectrogramExtractor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SoundInferenceEngine:
    """Orchestrates end-to-end sound detection inference pipeline."""

    def __init__(
        self,
        model: Optional[BaseSoundClassifier] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
        feature_extractor: Optional[SpectrogramExtractor] = None,
    ) -> None:
        """Initializes inference pipeline components.

        Args:
            model: Instance of BaseSoundClassifier. Defaults to ModelFactory starter model.
            preprocessor: Audio preprocessor. Defaults to standard AudioPreprocessor.
            feature_extractor: Spectrogram feature extractor. Defaults to standard SpectrogramExtractor.
        """
        self._preprocessor = preprocessor or AudioPreprocessor()
        self._feature_extractor = feature_extractor or SpectrogramExtractor()
        self._model = model or ModelFactory.create_model("starter")
        logger.info("SoundInferenceEngine initialized successfully.")

    def run_inference(self, raw_audio_pcm: Any) -> Tuple[str, float, List[float]]:
        """Processes raw audio PCM frame through full pipeline to produce prediction.

        Pipeline Stages:
        Raw PCM -> Preprocessing -> Spectrogram Extraction -> Model Classification

        Args:
            raw_audio_pcm: Raw 1D waveform input audio chunk.

        Returns:
            Tuple of (sound_label: str, confidence_score: float, probability_distribution: List[float]).
        """
        cleaned_signal = self._preprocessor.process(raw_audio_pcm)
        features = self._feature_extractor.extract(cleaned_signal)
        label, confidence, probs = self._model.predict(features)

        logger.debug(f"Inference output: detected '{label}' with confidence {confidence:.2f}")
        return label, confidence, probs
