"""
Real-Time Sound Recognition Engine.

Coordinates microphone audio capture, audio windowing, Phase 3 preprocessing,
Phase 4 feature extraction, CNN model inference, confidence gating, stability consensus,
and end-to-end performance measurement.
"""

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import librosa

from config import Config, InferenceConfig, settings
from app.ai.inference.audio_capture import MicrophoneAudioCapture
from app.ai.inference.exceptions import (
    PredictionError,
    RealtimeInferenceError,
)
from app.ai.inference.feature_pipeline import RealtimeFeaturePipeline
from app.ai.inference.model_loader import InferenceModelLoader
from app.ai.inference.prediction import (
    LatencyMetrics,
    PredictionResult,
    PredictionStabilizer,
    PredictionStatus,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RealtimeSoundRecognizer:
    """
    Master Real-Time Sound Recognition Controller.

    Supports:
    - Continuous microphone recognition loop with non-blocking circular windowing.
    - Offline standalone test-file inference (WAV input).
    - Configurable confidence gating and multi-window prediction stability.
    - Detailed latency telemetry profiling.
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        model_loader: Optional[InferenceModelLoader] = None,
        feature_pipeline: Optional[RealtimeFeaturePipeline] = None,
        audio_capture: Optional[MicrophoneAudioCapture] = None,
        stabilizer: Optional[PredictionStabilizer] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        self.config = config or getattr(settings, "inference", None)

        # 1. Model Loader
        self.model_loader = model_loader or InferenceModelLoader(config=self.config)

        # 2. Feature Pipeline
        self.feature_pipeline = feature_pipeline or RealtimeFeaturePipeline(config=self.config)

        # 3. Audio Capture (Lazy-loaded if needed for streaming)
        self._audio_capture = audio_capture

        # 4. Confidence Threshold
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)
        elif self.config and hasattr(self.config, "confidence_threshold"):
            self.confidence_threshold = float(self.config.confidence_threshold)
        else:
            self.confidence_threshold = 0.70

        # 5. Prediction Stabilizer
        buf_size = (
            self.config.stability_buffer_size
            if self.config and hasattr(self.config, "stability_buffer_size")
            else 3
        )
        req_agree = (
            self.config.required_agreement
            if self.config and hasattr(self.config, "required_agreement")
            else 2
        )
        self.stabilizer = stabilizer or PredictionStabilizer(
            buffer_size=buf_size,
            required_agreement=req_agree,
        )

        # Streaming thread control
        self._streaming_active = False
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Telemetry metrics accumulator
        self._session_results: List[PredictionResult] = []
        self._lock = threading.Lock()

        # Eagerly load model on startup
        self.model_loader.load()

    @property
    def audio_capture(self) -> MicrophoneAudioCapture:
        """Lazily instantiates audio capture device."""
        if self._audio_capture is None:
            self._audio_capture = MicrophoneAudioCapture(config=self.config)
        return self._audio_capture

    @property
    def is_streaming(self) -> bool:
        """Returns True if continuous microphone recognition loop is active."""
        return self._streaming_active

    def recognize_window(
        self,
        raw_waveform: Union[np.ndarray, List[float]],
        orig_sr: Optional[int] = None,
    ) -> PredictionResult:
        """
        Executes end-to-end inference on a single audio window:
        Audio Window -> Preprocessing -> Features -> CNN Prediction -> Confidence & Stability.

        Args:
            raw_waveform: 1D audio sample array.
            orig_sr: Input sample rate (defaults to 22050 Hz).

        Returns:
            PredictionResult dataclass with class predictions, probabilities, and latency breakdown.
        """
        t_start = time.perf_counter()

        # 1. Preprocessing & Feature Extraction
        feature_tensor, prep_time, feat_time = self.feature_pipeline.process(
            raw_waveform=raw_waveform,
            orig_sr=orig_sr,
        )

        # 2. Model Inference
        t_inf_start = time.perf_counter()
        try:
            classifier = self.model_loader.classifier
            raw_class, confidence, probs_list = classifier.predict(feature_tensor)
            inf_time = time.perf_counter() - t_inf_start
        except Exception as exc:
            raise PredictionError(f"Model inference execution failed: {exc}") from exc

        total_time = time.perf_counter() - t_start

        # 3. Construct probabilities dictionary
        class_names = self.model_loader.class_names
        prob_dict = {
            class_names[idx]: float(probs_list[idx])
            for idx in range(min(len(class_names), len(probs_list)))
        }

        # 4. Determine raw ID
        raw_id = class_names.index(raw_class) if raw_class in class_names else -1

        # 5. Confidence Gating
        is_confident = bool(confidence >= self.confidence_threshold)
        if is_confident:
            predicted_class = raw_class
            predicted_id = raw_id
        else:
            predicted_class = "Unknown / Low Confidence"
            predicted_id = -1

        # 6. Stability Consensus
        status = self.stabilizer.evaluate_stability(
            predicted_class=raw_class,
            is_confident=is_confident,
        )

        # 7. Latency Metrics
        latencies = LatencyMetrics(
            preprocessing_ms=prep_time * 1000.0,
            feature_extraction_ms=feat_time * 1000.0,
            inference_ms=inf_time * 1000.0,
            total_ms=total_time * 1000.0,
        )

        result = PredictionResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            predicted_class=predicted_class,
            predicted_id=predicted_id,
            raw_class=raw_class,
            raw_id=raw_id,
            confidence=float(confidence),
            is_confident=is_confident,
            status=status,
            probabilities=prob_dict,
            latency=latencies,
        )

        with self._lock:
            self._session_results.append(result)

        logger.debug(
            "Recognized '%s' (conf=%.2f%%, status=%s, latency=%.1fms)",
            result.predicted_class,
            result.confidence * 100.0,
            result.status.value,
            result.latency.total_ms,
        )

        return result

    def recognize_file(self, audio_file_path: Union[str, Path]) -> PredictionResult:
        """
        Test Mode: Evaluates a standalone WAV audio file without requiring microphone hardware.

        Args:
            audio_file_path: Path to target WAV file.

        Returns:
            PredictionResult dataclass.
        """
        file_path = Path(audio_file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio test file not found at: {file_path}")

        logger.info("Running offline test-file recognition on: %s", file_path)

        try:
            # Load with librosa
            waveform, sr = librosa.load(file_path, sr=None, mono=False)
        except Exception as exc:
            raise RealtimeInferenceError(f"Failed to load audio file '{file_path}': {exc}") from exc

        return self.recognize_window(waveform, orig_sr=sr)

    def start_streaming(
        self,
        callback: Optional[Callable[[PredictionResult], None]] = None,
        hop_duration_sec: Optional[float] = None,
    ) -> None:
        """
        Starts non-blocking continuous microphone sound recognition in a dedicated thread.

        Args:
            callback: Optional user function invoked whenever an audio window is processed.
            hop_duration_sec: Interval in seconds between successive recognition windows.
        """
        if self._streaming_active:
            logger.warning("Streaming recognition is already running.")
            return

        hop_sec = float(
            hop_duration_sec
            or (self.config.hop_duration_sec if self.config else 1.0)
        )

        self._stop_event.clear()
        self.stabilizer.reset()
        self.audio_capture.start()

        def _loop():
            logger.info("Real-time recognition loop active (hop_interval=%.2fs)...", hop_sec)
            self._streaming_active = True

            # Initial warm-up sleep to populate audio buffer
            time.sleep(min(1.0, hop_sec))

            while not self._stop_event.is_set():
                loop_start = time.perf_counter()

                try:
                    # Retrieve the latest 4.0s rolling buffer snapshot
                    window = self.audio_capture.get_latest_window()

                    # Check for non-zero audio content
                    if np.abs(window).max() > 1e-4:
                        res = self.recognize_window(window, orig_sr=self.audio_capture.sample_rate)
                        if callback:
                            try:
                                callback(res)
                            except Exception as cb_exc:
                                logger.warning("Exception inside user recognition callback: %s", cb_exc)

                except Exception as exc:
                    logger.error("Error during real-time recognition window iteration: %s", exc)

                # Maintain regular hop interval
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.05, hop_sec - elapsed)
                self._stop_event.wait(timeout=sleep_time)

            self._streaming_active = False
            logger.info("Real-time recognition streaming loop terminated.")

        self._streaming_thread = threading.Thread(
            target=_loop,
            name="RealtimeRecognizerWorker",
            daemon=True,
        )
        self._streaming_thread.start()

    def stop_streaming(self) -> None:
        """Stops the streaming recognition thread and halts microphone capture."""
        if not self._streaming_active and self._streaming_thread is None:
            return

        logger.info("Stopping real-time recognition stream...")
        self._stop_event.set()

        if self._streaming_thread is not None and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=3.0)
            self._streaming_thread = None

        self._streaming_active = False

        if self._audio_capture is not None:
            self._audio_capture.stop()

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Calculates session summary metrics and average latencies across all processed windows.

        Returns:
            Dictionary with counts and average latencies.
        """
        with self._lock:
            records = list(self._session_results)

        total = len(records)
        if total == 0:
            return {
                "total_windows_processed": 0,
                "confident_predictions": 0,
                "low_confidence_predictions": 0,
                "confirmed_alerts": 0,
                "tentative_alerts": 0,
                "average_preprocessing_ms": 0.0,
                "average_feature_extraction_ms": 0.0,
                "average_inference_ms": 0.0,
                "average_total_latency_ms": 0.0,
            }

        confident_count = sum(1 for r in records if r.is_confident)
        low_conf_count = total - confident_count
        confirmed_count = sum(1 for r in records if r.status == PredictionStatus.CONFIRMED)
        tentative_count = sum(1 for r in records if r.status == PredictionStatus.TENTATIVE)

        avg_prep = float(np.mean([r.latency.preprocessing_ms for r in records]))
        avg_feat = float(np.mean([r.latency.feature_extraction_ms for r in records]))
        avg_inf = float(np.mean([r.latency.inference_ms for r in records]))
        avg_total = float(np.mean([r.latency.total_ms for r in records]))

        return {
            "total_windows_processed": total,
            "confident_predictions": confident_count,
            "low_confidence_predictions": low_conf_count,
            "confirmed_alerts": confirmed_count,
            "tentative_alerts": tentative_count,
            "average_preprocessing_ms": round(avg_prep, 2),
            "average_feature_extraction_ms": round(avg_feat, 2),
            "average_inference_ms": round(avg_inf, 2),
            "average_total_latency_ms": round(avg_total, 2),
        }

    def clear_session(self) -> None:
        """Clears accumulated session telemetry history."""
        with self._lock:
            self._session_results.clear()
        self.stabilizer.reset()
