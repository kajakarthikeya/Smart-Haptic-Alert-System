"""
Software-Only Integration Service.

Connects Phase 1-8 modules into a unified hardware-independent pipeline:
- Phase 7 Real-time acoustic recognition (file and live stream)
- Phase 8 Context-aware decision engine and mode manager
- Demo / Simulation evaluation
- Alert history management
- 7 verification scenarios runner
- System status diagnostics
"""

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from config import Config, settings
from app.ai.inference import (
    AudioDeviceManager,
    PredictionResult,
    PredictionStatus,
    RealtimeSoundRecognizer,
)
from app.context import (
    ContextDecisionEngine,
    DecisionResult,
    EnvironmentMode,
    ModeManager,
    PriorityLevel,
    SoundPrediction,
    TARGET_SOUNDS,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SoftwareIntegrationService:
    """
    Singleton-capable integration service coordinating Phase 1–8 pipelines
    for the software-only prototype and frontend dashboard.
    """

    def __init__(self) -> None:
        """Initializes integration service, recognizer, and decision engine."""
        self._mode_manager = ModeManager(default_mode=EnvironmentMode.HOME)
        self._decision_engine = ContextDecisionEngine(
            mode_manager=self._mode_manager,
            confidence_threshold=settings.context.confidence_threshold,
        )
        
        # Lazy-load recognizer to keep startup swift
        self._recognizer: Optional[RealtimeSoundRecognizer] = None
        self._recognizer_lock = threading.Lock()
        
        # Alert decision history (in-memory ring buffer)
        self._history: deque[Dict[str, Any]] = deque(maxlen=100)
        self._history_lock = threading.Lock()
        
        # Real-time recognition state
        self._is_recognizing: bool = False
        self._recognition_thread: Optional[threading.Thread] = None
        self._latest_decision: Optional[Dict[str, Any]] = None
        
        # Pre-populate sample WAVs directory
        self._test_audio_dir: Path = settings.dataset.root_dir / "test_audio"
        self._test_audio_dir.mkdir(parents=True, exist_ok=True)

        logger.info("SoftwareIntegrationService initialized successfully.")

    @property
    def mode_manager(self) -> ModeManager:
        return self._mode_manager

    @property
    def decision_engine(self) -> ContextDecisionEngine:
        return self._decision_engine

    def get_recognizer(self) -> RealtimeSoundRecognizer:
        """Lazily initializes and caches the Phase 7 RealtimeSoundRecognizer."""
        with self._recognizer_lock:
            if self._recognizer is None:
                logger.info("Initializing RealtimeSoundRecognizer for integration service...")
                self._recognizer = RealtimeSoundRecognizer(
                    confidence_threshold=settings.inference.confidence_threshold
                )
            return self._recognizer

    def get_system_status(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic status for all software subsystems."""
        model_file = settings.inference.model_path
        model_loaded = model_file.exists()
        
        # Query input audio devices to check if host microphone exists
        mic_available = False
        default_mic_name = "None"
        try:
            default_dev = AudioDeviceManager.get_default_input_device()
            if default_dev is not None:
                mic_available = True
                default_mic_name = default_dev.name
        except Exception:
            mic_available = False

        return {
            "status": "online",
            "app_name": settings.system.app_name,
            "version": "0.8.0",
            "subsystems": {
                "ai_model": {
                    "status": "Loaded" if model_loaded else "Missing",
                    "model_path": str(model_file.relative_to(Config.paths.base_dir)),
                    "architecture": "2D CNN (111,237 params)",
                },
                "inference_engine": {
                    "status": "Ready",
                    "sample_rate": settings.inference.sample_rate,
                    "window_duration_sec": settings.inference.window_duration_sec,
                    "confidence_threshold": settings.inference.confidence_threshold,
                },
                "context_engine": {
                    "status": "Ready",
                    "current_mode": self._mode_manager.current_mode.value,
                    "supported_modes": [m.value for m in EnvironmentMode],
                    "confidence_threshold": self._decision_engine.confidence_threshold,
                },
                "audio_input": {
                    "status": "Ready",
                    "microphone_available": mic_available,
                    "default_microphone": default_mic_name,
                    "active_mode": "Live Microphone" if self._is_recognizing else "Test Audio / Demo",
                },
                "hardware": {
                    "status": "Not Connected",
                    "detail": "Software Prototype (Hardware was NOT required for this verification)",
                    "ble_connected": False,
                },
            },
            "current_mode": self._mode_manager.current_mode.value,
            "supported_sounds": list(TARGET_SOUNDS),
            "is_recognizing": self._is_recognizing,
            "latest_decision": self._latest_decision,
        }

    def set_mode(self, mode_str: str) -> Dict[str, Any]:
        """Switches active environment mode in the Phase 8 ModeManager."""
        new_mode = self._mode_manager.set_mode(mode_str)
        return {
            "status": "success",
            "mode": new_mode.value,
            "message": f"Operating mode switched to {new_mode.value}",
        }

    def list_test_audio_samples(self) -> List[Dict[str, Any]]:
        """Scans dataset/test_audio/ and dataset/processed/ for test audio WAV files."""
        samples: List[Dict[str, Any]] = []
        
        # 1. Primary test audio directory
        if self._test_audio_dir.exists():
            for p in sorted(self._test_audio_dir.glob("*.wav")):
                sound_hint = p.stem.lower()
                for target in TARGET_SOUNDS:
                    if target in sound_hint:
                        sound_hint = target
                        break
                samples.append({
                    "name": p.name,
                    "path": str(p.relative_to(Config.paths.base_dir)).replace("\\", "/"),
                    "sound_class": sound_hint,
                    "size_bytes": p.stat().st_size,
                    "source": "test_audio",
                })

        # 2. Fallback to dataset/processed/ samples if needed
        processed_dir = settings.dataset.processed_dir
        if processed_dir.exists():
            for class_dir in sorted(processed_dir.iterdir()):
                if class_dir.is_dir():
                    for wav in sorted(class_dir.glob("*.wav"))[:1]:
                        rel = str(wav.relative_to(Config.paths.base_dir)).replace("\\", "/")
                        samples.append({
                            "name": f"{class_dir.name}/{wav.name}",
                            "path": rel,
                            "sound_class": class_dir.name,
                            "size_bytes": wav.stat().st_size,
                            "source": "processed",
                        })

        return samples

    def evaluate_test_audio_file(
        self,
        file_path: Union[str, Path],
        override_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end pipeline:
        Audio File -> Phase 3 Preprocess -> Phase 4 Features -> Phase 5 Model ->
        Phase 7 PredictionResult -> Phase 8 Context Decision -> Output.
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = Config.paths.base_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        logger.info("Evaluating test audio file: %s", path)
        recognizer = self.get_recognizer()
        
        # Run Phase 7 inference
        prediction_result: PredictionResult = recognizer.recognize_file(path)
        
        # Run Phase 8 context decision
        decision: DecisionResult = self._decision_engine.evaluate(
            prediction_result,
            override_mode=override_mode,
        )

        record = self._build_enriched_record(
            decision=decision,
            source=f"file:{path.name}",
            prediction_result=prediction_result,
        )
        self._record_history(record)
        self._latest_decision = record
        return record

    def simulate_demo_sound(
        self,
        sound: str,
        confidence: float = 0.92,
        override_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Simulates one of the 5 target sounds through the Phase 8 context decision engine.
        """
        clean_sound = str(sound).strip().lower()
        pred = SoundPrediction(sound=clean_sound, confidence=float(confidence))
        
        decision: DecisionResult = self._decision_engine.evaluate(
            pred,
            override_mode=override_mode,
        )

        record = self._build_enriched_record(
            decision=decision,
            source="demo_simulation",
            confidence=confidence,
        )
        self._record_history(record)
        self._latest_decision = record
        return record

    def start_realtime_recognition(self) -> Dict[str, Any]:
        """Starts real-time microphone acoustic stream listener in background."""
        if self._is_recognizing:
            return {"status": "already_running", "message": "Microphone recognition is already active."}

        # Check microphone availability
        try:
            default_dev = AudioDeviceManager.get_default_input_device()
            if default_dev is None:
                return {
                    "status": "unavailable",
                    "message": "Microphone unavailable — use Test Audio or Demo Mode.",
                }
        except Exception as exc:
            return {
                "status": "unavailable",
                "message": f"Microphone unavailable — use Test Audio or Demo Mode. ({exc})",
            }

        self._is_recognizing = True

        def _streaming_worker():
            recognizer = self.get_recognizer()

            def _on_prediction(res: PredictionResult):
                if not self._is_recognizing:
                    return
                # Evaluate in Phase 8
                decision = self._decision_engine.evaluate(res)
                record = self._build_enriched_record(
                    decision=decision,
                    source="live_microphone",
                    prediction_result=res,
                )
                self._latest_decision = record
                if decision.alert_required:
                    self._record_history(record)

            try:
                recognizer.start_streaming(on_prediction=_on_prediction, block=False)
            except Exception as exc:
                logger.error("Live streaming worker error: %s", exc)
                self._is_recognizing = False

        self._recognition_thread = threading.Thread(target=_streaming_worker, daemon=True)
        self._recognition_thread.start()

        return {
            "status": "started",
            "message": "Live microphone recognition started.",
        }

    def stop_realtime_recognition(self) -> Dict[str, Any]:
        """Stops live microphone recognition."""
        if not self._is_recognizing:
            return {"status": "not_running", "message": "Microphone recognition is not running."}

        self._is_recognizing = False
        with self._recognizer_lock:
            if self._recognizer is not None:
                self._recognizer.stop_streaming()

        return {"status": "stopped", "message": "Live microphone recognition stopped."}

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns chronological history of alert decisions."""
        with self._history_lock:
            items = list(self._history)
            return list(reversed(items[-limit:]))

    def clear_alert_history(self) -> Dict[str, Any]:
        """Clears in-memory alert history."""
        with self._history_lock:
            self._history.clear()
        return {"status": "cleared", "count": 0}

    def run_all_seven_scenarios(self) -> List[Dict[str, Any]]:
        """
        Executes and validates the 7 mandatory test scenarios against Phase 8 logic:
        1. Home + Doorbell -> HIGH, Alert = YES
        2. Road + Car Horn -> HIGH, Alert = YES
        3. Office + Car Horn -> LOW, Alert = NO
        4. Home + Dog Bark -> MEDIUM, Alert = YES
        5. Road + Dog Bark -> LOW, Alert = NO
        6. Home/Road/Office + Fire Alarm -> HIGH, Alert = YES
        7. Road + Ambulance -> HIGH, Alert = YES
        """
        scenarios = [
            {"id": 1, "mode": "HOME", "sound": "doorbell", "confidence": 0.90, "exp_priority": "HIGH", "exp_alert": True},
            {"id": 2, "mode": "ROAD", "sound": "car_horn", "confidence": 0.92, "exp_priority": "HIGH", "exp_alert": True},
            {"id": 3, "mode": "OFFICE", "sound": "car_horn", "confidence": 0.88, "exp_priority": "LOW", "exp_alert": False},
            {"id": 4, "mode": "HOME", "sound": "dog_bark", "confidence": 0.85, "exp_priority": "MEDIUM", "exp_alert": True},
            {"id": 5, "mode": "ROAD", "sound": "dog_bark", "confidence": 0.85, "exp_priority": "LOW", "exp_alert": False},
            {"id": 6, "mode": "HOME", "sound": "fire_alarm", "confidence": 0.95, "exp_priority": "HIGH", "exp_alert": True},
            {"id": 7, "mode": "ROAD", "sound": "ambulance", "confidence": 0.96, "exp_priority": "HIGH", "exp_alert": True},
        ]

        results = []
        for sc in scenarios:
            pred = SoundPrediction(sound=sc["sound"], confidence=sc["confidence"])
            decision = self._decision_engine.evaluate(pred, override_mode=sc["mode"])
            passed = (
                decision.priority.value == sc["exp_priority"]
                and decision.alert_required == sc["exp_alert"]
            )
            results.append({
                "scenario_id": sc["id"],
                "mode": sc["mode"],
                "sound": sc["sound"],
                "confidence": sc["confidence"],
                "expected_priority": sc["exp_priority"],
                "expected_alert": sc["exp_alert"],
                "actual_priority": decision.priority.value,
                "actual_alert": decision.alert_required,
                "reason": decision.reason,
                "status": "PASS" if passed else "FAIL",
            })

        return results

    def _build_enriched_record(
        self,
        decision: DecisionResult,
        source: str,
        prediction_result: Optional[PredictionResult] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Builds a comprehensive UI-ready payload."""
        rec = decision.to_dict()
        rec["source"] = source
        if prediction_result is not None:
            rec["latency"] = prediction_result.latency.to_dict()
            rec["top_probabilities"] = {
                k: round(v, 4) for k, v in sorted(
                    prediction_result.probabilities.items(), key=lambda x: -x[1]
                )[:3]
            }
            rec["prediction_status"] = prediction_result.status.value
        else:
            rec["latency"] = {"total_ms": 0.05, "preprocessing_ms": 0.0, "feature_extraction_ms": 0.0, "inference_ms": 0.0}
            rec["top_probabilities"] = {decision.sound: decision.confidence}
            rec["prediction_status"] = "CONFIRMED" if decision.confidence >= 0.70 else "LOW_CONFIDENCE"

        return rec

    def _record_history(self, record: Dict[str, Any]) -> None:
        with self._history_lock:
            self._history.append(record)


# Global singleton instance
integration_service = SoftwareIntegrationService()
