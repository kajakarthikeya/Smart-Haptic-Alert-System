"""
FastAPI Delivery Layer Routers & Endpoint Specifications.

Provides endpoints for:
1. System diagnostics & status (model, inference, context, hardware: not connected).
2. Mode management (HOME, ROAD, OFFICE).
3. Test Audio pipeline execution (Phase 3 -> 4 -> 5 -> 7 -> 8).
4. Demo sound simulation (Phase 8 context engine).
5. Real-time microphone recognition controls.
6. Alert history retrieval & clearing.
7. Automated verification scenarios (7 benchmark scenarios).
"""

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.context.enums import EnvironmentMode
from app.context.exceptions import ContextError, InvalidModeError, UnknownSoundError
from app.services.integration_service import integration_service
from app.utils.logger import get_logger
from config import Config

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Smart Haptic Alert System API"])


# ---------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------

class ModeChangeRequest(BaseModel):
    """Payload model for mode change request."""
    mode: str = Field(..., description="Target mode: HOME, ROAD, or OFFICE", json_schema_extra={"example": "ROAD"})


class DemoSimulationRequest(BaseModel):
    """Payload model for simulating target sound via Phase 8 Context Engine."""
    sound: str = Field(..., description="Target sound name (ambulance, car_horn, fire_alarm, doorbell, dog_bark)")
    confidence: float = Field(default=0.92, ge=0.0, le=1.0, description="Inference confidence score")
    mode: Optional[str] = Field(default=None, description="Optional mode override")


class TestAudioEvaluateRequest(BaseModel):
    """Payload model for evaluating an existing test audio WAV file."""
    file_path: str = Field(..., description="Path to audio file (e.g. dataset/test_audio/ambulance.wav)")
    mode: Optional[str] = Field(default=None, description="Optional mode override")


class ManualAlertRequest(BaseModel):
    """Legacy manual alert request model."""
    sound_label: str = Field(..., description="Label of sound event", json_schema_extra={"example": "doorbell"})
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence score")


# ---------------------------------------------------------------------
# System Status & Mode Endpoints
# ---------------------------------------------------------------------

@router.get("/health", summary="System Health Check")
async def health_check() -> Dict[str, Any]:
    """Returns application health status and diagnostic state."""
    status = integration_service.get_system_status()
    return {"status": "online", "mode": status["current_mode"], "version": status["version"]}


@router.get("/system/status", summary="Full System Diagnostic Status")
async def get_system_status() -> Dict[str, Any]:
    """Retrieves full diagnostic status for AI model, inference, context engine, and hardware."""
    return integration_service.get_system_status()


@router.get("/mode", summary="Get Current Environment Mode")
async def get_mode() -> Dict[str, Any]:
    """Retrieves active operating mode profile details."""
    return {
        "mode": integration_service.mode_manager.current_mode.value,
        "supported_modes": [m.value for m in EnvironmentMode],
    }


@router.post("/mode", summary="Switch Operating Environment Mode")
async def switch_mode(payload: ModeChangeRequest) -> Dict[str, str]:
    """Switches active operating mode in Phase 8 ModeManager."""
    try:
        return integration_service.set_mode(payload.mode)
    except InvalidModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------
# Test Audio Mode Endpoints
# ---------------------------------------------------------------------

@router.get("/test-audio/samples", summary="List Available Test Audio Samples")
async def list_test_audio_samples() -> List[Dict[str, Any]]:
    """Returns list of test WAV audio files available in dataset/test_audio/."""
    return integration_service.list_test_audio_samples()


@router.post("/test-audio/evaluate", summary="Run Full Inference Pipeline on Audio File")
async def evaluate_test_audio(payload: TestAudioEvaluateRequest) -> Dict[str, Any]:
    """
    Executes full software-only pipeline on an audio file:
    Phase 3 Preprocessing -> Phase 4 Features -> Phase 5 Model -> Phase 7 Inference -> Phase 8 Decision.
    """
    try:
        return integration_service.evaluate_test_audio_file(
            file_path=payload.file_path,
            override_mode=payload.mode,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Test audio evaluation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


@router.post("/test-audio/upload", summary="Upload and Evaluate Custom WAV File")
async def upload_and_evaluate_audio(
    file: UploadFile = File(...),
    mode: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Uploads a custom WAV file and evaluates it through the full AI and context pipeline."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400, detail="Only .wav, .mp3, or .flac files are supported.")

    upload_dir = Config.paths.base_dir / "dataset" / "test_audio" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / file.filename

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return integration_service.evaluate_test_audio_file(
            file_path=destination,
            override_mode=mode,
        )
    except Exception as exc:
        logger.error("Uploaded file evaluation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")


# ---------------------------------------------------------------------
# Demo Mode & Simulation Endpoints
# ---------------------------------------------------------------------

@router.post("/demo/simulate", summary="Simulate Target Sound in Context Engine")
async def simulate_demo_sound(payload: DemoSimulationRequest) -> Dict[str, Any]:
    """Simulates a target environmental sound through the actual Phase 8 Decision Engine."""
    try:
        return integration_service.simulate_demo_sound(
            sound=payload.sound,
            confidence=payload.confidence,
            override_mode=payload.mode,
        )
    except (UnknownSoundError, InvalidModeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")


# ---------------------------------------------------------------------
# Real-Time Microphone Recognition Endpoints
# ---------------------------------------------------------------------

@router.post("/recognition/start", summary="Start Live Microphone Recognition")
async def start_recognition() -> Dict[str, Any]:
    """Starts live streaming recognition thread using Phase 7 recognizer."""
    return integration_service.start_realtime_recognition()


@router.post("/recognition/stop", summary="Stop Live Microphone Recognition")
async def stop_recognition() -> Dict[str, Any]:
    """Stops live streaming recognition thread."""
    return integration_service.stop_realtime_recognition()


@router.get("/recognition/latest", summary="Get Latest Detection Decision")
async def get_latest_decision() -> Dict[str, Any]:
    """Retrieves the most recent sound classification and context decision."""
    status = integration_service.get_system_status()
    return {
        "is_recognizing": status["is_recognizing"],
        "latest_decision": status["latest_decision"],
        "current_mode": status["current_mode"],
    }


# ---------------------------------------------------------------------
# Alerts History & Scenarios Endpoints
# ---------------------------------------------------------------------

@router.get("/alerts/history", summary="Get Alert History")
async def get_alert_history(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
    """Retrieves recent decision records."""
    history = integration_service.get_alert_history(limit)
    return {"count": len(history), "alerts": history}


@router.post("/alerts/clear", summary="Clear Alert History")
async def clear_alert_history() -> Dict[str, Any]:
    """Clears alert history log."""
    return integration_service.clear_alert_history()


@router.get("/scenarios/run", summary="Execute 7 Verification Scenarios")
async def run_verification_scenarios() -> Dict[str, Any]:
    """Runs the 7 mandatory verification test scenarios against Phase 8 logic."""
    results = integration_service.run_all_seven_scenarios()
    all_passed = all(r["status"] == "PASS" for r in results)
    return {
        "total": len(results),
        "all_passed": all_passed,
        "scenarios": results,
    }


# ---------------------------------------------------------------------
# Legacy Compatibility
# ---------------------------------------------------------------------

@router.post("/alerts/trigger", summary="Legacy Trigger Manual Alert")
async def trigger_legacy_alert(payload: ManualAlertRequest) -> Dict[str, Any]:
    """Legacy manual alert endpoint routing to integration service simulation."""
    record = integration_service.simulate_demo_sound(
        sound=payload.sound_label,
        confidence=payload.confidence,
    )
    return {"status": "success" if record["alert_required"] else "ignored", "alert": record}
