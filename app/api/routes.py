"""FastAPI Delivery Layer Routers & Endpoint Specifications."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict

from app.controllers.alert_controller import AlertController
from app.controllers.mode_controller import ModeController

router = APIRouter(prefix="/api/v1", tags=["Smart Haptic Alert System API"])

# Initialize controller singletons for route handling
_mode_controller = ModeController()
_alert_controller = AlertController()


class ModeChangeRequest(BaseModel):
    """Payload model for mode change request."""
    mode: str = Field(..., description="Target mode: HOME, ROAD, or OFFICE", example="ROAD")


class ManualAlertRequest(BaseModel):
    """Payload model for manual alert trigger."""
    sound_label: str = Field(..., description="Label of sound event", example="doorbell")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence score")


@router.get("/health", summary="System Health Check")
async def health_check() -> Dict[str, str]:
    """Returns application health status and current mode."""
    mode_info = _mode_controller.get_current_mode()
    return {"status": "online", "mode": mode_info["mode"]}


@router.get("/mode", summary="Get Current Environment Mode")
async def get_mode() -> Dict[str, str]:
    """Retrieves active operating profile details."""
    return _mode_controller.get_current_mode()


@router.post("/mode", summary="Switch Operating Environment Mode")
async def switch_mode(payload: ModeChangeRequest) -> Dict[str, str]:
    """Switches active operating mode (HOME, ROAD, OFFICE)."""
    result = _mode_controller.switch_mode(payload.mode)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.post("/alerts/trigger", summary="Trigger Manual Sound Alert")
async def trigger_alert(payload: ManualAlertRequest) -> Dict[str, Any]:
    """Manually evaluates a sound event and dispatches alert if authorized by active mode."""
    return await _alert_controller.trigger_manual_alert(payload.sound_label, payload.confidence)


@router.get("/alerts/history", summary="Get Alert History")
async def get_alert_history(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
    """Retrieves recent alert history entries."""
    return _alert_controller.get_history(limit)
