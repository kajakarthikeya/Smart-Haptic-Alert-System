"""Alert Controller for managing alert dispatches and history requests."""

from typing import Dict, List, Optional
from app.services.alert_service import AlertService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AlertController:
    """Controller handling alert dispatch commands from external API clients or UI triggers."""

    def __init__(self, alert_service: Optional[AlertService] = None) -> None:
        """Initializes controller with AlertService dependency.

        Args:
            alert_service: Core AlertService instance.
        """
        self._alert_service = alert_service or AlertService()
        logger.info("AlertController initialized.")

    async def trigger_manual_alert(self, sound_label: str, confidence: float = 1.0) -> Dict:
        """Manually triggers an alert evaluation request.

        Args:
            sound_label: Sound identifier label.
            confidence: Confidence score.

        Returns:
            Status dictionary containing result or failure message.
        """
        logger.info(f"Manual alert trigger requested for sound: '{sound_label}'")
        record = await self._alert_service.handle_sound_event(sound_label, confidence)
        if record:
            return {"status": "success", "alert": record}
        return {"status": "ignored", "message": f"Sound '{sound_label}' ignored under active mode profile."}

    def get_history(self, limit: int = 20) -> Dict[str, List[Dict]]:
        """Retrieves recent alert execution history.

        Args:
            limit: Maximum count.

        Returns:
            Dictionary containing list of alerts.
        """
        history = self._alert_service.get_alert_history(limit)
        return {"count": len(history), "alerts": history}
