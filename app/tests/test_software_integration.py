"""
Automated Integration and Verification Test Suite for Phases 1–8 Prototype.

Verifies:
1. Full System Status & Diagnostics (Hardware: Not Connected).
2. Mode Management API (HOME, ROAD, OFFICE, validation).
3. Test Audio Pipeline (Phase 3 Preprocess -> Phase 4 Features -> Phase 5 CNN -> Phase 7 Inference -> Phase 8 Context Decision).
4. Demo / Simulation Mode (Phase 8 Context Engine gating and priority).
5. All 7 Mandatory Verification Scenarios.
6. Alert Decision History Log.
7. FastAPI HTTP Endpoints & Web Dashboard Assets.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from config import Config, settings
from main import app
from app.context.enums import EnvironmentMode, PriorityLevel
from app.services.integration_service import SoftwareIntegrationService, integration_service


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def fresh_service():
    """Provides a fresh, isolated SoftwareIntegrationService."""
    return SoftwareIntegrationService()


# =====================================================================
# 1. System Diagnostics & Hardware-Independent Status
# =====================================================================

class TestSystemDiagnostics:
    """Verifies system diagnostics explicitly state hardware is NOT connected."""

    def test_system_status_structure(self, fresh_service):
        status = fresh_service.get_system_status()
        assert status["status"] == "online"
        assert status["version"] == "0.8.0"
        assert "subsystems" in status

        # Verify AI Model
        assert status["subsystems"]["ai_model"]["status"] == "Loaded"
        assert "2D CNN" in status["subsystems"]["ai_model"]["architecture"]

        # Verify Inference Engine
        assert status["subsystems"]["inference_engine"]["status"] == "Ready"

        # Verify Context Engine
        assert status["subsystems"]["context_engine"]["status"] == "Ready"

        # Verify Hardware is NOT connected
        hw = status["subsystems"]["hardware"]
        assert hw["status"] == "Not Connected"
        assert hw["ble_connected"] is False
        assert "Software Prototype" in hw["detail"]

    def test_status_endpoint_via_api(self, client):
        res = client.get("/api/v1/system/status")
        assert res.status_code == 200
        data = res.json()
        assert data["subsystems"]["hardware"]["status"] == "Not Connected"
        assert "ambulance" in data["supported_sounds"]


# =====================================================================
# 2. Operating Mode Management API
# =====================================================================

class TestModeManagementAPI:
    """Verifies mode transitions across HOME, ROAD, OFFICE."""

    def test_get_and_set_mode_via_api(self, client):
        # Set to ROAD
        res_set = client.post("/api/v1/mode", json={"mode": "ROAD"})
        assert res_set.status_code == 200
        assert res_set.json()["mode"] == "ROAD"

        # Verify GET reflects ROAD
        res_get = client.get("/api/v1/mode")
        assert res_get.status_code == 200
        assert res_get.json()["mode"] == "ROAD"

        # Set to OFFICE
        res_office = client.post("/api/v1/mode", json={"mode": "OFFICE"})
        assert res_office.status_code == 200
        assert res_office.json()["mode"] == "OFFICE"

        # Set back to HOME
        res_home = client.post("/api/v1/mode", json={"mode": "HOME"})
        assert res_home.status_code == 200
        assert res_home.json()["mode"] == "HOME"

    def test_invalid_mode_rejected_by_api(self, client):
        res = client.post("/api/v1/mode", json={"mode": "GYM"})
        assert res.status_code == 400
        assert "Invalid environment mode" in res.json()["detail"]


# =====================================================================
# 3. Test Audio File Execution (Phase 3 -> 4 -> 5 -> 7 -> 8)
# =====================================================================

class TestAudioFileExecution:
    """Verifies end-to-end execution of real WAV files through all 8 phases."""

    def test_list_test_audio_samples(self, client):
        res = client.get("/api/v1/test-audio/samples")
        assert res.status_code == 200
        samples = res.json()
        assert len(samples) >= 5
        classes_found = {s["sound_class"] for s in samples}
        assert "ambulance" in classes_found
        assert "car_horn" in classes_found

    def test_evaluate_sample_file_through_full_pipeline(self, client):
        """Processes real WAV file through Phase 3 -> 4 -> 5 -> 7 -> 8."""
        sample_path = "dataset/test_audio/ambulance.wav"
        res = client.post(
            "/api/v1/test-audio/evaluate",
            json={"file_path": sample_path, "mode": "ROAD"},
        )
        assert res.status_code == 200
        data = res.json()

        # Check required fields
        assert "sound" in data
        assert "confidence" in data
        assert data["mode"] == "ROAD"
        assert "priority" in data
        assert "alert_required" in data
        assert "reason" in data
        assert "latency" in data
        assert data["source"] == "file:ambulance.wav"

        # Latency breakdown exists
        assert data["latency"]["total_ms"] > 0

    def test_missing_audio_file_returns_404(self, client):
        res = client.post(
            "/api/v1/test-audio/evaluate",
            json={"file_path": "dataset/test_audio/non_existent.wav"},
        )
        assert res.status_code == 404


# =====================================================================
# 4. Demo / Simulation Mode Tests
# =====================================================================

class TestDemoSimulation:
    """Verifies simulation mode routes through the actual Phase 8 Decision Engine."""

    def test_demo_simulation_high_confidence(self, client):
        res = client.post(
            "/api/v1/demo/simulate",
            json={"sound": "car_horn", "confidence": 0.94, "mode": "ROAD"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["sound"] == "car_horn"
        assert data["priority"] == "HIGH"
        assert data["alert_required"] is True
        assert "Car Horn has HIGH priority in ROAD mode" in data["reason"]

    def test_demo_simulation_low_confidence_gated(self, client):
        res = client.post(
            "/api/v1/demo/simulate",
            json={"sound": "car_horn", "confidence": 0.45, "mode": "ROAD"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["sound"] == "car_horn"
        assert data["priority"] == "IGNORE"
        assert data["alert_required"] is False
        assert "below the configured threshold" in data["reason"]

    def test_demo_unknown_sound_rejected(self, client):
        res = client.post(
            "/api/v1/demo/simulate",
            json={"sound": "helicopter", "confidence": 0.90},
        )
        assert res.status_code == 400


# =====================================================================
# 5. The 7 Mandatory Verification Scenarios (Step 12)
# =====================================================================

class TestSevenScenarios:
    """Verifies the seven required integration scenarios."""

    def test_run_all_seven_scenarios_endpoint(self, client):
        res = client.get("/api/v1/scenarios/run")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 7
        assert data["all_passed"] is True

        scenarios = {s["scenario_id"]: s for s in data["scenarios"]}

        # Scenario 1: Home + Doorbell -> HIGH, Alert = YES
        assert scenarios[1]["sound"] == "doorbell"
        assert scenarios[1]["mode"] == "HOME"
        assert scenarios[1]["actual_priority"] == "HIGH"
        assert scenarios[1]["actual_alert"] is True
        assert scenarios[1]["status"] == "PASS"

        # Scenario 2: Road + Car Horn -> HIGH, Alert = YES
        assert scenarios[2]["sound"] == "car_horn"
        assert scenarios[2]["mode"] == "ROAD"
        assert scenarios[2]["actual_priority"] == "HIGH"
        assert scenarios[2]["actual_alert"] is True
        assert scenarios[2]["status"] == "PASS"

        # Scenario 3: Office + Car Horn -> LOW, Alert = NO
        assert scenarios[3]["sound"] == "car_horn"
        assert scenarios[3]["mode"] == "OFFICE"
        assert scenarios[3]["actual_priority"] == "LOW"
        assert scenarios[3]["actual_alert"] is False
        assert scenarios[3]["status"] == "PASS"

        # Scenario 4: Home + Dog Bark -> MEDIUM, Alert = YES
        assert scenarios[4]["sound"] == "dog_bark"
        assert scenarios[4]["mode"] == "HOME"
        assert scenarios[4]["actual_priority"] == "MEDIUM"
        assert scenarios[4]["actual_alert"] is True
        assert scenarios[4]["status"] == "PASS"

        # Scenario 5: Road + Dog Bark -> LOW, Alert = NO
        assert scenarios[5]["sound"] == "dog_bark"
        assert scenarios[5]["mode"] == "ROAD"
        assert scenarios[5]["actual_priority"] == "LOW"
        assert scenarios[5]["actual_alert"] is False
        assert scenarios[5]["status"] == "PASS"

        # Scenario 6: Home/Road/Office + Fire Alarm -> HIGH, Alert = YES
        assert scenarios[6]["sound"] == "fire_alarm"
        assert scenarios[6]["actual_priority"] == "HIGH"
        assert scenarios[6]["actual_alert"] is True
        assert scenarios[6]["status"] == "PASS"

        # Scenario 7: Road + Ambulance -> HIGH, Alert = YES
        assert scenarios[7]["sound"] == "ambulance"
        assert scenarios[7]["mode"] == "ROAD"
        assert scenarios[7]["actual_priority"] == "HIGH"
        assert scenarios[7]["actual_alert"] is True
        assert scenarios[7]["status"] == "PASS"


# =====================================================================
# 6. Alert History & Web Dashboard Assets
# =====================================================================

class TestDashboardAndHistory:
    """Verifies frontend assets serving and alert history management."""

    def test_alert_history_recording_and_clearing(self, client):
        # Generate two simulated alerts
        client.post("/api/v1/demo/simulate", json={"sound": "fire_alarm", "confidence": 0.95})
        client.post("/api/v1/demo/simulate", json={"sound": "ambulance", "confidence": 0.92})

        # Check history
        res_hist = client.get("/api/v1/alerts/history?limit=10")
        assert res_hist.status_code == 200
        alerts = res_hist.json()["alerts"]
        assert len(alerts) >= 2
        assert any(a["sound"] == "fire_alarm" for a in alerts)

        # Clear history
        res_clear = client.post("/api/v1/alerts/clear")
        assert res_clear.status_code == 200
        assert res_clear.json()["count"] == 0

    def test_dashboard_html_served(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "Smart Haptic Alert System" in res.text
        assert "Software-Only Prototype" in res.text
        assert "Hardware: Not Required" in res.text

    def test_static_assets_served(self, client):
        res_css = client.get("/static/style.css")
        assert res_css.status_code == 200
        assert "--bg-dark" in res_css.text

        res_js = client.get("/static/app.js")
        assert res_js.status_code == 200
        assert "fetchSystemStatus" in res_js.text
