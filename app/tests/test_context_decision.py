"""
Comprehensive Unit & Integration Test Suite for Phase 8 Context-Aware Decision Module.

Tests:
1. Mode Management (Default Home, transitions, invalid rejection, observer callbacks, reset).
2. Priority Rules (All 15 sound x mode combinations validated).
3. Confidence Handling (Valid 0.0 to 1.0, low-confidence gating, NaN/Inf and out-of-bounds rejection).
4. Decision Engine (Alert policies, reason strings, unknown sound handling).
5. Dynamic Configuration Testing (Matrix customization).
6. Phase 7 -> Phase 8 Integration Testing (Real PredictionResult ingestion).
7. Sub-millisecond Execution Performance.
"""

import math
import time
import pytest
from app.context.enums import EnvironmentMode, PriorityLevel
from app.context.exceptions import (
    ConfigurationError,
    InvalidConfidenceError,
    InvalidModeError,
    PriorityRuleError,
    UnknownSoundError,
)
from app.context.mode_manager import ModeManager
from app.context.models import AlertPolicy, DecisionResult, SoundPrediction
from app.context.priority_engine import PriorityEngine
from app.context.decision_engine import ContextDecisionEngine
from app.context.rules import DEFAULT_PRIORITY_MATRIX
from app.ai.inference.prediction import (
    LatencyMetrics,
    PredictionResult,
    PredictionStatus,
)


# =====================================================================
# 1. Mode Management Tests
# =====================================================================

class TestModeManagement:
    """Tests for ModeManager state tracking, transitions, and validation."""

    def test_default_mode_is_home(self):
        """1. Default mode should be HOME."""
        mgr = ModeManager()
        assert mgr.current_mode == EnvironmentMode.HOME
        assert mgr.get_mode() == EnvironmentMode.HOME

    def test_mode_transition_home_to_road(self):
        """2. Transition from HOME to ROAD."""
        mgr = ModeManager(default_mode=EnvironmentMode.HOME)
        new_mode = mgr.set_mode(EnvironmentMode.ROAD)
        assert new_mode == EnvironmentMode.ROAD
        assert mgr.current_mode == EnvironmentMode.ROAD

    def test_mode_transition_road_to_office(self):
        """3. Transition from ROAD to OFFICE."""
        mgr = ModeManager(default_mode=EnvironmentMode.ROAD)
        new_mode = mgr.set_mode("OFFICE")
        assert new_mode == EnvironmentMode.OFFICE
        assert mgr.current_mode == EnvironmentMode.OFFICE

    def test_invalid_mode_rejected(self):
        """4. Invalid mode like 'School' must be rejected with InvalidModeError."""
        mgr = ModeManager()
        with pytest.raises(InvalidModeError) as exc_info:
            mgr.set_mode("School")
        assert "Invalid environment mode" in str(exc_info.value)

    def test_mode_change_observer_notification(self):
        """Observer callback receives old and new mode transitions."""
        mgr = ModeManager()
        history = []

        def on_change(old, new):
            history.append((old, new))

        mgr.register_listener(on_change)
        mgr.set_mode(EnvironmentMode.ROAD)
        mgr.set_mode(EnvironmentMode.OFFICE)

        assert len(history) == 2
        assert history[0] == (EnvironmentMode.HOME, EnvironmentMode.ROAD)
        assert history[1] == (EnvironmentMode.ROAD, EnvironmentMode.OFFICE)

        # Unregister
        mgr.unregister_listener(on_change)
        mgr.set_mode(EnvironmentMode.HOME)
        assert len(history) == 2  # No additional callback

    def test_mode_reset_to_default(self):
        """Reset reverts mode back to initial default."""
        mgr = ModeManager(default_mode=EnvironmentMode.HOME)
        mgr.set_mode(EnvironmentMode.ROAD)
        assert mgr.current_mode == EnvironmentMode.ROAD
        mgr.reset()
        assert mgr.current_mode == EnvironmentMode.HOME


# =====================================================================
# 2. Priority Rules Matrix Tests (All 15 Sound x Mode combinations)
# =====================================================================

class TestPriorityMatrix:
    """Tests all 15 permutations of (5 target sounds x 3 operating modes)."""

    @pytest.fixture
    def priority_engine(self):
        return PriorityEngine()

    @pytest.mark.parametrize(
        "sound,mode,expected_priority",
        [
            # Ambulance in all modes -> HIGH
            ("ambulance", EnvironmentMode.HOME, PriorityLevel.HIGH),    # 5
            ("ambulance", EnvironmentMode.ROAD, PriorityLevel.HIGH),    # 6
            ("ambulance", EnvironmentMode.OFFICE, PriorityLevel.HIGH),  # 7
            # Car Horn
            ("car_horn", EnvironmentMode.HOME, PriorityLevel.MEDIUM),   # 8
            ("car_horn", EnvironmentMode.ROAD, PriorityLevel.HIGH),     # 9
            ("car_horn", EnvironmentMode.OFFICE, PriorityLevel.LOW),    # 10
            # Fire Alarm in all modes -> HIGH
            ("fire_alarm", EnvironmentMode.HOME, PriorityLevel.HIGH),   # 11a
            ("fire_alarm", EnvironmentMode.ROAD, PriorityLevel.HIGH),   # 11b
            ("fire_alarm", EnvironmentMode.OFFICE, PriorityLevel.HIGH), # 11c
            # Doorbell
            ("doorbell", EnvironmentMode.HOME, PriorityLevel.HIGH),     # 12
            ("doorbell", EnvironmentMode.ROAD, PriorityLevel.LOW),      # 13
            ("doorbell", EnvironmentMode.OFFICE, PriorityLevel.LOW),    # 14
            # Dog Bark
            ("dog_bark", EnvironmentMode.HOME, PriorityLevel.MEDIUM),   # 15
            ("dog_bark", EnvironmentMode.ROAD, PriorityLevel.LOW),      # 16
            ("dog_bark", EnvironmentMode.OFFICE, PriorityLevel.LOW),    # 17
        ],
    )
    def test_fifteen_sound_mode_combinations(self, priority_engine, sound, mode, expected_priority):
        priority = priority_engine.get_priority(sound, mode)
        assert priority == expected_priority


# =====================================================================
# 3. Confidence Gating & Safety Handling Tests
# =====================================================================

class TestConfidenceHandling:
    """Tests confidence thresholds, edge cases, and out-of-bounds rejection."""

    @pytest.fixture
    def decision_engine(self):
        mode_mgr = ModeManager(default_mode=EnvironmentMode.ROAD)
        return ContextDecisionEngine(mode_manager=mode_mgr, confidence_threshold=0.70)

    def test_high_confidence_produces_normal_decision(self, decision_engine):
        """18. High confidence prediction produces normal prioritized alert."""
        pred = SoundPrediction(sound="car_horn", confidence=0.85)
        decision = decision_engine.evaluate(pred)
        assert decision.priority == PriorityLevel.HIGH
        assert decision.alert_required is True
        assert "Car Horn has HIGH priority in ROAD mode" in decision.reason

    def test_low_confidence_suppresses_alert(self, decision_engine):
        """19. Low confidence prediction suppresses alert."""
        pred = SoundPrediction(sound="car_horn", confidence=0.45)
        decision = decision_engine.evaluate(pred)
        assert decision.priority == PriorityLevel.IGNORE
        assert decision.alert_required is False

    def test_confidence_below_threshold_reason(self, decision_engine):
        """20. Decision reason indicates confidence below threshold."""
        pred = SoundPrediction(sound="fire_alarm", confidence=0.69)
        decision = decision_engine.evaluate(pred)
        assert decision.priority == PriorityLevel.IGNORE
        assert decision.alert_required is False
        assert "below the configured threshold" in decision.reason

    def test_confidence_boundary_zero(self, decision_engine):
        """21. Confidence = 0.0 is valid input, safely filtered."""
        pred = SoundPrediction(sound="ambulance", confidence=0.0)
        decision = decision_engine.evaluate(pred)
        assert decision.priority == PriorityLevel.IGNORE
        assert decision.alert_required is False

    def test_confidence_boundary_one(self, decision_engine):
        """22. Confidence = 1.0 is handled correctly."""
        pred = SoundPrediction(sound="ambulance", confidence=1.0)
        decision = decision_engine.evaluate(pred)
        assert decision.priority == PriorityLevel.HIGH
        assert decision.alert_required is True

    def test_invalid_confidence_above_one_rejected(self):
        """23. Confidence > 1.0 raises InvalidConfidenceError."""
        with pytest.raises(InvalidConfidenceError):
            SoundPrediction(sound="ambulance", confidence=1.05)

    def test_invalid_confidence_below_zero_rejected(self):
        """24. Confidence < 0.0 raises InvalidConfidenceError."""
        with pytest.raises(InvalidConfidenceError):
            SoundPrediction(sound="ambulance", confidence=-0.1)

    def test_invalid_confidence_nan_or_inf_rejected(self):
        """NaN and Inf confidences are strictly rejected."""
        with pytest.raises(InvalidConfidenceError):
            SoundPrediction(sound="ambulance", confidence=float("nan"))
        with pytest.raises(InvalidConfidenceError):
            SoundPrediction(sound="ambulance", confidence=float("inf"))


# =====================================================================
# 4. Decision Engine Logic Tests
# =====================================================================

class TestDecisionEngine:
    """Tests sound/mode validation, alert policies, and error handling."""

    @pytest.fixture
    def engine(self):
        return ContextDecisionEngine(confidence_threshold=0.70)

    def test_correct_sound_mode_combination(self, engine):
        """25. Valid sound and mode evaluation."""
        decision = engine.evaluate(
            SoundPrediction(sound="doorbell", confidence=0.85),
            override_mode=EnvironmentMode.HOME,
        )
        assert decision.sound == "doorbell"
        assert decision.mode == EnvironmentMode.HOME

    def test_correct_priority(self, engine):
        """26. Doorbell in HOME mode has HIGH priority."""
        decision = engine.evaluate(
            SoundPrediction(sound="doorbell", confidence=0.85),
            override_mode=EnvironmentMode.HOME,
        )
        assert decision.priority == PriorityLevel.HIGH

    def test_correct_alert_required(self, engine):
        """27. HIGH priority triggers alert_required=True."""
        decision = engine.evaluate(
            SoundPrediction(sound="doorbell", confidence=0.85),
            override_mode=EnvironmentMode.HOME,
        )
        assert decision.alert_required is True

        # Doorbell in ROAD mode is LOW -> alert_required=False
        decision_road = engine.evaluate(
            SoundPrediction(sound="doorbell", confidence=0.85),
            override_mode=EnvironmentMode.ROAD,
        )
        assert decision_road.priority == PriorityLevel.LOW
        assert decision_road.alert_required is False

    def test_correct_decision_reason(self, engine):
        """28. Informative reason generated in output."""
        decision = engine.evaluate(
            SoundPrediction(sound="dog_bark", confidence=0.80),
            override_mode=EnvironmentMode.HOME,
        )
        assert "Dog Bark has MEDIUM priority in HOME mode" in decision.reason

    def test_unknown_sound_rejected(self, engine):
        """29. Unknown sound label raises UnknownSoundError."""
        with pytest.raises(UnknownSoundError) as exc_info:
            engine.evaluate(SoundPrediction(sound="cat_meow", confidence=0.90))
        assert "Unknown sound 'cat_meow'" in str(exc_info.value)

    def test_missing_rule_handled_safely(self):
        """30. Priority matrix missing a target sound raises ConfigurationError on startup."""
        incomplete_matrix = {
            EnvironmentMode.HOME: {
                "ambulance": PriorityLevel.HIGH,
                # Missing car_horn, fire_alarm, doorbell, dog_bark
            },
            EnvironmentMode.ROAD: DEFAULT_PRIORITY_MATRIX[EnvironmentMode.ROAD],
            EnvironmentMode.OFFICE: DEFAULT_PRIORITY_MATRIX[EnvironmentMode.OFFICE],
        }
        with pytest.raises(ConfigurationError):
            PriorityEngine(matrix=incomplete_matrix)


# =====================================================================
# 5. Configuration Testing (Step 17)
# =====================================================================

class TestConfigurationFlexibility:
    """Verifies that changing the priority matrix changes decision behavior dynamically."""

    def test_custom_rule_modifies_behavior_without_code_changes(self):
        """
        Modifying Road + Dog Bark from LOW to MEDIUM in a custom matrix
        should immediately reflect in the PriorityEngine and DecisionEngine.
        """
        custom_matrix = {
            EnvironmentMode.HOME: dict(DEFAULT_PRIORITY_MATRIX[EnvironmentMode.HOME]),
            EnvironmentMode.ROAD: dict(DEFAULT_PRIORITY_MATRIX[EnvironmentMode.ROAD]),
            EnvironmentMode.OFFICE: dict(DEFAULT_PRIORITY_MATRIX[EnvironmentMode.OFFICE]),
        }
        # Change Road + dog_bark to MEDIUM
        custom_matrix[EnvironmentMode.ROAD]["dog_bark"] = PriorityLevel.MEDIUM

        custom_priority_engine = PriorityEngine(matrix=custom_matrix)
        mode_mgr = ModeManager(default_mode=EnvironmentMode.ROAD)
        engine = ContextDecisionEngine(
            mode_manager=mode_mgr,
            priority_engine=custom_priority_engine,
            confidence_threshold=0.70,
        )

        decision = engine.evaluate(SoundPrediction(sound="dog_bark", confidence=0.90))
        # Now dog_bark in ROAD mode should be MEDIUM with alert=True!
        assert decision.priority == PriorityLevel.MEDIUM
        assert decision.alert_required is True
        assert "Dog Bark has MEDIUM priority in ROAD mode" in decision.reason


# =====================================================================
# 6. Phase 7 -> Phase 8 Integration Tests (Step 18)
# =====================================================================

class TestPhase7Phase8Integration:
    """Integration tests consuming Phase 7 PredictionResult objects."""

    @pytest.fixture
    def decision_engine(self):
        return ContextDecisionEngine(confidence_threshold=0.70)

    def _create_mock_phase7_result(self, sound: str, confidence: float) -> PredictionResult:
        """Helper to create a realistic Phase 7 PredictionResult."""
        return PredictionResult(
            timestamp="2026-09-05T07:00:00.000000+00:00",
            predicted_class=sound,
            predicted_id=1,
            raw_class=sound,
            raw_id=1,
            confidence=confidence,
            is_confident=(confidence >= 0.70),
            status=PredictionStatus.CONFIRMED if confidence >= 0.70 else PredictionStatus.LOW_CONFIDENCE,
            probabilities={sound: confidence},
            latency=LatencyMetrics(15.0, 45.0, 20.0, 80.0),
        )

    def test_integration_test_1_car_horn_road(self, decision_engine):
        """Test 1: Car Horn, 95%, Road -> HIGH + Alert."""
        decision_engine.mode_manager.set_mode(EnvironmentMode.ROAD)
        p7_result = self._create_mock_phase7_result("car_horn", 0.95)

        decision = decision_engine.evaluate(p7_result)
        assert decision.sound == "car_horn"
        assert decision.confidence == 0.95
        assert decision.mode == EnvironmentMode.ROAD
        assert decision.priority == PriorityLevel.HIGH
        assert decision.alert_required is True

    def test_integration_test_2_doorbell_home(self, decision_engine):
        """Test 2: Doorbell, 90%, Home -> HIGH + Alert."""
        decision_engine.mode_manager.set_mode(EnvironmentMode.HOME)
        p7_result = self._create_mock_phase7_result("doorbell", 0.90)

        decision = decision_engine.evaluate(p7_result)
        assert decision.sound == "doorbell"
        assert decision.confidence == 0.90
        assert decision.mode == EnvironmentMode.HOME
        assert decision.priority == PriorityLevel.HIGH
        assert decision.alert_required is True

    def test_integration_test_3_dog_bark_office(self, decision_engine):
        """Test 3: Dog Bark, 90%, Office -> LOW + No immediate alert."""
        decision_engine.mode_manager.set_mode(EnvironmentMode.OFFICE)
        p7_result = self._create_mock_phase7_result("dog_bark", 0.90)

        decision = decision_engine.evaluate(p7_result)
        assert decision.sound == "dog_bark"
        assert decision.confidence == 0.90
        assert decision.mode == EnvironmentMode.OFFICE
        assert decision.priority == PriorityLevel.LOW
        assert decision.alert_required is False

    def test_integration_test_4_car_horn_low_confidence(self, decision_engine):
        """Test 4: Car Horn, 45%, Road -> IGNORE + No alert."""
        decision_engine.mode_manager.set_mode(EnvironmentMode.ROAD)
        p7_result = self._create_mock_phase7_result("car_horn", 0.45)

        decision = decision_engine.evaluate(p7_result)
        assert decision.sound == "car_horn"
        assert decision.confidence == 0.45
        assert decision.mode == EnvironmentMode.ROAD
        assert decision.priority == PriorityLevel.IGNORE
        assert decision.alert_required is False


# =====================================================================
# 7. Performance Benchmark (Step 19)
# =====================================================================

class TestPerformance:
    """Verifies negligible processing latency (< 1.0 ms per evaluation)."""

    def test_sub_millisecond_evaluation_speed(self):
        engine = ContextDecisionEngine(confidence_threshold=0.70)
        pred = SoundPrediction(sound="ambulance", confidence=0.92)

        # Warmup
        engine.evaluate(pred)

        # Benchmark 1,000 iterations
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            engine.evaluate(pred)
        duration = time.perf_counter() - start

        avg_latency_ms = (duration / iterations) * 1000.0
        # Average latency must be well under 1.0 millisecond
        assert avg_latency_ms < 1.0, f"Average latency too high: {avg_latency_ms:.4f} ms"
