"""Unit tests for ContextManager and mode profiles."""

import unittest
from app.context.context_manager import ContextManager
from app.context.config.mode_profiles import EnvironmentMode, SoundPriority


class TestContext(unittest.TestCase):
    """Test suite for environmental context prioritization."""

    def test_initial_mode_setting(self) -> None:
        """Verifies ContextManager initializes with expected mode."""
        ctx = ContextManager(initial_mode=EnvironmentMode.HOME)
        self.assertEqual(ctx.active_mode, EnvironmentMode.HOME)

    def test_home_mode_evaluation(self) -> None:
        """Verifies sound priority logic in HOME mode."""
        ctx = ContextManager(initial_mode=EnvironmentMode.HOME)

        # Baby crying should trigger HIGH priority alert in HOME mode
        should_alert, priority = ctx.evaluate_sound("baby_crying", confidence=0.90)
        self.assertTrue(should_alert)
        self.assertEqual(priority, SoundPriority.HIGH)

        # Low confidence car horn should be ignored in HOME mode
        should_alert, priority = ctx.evaluate_sound("car_horn", confidence=0.40)
        self.assertFalse(should_alert)

    def test_mode_switch_to_road(self) -> None:
        """Verifies prioritization changes when switching to ROAD mode."""
        ctx = ContextManager(initial_mode=EnvironmentMode.HOME)
        ctx.set_mode(EnvironmentMode.ROAD)

        self.assertEqual(ctx.active_mode, EnvironmentMode.ROAD)
        # Siren should trigger CRITICAL alert in ROAD mode
        should_alert, priority = ctx.evaluate_sound("siren", confidence=0.85)
        self.assertTrue(should_alert)
        self.assertEqual(priority, SoundPriority.CRITICAL)


if __name__ == "__main__":
    unittest.main()
