"""Unit tests for configuration manager."""

import unittest
from config import settings


class TestConfig(unittest.TestCase):
    """Test suite for system and path configuration."""

    def test_system_config_defaults(self) -> None:
        """Verifies system settings load valid default values."""
        self.assertEqual(settings.system.app_name, "Smart Haptic Alert System")
        self.assertIn(settings.system.initial_mode, {"HOME", "ROAD", "OFFICE"})
        self.assertEqual(settings.audio.sample_rate, 16000)
        self.assertEqual(settings.audio.channels, 1)

    def test_path_config_directories(self) -> None:
        """Verifies output and log paths are properly resolved."""
        self.assertTrue(settings.paths.log_dir.exists())
        self.assertTrue(settings.paths.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
