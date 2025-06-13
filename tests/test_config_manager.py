import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
from app.core.config_manager import ConfigManager, get_config, get_config_manager

@pytest.fixture
def temp_config_dir():
    """Creates a temporary directory with dummy config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir)
        
        # Create dummy files
        with open(config_path / "settings.yaml", "w") as f:
            yaml.dump({"version": "1.0", "logging": {"level": "INFO"}}, f)
        
        with open(config_path / "patterns.yaml", "w") as f:
            yaml.dump({"pii_patterns": {"email": {"regex": ".+@.+\\..+"}}}, f)
            
        (config_path / "cities.yaml").touch()
        yield config_path

class TestConfigManager:
    """Focused unit tests for the ConfigManager."""

    def test_initialization(self, temp_config_dir):
        """Test that the manager initializes and loads from a given directory."""
        cm = ConfigManager(config_dir=temp_config_dir)
        assert cm.config_dir == temp_config_dir
        assert "email" in cm.patterns
        assert "logging" in cm.settings

    def test_fallback_to_defaults(self):
        """Test that the manager creates default files if none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)
            
            with patch.object(ConfigManager, 'get_default_patterns') as mock_get_patterns, \
                 patch.object(ConfigManager, 'get_default_cities') as mock_get_cities, \
                 patch.object(ConfigManager, 'get_default_settings') as mock_get_settings:
                
                mock_get_patterns.return_value = {}
                mock_get_cities.return_value = []
                mock_get_settings.return_value = {}

                cm = ConfigManager(config_dir=config_path)
                
                mock_get_patterns.assert_called_once()
                mock_get_cities.assert_called_once()
                mock_get_settings.assert_called_once()

    def test_singleton_behavior(self):
        """Test that get_config_manager returns a singleton instance."""
        # Reset the global singleton instance for isolated testing
        with patch('app.core.config_manager._config_manager_instance', None):
            cm1 = get_config_manager()
            cm2 = get_config_manager()
            assert cm1 is cm2

            # Ensure get_config also uses the singleton
            settings = get_config()
            assert settings == cm1.settings
