import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, Mock
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

    @patch('app.core.config_manager.StructuredLogger')
    def test_initialization(self, mock_logger, temp_config_dir):
        """Test that the manager initializes and loads from a given directory."""
        cm = ConfigManager(config_dir=temp_config_dir)
        assert cm.config_dir == temp_config_dir
        assert "email" in cm.patterns
        assert "logging" in cm.settings

    @patch('app.core.config_manager.ConfigManager.get_default_settings', return_value={'mock': 'setting'})
    @patch('app.core.config_manager.ConfigManager.get_default_cities', return_value=['mock_city'])
    @patch('app.core.config_manager.ConfigManager.get_default_patterns')
    @patch('app.core.config_manager.ConfigManager.save_settings')
    @patch('app.core.config_manager.ConfigManager.save_cities')
    @patch('app.core.config_manager.ConfigManager.save_patterns')
    @patch('app.core.config_manager.Path.exists', return_value=False)
    @patch('app.core.config_manager.StructuredLogger')
    def test_fallback_to_defaults(self, mock_logger, mock_exists, mock_save_patterns, mock_save_cities, mock_save_settings, mock_get_patterns, mock_get_cities, mock_get_settings):
        """Test that the manager creates default files if none exist."""
        mock_regex = Mock()
        mock_regex.pattern = 'mock_regex_string'
        mock_get_patterns.return_value = {'mock_pattern_name': mock_regex}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)
            cm = ConfigManager(config_dir=config_path)

            mock_get_patterns.assert_called_once()
            mock_save_patterns.assert_called_once_with({'mock_pattern_name': 'mock_regex_string'})
            
            mock_get_cities.assert_called_once()
            mock_save_cities.assert_called_once_with(['mock_city'])
            
            mock_get_settings.assert_called_once()
            mock_save_settings.assert_called_once_with({'mock': 'setting'})

            assert cm.patterns == {'mock_pattern_name': mock_regex}
            assert cm.cities == ['mock_city']
            assert cm.settings == {'mock': 'setting'}

    def test_singleton_behavior(self):
        """Test that get_config_manager returns a singleton instance."""
        # Reset the global singleton instance for isolated testing
        # by clearing the lru_cache
        get_config_manager.cache_clear()
        
        cm1 = get_config_manager()
        cm2 = get_config_manager()
        assert cm1 is cm2

        # Ensure get_config also uses the singleton
        with patch('warnings.warn') as mock_warn:
            settings = get_config()
            assert settings == cm1.settings
            mock_warn.assert_called_once()
