import pytest
import tempfile
import yaml
from pathlib import Path
from app.core.config_manager import ConfigManager


@pytest.mark.config
@pytest.mark.unit
class TestConfigManager:
    """Test configuration manager functionality."""

    def test_config_manager_initialization(self, test_config_dir: Path):
        """Test configuration manager initialization."""
        config_manager = ConfigManager(config_dir=test_config_dir)

        assert config_manager.config_dir == test_config_dir
        assert isinstance(config_manager.patterns, dict)
        assert isinstance(config_manager.cities, list)
        assert isinstance(config_manager.settings, dict)
        assert len(config_manager.patterns) > 0
        assert len(config_manager.cities) > 0

    def test_pattern_validation(self, test_config_dir: Path, test_patterns: dict):
        """Test pattern validation functionality."""
        config_manager = ConfigManager(config_dir=test_config_dir)

        # Test valid patterns
        valid_patterns = {k: v for k, v in test_patterns.items() if k != "invalid_pattern"}
        validated = config_manager.validate_patterns(valid_patterns)

        assert len(validated) == len(valid_patterns)
        for pattern_name in valid_patterns:
            assert pattern_name in validated

        # Test invalid pattern
        invalid_patterns = {"invalid_pattern": test_patterns["invalid_pattern"]}
        validated_invalid = config_manager.validate_patterns(invalid_patterns)

        assert len(validated_invalid) == 0  # Invalid pattern should be filtered out

    def test_pattern_categories(self, test_config_manager: ConfigManager):
        """Test pattern categorization."""
        categories = test_config_manager.get_pattern_categories()

        expected_categories = [
            "personal",
            "business",
            "location",
            "contact",
            "financial",
            "healthcare",
            "automotive",
            "legal",
            "dates",
            "generic",
        ]

        for category in expected_categories:
            assert category in categories
            assert isinstance(categories[category], list)

    def test_configuration_validation(self, test_config_manager: ConfigManager):
        """Test configuration validation."""
        is_valid, errors = test_config_manager.validate_configuration()

        assert is_valid is True
        assert len(errors) == 0

    def test_configuration_backup(self, test_config_manager: ConfigManager):
        """Test configuration backup functionality."""
        backup_success = test_config_manager.backup_configuration()
        assert backup_success is True

        # Check if backup directory was created
        backup_dir = test_config_manager.config_dir / "backups"
        assert backup_dir.exists()

        # Check if backup files were created
        backup_files = list(backup_dir.glob("*.yaml"))
        assert len(backup_files) > 0

    def test_configuration_reload(self, test_config_manager: ConfigManager):
        """Test configuration reload functionality."""
        original_patterns_count = len(test_config_manager.patterns)

        # Reload configuration
        reload_success = test_config_manager.reload_configuration()
        assert reload_success is True

        # Verify patterns are still loaded
        assert len(test_config_manager.patterns) == original_patterns_count

    def test_save_and_load_patterns(self, test_config_dir: Path, test_patterns: dict):
        """Test saving and loading patterns."""
        config_manager = ConfigManager(config_dir=test_config_dir)

        # Save test patterns
        valid_patterns = {k: v for k, v in test_patterns.items() if k != "invalid_pattern"}
        save_success = config_manager.save_patterns(valid_patterns)
        assert save_success is True

        # Create new config manager to test loading
        new_config_manager = ConfigManager(config_dir=test_config_dir)

        # Verify patterns were loaded correctly
        for pattern_name, pattern in valid_patterns.items():
            assert pattern_name in new_config_manager.patterns
            assert new_config_manager.patterns[pattern_name] == pattern

    def test_save_and_load_cities(self, test_config_dir: Path, test_cities: list):
        """Test saving and loading cities."""
        config_manager = ConfigManager(config_dir=test_config_dir)

        # Save test cities
        save_success = config_manager.save_cities(test_cities)
        assert save_success is True

        # Create new config manager to test loading
        new_config_manager = ConfigManager(config_dir=test_config_dir)

        # Verify cities were loaded correctly
        for city in test_cities:
            assert city in new_config_manager.cities

    def test_save_and_load_settings(self, test_config_dir: Path):
        """Test saving and loading settings."""
        config_manager = ConfigManager(config_dir=test_config_dir)

        test_settings = {"version": "2.0.0", "test_setting": True, "nested": {"value": 42}}

        # Save test settings
        save_success = config_manager.save_settings(test_settings)
        assert save_success is True

        # Create new config manager to test loading
        new_config_manager = ConfigManager(config_dir=test_config_dir)

        # Verify settings were loaded correctly
        assert new_config_manager.settings["version"] == "2.0.0"
        assert new_config_manager.settings["test_setting"] is True
        assert new_config_manager.settings["nested"]["value"] == 42

    def test_fallback_to_defaults(self):
        """Test fallback to default configuration when files don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_config_dir = Path(temp_dir) / "empty_config"
            empty_config_dir.mkdir()

            config_manager = ConfigManager(config_dir=empty_config_dir)

            # Should have default patterns and cities
            assert len(config_manager.patterns) > 0
            assert len(config_manager.cities) > 0
            assert len(config_manager.settings) > 0

            # Check that config files were created
            assert (empty_config_dir / "patterns.yaml").exists()
            assert (empty_config_dir / "cities.yaml").exists()
            assert (empty_config_dir / "settings.yaml").exists()


@pytest.mark.config
@pytest.mark.integration
class TestConfigManagerIntegration:
    """Integration tests for configuration manager."""

    def test_config_file_format(self, test_config_manager: ConfigManager):
        """Test that config files are properly formatted YAML."""
        # Test patterns file
        with open(test_config_manager.patterns_file, "r", encoding="utf-8") as f:
            patterns_data = yaml.safe_load(f)
            assert "metadata" in patterns_data
            assert "pii_patterns" in patterns_data
            assert "version" in patterns_data["metadata"]
            assert "last_updated" in patterns_data["metadata"]

        # Test cities file
        with open(test_config_manager.cities_file, "r", encoding="utf-8") as f:
            cities_data = yaml.safe_load(f)
            assert "metadata" in cities_data
            assert "lithuanian_cities" in cities_data
            assert isinstance(cities_data["lithuanian_cities"], list)

        # Test settings file
        with open(test_config_manager.settings_file, "r", encoding="utf-8") as f:
            settings_data = yaml.safe_load(f)
            assert isinstance(settings_data, dict)

    def test_configuration_persistence(self, test_config_dir: Path):
        """Test that configuration changes persist across instances."""
        # Create first config manager and modify configuration
        config1 = ConfigManager(config_dir=test_config_dir)
        original_pattern_count = len(config1.patterns)

        # Add a new pattern
        new_patterns = config1.patterns.copy()
        new_patterns["test_pattern"] = r"\bTEST\d+\b"
        config1.save_patterns(new_patterns)

        # Create second config manager
        config2 = ConfigManager(config_dir=test_config_dir)

        # Verify the new pattern persists
        assert len(config2.patterns) == original_pattern_count + 1
        assert "test_pattern" in config2.patterns
        assert config2.patterns["test_pattern"] == r"\bTEST\d+\b"

    def test_error_handling_corrupted_files(self, test_config_dir: Path):
        """Test error handling when config files are corrupted."""
        config_manager = ConfigManager(config_dir=test_config_dir)

        # Corrupt the patterns file
        with open(config_manager.patterns_file, "w", encoding="utf-8") as f:
            f.write("invalid: yaml: content: [unclosed")

        # Create new config manager - should fall back to defaults
        new_config_manager = ConfigManager(config_dir=test_config_dir)

        # Should still have patterns (from defaults)
        assert len(new_config_manager.patterns) > 0
