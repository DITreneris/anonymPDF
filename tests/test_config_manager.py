import pytest
import tempfile
import yaml
import re
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, mock_open, MagicMock
from datetime import datetime
from typing import Dict, List, Any
import threading
import warnings

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
            yaml.dump({"pii_patterns": {"emails": {"regex": ".+@.+\\..+"}}}, f)
            
        (config_path / "cities.yaml").touch()
        yield config_path

@pytest.fixture
def temp_empty_config_dir():
    """Creates a temporary empty directory for testing defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_patterns():
    """Sample patterns for testing."""
    return {
        "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}-\d{3}-\d{4}\b",
        "invalid_pattern": r"[unclosed",  # Invalid regex for testing
        "dict_pattern": {"regex": r"\d{4}-\d{4}-\d{4}-\d{4}"}
    }

@pytest.fixture
def sample_cities():
    """Sample cities for testing."""
    return ["Vilnius", "Kaunas", "Klaipėda", "Šiauliai"]

@pytest.fixture
def sample_settings():
    """Sample settings for testing."""
    return {
        "version": "1.0.0",
        "language_detection": {"enabled": True},
        "processing": {"max_file_size_mb": 50},
        "logging": {"level": "INFO"},
        "patterns": {"case_sensitive": False},
        "anti_overredaction": {"technical_terms_whitelist": ["test"]},
        "adaptive_learning": {
            "enabled": True,
            "databases": {"test": "test.db"},
            "thresholds": {"confidence": 0.8},
            "performance": {"cache_size": 1000, "cache_ttl_seconds": 3600, "max_workers": 4},
            "monitoring": {"enabled": True}
        }
    }


class TestConfigManager:
    """Focused unit tests for the ConfigManager."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_initialization(self, mock_logger, temp_config_dir):
        """Test that the manager initializes and loads from a given directory."""
        cm = ConfigManager(config_dir=temp_config_dir)
        assert cm.config_dir == temp_config_dir
        assert "emails" in cm.patterns
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


class TestPatternCompilation:
    """Tests for pattern compilation and validation."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_compile_and_validate_patterns_string_patterns(self, mock_logger, temp_empty_config_dir):
        """Test compilation of string patterns."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        patterns = {
            "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}-\d{3}-\d{4}\b"
        }
        
        compiled = cm.compile_and_validate_patterns(patterns)
        
        assert len(compiled) == 2
        assert isinstance(compiled["emails"], re.Pattern)
        assert isinstance(compiled["phone"], re.Pattern)

    @patch('app.core.config_manager.StructuredLogger')
    def test_compile_and_validate_patterns_dict_patterns(self, mock_logger, temp_empty_config_dir):
        """Test compilation of dictionary-based patterns."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        patterns = {
            "credit_card": {"regex": r"\d{4}-\d{4}-\d{4}-\d{4}"},
            "ssn": {"regex": r"\d{3}-\d{2}-\d{4}"}
        }
        
        compiled = cm.compile_and_validate_patterns(patterns)
        
        assert len(compiled) == 2
        assert isinstance(compiled["credit_card"], re.Pattern)
        assert isinstance(compiled["ssn"], re.Pattern)

    @patch('app.core.config_manager.StructuredLogger')
    def test_compile_and_validate_patterns_invalid_regex(self, mock_logger, temp_empty_config_dir):
        """Test handling of invalid regex patterns."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        patterns = {
            "valid": r"\d+",
            "invalid": r"[unclosed",  # Invalid regex
            "malformed": {"no_regex_key": "value"}  # Missing regex key
        }
        
        compiled = cm.compile_and_validate_patterns(patterns)
        
        # Only valid pattern should be compiled
        assert len(compiled) == 1
        assert "valid" in compiled
        assert "invalid" not in compiled
        assert "malformed" not in compiled

    @patch('app.core.config_manager.StructuredLogger')
    def test_get_default_patterns(self, mock_logger, temp_empty_config_dir):
        """Test default patterns generation."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        defaults = cm.get_default_patterns()
        
        # Should have many default patterns
        assert len(defaults) > 10
        
        # Check some key patterns exist
        assert "emails" in defaults
        assert "phone" in defaults
        assert "lithuanian_personal_codes" in defaults
        
        # All should be compiled regex objects
        for pattern in defaults.values():
            assert isinstance(pattern, re.Pattern)


class TestFileLoading:
    """Tests for file loading functionality."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_load_patterns_from_file(self, mock_logger, temp_config_dir):
        """Test loading patterns from existing file."""
        # Create patterns file
        patterns_data = {
            "pii_patterns": {
                "test_pattern": r"\d+",
                "email_pattern": {"regex": r".+@.+"}
            }
        }
        patterns_file = temp_config_dir / "patterns.yaml"
        with open(patterns_file, "w") as f:
            yaml.dump(patterns_data, f)
        
        cm = ConfigManager(config_dir=temp_config_dir)
        
        assert "test_pattern" in cm.patterns
        assert "email_pattern" in cm.patterns

    @patch('app.core.config_manager.StructuredLogger')
    def test_load_patterns_file_error(self, mock_logger, temp_config_dir):
        """Test loading patterns when file has YAML errors."""
        # Create malformed YAML file
        patterns_file = temp_config_dir / "patterns.yaml"
        with open(patterns_file, "w") as f:
            f.write("invalid: yaml: content: [")
        
        cm = ConfigManager(config_dir=temp_config_dir)
        
        # Should fall back to defaults
        assert len(cm.patterns) > 0
        assert "emails" in cm.patterns  # From defaults

    @patch('app.core.config_manager.StructuredLogger')
    def test_load_cities_from_file(self, mock_logger, temp_config_dir):
        """Test loading cities from existing file."""
        cities_data = {"lithuanian_cities": ["Vilnius", "Kaunas", "Test City"]}
        cities_file = temp_config_dir / "cities.yaml"
        with open(cities_file, "w") as f:
            yaml.dump(cities_data, f)
        
        cm = ConfigManager(config_dir=temp_config_dir)
        
        assert len(cm.cities) == 3
        assert "Vilnius" in cm.cities
        assert "Test City" in cm.cities

    @patch('app.core.config_manager.StructuredLogger')
    def test_load_cities_file_error(self, mock_logger, temp_config_dir):
        """Test loading cities when file has errors."""
        cities_file = temp_config_dir / "cities.yaml"
        with open(cities_file, "w") as f:
            f.write("invalid yaml content [")
        
        cm = ConfigManager(config_dir=temp_config_dir)
        
        # Should fall back to defaults
        default_cities = cm.get_default_cities()
        assert len(cm.cities) == len(default_cities)

    @patch('app.core.config_manager.StructuredLogger')
    def test_load_settings_from_file(self, mock_logger, temp_config_dir):
        """Test loading settings from existing file."""
        settings_data = {"version": "2.0", "custom_setting": {"value": True}}
        settings_file = temp_config_dir / "settings.yaml"
        with open(settings_file, "w") as f:
            yaml.dump(settings_data, f)
        
        cm = ConfigManager(config_dir=temp_config_dir)
        
        assert cm.settings["version"] == "2.0"
        assert cm.settings["custom_setting"]["value"] == True

    @patch('app.core.config_manager.StructuredLogger')
    def test_load_brand_names_from_file(self, mock_logger, temp_config_dir):
        """Test loading brand names from existing file."""
        brand_data = {"brand_names": ["Microsoft", "Google", "TestBrand"]}
        brand_file = temp_config_dir / "brand_names.yaml"
        with open(brand_file, "w") as f:
            yaml.dump(brand_data, f)
        
        cm = ConfigManager(config_dir=temp_config_dir)
        
        assert len(cm.brand_names) == 3
        assert "TestBrand" in cm.brand_names


class TestSavingMethods:
    """Tests for configuration saving functionality."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_save_patterns_success(self, mock_logger, temp_empty_config_dir):
        """Test successful pattern saving."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        patterns = {
            "test_pattern": r"\d+",
            "email_pattern": r".+@.+"
        }
        
        result = cm.save_patterns(patterns)
        
        assert result == True
        assert cm.patterns_file.exists()
        
        # Verify file content
        with open(cm.patterns_file, "r") as f:
            saved_data = yaml.safe_load(f)
        
        assert "pii_patterns" in saved_data
        assert "test_pattern" in saved_data["pii_patterns"]

    @patch('app.core.config_manager.StructuredLogger')
    def test_save_patterns_with_compiled_objects(self, mock_logger, temp_empty_config_dir):
        """Test saving patterns when input contains compiled regex objects."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        compiled_pattern = re.compile(r"\d+")
        patterns = {
            "string_pattern": r".+@.+",
            "compiled_pattern": compiled_pattern
        }
        
        result = cm.save_patterns(patterns)
        
        assert result == True
        
        # Verify file content - compiled patterns should be converted to strings
        with open(cm.patterns_file, "r") as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["pii_patterns"]["compiled_pattern"] == r"\d+"

    @patch('app.core.config_manager.StructuredLogger')
    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_save_patterns_permission_error(self, mock_open, mock_logger, temp_empty_config_dir):
        """Test pattern saving when file permission error occurs."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        patterns = {"test": r"\d+"}
        result = cm.save_patterns(patterns)
        
        assert result == False

    @patch('app.core.config_manager.StructuredLogger')
    def test_save_cities_success(self, mock_logger, temp_empty_config_dir):
        """Test successful cities saving."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        cities = ["Vilnius", "Kaunas", "Test City"]
        result = cm.save_cities(cities)
        
        assert result == True
        assert cm.cities_file.exists()
        
        # Verify file content
        with open(cm.cities_file, "r") as f:
            saved_data = yaml.safe_load(f)
        
        assert "lithuanian_cities" in saved_data
        assert len(saved_data["lithuanian_cities"]) == 3
        assert "metadata" in saved_data

    @patch('app.core.config_manager.StructuredLogger')
    def test_save_brand_names_success(self, mock_logger, temp_empty_config_dir):
        """Test successful brand names saving."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        brands = ["Microsoft", "Google", "TestBrand"]
        result = cm.save_brand_names(brands)
        
        assert result == True
        assert cm.brand_names_file.exists()
        
        # Verify file content
        with open(cm.brand_names_file, "r") as f:
            saved_data = yaml.safe_load(f)
        
        assert "brand_names" in saved_data
        assert "TestBrand" in saved_data["brand_names"]

    @patch('app.core.config_manager.StructuredLogger')
    def test_save_settings_success(self, mock_logger, temp_empty_config_dir):
        """Test successful settings saving."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        settings = {"version": "2.0", "custom": {"enabled": True}}
        result = cm.save_settings(settings)
        
        assert result == True
        assert cm.settings_file.exists()
        
        # Verify file content
        with open(cm.settings_file, "r") as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["version"] == "2.0"
        assert saved_data["custom"]["enabled"] == True


class TestReloadAndBackup:
    """Tests for reload and backup functionality."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_reload_configuration_success(self, mock_logger, temp_config_dir):
        """Test successful configuration reload."""
        cm = ConfigManager(config_dir=temp_config_dir)
        
        # Modify files after initialization
        new_patterns = {"pii_patterns": {"new_pattern": r"\d+"}}
        with open(cm.patterns_file, "w") as f:
            yaml.dump(new_patterns, f)
        
        new_cities = {"lithuanian_cities": ["NewCity"]}
        with open(cm.cities_file, "w") as f:
            yaml.dump(new_cities, f)
        
        result = cm.reload_configuration()
        
        assert result == True
        assert "new_pattern" in cm.patterns
        assert "NewCity" in cm.cities

    @patch('app.core.config_manager.StructuredLogger')
    def test_reload_configuration_error(self, mock_logger, temp_empty_config_dir):
        """Test reload when files have errors."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        # Create malformed file
        with open(cm.patterns_file, "w") as f:
            f.write("invalid: yaml: [")
        
        # Mock load_patterns to raise exception
        with patch.object(cm, 'load_patterns', side_effect=Exception("Load error")):
            result = cm.reload_configuration()
        
        assert result == False

    @patch('app.core.config_manager.StructuredLogger')
    def test_backup_configuration_default_dir(self, mock_logger, temp_config_dir):
        """Test backup with default backup directory."""
        cm = ConfigManager(config_dir=temp_config_dir)
        
        result = cm.backup_configuration()
        
        assert result == True
        backup_dir = cm.config_dir / "backups"
        assert backup_dir.exists()
        
        # Check if backup files were created
        backup_files = list(backup_dir.glob("*.yaml"))
        assert len(backup_files) > 0

    @patch('app.core.config_manager.StructuredLogger')
    def test_backup_configuration_custom_dir(self, mock_logger, temp_config_dir):
        """Test backup with custom backup directory."""
        cm = ConfigManager(config_dir=temp_config_dir)
        
        custom_backup_dir = temp_config_dir / "custom_backups"
        result = cm.backup_configuration(backup_dir=custom_backup_dir)
        
        assert result == True
        assert custom_backup_dir.exists()

    @patch('app.core.config_manager.StructuredLogger')
    def test_backup_configuration_error(self, mock_logger, temp_config_dir):
        """Test backup when directory creation fails."""
        cm = ConfigManager(config_dir=temp_config_dir)
        
        # Patch the mkdir method specifically on the backup directory
        with patch.object(Path, 'mkdir', side_effect=PermissionError("Cannot create directory")):
            result = cm.backup_configuration()
        
        assert result == False


class TestPatternCategorization:
    """Tests for pattern categorization functionality."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_get_pattern_categories(self, mock_logger, temp_empty_config_dir):
        """Test pattern categorization logic."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
                 # Override patterns with test data
        test_patterns = {
            "personal_code": re.compile(r"\d{11}"),
            "business_vat": re.compile(r"LT\d+"),
            "email_contact": re.compile(r".+@.+"),
            "address_location": re.compile(r".+ street"),
            "credit_card_financial": re.compile(r"\d{4}-\d{4}"),
            "health_record": re.compile(r"HR\d+"),
            "car_plate_automotive": re.compile(r"[A-Z]{3}\d{3}"),
            "legal_document": re.compile(r"LE\d+"),
            "date_birth": re.compile(r"\d{4}-\d{2}-\d{2}"),
            "unknown_pattern": re.compile(r".+")
        }
        cm.patterns = test_patterns
        
        categories = cm.get_pattern_categories()
        
        assert "personal_code" in categories["personal"]
        assert "business_vat" in categories["business"]
        assert "email_contact" in categories["contact"]
        assert "address_location" in categories["location"]
        assert "credit_card_financial" in categories["financial"]
        assert "health_record" in categories["healthcare"]
        assert "car_plate_automotive" in categories["automotive"]
        assert "legal_document" in categories["legal"]
        assert "date_birth" in categories["dates"]
        assert "unknown_pattern" in categories["generic"]


class TestConfigurationValidation:
    """Tests for configuration validation functionality."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_validate_configuration_success(self, mock_logger, temp_empty_config_dir):
        """Test successful configuration validation."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        # Set up valid configuration
        cm.patterns = {"test": re.compile(r"\d+")}
        cm.cities = ["Vilnius", "Kaunas"]
        cm.settings = {
            "version": "1.0",
            "language_detection": {},
            "processing": {},
            "logging": {},
            "patterns": {},
            "anti_overredaction": {},
            "adaptive_learning": {
                "enabled": True,
                "databases": {"test": "test.db"},
                "thresholds": {"confidence": 0.8},
                "performance": {"cache_size": 1000, "cache_ttl_seconds": 3600, "max_workers": 4},
                "monitoring": {}
            }
        }
        
        is_valid, errors = cm.validate_configuration()
        
        assert is_valid == True
        assert len(errors) == 0

    @patch('app.core.config_manager.StructuredLogger')
    def test_validate_configuration_invalid_patterns(self, mock_logger, temp_empty_config_dir):
        """Test validation with invalid patterns."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        # Mock invalid pattern that would raise re.error
        invalid_pattern = Mock()
        invalid_pattern.side_effect = re.error("Invalid regex")
        
        # Set up patterns with mock
        cm.patterns = {"invalid": invalid_pattern}
        cm.cities = ["Vilnius"]
        cm.settings = {}
        
        with patch('re.compile', side_effect=re.error("Invalid regex")):
            is_valid, errors = cm.validate_configuration()
        
        assert is_valid == False
        assert len(errors) > 0
        assert any("Invalid pattern" in error for error in errors)

    @patch('app.core.config_manager.StructuredLogger')
    def test_validate_configuration_invalid_cities(self, mock_logger, temp_empty_config_dir):
        """Test validation with invalid cities configuration."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        cm.patterns = {"test": re.compile(r"\d+")}
        cm.cities = "not_a_list"  # Invalid: should be list
        cm.settings = {}
        
        is_valid, errors = cm.validate_configuration()
        
        assert is_valid == False
        assert any("Cities configuration must be a list" in error for error in errors)

    @patch('app.core.config_manager.StructuredLogger')
    def test_validate_configuration_empty_cities(self, mock_logger, temp_empty_config_dir):
        """Test validation with empty cities list."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        cm.patterns = {"test": re.compile(r"\d+")}
        cm.cities = []  # Empty list
        cm.settings = {}
        
        is_valid, errors = cm.validate_configuration()
        
        assert is_valid == False
        assert any("Cities list is empty" in error for error in errors)

    @patch('app.core.config_manager.StructuredLogger')
    def test_validate_configuration_missing_settings(self, mock_logger, temp_empty_config_dir):
        """Test validation with missing required settings."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        cm.patterns = {"test": re.compile(r"\d+")}
        cm.cities = ["Vilnius"]
        cm.settings = {"version": "1.0"}  # Missing many required settings
        
        is_valid, errors = cm.validate_configuration()
        
        assert is_valid == False
        assert any("Missing required setting" in error for error in errors)

    @patch('app.core.config_manager.StructuredLogger')
    def test_validate_configuration_invalid_adaptive_learning(self, mock_logger, temp_empty_config_dir):
        """Test validation with invalid adaptive learning settings."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        cm.patterns = {"test": re.compile(r"\d+")}
        cm.cities = ["Vilnius"]
        cm.settings = {
            "version": "1.0",
            "language_detection": {},
            "processing": {},
            "logging": {},
            "patterns": {},
            "anti_overredaction": {},
            "adaptive_learning": {
                "enabled": True,
                "databases": {"test": 123},  # Invalid: should be string
                "thresholds": {"confidence": 1.5},  # Invalid: should be 0-1
                "performance": {"cache_size": -1, "cache_ttl_seconds": "invalid", "max_workers": 0},  # Invalid values
                "monitoring": {}
            }
        }
        
        is_valid, errors = cm.validate_configuration()
        
        assert is_valid == False
        assert any("Invalid database path" in error for error in errors)
        assert any("Invalid threshold value" in error for error in errors)
        assert any("Invalid cache_size" in error for error in errors)


class TestDefaultDataProviders:
    """Tests for default data provider methods."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_get_default_cities(self, mock_logger, temp_empty_config_dir):
        """Test default cities generation."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        cities = cm.get_default_cities()
        
        assert len(cities) > 50  # Should have many cities
        assert "Vilnius" in cities
        assert "Kaunas" in cities
        assert "Klaipėda" in cities

    @patch('app.core.config_manager.StructuredLogger')
    def test_get_default_brand_names(self, mock_logger, temp_empty_config_dir):
        """Test default brand names generation."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        brands = cm.get_default_brand_names()
        
        assert len(brands) > 10
        assert "Microsoft" in brands
        assert "Google" in brands
        assert "Apple" in brands

    @patch('app.core.config_manager.StructuredLogger')
    def test_get_default_settings(self, mock_logger, temp_empty_config_dir):
        """Test default settings generation."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        settings = cm.get_default_settings()
        
        assert "version" in settings
        assert "language_detection" in settings
        assert "processing" in settings
        assert "logging" in settings
        assert "patterns" in settings
        assert "anti_overredaction" in settings


class TestUserConfig:
    """Tests for user configuration retrieval."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_get_user_config(self, mock_logger, temp_config_dir):
        """Test user configuration retrieval."""
        cm = ConfigManager(config_dir=temp_config_dir)
        
        user_config = cm.get_user_config()
        
        assert "patterns" in user_config
        assert "cities" in user_config
        assert "settings" in user_config
        
        assert user_config["patterns"] == cm.patterns
        assert user_config["cities"] == cm.cities
        assert user_config["settings"] == cm.settings


class TestGlobalSingletonManagement:
    """Tests for global singleton management."""

    def test_singleton_with_custom_path(self):
        """Test singleton behavior with custom config path."""
        get_config_manager.cache_clear()
        
        # Clear the global singleton instance
        import app.core.config_manager
        app.core.config_manager._config_manager_instance = None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cm1 = get_config_manager(tmpdir)
            cm2 = get_config_manager(tmpdir)  # Same arguments should return same instance
            
            assert cm1 is cm2
            assert str(cm1.config_dir) == tmpdir

    def test_deprecated_get_config_warning(self):
        """Test that get_config() issues deprecation warning."""
        get_config_manager.cache_clear()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            config = get_config()
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)


class TestThreadSafety:
    """Tests for thread safety of singleton management."""

    def test_singleton_thread_safety(self):
        """Test that singleton creation is thread-safe."""
        get_config_manager.cache_clear()
        
        instances = []
        
        def create_instance():
            cm = get_config_manager()
            instances.append(cm)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All instances should be the same
        assert len(instances) == 10
        assert all(instance is instances[0] for instance in instances)


class TestErrorHandling:
    """Tests for error handling in various scenarios."""

    @patch('app.core.config_manager.StructuredLogger')
    @patch('pathlib.Path.mkdir', side_effect=PermissionError("Cannot create directory"))
    def test_initialization_directory_creation_error(self, mock_mkdir, mock_logger):
        """Test initialization when config directory cannot be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent"
            
            # Should handle error gracefully
            with pytest.raises(PermissionError):
                ConfigManager(config_dir=config_path)

    @patch('app.core.config_manager.StructuredLogger')
    def test_file_operations_with_readonly_directory(self, mock_logger, temp_config_dir):
        """Test file operations when file writing fails."""
        cm = ConfigManager(config_dir=temp_config_dir)
        
        # Mock file writing to simulate permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            result = cm.save_patterns({"test": r"\d+"})
            assert result == False

    @patch('app.core.config_manager.StructuredLogger')
    def test_malformed_yaml_handling(self, mock_logger, temp_config_dir):
        """Test handling of various malformed YAML files."""
        test_cases = [
            "invalid: yaml: content: [",  # Unclosed bracket
            "key: value\n  invalid_indent",  # Invalid indentation
            "key: value\nkey: value",  # Duplicate keys
            "",  # Empty file
            "\t\tinvalid tabs",  # Invalid characters
        ]
        
        for i, invalid_yaml in enumerate(test_cases):
            # Test each malformed YAML case
            patterns_file = temp_config_dir / f"patterns_{i}.yaml"
            with open(patterns_file, "w") as f:
                f.write(invalid_yaml)
            
            # Should handle gracefully by falling back to defaults
            cm = ConfigManager(config_dir=temp_config_dir)
            cm.patterns_file = patterns_file  # Override to test specific file
            
            try:
                patterns = cm.load_patterns()
                # Should either load defaults or handle error gracefully
                assert isinstance(patterns, dict)
            except Exception:
                # Should not raise unhandled exceptions
                pytest.fail(f"Unhandled exception for malformed YAML case {i}")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch('app.core.config_manager.StructuredLogger')
    def test_empty_pattern_compilation(self, mock_logger, temp_empty_config_dir):
        """Test compilation of empty patterns dictionary."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        compiled = cm.compile_and_validate_patterns({})
        
        assert compiled == {}

    @patch('app.core.config_manager.StructuredLogger')
    def test_none_pattern_values(self, mock_logger, temp_empty_config_dir):
        """Test compilation with None pattern values."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        patterns = {
            "valid": r"\d+",
            "none_value": None,
            "empty_dict": {},
            "dict_with_none": {"regex": None}
        }
        
        compiled = cm.compile_and_validate_patterns(patterns)
        
        # Only valid pattern should be compiled
        assert len(compiled) == 1
        assert "valid" in compiled

    @patch('app.core.config_manager.StructuredLogger')
    def test_large_pattern_set(self, mock_logger, temp_empty_config_dir):
        """Test handling of large pattern sets."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        # Create large pattern set
        large_patterns = {f"pattern_{i}": r"\d+" for i in range(1000)}
        
        compiled = cm.compile_and_validate_patterns(large_patterns)
        
        assert len(compiled) == 1000

    @patch('app.core.config_manager.StructuredLogger')  
    def test_unicode_in_patterns(self, mock_logger, temp_empty_config_dir):
        """Test handling of Unicode characters in patterns."""
        cm = ConfigManager(config_dir=temp_empty_config_dir)
        
        unicode_patterns = {
            "lithuanian_chars": r"[ąčęėįšųūž]+",
            "emoji_pattern": r"😀|😁|😂",
            "mixed_unicode": r"[A-Za-ząčęėįšųūž0-9]+"
        }
        
        compiled = cm.compile_and_validate_patterns(unicode_patterns)
        
        assert len(compiled) == 3
        for pattern in compiled.values():
            assert isinstance(pattern, re.Pattern)
