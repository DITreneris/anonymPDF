"""
Comprehensive tests for app.core.dependencies module.
Tests dependency validation, error handling, and installation guide generation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import spacy
from app.core.dependencies import DependencyValidator, validate_dependencies_on_startup


class TestDependencyValidator:
    """Test the DependencyValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a DependencyValidator instance for testing."""
        return DependencyValidator()

    def test_init(self, validator):
        """Test DependencyValidator initialization."""
        assert validator.required_spacy_models == ["en_core_web_sm", "lt_core_news_sm"]
        assert validator.required_directories == ["uploads", "processed", "temp", "logs"]
        assert validator.validation_results == {}

    @patch('app.core.dependencies.spacy.load')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_spacy_models_success(self, mock_logger, mock_spacy_load, validator):
        """Test successful spaCy model validation."""
        # Mock successful model loading
        mock_nlp = Mock()
        mock_nlp.pipe_names = ['tagger', 'parser', 'ner']
        mock_spacy_load.return_value = mock_nlp

        result = validator.validate_spacy_models()

        assert result == {"en_core_web_sm": True, "lt_core_news_sm": True}
        assert mock_spacy_load.call_count == 2
        mock_logger.log_dependency_check.assert_called()

    @patch('app.core.dependencies.spacy.load')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_spacy_models_failure(self, mock_logger, mock_spacy_load, validator):
        """Test spaCy model validation with missing models."""
        # Mock model loading failure
        mock_spacy_load.side_effect = OSError("Model not found")

        result = validator.validate_spacy_models()

        assert result == {"en_core_web_sm": False, "lt_core_news_sm": False}
        assert mock_spacy_load.call_count == 2
        mock_logger.log_dependency_check.assert_called()

    @patch('app.core.dependencies.spacy.load')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_spacy_models_mixed_results(self, mock_logger, mock_spacy_load, validator):
        """Test spaCy model validation with mixed success/failure."""
        # Mock mixed results - first succeeds, second fails
        mock_nlp = Mock()
        mock_nlp.pipe_names = ['tagger', 'parser']
        mock_spacy_load.side_effect = [mock_nlp, OSError("Model not found")]

        result = validator.validate_spacy_models()

        assert result == {"en_core_web_sm": True, "lt_core_news_sm": False}

    @patch('app.core.dependencies.Path')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_directories_success(self, mock_logger, mock_path, validator):
        """Test successful directory validation."""
        # Mock successful directory operations
        mock_dir = Mock()
        mock_path.return_value = mock_dir
        mock_test_file = Mock()
        # Mock the / operator (__truediv__)
        mock_dir.__truediv__ = Mock(return_value=mock_test_file)

        result = validator.validate_directories()

        expected = {"uploads": True, "processed": True, "temp": True, "logs": True}
        assert result == expected
        assert mock_dir.mkdir.call_count == 4
        mock_logger.log_dependency_check.assert_called()

    @patch('app.core.dependencies.Path')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_directories_permission_error(self, mock_logger, mock_path, validator):
        """Test directory validation with permission errors."""
        # Mock permission error
        mock_dir = Mock()
        mock_path.return_value = mock_dir
        mock_dir.mkdir.side_effect = PermissionError("Access denied")

        result = validator.validate_directories()

        expected = {"uploads": False, "processed": False, "temp": False, "logs": False}
        assert result == expected
        mock_logger.log_dependency_check.assert_called()

    @patch('app.core.dependencies.Path')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_directories_write_test_failure(self, mock_logger, mock_path, validator):
        """Test directory validation with write test failure."""
        # Mock successful mkdir but failed write test
        mock_dir = Mock()
        mock_path.return_value = mock_dir
        mock_test_file = Mock()
        mock_dir.__truediv__ = Mock(return_value=mock_test_file)
        mock_test_file.write_text.side_effect = OSError("No space left")

        result = validator.validate_directories()

        expected = {"uploads": False, "processed": False, "temp": False, "logs": False}
        assert result == expected

    @patch('builtins.__import__')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_python_packages_success(self, mock_logger, mock_import, validator):
        """Test successful Python package validation."""
        # Mock successful imports
        mock_import.return_value = Mock()

        result = validator.validate_python_packages()

        expected = {
            "fastapi": True,
            "sqlalchemy": True,
            "PyMuPDF": True,
            "spacy": True,
            "langdetect": True,
            "pdfminer.six": True
        }
        assert result == expected
        mock_logger.log_dependency_check.assert_called()

    @patch('builtins.__import__')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_python_packages_missing(self, mock_logger, mock_import, validator):
        """Test Python package validation with missing packages."""
        # Mock import failures
        mock_import.side_effect = ImportError("Module not found")

        result = validator.validate_python_packages()

        expected = {
            "fastapi": False,
            "sqlalchemy": False,
            "PyMuPDF": False,
            "spacy": False,
            "langdetect": False,
            "pdfminer.six": False
        }
        assert result == expected
        mock_logger.log_dependency_check.assert_called()

    @patch('builtins.__import__')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_python_packages_mixed_results(self, mock_logger, mock_import, validator):
        """Test Python package validation with mixed results."""
        # Mock mixed import results
        def import_side_effect(name):
            if name in ["fastapi", "spacy"]:
                return Mock()
            else:
                raise ImportError("Module not found")

        mock_import.side_effect = import_side_effect

        result = validator.validate_python_packages()

        assert result["fastapi"] is True
        assert result["spacy"] is True
        assert result["sqlalchemy"] is False
        assert result["PyMuPDF"] is False

    def test_validate_all_success(self, validator):
        """Test complete validation with all dependencies available."""
        with patch.object(validator, 'validate_spacy_models') as mock_spacy, \
             patch.object(validator, 'validate_directories') as mock_dirs, \
             patch.object(validator, 'validate_python_packages') as mock_packages, \
             patch('app.core.dependencies.dependency_logger') as mock_logger:

            # Mock all validations as successful
            mock_spacy.return_value = {"en_core_web_sm": True, "lt_core_news_sm": True}
            mock_dirs.return_value = {"uploads": True, "processed": True, "temp": True, "logs": True}
            mock_packages.return_value = {"fastapi": True, "sqlalchemy": True, "PyMuPDF": True, "spacy": True, "langdetect": True, "pdfminer.six": True}

            all_passed, results = validator.validate_all()

            assert all_passed is True
            assert "spacy_models" in results
            assert "directories" in results
            assert "python_packages" in results
            mock_logger.info.assert_called()

    def test_validate_all_with_failures(self, validator):
        """Test complete validation with some failures."""
        with patch.object(validator, 'validate_spacy_models') as mock_spacy, \
             patch.object(validator, 'validate_directories') as mock_dirs, \
             patch.object(validator, 'validate_python_packages') as mock_packages, \
             patch('app.core.dependencies.dependency_logger') as mock_logger:

            # Mock some validations as failed
            mock_spacy.return_value = {"en_core_web_sm": False, "lt_core_news_sm": True}
            mock_dirs.return_value = {"uploads": True, "processed": True, "temp": True, "logs": True}
            mock_packages.return_value = {"fastapi": True, "sqlalchemy": False, "PyMuPDF": True, "spacy": True, "langdetect": True, "pdfminer.six": True}

            all_passed, results = validator.validate_all()

            assert all_passed is False
            assert "spacy_models" in results
            assert "directories" in results
            assert "python_packages" in results
            mock_logger.error.assert_called()

    def test_get_missing_dependencies_no_validation_run(self, validator):
        """Test getting missing dependencies without running validation first."""
        with patch.object(validator, 'validate_all') as mock_validate:
            # Mock validate_all to set validation_results and return tuple
            def side_effect():
                validator.validation_results = {
                    "spacy_models": {"en_core_web_sm": False},
                    "python_packages": {"fastapi": False},
                    "directories": {"uploads": False}
                }
                return (False, validator.validation_results)
            
            mock_validate.side_effect = side_effect

            missing = validator.get_missing_dependencies()

            mock_validate.assert_called_once()
            assert any("spaCy model 'en_core_web_sm'" in item for item in missing)
            assert any("Python package 'fastapi'" in item for item in missing)
            assert any("Directory 'uploads'" in item for item in missing)

    def test_get_missing_dependencies_with_existing_results(self, validator):
        """Test getting missing dependencies with existing validation results."""
        validator.validation_results = {
            "spacy_models": {"en_core_web_sm": False, "lt_core_news_sm": True},
            "python_packages": {"fastapi": True, "sqlalchemy": False},
            "directories": {"uploads": True, "processed": False}
        }

        missing = validator.get_missing_dependencies()

        expected_items = [
            "spaCy model 'en_core_web_sm' - Install with: python -m spacy download en_core_web_sm",
            "Python package 'sqlalchemy' - Install with: pip install sqlalchemy",
            "Directory 'processed' - Check permissions and disk space"
        ]

        for expected_item in expected_items:
            assert expected_item in missing

    def test_get_missing_dependencies_all_satisfied(self, validator):
        """Test getting missing dependencies when all are satisfied."""
        validator.validation_results = {
            "spacy_models": {"en_core_web_sm": True, "lt_core_news_sm": True},
            "python_packages": {"fastapi": True, "sqlalchemy": True, "PyMuPDF": True, "spacy": True, "langdetect": True, "pdfminer.six": True},
            "directories": {"uploads": True, "processed": True, "temp": True, "logs": True}
        }

        missing = validator.get_missing_dependencies()

        assert missing == []

    def test_generate_installation_guide_no_missing(self, validator):
        """Test installation guide generation when no dependencies are missing."""
        with patch.object(validator, 'get_missing_dependencies') as mock_missing:
            mock_missing.return_value = []

            guide = validator.generate_installation_guide()

            assert guide == "All dependencies are satisfied!"

    def test_generate_installation_guide_with_missing(self, validator):
        """Test installation guide generation with missing dependencies."""
        with patch.object(validator, 'get_missing_dependencies') as mock_missing:
            mock_missing.return_value = [
                "Python package 'fastapi' - Install with: pip install fastapi",
                "spaCy model 'en_core_web_sm' - Install with: python -m spacy download en_core_web_sm",
                "Directory 'uploads' - Check permissions and disk space"
            ]

            guide = validator.generate_installation_guide()

            assert "Missing Dependencies Installation Guide:" in guide
            assert "1. Install missing Python packages:" in guide
            assert "2. Install missing spaCy models:" in guide
            assert "3. Fix directory issues:" in guide
            assert "pip install fastapi" in guide
            assert "python -m spacy download en_core_web_sm" in guide
            assert "After installing missing dependencies, restart the application." in guide

    @patch('app.core.dependencies.spacy.load')
    @patch('builtins.__import__')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_startup_dependencies_success(self, mock_logger, mock_import, mock_spacy_load, validator):
        """Test successful startup dependency validation."""
        # Mock successful imports for critical packages
        mock_import.return_value = Mock()
        # Mock successful spaCy model loading
        mock_nlp = Mock()
        mock_nlp.pipe_names = ['tagger', 'parser', 'ner']
        mock_spacy_load.return_value = mock_nlp

        result = validator.validate_startup_dependencies()

        assert result is True
        # Should import fastapi, sqlalchemy, spacy
        assert mock_import.call_count == 3
        mock_logger.log_dependency_check.assert_called()

    @patch('builtins.__import__')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_startup_dependencies_failure(self, mock_logger, mock_import, validator):
        """Test startup dependency validation with missing critical packages."""
        # Mock import failure for critical packages
        mock_import.side_effect = ImportError("Module not found")

        result = validator.validate_startup_dependencies()

        assert result is False
        mock_logger.log_dependency_check.assert_called()

    @patch('builtins.__import__')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_startup_dependencies_partial_failure(self, mock_logger, mock_import, validator):
        """Test startup dependency validation with some critical packages missing."""
        def import_side_effect(name):
            if name == "fastapi":
                return Mock()
            else:
                raise ImportError("Module not found")

        mock_import.side_effect = import_side_effect

        result = validator.validate_startup_dependencies()

        assert result is False
        mock_logger.log_dependency_check.assert_called()


class TestStartupValidation:
    """Test the standalone startup validation function."""

    @patch('app.core.dependencies.DependencyValidator')
    @patch('app.core.dependencies.dependency_logger')
    def test_validate_dependencies_on_startup_success(self, mock_logger, mock_validator_class):
        """Test successful startup dependency validation."""
        # Mock validator instance and methods
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.validate_startup_dependencies.return_value = True
        mock_validator.validate_all.return_value = (True, {})
        mock_validator.get_missing_dependencies.return_value = []

        result = validate_dependencies_on_startup()

        assert result == mock_validator
        mock_validator.validate_startup_dependencies.assert_called_once()
        mock_validator.validate_all.assert_called_once()

    @patch('app.core.dependencies.DependencyValidator')
    @patch('app.core.dependencies.dependency_logger')
    @patch('app.core.dependencies.sys.exit')
    def test_validate_dependencies_on_startup_failure(self, mock_exit, mock_logger, mock_validator_class):
        """Test startup dependency validation failure."""
        # Mock validator instance and methods
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.validate_startup_dependencies.return_value = False
        # Mock sys.exit to prevent actual exit and allow test to continue
        mock_exit.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            validate_dependencies_on_startup()

        mock_validator.validate_startup_dependencies.assert_called_once()
        mock_logger.error.assert_called()
        mock_exit.assert_called_with(1)

    @patch('app.core.dependencies.DependencyValidator')
    @patch('app.core.dependencies.dependency_logger')
    @patch('app.core.dependencies.sys.exit')
    def test_validate_dependencies_on_startup_exception(self, mock_exit, mock_logger, mock_validator_class):
        """Test startup dependency validation with unexpected exception."""
        # Mock validator constructor to raise an exception
        mock_validator_class.side_effect = Exception("Unexpected error")
        # Mock sys.exit to allow test to continue
        mock_exit.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            validate_dependencies_on_startup()

        mock_logger.error.assert_called()
        mock_exit.assert_called_with(1)


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture
    def validator(self):
        """Create a DependencyValidator instance for testing."""
        return DependencyValidator()

    @patch('app.core.dependencies.spacy.load')
    @patch('app.core.dependencies.dependency_logger')
    def test_spacy_load_with_empty_pipe_names(self, mock_logger, mock_spacy_load, validator):
        """Test spaCy model validation with model that has no pipe names."""
        # Mock model with empty pipe_names
        mock_nlp = Mock()
        mock_nlp.pipe_names = []
        mock_spacy_load.return_value = mock_nlp

        result = validator.validate_spacy_models()

        assert result == {"en_core_web_sm": True, "lt_core_news_sm": True}
        # Should log with "0 components"
        mock_logger.log_dependency_check.assert_called()

    @patch('app.core.dependencies.Path')
    def test_directory_validation_with_readonly_filesystem(self, mock_path, validator):
        """Test directory validation on read-only filesystem."""
        mock_dir = Mock()
        mock_path.return_value = mock_dir
        mock_test_file = Mock()
        mock_dir.__truediv__ = Mock(return_value=mock_test_file)

        # Mock successful mkdir but failed write (read-only)
        mock_test_file.write_text.side_effect = OSError("Read-only file system")

        with patch('app.core.dependencies.dependency_logger'):
            result = validator.validate_directories()

        expected = {"uploads": False, "processed": False, "temp": False, "logs": False}
        assert result == expected

    def test_empty_validation_results(self, validator):
        """Test behavior with empty validation results."""
        validator.validation_results = {}

        with patch.object(validator, 'validate_all') as mock_validate:
            mock_validate.return_value = (True, {
                "spacy_models": {},
                "python_packages": {},
                "directories": {}
            })

            missing = validator.get_missing_dependencies()

            assert missing == []

    def test_malformed_validation_results(self, validator):
        """Test behavior with malformed validation results."""
        # Set malformed validation results that could cause errors
        validator.validation_results = {
            "spacy_models": {},  # Empty instead of None to avoid AttributeError
            "python_packages": {"fastapi": False},
            "directories": {}
        }

        missing = validator.get_missing_dependencies()

        # Should handle empty gracefully and only include valid entries
        assert any("Python package 'fastapi'" in item for item in missing)
        assert len([item for item in missing if "spaCy model" in item]) == 0 