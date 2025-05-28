import spacy
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from app.core.logging import dependency_logger


class DependencyValidator:
    """Validates all required dependencies for AnonymPDF application."""

    def __init__(self):
        self.required_spacy_models = ["en_core_web_sm", "lt_core_news_sm"]
        self.required_directories = ["uploads", "processed", "temp", "logs"]
        self.validation_results = {}

    def validate_spacy_models(self) -> Dict[str, bool]:
        """Validate that required spaCy models are installed."""
        results = {}

        for model in self.required_spacy_models:
            try:
                nlp = spacy.load(model)
                results[model] = True
                dependency_logger.log_dependency_check(
                    dependency=model,
                    status="available",
                    details=f"Model loaded successfully with {len(nlp.pipe_names)} components",
                )
            except OSError as e:
                results[model] = False
                dependency_logger.log_dependency_check(
                    dependency=model, status="missing", details=f"Model not found: {str(e)}"
                )

        return results

    def validate_directories(self) -> Dict[str, bool]:
        """Ensure required directories exist and are writable."""
        results = {}

        for directory in self.required_directories:
            dir_path = Path(directory)
            try:
                # Create directory if it doesn't exist
                dir_path.mkdir(exist_ok=True, parents=True)

                # Test write permissions
                test_file = dir_path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()

                results[directory] = True
                dependency_logger.log_dependency_check(
                    dependency=f"directory_{directory}",
                    status="available",
                    details="Directory exists and is writable",
                )
            except Exception as e:
                results[directory] = False
                dependency_logger.log_dependency_check(
                    dependency=f"directory_{directory}",
                    status="failed",
                    details=f"Directory validation failed: {str(e)}",
                )

        return results

    def validate_python_packages(self) -> Dict[str, bool]:
        """Validate that required Python packages are available."""
        required_packages = [
            "fastapi",
            "sqlalchemy",
            "PyMuPDF",
            "spacy",
            "langdetect",
            "pdfminer.six",
        ]

        results = {}

        for package in required_packages:
            try:
                __import__(package.replace("-", "_").replace(".", "_"))
                results[package] = True
                dependency_logger.log_dependency_check(
                    dependency=f"package_{package}", status="available"
                )
            except ImportError as e:
                results[package] = False
                dependency_logger.log_dependency_check(
                    dependency=f"package_{package}",
                    status="missing",
                    details=f"Import failed: {str(e)}",
                )

        return results

    def validate_all(self) -> Tuple[bool, Dict[str, Dict[str, bool]]]:
        """Run all dependency validations."""
        dependency_logger.info("Starting comprehensive dependency validation")

        results = {
            "spacy_models": self.validate_spacy_models(),
            "directories": self.validate_directories(),
            "python_packages": self.validate_python_packages(),
        }

        # Check if all validations passed
        all_passed = True
        for category, category_results in results.items():
            for item, status in category_results.items():
                if not status:
                    all_passed = False
                    break
            if not all_passed:
                break

        self.validation_results = results

        if all_passed:
            dependency_logger.info("All dependency validations passed successfully")
        else:
            dependency_logger.error("Some dependency validations failed")

        return all_passed, results

    def get_missing_dependencies(self) -> List[str]:
        """Get a list of missing dependencies with installation instructions."""
        missing = []

        if not self.validation_results:
            self.validate_all()

        # Check spaCy models
        for model, status in self.validation_results.get("spacy_models", {}).items():
            if not status:
                missing.append(
                    f"spaCy model '{model}' - Install with: python -m spacy download {model}"
                )

        # Check Python packages
        for package, status in self.validation_results.get("python_packages", {}).items():
            if not status:
                missing.append(f"Python package '{package}' - Install with: pip install {package}")

        # Check directories
        for directory, status in self.validation_results.get("directories", {}).items():
            if not status:
                missing.append(f"Directory '{directory}' - Check permissions and disk space")

        return missing

    def generate_installation_guide(self) -> str:
        """Generate a comprehensive installation guide for missing dependencies."""
        missing = self.get_missing_dependencies()

        if not missing:
            return "All dependencies are satisfied!"

        guide = "Missing Dependencies Installation Guide:\n"
        guide += "=" * 50 + "\n\n"

        spacy_models = [item for item in missing if "spaCy model" in item]
        python_packages = [item for item in missing if "Python package" in item]
        directories = [item for item in missing if "Directory" in item]

        if python_packages:
            guide += "1. Install missing Python packages:\n"
            for package in python_packages:
                guide += f"   {package}\n"
            guide += "\n"

        if spacy_models:
            guide += "2. Install missing spaCy models:\n"
            for model in spacy_models:
                guide += f"   {model}\n"
            guide += "\n"

        if directories:
            guide += "3. Fix directory issues:\n"
            for directory in directories:
                guide += f"   {directory}\n"
            guide += "\n"

        guide += "After installing missing dependencies, restart the application.\n"

        return guide

    def validate_startup_dependencies(self) -> bool:
        """Validate critical dependencies required for application startup."""
        dependency_logger.info("Validating startup dependencies")

        critical_packages = ["fastapi", "sqlalchemy", "spacy"]

        for package in critical_packages:
            try:
                __import__(package)
                dependency_logger.log_dependency_check(
                    dependency=f"critical_package_{package}", status="available"
                )
            except ImportError:
                dependency_logger.log_dependency_check(
                    dependency=f"critical_package_{package}",
                    status="missing",
                    details="Critical package missing",
                )
                return False

        # Validate at least English spaCy model is available
        try:
            spacy.load("en_core_web_sm")
            dependency_logger.log_dependency_check(dependency="en_core_web_sm", status="available")
        except OSError:
            dependency_logger.log_dependency_check(
                dependency="en_core_web_sm",
                status="missing",
                details="Critical spaCy model missing",
            )
            return False

        return True


def validate_dependencies_on_startup():
    """Function to be called on application startup to validate dependencies."""
    validator = DependencyValidator()

    # First check critical dependencies
    if not validator.validate_startup_dependencies():
        error_msg = (
            "Critical dependencies are missing. The application cannot start.\n"
            "Please install the required dependencies:\n"
            "1. pip install fastapi sqlalchemy spacy\n"
            "2. python -m spacy download en_core_web_sm\n"
            "3. Restart the application"
        )
        dependency_logger.error(error_msg)
        print(f"ERROR: {error_msg}")
        sys.exit(1)

    # Then check all dependencies and warn about non-critical missing ones
    all_passed, results = validator.validate_all()

    if not all_passed:
        missing = validator.get_missing_dependencies()
        warning_msg = (
            "Some optional dependencies are missing. "
            "The application will start but some features may be limited.\n"
            f"Missing dependencies: {len(missing)}\n"
        )

        for item in missing:
            warning_msg += f"  - {item}\n"

        dependency_logger.warning(warning_msg)
        print(f"WARNING: {warning_msg}")

    return validator


# Global validator instance
dependency_validator = DependencyValidator()
