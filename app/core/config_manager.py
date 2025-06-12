from pathlib import Path
import yaml
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from app.core.logging import StructuredLogger

config_logger = StructuredLogger("config_manager")


class ConfigManager:
    """
    Manages configuration for PII patterns and application settings.
    Supports loading from YAML files with validation and fallback to defaults.
    """

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.config_dir.mkdir(exist_ok=True)

        self.patterns_file = self.config_dir / "patterns.yaml"
        self.cities_file = self.config_dir / "cities.yaml"
        self.settings_file = self.config_dir / "settings.yaml"

        # Initialize configuration
        self.patterns = self.load_patterns()
        self.cities = self.load_cities()
        self.settings = self.load_settings()

        config_logger.info(
            "Configuration manager initialized",
            config_dir=str(self.config_dir),
            patterns_count=len(self.patterns),
            cities_count=len(self.cities),
        )

    def load_patterns(self) -> Dict[str, str]:
        """Load PII patterns from YAML file with fallback to defaults."""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    patterns = config.get("pii_patterns", {})

                    # Validate patterns
                    validated_patterns = self.validate_patterns(patterns)

                    config_logger.info(
                        "Patterns loaded from file",
                        file=str(self.patterns_file),
                        patterns_count=len(validated_patterns),
                    )
                    return validated_patterns
            else:
                config_logger.info(
                    "Patterns file not found, creating default", file=str(self.patterns_file)
                )
                default_patterns = self.get_default_patterns()
                self.save_patterns(default_patterns)
                return default_patterns

        except Exception as e:
            config_logger.error(
                "Failed to load patterns, using defaults",
                error=str(e),
                file=str(self.patterns_file),
            )
            return self.get_default_patterns()

    def load_cities(self) -> List[str]:
        """Load Lithuanian cities from YAML file with fallback to defaults."""
        try:
            if self.cities_file.exists():
                with open(self.cities_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    cities = config.get("lithuanian_cities", [])

                    config_logger.info(
                        "Cities loaded from file",
                        file=str(self.cities_file),
                        cities_count=len(cities),
                    )
                    return cities
            else:
                config_logger.info(
                    "Cities file not found, creating default", file=str(self.cities_file)
                )
                default_cities = self.get_default_cities()
                self.save_cities(default_cities)
                return default_cities

        except Exception as e:
            config_logger.error(
                "Failed to load cities, using defaults", error=str(e), file=str(self.cities_file)
            )
            return self.get_default_cities()

    def load_settings(self) -> Dict[str, Any]:
        """Load application settings from YAML file with fallback to defaults."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    settings = yaml.safe_load(f)

                    config_logger.info("Settings loaded from file", file=str(self.settings_file))
                    return settings or {}
            else:
                config_logger.info(
                    "Settings file not found, creating default", file=str(self.settings_file)
                )
                default_settings = self.get_default_settings()
                self.save_settings(default_settings)
                return default_settings

        except Exception as e:
            config_logger.error(
                "Failed to load settings, using defaults",
                error=str(e),
                file=str(self.settings_file),
            )
            return self.get_default_settings()

    def validate_patterns(self, patterns: Dict[str, str]) -> Dict[str, str]:
        """Validate regex patterns and return only valid ones."""
        validated = {}

        for name, pattern in patterns.items():
            try:
                # Test if the pattern compiles
                re.compile(pattern)
                validated[name] = pattern
                config_logger.debug(
                    "Pattern validated",
                    pattern_name=name,
                    pattern=pattern[:50] + "..." if len(pattern) > 50 else pattern,
                )
            except re.error as e:
                config_logger.warning(
                    "Invalid regex pattern skipped",
                    pattern_name=name,
                    pattern=pattern,
                    error=str(e),
                )

        return validated

    def get_default_patterns(self) -> Dict[str, str]:
        """Return default PII patterns."""
        return {
            # Contact Information
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "phone_international": r"\+\d{1,3}\s+\d{3}\s+\d{5,6}\b",
            # Lithuanian Phone Numbers
            "lithuanian_mobile_prefixed": r"Tel\.\s*(?:nr\.\s*:?\s*)?\+370\s+\d{3}\s+\d{5}\b",
            "lithuanian_phone_generic": r"\+370\s+\d{3}\s+\d{5}\b",
            "lithuanian_phone_compact": r"\+370\d{8}\b",
            # Lithuanian Business Information
            "lithuanian_vat_code_labeled": r"PVM\s+kodas:?\s*LT\d{9,12}",
            "lithuanian_vat_code": r"\bLT\d{9,12}\b",
            "lithuanian_iban": r"\bLT\d{18}\b",
            "lithuanian_business_cert": r"\bAF\d{6}-\d\b",
            "lithuanian_business_cert_alt": r"\b\d{9}\b",
            # Lithuanian Personal Information
            "lithuanian_personal_code": r"\b[3-6]\d{10}\b",
            "lithuanian_passport": r"\b[A-Z]{2}\d{7}\b",
            "lithuanian_driver_license": r"\b[A-Z]{1,2}\d{6,7}[A-Z]?\b",
            # Lithuanian Addresses
            "lithuanian_address_prefixed": (
                r"Adresas:\s*[^,\n\r]+(?:g|pr|al)\.\s*[^,\n\r]+(?:,\s*LT-\d{5})?(?:,\s*[^\n\r]+)?"
            ),
            "lithuanian_address_generic": (
                r"\b[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]{2,}\s+(?:g|pr|al)\.\s*\d+[A-Za-z]?"
                r"(?:-\d+)?(?:,\s*LT-\d{5})?(?:,\s*[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž\s]+)?"
            ),
            "lithuanian_postal_code": r"\bLT-\d{5}\b",
            # Healthcare & Medical
            "health_insurance_number": r"\b\d{6,12}\b",
            "blood_group": r"(?<!\w)(?:A|B|AB|O)[\+\-](?!\w)",
            "medical_record_number": r"\b\d{6,10}\b",
            # Automotive
            "lithuanian_car_plate": r"\b[A-Z]{3}[-\s]?\d{3}\b",
            # Morning Session 5 Improvements - Enhanced car plate detection
            "lithuanian_car_plate_contextual": r"(?i)[Vv]alst\.?\s*[Nn]r\.?\s*[:\-]?\s*([A-Z]{3}[-\s]?\d{3})\b",
            "lithuanian_car_plate_enhanced": r"(?i)(?:valst\.?\s*nr\.?|automobilio\s+nr\.?|numeris)[\s:–-]*([A-Z]{3}[-\s]?\d{3})\b",
            # Financial
            "swift_bic": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
            "iban_eu": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
            "credit_card_enhanced": (
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
                r"6(?:011|5[0-9]{2})[0-9]{12})\b"
            ),
            # Legal
            "legal_entity_code": r"\b\d{8,9}\b",
            # Dates
            "date_yyyy_mm_dd": r"\b\d{4}-\d{2}-\d{2}\b",
            "date_yyyy_mm_dd_dots": r"\d{4}\.\d{2}\.\d{2}",
            # Generic patterns
            "eleven_digit_numeric": r"\b\d{11}\b",
            "ssn": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
            "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            # Morning Session 5 Improvements - Enhanced Lithuanian name detection
            "lithuanian_name_all_caps": r"\b([A-ZČĘĖĮŠŲŪŽ]{2,}(?:IENĖ|AITĖ|YTĖ|UTĖ|ŪTĖ|AS|IS|YS|US|IUS|Ė|A))\s+([A-ZČĘĖĮŠŲŪŽ]{2,}(?:IENĖ|AITĖ|YTĖ|UTĖ|ŪTĖ|AS|IS|YS|US|IUS|Ė|A))|\b([A-ZČĘĖĮŠŲŪŽ]{2,})\s+([A-ZČĘĖĮŠŲŪŽ]{2,}(?:IENĖ|AITĖ|YTĖ|UTĖ|ŪTĖ|AS|IS|YS|US|IUS|Ė|A))\b",
            "lithuanian_name_contextual": r"(?:Draudėjas|Vardas|Pavardė|Sutartį\s+sudarė)[\s:–-]*([A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+(?:\s+[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+)*(?:ienė|aitė|ytė|utė|ūtė|as|is|ys|us|ius|ė|a))\b",
            "lithuanian_name_contextual_caps": r"(?:Draudėjas|DRAUDĖJAS|Vardas|VARDAS)[\s:–-]*([A-ZČĘĖĮŠŲŪŽ]{3,}\s+[A-ZČĘĖĮŠŲŪŽ]{3,})\b",
            # Morning Session 5 Improvements - Enhanced address detection
            "lithuanian_address_flexible": r"\b([A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]{3,}(?:io|ės)?)\s*(?:g\.|gatvė|pr\.|prospektas|al\.|alėja)?(?:\s*\d+(?:[A-Za-z]?(?:-\d+)?)?)?",
            # Morning Session 5 Improvements - Enhanced personal code with context
            "lithuanian_personal_code_contextual": r"(?:asmens\s+kodas|asmens/įmonės\s+kodas|A\.K\.)[\s:–-]*(\d{11})\b",
            # A generic pattern to catch potential unknown IDs for the adaptive system
            "UNKNOWN_ID": r"\b[A-Z]{2,5}(?:-|\s)?\d{4,}\b"
        }

    def get_default_cities(self) -> List[str]:
        """Return default Lithuanian cities list."""
        return [
            # Major cities
            "Vilnius",
            "Kaunas",
            "Klaipėda",
            "Šiauliai",
            "Panevėžys",
            "Alytus",
            "Marijampolė",
            "Mažeikiai",
            "Jonava",
            "Utena",
            "Kėdainiai",
            "Telšiai",
            "Tauragė",
            "Ukmergė",
            "Visaginas",
            "Plungė",
            "Kretinga",
            "Radviliškis",
            "Palanga",
            "Druskininkai",
            "Gargždai",
            "Rokiškis",
            "Biržai",
            "Jurbarkas",
            "Elektrėnai",
            "Kuršėnai",
            "Garliava",
            "Vilkaviškis",
            "Anykščiai",
            "Lentvaris",
            "Prienai",
            "Joniškis",
            "Šilutė",
            "Šilalė",
            "Raseiniai",
            "Kelmė",
            "Varėna",
            "Kaišiadorys",
            "Pasvalys",
            "Kupiškis",
            "Zarasai",
            "Širvintos",
            "Molėtai",
            "Ignalina",
            "Švenčionys",
            "Trakai",
            "Šakiai",
            "Kazlų Rūda",
            "Vilkija",
            "Grigiškės",
            "Nemenčinė",
            # Districts and regions
            "Vilniaus",
            "Kauno",
            "Klaipėdos",
            "Šiaulių",
            "Panevėžio",
            "Alytaus",
            "Marijampolės",
            "Tauragės",
            "Telšių",
            "Utenos",
            # Common location suffixes
            "rajonas",
            "raj.",
            "sav.",
            "savivaldybė",
            "apskritis",
            "aps.",
            # Neighborhoods and areas in major cities
            "Antakalnis",
            "Žirmūnai",
            "Lazdynai",
            "Fabijoniškės",
            "Pilaitė",
            "Justiniškės",
            "Viršuliškės",
            "Šeškinė",
            "Naujamiestis",
            "Senamiestis",
            "Užupis",
            "Žvėrynas",
        ]

    def get_default_settings(self) -> Dict[str, Any]:
        """Return default application settings."""
        return {
            "version": "1.0.0",
            "language_detection": {"enabled": True, "sample_size": 1000},
            "processing": {"max_file_size_mb": 50, "timeout_seconds": 300, "temp_cleanup": True},
            "logging": {"level": "INFO", "max_file_size_mb": 10, "backup_count": 5},
            "patterns": {
                "case_sensitive": False,
                "word_boundaries": True,
                "validate_on_load": True,
            },
        }

    def save_patterns(self, patterns: Dict[str, str]) -> bool:
        """Save patterns to YAML file."""
        try:
            config_data = {
                "metadata": {
                    "version": "1.0.0",
                    "last_updated": datetime.now().isoformat(),
                    "description": "PII detection patterns for AnonymPDF",
                },
                "pii_patterns": patterns,
            }

            with open(self.patterns_file, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            config_logger.info(
                "Patterns saved successfully",
                file=str(self.patterns_file),
                patterns_count=len(patterns),
            )
            return True

        except Exception as e:
            config_logger.error(
                "Failed to save patterns", error=str(e), file=str(self.patterns_file)
            )
            return False

    def save_cities(self, cities: List[str]) -> bool:
        """Save cities to YAML file."""
        try:
            config_data = {
                "metadata": {
                    "version": "1.0.0",
                    "last_updated": datetime.now().isoformat(),
                    "description": "Lithuanian cities and locations for PII detection",
                },
                "lithuanian_cities": cities,
            }

            with open(self.cities_file, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            config_logger.info(
                "Cities saved successfully", file=str(self.cities_file), cities_count=len(cities)
            )
            return True

        except Exception as e:
            config_logger.error("Failed to save cities", error=str(e), file=str(self.cities_file))
            return False

    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to YAML file."""
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                yaml.dump(settings, f, default_flow_style=False, allow_unicode=True)

            config_logger.info("Settings saved successfully", file=str(self.settings_file))
            return True

        except Exception as e:
            config_logger.error(
                "Failed to save settings", error=str(e), file=str(self.settings_file)
            )
            return False

    def reload_configuration(self) -> bool:
        """Reload all configuration from files."""
        try:
            self.patterns = self.load_patterns()
            self.cities = self.load_cities()
            self.settings = self.load_settings()

            config_logger.info(
                "Configuration reloaded successfully",
                patterns_count=len(self.patterns),
                cities_count=len(self.cities),
            )
            return True

        except Exception as e:
            config_logger.error("Failed to reload configuration", error=str(e))
            return False

    def backup_configuration(self, backup_dir: Optional[Path] = None) -> bool:
        """Create backup of current configuration."""
        try:
            if backup_dir is None:
                backup_dir = self.config_dir / "backups"

            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Backup each config file
            for config_file in [self.patterns_file, self.cities_file, self.settings_file]:
                if config_file.exists():
                    backup_file = backup_dir / f"{config_file.stem}_{timestamp}.yaml"
                    backup_file.write_text(
                        config_file.read_text(encoding="utf-8"), encoding="utf-8"
                    )

            config_logger.info(
                "Configuration backed up successfully",
                backup_dir=str(backup_dir),
                timestamp=timestamp,
            )
            return True

        except Exception as e:
            config_logger.error("Failed to backup configuration", error=str(e))
            return False

    def get_pattern_categories(self) -> Dict[str, List[str]]:
        """Return patterns organized by categories."""
        categories = {
            "personal": [],
            "business": [],
            "location": [],
            "contact": [],
            "financial": [],
            "healthcare": [],
            "automotive": [],
            "legal": [],
            "dates": [],
            "generic": [],
        }

        for pattern_name in self.patterns.keys():
            if any(
                keyword in pattern_name.lower() for keyword in ["personal", "passport", "driver"]
            ):
                categories["personal"].append(pattern_name)
            elif any(keyword in pattern_name.lower() for keyword in ["vat", "business", "entity"]):
                categories["business"].append(pattern_name)
            elif any(keyword in pattern_name.lower() for keyword in ["address", "postal", "city"]):
                categories["location"].append(pattern_name)
            elif any(keyword in pattern_name.lower() for keyword in ["phone", "email", "mobile"]):
                categories["contact"].append(pattern_name)
            elif any(
                keyword in pattern_name.lower()
                for keyword in ["iban", "credit", "swift", "financial"]
            ):
                categories["financial"].append(pattern_name)
            elif any(keyword in pattern_name.lower() for keyword in ["health", "medical", "blood"]):
                categories["healthcare"].append(pattern_name)
            elif any(keyword in pattern_name.lower() for keyword in ["car", "plate", "automotive"]):
                categories["automotive"].append(pattern_name)
            elif any(keyword in pattern_name.lower() for keyword in ["legal", "entity"]):
                categories["legal"].append(pattern_name)
            elif any(keyword in pattern_name.lower() for keyword in ["date"]):
                categories["dates"].append(pattern_name)
            else:
                categories["generic"].append(pattern_name)

        return categories

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate entire configuration and return status with error messages."""
        errors = []

        # Validate patterns
        for name, pattern in self.patterns.items():
            try:
                re.compile(pattern)
            except re.error as e:
                errors.append(f"Invalid pattern '{name}': {str(e)}")

        # Validate cities (basic check)
        if not isinstance(self.cities, list):
            errors.append("Cities configuration must be a list")
        elif len(self.cities) == 0:
            errors.append("Cities list is empty")

        # Validate settings structure
        required_settings = ["version", "language_detection", "processing", "logging", "patterns"]
        for setting in required_settings:
            if setting not in self.settings:
                errors.append(f"Missing required setting: {setting}")

        is_valid = len(errors) == 0

        if is_valid:
            config_logger.info("Configuration validation passed")
        else:
            config_logger.warning(
                "Configuration validation failed", errors_count=len(errors), errors=errors
            )

        return is_valid, errors

    def get_user_config(self) -> Dict[str, Any]:
        """Return the user-defined configuration."""
        # ... existing code ...


# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> Dict[str, Any]:
    """Get configuration dictionary for easy access."""
    config_manager = get_config_manager()
    return {
        'patterns': config_manager.patterns,
        'cities': config_manager.cities,
        'settings': config_manager.settings,
        
        # Priority 3 specific configuration
        'ml_engine': {
            'model_type': 'xgboost',
            'model_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'confidence_calibration': {
                'method': 'isotonic',
                'cv_folds': 3
            },
            'training': {
                'test_size': 0.2,
                'cv_folds': 5
            }
        },
        'feature_engineering': {
            'context_features': {
                'window_size': 50
            },
            'linguistic_features': {
                'spacy_models': ['lt_core_news_sm', 'en_core_web_sm']
            }
        },
        'training_data': {
            'synthetic_generation': {
                'default_count': 500,
                'balance_ratio': 0.5
            },
            'data_sources': ['priority2', 'feedback', 'synthetic']
        },
        'performance': {
            'parallel_processing': {
                'max_workers': 4,
                'chunk_size': 100
            },
            'caching': {
                'max_size': 1000,
                'ttl_seconds': 3600
            }
        }
    }
