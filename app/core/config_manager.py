from pathlib import Path
import yaml
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from app.core.logging import StructuredLogger
import threading
import warnings
import functools

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
        self.brand_names_file = self.config_dir / "brand_names.yaml"

        # Initialize configuration
        self.patterns = self.load_patterns()
        self.cities = self.load_cities()
        self.settings = self.load_settings()
        self.brand_names = self.load_brand_names()

        config_logger.info(
            "Configuration manager initialized",
            config_dir=str(self.config_dir),
            patterns_count=len(self.patterns),
            cities_count=len(self.cities),
        )

    def load_patterns(self) -> Dict[str, re.Pattern]:
        """Load PII patterns from YAML file, compile them, and return them."""
        # First, try to load from file if it exists.
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    patterns = config.get("pii_patterns", {})

                    # Compile and validate patterns
                    compiled_patterns = self.compile_and_validate_patterns(patterns)

                    config_logger.info(
                        "Patterns loaded and compiled from file",
                        file=str(self.patterns_file),
                        patterns_count=len(compiled_patterns),
                    )
                    return compiled_patterns
            except Exception as e:
                config_logger.error(
                    "Failed to load patterns from existing file, falling back to defaults.",
                    error=str(e),
                    file=str(self.patterns_file),
                )
        
        # This block is reached if file doesn't exist OR if loading from file failed.
        config_logger.info(
            "Loading default patterns because file was not found or failed to load."
        )
        default_patterns = self.get_default_patterns()
        
        try:
            # Attempt to save the defaults for next time, but don't fail if this doesn't work.
            # The user should still get a working system with default patterns.
            patterns_to_save = {k: v.pattern for k, v in default_patterns.items()}
            self.save_patterns(patterns_to_save)
        except Exception as e:
            config_logger.error(
                "Failed to save default patterns file. The application will use defaults for this session.",
                error=str(e),
                file=str(self.patterns_file),
            )
            
        return default_patterns

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

    def compile_and_validate_patterns(self, patterns: Dict[str, Any]) -> Dict[str, re.Pattern]:
        """
        Compile and validate regex patterns, returning only valid compiled objects.
        Accepts dicts with either string patterns or dicts with a 'regex' key.
        """
        compiled = {}

        for name, value in patterns.items():
            pattern_str = None
            if isinstance(value, str):
                pattern_str = value
            elif isinstance(value, dict) and 'regex' in value:
                pattern_str = value['regex']
            
            if not pattern_str:
                config_logger.warning("Pattern entry is malformed, skipping", pattern_name=name, pattern_value=value)
                continue

            try:
                # Let the pattern string define its own flags (e.g., (?i) for case-insensitivity)
                compiled_pattern = re.compile(pattern_str)
                compiled[name] = compiled_pattern
            except re.error as e:
                config_logger.warning(
                    "Invalid regex pattern skipped",
                    pattern_name=name,
                    pattern=pattern_str,
                    error=str(e),
                )

        return compiled

    def get_default_patterns(self) -> Dict[str, re.Pattern]:
        """Return default PII patterns as compiled regex objects."""
        raw_patterns = {
            # Contact Information
            "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "phone_international": r"\+\d{1,3}\s+\d{3}\s+\d{5,6}\b",
            # Lithuanian Phone Numbers
            "lithuanian_mobile_prefixed": r"Tel\.\s*(?:nr\.\s*:?\s*)?\+370\s+\d{3}\s+\d{5}\b",
            "lithuanian_phone_generic": r"\+370\s+\d{3}\s+\d{5}\b",
            "lithuanian_phone_compact": r"\+370\d{8}\b",
            # Lithuanian Business Information
            "lithuanian_vat_code_labeled": r"PVM\s+kodas:?\s*LT\d{9,12}",
            "lithuanian_vat_code": r"\bLT\d{9,12}\b",
            "lithuanian_iban": r"\bLT\d{2}(?:\s?\d{4}){4}\b",
            "lithuanian_business_cert": r"\bAF\d{6}-\d\b",
            "lithuanian_business_cert_alt": r"\b\d{9}\b",
            # Lithuanian Personal Information
            "lithuanian_personal_codes": r"\b[3-6]\d{10}\b",
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
            "blood_group": r"(?<!\w)(?:A|B|AB|O)[+\-](?!\w)",
            "medical_record_number": r"\b\d{6,10}\b",
            # Automotive
            "lithuanian_license_plate": r"\b[A-Z]{3}[-\s]?\d{3}\b",
            # Morning Session 5 Improvements - Enhanced car plate detection
            "lithuanian_car_plate_contextual": r"(?i)[Vv]alst\.?\s*[Nn]r\.?\s*[:\-]?\s*([A-Z]{3}[-\s]?\d{3})\b",
            "lithuanian_car_plate_enhanced": r"(?i)(?:valst\.?\s*nr\.?|automobilio\s+nr\.?|numeris)[:–-]*([A-Z]{3}[-\s]?\d{3})\b",
            # Financial
            "swift_bic": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
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
            "lithuanian_name_all_caps": r"\b([A-ZČĘĖĮŠŲŪŽ]{2,}(?:IENĖ|AITĖ|YTĖ|UTĖ|ŪTĖ|AS|IS|YS|US|IUS|Ė|A)\s+[A-ZČĘĖĮŠŲŪŽ]{2,}(?:IENĖ|AITĖ|YTĖ|UTĖ|ŪTĖ|AS|IS|YS|US|IUS|Ė|A))|\b([A-ZČĘĖĮŠŲŪŽ]{2,}\s+[A-ZČĘĖĮŠŲŪŽ]{2,}(?:IENĖ|AITĖ|YTĖ|UTĖ|ŪTĖ|AS|IS|YS|US|IUS|Ė|A))\b",
            "lithuanian_name_contextual": r"(?:Draudėjas|Vardas|Pavardė|Sutartį\s+sudarė)[:–-]?\s*([A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+(?:\s+[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+)*?)(?=\s*\n|\s*$|\s*,|\s*\.|asmens\s+kodas|tel\.|phone|email)",
            "lithuanian_name_simple": r"\b([A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+\s+[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+(?:ienė|aitė|ytė|utė|ūtė|as|is|ys|us|ius|ė|a)?)\b",
            "lithuanian_name_contextual_caps": r"(?:Draudėjas|DRAUDĖJAS|Vardas|VARDAS)[:–-]?\s*([A-ZČĘĖĮŠŲŪŽ]{3,}\s+[A-ZČĘĖĮŠŲŪŽ]{3,})\b",
            # Morning Session 5 Improvements - Enhanced address detection
            "lithuanian_address_flexible": r"\b([A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]{3,}(?:io|ės)?\s*(?:g\.|gatvė|pr\.|prospektas|al\.|alėja)?(?:\s*\d+(?:[A-Za-z]?(?:-\d+)?)?))",
            # Morning Session 5 Improvements - Enhanced personal code with context
            "lithuanian_personal_code_contextual": r"(?:asmens\s+kodas|asmens/įmonės\s+kodas|A\.K\.)[:–-]?\s*(\d{11})\b",
            # Lithuanian cities and locations
            "locations": r"\b(Vilni(?:us|aus|uje|ų)|Kaun(?:as|o|e|ą)|Klaipėd(?:a|os|oje|ą)|Šiauli(?:ai|ų|uose|us)|Panevėž(?:ys|io|yje|į))\b",
            # A generic pattern to catch potential unknown IDs for the adaptive system
            "UNKNOWN_ID": r"\b[A-Z]{2,5}(?:-|\s)?\d{4,}\b"
        }
        # We manually remove the old iban pattern if it exists to avoid conflicts.
        raw_patterns.pop("iban_eu", None)
        
        return self.compile_and_validate_patterns(raw_patterns)

    def load_brand_names(self) -> List[str]:
        """Load brand names from YAML file with fallback to defaults."""
        try:
            if self.brand_names_file.exists():
                with open(self.brand_names_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    brand_names = config.get("brand_names", [])
                    config_logger.info(
                        "Brand names loaded from file",
                        file=str(self.brand_names_file),
                        count=len(brand_names),
                    )
                    return brand_names
            else:
                config_logger.info(
                    "Brand names file not found, creating default", file=str(self.brand_names_file)
                )
                default_brand_names = self.get_default_brand_names()
                self.save_brand_names(default_brand_names)
                return default_brand_names
        except Exception as e:
            config_logger.error(
                "Failed to load brand names, using defaults", error=str(e), file=str(self.brand_names_file)
            )
            return self.get_default_brand_names()

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
            "Neringa",
            "Birštonas",
        ]

    def get_default_brand_names(self) -> List[str]:
        """Return a default list of common brand names to prevent over-redaction."""
        return [
            "Microsoft", "Windows", "Office", "Excel", "Word", "PowerPoint",
            "Apple", "iPhone", "iPad", "MacBook", "macOS",
            "Google", "Android", "Chrome", "Gmail", "Google Maps",
            "Amazon", "AWS", "Kindle", "Echo",
            "Facebook", "Instagram", "WhatsApp", "Meta",
            "Adobe", "Photoshop", "Acrobat", "Illustrator",
            "Oracle", "Java", "MySQL",
            "IBM", "Intel", "NVIDIA", "AMD",
            "Tesla", "SpaceX",
            "Toyota", "Honda", "Ford", "Volkswagen", "BMW", "Mercedes-Benz",
            "Samsung", "Sony", "LG",
            "Coca-Cola", "Pepsi",
            "Nike", "Adidas",
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
            "anti_overredaction": {
                "technical_terms_whitelist": [
                    "kW", "Nm", "g/km", "l/100 km", "mm", "kg", "cm³",
                    "CO2 emisijos", "Maks. variklio galia", "Degalų sąnaudos",
                    "SVORIS", "VAŽ.", "Pareigos", "Kaina", "Netto atlyginimas",
                    "Kitos pajamos", "Gyvenamoji vieta"
                ],
                "pii_field_labels": [
                    "Vardas", "Pavardė", "Vardas, pavardė", "Asmens kodas",
                    "Paso Nr", "Adresas", "Telefono numeris", "El. paštas",
                    "Banko sąskaita", "Asmens socialinio draudimo pažymėjimo Nr",
                    "Draudėjas", "Valst. Nr", "VIN"
                ],
                "technical_sections": [
                    "SVORIS", "VAŽ.", "Degalų sąnaudos", "Techniniai duomenys",
                    "Automobilio duomenys", "Variklio duomenys"
                ]
            },
            "adaptive_learning": {
                "enabled": True,
                "databases": {
                    "patterns_db": "data/adaptive/patterns.db",
                    "ab_tests_db": "data/adaptive/ab_tests.db",
                    "analytics_db": "data/adaptive/analytics.db"
                },
                "thresholds": {
                    "min_confidence_to_validate": 0.95,
                    "min_samples_for_learning": 10,
                    "ab_test_confidence_level": 0.95
                },
                "performance": {
                    "cache_size": 1000,
                    "cache_ttl_seconds": 3600,
                    "max_workers": 4
                },
                "monitoring": {
                    "log_learning_events": True,
                    "track_pattern_usage": True,
                    "alert_on_anomalies": True
                }
            },
            "performance": {
                "cache_size": 1000,
                "cache_ttl_seconds": 3600,
                "max_workers": 4,
                "parallel_processing": {
                    "max_workers": 4,
                    "chunk_size": 100
                },
                "batch_engine": {
                    "max_batch_size": 100,
                    "batch_timeout_seconds": 10
                }
            },
            "feedback_system": {
                "storage_path": "data/user_feedback.db",
                "min_feedback_for_retraining": 50,
                "auto_retrain_enabled": True
            }
        }

    def save_patterns(self, patterns: Dict[str, str]) -> bool:
        """Save patterns to YAML file."""
        try:
            # Ensure we are saving the raw string, not the compiled object
            patterns_to_save = {k: v if isinstance(v, str) else v.pattern for k, v in patterns.items()}
            with open(self.patterns_file, "w", encoding="utf-8") as f:
                yaml.dump({"pii_patterns": patterns_to_save}, f, allow_unicode=True, sort_keys=False)
            config_logger.info("Patterns saved successfully", file=str(self.patterns_file))
            return True
        except Exception as e:
            config_logger.error("Failed to save patterns", error=str(e), file=str(self.patterns_file))
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

    def save_brand_names(self, brand_names: List[str]) -> bool:
        """Save brand names list to YAML file."""
        try:
            with open(self.brand_names_file, "w", encoding="utf-8") as f:
                yaml.dump({"brand_names": brand_names}, f, allow_unicode=True, sort_keys=False)
            config_logger.info("Saved brand names to file", file=str(self.brand_names_file))
            return True
        except Exception as e:
            config_logger.error("Failed to save brand names", file=str(self.brand_names_file), error=str(e))
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
            self.brand_names = self.load_brand_names()

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
        required_settings = [
            "version", 
            "language_detection", 
            "processing", 
            "logging", 
            "patterns",
            "anti_overredaction",
            "adaptive_learning"
        ]
        for setting in required_settings:
            if setting not in self.settings:
                errors.append(f"Missing required setting: {setting}")

        # Validate adaptive learning settings
        if "adaptive_learning" in self.settings:
            adaptive = self.settings["adaptive_learning"]
            required_adaptive = ["enabled", "databases", "thresholds", "performance", "monitoring"]
            for setting in required_adaptive:
                if setting not in adaptive:
                    errors.append(f"Missing required adaptive learning setting: {setting}")

            # Validate database paths
            if "databases" in adaptive:
                for db_name, db_path in adaptive["databases"].items():
                    if not isinstance(db_path, str):
                        errors.append(f"Invalid database path for {db_name}: must be a string")
                    elif not db_path.endswith(".db"):
                        errors.append(f"Invalid database path for {db_name}: must end with .db")

            # Validate thresholds
            if "thresholds" in adaptive:
                thresholds = adaptive["thresholds"]
                for threshold_name, value in thresholds.items():
                    if not isinstance(value, (int, float)):
                        errors.append(f"Invalid threshold value for {threshold_name}: must be numeric")
                    elif value < 0 or value > 1:
                        errors.append(f"Invalid threshold value for {threshold_name}: must be between 0 and 1")

            # Validate performance settings
            if "performance" in adaptive:
                perf = adaptive["performance"]
                if not isinstance(perf.get("cache_size"), int) or perf["cache_size"] < 0:
                    errors.append("Invalid cache_size: must be a positive integer")
                if not isinstance(perf.get("cache_ttl_seconds"), int) or perf["cache_ttl_seconds"] < 0:
                    errors.append("Invalid cache_ttl_seconds: must be a positive integer")
                if not isinstance(perf.get("max_workers"), int) or perf["max_workers"] < 1:
                    errors.append("Invalid max_workers: must be a positive integer")

        is_valid = len(errors) == 0

        if is_valid:
            config_logger.info("Configuration validation passed")
        else:
            config_logger.warning(
                "Configuration validation failed",
                error_count=len(errors),
                errors=errors
            )

        return is_valid, errors

    def get_user_config(self) -> Dict[str, Any]:
        """Return the user-defined configuration."""
        return {
            'patterns': self.patterns,
            'cities': self.cities,
            'settings': self.settings,
        }


# --- Global Singleton Management ---

# Global config manager instance, initialized lazily.
_config_manager_instance = None
_config_lock = threading.Lock()

@functools.lru_cache(maxsize=1)
def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Returns a singleton instance of the ConfigManager.
    If a config_path is provided, it must be provided on the first call.
    """
    global _config_manager_instance
    if _config_manager_instance is None:
        with _config_lock:
            if _config_manager_instance is None:
                config_dir = Path(config_path) if config_path else Path("config")
                _config_manager_instance = ConfigManager(config_dir=config_dir)
    return _config_manager_instance

def get_config() -> Dict[str, Any]:
    """
    DEPRECATED: Returns the settings dictionary from the ConfigManager.
    
    This function is deprecated in favor of directly accessing `get_config_manager().settings`.
    It is maintained for backward compatibility but will be removed in a future version.
    """
    warnings.warn(
        "`get_config()` is deprecated. Please use `get_config_manager().settings` instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_config_manager().settings
