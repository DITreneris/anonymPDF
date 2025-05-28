from typing import List, Tuple, Dict
from pdfminer.high_level import extract_text
from fastapi import HTTPException
from pathlib import Path
import spacy
import re
from datetime import datetime
from langdetect import detect, LangDetectException
from app.pdf_processor import redact_pdf
import time
from app.core.logging import pdf_logger
from app.core.config_manager import ConfigManager
from app.core.performance import file_processing_metrics, performance_monitor


class PDFProcessor:
    def __init__(self):
        pdf_logger.info("Initializing PDF processor")

        # Initialize configuration manager
        self.config_manager = ConfigManager()
        pdf_logger.info(
            "Configuration manager loaded",
            patterns_count=len(self.config_manager.patterns),
            cities_count=len(self.config_manager.cities),
        )

        # Load English model by default
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            pdf_logger.info(
                "English spaCy model loaded successfully",
                model="en_core_web_sm",
                components=len(self.nlp_en.pipe_names),
            )
        except OSError as e:
            pdf_logger.error("Failed to load English spaCy model", error=str(e))
            raise RuntimeError("Critical dependency missing: en_core_web_sm spaCy model")

        # Attempt to load Lithuanian model, handle if not found
        try:
            self.nlp_lt = spacy.load("lt_core_news_sm")
            pdf_logger.info(
                "Lithuanian spaCy model loaded successfully",
                model="lt_core_news_sm",
                components=len(self.nlp_lt.pipe_names),
            )
        except OSError as e:
            pdf_logger.warning(
                "Lithuanian spaCy model not found - Lithuanian detection will be limited",
                model="lt_core_news_sm",
                error=str(e),
            )
            self.nlp_lt = None

        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        self.processed_dir = Path("processed")
        self.processed_dir.mkdir(exist_ok=True)

    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        try:
            # Use a sample of the text for detection
            sample = text[:1000] if len(text) > 1000 else text
            detected_lang = detect(sample)
            pdf_logger.info("Language detected", language=detected_lang, sample_length=len(sample))
            return detected_lang
        except LangDetectException as e:
            pdf_logger.warning("Language detection failed", error=str(e))
            return "unknown"

    @performance_monitor("pii_detection")
    def find_personal_info(
        self, text: str, language: str = "en"
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Find personal information in text using spaCy NER and regex patterns.
        Uses language-specific NLP model if available.
        """

        nlp_to_use = self.nlp_en  # Default to English
        if language == "lt" and self.nlp_lt:
            nlp_to_use = self.nlp_lt
            pdf_logger.info("Using Lithuanian NLP model for processing")
        elif language == "lt" and not self.nlp_lt:
            pdf_logger.warning(
                "Lithuanian model not available, using English NLP for Lithuanian text"
            )
        else:
            pdf_logger.info("Using English NLP model for processing", language=language)

        doc = nlp_to_use(text)
        personal_info = {
            "names": [],
            "locations": [],  # SpaCy GPE can also find locations/cities
            "organizations": [],
            "emails": [],
            "phones": [],  # For generic phone numbers
            "phones_international": [],  # For international phone numbers
            "mobile_phones_prefixed": [],  # For Tel. nr.: ...
            "lithuanian_phones_generic": [],  # For +370 XXX XXXXX format
            "lithuanian_phones_compact": [],  # For +370XXXXXXXX format
            "addresses_prefixed": [],  # For Adresas: ...
            "lithuanian_personal_codes": [],
            "lithuanian_vat_codes": [],  # Added for VAT codes
            "identity_documents": [],  # Passports, driver licenses
            "healthcare_medical": [],  # Health insurance, blood groups, medical records
            "automotive": [],  # Car plates
            "financial_enhanced": [],  # SWIFT/BIC, enhanced credit cards, IBANs
            "legal_entities": [],  # Legal entity codes
            "eleven_digit_numerics": [],
            "dates_yyyy_mm_dd": [],
            "ssns": [],
            "credit_cards": [],
        }

        # Extract entities using spaCy (PERSON, GPE, ORG)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                personal_info["names"].append((ent.text, ent.label_))
            elif ent.label_ == "GPE":
                personal_info["locations"].append((ent.text, ent.label_))
            elif ent.label_ == "ORG":
                personal_info["organizations"].append((ent.text, ent.label_))

        # Extract sensitive information using regex patterns from configuration
        for pattern_type, pattern in self.config_manager.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                if pattern_type == "email":
                    personal_info["emails"].append((match.group(), "EMAIL"))
                elif pattern_type == "phone":
                    personal_info["phones"].append((match.group(), "PHONE"))
                elif pattern_type == "phone_international":
                    personal_info["phones_international"].append(
                        (match.group(), "PHONE_INTERNATIONAL")
                    )
                elif pattern_type == "lithuanian_mobile_prefixed":
                    personal_info["mobile_phones_prefixed"].append(
                        (match.group(), "MOBILE_PHONE_PREFIXED")
                    )
                elif pattern_type == "lithuanian_phone_generic":
                    personal_info["lithuanian_phones_generic"].append(
                        (match.group(), "LITHUANIAN_PHONE_GENERIC")
                    )
                elif pattern_type == "lithuanian_phone_compact":
                    personal_info["lithuanian_phones_compact"].append(
                        (match.group(), "LITHUANIAN_PHONE_COMPACT")
                    )
                elif pattern_type == "lithuanian_address_prefixed":
                    personal_info["addresses_prefixed"].append((match.group(), "ADDRESS_PREFIXED"))
                elif pattern_type == "lithuanian_address_generic":
                    personal_info["addresses_prefixed"].append((match.group(), "ADDRESS_GENERIC"))
                elif pattern_type == "lithuanian_postal_code":
                    personal_info["addresses_prefixed"].append((match.group(), "POSTAL_CODE"))
                elif pattern_type == "lithuanian_personal_code":
                    personal_info["lithuanian_personal_codes"].append(
                        (match.group(), "LITHUANIAN_PERSONAL_CODE")
                    )
                elif (
                    pattern_type == "lithuanian_vat_code"
                    or pattern_type == "lithuanian_vat_code_labeled"
                ):
                    personal_info["lithuanian_vat_codes"].append(
                        (match.group(), "LITHUANIAN_VAT_CODE")
                    )
                elif pattern_type == "lithuanian_iban":
                    personal_info["lithuanian_vat_codes"].append((match.group(), "LITHUANIAN_IBAN"))
                elif (
                    pattern_type == "lithuanian_business_cert"
                    or pattern_type == "lithuanian_business_cert_alt"
                ):
                    personal_info["lithuanian_vat_codes"].append((match.group(), "BUSINESS_CERT"))
                elif pattern_type == "eleven_digit_numeric":
                    # Avoid double-counting if already caught as a personal code
                    is_already_personal_code = False
                    for pc_match, _ in personal_info["lithuanian_personal_codes"]:
                        if match.group() == pc_match:
                            is_already_personal_code = True
                            break
                    if not is_already_personal_code:
                        personal_info["eleven_digit_numerics"].append(
                            (match.group(), "ELEVEN_DIGIT_NUMERIC")
                        )
                elif pattern_type == "date_yyyy_mm_dd" or pattern_type == "date_yyyy_mm_dd_dots":
                    personal_info["dates_yyyy_mm_dd"].append((match.group(), "DATE"))
                elif pattern_type == "ssn":
                    personal_info["ssns"].append((match.group(), "SSN"))
                elif pattern_type == "credit_card":
                    personal_info["credit_cards"].append((match.group(), "CREDIT_CARD"))
                # New pattern categories
                elif pattern_type == "lithuanian_passport":
                    personal_info["identity_documents"].append(
                        (match.group(), "LITHUANIAN_PASSPORT")
                    )
                elif pattern_type == "lithuanian_driver_license":
                    personal_info["identity_documents"].append((match.group(), "DRIVER_LICENSE"))
                elif pattern_type == "health_insurance_number":
                    # Avoid conflict with personal codes and other specific patterns
                    is_already_detected = False
                    for pc_match, _ in personal_info["lithuanian_personal_codes"]:
                        if match.group() == pc_match:
                            is_already_detected = True
                            break
                    if not is_already_detected:
                        personal_info["healthcare_medical"].append(
                            (match.group(), "HEALTH_INSURANCE")
                        )
                elif pattern_type == "blood_group":
                    personal_info["healthcare_medical"].append((match.group(), "BLOOD_GROUP"))
                elif pattern_type == "medical_record_number":
                    # Avoid conflict with other specific numeric patterns
                    is_already_detected = False
                    for pc_match, _ in personal_info["lithuanian_personal_codes"]:
                        if match.group() == pc_match:
                            is_already_detected = True
                            break
                    for vat_match, _ in personal_info["lithuanian_vat_codes"]:
                        if match.group() in vat_match:
                            is_already_detected = True
                            break
                    if not is_already_detected:
                        personal_info["healthcare_medical"].append(
                            (match.group(), "MEDICAL_RECORD")
                        )
                elif pattern_type == "lithuanian_car_plate":
                    personal_info["automotive"].append((match.group(), "CAR_PLATE"))
                elif pattern_type == "swift_bic":
                    personal_info["financial_enhanced"].append((match.group(), "SWIFT_BIC"))
                elif pattern_type == "iban_eu":
                    personal_info["financial_enhanced"].append((match.group(), "IBAN_EU"))
                elif pattern_type == "credit_card_enhanced":
                    personal_info["financial_enhanced"].append(
                        (match.group(), "CREDIT_CARD_ENHANCED")
                    )
                elif pattern_type == "legal_entity_code":
                    # Avoid conflict with business certificates and other specific patterns
                    is_already_detected = False
                    for bc_match, _ in personal_info["lithuanian_vat_codes"]:
                        if match.group() in bc_match:
                            is_already_detected = True
                            break
                    if not is_already_detected:
                        personal_info["legal_entities"].append((match.group(), "LEGAL_ENTITY_CODE"))

        # Detect Lithuanian cities and locations
        city_detections = self.detect_lithuanian_cities(text)
        personal_info["locations"].extend(city_detections)

        return personal_info

    def detect_lithuanian_cities(self, text: str) -> List[Tuple[str, str]]:
        """Detect Lithuanian city names and locations."""
        city_detections = []

        # Create a pattern for all city names (case-insensitive) from configuration
        for city in self.config_manager.cities:
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(city) + r"\b"
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                city_detections.append((match.group(), "LITHUANIAN_LOCATION"))
                pdf_logger.info(
                    "Lithuanian location detected",
                    city=match.group(),
                    start=match.start(),
                    end=match.end(),
                )

        return city_detections

    def anonymize_pdf(self, input_path: Path, output_path: Path) -> Tuple[bool, Dict]:
        """Anonymize PDF by redacting personal information."""
        pdf_logger.info(
            "Starting PDF anonymization", input_path=str(input_path), output_path=str(output_path)
        )

        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(input_path)
            pdf_logger.info(
                "Text extracted from PDF", text_length=len(text), input_path=str(input_path)
            )

            # Detect language
            language = self.detect_language(text)

            # Find personal information using the detected language
            personal_info = self.find_personal_info(text, language=language)

            # Log PII detection results
            total_pii_found = sum(len(items) for items in personal_info.values())
            pdf_logger.info(
                "PII detection completed",
                language=language,
                total_pii_items=total_pii_found,
                categories_found=len([k for k, v in personal_info.items() if v]),
            )

            # Collect all sensitive words
            sensitive_words = []
            for category_items in personal_info.values():
                for item_text, _ in category_items:
                    sensitive_words.append(item_text)

            pdf_logger.info(
                "Sensitive words collected for redaction",
                sensitive_words_count=len(sensitive_words),
            )

            # Redact the PDF using PyMuPDF
            redaction_successful = redact_pdf(str(input_path), str(output_path), sensitive_words)

            if not redaction_successful:
                pdf_logger.error(
                    "PDF redaction failed",
                    input_path=str(input_path),
                    output_path=str(output_path),
                    sensitive_words_count=len(sensitive_words),
                )
                return False, {
                    "error": "PDF redaction failed",
                    "details": "redact_pdf returned False",
                }

            # Generate report of redacted information
            report = self.generate_redaction_report(personal_info, language)

            pdf_logger.info(
                "PDF anonymization completed successfully",
                input_path=str(input_path),
                output_path=str(output_path),
                total_redactions=report.get("totalRedactions", 0),
            )

            return True, report

        except Exception as e:
            pdf_logger.log_error(
                "anonymize_pdf", e, input_path=str(input_path), output_path=str(output_path)
            )
            return False, {"error": "An exception occurred during anonymization", "details": str(e)}

    def generate_redaction_report(
        self, personal_info: Dict[str, List[Tuple[str, str]]], language: str
    ) -> Dict:
        """Generate a structured report of redacted information."""
        total_redactions = 0
        categories = {}

        for category, items in personal_info.items():
            if items:
                category_name = category.upper()
                categories[category_name] = len(items)
                total_redactions += len(items)

        report_data = {
            "title": f"Redaction Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "detectedLanguage": language,
            "totalRedactions": total_redactions,
            "categories": categories,
            "details": personal_info,  # Include raw details for potential future use or deeper inspection
        }
        return report_data

    async def process_pdf(self, file_path: str) -> dict:
        """Process PDF file at file_path and extract text."""
        start_time = time.time()

        pdf_logger.info("Starting PDF processing", filename=file_path)

        if not str(file_path).lower().endswith(".pdf"):
            pdf_logger.error("Invalid file type for processing", filename=file_path)
            raise HTTPException(status_code=400, detail="File must be a PDF")

        file_path_obj = Path(file_path)
        temp_path = self.temp_dir / file_path_obj.name

        # Start performance tracking
        file_size = 0
        performance_tracking = None

        try:
            # Copy the file to temp_path for processing
            with open(file_path_obj, "rb") as src, open(temp_path, "wb") as dst:
                content = src.read()
                dst.write(content)
                file_size = len(content)

            # Initialize performance tracking with file metrics
            performance_tracking = file_processing_metrics.track_file_processing(
                file_path_obj, file_size, "pdf_processing"
            )

            pdf_logger.info(
                "File copied to temp directory",
                filename=file_path_obj.name,
                temp_path=str(temp_path),
                file_size=file_size,
            )

            # Process the PDF
            processing_start = time.time()
            success, report_data = self.anonymize_pdf(
                temp_path, self.processed_dir / f"anonymized_{file_path_obj.name}"
            )
            processing_time = time.time() - processing_start

            if success:
                total_time = time.time() - start_time

                # Count total redactions from report
                redactions_count = report_data.get("totalRedactions", 0)

                pdf_logger.info(
                    "PDF processing completed successfully",
                    filename=file_path_obj.name,
                    processing_time=processing_time,
                    total_time=total_time,
                    redactions_count=redactions_count,
                )

                return {
                    "filename": file_path_obj.name,
                    "status": "processed",
                    "report": report_data,  # Return as dict, not JSON string
                    "processing_time": processing_time,
                    "redactions_count": redactions_count,
                    "metadata": {
                        "total_time": total_time,
                        "file_size": len(content),
                        "detected_language": report_data.get("detectedLanguage", "unknown"),
                    },
                }
            else:
                pdf_logger.error(
                    "PDF processing failed",
                    filename=file_path_obj.name,
                    error=report_data.get("error", "Unknown error"),
                    processing_time=processing_time,
                )

                return {
                    "filename": file_path_obj.name,
                    "status": "failed",
                    "error": report_data.get("error", "Unknown error during PDF processing"),
                    "details": report_data.get("details", "No additional details available"),
                    "processing_time": processing_time,
                }

        except Exception as e:
            total_time = time.time() - start_time
            pdf_logger.log_error(
                "process_pdf", e, filename=file_path_obj.name, processing_time=total_time
            )

            return {
                "filename": file_path_obj.name,
                "status": "failed",
                "error": "An exception occurred during PDF processing",
                "details": str(e),
                "processing_time": total_time,
            }

        finally:
            # End performance tracking
            if performance_tracking:
                performance_tracking["end_tracking"]()

            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
                pdf_logger.info("Temporary file cleaned up", temp_path=str(temp_path))

    def cleanup(self):
        """Clean up temporary files."""
        for file in self.temp_dir.glob("*"):
            if file.is_file():
                file.unlink()

    @performance_monitor("text_extraction")
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        return extract_text(str(pdf_path))

    def process_pdf_for_anonymization(self, file_path: Path) -> Tuple[bool, str]:
        """Process a PDF file for anonymization."""
        try:
            # Generate output filename
            output_filename = f"anonymized_{file_path.name}"
            output_path = self.processed_dir / output_filename

            # Process the PDF
            success, report_data = self.anonymize_pdf(file_path, output_path)

            if success:
                return True, str(output_path)
            else:
                return False, report_data.get("error", "Failed to anonymize PDF")

        except Exception as e:
            return False, str(e)
