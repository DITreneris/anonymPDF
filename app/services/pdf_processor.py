from typing import List, Tuple, Dict
from pdfminer.high_level import extract_text
from fastapi import HTTPException
from pathlib import Path
import spacy
import re
import json
import fitz  # PyMuPDF
import time
from datetime import datetime
from langdetect import detect, LangDetectException
from collections import defaultdict

from app.pdf_processor import redact_pdf
from app.core.logging import pdf_logger
from app.core.config_manager import ConfigManager
from app.core.performance import file_processing_metrics, performance_monitor
from app.core.validation_utils import (
    validate_person_name,
    validate_swift_bic,
    validate_organization_name,
    validate_detection_context,
    deduplicate_detections,
    validate_organization_with_brand_context,
    is_brand_name,
    GEOGRAPHIC_EXCLUSIONS,
    DOCUMENT_TERMS,
    COMMON_ALL_CAPS_NON_NAMES
)
# Priority 2 imports
from app.core.context_analyzer import (
    ContextualValidator,
    AdvancedPatternRefinement,
    DocumentStructureAnalyzer,
    DetectionContext,
    ConfidenceLevel,
    create_context_aware_detection
)
from app.core.lithuanian_enhancements import (
    LithuanianLanguageEnhancer,
    LithuanianContextAnalyzer
)
from app.core.text_extraction import extract_text_enhanced
from app.core.salutation_detector import detect_lithuanian_salutations

class PDFProcessor:
    def __init__(self):
        pdf_logger.info("Initializing PDF processor with Priority 2 enhancements")

        # Initialize configuration manager
        self.config_manager = ConfigManager()
        pdf_logger.info(
            "Configuration manager loaded",
            patterns_count=len(self.config_manager.patterns),
            cities_count=len(self.config_manager.cities),
        )

        # Priority 2: Initialize context-aware components
        self.contextual_validator = ContextualValidator()
        self.advanced_patterns = AdvancedPatternRefinement()
        self.document_analyzer = DocumentStructureAnalyzer()
        self.lithuanian_enhancer = LithuanianLanguageEnhancer()
        self.lithuanian_analyzer = LithuanianContextAnalyzer()
        
        pdf_logger.info("Priority 2 context-aware components initialized")

        # Load spaCy models with PyInstaller compatibility
        self.nlp_en = self._load_spacy_model_safe("en_core_web_sm", "English")
        self.nlp_lt = self._load_spacy_model_safe("lt_core_news_sm", "Lithuanian")
        
        # Ensure at least one model is available
        if not self.nlp_en and not self.nlp_lt:
            raise RuntimeError("No spaCy models available - at least English model is required")

        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        self.processed_dir = Path("processed")
        self.processed_dir.mkdir(exist_ok=True)

    def _load_spacy_model_safe(self, model_name: str, language_name: str):
        """Safely load spaCy model with PyInstaller compatibility"""
        import sys
        import os
        from pathlib import Path
        
        try:
            # Method 1: Standard loading
            nlp = spacy.load(model_name)
            pdf_logger.info(
                f"{language_name} spaCy model loaded successfully",
                model=model_name,
                components=len(nlp.pipe_names),
                method="standard"
            )
            return nlp
        except OSError:
            pdf_logger.warning(f"Standard loading failed for {model_name}, trying PyInstaller compatibility...")
            
        # Method 2: PyInstaller bundle path
        try:
            if hasattr(sys, '_MEIPASS'):
                bundle_dir = Path(sys._MEIPASS)
                model_path = bundle_dir / model_name
                if model_path.exists():
                    nlp = spacy.load(str(model_path))
                    pdf_logger.info(
                        f"{language_name} spaCy model loaded from bundle",
                        model=model_name,
                        path=str(model_path),
                        components=len(nlp.pipe_names),
                        method="bundle"
                    )
                    return nlp
        except Exception as e:
            pdf_logger.warning(f"Bundle loading failed for {model_name}: {e}")
            
        # Method 3: Environment variable paths
        try:
            env_var = f"SPACY_MODEL_{language_name.upper()[:2]}"
            if env_var in os.environ:
                model_path = os.environ[env_var]
                nlp = spacy.load(model_path)
                pdf_logger.info(
                    f"{language_name} spaCy model loaded from environment",
                    model=model_name,
                    path=model_path,
                    components=len(nlp.pipe_names),
                    method="environment"
                )
                return nlp
        except Exception as e:
            pdf_logger.warning(f"Environment loading failed for {model_name}: {e}")
            
        # Method 4: Try loading by package name
        try:
            import importlib
            model_module = importlib.import_module(model_name)
            nlp = model_module.load()
            pdf_logger.info(
                f"{language_name} spaCy model loaded via import",
                model=model_name,
                components=len(nlp.pipe_names),
                method="import"
            )
            return nlp
        except Exception as e:
            pdf_logger.warning(f"Import loading failed for {model_name}: {e}")
        
        # All methods failed
        if language_name == "English":
            pdf_logger.error(f"Failed to load {language_name} spaCy model - this is required", model=model_name)
            return None
        else:
            pdf_logger.warning(
                f"{language_name} spaCy model not available - {language_name.lower()} detection will be limited",
                model=model_name
            )
            return None

    def _add_detection(self, personal_info: Dict, category: str, text: str, context: str, confidence: float):
        """Helper to standardize adding detections."""
        if category in personal_info:
            validated_text = text.strip()
            # Basic validation to avoid adding empty or junk strings
            if validated_text:
                personal_info[category].append((
                    validated_text,
                    f"{context}_{confidence:.2f}"
                ))

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
        Priority 2: Enhanced with context-aware detection and confidence scoring.
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

        # Priority 2: Store context-aware detections for advanced processing
        context_aware_detections = []

        # Extract entities using spaCy (PERSON, GPE, ORG) with enhanced validation
        for ent in doc.ents:
            # Priority 2: Create context-aware detection
            detection_context = create_context_aware_detection(
                ent.text, ent.label_, ent.start_char, ent.end_char, text, self.contextual_validator
            )
            context_aware_detections.append(detection_context)
            
        # Priority 2: Apply enhanced pattern detection
        enhanced_detections = self.advanced_patterns.find_enhanced_patterns(text)
        for detection in enhanced_detections:
            # Create context-aware detection for enhanced patterns
            detection_context = create_context_aware_detection(
                detection['text'], detection['category'], 
                detection['start'], detection['end'], text, self.contextual_validator
            )
            
            # Apply confidence boost from enhanced pattern
            detection_context.confidence += detection.get('confidence_boost', 0.0)
            
            # Add to appropriate category using the helper
            self._add_detection(
                personal_info,
                detection['category'],
                detection['text'],
                "ENHANCED_PATTERN",
                detection_context.confidence
            )
            context_aware_detections.append(detection_context)

        # Priority 2: Apply Lithuanian enhanced patterns if Lithuanian language
        if language == "lt":
            lt_detections = self.lithuanian_enhancer.find_enhanced_lithuanian_patterns(text)
            for detection in lt_detections:
                detection_context = create_context_aware_detection(
                    detection['text'], detection['category'],
                    detection['start'], detection['end'], text, self.contextual_validator
                )
                detection_context.confidence += detection.get('confidence_boost', 0.0)
                
                # Use the helper to ensure consistent format
                self._add_detection(
                    personal_info,
                    detection['category'],
                    detection['text'],
                    "LT_ENHANCED",
                    detection_context.confidence
                )
                context_aware_detections.append(detection_context)

        # Apply regex patterns for various information types
        for pattern_type, regex_list in self.config_manager.patterns.items():
            # Ensure regex_list is actually a list of regex patterns
            if not isinstance(regex_list, list):
                regex_list = [regex_list]

            for regex in regex_list:
                try:
                    # Find all matches for the current regex
                    for match in re.finditer(regex, text):
                        matched_text = match.group(0)
                        
                        # Use helper to add detection
                        self._add_detection(
                            personal_info,
                            pattern_type,
                            matched_text,
                            "REGEX",
                            0.5 # Base confidence for regex, can be improved
                        )
                except re.error as e:
                    pdf_logger.warning(
                        "Regex error in pattern",
                        pattern_type=pattern_type,
                        regex=regex,
                        error=str(e)
                    )

        # Process context-aware detections for final validation
        for detection in context_aware_detections:
            # Final validation before adding to the list
            if detection.confidence >= 0.4:  # Confidence threshold
                # Use helper to add validated detection
                self._add_detection(
                    personal_info,
                    detection.category,
                    detection.text,
                    "CONTEXT_VALIDATED",
                    detection.confidence
                )

        # Deduplicate and refine results
        return self.deduplicate_with_confidence(personal_info, context_aware_detections)

    def detect_lithuanian_cities_enhanced(self, text: str, language: str) -> List[Tuple[str, str]]:
        """Enhanced Lithuanian city detection with confidence scoring."""
        city_detections = []

        # Create a pattern for all city names (case-insensitive) from configuration
        for city in self.config_manager.cities:
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(city) + r"\b"
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                # Priority 2: Calculate confidence for city detection
                base_confidence = 0.8  # Cities are generally high confidence
                
                # Apply Lithuanian geographic validation if Lithuanian language
                if language == "lt" and self.lithuanian_enhancer.is_lithuanian_geographic_term(match.group()):
                    base_confidence += 0.1
                
                city_detections.append((match.group(), f"LITHUANIAN_LOCATION_CONF_{base_confidence:.2f}"))
                pdf_logger.info(
                    "Enhanced Lithuanian location detected",
                    city=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=base_confidence
                )

        return city_detections

    def deduplicate_with_confidence(self,
                                  personal_info: Dict[str, List[Tuple[str, str]]],
                                  context_detections: List[DetectionContext]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Deduplicates detections within each category, keeping the one with the highest confidence.
        This version is more robust and handles malformed data gracefully.
        """
        final_detections = defaultdict(list)
        
        # Use a dictionary to track the best detection for each unique text
        best_detections = {}

        all_detections = []
        # First, gather all context-aware detections
        for detection in context_detections:
            all_detections.append({
                "text": detection.text,
                "category": detection.category,
                "confidence": detection.confidence
            })

        # Then, gather all regex/other detections from the main dict
        for category, items in personal_info.items():
            if not isinstance(items, list):
                continue # Skip malformed entries
            for item in items:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    text, _ = item
                    # Assign a baseline confidence if not available
                    all_detections.append({
                        "text": text, "category": category, "confidence": 0.5
                    })

        # Now, iterate and find the best detection for each text
        for detection in all_detections:
            text = detection.get("text", "").strip()
            category = detection.get("category")
            confidence = detection.get("confidence", 0.0)

            if not text or not category:
                continue

            key = (text, category)
            if key not in best_detections or confidence > best_detections[key]["confidence"]:
                best_detections[key] = {
                    "text": text,
                    "category": category,
                    "confidence": confidence
                }

        # Finally, build the output dictionary from the best detections
        for key, detection in best_detections.items():
            final_detections[detection["category"]].append(
                (detection["text"], f"CONF_{detection['confidence']:.2f}")
            )
            
        return dict(final_detections)

    def should_preserve_detection(self, text: str, pattern_type: str, surrounding_text: str) -> bool:
        """
        Determines if a detected PII should be preserved (not redacted).
        Morning Session 5 Improvement: Anti-overredaction logic.
        
        Args:
            text: The detected text
            pattern_type: Type of pattern that detected this text
            surrounding_text: Context around the detection
            
        Returns:
            bool: True if should preserve (not redact), False if should redact
        """
        # Get anti-overredaction settings
        anti_overredaction = self.config_manager.settings.get('anti_overredaction', {})
        technical_terms = set(anti_overredaction.get('technical_terms_whitelist', []))
        technical_sections = anti_overredaction.get('technical_sections', [])
        pii_field_labels = anti_overredaction.get('pii_field_labels', [])
        
        # Check if the text or surrounding context contains technical terms
        text_lower = text.lower()
        surrounding_lower = surrounding_text.lower()
        
        # If the detected text is adjacent to technical terms, preserve it
        for tech_term in technical_terms:
            tech_term_lower = tech_term.lower()
            if tech_term_lower in surrounding_lower:
                pdf_logger.info(
                    "Detection preserved due to technical term context",
                    text=text,
                    pattern_type=pattern_type,
                    technical_term=tech_term
                )
                return True
        
        # Check if we're in a technical section
        for tech_section in technical_sections:
            if tech_section.lower() in surrounding_lower:
                # Only preserve if it's not a PII field
                is_pii_field = any(pii_label.lower() in surrounding_lower for pii_label in pii_field_labels)
                if not is_pii_field:
                    pdf_logger.info(
                        "Detection preserved due to technical section context",
                        text=text,
                        pattern_type=pattern_type,
                        technical_section=tech_section
                    )
                    return True
        
        # Special handling for numeric patterns in technical contexts
        if pattern_type in ['eleven_digit_numeric', 'health_insurance_number', 'medical_record_number']:
            # Look for technical units or contexts that suggest this is not PII
            technical_indicators = ['kw', 'nm', 'kg', 'mm', 'cm³', 'g/km', 'l/100', 'co2', 'emisijos', 'galia', 'sąnaudos']
            if any(indicator in surrounding_lower for indicator in technical_indicators):
                pdf_logger.info(
                    "Numeric detection preserved due to technical indicators",
                    text=text,
                    pattern_type=pattern_type
                )
                return True
        
        return False

    def detect_lithuanian_cities(self, text: str) -> List[Tuple[str, str]]:
        """Legacy method - kept for backward compatibility."""
        return self.detect_lithuanian_cities_enhanced(text, "lt")

    def anonymize_pdf(self, input_path: Path, output_path: Path) -> Tuple[bool, Dict]:
        """Anonymize PDF by redacting personal information."""
        pdf_logger.info(
            "Starting PDF anonymization", input_path=str(input_path), output_path=str(output_path)
        )
        anonymization_start_time = time.time()

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
            for category, items in personal_info.items():
                sensitive_words.extend([item[0] for item in items])

            # A simple log to ensure we have words to redact, especially for PERSON category
            if personal_info.get("PERSON"):
                pdf_logger.info(f"Found {len(personal_info.get('PERSON', []))} person names to redact.")

            # Check if sensitive words were found
            if not sensitive_words:
                pdf_logger.warning("No sensitive words found for redaction")
                return False, {
                    "error": "No sensitive words found for redaction",
                    "details": "No sensitive words found for redaction",
                }

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

            anonymization_duration = time.time() - anonymization_start_time
            pdf_logger.info(f"BENCHMARK: anonymize_pdf for {input_path.name} took {anonymization_duration:.4f} seconds.")

            return True, report

        except Exception as e:
            pdf_logger.log_error(
                "anonymize_pdf", e, input_path=str(input_path), output_path=str(output_path)
            )
            anonymization_duration = time.time() - anonymization_start_time
            pdf_logger.error(f"BENCHMARK: anonymize_pdf for {input_path.name} failed after {anonymization_duration:.4f} seconds.")
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
        """Extract text from PDF file with enhanced Lithuanian character support."""
        return extract_text_enhanced(pdf_path)

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
