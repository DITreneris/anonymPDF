import threading
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
    def __init__(self, config_manager: ConfigManager):
        pdf_logger.info("Initializing PDF processor with Priority 2 enhancements")

        # Initialize configuration manager
        self.config_manager = config_manager
        pdf_logger.info(
            "Configuration manager loaded",
            patterns_count=len(self.config_manager.patterns),
            cities_count=len(self.config_manager.cities),
        )

        # Priority 2: Initialize context-aware components with injected config
        self.contextual_validator = ContextualValidator(
            cities=self.config_manager.cities,
            brand_names=self.config_manager.brand_names
        )
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
                detection['start'], detection['end'], 
                text, self.contextual_validator,
                confidence=detection.get('confidence', 0.5)
            )
            context_aware_detections.append(detection_context)

        # Priority 2: Apply Lithuanian-specific enhancements if language is 'lt'
        if language == "lt":
            lithuanian_detections = self.lithuanian_enhancer.find_enhanced_lithuanian_patterns(text)
            for detection in lithuanian_detections:
                detection_context = create_context_aware_detection(
                    detection['text'],
                    detection['category'],
                    detection['start'],
                    detection['end'],
                    text,
                    self.contextual_validator,
                    confidence=detection.get('confidence', 0.5)
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
        """
        Enhanced city detection, leveraging contextual analysis to reduce false positives.
        Only runs for Lithuanian text.
        """
        if language != "lt":
            return []
            
        detected_cities = []
        for city in self.config_manager.cities:
            # Use word boundaries to avoid matching parts of words
            pattern = r'\b' + re.escape(city) + r'\b'
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Priority 2: Use ContextualValidator for enhanced validation
                is_valid, reason = self.lithuanian_analyzer.validate_city_context(
                    city, text, match.start()
                )
                if is_valid:
                    detected_cities.append((city, "CITY_VALIDATED_BY_CONTEXT"))
                else:
                    pdf_logger.info("City detection skipped due to context", city=city, reason=reason)
                    
        return detected_cities

    def deduplicate_with_confidence(self,
                                  personal_info: Dict[str, List[Tuple[str, str]]],
                                  context_detections: List[DetectionContext]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Deduplicate detections based on position and confidence score.
        Highest confidence detection for an overlapping area is kept.
        """
        
        # Sort detections by confidence score (highest first)
        sorted_detections = sorted(context_detections, key=lambda d: d.confidence, reverse=True)
        
        # Keep track of character positions that have been redacted
        redacted_positions = set()
        
        final_detections = defaultdict(list)

        for detection in sorted_detections:
            # Check if this detection's range overlaps with an already claimed range
            detection_range = set(range(detection.start_char, detection.end_char))
            if not detection_range.intersection(redacted_positions):
                
                # Check for document term exclusion, unless it's a high-confidence pattern
                if (detection.text.lower() in DOCUMENT_TERMS and 
                    detection.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]):
                    pdf_logger.info(
                        "Skipping common document term with low/medium confidence",
                        term=detection.text
                    )
                    continue

                # Add to final list
                final_detections[detection.category].append(
                    (detection.text, f"CONTEXT_{detection.confidence:.2f}")
                )
                
                # Claim this character range
                redacted_positions.update(detection_range)
        
        return dict(final_detections)

    def should_preserve_detection(self, text: str, pattern_type: str, surrounding_text: str) -> bool:
        """
        Determine if a detection should be preserved based on its context,
        e.g., to prevent over-redaction of technical terms.
        """
        # Example rule: if it looks like a file path, don't redact city names
        if pattern_type == "GPE" and any(term in surrounding_text for term in ["path:", "file:", "C:\\"]):
            return False
            
        # Example rule: preserve common words if they are not part of a name-like structure
        if text.lower() in ["summary", "total"] and pattern_type == "PERSON":
            # Very basic check, could be improved
            if not text.istitle():
                return False

        return True

    def _extract_and_validate_with_patterns(
        self, text: str, patterns: Dict, language: str, text_doc=None
    ) -> List[DetectionContext]:
        """
        Internal helper to extract PII using regex patterns and perform contextual validation.
        This is a refactored and enhanced version for Priority 2.
        """
        detections = []
        
        for pii_type, pattern in patterns.items():
            try:
                # Use finditer to get match objects with positions
                for match in re.finditer(pattern, text):
                    matched_text = match.group(0)
                    
                    # Create a context-aware detection object
                    context_detection = create_context_aware_detection(
                        matched_text,
                        pii_type,
                        match.start(),
                        match.end(),
                        text,
                        self.contextual_validator,
                    )
                    
                    # Validate the context-aware detection
                    is_valid, reason = self.contextual_validator.validate_with_context(context_detection)
                    
                    if is_valid:
                        detections.append(context_detection)
                    else:
                        pdf_logger.info(
                            "Pattern detection skipped due to context",
                            text=matched_text,
                            pattern=pii_type,
                            reason=reason,
                        )
            except re.error as e:
                pdf_logger.error("Invalid regex pattern", pattern=pattern, pii_type=pii_type, error=str(e))
                
        return detections

    def detect_lithuanian_cities(self, text: str) -> List[Tuple[str, str]]:
        """Find Lithuanian city names in the text."""
        # This function is now superseded by detect_lithuanian_cities_enhanced
        # but kept for backward compatibility or simple use cases.
        cities_found = []
        for city in self.config_manager.cities:
            if re.search(r"\b" + re.escape(city) + r"\b", text, re.IGNORECASE):
                cities_found.append((city, "CITY"))
        return cities_found

    def anonymize_pdf(self, input_path: Path, output_path: Path) -> Tuple[bool, Dict]:
        """
        Anonymizes a PDF by redacting detected personal information.
        This is the core redaction logic.
        """
        
        start_time = time.time()
        
        try:
            # 1. Extract text and identify PII
            text = self.extract_text_from_pdf(input_path)
            if not text.strip():
                return False, {"error": "No content found in PDF"}

            language = self.detect_language(text)
            personal_info = self.find_personal_info(text, language)
            
            # 2. Check if there is anything to redact
            total_redactions = sum(len(v) for v in personal_info.values())
            if total_redactions == 0:
                pdf_logger.info("No PII found, no redaction needed.", file=input_path.name)
                # If no redactions are needed, we can consider it a success.
                # The "anonymized" file will be a copy of the original.
                import shutil
                shutil.copy(str(input_path), str(output_path))
                return True, {
                    "status": "success_no_pii",
                    "redactions": 0,
                    "processing_time": time.time() - start_time
                }

            # 3. Perform the redaction using PyMuPDF
            doc = fitz.open(input_path)
            redactions_applied = 0
            
            # Consolidate all PII text to search for
            all_pii_texts = set()
            for category, items in personal_info.items():
                for item_text, _ in items:
                    all_pii_texts.add(item_text)

            for page in doc:
                for pii_text in all_pii_texts:
                    # Search for the text on the page
                    text_instances = page.search_for(pii_text)
                    
                    # Apply redaction to all found instances
                    for inst in text_instances:
                        page.add_redact_annot(inst, fill=(0, 0, 0)) # Black fill
                        redactions_applied += 1
                
                # Apply the redactions for the current page
                page.apply_redactions()
            
            # 4. Save the redacted PDF
            doc.save(str(output_path), garbage=4, deflate=True)
            doc.close()

            pdf_logger.info(
                "PDF anonymized successfully",
                input_file=input_path.name,
                output_file=output_path.name,
                redactions_applied=redactions_applied
            )
            
            # 5. Generate and return the report
            report = self.generate_redaction_report(personal_info, language)
            
            return True, {
                "status": "success",
                "report": report,
                "redactions_applied": redactions_applied,
                "output_path": str(output_path),
                "processing_time": time.time() - start_time
            }

        except Exception as e:
            pdf_logger.error("Failed to anonymize PDF", file=input_path.name, error=str(e), exc_info=True)
            return False, {"error": str(e), "processing_time": time.time() - start_time}

    def generate_redaction_report(
        self, personal_info: Dict[str, List[Tuple[str, str]]], language: str
    ) -> Dict:
        """Generate a detailed redaction report."""
        report = {
            "total_redactions": sum(len(v) for v in personal_info.values()),
            "language": language,
            "categories": {k: len(v) for k, v in personal_info.items() if v},
            "details": defaultdict(list)
        }

        # Priority 3: Enhanced report with confidence scores
        for category, items in personal_info.items():
            if not items:
                continue
            for text, context_str in items:
                # Attempt to parse confidence from context string (e.g., "NER_0.85")
                try:
                    confidence = float(context_str.split('_')[-1])
                except (ValueError, IndexError):
                    confidence = 0.5  # Default confidence if parsing fails
                
                report["details"][category].append((text, f"CONF_{confidence:.2f}"))
        
        pdf_logger.info("Redaction report generated", total_redactions=report["total_redactions"], language=language)
        return report

    async def process_pdf(self, file_path: str) -> dict:
        """
        Main orchestration function to process a PDF file asynchronously.
        It handles text extraction, PII detection, anonymization, and reporting.
        """
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        file_size_bytes = pdf_path.stat().st_size
        tracker = file_processing_metrics.track_file_processing(pdf_path, file_size_bytes)
        response_data = {}
        
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                raise HTTPException(status_code=400, detail="PDF content is empty or could not be extracted")

            # Detect language
            language = self.detect_language(text)

            # Find personal information
            personal_info = self.find_personal_info(text, language)

            # Generate a unique output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{pdf_path.stem}_redacted_{timestamp}.pdf"
            output_path = self.processed_dir / output_filename

            # Anonymize the PDF
            success, anonymization_details = self.anonymize_pdf(pdf_path, output_path)
            
            if not success:
                response_data = {
                    "status": "error",
                    "error": anonymization_details.get("error", "Unknown error during anonymization"),
                    "redactions": 0,
                }
            else:
                # Generate redaction report
                report = self.generate_redaction_report(personal_info, language)

                response_data = {
                    "anonymized_file_path": str(output_path),
                    "status": "processed",
                    "report": report,
                }

        except Exception as e:
            pdf_logger.error("Error processing PDF", file=file_path, error=str(e), exc_info=True)
            
            response_data = {
                "status": "error",
                "error": f"An unexpected error occurred: {type(e).__name__}",
                "detail": str(e),
            }
            if isinstance(e, HTTPException):
                # Re-raise HTTPException to be handled by FastAPI's exception middleware
                raise e
        finally:
            # Ensure performance tracking is always finalized
            metrics = tracker['end_tracking']()
            processing_time = metrics['duration_seconds']
            response_data["processing_time"] = round(processing_time, 2)

        return response_data

    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_dir.glob("*"):
            try:
                temp_file.unlink()
            except OSError as e:
                pdf_logger.warning(
                    f"Could not remove temp file: {temp_file}", error=str(e)
                )

    @performance_monitor("text_extraction")
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract all text from a PDF file."""
        return extract_text_enhanced(pdf_path)

    def process_pdf_for_anonymization(self, file_path: Path) -> Tuple[bool, str]:
        """
        Simplified processing for direct anonymization use cases.
        Returns success status and the path to the anonymized file.
        """
        output_filename = f"{file_path.stem}_anonymized_{int(time.time())}.pdf"
        output_path = self.processed_dir / output_filename
        
        success, details = self.anonymize_pdf(file_path, output_path)
        
        if success:
            return True, str(output_path)
        else:
            return False, details.get("error", "Anonymization failed")

# Singleton instance of the processor
_pdf_processor_instance = None
_pdf_processor_lock = threading.Lock()

def get_pdf_processor() -> PDFProcessor:
    """
    Get a singleton instance of the PDFProcessor.
    This is thread-safe and ensures that spaCy models are loaded only once.
    """
    global _pdf_processor_instance
    with _pdf_processor_lock:
        if _pdf_processor_instance is None:
            config_manager = get_config_manager()
            _pdf_processor_instance = PDFProcessor(config_manager=config_manager)
    return _pdf_processor_instance
