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
            detection_context.confidence += detection['confidence_boost']
            
            # Add to appropriate category
            category = detection['category']
            if category in personal_info:
                personal_info[category].append((
                    detection['text'], 
                    f"ENHANCED_CONF_{detection_context.confidence:.2f}"
                ))
                context_aware_detections.append(detection_context)

        # Priority 2: Apply Lithuanian enhanced patterns if Lithuanian language
        if language == "lt":
            lithuanian_detections = self.lithuanian_enhancer.find_enhanced_lithuanian_patterns(text)
            for detection in lithuanian_detections:
                # Check for brand names in Lithuanian patterns (especially organizations)
                if detection['category'] == 'organizations' and is_brand_name(detection['text']):
                    pdf_logger.info(
                        "Lithuanian organization pattern identified as brand name - preserving",
                        text=detection['text'],
                        pattern=detection.get('pattern_name', 'unknown'),
                        reason="brand_name_preservation"
                    )
                    continue  # Skip redaction for brand names
                
                # Create context-aware detection for Lithuanian patterns
                detection_context = create_context_aware_detection(
                    detection['text'], detection['category'], 
                    detection['start'], detection['end'], text, self.contextual_validator
                )
                
                # Apply Lithuanian confidence modifier
                lithuanian_confidence = self.lithuanian_analyzer.calculate_lithuanian_confidence(
                    detection['text'], detection['category'],
                    detection_context.context_before + detection_context.context_after,
                    self.lithuanian_analyzer.identify_lithuanian_section(text, detection['start'])
                )
                detection_context.confidence += lithuanian_confidence + detection['confidence_modifier']
                
                # Add to appropriate category
                category = detection['category']
                if category in personal_info:
                    personal_info[category].append((
                        detection['text'], 
                        f"LT_ENHANCED_CONF_{detection_context.confidence:.2f}"
                    ))
                    context_aware_detections.append(detection_context)

            # Priority 2: Apply Lithuanian salutation detection for Lithuanian documents
            salutation_detections = detect_lithuanian_salutations(text)
            if salutation_detections:
                pdf_logger.info(
                    "Lithuanian salutation detection completed",
                    total_salutations=len(salutation_detections),
                    unique_names=len(set(s.base_name for s in salutation_detections))
                )
                
                for salutation in salutation_detections:
                    # Create context-aware detection for salutation names
                    detection_context = create_context_aware_detection(
                        salutation.base_name, 'SALUTATION_NAME',
                        salutation.start_pos, salutation.end_pos, text, self.contextual_validator
                    )
                    
                    # Use salutation confidence
                    detection_context.confidence = salutation.confidence
                    
                    # Add both the detected form and base form to names
                    personal_info["names"].append((
                        salutation.extracted_name, 
                        f"SALUTATION_DETECTED_CONF_{salutation.confidence:.2f}"
                    ))
                    
                    if salutation.base_name != salutation.extracted_name:
                        personal_info["names"].append((
                            salutation.base_name, 
                            f"SALUTATION_BASE_CONF_{salutation.confidence:.2f}"
                        ))
                    
                    context_aware_detections.append(detection_context)
                    
                    pdf_logger.info(
                        "Salutation name added to PII detection",
                        salutation_type=salutation.salutation_type,
                        extracted_name=salutation.extracted_name,
                        base_name=salutation.base_name,
                        confidence=salutation.confidence
                    )

        # Extract sensitive information using regex patterns from configuration
        for pattern_type, pattern_from_config in self.config_manager.patterns.items():
            
            current_regex_text = pattern_from_config
            is_lithuanian_address_flexible_special = False

            if pattern_type == "lithuanian_address_flexible":
                is_lithuanian_address_flexible_special = True
                # Override with the corrected regex that enforces prefix and captures groups
                current_regex_text = r'(G\\.|g\\.|Gatvė|gatvė|Al\\.|al\\.|Alėja|alėja|Pr\\.|pr\\.|Prospektas|prospektas|Pl\\.|pl\\.|Plentas|plentas|Tak\\.|tak\\.|Takas|takas|Sk\\.|sk\\.|Sodų bendrija|Skersgatvis|skersgatvis|Kelias|kelias|Akademija|aikštė|Skveras|skveras)\\s*([A-Ža-zĄ-žĀ-žČ-čĒ-ēĘ-ęĖ-ėĮ-įŠ-šŪ-ūŲ-ųŽ-ž0-9\\s\\.\\-\\(\\)]+?)(?:,\\s*(?:N\\.?\\s*\\d+|Butas Nr\\.\\s*\\d+|Butas\\s*\\d+|B\\.\\s*\\d+|K\\.\\s*\\d+|P\\.O\\.\\s*Box\\s*\\d+|Pašto dėžutė\\s*\\d+))?'
            
            matches = re.finditer(current_regex_text, text)

            for match in matches:
                # For context creation, always use the full text matched by the pattern.
                full_match_text = match.group()

                # This context is for the raw match before specific PII content extraction for some patterns
                detection_context = create_context_aware_detection(
                    full_match_text, pattern_type, 
                    match.start(), match.end(), text, self.contextual_validator
                )

                if detection_context.confidence < 0.4: # Regex patterns use 0.4 threshold
                    continue

                # Determine the actual PII text to consider for preservation and addition.
                pii_content = full_match_text # Default to the full match

                # After validation, use the category from the detection_context, which may have been overridden by adaptive logic
                final_category = detection_context.category

                if pattern_type == "lithuanian_address_flexible": # Special handling for this pattern
                    # is_lithuanian_address_flexible_special is defined earlier in the loop
                    if is_lithuanian_address_flexible_special: # Check the flag
                        if match.groups() and len(match.groups()) >= 2:
                            pii_content = match.group(2).strip() 
                        else:
                            continue 
                
                if not pii_content: 
                    continue

                # Ensure the category exists in the results dictionary
                if final_category not in personal_info:
                    personal_info[final_category] = []
                    
                surrounding_context_for_preservation = detection_context.context_before + pii_content + detection_context.context_after
                if self.should_preserve_detection(pii_content, pattern_type, surrounding_context_for_preservation):
                    continue
                
                personal_info[final_category].append(
                    (pii_content, f"{final_category}_CONF_{detection_context.confidence:.2f}")
                )
                context_aware_detections.append(detection_context)

        city_detections = self.detect_lithuanian_cities_enhanced(text, language)
        personal_info["locations"].extend(city_detections)

        # --- Final Processing Loop ---
        # This new loop becomes the single point of truth for adding detections to the final result.
        # It iterates over all gathered contexts, applies final validation, and populates `personal_info`.
        
        temp_personal_info = {}

        for context in context_aware_detections:
            final_category = context.category
            pii_content = context.text
            confidence = context.confidence
            
            # Apply a general confidence threshold
            if confidence < 0.3:
                continue

            # Apply post-validation logic similar to the original structure, but unified.
            # This is to prevent false positives from spaCy that the validator didn't catch.
            is_valid = True
            # Here you could re-add specific logic like `validate_person_name` if needed,
            # but for now we rely on the validator's output and confidence.

            if is_valid:
                if final_category not in temp_personal_info:
                    temp_personal_info[final_category] = []
                temp_personal_info[final_category].append(
                    (pii_content, f"{final_category}_CONF_{confidence:.2f}")
                )
        
        # Merge results into the main personal_info dictionary, then deduplicate
        for category, items in temp_personal_info.items():
            if category not in personal_info:
                personal_info[category] = []
            personal_info[category].extend(items)

        personal_info = self.deduplicate_with_confidence(personal_info, context_aware_detections)
        
        return personal_info

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
        Deduplicates detections based on confidence scores, ensuring that for any
        given text segment, only the detection with the highest confidence across
        all categories is retained.
        """
        # Invert the context_detections for quick lookup: text -> best_context
        # This ensures we always use the highest-confidence context for any given text string.
        best_contexts: Dict[str, DetectionContext] = {}

        all_detections_flat = []
        for category, detections_in_category in personal_info.items():
            for text, _ in detections_in_category:
                all_detections_flat.append(text)

        # Find the best context for each unique detected string
        for text in set(all_detections_flat):
            # Find all contexts related to this specific text
            related_contexts = [ctx for ctx in context_detections if ctx.text == text]
            if not related_contexts:
                continue

            # Find the one with the highest confidence
            best_context = max(related_contexts, key=lambda ctx: ctx.confidence)
            best_contexts[text] = best_context
            pdf_logger.debug(
                f"Deduplicating text '{text}'. Best context found in category "
                f"'{best_context.category}' with confidence {best_context.confidence:.2f}"
            )
        
        # Initialize a new dict to hold the final, confidence-based results.
        final_results: Dict[str, List[Tuple[str, str]]] = {
            category: [] for category in personal_info.keys()
        }

        # Now, rebuild the final_results using the best context for each unique detection
        for detection_text, best_context in best_contexts.items():
            final_category = best_context.category
            
            # Ensure the final category key exists in the final_results dict, as it might be a new adaptive one.
            if final_category not in final_results:
                final_results[final_category] = []
                
            final_results[final_category].append(
                (detection_text, f"{final_category}_CONF_{best_context.confidence:.2f}")
            )
            
        # Clean up empty categories from the final dictionary
        return {k: v for k, v in final_results.items() if v}

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
