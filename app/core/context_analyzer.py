"""
Context-aware detection engine for Priority 2 improvements.

This module provides advanced context analysis, confidence scoring,
and document structure awareness to improve PII detection accuracy.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from app.core.validation_utils import GEOGRAPHIC_EXCLUSIONS
from app.core.config_manager import ConfigManager, get_config_manager
from app.core.lithuanian_enhancements import LithuanianLanguageEnhancer

context_logger = logging.getLogger(__name__)


def select_longest_match(matches: List[Dict]) -> List[Dict]:
    """
    From a list of PII matches, select the longest one when overlaps occur.

    Example: If we have a match for "90210" (zip) and "Beverly Hills, 90210" (address),
    this function ensures only the latter is returned.
    """
    if not matches:
        return []

    # Sort by start position, then by length descending
    matches.sort(key=lambda m: (m['start'], -m['end']))

    # The list to store the final, non-overlapping matches
    final_matches = []
    
    # Sentinel to track the end position of the last added match
    last_match_end = -1

    for match in matches:
        # If the new match starts after or at the same place as the last one ended,
        # it's a valid, non-overlapping match.
        if match['start'] >= last_match_end:
            final_matches.append(match)
            last_match_end = match['end']
        # If a new match starts *before* the last one ended, it's an overlap.
        # Because we sorted by length descending, the one we already added
        # is the longer one, so we just ignore this new, shorter match.
    
    return final_matches


def deduplicate_by_full_match_priority(matches: List[Dict]) -> List[Dict]:
    """
    From a list of PII matches, select the longest one based on the full regex
    match coordinates when overlaps occur. This correctly prioritizes patterns
    that match more context (e.g., 'Asmens kodas: 123' over just '123').

    Args:
        matches: A list of dicts, where each dict must contain:
                 'pii_start', 'pii_end', 'full_match_start', 'full_match_end',
                 'text', and 'category'.
    """
    if not matches:
        return []

    # Sort by the start of the full match, then by the length of the full match (descending).
    matches.sort(key=lambda m: (m['full_match_start'], -(m['full_match_end'] - m['full_match_start'])))

    final_matches = []
    last_full_match_end = -1

    for match in matches:
        # If the new full match starts after the last one ended, it's non-overlapping.
        if match['full_match_start'] >= last_full_match_end:
            final_matches.append(match)
            last_full_match_end = match['full_match_end']
        # An overlapping match is found. Because we sorted by length descending,
        # the one already in final_matches is the longest, correct one. So we ignore this new one.
    
    return final_matches


class ConfidenceLevel(Enum):
    """Confidence levels for PII detections."""
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


@dataclass
class DetectionContext:
    """
    A dataclass-like object holding enriched information about a PII detection.
    This provides a structured way to pass around detection data.
    """
    __slots__ = ['text', 'category', 'start_char', 'end_char', 'confidence', 'context_before', 'context_after', 'validation_result', 'is_valid', 'full_text', 'validator', 'original_text', 'page_number', 'bounding_box', 'document_section']

    def __init__(self, text: str, category: str, start_char: int, end_char: int,
                 full_text: str, validator, confidence: float = 0.5,
                 context_window: int = 50, page_number: Optional[int] = None, bounding_box: Optional[Any] = None):
        self.text = text
        self.category = category
        self.start_char = start_char
        self.end_char = end_char
        self.full_text = full_text
        self.validator = validator
        self.confidence = confidence  # Initial confidence
        self.context_before = full_text[max(0, start_char - context_window):start_char]
        self.context_after = full_text[end_char:end_char + context_window]
        self.validation_result = None  # To be filled by validator
        self.is_valid = True  # Default to valid
        self.original_text = text # Preserve original form if needed
        self.page_number = page_number
        self.bounding_box = bounding_box
        self.document_section = None # To be filled by validator

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the object to a dictionary."""
        return {
            "text": self.text,
            "category": self.category,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": round(self.confidence, 3),
            "is_valid": self.is_valid,
            "validation_result": self.validation_result,
            "page_number": self.page_number,
            "document_section": self.document_section
        }

    def get_confidence_level(self) -> ConfidenceLevel:
        """Return the confidence level based on the score."""
        if self.confidence >= ConfidenceLevel.VERY_HIGH.value:
            return ConfidenceLevel.VERY_HIGH
        if self.confidence >= ConfidenceLevel.HIGH.value:
            return ConfidenceLevel.HIGH
        if self.confidence >= ConfidenceLevel.MEDIUM.value:
            return ConfidenceLevel.MEDIUM
        if self.confidence >= ConfidenceLevel.LOW.value:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW

    def get_context(self, window_size: int) -> str:
        """Returns the context window around the detection."""
        # This method assumes context_before and context_after are populated.
        # It reconstructs the context window for external use.
        return f"{self.context_before}{self.text}{self.context_after}"


class DocumentStructureAnalyzer:
    """Analyzes document structure to improve context awareness."""
    
    def __init__(self):
        # Document section indicators - Reordered for priority
        self.section_patterns = {
            'form_field': [
                r'^\s*(Name|Vardas)\s*:\s*',
                r'^\s*(Address|Adresas)\s*:\s*',
                r'^\s*(Phone|Telefonas)\s*:\s*',
                r'^\s*(Email|El\. paštas)\s*:\s*',
                r'^\s*(Date of Birth|Gimimo data)\s*:\s*',
                r'^\s*(Personal Code|Asmens kodas)\s*:\s*'
            ],
            'footer': [
                r'(Page|Puslapis)\s+\d+(?:\s+of\s+\d+)?', 
                r'\b(www\.[\w\.-]+|https?://[\w\.-]+)\b', 
                r'(©|Copyright|Autorių teisės)\s+\d{4}', 
                r'^[\s\-_]*Confidential[\s\-_]*$', 
                r'^[\s\-_]*Internal Use Only[\s\-_]*$' 
            ],
            'table_header': [
                r'^\s*(Description|Aprašymas|Item|Prekė|Quantity|Kiekis|Unit Price|Vieneto kaina|Total|Iš viso)(?!\s*:[^\n]*$)',
                r'^\s*(Date|Data|Time|Laikas|Period|Laikotarpis)(?!\s*:[^\n]*$)',
                r'^\s*(Number|Numeris|Code|Kodas|ID|Reference|Nuoroda)(?!\s*:[^\n]*$)',
                r'^\s*(Status|Būsena|Action|Veiksmas|Notes|Pastabos)\b' 
            ],
            'header': [
                r'^\s*(CERTIFICATE|PAŽYMĖJIMAS|AGREEMENT|SUTARTIS|INVOICE|SĄSKAITA FAKTŪRA)(?:\s+|$)',
                r'^\s*(INSURANCE|DRAUDIMO|POLICY|POLISAS)(?:\s+|$)',
                r'^\s*(COMPANY|ĮMONĖS|ORGANIZATION|ORGANIZACIJOS)(?:\s+DATA|DUOMENYS)?(?:\s+|$)',
                r'^\s*(DOCUMENT|DOKUMENTAS)(?!\s+(?:Number|Numeris|No\.|Nr\.))(\s+.*)?$'
            ],
            'legal_text': [
                r'(Article|Straipsnis)\s+\d+',
                r'(Section|Skyrius)\s+\d+',
                r'(Paragraph|Paragrafas)\s+\d+',
                r'(Clause|Punktas)\s+\d+'
            ]
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        for section_type, patterns in self.section_patterns.items():
            self.compiled_patterns[section_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
    
    def identify_document_section(self, text: str, position: int, window_size: int = 100) -> Optional[str]:
        """
        Identify what section of the document a position belongs to.
    
        Args:
            text: Full document text
            position: Character position to analyze
            window_size: Size of context window to analyze
        
        Returns:
        Section type or None if not identifiable
        """
        context_logger.debug(f"Identifying section for position {position}")

        # Determine the primary line text based on the original position in the full text
        line_start_orig = text.rfind('\n', 0, position) + 1
        line_end_orig = text.find('\n', position)
        if line_end_orig == -1:  # Position is in the last line
            line_end_orig = len(text)
        
        primary_line_text = text[line_start_orig:line_end_orig].strip()
        context_logger.debug(f"Primary line text: '{primary_line_text}'")

        if primary_line_text:
            # First pass: check the primary line specifically
            for section_type, pattern_objects in self.compiled_patterns.items():
                for pattern_obj in pattern_objects:
                    matched = False
                    if section_type in ['form_field', 'table_header']:
                        if pattern_obj.match(primary_line_text):
                            matched = True
                    else:  # footer, header, legal_text
                        if pattern_obj.search(primary_line_text):
                            matched = True
                    
                    if matched:
                        context_logger.debug(
                            "Document section identified (primary line)",
                            section=section_type,
                            position=position,
                            line_checked=primary_line_text,
                            pattern=pattern_obj.pattern
                        )
                        return section_type

        context_logger.debug("Primary line search failed - initiating context window fallback.")

        # Fallback: expand search to full context window
        context_start_win = max(0, position - window_size)
        context_end_win = min(len(text), position + window_size)
        context_window_text = text[context_start_win:context_end_win]

        # Log context preview with conditional ellipsis
        ellipsis = "..." if len(context_window_text) > 200 else ""
        context_logger.debug(f"Fallback context window ({len(context_window_text)} chars): '{context_window_text[:200]}{ellipsis}'")

        for section_type, pattern_objects in self.compiled_patterns.items():
            # Exclude form_field and table_header from broad window search if they are strictly line-anchored
            if section_type in ['form_field', 'table_header']:
                continue 
            for pattern_obj in pattern_objects:
                if pattern_obj.search(context_window_text):
                    context_logger.debug(
                        "Document section identified (context window fallback)",
                        section=section_type,
                        position=position,
                        pattern=pattern_obj.pattern
                    )
                    return section_type
        
        context_logger.debug(f"No section identified for position {position}")
        return None
    
    def is_document_metadata(self, text: str, context: str) -> bool:
        """Check if text appears to be document metadata rather than content."""
        metadata_indicators = [
            r'(Document|Dokumentas)\s+(Number|Numeris|Nr\.)',
            r'(Page|Puslapis)\s+\d+',
            r'(Version|Versija)\s+\d+',
            r'(Date|Data):\s*\d',
            r'(Reference|Nuoroda):\s*[A-Z0-9]',
            r'(File|Failas):\s*\w+\.(pdf|doc|docx)',
        ]
        
        for pattern in metadata_indicators:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        
        return False


class ContextualValidator:
    """Provides context-aware validation for PII detections."""
    
    def __init__(self, cities: List[str], brand_names: List[str]):
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.cities = {city.lower() for city in cities}
        self.brand_names = {brand.lower() for brand in brand_names}
        
        # Context patterns that indicate false positives
        self.false_positive_contexts = {
            'person_name': [
                r'(Section|Skyrius|Chapter|Skyrius)\s+{text}',
                r'(Article|Straipsnis)\s+{text}',
                r'(Paragraph|Paragrafas)\s+{text}',
                r'(Figure|Pav\.)\s+{text}',
                r'(Table|Lentelė)\s+{text}',
                r'(Appendix|Priedas)\s+{text}',
            ],
            'organization': [
                r'(When|Kada)\s+{text}',
                r'(Where|Kur)\s+{text}',
                r'(What|Kas)\s+{text}',
                r'(Which|Kuris)\s+{text}',
                r'(How|Kaip)\s+{text}',
            ]
        }
        
        # Context patterns that indicate true positives
        self.true_positive_contexts = {
            'person_name': [
                r'(Mr\.|Mrs\.|Ms\.|Dr\.)\s+{text}',
                r'(Ponas|Ponia|Daktaras)\s+{text}',
                r'{text}\s+(born|gimęs|gimusi)',
                r'(Name|Vardas):\s*{text}',
                r'(Signature|Parašas):\s*{text}',
            ],
            'organization': [
                r'(Company|Įmonė):\s*{text}',
                r'(Organization|Organizacija):\s*{text}',
                r'{text}\s+(Ltd\.|UAB|AB|VšĮ)',
                r'(Employer|Darbdavys):\s*{text}',
            ],
            'email': [
                r'(Email|El\. paštas):\s*{text}',
                r'(Contact|Kontaktai):\s*{text}',
                r'(Send to|Siųsti)\s+{text}',
            ],
            'phone': [
                r'(Phone|Telefonas):\s*{text}',
                r'(Mobile|Mobilus):\s*{text}',
                r'(Call|Skambinti)\s+{text}',
            ]
        }
    
    def calculate_confidence(self, detection: str, category: str, context: str, 
                           document_section: Optional[str] = None) -> float:
        """
        Calculate confidence score for a PII detection based on context.
        
        Args:
            detection: The detected text
            category: PII category (person_name, organization, etc.)
            context: Surrounding text context
            document_section: Document section type if identified
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.7  # Default confidence
        confidence_adjustments = []
        
        detection_lower = detection.lower()

        # Penalty for geographic exclusions misclassified as person names
        if category == 'person_name' and detection in GEOGRAPHIC_EXCLUSIONS:
            confidence_adjustments.append(-0.5) 
            context_logger.debug(
                "PII penalized: Geographic term misclassified as person_name",
                detection=detection, category=category, adjustment=-0.5
            )

        # Specific check for structural keywords misclassified as person_name
        structural_keywords_person = {
            "section", "skyrius", "chapter", "article", "straipsnis", 
            "paragraph", "paragrafas", "figure", "pav.", "table", "lentelė", 
            "appendix", "priedas"
        }
        if category == 'person_name' and detection_lower in structural_keywords_person:
            # Check if followed by a number or is at the start of a line often indicative of a header
            if re.search(r'^\s*' + re.escape(detection) + r'\s*\d+', context, re.IGNORECASE | re.MULTILINE) or \
               re.search(r'\n\s*' + re.escape(detection) + r'\s*\d+', context, re.IGNORECASE | re.MULTILINE) or \
               document_section in ['header', 'table_header']:
                confidence_adjustments.append(-0.4) # Strong penalty
                context_logger.debug(
                    "Structural keyword misclassified as person_name penalized",
                    detection=detection, category=category
                )

        # Check for false positive indicators (original logic, adjusted to avoid double penalty)
        if not (category == 'person_name' and detection_lower in structural_keywords_person):
            if category in self.false_positive_contexts:
                for pattern_template in self.false_positive_contexts[category]:
                    pattern = pattern_template.format(text=re.escape(detection))
                    if re.search(pattern, context, re.IGNORECASE):
                        confidence_adjustments.append(-0.3)
                        context_logger.debug(
                            "False positive context detected",
                            detection=detection,
                            category=category,
                            pattern=pattern
                        )
        
        # Check for true positive indicators
        if category in self.true_positive_contexts:
            for pattern_template in self.true_positive_contexts[category]:
                pattern = pattern_template.format(text=re.escape(detection))
                if re.search(pattern, context, re.IGNORECASE):
                    confidence_adjustments.append(0.2)
                    context_logger.debug(
                        "True positive context detected",
                        detection=detection,
                        category=category,
                        pattern=pattern
                    )
        
        # Document section adjustments
        if document_section:
            section_adjustments = {
                'header': -0.1,  # Headers often contain non-PII
                'footer': -0.2,  # Footers rarely contain PII
                'table_header': -0.3,  # Table headers are usually labels
                'form_field': 0.2,  # Form fields often contain PII
                'legal_text': -0.1,  # Legal text structure references
            }
            
            if document_section in section_adjustments:
                confidence_adjustments.append(section_adjustments[document_section])
                context_logger.debug(
                    "Document section adjustment applied",
                    detection=detection,
                    section=document_section,
                    adjustment=section_adjustments[document_section]
                )
        
        # Check if it's document metadata
        if self.structure_analyzer.is_document_metadata(detection, context):
            confidence_adjustments.append(-0.4)
            context_logger.debug(
                "Document metadata detected",
                detection=detection,
                category=category
            )
        
        # Calculate final confidence
        confidence = min(1.0, max(0.0, base_confidence + sum(confidence_adjustments)))

        # Log the details of the confidence calculation for transparency
        context_logger.info(
            f"Confidence for '{detection}' ({category}): base={base_confidence}, "
            f"adjust={sum(confidence_adjustments):.2f}, final={confidence:.2f}, "
            f"section='{document_section}', "
        )
        return confidence

    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Return the confidence level based on the score."""
        if confidence >= ConfidenceLevel.VERY_HIGH.value:
            return ConfidenceLevel.VERY_HIGH
        if confidence >= ConfidenceLevel.HIGH.value:
            return ConfidenceLevel.HIGH
        if confidence >= ConfidenceLevel.MEDIUM.value:
            return ConfidenceLevel.MEDIUM
        if confidence >= ConfidenceLevel.LOW.value:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW
    
    def validate_with_context(self, detection: str, category: str, 
                            full_text: str, start_pos: int, end_pos: int,
                            context_window: int = 100) -> DetectionContext:
        """
        Perform comprehensive context validation for a detection.
        
        Args:
            detection: The detected text
            category: PII category
            full_text: Full document text
            start_pos: Start position of detection
            end_pos: End position of detection
            context_window: Size of context window to analyze
            
        Returns:
            DetectionContext with validation results
        """
        # Extract context
        context_start = max(0, start_pos - context_window)
        context_end = min(len(full_text), end_pos + context_window)
        
        context_before = full_text[context_start:start_pos]
        context_after = full_text[end_pos:context_end]
        full_context = full_text[context_start:context_end]
        
        # Identify document section
        document_section = self.structure_analyzer.identify_document_section(
            full_text, start_pos, context_window
        )
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            detection, category, full_context, document_section
        )
        
        # Generate validation flags
        validation_flags = []
        
        if confidence < 0.4:
            validation_flags.append("low_confidence")
        
        if document_section in ['header', 'footer', 'table_header']:
            validation_flags.append("structural_element")
        
        if self.structure_analyzer.is_document_metadata(detection, full_context):
            validation_flags.append("document_metadata")
        
        # Check for common false positive patterns
        if category == 'person_name' and len(detection.split()) > 3:
            validation_flags.append("overly_long_name")
        
        if category == 'organization' and detection.lower() in ['when', 'where', 'what', 'which', 'how']:
            validation_flags.append("common_word")
        
        detection_obj = DetectionContext(
            text=detection,
            category=category,
            start_char=start_pos,
            end_char=end_pos,
            confidence=confidence,
            full_text=full_text,
            validator=self,
            context_window=context_window,
            page_number=None,
            bounding_box=None
        )
        detection_obj.validation_result = validation_flags
        detection_obj.document_section = document_section
        return detection_obj

    def validate(self, detection: DetectionContext, full_text: str):
        """
        Validates a single detection by checking its surrounding context,
        and adjusts its confidence score.
        """
        # Define a window around the detection to analyze its context.
        window_size = 50  # characters
        start = max(0, detection.start_char - window_size)
        end = min(len(full_text), detection.end_char + window_size)
        context_window = full_text[start:end]

        # Example validation: check for keywords that boost or reduce confidence.
        if "forbidden" in context_window.lower():
            detection.confidence *= 0.5  # Reduce confidence
        elif "confirmed" in context_window.lower():
            detection.confidence = min(1.0, detection.confidence * 1.2)  # Increase confidence

        # Mark as validated
        detection.is_valid = True


class AdvancedPatternRefinement:
    """Refines PII detection using a library of advanced, context-aware patterns."""

    def __init__(self):
        """
        Initializes the AdvancedPatternRefinement, ensuring enhanced patterns
        from the LithuanianLanguageEnhancer overwrite any base patterns
        with the same name from the ConfigManager.
        """
        config_manager = get_config_manager()
        lithuanian_enhancer = LithuanianLanguageEnhancer()
        
        # Final structure: Dict[str, Tuple[re.Pattern, str]] # (pattern_name -> (compiled_regex, category))
        self.pattern_map: Dict[str, Tuple[re.Pattern, str]] = {}

        # 1. Load base patterns from config. The key is the category.
        for name, pattern in config_manager.patterns.items():
            self.pattern_map[name] = (pattern, name)

        # 2. Load enhanced patterns from enhancer. This uses the info object.
        # This will overwrite any base patterns if the `name` (the key) is the same.
        for name, pattern_info in lithuanian_enhancer.enhanced_lithuanian_patterns.items():
            # The enhancer patterns are designed to be compiled with IGNORECASE
            compiled_pattern = re.compile(pattern_info.pattern, re.IGNORECASE)
            self.pattern_map[name] = (compiled_pattern, pattern_info.category) 

        context_logger.info(f"AdvancedPatternRefinement initialized with {len(self.pattern_map)} patterns.")

    def find_enhanced_patterns(self, text: str) -> List[Dict]:
        """
        Find PII using the enhanced regex patterns from the configuration. This method
        now performs a two-stage de-duplication process to correctly prioritize
        context-aware patterns over simpler ones.
        """
        all_matches_with_context = []
        for pattern_name, (pattern, category) in self.pattern_map.items():
            try:
                for match in pattern.finditer(text):
                    # For de-duplication, we need the full match coordinates.
                    # For the final result, we need the PII (captured group) coordinates.
                    is_grouped = match.groups()
                    pii_text = match.group(1) if is_grouped else match.group(0)
                    
                    match_data = {
                        "text": pii_text,
                        "category": category,
                        "pii_start": match.start(1) if is_grouped else match.start(0),
                        "pii_end": match.end(1) if is_grouped else match.end(0),
                        "full_match_start": match.start(0),
                        "full_match_end": match.end(0)
                    }
                    all_matches_with_context.append(match_data)
            except re.error as e:
                context_logger.error(f"Regex error for pattern '{pattern_name}': {e}")
        
        # De-duplicate using the full match coordinates for prioritization.
        unique_longest_matches = deduplicate_by_full_match_priority(all_matches_with_context)
        
        # Transform the results into the final format, containing only the PII details.
        final_detections = [
            {
                "text": m["text"],
                "category": m["category"],
                "start": m["pii_start"],
                "end": m["pii_end"]
            }
            for m in unique_longest_matches
        ]
        
        context_logger.debug(f"Found {len(final_detections)} unique, prioritized patterns.")
        return final_detections


def create_context_aware_detection(
    text: str, category: str, start: int, end: int, full_text: str, validator: 'ContextualValidator',
    confidence: Optional[float] = None
) -> DetectionContext:
    """
    Factory function to create a DetectionContext object.
    This encapsulates the logic of creating and validating a detection.
    """
    detection = DetectionContext(
        text=text,
        category=category,
        start_char=start,
        end_char=end,
        full_text=full_text,
        validator=validator,
        confidence=confidence if confidence is not None else 0.5
    )
    
    # Perform validation immediately upon creation
    validator.validate(detection, full_text)
    
    return detection 