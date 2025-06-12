"""
Context-aware detection engine for Priority 2 improvements.

This module provides advanced context analysis, confidence scoring,
and document structure awareness to improve PII detection accuracy.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from app.core.validation_utils import GEOGRAPHIC_EXCLUSIONS

context_logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for PII detections."""
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


@dataclass
class DetectionContext:
    """Context information for a PII detection."""
    text: str
    category: str
    start_pos: int
    end_pos: int
    confidence: float
    context_before: str
    context_after: str
    validation_flags: List[str]
    document_section: Optional[str] = None


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
        
        context_logger.debug("No match on primary line, falling back to context window search.")
        # Second pass (fallback): search the entire context window
        context_start_win = max(0, position - window_size)
        context_end_win = min(len(text), position + window_size)
        context_window_text = text[context_start_win:context_end_win]
        context_logger.debug(f"Context window text for fallback: '{context_window_text[:200]}...' (first 200 chars)")

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
    
    def __init__(self):
        self.structure_analyzer = DocumentStructureAnalyzer()
        
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
        """Convert numeric confidence to confidence level enum."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.4:
            return ConfidenceLevel.LOW
        else:
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
        
        return DetectionContext(
            text=detection,
            category=category,
            start_pos=start_pos,
            end_pos=end_pos,
            confidence=confidence,
            context_before=context_before,
            context_after=context_after,
            validation_flags=validation_flags,
            document_section=document_section
        )


class AdvancedPatternRefinement:
    """Refines PII detection using advanced and contextual patterns."""
    
    def __init__(self):
        # Enhanced Lithuanian patterns with context awareness
        self.patterns = {
            'lithuanian_personal_code_contextual': {
                'pattern': r'(?:Asmens\s+kodas|Personal\s+code|A\.K\.):\s*(\d{11})',
                'flags': re.IGNORECASE, 
                'confidence_boost': 0.2,
                'description': 'Personal code with explicit label'
            },
            'lithuanian_vat_contextual': {
                'pattern': r'(?:PVM\s+kodas|VAT\s+code|PVM\s+Nr\.):\s*(LT\d{9,12})',
                'flags': re.IGNORECASE, 
                'confidence_boost': 0.2,
                'description': 'VAT code with explicit label'
            },
            'email_contextual': {
                'pattern': r'(?:El\.\s*paštas|Email|E-mail):\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                'flags': re.IGNORECASE, 
                'confidence_boost': 0.2,
                'description': 'Email with explicit label'
            },
            'phone_contextual': {
                'pattern': r'(?:Tel\.|Telefonas|Phone|Mob\.)\s*:\s*(\+?\d{1,4}(?:[\s-]+\d+)+)',
                'flags': re.IGNORECASE,
                'confidence_boost': 0.2, 
                'description': 'Phone number with explicit label' 
            },
            'address_contextual': {
                'pattern': r'(?:Adresas|Address):\s*([^,\n]+(?:,\s*[^,\n]+)*)',
                'flags': re.IGNORECASE, 
                'confidence_boost': 0.15,
                'description': 'Address with explicit label'
            },
            'lithuanian_name_contextual': { 
                'pattern': r'\b(?:vardas|pavardė|asmens?)[\s:]*([A-Ž][a-ž]+(?:\s+[A-Ž][a-ž]+)+)\b', 
                'flags': re.IGNORECASE,
                'confidence_boost': 0.15, 
                'description': 'Lithuanian name with contextual label' 
            },
        }
        
        self.compiled_patterns = {}
        for name, pattern_info in self.patterns.items():
            regex_str = pattern_info.get('pattern') or pattern_info.get('regex')
            flags = pattern_info.get('flags', 0)
            
            if not regex_str:
                context_logger.warning(f"Pattern {name} is missing 'pattern' or 'regex' key. Skipping compilation.")
                continue
            
            try:
                compiled_regex = re.compile(regex_str, flags)
            except re.error as e:
                context_logger.error(f"Failed to compile regex for pattern {name}: {regex_str} with flags {flags}. Error: {e}")
                continue

            self.compiled_patterns[name] = {
                'pattern': compiled_regex,
                'confidence_boost': pattern_info.get('confidence_boost', 0.0),
                'description': pattern_info.get('description', '')
            }
    
    def find_enhanced_patterns(self, text: str) -> List[Dict]:
        """
        Find PII using enhanced contextual patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of enhanced detections with confidence boosts
        """
        enhanced_detections = []
        
        for pattern_name, pattern_info in self.compiled_patterns.items():
            matches = pattern_info['pattern'].finditer(text)
            
            for match in matches:
                # Extract the actual PII (usually in group 1)
                pii_text = match.group(1) if match.groups() else match.group(0)
                
                detection = {
                    'text': pii_text,
                    'full_match': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'pattern_name': pattern_name,
                    'confidence_boost': pattern_info['confidence_boost'],
                    'description': pattern_info['description'],
                    'category': self._get_category_from_pattern(pattern_name)
                }
                
                enhanced_detections.append(detection)
                
                context_logger.info(
                    "Enhanced pattern detected",
                    pattern=pattern_name,
                    text=pii_text,
                    confidence_boost=pattern_info['confidence_boost']
                )
        
        return enhanced_detections
    
    def _get_category_from_pattern(self, pattern_name: str) -> str:
        """Map pattern name to PII category."""
        category_mapping = {
            'lithuanian_personal_code_contextual': 'lithuanian_personal_codes',
            'lithuanian_vat_contextual': 'lithuanian_vat_codes',
            'email_contextual': 'emails',
            'phone_contextual': 'phones',
            'address_contextual': 'addresses_prefixed',
            'lithuanian_name_contextual': 'lithuanian_names',
        }
        
        return category_mapping.get(pattern_name, 'unknown')


def create_context_aware_detection(text: str, category: str, start_pos: int, 
                                 end_pos: int, full_text: str, 
                                 validator: ContextualValidator) -> DetectionContext:
    """
    Create a context-aware detection with confidence scoring.
    
    Args:
        text: Detected text
        category: PII category
        start_pos: Start position in document
        end_pos: End position in document
        full_text: Full document text
        validator: ContextualValidator instance
        
    Returns:
        DetectionContext with comprehensive analysis
    """
    return validator.validate_with_context(
        text, category, full_text, start_pos, end_pos
    ) 