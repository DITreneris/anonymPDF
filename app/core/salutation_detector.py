"""
Lithuanian salutation detection module.

This module detects and extracts personal names from Lithuanian formal salutations
and greetings, handling different grammatical cases and formal address patterns.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from app.core.context_analyzer import DetectionContext, create_context_aware_detection, ContextualValidator
from app.core.config_manager import get_config_manager

# Setup logging
salutation_logger = logging.getLogger(__name__)

@dataclass
class SalutationDetection:
    """Represents a detected salutation with extracted name."""
    full_text: str          # Full salutation text
    extracted_name: str     # Extracted personal name
    base_name: str         # Base form of the name (nominative case)
    start_pos: int         # Start position in text
    end_pos: int           # End position in text
    confidence: float      # Detection confidence (0.0-1.0)
    salutation_type: str   # Type of salutation pattern

class LithuanianSalutationDetector:
    """
    Detects Lithuanian salutations and extracts personal names from formal greetings.
    
    Handles various Lithuanian grammatical cases and formal address patterns.
    """
    
    def __init__(self):
        """Initialize the Lithuanian salutation detector."""
        self.salutation_patterns = self._build_salutation_patterns()
        self.name_case_endings = self._build_name_case_patterns()
        self.formal_titles = self._build_formal_titles()
        
    def _build_salutation_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for different types of Lithuanian salutations."""
        return {
            'formal_address': [
                # Dative case patterns (Gerbiamam/Gerbiamai + Name)
                r'Gerbiamam\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ui?)',
                r'Gerbiamai\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?)',
                r'Gerbiamam\s+p\.\s*([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ui?)',
                r'Gerbiamai\s+p\.\s*([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?)',
                
                # Vocative case patterns (direct address)
                r'Gerbiamas\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?)',
                r'Gerbiama\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+e)',
                
                # With titles
                r'Gerbiamam\s+(?:ponaui?|p\.)\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ui?)',
                r'Gerbiamai\s+(?:poniai?|p\.)\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?)',
            ],
            
            'greeting_patterns': [
                # Sveiki/Labas + Name variations
                r'Sveiki,?\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?)',
                r'Labas,?\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?)',
                r'Laba\s+diena,?\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?)',
                
                # Hello + Name
                r'Sveika,?\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+e)',
                r'Laba\s+diena,?\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+e)',
            ],
            
            'closing_patterns': [
                # Su pagarba / Pagarbiai + Name
                r'Su\s+pagarba,?\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+s)',
                r'Pagarbiai,?\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+s)',
                r'Ačiū,?\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?)',
            ],
            
            'direct_address': [
                # Direct name addressing patterns
                r'([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?),?\s+(?:prašau|galėčiau|norėčiau)',
                r'([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+e),?\s+(?:prašau|galėčiau|norėčiau)',
                r'(?:Prašau|Galėčiau),?\s+([A-ZĄĘĖĮŠŲŪŽ][a-ząęėįšųūž]+ai?)',
            ]
        }
    
    def _build_name_case_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for Lithuanian name case endings."""
        return {
            'masculine_dative': ['ui', 'iui'],           # Tomui, Petrui
            'feminine_dative': ['ai', 'iai', 'ei'],     # Onai, Marijai, Rutei
            'masculine_vocative': ['ai', 'iau'],        # Tomai, Petriau  
            'feminine_vocative': ['e', 'a'],            # Ona, Rita
            'masculine_nominative': ['s', 'as', 'us'],  # Tomas, Petras, Julius
            'feminine_nominative': ['a', 'ė', 'ė'],     # Ona, Rūtė, Aušra
        }
    
    def _build_formal_titles(self) -> Set[str]:
        """Build set of formal titles that might appear with names."""
        return {
            'ponas', 'ponia', 'p.', 'ponai', 'ponioms',
            'daktaras', 'daktarė', 'dr.', 'prof.',
            'inžinierius', 'inžinierė', 'ing.',
            'magistras', 'magistrė', 'mag.',
        }
    
    def detect_salutations(self, text: str) -> List[SalutationDetection]:
        """
        Detect Lithuanian salutations and extract personal names.
        
        Args:
            text: Text to analyze for salutations
            
        Returns:
            List of detected salutations with extracted names
        """
        detections = []
        
        salutation_logger.info(
            "Starting Lithuanian salutation detection",
            text_length=len(text),
            pattern_categories=len(self.salutation_patterns)
        )
        
        # Process each category of salutation patterns
        for category, patterns in self.salutation_patterns.items():
            category_detections = self._detect_pattern_category(text, category, patterns)
            detections.extend(category_detections)
        
        # Deduplicate and sort by confidence
        detections = self._deduplicate_detections(detections)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        salutation_logger.info(
            "Salutation detection completed",
            total_detections=len(detections),
            unique_names=len(set(d.base_name for d in detections))
        )
        
        return detections
    
    def _detect_pattern_category(self, text: str, category: str, patterns: List[str]) -> List[SalutationDetection]:
        """Detect salutations for a specific pattern category."""
        detections = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                # Extract the name from the captured group
                if match.groups():
                    detected_name = match.group(1).strip()
                    full_salutation = match.group(0).strip()
                    
                    # Convert name to base form (nominative case)
                    base_name = self._convert_to_base_name(detected_name)
                    
                    # Calculate confidence based on pattern type and context
                    confidence = self._calculate_confidence(
                        full_salutation, detected_name, category, text, match.start()
                    )
                    
                    # Only include high-confidence detections
                    if confidence >= 0.6:
                        detection = SalutationDetection(
                            full_text=full_salutation,
                            extracted_name=detected_name,
                            base_name=base_name,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence,
                            salutation_type=category
                        )
                        detections.append(detection)
                        
                        salutation_logger.info(
                            "Salutation detected",
                            category=category,
                            full_text=full_salutation,
                            extracted_name=detected_name,
                            base_name=base_name,
                            confidence=confidence
                        )
        
        return detections
    
    def _convert_to_base_name(self, name: str) -> str:
        """
        Convert a name in any Lithuanian case to its base (nominative) form.
        
        This is a simplified conversion - in practice, you might want to use
        a proper Lithuanian morphological analyzer.
        """
        # Check if it's a formal title that shouldn't be converted
        if name.lower() in self.formal_titles:
            return name  # Don't convert titles
        
        # Remove common case endings to get the base
        name_lower = name.lower()
        
        # Masculine dative endings -> nominative
        if name_lower.endswith('ui'):
            if name_lower.endswith('iui'):
                return name[:-3] + 'ius'  # Liuciui -> Liucius
            else:
                return name[:-2] + 'as'   # Tomui -> Tomas
        elif name_lower.endswith('iui'):
            return name[:-3] + 'ius'      # Petrui -> Petrus (rare)
        
        # Feminine dative endings -> nominative  
        elif name_lower.endswith('ai') and not self._is_likely_masculine_vocative(name):
            return name[:-2] + 'a'        # Onai -> Ona
        elif name_lower.endswith('iai'):
            return name[:-3] + 'ija'      # Marijai -> Marija
        elif name_lower.endswith('ei'):
            return name[:-2] + 'ė'        # Rutei -> Rūtė
        
        # Vocative endings -> nominative (improved logic)
        elif name_lower.endswith('iau'):
            return name[:-3] + 'ius'      # Petriau -> Petrus
        elif name_lower.endswith('ai') and len(name) > 3 and self._is_likely_masculine_vocative(name):
            return name[:-2] + 'as'       # Tomai -> Tomas (fixed!)
        elif name_lower.endswith('e') and len(name) > 2:
            # Check if it's likely feminine vocative
            if self._is_likely_feminine_name(name):
                return name[:-1] + 'a'    # Rūte -> Rūta
            else:
                return name               # Some names end in 'e' naturally
        
        # If no clear pattern, return as is (might already be nominative)
        return name
    
    def _is_likely_masculine_vocative(self, name: str) -> bool:
        """Check if a name ending in 'ai' is likely masculine vocative rather than feminine dative."""
        # Common masculine name patterns that use vocative 'ai'
        masculine_patterns = [
            r'[Tt]oma[i]$',      # Tomai
            r'[Jj]ona[i]$',      # Jonai  
            r'[Pp]etra[i]$',     # Petrai (though less common)
            r'[Aa]ntana[i]$',    # Antanai
        ]
        
        for pattern in masculine_patterns:
            if re.match(pattern, name):
                return True
        
        # Default heuristic: if name has typical masculine stem, treat as vocative
        stem = name[:-2].lower()
        if len(stem) >= 3 and not stem.endswith(('ij', 'ar', 'er')):
            return True
            
        return False
    
    def _is_likely_feminine_name(self, name: str) -> bool:
        """Check if a name is likely feminine based on common patterns."""
        name_lower = name.lower()
        
        # Common feminine name endings and patterns
        feminine_indicators = [
            name_lower.endswith('a'),      # Most feminine names end in 'a'
            name_lower.endswith('ė'),      # Many feminine names end in 'ė'
            name_lower.endswith('ija'),    # Names like Marija
            name_lower.endswith('ana'),    # Names like Svetlana
            name_lower.endswith('ina'),    # Names like Kristina
        ]
        
        return any(feminine_indicators)
    
    def _calculate_confidence(self, full_salutation: str, extracted_name: str, 
                            category: str, full_text: str, position: int) -> float:
        """Calculate confidence score for a salutation detection."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on category
        category_boosts = {
            'formal_address': 0.3,     # "Gerbiamam Tomui" is very formal
            'greeting_patterns': 0.2,  # "Sveiki, Tomai" is clear
            'closing_patterns': 0.2,   # "Pagarbiai, Tomas" is formal
            'direct_address': 0.1      # Less certain context
        }
        confidence += category_boosts.get(category, 0.0)
        
        # Boost if name appears at document beginning (typical for salutations)
        if position < 500:  # First 500 characters
            confidence += 0.1
        
        # Boost if contains formal markers
        formal_markers = ['gerbiamam', 'gerbiamai', 'gerbiamas', 'gerbiama', 'su pagarba', 'pagarbiai']
        if any(marker in full_salutation.lower() for marker in formal_markers):
            confidence += 0.1
        
        # Boost for proper name capitalization
        if extracted_name and extracted_name[0].isupper():
            confidence += 0.05
        
        # Penalty for very short names (likely false positives)
        if len(extracted_name) < 3:
            confidence -= 0.2
        
        # Penalty for very long "names" (likely not names)
        if len(extracted_name) > 15:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _deduplicate_detections(self, detections: List[SalutationDetection]) -> List[SalutationDetection]:
        """Remove duplicate detections, keeping the highest confidence ones."""
        if not detections:
            return []
        
        # Group by base name
        name_groups = {}
        for detection in detections:
            base_name = detection.base_name.lower()
            if base_name not in name_groups:
                name_groups[base_name] = []
            name_groups[base_name].append(detection)
        
        # Keep the highest confidence detection for each base name
        deduplicated = []
        for base_name, group in name_groups.items():
            best_detection = max(group, key=lambda x: x.confidence)
            deduplicated.append(best_detection)
            
            if len(group) > 1:
                salutation_logger.info(
                    "Deduplicated salutation detections",
                    base_name=base_name,
                    kept_confidence=best_detection.confidence,
                    removed_count=len(group) - 1
                )
        
        return deduplicated
    
    def extract_names_for_redaction(self, detections: List[SalutationDetection]) -> List[Tuple[str, str]]:
        """
        Extract names from salutations for redaction purposes.
        
        Returns both the original detected form and the base form for comprehensive redaction.
        """
        redaction_names = []
        
        for detection in detections:
            # Add the detected name form (e.g., "Tomui")
            redaction_names.append((detection.extracted_name, f"SALUTATION_DETECTED_CONF_{detection.confidence:.2f}"))
            
            # Add the base name form (e.g., "Tomas") 
            if detection.base_name != detection.extracted_name:
                redaction_names.append((detection.base_name, f"SALUTATION_BASE_CONF_{detection.confidence:.2f}"))
            
            # Also add the full salutation text for context-aware redaction
            redaction_names.append((detection.full_text, f"SALUTATION_FULL_CONF_{detection.confidence:.2f}"))
        
        return redaction_names

def detect_lithuanian_salutations(text: str) -> List[DetectionContext]:
    """
    Top-level function to detect Lithuanian salutations and return them as
    DetectionContext objects, ready for the main PII processing pipeline.
    """
    if not hasattr(detect_lithuanian_salutations, "detector"):
        # Initialize the detector once and cache it as a function attribute
        detect_lithuanian_salutations.detector = LithuanianSalutationDetector()

    detector = detect_lithuanian_salutations.detector
    salutation_detections = detector.detect_salutations(text)
    
    context_detections = []
    
    # Correctly initialize the validator using the config manager
    config_manager = get_config_manager()
    validator = ContextualValidator(
        cities=config_manager.cities, 
        brand_names=config_manager.brand_names
    )

    for salutation in salutation_detections:
        # Convert each SalutationDetection into a standard DetectionContext
        detection = create_context_aware_detection(
            text=salutation.extracted_name,
            category='person_name',
            start=salutation.start_pos,
            end=salutation.end_pos,
            full_text=text,
            validator=validator
        )
        detection.confidence = salutation.confidence
        context_detections.append(detection)

    return context_detections 