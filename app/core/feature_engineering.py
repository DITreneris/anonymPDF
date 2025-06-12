"""
Feature Engineering Pipeline for Priority 3 ML Implementation

This module extracts comprehensive features from PII detections for use in
ML confidence scoring and adaptive pattern learning.
"""

import re
import string
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import spacy
from collections import Counter

# Third-party feature libraries
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

# Import existing components
from app.core.config_manager import get_config
from app.core.logging import get_logger
from app.core.context_analyzer import DocumentStructureAnalyzer

feature_logger = get_logger(__name__)


@dataclass
class FeatureSet:
    """Container for all extracted features."""
    text_features: Dict[str, float]
    context_features: Dict[str, float]
    linguistic_features: Dict[str, float]
    document_features: Dict[str, float]
    pattern_features: Dict[str, float]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert all features to a single dictionary."""
        all_features = {}
        all_features.update(self.text_features)
        all_features.update(self.context_features)
        all_features.update(self.linguistic_features)
        all_features.update(self.document_features)
        all_features.update(self.pattern_features)
        return all_features


class TextFeatureExtractor:
    """Extract text-based features from detection strings."""
    
    def __init__(self):
        self.feature_cache = {}
        
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """
        Extract text-based features from detection text.
        
        Args:
            text: The detected text string
            
        Returns:
            Dictionary of text features
        """
        if not text:
            return self._get_empty_text_features()
        
        # Cache key for performance
        cache_key = hash(text)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = {}
        
        # Basic text metrics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['whitespace_count'] = text.count(' ') + text.count('\t') + text.count('\n')
        
        # Character diversity and composition
        unique_chars = len(set(text.lower()))
        features['char_diversity'] = unique_chars / len(text) if len(text) > 0 else 0
        features['unique_char_count'] = unique_chars
        
        # Character type ratios
        total_chars = len(text)
        if total_chars > 0:
            features['digit_ratio'] = sum(c.isdigit() for c in text) / total_chars
            features['alpha_ratio'] = sum(c.isalpha() for c in text) / total_chars
            features['special_char_ratio'] = sum(not c.isalnum() and not c.isspace() for c in text) / total_chars
            features['uppercase_ratio'] = sum(c.isupper() for c in text) / total_chars
            features['lowercase_ratio'] = sum(c.islower() for c in text) / total_chars
            features['space_ratio'] = sum(c.isspace() for c in text) / total_chars
        else:
            features.update({
                'digit_ratio': 0, 'alpha_ratio': 0, 'special_char_ratio': 0,
                'uppercase_ratio': 0, 'lowercase_ratio': 0, 'space_ratio': 0
            })
        
        # Language-specific features
        features['has_lithuanian_chars'] = float(any(c in 'ąčęėįšųūž' for c in text.lower()))
        features['has_accented_chars'] = float(any(c in 'àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ' for c in text.lower()))
        
        # Pattern-based features
        features['has_digits'] = float(any(c.isdigit() for c in text))
        features['has_punctuation'] = float(any(c in string.punctuation for c in text))
        features['starts_with_capital'] = float(text[0].isupper() if text else 0)
        features['all_caps'] = float(text.isupper() if text else 0)
        features['title_case'] = float(text.istitle() if text else 0)
        
        # Common patterns
        features['has_phone_pattern'] = float(bool(re.search(r'\+?\d{1,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}', text)))
        features['has_email_pattern'] = float(bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)))
        features['has_url_pattern'] = float(bool(re.search(r'https?://|www\.', text)))
        features['has_date_pattern'] = float(bool(re.search(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', text)))
        features['has_id_pattern'] = float(bool(re.search(r'\b\d{11}\b|\b[A-Z]{2}\d{6}\b', text)))
        
        # Word-level features
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
            features['max_word_length'] = max(len(word) for word in words)
            features['min_word_length'] = min(len(word) for word in words)
            features['word_length_variance'] = np.var([len(word) for word in words])
        else:
            features.update({
                'avg_word_length': 0, 'max_word_length': 0,
                'min_word_length': 0, 'word_length_variance': 0
            })
        
        # Readability features (if textstat is available)
        if TEXTSTAT_AVAILABLE and len(text) > 10:
            try:
                features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
                features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
                features['automated_readability_index'] = textstat.automated_readability_index(text)
            except:
                features.update({
                    'flesch_reading_ease': 50, 'flesch_kincaid_grade': 10,
                    'automated_readability_index': 10
                })
        else:
            features.update({
                'flesch_reading_ease': 50, 'flesch_kincaid_grade': 10,
                'automated_readability_index': 10
            })
        
        # Cache the result
        self.feature_cache[cache_key] = features
        return features
    
    def _get_empty_text_features(self) -> Dict[str, float]:
        """Return empty text features for null/empty input."""
        return {
            'text_length': 0, 'word_count': 0, 'char_count': 0, 'whitespace_count': 0,
            'char_diversity': 0, 'unique_char_count': 0,
            'digit_ratio': 0, 'alpha_ratio': 0, 'special_char_ratio': 0,
            'uppercase_ratio': 0, 'lowercase_ratio': 0, 'space_ratio': 0,
            'has_lithuanian_chars': 0, 'has_accented_chars': 0,
            'has_digits': 0, 'has_punctuation': 0, 'starts_with_capital': 0,
            'all_caps': 0, 'title_case': 0,
            'has_phone_pattern': 0, 'has_email_pattern': 0, 'has_url_pattern': 0,
            'has_date_pattern': 0, 'has_id_pattern': 0,
            'avg_word_length': 0, 'max_word_length': 0, 'min_word_length': 0,
            'word_length_variance': 0,
            'flesch_reading_ease': 50, 'flesch_kincaid_grade': 10,
            'automated_readability_index': 10
        }


class ContextFeatureExtractor:
    """Extract context-based features from surrounding text."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.structure_analyzer = DocumentStructureAnalyzer()
        
    def extract_context_features(self, detection_text: str, context: str, 
                               position: int, full_text: str = None) -> Dict[str, float]:
        """
        Extract context-based features.
        
        Args:
            detection_text: The detected text
            context: Surrounding context
            position: Position in document
            full_text: Full document text
            
        Returns:
            Dictionary of context features
        """
        features = {}
        
        if not context:
            return self._get_empty_context_features()
        
        # Context window analysis
        before_text, after_text = self._split_context(context, detection_text)
        
        # Before/after context features
        features['context_before_length'] = len(before_text)
        features['context_after_length'] = len(after_text)
        features['context_total_length'] = len(context)
        
        # Context composition
        features['context_word_count'] = len(context.split())
        features['context_sentence_count'] = len(re.split(r'[.!?]+', context))
        features['context_paragraph_count'] = len(context.split('\n\n'))
        
        # Proximity features
        features['words_before'] = len(before_text.split())
        features['words_after'] = len(after_text.split())
        
        # Context indicators
        features['has_colon_before'] = float(':' in before_text[-10:] if len(before_text) > 0 else 0)
        features['has_comma_before'] = float(',' in before_text[-5:] if len(before_text) > 0 else 0)
        features['has_period_after'] = float('.' in after_text[:5] if len(after_text) > 0 else 0)
        features['has_newline_before'] = float('\n' in before_text[-20:] if len(before_text) > 0 else 0)
        features['has_newline_after'] = float('\n' in after_text[:20] if len(after_text) > 0 else 0)
        
        # Label/field indicators
        label_patterns = [
            r'(name|vardas|nombre|nome):\s*$',
            r'(address|adresas|dirección|indirizzo):\s*$',
            r'(phone|telefonas|teléfono|telefono):\s*$',
            r'(email|el\.\s*paštas):\s*$',
            r'(company|įmonė|empresa|società):\s*$'
        ]
        
        features['has_field_label'] = 0.0
        for pattern in label_patterns:
            if re.search(pattern, before_text, re.IGNORECASE):
                features['has_field_label'] = 1.0
                break
        
        # Document structure context
        if full_text and position >= 0:
            doc_section = self.structure_analyzer.identify_document_section(
                full_text, position, self.window_size
            )
            features['in_header'] = float(doc_section == 'header')
            features['in_footer'] = float(doc_section == 'footer')
            features['in_table'] = float(doc_section == 'table_header')
            features['in_form'] = float(doc_section == 'form_field')
            features['in_legal'] = float(doc_section == 'legal_text')
            
            # Position-based features
            relative_position = position / len(full_text) if full_text else 0.5
            features['relative_position'] = relative_position
            features['in_first_quarter'] = float(relative_position < 0.25)
            features['in_last_quarter'] = float(relative_position > 0.75)
        else:
            features.update({
                'in_header': 0, 'in_footer': 0, 'in_table': 0,
                'in_form': 0, 'in_legal': 0,
                'relative_position': 0.5, 'in_first_quarter': 0, 'in_last_quarter': 0
            })
        
        # Surrounding text quality
        if context:
            # Language consistency
            features['context_has_lithuanian'] = float(any(c in 'ąčęėįšųūž' for c in context.lower()))
            features['context_mixed_case'] = float(any(c.isupper() for c in context) and any(c.islower() for c in context))
            features['context_mostly_caps'] = float(sum(c.isupper() for c in context) / len(context) > 0.5 if context else 0)
        
        return features
    
    def _split_context(self, context: str, detection_text: str) -> Tuple[str, str]:
        """Split context into before and after the detection."""
        detection_pos = context.find(detection_text)
        if detection_pos == -1:
            # If detection not found in context, split in the middle
            mid = len(context) // 2
            return context[:mid], context[mid:]
        
        before = context[:detection_pos]
        after = context[detection_pos + len(detection_text):]
        return before, after
    
    def _get_empty_context_features(self) -> Dict[str, float]:
        """Return empty context features."""
        return {
            'context_before_length': 0, 'context_after_length': 0, 'context_total_length': 0,
            'context_word_count': 0, 'context_sentence_count': 0, 'context_paragraph_count': 0,
            'words_before': 0, 'words_after': 0,
            'has_colon_before': 0, 'has_comma_before': 0, 'has_period_after': 0,
            'has_newline_before': 0, 'has_newline_after': 0, 'has_field_label': 0,
            'in_header': 0, 'in_footer': 0, 'in_table': 0, 'in_form': 0, 'in_legal': 0,
            'relative_position': 0.5, 'in_first_quarter': 0, 'in_last_quarter': 0,
            'context_has_lithuanian': 0, 'context_mixed_case': 0, 'context_mostly_caps': 0
        }


class LinguisticFeatureExtractor:
    """Extract linguistic features using NLP models."""
    
    def __init__(self):
        self.nlp_models = {}
        self._load_nlp_models()
        
    def _load_nlp_models(self):
        """Load spaCy NLP models."""
        models_to_load = ['lt_core_news_sm', 'en_core_web_sm']
        
        for model_name in models_to_load:
            try:
                self.nlp_models[model_name] = spacy.load(model_name)
                feature_logger.debug(f"Loaded spaCy model: {model_name}")
            except OSError:
                feature_logger.warning(f"spaCy model not available: {model_name}")
    
    def extract_linguistic_features(self, text: str, context: str = None, 
                                  language: str = 'lt') -> Dict[str, float]:
        """
        Extract linguistic features using spaCy.
        
        Args:
            text: Text to analyze
            context: Optional context for better analysis
            language: Language code
            
        Returns:
            Dictionary of linguistic features
        """
        features = {}
        
        if not text:
            return self._get_empty_linguistic_features()
        
        # Select appropriate model
        model_name = self._get_model_for_language(language)
        nlp = self.nlp_models.get(model_name)
        
        if not nlp:
            feature_logger.warning(f"No spaCy model available for language: {language}")
            return self._get_empty_linguistic_features()
        
        try:
            # Process text
            doc = nlp(text)
            
            # POS tag features
            pos_counts = Counter(token.pos_ for token in doc)
            total_tokens = len(doc)
            
            if total_tokens > 0:
                features['noun_ratio'] = pos_counts.get('NOUN', 0) / total_tokens
                features['verb_ratio'] = pos_counts.get('VERB', 0) / total_tokens
                features['adj_ratio'] = pos_counts.get('ADJ', 0) / total_tokens
                features['propn_ratio'] = pos_counts.get('PROPN', 0) / total_tokens  # Proper nouns
                features['num_ratio'] = pos_counts.get('NUM', 0) / total_tokens
                features['punct_ratio'] = pos_counts.get('PUNCT', 0) / total_tokens
            else:
                features.update({
                    'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0,
                    'propn_ratio': 0, 'num_ratio': 0, 'punct_ratio': 0
                })
            
            # Named entity features
            ent_types = Counter(ent.label_ for ent in doc.ents)
            features['has_person_entity'] = float('PERSON' in ent_types)
            features['has_org_entity'] = float('ORG' in ent_types)
            features['has_gpe_entity'] = float('GPE' in ent_types)  # Geopolitical entity
            features['has_date_entity'] = float('DATE' in ent_types)
            features['has_time_entity'] = float('TIME' in ent_types)
            features['entity_count'] = len(doc.ents)
            features['entity_density'] = len(doc.ents) / total_tokens if total_tokens > 0 else 0
            
            # Morphological features
            features['has_title_case_token'] = float(any(token.is_title for token in doc))
            features['has_alpha_token'] = float(any(token.is_alpha for token in doc))
            features['has_digit_token'] = float(any(token.is_digit for token in doc))
            features['has_stop_word'] = float(any(token.is_stop for token in doc))
            
            # Dependency features (simplified for performance)
            dep_types = Counter(token.dep_ for token in doc)
            features['has_compound'] = float('compound' in dep_types)
            features['has_root'] = float('ROOT' in dep_types)
            features['has_det'] = float('det' in dep_types)  # Determiner
            
            # Token-level statistics
            if total_tokens > 0:
                token_lengths = [len(token.text) for token in doc]
                features['avg_token_length'] = np.mean(token_lengths)
                features['token_length_std'] = np.std(token_lengths)
                features['max_token_length'] = max(token_lengths)
            else:
                features.update({
                    'avg_token_length': 0, 'token_length_std': 0, 'max_token_length': 0
                })
            
        except Exception as e:
            feature_logger.warning(f"Linguistic feature extraction failed: {e}")
            return self._get_empty_linguistic_features()
        
        return features
    
    def _get_model_for_language(self, language: str) -> str:
        """Get appropriate spaCy model for language."""
        language_models = {
            'lt': 'lt_core_news_sm',
            'en': 'en_core_web_sm',
            'lv': 'en_core_web_sm',  # Fallback
            'et': 'en_core_web_sm',  # Fallback
            'pl': 'en_core_web_sm',  # Fallback
            'de': 'en_core_web_sm',  # Fallback
            'fr': 'en_core_web_sm',  # Fallback
        }
        
        return language_models.get(language.lower(), 'en_core_web_sm')
    
    def _get_empty_linguistic_features(self) -> Dict[str, float]:
        """Return empty linguistic features."""
        return {
            'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0,
            'propn_ratio': 0, 'num_ratio': 0, 'punct_ratio': 0,
            'has_person_entity': 0, 'has_org_entity': 0, 'has_gpe_entity': 0,
            'has_date_entity': 0, 'has_time_entity': 0, 'entity_count': 0, 'entity_density': 0,
            'has_title_case_token': 0, 'has_alpha_token': 0, 'has_digit_token': 0, 'has_stop_word': 0,
            'has_compound': 0, 'has_root': 0, 'has_det': 0,
            'avg_token_length': 0, 'token_length_std': 0, 'max_token_length': 0
        }


class PatternFeatureExtractor:
    """Extract pattern-based features specific to PII types."""
    
    def __init__(self):
        self.pii_patterns = self._initialize_pii_patterns()
        
    def _initialize_pii_patterns(self) -> Dict[str, List[str]]:
        """Initialize PII-specific patterns."""
        return {
            'person_name': [
                r'^[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+\s+[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+$',  # Lithuanian names
                r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Generic names
                r'^(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+',  # Titles + names
            ],
            'organization': [
                r'\b(UAB|AB|VšĮ|MB|IĮ)\b',  # Lithuanian company types
                r'\b(Ltd|LLC|Inc|Corp|GmbH|SA)\b',  # International company types
                r'\b[A-Z][a-z]+\s+(Company|Corp|Ltd)\b',
            ],
            'email': [
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            ],
            'phone': [
                r'^\+370\s?\d{8}$',  # Lithuanian mobile
                r'^\+\d{1,4}\s?\d{6,14}$',  # International
                r'^\d{8}$',  # Local format
            ],
            'address': [
                r'\b(gatvė|g\.|str\.|street|avenue|ave)\b',
                r'\d+[a-z]?\s+(gatvė|g\.|str\.)',
                r'LT-\d{5}',  # Lithuanian postal code
            ],
            'id_number': [
                r'^\d{11}$',  # Lithuanian personal code
                r'^[A-Z]{2}\d{6}$',  # Document numbers
            ]
        }
    
    def extract_pattern_features(self, text: str, category: str = None) -> Dict[str, float]:
        """
        Extract pattern-based features.
        
        Args:
            text: Text to analyze
            category: Optional PII category hint
            
        Returns:
            Dictionary of pattern features
        """
        features = {}
        
        if not text:
            return self._get_empty_pattern_features()
        
        # Check against all pattern types
        for pattern_type, patterns in self.pii_patterns.items():
            feature_name = f'matches_{pattern_type}_pattern'
            features[feature_name] = 0.0
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    features[feature_name] = 1.0
                    break
        
        # Category-specific pattern strength
        if category and category in self.pii_patterns:
            features['category_pattern_strength'] = self._calculate_pattern_strength(
                text, self.pii_patterns[category]
            )
        else:
            features['category_pattern_strength'] = 0.0
        
        # General pattern features
        features['has_repeated_chars'] = float(self._has_repeated_characters(text))
        features['has_sequential_chars'] = float(self._has_sequential_characters(text))
        features['char_pattern_complexity'] = self._calculate_pattern_complexity(text)
        
        return features
    
    def _calculate_pattern_strength(self, text: str, patterns: List[str]) -> float:
        """Calculate how strongly text matches given patterns."""
        match_scores = []
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Calculate match quality (simplified)
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    match_ratio = len(match.group()) / len(text)
                    match_scores.append(match_ratio)
        
        return max(match_scores) if match_scores else 0.0
    
    def _has_repeated_characters(self, text: str, threshold: int = 3) -> bool:
        """Check if text has repeated characters."""
        for char in set(text.lower()):
            if text.lower().count(char) >= threshold:
                return True
        return False
    
    def _has_sequential_characters(self, text: str) -> bool:
        """Check if text has sequential characters (like abc or 123)."""
        for i in range(len(text) - 2):
            if text[i:i+3].isdigit():
                digits = [int(d) for d in text[i:i+3]]
                if digits[1] == digits[0] + 1 and digits[2] == digits[1] + 1:
                    return True
            elif text[i:i+3].isalpha():
                chars = [ord(c.lower()) for c in text[i:i+3]]
                if chars[1] == chars[0] + 1 and chars[2] == chars[1] + 1:
                    return True
        return False
    
    def _calculate_pattern_complexity(self, text: str) -> float:
        """Calculate pattern complexity score."""
        if not text:
            return 0.0
        
        # Entropy-based complexity
        char_counts = Counter(text.lower())
        total_chars = len(text)
        entropy = -sum((count/total_chars) * np.log2(count/total_chars) 
                      for count in char_counts.values())
        
        # Normalize entropy by maximum possible entropy
        max_entropy = np.log2(len(char_counts)) if len(char_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _get_empty_pattern_features(self) -> Dict[str, float]:
        """Return empty pattern features."""
        features = {}
        
        # Pattern match features
        for pattern_type in self.pii_patterns.keys():
            features[f'matches_{pattern_type}_pattern'] = 0.0
        
        # Additional pattern features
        features.update({
            'category_pattern_strength': 0.0,
            'has_repeated_chars': 0.0,
            'has_sequential_chars': 0.0,
            'char_pattern_complexity': 0.0
        })
        
        return features


class FeatureExtractor:
    """Main feature extraction coordinator."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config().get('feature_engineering', {})
        
        # Initialize sub-extractors
        self.text_extractor = TextFeatureExtractor()
        self.context_extractor = ContextFeatureExtractor(
            window_size=self.config.get('context_features', {}).get('window_size', 50)
        )
        self.linguistic_extractor = LinguisticFeatureExtractor()
        self.pattern_extractor = PatternFeatureExtractor()
        
        # Performance tracking
        self.extraction_count = 0
        self.total_extraction_time = 0.0
        
    def extract_all_features(self, detection_text: str, category: str, context: str,
                           position: int = -1, full_text: str = None,
                           document_type: str = None, language: str = 'lt') -> FeatureSet:
        """
        Extract all features for a PII detection.
        
        Args:
            detection_text: The detected PII text
            category: PII category
            context: Surrounding context
            position: Position in document
            full_text: Full document text
            document_type: Type of document
            language: Document language
            
        Returns:
            FeatureSet with all extracted features
        """
        start_time = datetime.now()
        
        try:
            # Extract features from each component
            text_features = self.text_extractor.extract_text_features(detection_text)
            
            context_features = self.context_extractor.extract_context_features(
                detection_text, context, position, full_text
            )
            
            linguistic_features = self.linguistic_extractor.extract_linguistic_features(
                detection_text, context, language
            )
            
            pattern_features = self.pattern_extractor.extract_pattern_features(
                detection_text, category
            )
            
            # Document-level features
            document_features = self._extract_document_features(
                document_type, language, len(full_text) if full_text else 0
            )
            
            feature_set = FeatureSet(
                text_features=text_features,
                context_features=context_features,
                linguistic_features=linguistic_features,
                document_features=document_features,
                pattern_features=pattern_features
            )
            
            # Update performance tracking
            self.extraction_count += 1
            extraction_time = (datetime.now() - start_time).total_seconds()
            self.total_extraction_time += extraction_time
            
            if self.extraction_count % 100 == 0:
                avg_time = self.total_extraction_time / self.extraction_count
                feature_logger.debug(f"Feature extraction performance: {avg_time*1000:.2f}ms avg")
            
            return feature_set
            
        except Exception as e:
            feature_logger.error(f"Feature extraction failed: {e}")
            # Return empty features as fallback
            return FeatureSet(
                text_features=self.text_extractor._get_empty_text_features(),
                context_features=self.context_extractor._get_empty_context_features(),
                linguistic_features=self.linguistic_extractor._get_empty_linguistic_features(),
                document_features=self._get_empty_document_features(),
                pattern_features=self.pattern_extractor._get_empty_pattern_features()
            )
    
    def _extract_document_features(self, document_type: str, language: str, 
                                 document_length: int) -> Dict[str, float]:
        """Extract document-level features."""
        features = {}
        
        # Document type features
        doc_types = ['insurance', 'legal', 'financial', 'medical', 'personal', 'other']
        for doc_type in doc_types:
            features[f'is_{doc_type}_document'] = float(
                document_type and document_type.lower() == doc_type
            )
        
        # Language features
        languages = ['lt', 'en', 'lv', 'et', 'pl', 'de', 'fr']
        for lang in languages:
            features[f'is_{lang}_language'] = float(language == lang)
        
        # Document size features
        features['document_length'] = document_length
        features['is_short_document'] = float(document_length < 1000)
        features['is_medium_document'] = float(1000 <= document_length < 10000)
        features['is_long_document'] = float(document_length >= 10000)
        
        return features
    
    def _get_empty_document_features(self) -> Dict[str, float]:
        """Return empty document features."""
        features = {}
        
        # Document type features
        doc_types = ['insurance', 'legal', 'financial', 'medical', 'personal', 'other']
        for doc_type in doc_types:
            features[f'is_{doc_type}_document'] = 0.0
        
        # Language features
        languages = ['lt', 'en', 'lv', 'et', 'pl', 'de', 'fr']
        for lang in languages:
            features[f'is_{lang}_language'] = 0.0
        
        # Document size features
        features.update({
            'document_length': 0,
            'is_short_document': 0,
            'is_medium_document': 0,
            'is_long_document': 0
        })
        
        return features
    
    def get_feature_importance(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        """Calculate feature importance scores."""
        # Simple feature importance based on non-zero values and variance
        importance = {}
        
        for feature_name, value in feature_dict.items():
            # Higher importance for non-zero, non-default values
            if value != 0:
                importance[feature_name] = abs(value)
            else:
                importance[feature_name] = 0.01  # Small importance for zero values
        
        return importance


# Factory function for easy integration
def create_feature_extractor(config: Optional[Dict] = None) -> FeatureExtractor:
    """Create and return feature extractor instance."""
    return FeatureExtractor(config) 