"""
Adaptive Learning: Pattern Discovery Module

This module is responsible for discovering new PII patterns from text
and validating them before they are added to the adaptive pattern database.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from app.core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


@dataclass
class ValidatedPattern:
    """
    Represents a pattern that has been validated.
    This structure is designed to align with the schema in AdaptivePatternDB.
    """
    pattern_id: str
    regex: str
    pii_category: str
    confidence: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    positive_matches: int = 0
    negative_matches: int = 0
    version: int = 1

    created_at: datetime = field(default_factory=datetime.now)
    validated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


class PatternLearner:
    """
    Discovers and validates potential PII patterns from text corpora.
    This class is responsible for taking raw text and feedback, generating
    """

    def __init__(self, pattern_db: "AdaptivePatternDB"):
        """
        Initializes the PatternLearner.

        Args:
            pattern_db: An instance of AdaptivePatternDB to store validated patterns.
        """
        self.pattern_db = pattern_db
        # More sophisticated configuration could be loaded here
        self.min_confidence_to_validate = 0.85  # Precision must be >= 85%
        self.min_recall_to_validate = 0.85     # Recall must be >= 85%
        logger.info("PatternLearner initialized.")

    def discover_and_validate_patterns(self, text_corpus: List[str], pii_to_discover: Dict[str, str], ground_truth_pii: Dict[str, str]) -> List[ValidatedPattern]:
        """
        Discovers potential new patterns and validates them.

        Args:
            text_corpus: A list of text documents to validate against.
            pii_to_discover: A dictionary of PII strings to generate patterns from.
            ground_truth_pii: The complete dictionary of all known PII for validation.

        Returns:
            A list of validated patterns ready for database insertion.
        """
        validated_patterns = []
        logger.debug(f"Starting pattern discovery for {len(pii_to_discover)} confirmed PII samples.")

        for pii, category in pii_to_discover.items():
            try:
                # 1. Discover a potential pattern (simplified)
                escaped_pii = re.escape(pii)
                pattern_regex = f"\\b{escaped_pii}\\b"

                # 2. Validate the pattern against the full ground truth
                validation_results = self._validate_regex(pattern_regex, text_corpus, ground_truth_pii)
                precision = validation_results['precision']

                if precision >= self.min_confidence_to_validate:
                    new_pattern = ValidatedPattern(
                        pattern_id=f"p_{hash(pattern_regex)}",
                        regex=pattern_regex,
                        pii_category=category,
                        confidence=precision,
                        precision=precision,
                        recall=validation_results['recall'],
                        positive_matches=validation_results['true_positives'],
                        negative_matches=validation_results['false_positives']
                    )
                    validated_patterns.append(new_pattern)
                    logger.info(f"Discovered and validated new pattern: {pattern_regex} with precision {precision:.2f}")

            except re.error as e:
                logger.error(f"Could not process PII sample '{pii}': {e}")
                continue

        return validated_patterns

    def _validate_regex(self, regex: str, corpus: List[str], known_positives: Dict[str, str]) -> Dict[str, float]:
        """
        Calculates precision and recall for a given regex against a corpus.
        Metrics are based on unique strings found.
        """
        try:
            compiled_regex = re.compile(regex, re.IGNORECASE)
            
            # Use sets to store unique matches
            all_matches = set()
            for doc in corpus:
                matches_in_doc = compiled_regex.findall(doc)
                all_matches.update(matches_in_doc)

            found_true_positives = all_matches.intersection(known_positives.keys())
            found_false_positives = all_matches.difference(known_positives.keys())

            tp_count = len(found_true_positives)
            fp_count = len(found_false_positives)
            
            # Precision: Of all the unique strings we found, what percentage were real PII?
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            
            # Recall: Of all the unique real PII, what percentage did we find?
            recall = tp_count / len(known_positives) if len(known_positives) > 0 else 0.0

            # Count occurrences for the final report, but don't use for metrics
            positive_match_occurrences = sum(doc.count(p) for p in found_true_positives for doc in corpus)
            negative_match_occurrences = sum(doc.count(p) for p in found_false_positives for doc in corpus)

            return {
                "precision": precision,
                "recall": recall,
                "true_positives": positive_match_occurrences,
                "false_positives": negative_match_occurrences
            }
        except re.error:
            return {"precision": 0.0, "recall": 0.0, "true_positives": 0, "false_positives": 0} 