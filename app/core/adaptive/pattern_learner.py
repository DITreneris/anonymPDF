"""
Adaptive Learning: Pattern Discovery Module

This module is responsible for discovering new PII patterns from text
and validating them before they are added to the adaptive pattern database.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import re
from datetime import datetime

from app.core.logging import get_logger
from app.core.config_manager import get_config_manager
from app.core.adaptive.pattern_db import AdaptivePattern
from app.core.feedback_system import UserFeedbackProcessor

if TYPE_CHECKING:
    from app.core.adaptive.pattern_db import AdaptivePatternDB

# Initialize logger
logger = get_logger("anonympdf.pattern_learner")


@dataclass
class ValidatedPattern:
    """A pattern that has been discovered and validated against a corpus."""
    regex: str
    pii_category: str
    confidence: float
    positive_matches: int
    negative_matches: int
    created_at: datetime = field(default_factory=datetime.now)
    last_validated_at: Optional[datetime] = None


class PatternLearner:
    """
    Analyzes user feedback to discover and validate new PII patterns.
    """
    def __init__(self, pattern_db: AdaptivePatternDB, min_confidence: float = 0.95):
        self.pattern_db = pattern_db
        self.min_confidence = min_confidence
        self.feedback_processor = UserFeedbackProcessor()
        logger.info(f"PatternLearner initialized - {{'min_confidence': {self.min_confidence}}}")

    def discover_and_validate_patterns(
        self,
        text_corpus: List[str],
        pii_to_discover: Dict[str, str],
        ground_truth_pii: Dict[str, str],
        min_samples_for_learning: int = 1,
    ) -> List[AdaptivePattern]:
        """
        Discovers and validates new patterns from user feedback.
        """
        new_patterns = []

        if not pii_to_discover:
            return []

        for pii, category in pii_to_discover.items():
            try:
                # 1) Create a raw pattern for validation
                raw_pattern = r'\b' + re.escape(pii) + r'\b'
                logger.debug(f"Generated raw pattern: {raw_pattern}")

                # 2) Validate the raw pattern
                validation_results = self._validate_regex(
                    raw_pattern,
                    text_corpus,
                    pii,
                    ground_truth_pii
                )
                precision = validation_results['precision']
                recall = validation_results['recall']
                total_samples = validation_results['true_positives'] + validation_results['false_positives']

                # 3) Check if the pattern meets the criteria
                if precision >= self.min_confidence and recall >= 0.8 and total_samples >= min_samples_for_learning:
                    # 4) Create and add the new pattern object using the raw_pattern
                    new_pattern = AdaptivePattern(
                        pattern_id=f"p_{hash(raw_pattern)}",
                        regex=raw_pattern,
                        pii_category=category,
                        confidence=precision,
                        precision=precision,
                        recall=recall,
                        positive_matches=validation_results['true_positives'],
                        negative_matches=validation_results['false_positives'],
                        last_validated_at=datetime.now()
                    )
                    new_patterns.append(new_pattern)
                    
                    # 5) Log the newly discovered pattern
                    logger.info(
                        f"Discovered and validated new pattern: {raw_pattern} - "
                        f"{{'precision': '{precision:.2f}', 'recall': '{recall:.2f}'}}"
                    )
            except Exception as e:
                logger.error(
                    f"Error discovering pattern for PII: {pii}. Error: {e}",
                    exc_info=True
                )
                continue

        return new_patterns

    def _validate_regex(self, regex: str, corpus: List[str], target_pii: str, all_known_pii: Dict[str, str]) -> Dict:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # TP and FP
        for text in corpus:
            matches = re.findall(regex, text)
            for match in matches:
                if match in all_known_pii:
                    true_positives += 1
                else:
                    false_positives += 1
        
        # FN
        found_target_pii_in_corpus = any(target_pii in text for text in corpus)
        if found_target_pii_in_corpus and true_positives == 0:
             # This logic is simplified; a real system would need to check if the specific `target_pii` was found
             false_negatives = 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        return {"precision": precision, "recall": recall, "true_positives": true_positives, "false_positives": false_positives} 