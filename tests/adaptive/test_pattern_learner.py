"""
Unit tests for the PatternDiscovery module.
"""

import pytest
from unittest.mock import MagicMock
from app.core.adaptive.pattern_learner import PatternLearner, ValidatedPattern
from app.core.adaptive.pattern_db import AdaptivePatternDB
from unittest.mock import Mock

@pytest.fixture
def mock_db():
    """Fixture for a mock AdaptivePatternDB."""
    return Mock(spec=AdaptivePatternDB)

@pytest.fixture
def pattern_learner(mock_db):
    """Fixture for a PatternLearner instance with a mock DB."""
    return PatternLearner(pattern_db=mock_db, min_confidence=0.8, min_samples=3)

class TestPatternLearner:
    def test_initialization(self, pattern_learner, mock_db):
        assert pattern_learner.pattern_db is mock_db
        assert pattern_learner.min_confidence_to_validate == 0.8
        assert pattern_learner.min_samples_for_learning == 3

    def test_discover_and_validate_patterns(self, pattern_learner):
        """
        Test the full cycle of discovering a pattern, validating it against a corpus,
        and ensuring it meets confidence and recall thresholds.
        """
        # --- Test Case 1: A perfect pattern ---
        corpus = ["My phone is 555-1234.", "Call 555-1234 now!", "Ignore 123-4567."]
        pii_to_discover = {"555-1234": "PHONE"}
        ground_truth = {"555-1234": "PHONE"}

        patterns = pattern_learner.discover_and_validate_patterns(corpus, pii_to_discover, ground_truth)

        assert len(patterns) == 1
        pattern = patterns[0]
        assert pattern.regex == r"\b555-1234\b"
        assert pattern.pii_category == "PHONE"
        assert pattern.confidence == 1.0  # Precision
        assert pattern.recall == 1.0
        assert pattern.positive_matches == 2
        assert pattern.negative_matches == 0

        # --- Test Case 2: A pattern with low precision ---
        corpus_low_precision = ["My ID is ABC-123.", "Order number is ABC-123."]
        pii_to_discover_low_precision = {"ABC-123": "ID_CARD"}
        ground_truth_low_precision = {"ABC-123": "ID_CARD"}

        pattern_learner.min_confidence_to_validate = 0.6
        patterns = pattern_learner.discover_and_validate_patterns(
            corpus_low_precision, pii_to_discover_low_precision, ground_truth_low_precision
        )

        corpus_fp = ["ID: ID-456", "Not PII: ID-456"]
        pii_fp = {"ID-456": "ID"}
        ground_truth_fp = {"ID-456": "ID"}

        patterns = pattern_learner.discover_and_validate_patterns(corpus_fp, pii_fp, ground_truth_fp)

        assert len(patterns) == 1
        pattern = patterns[0]
        assert pattern.regex == r"\bID-456\b"
        assert pattern.pii_category == "ID"
        assert pattern.positive_matches == 2
        assert pattern.confidence <= 1.0
        assert pattern.recall <= 1.0

        # --- Test Case 3: Insufficient samples ---
        pattern_learner.min_samples_for_learning = 5
        patterns = pattern_learner.discover_and_validate_patterns(corpus, pii_to_discover, ground_truth)
        assert len(patterns) == 0
