import pytest
import re
from datetime import datetime

from app.core.adaptive.pattern_learner import PatternLearner
from app.core.adaptive.pattern_db import AdaptivePatternDB, AdaptivePattern

@pytest.fixture
def mock_db():
    """Fixture for a mock AdaptivePatternDB."""
    return pytest.MonkeyPatch().setenv  # placeholder, DB isn't used directly in discovery

@pytest.fixture
def pattern_learner(mock_db):
    """Fixture for a PatternLearner instance with a mock DB."""
    # The mock_db isn't used directly in tests since learner returns patterns without DB calls
    return PatternLearner(pattern_db=mock_db, min_confidence=0.8)

class TestValidateRegex:
    def test_perfect_match(self, pattern_learner):
        regex = r"\b555\-1234\b"
        corpus = [
            "Contact: 555-1234",
            "My number 555-1234"
        ]
        target_pii = "555-1234"
        known = {"555-1234": "PHONE"}

        results = pattern_learner._validate_regex(regex, corpus, target_pii, known)
        assert results["true_positives"] == 2
        assert results["false_positives"] == 0
        assert results["precision"] == pytest.approx(1.0)
        assert results["recall"] == pytest.approx(1.0)

    def test_false_positives(self, pattern_learner):
        regex = r"ID\-\d{3}"
        corpus = ["ID-456 is valid", "Ref ID-456", "False match: ID-999"]
        target_pii = "ID-456"
        known = {"ID-456": "ID"}

        results = pattern_learner._validate_regex(regex, corpus, target_pii, known)
        # Matches 'ID-456' twice (TP) and 'ID-999' once (FP)
        assert results["true_positives"] == 2
        assert results["false_positives"] == 1
        assert results["precision"] == pytest.approx(2/3)
        # FN is zero because TP > 0
        assert results["recall"] == pytest.approx(1.0)

    def test_false_negative(self, pattern_learner):
        regex = r"\b1234\b"
        corpus = ["No match here"]
        target_pii = "1234"
        known = {"1234": "NUM"}

        results = pattern_learner._validate_regex(regex, corpus, target_pii, known)
        assert results["true_positives"] == 0
        assert results["false_positives"] == 0
        # Since target appears nowhere, recall should be 0.0
        assert results["recall"] == pytest.approx(0.0)

class TestDiscoverAndValidatePatterns:
    def test_successful_discovery(self, pattern_learner):
        corpus = [
            "Call me at 555-1234.",
            "Her number is 555-1234.",
            "Office: 555-1234.",
            "Ignore 000-0000."
        ]
        pii_to_discover = {"555-1234": "PHONE"}
        ground_truth = {"555-1234": "PHONE"}

        patterns = pattern_learner.discover_and_validate_patterns(corpus, pii_to_discover, ground_truth)
        assert len(patterns) == 1
        pat = patterns[0]
        assert isinstance(pat, AdaptivePattern)
        assert pat.regex == r"\b555\-1234\b"
        assert pat.pii_category == "PHONE"
        assert pat.positive_matches == 3
        assert pat.negative_matches == 0
        assert pat.confidence == pytest.approx(1.0)
        assert pat.recall == pytest.approx(1.0)

    def test_low_precision_filtered(self, pattern_learner):
        corpus = ["A: ABC-123", "B: XYZ-789"]
        pii_to_discover = {"ABC-123": "CODE"}
        ground_truth = {"ABC-123": "CODE"}

        # For this corpus, one true positive and zero false positives
        pattern_learner.min_confidence = 0.9
        patterns = pattern_learner.discover_and_validate_patterns(
            corpus, pii_to_discover, ground_truth, min_samples_for_learning=3
        )
        # Precision = 1.0 but true positives = 1 < min_samples(3)
        assert patterns == []

    def test_insufficient_samples(self, pattern_learner):
        corpus = ["X: 999-0000", "Y: 999-0000"]
        pii_to_discover = {"999-0000": "PHONE"}
        ground_truth = {"999-0000": "PHONE"}

        # Only 2 samples but min_samples is 3
        patterns = pattern_learner.discover_and_validate_patterns(
            corpus, pii_to_discover, ground_truth, min_samples_for_learning=3
        )
        assert patterns == []

    def test_recall_threshold(self, pattern_learner):
        corpus = ["See PII: ABC-111"]
        pii_to_discover = {"ABC-111": "ID"}
        ground_truth = {"ABC-111": "ID"}

        # TP=1, FN=0 => recall=1.0 but TP<min_samples
        pattern_learner.min_samples = 1
        patterns = pattern_learner.discover_and_validate_patterns(corpus, pii_to_discover, ground_truth)
        # Even though TP>=min_samples, recall ok, precision ok, so pattern created
        assert len(patterns) == 1
        
        # Test case for when min_samples is NOT met
        patterns_not_enough_samples = pattern_learner.discover_and_validate_patterns(
            corpus, pii_to_discover, ground_truth, min_samples_for_learning=2
        )
        assert len(patterns_not_enough_samples) == 0

