"""
Unit tests for the PatternDiscovery module.
"""

import pytest
from app.core.adaptive.pattern_learner import PatternDiscovery, ValidatedPattern

@pytest.fixture
def pattern_discoverer():
    """Returns an instance of PatternDiscovery."""
    return PatternDiscovery(config={'min_confidence_to_validate': 0.8})

@pytest.fixture
def sample_corpus():
    """Provides a sample text corpus for validation."""
    return [
        "Contact me at john.doe@example.com for more details.",
        "My phone number is 123-456-7890.",
        "This is a regular sentence without any PII.",
        "Another email is jane.smith@work.net.",
        "The project code is ABC-123, not a phone number.",
        "Call me maybe? 123-456-7890 is the number.",
        "This is not an email: john.doe@.com"
    ]

@pytest.fixture
def confirmed_pii():
    """Provides a dictionary of confirmed PII samples and their categories."""
    return {
        "john.doe@example.com": "EMAIL",
        "123-456-7890": "PHONE_NUMBER",
        "jane.smith@work.net": "EMAIL"
    }

def test_initialization(pattern_discoverer):
    """Test that the PatternDiscovery class initializes correctly."""
    assert pattern_discoverer is not None
    assert pattern_discoverer.min_confidence_to_validate == 0.8

def test_discover_and_validate_good_pattern(pattern_discoverer, sample_corpus, confirmed_pii):
    """
    Test that a good, high-precision pattern is discovered and validated.
    '123-456-7890' is a unique string that is always PII in the corpus.
    """
    pii_to_discover = {"123-456-7890": "PHONE_NUMBER"}

    validated_patterns = pattern_discoverer.discover_and_validate_patterns(
        sample_corpus,
        pii_to_discover=pii_to_discover,
        ground_truth_pii=confirmed_pii
    )

    assert len(validated_patterns) == 1
    pattern = validated_patterns[0]
    assert pattern.precision == 1.0
    # Recall is how many of the PII types in the ground truth this one pattern found.
    # It found 1 (PHONE_NUMBER) of the 3 unique PII types in the ground truth.
    assert pattern.recall == 1/3

def test_discover_and_discard_ambiguous_pattern(pattern_discoverer, sample_corpus, confirmed_pii):
    """
    Test that an ambiguous pattern with low precision is correctly discarded.
    """
    # Let's try to discover a pattern for "work.net".
    pii_to_discover = {"work.net": "DOMAIN"}

    # The ground truth does NOT contain "work.net" as a known PII string.
    # Therefore, any match for "work.net" is a false positive.
    corpus = [
        "My email is jane.smith@work.net, a great domain.",
        "I love to work.net is my motto.",
    ]
    
    # discover_and_validate will create a regex for `\bwork\.net\b`.
    # It will find one unique match: "work.net".
    # This match is NOT in `confirmed_pii`, so it's a false positive.
    # Precision will be 0 / (0 + 1) = 0.0.
    validated_patterns = pattern_discoverer.discover_and_validate_patterns(
        corpus,
        pii_to_discover=pii_to_discover,
        ground_truth_pii=confirmed_pii
    )

    # 0.0 is less than the 0.8 threshold, so no pattern should be returned.
    assert len(validated_patterns) == 0

def test_empty_corpus_and_pii(pattern_discoverer):
    """Test that the method handles empty inputs gracefully."""
    patterns = pattern_discoverer.discover_and_validate_patterns([], {}, {})
    assert len(patterns) == 0

    patterns = pattern_discoverer.discover_and_validate_patterns(["some text"], {}, {})
    assert len(patterns) == 0

    patterns = pattern_discoverer.discover_and_validate_patterns([], {"pii": "CAT"}, {"pii": "CAT"})
    assert len(patterns) == 0

def test_internal_validate_regex(pattern_discoverer, sample_corpus, confirmed_pii):
    """Test the internal regex validation logic directly."""
    # Test a perfect regex
    good_regex = r"123-456-7890"
    results = pattern_discoverer._validate_regex(good_regex, sample_corpus, confirmed_pii)
    assert results['precision'] == 1.0
    # Recall is how many of the unique known PII we found with this one regex.
    assert results['recall'] == 1/3 # It finds 1 of the 3 unique PII types
    assert results['true_positives'] == 2
    assert results['false_positives'] == 0

    # Test a regex that has false positives
    bad_regex = r"\b\w{3}-\d{3}\b" # This will match "ABC-123" and "123-456"
    results = pattern_discoverer._validate_regex(bad_regex, sample_corpus, confirmed_pii)
    assert results['precision'] == 0.0 # No matches are in the ground truth
    assert results['recall'] == 0.0
    assert results['true_positives'] == 0
    # It uniquely matches '123-456' and 'ABC-123'
    assert results['false_positives'] == 3 # '123-456' occurs twice, 'ABC-123' once 