"""
Unit tests for the DocumentClassifier module.
"""

import pytest
from pathlib import Path

from app.core.adaptive.doc_classifier import DocumentClassifier, DocumentClassification
from app.core.adaptive.processing_rules import ProcessingRuleManager

@pytest.fixture
def mock_rules_config_content():
    return """
document_types:
  - doc_type: invoice
    keywords: ["invoice", "payment due", "total amount"]
    rules: []
  - doc_type: medical
    keywords: ["patient", "doctor", "diagnosis", "prescription"]
    rules: []
  - doc_type: general_text
    keywords: []
    rules: []
"""

@pytest.fixture
def classifier(tmp_path: Path, mock_rules_config_content: str) -> DocumentClassifier:
    """Provides a DocumentClassifier instance with a mocked config."""
    config_file = tmp_path / "test_rules.yaml"
    config_file.write_text(mock_rules_config_content)
    
    config = {
        'rules_config_path': str(config_file),
        'default_doc_type': 'general_text',
        'classification_threshold': 0.2
    }
    return DocumentClassifier(config=config)

def test_classify_invoice_document(classifier: DocumentClassifier):
    """Tests correct classification of an invoice document."""
    text = "This is an invoice for a recent purchase. The total amount is $50. Payment due soon."
    result = classifier.classify_document(text)
    assert isinstance(result, DocumentClassification)
    assert result.doc_type == "invoice"
    assert result.confidence == 3 / 3  # 3 of 3 keywords matched

def test_classify_medical_document(classifier: DocumentClassifier):
    """Tests correct classification of a medical document."""
    text = "The doctor saw the patient and provided a diagnosis."
    result = classifier.classify_document(text)
    assert result.doc_type == "medical"
    assert result.confidence == 3 / 4 # 3 of 4 keywords matched

def test_fallback_to_general_text_no_keywords(classifier: DocumentClassifier):
    """Tests fallback to the default type when no keywords are matched."""
    text = "This is a simple note about a meeting."
    result = classifier.classify_document(text)
    assert result.doc_type == "general_text"
    assert result.confidence == 0.1 # Default confidence for no match

def test_classification_confidence_scoring(classifier: DocumentClassifier):
    """Tests the confidence scoring logic."""
    text = "This invoice has a total amount to be paid." # 2 of 3 invoice keywords
    result = classifier.classify_document(text)
    assert result.doc_type == "invoice"
    assert pytest.approx(result.confidence) == 2 / 3

def test_classification_below_threshold(classifier: DocumentClassifier):
    """Tests that the classifier falls back if confidence is below threshold."""
    # This text only has one of the four medical keywords, so confidence will be 0.25.
    # We set the threshold in the fixture to 0.4 for this test.
    classifier.classification_threshold = 0.4
    text = "A patient was here."
    result = classifier.classify_document(text)
    assert result.doc_type == "general_text" # Should fall back
    assert result.confidence == 1 / 4 # Confidence is calculated but not used for selection

def test_get_processing_rules(classifier: DocumentClassifier):
    """Tests that the classifier can retrieve rules via the rule manager."""
    ruleset = classifier.get_processing_rules("invoice")
    assert ruleset is not None
    assert ruleset.doc_type == "invoice"

def test_classify_generic_document(classifier: DocumentClassifier):
    """Test that text without specific keywords is classified as general."""
    text = "This is a document about something interesting."
    result = classifier.classify_document(text)
    assert result.doc_type == "general_text"

def test_get_rules_for_unknown_type(classifier: DocumentClassifier):
    """Test that retrieving rules for an unknown doc type returns None."""
    ruleset = classifier.get_processing_rules("unknown_document_type")
    assert ruleset is None 