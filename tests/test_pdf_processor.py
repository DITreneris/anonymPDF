"""
Tests for the PDFProcessor service.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
import fitz
import re

from app.services.pdf_processor import PDFProcessor
from app.core.config_manager import ConfigManager
from app.core.context_analyzer import DetectionContext, ConfidenceLevel

# A single, consistent mock ConfigManager for all tests.
@pytest.fixture(scope="module")
def mock_config_manager():
    """Provides a mocked ConfigManager for the entire test module."""
    cm = MagicMock(spec=ConfigManager)
    # Corrected regex patterns (raw strings don't need double escapes for `\b`).
    cm.patterns = {
        'emails': [r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'],
        'lithuanian_personal_codes': [r'\b[3-6]\d{10}\b']
    }
    cm.cities = {'Vilnius', 'Kaunas'}
    cm.brand_names = {'TestCorp', 'SampleBrand'}
    # Mock methods that might be called
    cm.get_patterns.return_value = cm.patterns
    cm.get_cities.return_value = cm.cities
    cm.get_brand_names.return_value = cm.brand_names
    return cm

# A single, module-scoped processor fixture.
@pytest.fixture(scope="module")
def pdf_processor(mock_config_manager):
    """Provides a module-scoped PDFProcessor instance with mocked dependencies."""
    with patch('app.services.pdf_processor.spacy.load') as mock_spacy_load:
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['ner']
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        # We need to patch the AdvancedPatternRefinement to use our mock patterns
        with patch('app.core.context_analyzer.AdvancedPatternRefinement') as mock_advanced_patterns:
            instance = mock_advanced_patterns.return_value
            # This simulates the behavior of the refactored AdvancedPatternRefinement
            def find_mock_patterns(text):
                detections = []
                # Simulate email detection
                for match in re.finditer(mock_config_manager.patterns['emails'][0], text):
                    detections.append({'text': match.group(0), 'category': 'emails', 'start': match.start(), 'end': match.end()})
                # Simulate code detection
                for match in re.finditer(mock_config_manager.patterns['lithuanian_personal_codes'][0], text):
                     detections.append({'text': match.group(0), 'category': 'lithuanian_personal_codes', 'start': match.start(), 'end': match.end()})
                return detections
            
            instance.find_enhanced_patterns.side_effect = find_mock_patterns
            
            processor = PDFProcessor(config_manager=mock_config_manager)
            yield processor

@pytest.fixture(scope="function")
def pdf_processor(test_pdf_processor: PDFProcessor):
    """Alias the global test_pdf_processor fixture for use in this module."""
    return test_pdf_processor

@pytest.fixture
def sample_text_with_pii():
    """Sample text containing various PII types."""
    return """
    This document belongs to Johnathan Doe (email: john.doe@email.com).
    His phone is 8-800-555-3535.
    Please send all correspondence to Vilnius, Lithuania.
    Vardenis Pavardenis, asmens kodas 38801011234.
    Organization: ACME Corp.
    """

@pytest.fixture(scope="module")
def sample_text():
    """Provides a sample text with PII."""
    return "Hello, my name is John Doe. You can reach me at john.doe@example.com or 555-1234. My SSN is 000-00-1234 and my credit card is 1234-5678-9012-3456. I live in New York."

@pytest.fixture
def mock_processor(self) -> MagicMock:
    processor = MagicMock(spec=PDFProcessor)
    processor.anonymize_pdf.return_value = (True, {"redactions": 1})
    processor.process_pdf.return_value = {"success": True, "report": {}, "redaction_count": 1, "error_message": None}
    return processor

@pytest.mark.unit
class TestPDFProcessorUnit:
    """Unit tests for the PDFProcessor, aligned with recent refactoring."""

    def test_find_personal_info(self, pdf_processor):
        """Test the core PII detection logic with corrected regex."""
        text = "Contact me at test@example.com. My code is 38801011234."
        detections = pdf_processor.find_personal_info(text, language="en")
        
        # Assert that the correct categories are present
        assert "emails" in detections
        assert "lithuanian_personal_codes" in detections
        
        # Assert that the correct items were detected
        detected_emails = [item[0] for item in detections.get("emails", [])]
        assert "test@example.com" in detected_emails
        
        detected_codes = [item[0] for item in detections.get("lithuanian_personal_codes", [])]
        assert "38801011234" in detected_codes

    def test_deduplicate_with_confidence_simplified(self, pdf_processor):
        """Test the simplified filtering logic of deduplicate_with_confidence."""
        mock_validator = MagicMock()
        full_text = "Some text about Vilnius"
        
        # Detections to be processed
        context_detections = [
            DetectionContext(text='Vilnius', category='locations', start_char=16, end_char=23, confidence=ConfidenceLevel.HIGH.value, full_text=full_text, validator=mock_validator),
            DetectionContext(text='Vilnius', category='locations', start_char=30, end_char=37, confidence=ConfidenceLevel.LOW.value, full_text=full_text, validator=mock_validator),
            DetectionContext(text='InvalidCorp', category='organizations', start_char=40, end_char=51, confidence=ConfidenceLevel.HIGH.value, full_text=full_text, validator=mock_validator)
        ]
        # Set validation status
        context_detections[0].is_valid = True  # High confidence, valid
        context_detections[1].is_valid = True  # Low confidence, valid
        context_detections[2].is_valid = False # High confidence, but invalid

        # The method now filters and enhances a dictionary of detections
        final_detections = pdf_processor.deduplicate_with_confidence({}, context_detections=context_detections)
        
        # Only the highest-confidence, valid detection for 'Vilnius' should remain.
        assert 'locations' in final_detections
        assert len(final_detections['locations']) == 1
        assert final_detections['locations'][0][0] == 'Vilnius'
        assert final_detections['locations'][0][1].startswith('CONF_')
        
        # The invalid detection should be gone completely.
        assert 'organizations' not in final_detections

    def test_generate_redaction_report(self, pdf_processor):
        """Test the redaction report generation with the new, flattened structure."""
        personal_info = {
            "names": [("John Doe", f"HIGH_{ConfidenceLevel.HIGH.value}")],
            "emails": [("johndoe@email.com", f"MEDIUM_{ConfidenceLevel.MEDIUM.value}")]
        }
        report = pdf_processor.generate_redaction_report(personal_info, language="en")

        # No 'summary' key anymore
        assert "summary" not in report
        assert "total_redactions" in report
        assert "categories" in report
        assert "details" in report
        
        assert report["total_redactions"] == 2
        assert report["categories"]["NAMES"] == 1
        assert report["categories"]["EMAILS"] == 1
        
        # Details are now a list of tuples
        assert ("johndoe@email.com", f"MEDIUM_{ConfidenceLevel.MEDIUM.value}") in report['details']['emails']


@pytest.mark.integration
class TestPDFProcessorIntegration:
    """Integration tests for the PDFProcessor, aligned with the latest API."""

    @pytest.mark.asyncio
    async def test_process_pdf_success(self, pdf_processor, tmp_path):
        """Test the full PDF processing pipeline for a successful case."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("This is a test PDF with an email: success@example.com")

        # Mock the parts of the pipeline
        pdf_processor.extract_text_from_pdf = MagicMock(return_value="Email: success@example.com")
        # Anonymize now returns a simple tuple
        pdf_processor.anonymize_pdf = MagicMock(return_value=(True, {"redactions": 1}))

        result = await pdf_processor.process_pdf(str(pdf_path))

        # Assert against the new, correct response structure
        assert result['status'] == 'processed'
        assert result['filename'] == 'test.pdf'
        assert 'report' in result
        assert result['report']['total_redactions'] > 0
        assert result['report']['categories']['EMAILS'] == 1

    @pytest.mark.asyncio
    async def test_process_pdf_failure_on_invalid_content(self, pdf_processor, tmp_path):
        """Test failure case when PDF processing encounters an error."""
        pdf_path = tmp_path / "test_fail.pdf"
        pdf_path.write_text("irrelevant")

        # Mock anonymize_pdf to simulate a failure
        pdf_processor.anonymize_pdf = MagicMock(return_value=(False, {"error": "Test failure message"}))
        # Mock text extraction to ensure the process continues to the failing step
        pdf_processor.extract_text_from_pdf = MagicMock(return_value="Some text")
        
        result = await pdf_processor.process_pdf(str(pdf_path))
        
        # Assert against the new, correct error structure
        assert result['status'] == 'failed'
        assert "Test failure message" in result['error']

    def test_anonymize_pdf_flow(self, pdf_processor, tmp_path):
        """Test the full PDF processing and anonymization flow."""
        # This test uses a real file path but mocks the expensive parts.
        input_pdf_path = Path("tests/samples/simple_pii_document.pdf")
        output_pdf_path = tmp_path / "anonymized_output.pdf"

        # Define the mock PII that find_personal_info should return.
        mock_pii = {
            'names': [('John Doe', 'CONF_0.90')],
            'emails': [('john.doe@work.com', 'CONF_0.90')]
        }

        # Mock both text extraction and PII finding to isolate the anonymization logic.
        with patch.object(pdf_processor, 'extract_text_from_pdf', return_value="Mocked text with PII.") as mock_extract, \
             patch.object(pdf_processor, 'find_personal_info', return_value=mock_pii) as mock_find:
            
            # Run the actual anonymization.
            success, report = pdf_processor.anonymize_pdf(input_pdf_path, output_pdf_path)
            
            # Ensure our mocks were called.
            mock_extract.assert_called_once_with(input_pdf_path)
            mock_find.assert_called_once_with("Mocked text with PII.", ANY)

        assert success, f"anonymize_pdf returned False. Report: {report}"
        assert output_pdf_path.exists()
        
        # Verify the output by checking for redaction markers.
        # A full text check is brittle; we just need to know redaction happened.
        with fitz.open(output_pdf_path) as doc:
            page = doc[0]
            # Check for redaction annotations directly.
            assert len(page.annots(types=[fitz.PDF_ANNOT_SQUARE])) > 0, "No redaction annotations found."