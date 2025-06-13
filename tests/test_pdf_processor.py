"""
Tests for the PDFProcessor service.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import fitz

from app.services.pdf_processor import PDFProcessor
from app.core.text_extraction import extract_text_enhanced
from app.core.config_manager import ConfigManager

@pytest.fixture(scope="module")
def sample_text():
    """Provides a sample text with PII."""
    return "Hello, my name is John Doe. You can reach me at john.doe@example.com or 555-1234. My SSN is 000-00-1234 and my credit card is 1234-5678-9012-3456. I live in New York."

@pytest.fixture(scope="module")
def pdf_processor():
    """Fixture for a PDFProcessor instance."""
    # Use a real processor to test the integration of its components
    return PDFProcessor()

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

@pytest.mark.unit
class TestPDFProcessorUnit:
    """Unit tests for individual methods of the PDFProcessor."""

    def test_find_personal_info(self, pdf_processor, sample_text_with_pii):
        """Test the main PII finding logic."""
        personal_info = pdf_processor.find_personal_info(sample_text_with_pii, language='en')

        # Check that major categories have been found (using new spaCy/custom keys)
        assert "PERSON" in personal_info
        assert "emails" in personal_info
        assert "lithuanian_personal_codes" in personal_info
        
        # Check for specific detected items
        assert any("Johnathan Doe" in item[0] for item in personal_info["PERSON"])
        assert any("john.doe@email.com" in item[0] for item in personal_info["emails"])
        assert any("38801011234" in item[0] for item in personal_info["lithuanian_personal_codes"])

    def test_deduplication(self, pdf_processor):
        """Test the deduplication logic."""
        # This test should be updated to reflect `deduplicate_with_confidence`
        # For now, we test the public-facing `find_personal_info` which uses it
        text = "Call John Smith. John Smith is the manager. John Smith's email is j.smith@example.com."
        personal_info = pdf_processor.find_personal_info(text, language='en')
        
        # "John Smith" should only appear once after deduplication
        name_detections = [item[0] for item in personal_info.get("PERSON", [])]
        assert name_detections.count("John Smith") <= 1


@pytest.mark.integration
class TestPDFProcessorIntegration:
    """Integration tests for the PDFProcessor with external dependencies."""

    def test_anonymize_pdf_flow(self, pdf_processor, tmp_path):
        """Test the full PDF processing and anonymization flow."""
        # Create a dummy PDF for testing
        input_pdf_path = tmp_path / "input.pdf"
        output_pdf_path = tmp_path / "anonymized.pdf"
        
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This is a test PDF with the name Vardenis Pavardenis.")
        doc.save(str(input_pdf_path))
        doc.close()

        # Run the anonymization
        success, report = pdf_processor.anonymize_pdf(input_pdf_path, output_pdf_path)

        assert success
        assert output_pdf_path.exists()
        assert "PERSON" in report["details"]
        assert len(report["details"]["PERSON"]) > 0

        # Verify the output PDF is redacted
        anonymized_text = extract_text_enhanced(str(output_pdf_path))
        assert "Vardenis Pavardenis" not in anonymized_text
        assert "[REDACTED]" in anonymized_text

    def test_generate_redaction_report(self, pdf_processor, sample_text):
        """Test generating a redaction report."""
        personal_info = pdf_processor.find_personal_info(sample_text, language="en")
        report = pdf_processor.generate_redaction_report(personal_info, "en")

        # Verify the report structure and content safely.
        assert "total_redactions" in report
        assert "categories" in report
        assert report["total_redactions"] > 0
        
        categories = report.get("categories", {})
        assert "PERSON" in categories
        assert "EMAILS" in categories
        assert "PHONES" in categories
        assert "SSNS" in categories
        assert "CREDIT_CARDS" in categories

    @pytest.mark.asyncio
    @patch('app.services.pdf_processor.extract_text_enhanced')
    async def test_process_pdf_failure_on_invalid_content(self, mock_extract_text, pdf_processor, tmp_path):
        """Test PDF processing failure on invalid content."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("invalid content")

        # Simulate the specific logger error from the traceback.
        mock_extract_text.side_effect = TypeError("Logger._log() got an unexpected keyword argument 'error'")

        result = await pdf_processor.process_pdf(str(pdf_path))

        # Check for failure status and the specific error message.
        assert result.get("status") == "error"
        assert "Logger._log() got an unexpected keyword argument 'error'" in result.get("error", ""), \
            "Error message should indicate the specific TypeError"
            
    def test_deduplicate_with_confidence(self, pdf_processor):
        """Test that deduplication prefers the highest confidence detection."""
        from app.core.context_analyzer import DetectionContext

        # Low confidence 'name' detection
        low_conf_name = DetectionContext(text="123 Corp", category="PERSON", start=0, end=8, confidence=0.4)
        
        # High confidence 'organization' detection for the same text
        high_conf_org = DetectionContext(text="123 Corp", category="organizations", start=0, end=8, confidence=0.9)

        # Initial dict has the low confidence detection
        initial_pii = {"PERSON": [("123 Corp", "CONF_0.40")], "organizations": []}
        
        # Run deduplication
        final_pii = pdf_processor.deduplicate_with_confidence(
            initial_pii, [low_conf_name, high_conf_org]
        )

        # The final result should only contain the high-confidence organization
        assert "PERSON" not in final_pii
        assert "organizations" in final_pii
        assert len(final_pii["organizations"]) == 1
        assert final_pii["organizations"][0][0] == "123 Corp"
