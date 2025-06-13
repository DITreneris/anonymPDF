"""
Tests for Morning Session 5 improvements: Enhanced regex patterns and anti-overredaction logic.
Based on analysis in morning_ses5.md
"""

import pytest
import re
from app.core.config_manager import ConfigManager
from app.services.pdf_processor import PDFProcessor
from pathlib import Path


@pytest.fixture(scope="module")
def lithuanian_invoice_text():
    """Provides a sample Lithuanian invoice text."""
    return """
        Draudėjas: STANIULIS TOMAS
        asmens/įmonės kodas: 38901234567
        El. paštas: tomas.s@email.com
        Adresas: P. Vileišio g. 11-4, Vilnius
        Automobilio duomenys:
        Valst. Nr.: HRV249
    """

@pytest.fixture(scope="module")
def lithuanian_technical_text():
    """Provides a sample Lithuanian technical document text."""
    return """
        Sutartį sudarė Vardenis Pavardenis, asmens kodas 38901234567.
        Techniniai duomenys:
        Variklio numeris: 12345678901
        Galia: 150 kW
        Svoris: 1500 kg
    """


class TestMorningSes5Patterns:
    """Test the new patterns added in Morning Session 5."""

    @pytest.fixture
    def config_manager(self):
        """Fixture for ConfigManager."""
        return ConfigManager()

    @pytest.fixture
    def pdf_processor(self):
        """Fixture for PDFProcessor."""
        return PDFProcessor()

    def test_lithuanian_name_all_caps_pattern(self, config_manager):
        """Test ALL CAPS Lithuanian name detection."""
        pattern = config_manager.patterns["lithuanian_name_all_caps"]
        assert re.search(pattern, "STANIULIS TOMAS") is not None

    def test_lithuanian_name_contextual_pattern(self, config_manager):
        """Test Lithuanian name detection with context keywords."""
        pattern = config_manager.patterns["lithuanian_name_contextual"]
        assert re.search(pattern, "Draudėjas: Jonas Petraitis") is not None

    def test_lithuanian_address_flexible_pattern(self, config_manager):
        """Test flexible address detection."""
        pattern = config_manager.patterns["lithuanian_address_flexible"]
        assert re.search(pattern, "Vileišio g. 11-4") is not None

    def test_lithuanian_personal_code_contextual_pattern(self, config_manager):
        """Test personal code with explicit context."""
        pattern = config_manager.patterns["lithuanian_personal_code_contextual"]
        match = re.search(pattern, "asmens kodas: 38901234567")
        assert match is not None
        assert match.group(1) == "38901234567"


class TestMorningSes5AntiOverredaction:
    """Test anti-overredaction logic from Morning Session 5."""

    @pytest.fixture
    def pdf_processor(self):
        """Fixture for PDFProcessor with test config."""
        processor = PDFProcessor()
        processor.config_manager.settings['anti_overredaction'] = {
            'technical_terms_whitelist': ['kW', 'Nm', 'g/km', 'CO2 emisijos', 'kg'],
            'technical_sections': ['SVORIS', 'VAŽ.', 'Techniniai duomenys'],
            'pii_field_labels': ['Vardas', 'Asmens kodas', 'Draudėjas']
        }
        return processor

    def test_no_preserve_pii_in_technical_sections(self, pdf_processor):
        """Test that PII fields are still redacted even in technical sections."""
        assert not pdf_processor.should_preserve_detection(
            "38901234567", "lithuanian_personal_code", "SVORIS: Asmens kodas: 38901234567"
        )


class TestMorningSes5Integration:
    """Integration tests for morning session 5 improvements."""

    @pytest.fixture
    def pdf_processor(self):
        """Fixture for PDFProcessor."""
        return PDFProcessor()

    def test_comprehensive_lithuanian_document_processing(self, pdf_processor, lithuanian_invoice_text):
        """Test comprehensive Lithuanian document processing."""
        personal_info = pdf_processor.find_personal_info(lithuanian_invoice_text, language="lt")

        # The 'names' key may not exist if no names are found.
        detected_names = personal_info.get("names", [])
        assert len(detected_names) > 0, "Should detect at least one name"

        assert len(personal_info.get("lithuanian_personal_codes", [])) > 0
        assert len(personal_info.get("locations", [])) > 0
        assert len(personal_info.get("emails", [])) > 0

    def test_contextual_validation_of_technical_terms(self, pdf_processor):
        """Test that common technical terms are not flagged as names."""
        text = "This is a test of Python code. Contact author@example.com"
        personal_info = pdf_processor.find_personal_info(text, language="en")

        # 'Python' should not be a name. The 'names' key might be absent altogether.
        assert "names" not in personal_info, "Should not detect 'Python' as a name"

    def test_anti_overredaction_in_technical_context(self, pdf_processor, lithuanian_technical_text):
        """Test that PII is correctly identified and redacted within technical documents."""
        personal_info = pdf_processor.find_personal_info(lithuanian_technical_text, language="lt")

        # Check that the name and personal code are detected
        detections = personal_info.get("names", [])
        assert len(detections) > 0, "Expected names to be detected"
        assert "Vardenis Pavardenis" in [d[0] for d in detections], "The specific name was not found"

        detected_codes = personal_info.get("lithuanian_personal_codes", [])
        assert len(detected_codes) >= 1, "Should detect personal code"
        assert any("38901234567" in code for code, conf in detected_codes)

        # Ensure that non-PII numbers are NOT detected
        assert "eleven_digit_numerics" not in personal_info, "Should not detect non-PII numbers"

    def test_anti_overredaction_of_common_words(self, pdf_processor):
        """Test that common document words are not redacted as names."""
        text = "This document is a summary of the Certificate and the Agreement."
        personal_info = pdf_processor.find_personal_info(text, language="en")
        
        # These words are in the exclusion list and should not be detected as names.
        name_detections = personal_info.get("names", [])
        detected_texts = {d[0].lower() for d in name_detections}
        
        assert "summary" not in detected_texts
        assert "certificate" not in detected_texts
        assert "agreement" not in detected_texts

    def test_redaction_report_generation(self, pdf_processor):
        """Test that the redaction report is generated correctly."""
        text = "Contact Jonas Petraitis at j.p@email.com for details."
        personal_info = pdf_processor.find_personal_info(text, language="lt")

        report = pdf_processor.generate_redaction_report(personal_info, "lt")

        assert "names" in report
        assert "emails" in report
        assert report["names"][0] == "Jonas Petraitis"
        assert report["emails"][0] == "j.p@email.com"
        assert report["summary"]["total_redactions"] >= 2
        assert report["summary"]["language"] == "lt"


if __name__ == "__main__":
    pytest.main([__file__]) 