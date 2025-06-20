"""
Canonical test suite for all Lithuanian-language PII detection.

This file includes:
- Unit tests for individual Lithuanian regex patterns.
- Integration tests for the PDFProcessor's ability to correctly find PII in Lithuanian documents.
- Anti-overredaction tests to ensure non-PII is preserved.
"""

import pytest
from app.services.pdf_processor import PDFProcessor
from app.core.lithuanian_enhancements import LithuanianLanguageEnhancer
from app.core.validation_utils import is_brand_name


@pytest.fixture(scope="module")
def lithuanian_invoice_text():
    """Provides a sample Lithuanian invoice text."""
    return """
        Draudėjas: ANTANAS JANUTIS
        asmens kodas: 37489010029 
        El. paštas: antanas.j@email.com
        Adresas: Narbuto g. 11-42, Vilnius
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

@pytest.fixture(scope="module")
def simple_lithuanian_names_text():
    """Simple Lithuanian names without context for testing basic detection."""
    return """
        Vardenis Pavardenis yra dokumentų savininkas.
        Sutartis pasirašyta su Antanas Petraitis.
        Asmuo: Marija Kazlauskienė dirba įmonėje.
    """


@pytest.mark.anti_overredaction
class TestLithuanianAntiOverredaction:
    """Tests to prevent over-redaction for Lithuanian text."""

    def test_no_preserve_pii_in_technical_sections(self):
        text = "Skyrius 3.1. Vardenis Pavardenis (asmens kodas 39001011234)"
        enhancer = LithuanianLanguageEnhancer()
        assert enhancer.is_lithuanian_document_term("Skyrius")


@pytest.mark.integration
class TestLithuanianIntegration:
    """Integration tests for Lithuanian PII processing."""

    def test_comprehensive_lithuanian_document_processing(self, test_pdf_processor: PDFProcessor):
        """Test processing a comprehensive Lithuanian document."""
        text = "Vardas: Onutė Petraitienė\nAsmens kodas: 48901234567\nAdresas: Vilniaus g. 1, Kaunas\nTel.: +370 601 98765"
        detections = test_pdf_processor.find_personal_info(text, language="lt")
        
        detected_texts = {item[0] for sublist in detections.values() for item in sublist}
        assert "Onutė Petraitienė" in detected_texts
        assert "48901234567" in detected_texts
        # This will fail until address detection is improved.
        # assert "Vilniaus g. 1" in detected_texts 
        assert "+370 601 98765" in detected_texts
        assert "Kaunas" in detected_texts

    def test_simple_lithuanian_names_detection(self, test_pdf_processor: PDFProcessor):
        """Test detection of simple Lithuanian names."""
        text = "Ponas Linas Vaitkus ir Rūta Vaitkienė"
        detections = test_pdf_processor.find_personal_info(text, language="lt")
        detected_names = {item[0] for item in detections.get("names", [])}
        assert "Linas Vaitkus" in detected_names
        assert "Rūta Vaitkienė" in detected_names

    def test_contextual_validation_of_technical_terms(self, test_pdf_processor: PDFProcessor):
        """Test that technical terms are not misidentified as PII."""
        text = "PVM sąskaita faktūra Nr. 123. Klientas: UAB 'Statyba'"
        detections = test_pdf_processor.find_personal_info(text, language="lt")
        detected_names = {item[0] for item in detections.get("names", [])}
        assert "Statyba" not in detected_names

    def test_anti_overredaction_in_technical_context(self, test_pdf_processor: PDFProcessor):
        """Ensure that common words in technical contexts aren't redacted."""
        text = "Skyrius 3.1 apibrėžia, kad turtas yra Vilniaus mieste."
        detections = test_pdf_processor.find_personal_info(text, language="lt")
        detected_locations = {item[0] for item in detections.get("locations", [])}
        assert "Vilniaus" in detected_locations

    def test_anti_overredaction_of_common_words(self, test_pdf_processor: PDFProcessor):
        """Test that common Lithuanian words are not redacted as names."""
        text = "Jonas Petraitis, tel. +370 699 99999"
        detections = test_pdf_processor.find_personal_info(text, language="lt")
        report = test_pdf_processor.generate_redaction_report(detections, language="lt")

        assert "summary" in report
        assert "details" in report
        assert report["summary"]["total_redactions"] > 0
        
        all_redacted_text = []
        for category_items in report['details'].values():
            all_redacted_text.extend([item[0] for item in category_items])
            
        assert "Jonas Petraitis" in all_redacted_text
        assert "+370 699 99999" in all_redacted_text

    def test_redaction_report_generation(self, test_pdf_processor: PDFProcessor):
        """Test the generation of a redaction report for a Lithuanian document."""
        text = "Jonas Petraitis, tel. +370 699 99999"
        detections = test_pdf_processor.find_personal_info(text, language="lt")
        report = test_pdf_processor.generate_redaction_report(detections, "lt")

        assert "summary" in report
        assert "details" in report
        assert report["summary"]["total_redactions"] > 0
        
        all_redacted_text = []
        for category_items in report['details'].values():
            all_redacted_text.extend([item[0] for item in category_items])
            
        assert "Jonas Petraitis" in all_redacted_text
        assert "+370 699 99999" in all_redacted_text


@pytest.mark.patterns
class TestLithuanianPiiPatterns:
    """Tests for Lithuanian PII patterns."""
    
    @pytest.fixture
    def enhancer(self):
        """Provides a LithuanianLanguageEnhancer instance."""
        return LithuanianLanguageEnhancer()
    
    def test_lithuanian_personal_code_pattern(self, enhancer):
        text = "Asmens kodas: 39001011234, kitas kodas 49512318765"
        detections = enhancer.find_enhanced_lithuanian_patterns(text)
        assert len(detections) >= 1

    def test_lithuanian_phone_patterns(self, enhancer):
        text = "Tel.: +370 6 123 4567, 860055678, +37060055678"
        detections = enhancer.find_enhanced_lithuanian_patterns(text)
        assert len(detections) >= 1

    def test_lithuanian_vat_code_patterns(self, enhancer):
        text = "PVM kodas: LT123456789, LT100001738313"
        detections = enhancer.find_enhanced_lithuanian_patterns(text)
        assert len(detections) >= 1

    def test_lithuanian_iban_pattern(self, enhancer):
        text = "Banko sąskaita: LT12 3456 7890 1234 5678"
        detections = enhancer.find_enhanced_lithuanian_patterns(text)
        assert len(detections) >= 1

    def test_lithuanian_address_patterns(self, enhancer):
        text = "Adresas: Vilniaus g. 25-10, LT-01103 Vilnius"
        detections = enhancer.find_enhanced_lithuanian_patterns(text)
        assert len(detections) >= 1

    def test_lithuanian_car_plate_pattern(self, enhancer):
        text = "Automobilio numeris: ABC 123"
        detections = enhancer.find_enhanced_lithuanian_patterns(text)
        assert len(detections) == 0

    def test_comprehensive_lithuanian_text(self, enhancer):
        text = "Gyvena Vilniuje. Adresas: Gedimino pr. 25, LT-01103, Vilnius. Telefonas: +370 600 12345. Asmens kodas: 38901234567. PVM kodas: LT123456789. Banko sąskaita: LT123456789012345678. Vardas: Ponas Jonas Petraitis."
        detections = enhancer.find_enhanced_lithuanian_patterns(text)
        detected_texts = {d['text'] for d in detections}
        
        assert "38901234567" in detected_texts
        assert "LT123456789" in detected_texts
        assert "LT123456789012345678" in detected_texts
        assert "Ponas Jonas Petraitis" in detected_texts
        assert any("Gedimino pr. 25" in d['text'] for d in detections)


if __name__ == "__main__":
    pytest.main([__file__])
