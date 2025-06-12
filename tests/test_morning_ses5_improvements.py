"""
Tests for Morning Session 5 improvements: Enhanced regex patterns and anti-overredaction logic.
Based on analysis in morning_ses5.md
"""

import pytest
import re
from app.core.config_manager import ConfigManager
from app.services.pdf_processor import PDFProcessor
from pathlib import Path


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

    def test_enhanced_car_plate_contextual_pattern(self, config_manager):
        """Test enhanced car plate detection with 'Valst Nr' context."""
        pattern = config_manager.patterns.get("lithuanian_car_plate_contextual")
        assert pattern is not None
        
        test_texts = [
            "Valst. Nr.: HRV249",
            "valst. Nr: ABC123", 
            "Valst Nr HRV249",
            "valst.nr.:DEF456"
        ]
        
        for text in test_texts:
            matches = re.findall(pattern, text)
            assert len(matches) >= 1, f"Should detect car plate in: {text}"
            # Should extract just the plate number from capture group
            assert matches[0] in ["HRV249", "ABC123", "DEF456"], f"Should extract plate number from: {text}"

    def test_enhanced_car_plate_enhanced_pattern(self, config_manager):
        """Test comprehensive car plate detection patterns."""
        pattern = config_manager.patterns.get("lithuanian_car_plate_enhanced")
        assert pattern is not None
        
        test_texts = [
            "automobilio nr.: XYZ789",
            "Automobilio Nr. GHI456",
            "numeris: JKL123"
        ]
        
        for text in test_texts:
            matches = re.findall(pattern, text)
            assert len(matches) >= 1, f"Should detect car plate in: {text}"

    def test_lithuanian_name_all_caps_pattern(self, config_manager):
        """Test ALL CAPS Lithuanian name detection."""
        pattern = config_manager.patterns.get("lithuanian_name_all_caps")
        assert pattern is not None
        
        test_texts = [
            "STANIULIS TOMAS",
            "PETRAITĖ ONA", 
            "KAZLAUSKAS JONAS",
            "STANKEVIČIENĖ ŽANETA"
        ]
        
        for text in test_texts:
            matches = re.findall(pattern, text)
            assert len(matches) >= 1, f"Should detect ALL CAPS Lithuanian name: {text}"
            # The pattern has multiple groups, so we need to check if any group captured content
            found_name = False
            for match_groups in matches:
                if any(group for group in match_groups if group):  # Check if any group has content
                    found_name = True
                    break
            assert found_name, f"Should extract name components from: {text}"

    def test_lithuanian_name_contextual_pattern(self, config_manager):
        """Test Lithuanian name detection with context keywords."""
        pattern = config_manager.patterns.get("lithuanian_name_contextual")
        assert pattern is not None
        
        test_texts = [
            "Draudėjas: Jonas Petraitis",
            "Vardas: Žaneta Stankevičienė",
            "Pavardė: Kazlauskas",
            "Sutartį sudarė Ona Petraitė"
        ]
        
        for text in test_texts:
            matches = re.findall(pattern, text)
            assert len(matches) >= 1, f"Should detect contextual Lithuanian name: {text}"

    def test_lithuanian_name_contextual_caps_pattern(self, config_manager):
        """Test ALL CAPS Lithuanian name with context."""
        pattern = config_manager.patterns.get("lithuanian_name_contextual_caps")
        assert pattern is not None
        
        test_texts = [
            "Draudėjas: STANIULIS TOMAS",
            "DRAUDĖJAS: PETRAITĖ ONA",
            "Vardas: KAZLAUSKAS JONAS"
        ]
        
        for text in test_texts:
            matches = re.findall(pattern, text)
            assert len(matches) >= 1, f"Should detect contextual ALL CAPS Lithuanian name: {text}"

    def test_lithuanian_address_flexible_pattern(self, config_manager):
        """Test flexible address detection."""
        pattern = config_manager.patterns.get("lithuanian_address_flexible")
        assert pattern is not None
        
        test_texts = [
            "Vileišio g. 11-4",
            "Gedimino pr. 25",
            "Konstitucijos al. 7",
            "Paupio",  # Just street name
            "Vilniaus gatvė 123"
        ]
        
        for text in test_texts:
            matches = re.findall(pattern, text)
            assert len(matches) >= 1, f"Should detect flexible address: {text}"

    def test_lithuanian_personal_code_contextual_pattern(self, config_manager):
        """Test personal code with explicit context."""
        pattern = config_manager.patterns.get("lithuanian_personal_code_contextual")
        assert pattern is not None
        
        test_texts = [
            "asmens kodas: 38901234567",
            "asmens/įmonės kodas: 49012345678",
            "A.K.: 50001234567"
        ]
        
        for text in test_texts:
            matches = re.findall(pattern, text)
            assert len(matches) >= 1, f"Should detect contextual personal code: {text}"
            # Should extract just the personal code
            assert len(matches[0]) == 11, f"Should extract 11-digit personal code from: {text}"


class TestMorningSes5AntiOverredaction:
    """Test anti-overredaction logic from Morning Session 5."""

    @pytest.fixture
    def pdf_processor(self):
        """Fixture for PDFProcessor with test config."""
        processor = PDFProcessor()
        # Add test anti-overredaction settings
        processor.config_manager.settings['anti_overredaction'] = {
            'technical_terms_whitelist': ['kW', 'Nm', 'g/km', 'CO2 emisijos', 'kg'],
            'technical_sections': ['SVORIS', 'VAŽ.', 'Techniniai duomenys'],
            'pii_field_labels': ['Vardas', 'Asmens kodas', 'Draudėjas']
        }
        return processor

    def test_preserve_detection_technical_terms(self, pdf_processor):
        """Test preservation of detections near technical terms."""
        # Should preserve due to technical term context
        assert pdf_processor.should_preserve_detection(
            "150", "eleven_digit_numeric", "Maks. variklio galia: 150 kW"
        )
        
        assert pdf_processor.should_preserve_detection(
            "250", "health_insurance_number", "Sukimo momentas: 250 Nm"
        )
        
        assert pdf_processor.should_preserve_detection(
            "120", "medical_record_number", "CO2 emisijos: 120 g/km"
        )

    def test_preserve_detection_technical_sections(self, pdf_processor):
        """Test preservation in technical sections."""
        # Should preserve in technical sections (when no PII field present)
        assert pdf_processor.should_preserve_detection(
            "1500", "eleven_digit_numeric", "SVORIS: Nuosavas svoris 1500 kg"
        )
        
        assert pdf_processor.should_preserve_detection(
            "2000", "health_insurance_number", "VAŽ. specifikacija: 2000"
        )

    def test_no_preserve_pii_in_technical_sections(self, pdf_processor):
        """Test that PII fields are still redacted even in technical sections."""
        # Should NOT preserve PII field even in technical section
        assert not pdf_processor.should_preserve_detection(
            "38901234567", "lithuanian_personal_code", "SVORIS: Asmens kodas: 38901234567"
        )

    def test_preserve_detection_technical_indicators(self, pdf_processor):
        """Test preservation based on technical indicators."""
        # Should preserve numeric patterns near technical indicators
        assert pdf_processor.should_preserve_detection(
            "12345678901", "eleven_digit_numeric", "Variklio galia: 12345678901 specifikacija"
        )
        
        assert pdf_processor.should_preserve_detection(
            "987654", "health_insurance_number", "Degalų sąnaudos: 987654 l/100 km"
        )

    def test_no_preserve_regular_detections(self, pdf_processor):
        """Test that regular PII detections are not preserved."""
        # Should NOT preserve regular PII without technical context
        assert not pdf_processor.should_preserve_detection(
            "38901234567", "lithuanian_personal_code", "Vardas: Jonas, asmens kodas: 38901234567"
        )
        
        assert not pdf_processor.should_preserve_detection(
            "ABC123", "lithuanian_car_plate", "Valst. Nr.: ABC123"
        )


class TestMorningSes5Integration:
    """Integration tests for Morning Session 5 improvements."""

    @pytest.fixture
    def pdf_processor(self):
        """Fixture for PDFProcessor."""
        return PDFProcessor()

    def test_comprehensive_lithuanian_document_processing(self, pdf_processor):
        """Test comprehensive processing of Lithuanian document with Morning Session 5 improvements."""
        # Sample text representing the issues from morning_ses5.md
        sample_text = """
        Draudėjas: STANIULIS TOMAS
        asmens/įmonės kodas: 38901234567
        
        Automobilio duomenys:
        Valst. Nr.: HRV249
        Adresas: P. Vileišio g. 11-4
        
        Techniniai duomenys:
        Maks. variklio galia: 150 kW
        Sukimo momentas: 250 Nm
        CO2 emisijos: 120 g/km
        SVORIS: 1500 kg
        """
        
        personal_info = pdf_processor.find_personal_info(sample_text, language="lt")
        
        # Should detect the enhanced patterns
        assert len(personal_info["names"]) >= 1, "Should detect STANIULIS TOMAS"
        assert len(personal_info["automotive"]) >= 1, "Should detect HRV249"
        assert len(personal_info["lithuanian_personal_codes"]) >= 1, "Should detect personal code"
        assert len(personal_info["addresses_prefixed"]) >= 1, "Should detect address"
        
        # Verify specific detections
        detected_names = [item[0] for item in personal_info["names"]]
        detected_plates = [item[0] for item in personal_info["automotive"]]
        detected_codes = [item[0] for item in personal_info["lithuanian_personal_codes"]]
        
        assert any("STANIULIS TOMAS" in name for name in detected_names), "Should detect ALL CAPS name"
        assert "HRV249" in detected_plates, "Should detect car plate from Valst. Nr. context"
        assert "38901234567" in detected_codes, "Should detect contextual personal code"

    def test_anti_overredaction_in_technical_context(self, pdf_processor):
        """Test that technical values are preserved while PII is redacted."""
        # Set up anti-overredaction config
        pdf_processor.config_manager.settings['anti_overredaction'] = {
            'technical_terms_whitelist': ['kW', 'Nm', 'g/km', 'kg'],
            'technical_sections': ['Techniniai duomenys'],
            'pii_field_labels': ['Draudėjas', 'asmens kodas']
        }
        
        technical_text = """
        Draudėjas: Jonas Petraitis
        asmens kodas: 38901234567
        
        Techniniai duomenys:
        Variklio galia: 150 kW
        Sukimo momentas: 250 Nm
        Emisijos: 120 g/km
        """
        
        personal_info = pdf_processor.find_personal_info(technical_text, language="lt")
        
        # Should detect PII 
        assert len(personal_info["names"]) >= 1, "Should detect name"
        assert len(personal_info["lithuanian_personal_codes"]) >= 1, "Should detect personal code"
        
        # Technical values should be preserved (not detected/redacted)
        # This is harder to test directly since preserved items don't appear in personal_info
        # but we can verify the logic works through the should_preserve_detection method
        assert pdf_processor.should_preserve_detection("150", "eleven_digit_numeric", "Variklio galia: 150 kW")


if __name__ == "__main__":
    pytest.main([__file__]) 