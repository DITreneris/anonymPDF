"""
Comprehensive tests for Priority 2 improvements.

Tests context-aware detection, Lithuanian language enhancements,
confidence scoring, and advanced pattern refinement.
"""

import pytest
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.context_analyzer import (
    ContextualValidator,
    AdvancedPatternRefinement,
    DocumentStructureAnalyzer,
    DetectionContext,
    ConfidenceLevel,
    create_context_aware_detection
)
from app.core.lithuanian_enhancements import (
    LithuanianLanguageEnhancer,
    LithuanianContextAnalyzer
)


class TestContextualValidator:
    """Test context-aware validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ContextualValidator()
    
    def test_confidence_calculation_person_name(self):
        """Test confidence calculation for person names."""
        # High confidence case - name with title
        detection = "Jonas Petraitis"
        context = "Ponas Jonas Petraitis gimęs 1980 metais"
        confidence = self.validator.calculate_confidence(detection, "person_name", context)
        assert confidence > 0.8, f"Expected high confidence, got {confidence}"
    
    def test_confidence_calculation_false_positive(self):
        """Test confidence calculation for false positive cases."""
        # Low confidence case - document section reference
        detection = "Section"
        context = "Section 5 of the document describes"
        confidence = self.validator.calculate_confidence(detection, "person_name", context)
        assert confidence < 0.5, f"Expected low confidence, got {confidence}"
    
    def test_document_section_adjustment(self):
        """Test document section confidence adjustments."""
        detection = "Test Name"
        context = "CERTIFICATE OF INSURANCE Test Name appears here"
        confidence = self.validator.calculate_confidence(detection, "person_name", context, "header")
        
        # Header should reduce confidence
        base_context = "Test Name appears in the document"
        base_confidence = self.validator.calculate_confidence(detection, "person_name", base_context)
        assert confidence < base_confidence, "Header section should reduce confidence"
    
    def test_validate_with_context(self):
        """Test comprehensive context validation."""
        full_text = "INSURANCE CERTIFICATE\n\nName: Jonas Petraitis\nAddress: Vilnius, Lithuania"
        detection_context = self.validator.validate_with_context(
            "Jonas Petraitis", "person_name", full_text, 30, 45
        )
        
        assert isinstance(detection_context, DetectionContext)
        assert detection_context.text == "Jonas Petraitis"
        assert detection_context.category == "person_name"
        assert detection_context.confidence > 0.5
        assert len(detection_context.validation_flags) >= 0
    
    def test_confidence_level_mapping(self):
        """Test confidence level enum mapping."""
        assert self.validator.get_confidence_level(0.95) == ConfidenceLevel.VERY_HIGH
        assert self.validator.get_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert self.validator.get_confidence_level(0.65) == ConfidenceLevel.MEDIUM
        assert self.validator.get_confidence_level(0.45) == ConfidenceLevel.LOW
        assert self.validator.get_confidence_level(0.25) == ConfidenceLevel.VERY_LOW


class TestDocumentStructureAnalyzer:
    """Test document structure analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DocumentStructureAnalyzer()
    
    def test_identify_header_section(self):
        """Test header section identification."""
        text = "CERTIFICATE OF INSURANCE\n\nThis document certifies..."
        section = self.analyzer.identify_document_section(text, 10)
        assert section == "header"
    
    def test_identify_form_field_section(self):
        """Test form field section identification."""
        text = "Personal Information\nName: \nAddress: \nPhone: "
        section = self.analyzer.identify_document_section(text, 25)
        assert section == "form_field"
    
    def test_identify_footer_section(self):
        """Test footer section identification."""
        text = "Document content here\n\nPage 1 of 3\nwww.company.com"
        section = self.analyzer.identify_document_section(text, 40)
        assert section == "footer"
    
    def test_is_document_metadata(self):
        """Test document metadata detection."""
        # Should detect document metadata
        assert self.analyzer.is_document_metadata("Document Number", "Document Number: 12345")
        assert self.analyzer.is_document_metadata("Page 1", "Page 1 of 10")
        assert self.analyzer.is_document_metadata("Version 2", "Version 2.1")
        
        # Should not detect regular content as metadata
        assert not self.analyzer.is_document_metadata("Jonas Petraitis", "Name: Jonas Petraitis")


class TestAdvancedPatternRefinement:
    """Test advanced pattern refinement functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pattern_refiner = AdvancedPatternRefinement()
    
    def test_enhanced_email_detection(self):
        """Test enhanced email pattern detection."""
        text = "El. paštas: jonas.petraitis@example.com ir kitas tekstas"
        detections = self.pattern_refiner.find_enhanced_patterns(text)
        
        email_detections = [d for d in detections if d['pattern_name'] == 'email_contextual']
        assert len(email_detections) == 1
        assert email_detections[0]['text'] == "jonas.petraitis@example.com"
        assert email_detections[0]['confidence_boost'] == 0.2
    
    def test_enhanced_phone_detection(self):
        """Test enhanced phone pattern detection."""
        text = "Tel.: +370 6 123 4567 yra kontaktinis numeris"
        detections = self.pattern_refiner.find_enhanced_patterns(text)
        
        phone_detections = [d for d in detections if d['pattern_name'] == 'phone_contextual']
        assert len(phone_detections) == 1
        assert phone_detections[0]['text'] == "+370 6 123 4567"
        assert phone_detections[0]['confidence_boost'] == 0.2
    
    def test_enhanced_personal_code_detection(self):
        """Test enhanced personal code pattern detection."""
        text = "Asmens kodas: 38901234567 yra registruotas"
        detections = self.pattern_refiner.find_enhanced_patterns(text)
        
        code_detections = [d for d in detections if d['pattern_name'] == 'lithuanian_personal_code_contextual']
        assert len(code_detections) == 1
        assert code_detections[0]['text'] == "38901234567"
        assert code_detections[0]['confidence_boost'] == 0.2
    
    def test_enhanced_address_detection(self):
        """Test enhanced address pattern detection."""
        text = "Adresas: Paupio g. 50-136, LT-11341 Vilnius"
        detections = self.pattern_refiner.find_enhanced_patterns(text)
        
        address_detections = [d for d in detections if d['pattern_name'] == 'address_contextual']
        assert len(address_detections) == 1
        assert "Paupio g. 50-136, LT-11341 Vilnius" in address_detections[0]['text']
        assert address_detections[0]['confidence_boost'] == 0.15


class TestLithuanianLanguageEnhancer:
    """Test Lithuanian language enhancement functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enhancer = LithuanianLanguageEnhancer()
    
    def test_lithuanian_geographic_terms(self):
        """Test Lithuanian geographic term detection."""
        assert self.enhancer.is_lithuanian_geographic_term("Vilnius")
        assert self.enhancer.is_lithuanian_geographic_term("Kaunas")
        assert self.enhancer.is_lithuanian_geographic_term("Lietuva")
        assert not self.enhancer.is_lithuanian_geographic_term("Jonas")
    
    def test_lithuanian_document_terms(self):
        """Test Lithuanian document term detection."""
        assert self.enhancer.is_lithuanian_document_term("Dokumentas")
        assert self.enhancer.is_lithuanian_document_term("Pažymėjimas")
        assert self.enhancer.is_lithuanian_document_term("Draudimas")
        assert not self.enhancer.is_lithuanian_document_term("Petraitis")
    
    def test_lithuanian_common_words(self):
        """Test Lithuanian common word detection."""
        assert self.enhancer.is_lithuanian_common_word("Asmuo")
        assert self.enhancer.is_lithuanian_common_word("Vardas")
        assert self.enhancer.is_lithuanian_common_word("Numeris")
        assert not self.enhancer.is_lithuanian_common_word("Petraitis")
    
    def test_validate_lithuanian_name_valid(self):
        """Test validation of valid Lithuanian names."""
        # Valid Lithuanian male name
        is_valid, confidence = self.enhancer.validate_lithuanian_name("Jonas Petraitis")
        assert is_valid
        assert confidence > 0
        
        # Valid Lithuanian female name
        is_valid, confidence = self.enhancer.validate_lithuanian_name("Žaneta Stankevičienė")
        assert is_valid
        assert confidence > 0
    
    def test_validate_lithuanian_name_invalid(self):
        """Test validation of invalid Lithuanian names."""
        # Geographic term should be invalid
        is_valid, confidence = self.enhancer.validate_lithuanian_name("Vilnius")
        assert not is_valid
        assert confidence < 0
        
        # Document term should be invalid
        is_valid, confidence = self.enhancer.validate_lithuanian_name("Dokumentas")
        assert not is_valid
        assert confidence < 0
    
    def test_validate_lithuanian_name_with_context(self):
        """Test Lithuanian name validation with context."""
        # Name with title should get confidence boost
        context = "Ponas Jonas Petraitis gimęs 1980 metais"
        is_valid, confidence = self.enhancer.validate_lithuanian_name("Jonas Petraitis", context)
        assert is_valid
        assert confidence > 0.2  # Should get boost from title
    
    def test_lithuanian_swift_bic_validation(self):
        """Test Lithuanian SWIFT/BIC validation."""
        # Valid Lithuanian bank SWIFT code
        assert self.enhancer.validate_lithuanian_swift_bic("HABALT2X")
        assert self.enhancer.validate_lithuanian_swift_bic("CBVILT2X")
        
        # Invalid - Lithuanian common word
        assert not self.enhancer.validate_lithuanian_swift_bic("DRAUDIMO")
        assert not self.enhancer.validate_lithuanian_swift_bic("PRIVALOMOJO")
        
        # Invalid format
        assert not self.enhancer.validate_lithuanian_swift_bic("ABC")
        assert not self.enhancer.validate_lithuanian_swift_bic("12345678")
    
    def test_enhanced_lithuanian_patterns(self):
        """Test enhanced Lithuanian pattern detection."""
        text = """
        Ponas Jonas Petraitis
        Adresas: Paupio g. 50-136, LT-11341 Vilnius
        UAB "Lietuvos Technologijos"
        A.K.: 38901234567
        Tel.: +370 6 123 4567
        El. paštas: jonas@example.lt
        Banko sąskaita: LT12 3456 7890 1234 5678
        PVM kodas: LT100001738313
        2024 m. sausio 15 d.
        """
        
        detections = self.enhancer.find_enhanced_lithuanian_patterns(text)
        
        # Should find multiple enhanced patterns
        assert len(detections) > 5
        
        # Check specific patterns
        pattern_names = [d['pattern_name'] for d in detections]
        assert 'lithuanian_name_with_title' in pattern_names
        assert 'lithuanian_address_full' in pattern_names
        assert 'lithuanian_company_full' in pattern_names
        assert 'lithuanian_personal_code_labeled' in pattern_names


class TestLithuanianContextAnalyzer:
    """Test Lithuanian context analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = LithuanianContextAnalyzer()
    
    def test_identify_lithuanian_section_insurance(self):
        """Test Lithuanian insurance document section identification."""
        text = "PRIVALOMOJO CIVILĖS ATSAKOMYBĖS DRAUDIMO PAŽYMĖJIMAS"
        section = self.analyzer.identify_lithuanian_section(text, 20)
        assert section == "insurance_header"
    
    def test_identify_lithuanian_section_personal_info(self):
        """Test Lithuanian personal info section identification."""
        text = "DRAUDĖJO DUOMENYS\nVardas: Jonas Petraitis"
        section = self.analyzer.identify_lithuanian_section(text, 25)
        assert section == "personal_info_section"
    
    def test_identify_lithuanian_section_company_info(self):
        """Test Lithuanian company info section identification."""
        text = "ĮMONĖS DUOMENYS\nPavadinimas: UAB Technologijos"
        section = self.analyzer.identify_lithuanian_section(text, 20)
        assert section == "company_info_section"
    
    def test_calculate_lithuanian_confidence_personal_section(self):
        """Test confidence calculation for personal info section."""
        confidence = self.analyzer.calculate_lithuanian_confidence(
            "Jonas Petraitis", "names", "DRAUDĖJO DUOMENYS Jonas Petraitis", "personal_info_section"
        )
        assert confidence > 0.2  # Should get boost from personal info section
    
    def test_calculate_lithuanian_confidence_header_section(self):
        """Test confidence calculation for header section."""
        confidence = self.analyzer.calculate_lithuanian_confidence(
            "Draudimas", "names", "DRAUDIMO PAŽYMĖJIMAS Draudimas", "insurance_header"
        )
        assert confidence < 0  # Should get penalty from header section


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple Priority 2 features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ContextualValidator()
        self.pattern_refiner = AdvancedPatternRefinement()
        self.lithuanian_enhancer = LithuanianLanguageEnhancer()
        self.lithuanian_analyzer = LithuanianContextAnalyzer()
    
    def test_comprehensive_lithuanian_document_analysis(self):
        """Test comprehensive analysis of Lithuanian document."""
        document_text = """
        PRIVALOMOJO CIVILĖS ATSAKOMYBĖS DRAUDIMO PAŽYMĖJIMAS
        
        DRAUDĖJO DUOMENYS:
        Vardas: Jonas Petraitis
        Asmens kodas: 38901234567
        Adresas: Paupio g. 50-136, LT-11341 Vilnius
        Tel.: +370 6 123 4567
        El. paštas: jonas.petraitis@example.com
        
        ĮMONĖS DUOMENYS:
        UAB "Lietuvos Technologijos"
        PVM kodas: LT100001738313
        Banko sąskaita: LT12 3456 7890 1234 5678
        
        Data: 2024 m. sausio 15 d.
        """
        
        # Test enhanced pattern detection
        enhanced_detections = self.pattern_refiner.find_enhanced_patterns(document_text)
        lithuanian_detections = self.lithuanian_enhancer.find_enhanced_lithuanian_patterns(document_text)
        
        # Should find multiple high-confidence detections
        assert len(enhanced_detections) >= 3
        assert len(lithuanian_detections) >= 5
        
        # Test context-aware validation for a name
        name_start = document_text.find("Jonas Petraitis")
        name_end = name_start + len("Jonas Petraitis")
        
        detection_context = create_context_aware_detection(
            "Jonas Petraitis", "person_name", name_start, name_end, 
            document_text, self.validator
        )
        
        # Should have high confidence due to context
        assert detection_context.confidence > 0.7
        assert detection_context.document_section is not None
    
    def test_false_positive_filtering(self):
        """Test that Priority 2 improvements filter false positives."""
        document_text = """
        DOCUMENT HEADER
        
        Section 5: When processing documents, ensure that Gibraltar
        and United Kingdom are properly handled. The Document
        Number should be recorded.
        
        Table 1: Reference data
        """
        
        # Test that geographic terms in document context get low confidence
        gibraltar_context = create_context_aware_detection(
            "Gibraltar", "person_name", 50, 59, document_text, self.validator
        )
        assert gibraltar_context.confidence < 0.5
        
        # Test that document terms get flagged
        document_context = create_context_aware_detection(
            "Document", "person_name", 100, 108, document_text, self.validator
        )
        assert document_context.confidence < 0.5
        assert "structural_element" in document_context.validation_flags or \
               "document_metadata" in document_context.validation_flags
    
    def test_confidence_based_prioritization(self):
        """Test that higher confidence detections are prioritized."""
        # High confidence detection with explicit label
        high_conf_text = "Asmens kodas: 38901234567"
        high_detections = self.pattern_refiner.find_enhanced_patterns(high_conf_text)
        
        # Lower confidence detection without label
        low_conf_text = "Random number 38901234567 in text"
        low_context = create_context_aware_detection(
            "38901234567", "lithuanian_personal_codes", 14, 25, 
            low_conf_text, self.validator
        )
        
        # Enhanced pattern should have higher confidence
        if high_detections:
            high_confidence = 0.7 + high_detections[0]['confidence_boost']  # Base + boost
            assert high_confidence > low_context.confidence


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([__file__ + "::TestContextualValidator", "-v"])
    pytest.main([__file__ + "::TestLithuanianLanguageEnhancer", "-v"])
    pytest.main([__file__ + "::TestIntegrationScenarios", "-v"]) 