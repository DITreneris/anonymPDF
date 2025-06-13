"""
Tests for validation utilities that improve PII detection accuracy.
"""

import pytest
from app.core.validation_utils import (
    validate_person_name,
    validate_swift_bic,
    validate_organization_name,
    validate_detection_context,
    deduplicate_detections,
    GEOGRAPHIC_EXCLUSIONS,
    DOCUMENT_TERMS,
    LITHUANIAN_COMMON_WORDS
)


class TestPersonNameValidation:
    """Test person name validation function."""

    def test_valid_person_names(self):
        """Test that valid person names pass validation."""
        valid_names = [
            "Jonas Petraitis",
            "Žaneta Stankevičienė",
            "John Smith",
            "Mary Jane Watson",
            "Ana",
            "José María García"
        ]

        for name in valid_names:
            assert validate_person_name(name), f"Valid name should pass: {name}"

    def test_geographic_exclusions(self):
        """Test that geographic terms are filtered out."""
        geographic_terms = [
            "Gibraltar", "United Kingdom", "Lithuania", "Germany", "Poland",
            "Lietuvos", "Europos", "Baltijos", "Vilnius", "Kaunas"
        ]

        for term in geographic_terms:
            assert not validate_person_name(term), f"Geographic term should be filtered: {term}"

    def test_document_terms_exclusions(self):
        """Test that document terminology is filtered out."""
        document_terms = [
            "Document", "Certificate", "Policy", "Agreement", "Contract",
            "Dokumentas", "Pažymėjimas", "Sutartis", "Draudimo", "When", "Where"
        ]

        for term in document_terms:
            assert not validate_person_name(term), f"Document term should be filtered: {term}"

    def test_valid_names(self):
        """Test that common valid names pass validation."""
        assert validate_person_name("John Doe") is True
        assert validate_person_name("Vardenis Pavardenis") is True
        assert validate_person_name("O'Malley") is True
        assert validate_person_name("Anne-Marie") is True

    def test_length_validation(self):
        """Test the name length validation logic."""
        # Test short name (should fail as per logic: len < 2)
        assert not validate_person_name("A"), "Single character should not be a valid name"

        # Test long name (should fail as per logic: len > 100)
        long_name = "A" * 101
        assert not validate_person_name(long_name), "Name longer than 100 chars should be invalid"

        # Test names at the boundary (should pass)
        assert validate_person_name("Jo"), "Two characters should be a valid name"
        assert validate_person_name("B" * 100), "Name with 100 chars should be valid"

    def test_excluded_prefixes(self):
        """Test that names with excluded prefixes are filtered out."""
        excluded_names = [
            "Nr. 123", "No. 456", "Art. 7", "Sec. 4", "Ch. 2", "Par. 3", "§ 15"
        ]

        for name in excluded_names:
            assert not validate_person_name(name), f"Name with excluded prefix should be filtered: {name}"

    def test_digit_validation(self):
        """Test that names with digits are rejected."""
        assert not validate_person_name("John123"), "Name with digits should be invalid: John123"
        assert not validate_person_name("MaryJaneWatson123"), "Name with digits should be invalid: MaryJaneWatson123"
        assert not validate_person_name("Ana1"), "Name with digits should be invalid: Ana1"
        assert not validate_person_name("JoséMaríaGarcía1"), "Name with digits should be invalid: JoséMaríaGarcía1"
        assert not validate_person_name("JonasPetraitis123"), "Name with digits should be invalid: JonasPetraitis123"
        assert not validate_person_name("ŽanetaStankevičienė123"), "Name with digits should be invalid: ŽanetaStankevičienė123"
        assert not validate_person_name("JohnSmith123"), "Name with digits should be invalid: JohnSmith123"


class TestSwiftBicValidation:
    """Test SWIFT/BIC code validation function."""

    def test_valid_swift_codes(self):
        """Test that valid SWIFT codes pass validation."""
        valid_codes = [
            "DEUTDEFF",     # 8 chars - Deutsche Bank Frankfurt
            "CHASUS33",     # 8 chars - Chase Bank
            "BNPAFRPP",     # 8 chars - BNP Paribas
            "DEUTDEFF500",  # 11 chars - Deutsche Bank with branch
            "CHASUS33XXX",  # 11 chars - Chase Bank with branch
        ]

        for code in valid_codes:
            assert validate_swift_bic(code), f"Valid SWIFT code should pass: {code}"

    def test_lithuanian_common_words(self):
        """Test that Lithuanian common words are filtered out."""
        lithuanian_words = [
            "PRIVALOMOJO", "DRAUDIMO", "SUTARTIS", "LIETUVOS", "RESPUBLIKA",
            "BENDROVĖ", "UŽDAROJI", "TECHNOLOG"
        ]

        for word in lithuanian_words:
            assert not validate_swift_bic(word), f"Lithuanian word should be filtered: {word}"

    def test_invalid_length(self):
        """Test that codes with invalid length are rejected."""
        invalid_lengths = [
            "DEUT",      # Too short (4 chars)
            "DEUTDE",    # Too short (6 chars)
            "DEUTDEFF5001234"  # Too long (15 chars)
        ]

        for code in invalid_lengths:
            assert not validate_swift_bic(code), f"Invalid length code should be rejected: {code}"

    def test_invalid_format(self):
        """Test that codes with invalid format are rejected."""
        invalid_formats = [
            "1EUTDEFF",    # First 4 not alphabetic
            "DEUT1EFF",    # Country code contains number
            "DEUTDE1F",    # Location code invalid (was previously failing, should be caught by refined logic)
            # "DEUTDEFF50A" was here - removed as it's a valid format
        ]

        for code in invalid_formats:
            assert not validate_swift_bic(code), f"Invalid format code should be rejected: {code}"


class TestOrganizationValidation:
    """Test organization name validation function."""

    def test_valid_organizations(self):
        """Test that valid organization names pass validation."""
        valid_orgs = [
            "P&C Insurance AS",
            "UAB Lietuvos Technologijos",
            "Microsoft Corporation",
            "Google LLC",
            "Bank of Lithuania"
        ]

        for org in valid_orgs:
            assert validate_organization_name(org), f"Valid organization should pass: {org}"

    def test_geographic_exclusions_for_orgs(self):
        """Test that geographic terms are filtered out for organizations."""
        geographic_terms = [
            "Lithuania", "Germany", "United Kingdom", "Lietuvos", "Europos"
        ]

        for term in geographic_terms:
            assert not validate_organization_name(term), f"Geographic term should be filtered: {term}"

    def test_short_organizations(self):
        """Test that very short organization names are filtered out."""
        short_orgs = ["A", "AB", "1", ""]

        for org in short_orgs:
            assert not validate_organization_name(org), f"Short org name should be filtered: {org}"


class TestContextValidation:
    """Test context-based validation function."""

    def test_person_context_validation(self):
        """Test person name validation based on context."""
        # Valid context
        valid_context = "The insurance policy holder is Jonas Petraitis who lives in Vilnius."
        assert validate_detection_context("Jonas Petraitis", valid_context, "PERSON")

        # Invalid context - document structure
        invalid_context = "Section 3.1 defines Jonas as the primary beneficiary in chapter 5."
        assert not validate_detection_context("Jonas", invalid_context, "PERSON")

    def test_organization_context_validation(self):
        """Test organization validation based on context."""
        # Valid context
        valid_context = "The insurance is provided by P&C Insurance AS, a licensed company."
        assert validate_detection_context("P&C Insurance AS", valid_context, "ORG")

        # Invalid context - common word
        invalid_context = "When will the policy expire and where can I renew it?"
        assert not validate_detection_context("When", invalid_context, "ORG")


class TestDeduplication:
    """Test deduplication functionality."""

    def test_basic_deduplication(self):
        """Test basic deduplication removes exact duplicates."""
        detections = {
            "names": [("John Smith", "PERSON")],
            "organizations": [("John Smith", "ORG")],  # Same text, different category
            "emails": [("test@example.com", "EMAIL")],
        }

        result = deduplicate_detections(detections)

        # John Smith should only appear once, in the higher specificity category
        john_smith_count = sum(
            1 for items in result.values()
            for text, _ in items
            if text.lower() == "john smith"
        )
        assert john_smith_count == 1

    def test_specificity_prioritization(self):
        """Test that more specific patterns are prioritized."""
        detections = {
            "lithuanian_personal_codes": [("37611010029", "LITHUANIAN_PERSONAL_CODE")],
            "healthcare_medical": [("37611010029", "HEALTH_INSURANCE")],
            "eleven_digit_numerics": [("37611010029", "ELEVEN_DIGIT_NUMERIC")],
            "emails": [("test@example.com", "EMAIL")],
        }

        result = deduplicate_detections(detections)

        # Personal code should only appear in the most specific category
        personal_code_appearances = []
        for category, items in result.items():
            for text, _ in items:
                if text == "37611010029":
                    personal_code_appearances.append(category)

        assert len(personal_code_appearances) == 1
        assert personal_code_appearances[0] == "lithuanian_personal_codes"

    def test_deduplication_preserves_unique_items(self):
        """Test that unique items are preserved during deduplication."""
        detections = {
            "emails": [("user1@example.com", "EMAIL"), ("user2@example.com", "EMAIL")],
            "phones": [("123-456-7890", "PHONE")],
            "names": [("John Smith", "PERSON"), ("Jane Doe", "PERSON")],
        }

        result = deduplicate_detections(detections)

        # All unique items should be preserved
        assert len(result["emails"]) == 2
        assert len(result["phones"]) == 1
        assert len(result["names"]) == 2

    def test_empty_deduplication(self):
        """Test deduplication with empty input."""
        empty_detections = {
            "names": [],
            "emails": [],
            "phones": []
        }

        result = deduplicate_detections(empty_detections)

        assert all(len(items) == 0 for items in result.values())

    def test_deduplication_logging_metrics(self):
        """Test that deduplication provides correct metrics."""
        detections = {
            "names": [("Test", "PERSON")],
            "organizations": [("Test", "ORG")],  # Duplicate
            "emails": [("unique@example.com", "EMAIL")],
        }

        result = deduplicate_detections(detections)

        original_count = sum(len(items) for items in detections.values())
        final_count = sum(len(items) for items in result.values())

        assert original_count == 3  # 2 duplicates + 1 unique
        assert final_count == 2    # 1 deduplicated + 1 unique


class TestExclusionLists:
    """Test that exclusion lists contain expected items."""

    def test_geographic_exclusions_coverage(self):
        """Test that geographic exclusions include key terms."""
        expected_terms = [
            "Lithuania", "Germany", "Poland", "Gibraltar", "United", "Kingdom",
            "Lietuvos", "Europos", "Baltijos"
        ]
        for term in expected_terms:
            assert term in GEOGRAPHIC_EXCLUSIONS, f"Expected geographic term missing: {term}"

    def test_document_terms_coverage(self):
        """Test that document terms include key terms."""
        expected_terms = [
            "Document", "Certificate", "Policy", "Contract", "Insurance",
            "Dokumentas", "Draudimo", "Sutartis", "When", "Where", "What"
        ]
        for term in expected_terms:
            assert term in DOCUMENT_TERMS, f"Expected document term missing: {term}"

    def test_lithuanian_words_coverage(self):
        """Test that Lithuanian words include key terms."""
        expected_terms = [
            "PRIVALOMOJO", "DRAUDIMO", "SUTARTIS", "LIETUVOS", "RESPUBLIKA"
        ]
        for term in expected_terms:
            assert term in LITHUANIAN_COMMON_WORDS, f"Expected Lithuanian word missing: {term}"
