import pytest
import re
from app.core.config_manager import ConfigManager


@pytest.mark.pii
@pytest.mark.unit
class TestPIIPatterns:
    """Test PII detection patterns."""

    def test_email_pattern(self, test_config_manager: ConfigManager):
        """Test email pattern detection."""
        pattern = test_config_manager.patterns["email"]

        # Valid emails
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "firstname+lastname@company.org",
            "user123@test-domain.com",
        ]

        for email in valid_emails:
            matches = re.findall(pattern, email)
            assert len(matches) == 1, f"Should detect email: {email}"
            assert matches[0] == email

        # Invalid emails (should not match)
        invalid_emails = ["notanemail", "@domain.com", "user@", "user@domain", "user.domain.com"]

        for invalid in invalid_emails:
            matches = re.findall(pattern, invalid)
            assert len(matches) == 0, f"Should not detect invalid email: {invalid}"

    def test_lithuanian_personal_code_pattern(self, test_config_manager: ConfigManager):
        """Test Lithuanian personal code pattern detection."""
        pattern = test_config_manager.patterns["lithuanian_personal_code"]

        # Valid personal codes (starting with 3-6)
        valid_codes = [
            "38901234567",  # Male born in 1989
            "48901234567",  # Female born in 1989
            "50001234567",  # Male born in 2000
            "60001234567",  # Female born in 2000
        ]

        for code in valid_codes:
            matches = re.findall(pattern, code)
            assert len(matches) == 1, f"Should detect personal code: {code}"
            assert matches[0] == code

        # Invalid personal codes
        invalid_codes = [
            "12345678901",  # Starts with 1
            "28901234567",  # Starts with 2
            "78901234567",  # Starts with 7
            "3890123456",  # Too short
            "389012345678",  # Too long
            "abcd1234567",  # Contains letters
        ]

        for invalid in invalid_codes:
            matches = re.findall(pattern, invalid)
            assert len(matches) == 0, f"Should not detect invalid personal code: {invalid}"

    def test_lithuanian_phone_patterns(self, test_config_manager: ConfigManager):
        """Test Lithuanian phone number patterns."""

        # Test generic Lithuanian phone pattern
        generic_pattern = test_config_manager.patterns["lithuanian_phone_generic"]
        valid_generic = ["+370 600 12345", "+370 612 34567", "+370 698 76543"]

        for phone in valid_generic:
            matches = re.findall(generic_pattern, phone)
            assert len(matches) == 1, f"Should detect generic Lithuanian phone: {phone}"

        # Test compact Lithuanian phone pattern
        compact_pattern = test_config_manager.patterns["lithuanian_phone_compact"]
        valid_compact = ["+37060012345", "+37061234567", "+37069876543"]

        for phone in valid_compact:
            matches = re.findall(compact_pattern, phone)
            assert len(matches) == 1, f"Should detect compact Lithuanian phone: {phone}"

        # Test prefixed Lithuanian phone pattern
        prefixed_pattern = test_config_manager.patterns["lithuanian_mobile_prefixed"]
        valid_prefixed = [
            "Tel. nr.: +370 600 12345",
            "Tel. +370 612 34567",
            "Tel.nr.:+370 698 76543",
        ]

        for phone in valid_prefixed:
            matches = re.findall(prefixed_pattern, phone)
            assert len(matches) >= 1, f"Should detect prefixed Lithuanian phone: {phone}"

    def test_lithuanian_vat_code_patterns(self, test_config_manager: ConfigManager):
        """Test Lithuanian VAT code patterns."""

        # Test labeled VAT code pattern
        labeled_pattern = test_config_manager.patterns["lithuanian_vat_code_labeled"]
        valid_labeled = [
            "PVM kodas: LT100001738313",
            "PVM kodas:LT123456789012",
            "PVM kodas LT987654321098",
        ]

        for vat in valid_labeled:
            matches = re.findall(labeled_pattern, vat)
            assert len(matches) >= 1, f"Should detect labeled VAT code: {vat}"

        # Test standalone VAT code pattern
        standalone_pattern = test_config_manager.patterns["lithuanian_vat_code"]
        valid_standalone = ["LT100001738313", "LT123456789012", "LT987654321098"]

        for vat in valid_standalone:
            matches = re.findall(standalone_pattern, vat)
            assert len(matches) == 1, f"Should detect standalone VAT code: {vat}"

    def test_lithuanian_iban_pattern(self, test_config_manager: ConfigManager):
        """Test Lithuanian IBAN pattern."""
        pattern = test_config_manager.patterns["lithuanian_iban"]

        valid_ibans = ["LT123456789012345678", "LT987654321098765432", "LT111111111111111111"]

        for iban in valid_ibans:
            matches = re.findall(pattern, iban)
            assert len(matches) == 1, f"Should detect Lithuanian IBAN: {iban}"

        # Invalid IBANs
        invalid_ibans = [
            "LT12345678901234567",  # Too short
            "LT1234567890123456789",  # Too long
            "LV123456789012345678",  # Wrong country code
            "lt123456789012345678",  # Lowercase
        ]

        for invalid in invalid_ibans:
            matches = re.findall(pattern, invalid)
            assert len(matches) == 0, f"Should not detect invalid IBAN: {invalid}"

    def test_lithuanian_address_patterns(self, test_config_manager: ConfigManager):
        """Test Lithuanian address patterns."""

        # Test prefixed address pattern
        prefixed_pattern = test_config_manager.patterns["lithuanian_address_prefixed"]
        valid_prefixed = [
            "Adresas: Gedimino pr. 25, LT-01103, Vilnius",
            "Adresas: Paupio g. 50-136",
            "Adresas: Konstitucijos al. 7",
        ]

        for address in valid_prefixed:
            matches = re.findall(prefixed_pattern, address)
            assert len(matches) >= 1, f"Should detect prefixed address: {address}"

        # Test postal code pattern
        postal_pattern = test_config_manager.patterns["lithuanian_postal_code"]
        valid_postal = ["LT-01103", "LT-44001", "LT-92111"]

        for postal in valid_postal:
            matches = re.findall(postal_pattern, postal)
            assert len(matches) == 1, f"Should detect postal code: {postal}"

    def test_date_patterns(self, test_config_manager: ConfigManager):
        """Test date patterns."""

        # Test YYYY-MM-DD pattern
        dash_pattern = test_config_manager.patterns["date_yyyy_mm_dd"]
        valid_dash_dates = ["2024-01-15", "1989-12-31", "2000-06-15"]

        for date in valid_dash_dates:
            matches = re.findall(dash_pattern, date)
            assert len(matches) == 1, f"Should detect dash date: {date}"

        # Test YYYY.MM.DD pattern
        dot_pattern = test_config_manager.patterns["date_yyyy_mm_dd_dots"]
        valid_dot_dates = ["2024.01.15", "1989.12.31", "2000.06.15"]

        for date in valid_dot_dates:
            matches = re.findall(dot_pattern, date)
            assert len(matches) == 1, f"Should detect dot date: {date}"

    def test_business_certificate_patterns(self, test_config_manager: ConfigManager):
        """Test business certificate patterns."""

        # Test AF format business certificates
        af_pattern = test_config_manager.patterns["lithuanian_business_cert"]
        valid_af = ["AF123456-1", "AF987654-9", "AF555555-5"]

        for cert in valid_af:
            matches = re.findall(af_pattern, cert)
            assert len(matches) == 1, f"Should detect AF business certificate: {cert}"

    def test_healthcare_patterns(self, test_config_manager: ConfigManager):
        """Test healthcare-related patterns."""

        # Test blood group pattern
        blood_pattern = test_config_manager.patterns["blood_group"]
        valid_blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

        for blood in valid_blood_groups:
            # Test in context to ensure word boundaries work
            text = f"Blood type: {blood} patient"
            matches = re.findall(blood_pattern, text)
            assert len(matches) == 1, f"Should detect blood group: {blood}"

    def test_automotive_patterns(self, test_config_manager: ConfigManager):
        """Test automotive patterns."""

        # Test Lithuanian car plate pattern
        plate_pattern = test_config_manager.patterns["lithuanian_car_plate"]
        valid_plates = ["ABC-123", "XYZ 456", "DEF-789"]

        for plate in valid_plates:
            matches = re.findall(plate_pattern, plate)
            assert len(matches) == 1, f"Should detect car plate: {plate}"

    def test_financial_patterns(self, test_config_manager: ConfigManager):
        """Test financial patterns."""

        # Test enhanced credit card pattern
        cc_pattern = test_config_manager.patterns["credit_card_enhanced"]
        valid_cards = [
            "4111111111111111",  # Visa
            "5555555555554444",  # MasterCard
            "378282246310005",  # American Express
            "6011111111111117",  # Discover
        ]

        for card in valid_cards:
            matches = re.findall(cc_pattern, card)
            assert len(matches) == 1, f"Should detect credit card: {card}"

        # Test SWIFT/BIC pattern
        swift_pattern = test_config_manager.patterns["swift_bic"]
        valid_swift = ["DEUTDEFF", "CHASUS33", "BNPAFRPP"]

        for swift in valid_swift:
            matches = re.findall(swift_pattern, swift)
            assert len(matches) == 1, f"Should detect SWIFT code: {swift}"


@pytest.mark.pii
@pytest.mark.integration
class TestPIIPatternIntegration:
    """Integration tests for PII pattern detection."""

    def test_comprehensive_lithuanian_text(
        self, test_config_manager: ConfigManager, sample_lithuanian_text: str
    ):
        """Test comprehensive PII detection in Lithuanian text."""
        patterns = test_config_manager.patterns

        # Count expected detections
        expected_detections = {
            "email": 1,  # jonas.petraitis@example.com
            "lithuanian_personal_code": 1,  # 38901234567
            "lithuanian_phone_generic": 1,  # +370 600 12345
            "lithuanian_vat_code": 1,  # LT100001738313
            "lithuanian_iban": 1,  # LT123456789012345678
            "date_yyyy_mm_dd": 1,  # 1989-01-23
            "lithuanian_car_plate": 1,  # ABC-123
            "lithuanian_postal_code": 1,  # LT-01103
        }

        for pattern_name, expected_count in expected_detections.items():
            if pattern_name in patterns:
                matches = re.findall(patterns[pattern_name], sample_lithuanian_text)
                assert len(matches) >= expected_count, (
                    f"Should detect {expected_count} {pattern_name} in Lithuanian text, "
                    f"found {len(matches)}"
                )

    def test_comprehensive_english_text(
        self, test_config_manager: ConfigManager, sample_english_text: str
    ):
        """Test comprehensive PII detection in English text."""
        patterns = test_config_manager.patterns

        # Count expected detections
        expected_detections = {
            "email": 1,  # john.smith@example.com
            "ssn": 1,  # 123-45-6789
            "credit_card": 1,  # 4111 1111 1111 1111
            "date_yyyy_mm_dd": 1,  # 2024-01-15
        }

        for pattern_name, expected_count in expected_detections.items():
            if pattern_name in patterns:
                matches = re.findall(patterns[pattern_name], sample_english_text)
                assert len(matches) >= expected_count, (
                    f"Should detect {expected_count} {pattern_name} in English text, "
                    f"found {len(matches)}"
                )
