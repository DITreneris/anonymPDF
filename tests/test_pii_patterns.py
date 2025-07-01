import pytest
import re
from app.core.config_manager import ConfigManager


@pytest.mark.pii
@pytest.mark.unit
class TestPIIPatterns:
    """Test PII detection patterns."""

    def test_email_pattern(self, test_config_manager: ConfigManager):
        """Test email pattern detection."""
        pattern = test_config_manager.patterns["emails"]

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

    def test_comprehensive_english_text(
        self, test_config_manager: ConfigManager, sample_english_text: str
    ):
        """Test comprehensive PII detection in English text."""

        expected_counts = {
            "emails": 1,
            "phone": 1,
            "ssn": 1,
            "credit_card": 1,
            "date_yyyy_mm_dd": 1,
        }

        for category, count in expected_counts.items():
            pattern = test_config_manager.patterns[category]
            matches = re.findall(pattern, sample_english_text)
            assert len(matches) == count, f"Expected {count} for '{category}', found {len(matches)}"
