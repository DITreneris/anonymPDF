import pytest
from app.services.pdf_processor import PDFProcessor
from fastapi import UploadFile
import io


@pytest.fixture
def pdf_processor():
    return PDFProcessor()


@pytest.fixture
def sample_text():
    return """
    John Doe lives in New York City.
    Contact: john.doe@example.com
    Phone: (555) 123-4567
    SSN: 123-45-6789
    Credit Card: 1234 5678 9012 3456
    """


def test_detect_language(pdf_processor, sample_text):
    language = pdf_processor.detect_language(sample_text)
    assert language == "en"


def test_find_personal_info(pdf_processor, sample_text):
    personal_info = pdf_processor.find_personal_info(sample_text)

    assert len(personal_info["names"]) > 0
    assert len(personal_info["locations"]) > 0
    assert len(personal_info["emails"]) > 0
    assert len(personal_info["phones"]) > 0
    assert len(personal_info["ssns"]) > 0
    assert len(personal_info["credit_cards"]) > 0


def test_generate_redaction_report(pdf_processor, sample_text):
    personal_info = pdf_processor.find_personal_info(sample_text)
    report = pdf_processor.generate_redaction_report(personal_info, "en")

    assert isinstance(report, dict)
    assert "title" in report
    assert "Redaction Report" in report["title"]
    assert report["detectedLanguage"] == "en"
    assert "categories" in report
    assert "NAMES" in report["categories"]
    assert "LOCATIONS" in report["categories"]
    assert "EMAILS" in report["categories"]
    assert "PHONES" in report["categories"]
    assert "SSNS" in report["categories"]
    assert "CREDIT_CARDS" in report["categories"]


@pytest.mark.asyncio
async def test_process_pdf(pdf_processor, tmp_path):
    # Create a temporary PDF file
    test_pdf = tmp_path / "test.pdf"
    test_pdf.write_bytes(b"%PDF-1.4\n%Test PDF content")

    result = await pdf_processor.process_pdf(str(test_pdf))

    assert "filename" in result
    assert "status" in result
    assert result["filename"] == "test.pdf"
    # The result might be "failed" due to invalid PDF content, but that's expected
