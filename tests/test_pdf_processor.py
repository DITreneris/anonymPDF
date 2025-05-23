import pytest
from pathlib import Path
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
    
    assert len(personal_info['names']) > 0
    assert len(personal_info['locations']) > 0
    assert len(personal_info['emails']) > 0
    assert len(personal_info['phones']) > 0
    assert len(personal_info['ssns']) > 0
    assert len(personal_info['credit_cards']) > 0

def test_generate_redaction_report(pdf_processor, sample_text):
    personal_info = pdf_processor.find_personal_info(sample_text)
    report = pdf_processor.generate_redaction_report(personal_info, "en")
    
    assert "Redaction Report" in report
    assert "Detected Language: en" in report
    assert "NAMES:" in report
    assert "LOCATIONS:" in report
    assert "EMAILS:" in report
    assert "PHONES:" in report
    assert "SSNS:" in report
    assert "CREDIT_CARDS:" in report

@pytest.mark.asyncio
async def test_process_pdf(pdf_processor):
    # Create a mock PDF file
    content = b"%PDF-1.4\n%Test PDF content"
    file = UploadFile(
        filename="test.pdf",
        file=io.BytesIO(content)
    )
    
    result = await pdf_processor.process_pdf(file)
    
    assert "filename" in result
    assert "status" in result
    assert "report" in result
    assert result["filename"] == "test.pdf"
    assert result["status"] == "processed" 