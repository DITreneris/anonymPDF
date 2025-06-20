import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app  # Assuming the FastAPI app instance is in app.main

SAMPLES_DIR = Path(__file__).parent.parent / "samples"

@pytest.mark.api
class TestPdfApiEndpoints:

    def test_post_lithuanian_pdf_for_processing_success(self, client: TestClient, monkeypatch):
        """
        Tests the /upload endpoint with a valid Lithuanian PDF.
        Asserts a 200 OK status and a valid JSON response structure.
        """
        # ARRANGE
        # Mock the celery task to avoid dependency on a running worker and redis
        mock_task = MagicMock()
        monkeypatch.setattr("app.api.endpoints.pdf.process_pdf_task.delay", mock_task)

        pii_pdf_path = SAMPLES_DIR / "lithuanian_pii_document.pdf"
        
        if not pii_pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pii_pdf_path}")

        # ACT
        with open(pii_pdf_path, "rb") as f:
            response = client.post("/api/v1/upload", files={"file": (pii_pdf_path.name, f, "application/pdf")})

        # ASSERT
        assert response.status_code == 202
        response_data = response.json()
        
        # The /upload endpoint returns a PDFDocument schema object
        assert "id" in response_data
        assert "status" in response_data
        assert response_data["original_filename"] == pii_pdf_path.name
        
        # The status should reflect that processing has been queued
        assert response_data["status"] == "queued"

    def test_post_non_pdf_file_returns_4xx(self, client: TestClient):
        """
        Tests the /upload endpoint with a non-PDF file.
        Asserts that the server returns a 4xx client error.
        """
        # ARRANGE
        non_pdf_path = SAMPLES_DIR / "not_a_pdf.txt"

        if not non_pdf_path.exists():
            # Create a dummy file if it doesn't exist
            non_pdf_path.touch()
            non_pdf_path.write_text("this is not a pdf")

        # ACT
        with open(non_pdf_path, "rb") as f:
            response = client.post("/api/v1/upload", files={"file": (non_pdf_path.name, f, "text/plain")})

        # ASSERT
        assert 400 <= response.status_code < 500 