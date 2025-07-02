"""
Comprehensive tests for app.api.endpoints.feedback module.
Tests feedback submission endpoint with database mocking and error handling.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException

from app.api.endpoints.feedback import FeedbackItem, FeedbackPayload
from app.core.feedback_system import UserFeedback, FeedbackType, FeedbackSeverity


class TestFeedbackModels:
    """Test the Pydantic models for feedback."""

    def test_feedback_item_creation(self):
        """Test FeedbackItem model creation."""
        item = FeedbackItem(
            text_segment="John Doe",
            original_category="names",
            is_correct=True
        )
        
        assert item.text_segment == "John Doe"
        assert item.original_category == "names"
        assert item.is_correct is True

    def test_feedback_payload_creation(self):
        """Test FeedbackPayload model creation."""
        items = [
            FeedbackItem(text_segment="John Doe", original_category="names", is_correct=True),
            FeedbackItem(text_segment="Not PII", original_category="names", is_correct=False)
        ]
        
        payload = FeedbackPayload(
            document_id=123,
            feedback_items=items
        )
        
        assert payload.document_id == 123
        assert len(payload.feedback_items) == 2


class TestFeedbackEndpoint:
    """Test the feedback submission endpoint."""

    @pytest.fixture
    def sample_payload(self):
        """Sample feedback payload."""
        items = [
            FeedbackItem(text_segment="John Doe", original_category="names", is_correct=True),
            FeedbackItem(text_segment="Not PII", original_category="names", is_correct=False)
        ]
        return FeedbackPayload(document_id=123, feedback_items=items)

    @patch('app.api.endpoints.feedback.api_logger')
    def test_submit_feedback_adaptive_disabled(self, mock_logger, sample_payload):
        """Test feedback when adaptive learning is disabled."""
        from app.api.endpoints.feedback import submit_feedback
        
        # Mock coordinator as disabled
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = False
        
        result = submit_feedback(
            payload=sample_payload,
            coordinator=mock_coordinator,
            pdf_processor=Mock(),
            db=Mock()
        )
        
        assert result["message"] == "Feedback received, but adaptive learning is disabled."
        mock_logger.warning.assert_called()

    @patch('app.api.endpoints.feedback.api_logger')
    def test_submit_feedback_document_not_found(self, mock_logger, sample_payload):
        """Test feedback with non-existent document."""
        from app.api.endpoints.feedback import submit_feedback
        
        # Mock coordinator as enabled
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = True
        
        # Mock database session
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_payload,
                coordinator=mock_coordinator,
                pdf_processor=Mock(),
                db=mock_db
            )
        
        assert exc_info.value.status_code == 404
        assert "Original document not found" in str(exc_info.value.detail)

    @patch('app.api.endpoints.feedback.api_logger')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_success(self, mock_path, mock_logger, sample_payload):
        """Test successful feedback submission."""
        from app.api.endpoints.feedback import submit_feedback
        
        # Mock coordinator as enabled
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = True
        
        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        
        # Mock database session
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        
        # Mock PDF processor
        mock_pdf_processor = Mock()
        mock_pdf_processor.extract_text_from_pdf.return_value = "Sample text content"
        
        # Mock Path operations properly
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        # Mock the / operator (__truediv__)
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_file_path = Mock()
        mock_upload_dir.glob.return_value = [mock_file_path]
        
        result = submit_feedback(
            payload=sample_payload,
            coordinator=mock_coordinator,
            pdf_processor=mock_pdf_processor,
            db=mock_db
        )
        
        assert result["message"] == "Feedback submitted successfully and is being processed."
        mock_coordinator.process_feedback_and_learn.assert_called_once()
        mock_pdf_processor.extract_text_from_pdf.assert_called_once_with(mock_file_path)

    @patch('app.api.endpoints.feedback.api_logger')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_file_not_found_on_disk(self, mock_path, mock_logger, sample_payload):
        """Test feedback when file is not found on disk."""
        from app.api.endpoints.feedback import submit_feedback
        
        # Mock coordinator as enabled
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = True
        
        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        
        # Mock database session
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        
        # Mock Path operations properly
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        # Mock the / operator
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_upload_dir.glob.return_value = []  # Empty list - no files found
        
        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_payload,
                coordinator=mock_coordinator,
                pdf_processor=Mock(),
                db=mock_db
            )
        
        assert exc_info.value.status_code == 404
        assert "not found in uploads directory" in str(exc_info.value.detail["message"])

    @patch('app.api.endpoints.feedback.api_logger')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_text_extraction_error(self, mock_path, mock_logger, sample_payload):
        """Test feedback when text extraction fails."""
        from app.api.endpoints.feedback import submit_feedback
        
        # Mock coordinator as enabled
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = True
        
        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        
        # Mock database session
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        
        # Mock PDF processor with extraction failure
        mock_pdf_processor = Mock()
        mock_pdf_processor.extract_text_from_pdf.side_effect = Exception("PDF extraction failed")
        
        # Mock Path operations properly
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        # Mock the / operator
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_file_path = Mock()
        mock_upload_dir.glob.return_value = [mock_file_path]
        
        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_payload,
                coordinator=mock_coordinator,
                pdf_processor=mock_pdf_processor,
                db=mock_db
            )
        
        assert exc_info.value.status_code == 500
        assert "PDF extraction failed" in str(exc_info.value.detail["message"])

    @patch('app.api.endpoints.feedback.api_logger')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_coordinator_error(self, mock_path, mock_logger, sample_payload):
        """Test feedback when coordinator processing fails."""
        from app.api.endpoints.feedback import submit_feedback
        
        # Mock coordinator as enabled but with processing error
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = True
        mock_coordinator.process_feedback_and_learn.side_effect = Exception("Coordinator error")
        
        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        
        # Mock database session
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        
        # Mock PDF processor
        mock_pdf_processor = Mock()
        mock_pdf_processor.extract_text_from_pdf.return_value = "Sample text"
        
        # Mock Path operations properly
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        # Mock the / operator
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_file_path = Mock()
        mock_upload_dir.glob.return_value = [mock_file_path]
        
        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_payload,
                coordinator=mock_coordinator,
                pdf_processor=mock_pdf_processor,
                db=mock_db
            )
        
        assert exc_info.value.status_code == 500
        assert "Coordinator error" in str(exc_info.value.detail["message"])

    @patch('app.api.endpoints.feedback.api_logger')
    def test_submit_feedback_document_no_filename(self, mock_logger, sample_payload):
        """Test feedback with document that has no filename."""
        from app.api.endpoints.feedback import submit_feedback
        
        # Mock coordinator as enabled
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = True
        
        # Mock database document without filename
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = None
        
        # Mock database session
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        
        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_payload,
                coordinator=mock_coordinator,
                pdf_processor=Mock(),
                db=mock_db
            )
        
        assert exc_info.value.status_code == 404
        assert "Original document not found" in str(exc_info.value.detail["message"])

    def test_feedback_type_conversion(self, sample_payload):
        """Test feedback type conversion logic."""
        # Test confirmed PII
        confirmed_item = sample_payload.feedback_items[0]  # is_correct=True
        feedback_type = FeedbackType.CONFIRMED_PII if confirmed_item.is_correct else FeedbackType.FALSE_POSITIVE
        assert feedback_type == FeedbackType.CONFIRMED_PII
        
        # Test false positive
        false_positive_item = sample_payload.feedback_items[1]  # is_correct=False
        feedback_type = FeedbackType.CONFIRMED_PII if false_positive_item.is_correct else FeedbackType.FALSE_POSITIVE
        assert feedback_type == FeedbackType.FALSE_POSITIVE


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_feedback_payload_empty_items(self):
        """Test feedback payload with empty items list."""
        payload = FeedbackPayload(document_id=456, feedback_items=[])
        assert payload.document_id == 456
        assert len(payload.feedback_items) == 0

    def test_feedback_item_special_characters(self):
        """Test feedback item with special characters."""
        item = FeedbackItem(
            text_segment="José María García-López",
            original_category="names",
            is_correct=True
        )
        assert item.text_segment == "José María García-López"

    @patch('app.api.endpoints.feedback.api_logger')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_multiple_files_found(self, mock_path, mock_logger):
        """Test feedback when multiple files match the pattern."""
        from app.api.endpoints.feedback import submit_feedback
        
        sample_payload = FeedbackPayload(
            document_id=123,
            feedback_items=[FeedbackItem(text_segment="test", original_category="names", is_correct=True)]
        )
        
        # Mock coordinator as enabled
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = True
        
        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        
        # Mock database session
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        
        # Mock PDF processor
        mock_pdf_processor = Mock()
        mock_pdf_processor.extract_text_from_pdf.return_value = "Sample text"
        
        # Mock Path operations properly
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        # Mock the / operator
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_file_path1 = Mock()
        mock_file_path2 = Mock()
        mock_upload_dir.glob.return_value = [mock_file_path1, mock_file_path2]
        
        result = submit_feedback(
            payload=sample_payload,
            coordinator=mock_coordinator,
            pdf_processor=mock_pdf_processor,
            db=mock_db
        )
        
        # Should use the first file found
        mock_pdf_processor.extract_text_from_pdf.assert_called_once_with(mock_file_path1)
        assert result["message"] == "Feedback submitted successfully and is being processed." 