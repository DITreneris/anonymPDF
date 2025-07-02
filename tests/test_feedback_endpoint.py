"""
Comprehensive tests for app.api.endpoints.feedback module.
Tests feedback submission endpoint with database mocking and error handling.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from pathlib import Path
from sqlalchemy.orm import Session

from app.api.endpoints.feedback import router, FeedbackItem, FeedbackPayload
from app.models.pdf_document import PDFDocument
from app.core.adaptive.coordinator import AdaptiveLearningCoordinator
from app.services.pdf_processor import PDFProcessor
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

    def test_feedback_item_creation_incorrect(self):
        """Test FeedbackItem model with incorrect detection."""
        item = FeedbackItem(
            text_segment="Not PII",
            original_category="names",
            is_correct=False
        )
        
        assert item.text_segment == "Not PII"
        assert item.original_category == "names"
        assert item.is_correct is False

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
        assert payload.feedback_items[0].text_segment == "John Doe"

    def test_feedback_payload_empty_items(self):
        """Test FeedbackPayload with empty feedback items."""
        payload = FeedbackPayload(
            document_id=456,
            feedback_items=[]
        )
        
        assert payload.document_id == 456
        assert len(payload.feedback_items) == 0


class TestFeedbackEndpoint:
    """Test the feedback submission endpoint."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return Mock(spec=Session)

    @pytest.fixture
    def mock_coordinator(self):
        """Mock adaptive learning coordinator."""
        coordinator = Mock(spec=AdaptiveLearningCoordinator)
        coordinator.is_enabled = True
        return coordinator

    @pytest.fixture
    def mock_pdf_processor(self):
        """Mock PDF processor."""
        processor = Mock(spec=PDFProcessor)
        processor.extract_text_from_pdf.return_value = "Sample document text content"
        return processor

    @pytest.fixture
    def sample_feedback_payload(self):
        """Sample feedback payload for testing."""
        items = [
            FeedbackItem(text_segment="John Doe", original_category="names", is_correct=True),
            FeedbackItem(text_segment="Not PII", original_category="names", is_correct=False),
            FeedbackItem(text_segment="john@example.com", original_category="emails", is_correct=True)
        ]
        
        return FeedbackPayload(
            document_id=123,
            feedback_items=items
        )

    @patch('app.api.endpoints.feedback.get_db')
    @patch('app.api.endpoints.feedback.get_adaptive_learning_coordinator')
    @patch('app.api.endpoints.feedback.get_pdf_processor')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_success(self, mock_path, mock_get_processor, mock_get_coordinator, 
                                   mock_get_db, mock_db_session, mock_coordinator, 
                                   mock_pdf_processor, sample_feedback_payload):
        """Test successful feedback submission."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_coordinator.return_value = mock_coordinator
        mock_get_processor.return_value = mock_pdf_processor

        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_document

        # Mock file path operations properly
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        mock_file_path = Mock()
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_upload_dir.glob.return_value = [mock_file_path]  # Fixed: glob on upload_dir, not constructed path

        # Import the endpoint function
        from app.api.endpoints.feedback import submit_feedback

        # Call the endpoint
        result = submit_feedback(
            payload=sample_feedback_payload,
            coordinator=mock_coordinator,
            pdf_processor=mock_pdf_processor,
            db=mock_db_session
        )

        # Assertions
        assert result["message"] == "Feedback submitted successfully and is being processed."
        mock_coordinator.process_feedback_and_learn.assert_called_once()
        mock_pdf_processor.extract_text_from_pdf.assert_called_once_with(mock_file_path)

        # Verify feedback conversion
        call_args = mock_coordinator.process_feedback_and_learn.call_args
        feedback_list = call_args[0][0]
        text_corpus = call_args[0][1]

        assert len(feedback_list) == 3
        assert feedback_list[0].text_segment == "John Doe"
        assert feedback_list[0].feedback_type == FeedbackType.CONFIRMED_PII
        assert feedback_list[1].feedback_type == FeedbackType.FALSE_POSITIVE
        assert text_corpus == ["Sample document text content"]

    @patch('app.api.endpoints.feedback.get_db')
    @patch('app.api.endpoints.feedback.get_adaptive_learning_coordinator')
    def test_submit_feedback_adaptive_learning_disabled(self, mock_get_coordinator, mock_get_db, 
                                                       mock_db_session, sample_feedback_payload):
        """Test feedback submission when adaptive learning is disabled."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = False
        mock_get_coordinator.return_value = mock_coordinator

        from app.api.endpoints.feedback import submit_feedback

        result = submit_feedback(
            payload=sample_feedback_payload,
            coordinator=mock_coordinator,
            pdf_processor=Mock(),
            db=mock_db_session
        )

        assert result["message"] == "Feedback received, but adaptive learning is disabled."
        mock_coordinator.process_feedback_and_learn.assert_not_called()

    @patch('app.api.endpoints.feedback.get_db')
    @patch('app.api.endpoints.feedback.get_adaptive_learning_coordinator')
    def test_submit_feedback_document_not_found(self, mock_get_coordinator, mock_get_db, 
                                               mock_db_session, sample_feedback_payload):
        """Test feedback submission with non-existent document."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = True
        mock_get_coordinator.return_value = mock_coordinator

        # Mock database query returning None
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        from app.api.endpoints.feedback import submit_feedback

        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_feedback_payload,
                coordinator=mock_coordinator,
                pdf_processor=Mock(),
                db=mock_db_session
            )

        assert exc_info.value.status_code == 404
        assert "Original document not found" in str(exc_info.value.detail)

    @patch('app.api.endpoints.feedback.get_db')
    @patch('app.api.endpoints.feedback.get_adaptive_learning_coordinator')
    def test_submit_feedback_document_no_filename(self, mock_get_coordinator, mock_get_db, 
                                                 mock_db_session, sample_feedback_payload):
        """Test feedback submission with document that has no filename."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_coordinator = Mock()
        mock_coordinator.is_enabled = True
        mock_get_coordinator.return_value = mock_coordinator

        # Mock database document without filename
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = None
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_document

        from app.api.endpoints.feedback import submit_feedback

        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_feedback_payload,
                coordinator=mock_coordinator,
                pdf_processor=Mock(),
                db=mock_db_session
            )

        assert exc_info.value.status_code == 404
        assert "Original document not found" in str(exc_info.value.detail)

    @patch('app.api.endpoints.feedback.get_db')
    @patch('app.api.endpoints.feedback.get_adaptive_learning_coordinator')
    @patch('app.api.endpoints.feedback.get_pdf_processor')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_file_not_found_on_disk(self, mock_path, mock_get_processor, 
                                                   mock_get_coordinator, mock_get_db, 
                                                   mock_db_session, mock_coordinator, 
                                                   mock_pdf_processor, sample_feedback_payload):
        """Test feedback submission when file is not found on disk."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_coordinator.return_value = mock_coordinator
        mock_get_processor.return_value = mock_pdf_processor

        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_document

        # Mock file path operations - no files found
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_upload_dir.glob.return_value = []  # Empty list - no files found

        from app.api.endpoints.feedback import submit_feedback

        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_feedback_payload,
                coordinator=mock_coordinator,
                pdf_processor=mock_pdf_processor,
                db=mock_db_session
            )

        assert exc_info.value.status_code == 404
        assert "not found in uploads directory" in str(exc_info.value.detail["message"])

    @patch('app.api.endpoints.feedback.get_db')
    @patch('app.api.endpoints.feedback.get_adaptive_learning_coordinator')
    @patch('app.api.endpoints.feedback.get_pdf_processor')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_text_extraction_error(self, mock_path, mock_get_processor, 
                                                   mock_get_coordinator, mock_get_db, 
                                                   mock_db_session, mock_coordinator, 
                                                   mock_pdf_processor, sample_feedback_payload):
        """Test feedback submission when text extraction fails."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_coordinator.return_value = mock_coordinator
        mock_get_processor.return_value = mock_pdf_processor

        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_document

        # Mock file path operations properly
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        mock_file_path = Mock()
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_upload_dir.glob.return_value = [mock_file_path]  # Fixed: glob on upload_dir, not constructed path

        # Mock text extraction failure
        mock_pdf_processor.extract_text_from_pdf.side_effect = Exception("PDF extraction failed")

        from app.api.endpoints.feedback import submit_feedback

        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_feedback_payload,
                coordinator=mock_coordinator,
                pdf_processor=mock_pdf_processor,
                db=mock_db_session
            )

        assert exc_info.value.status_code == 500
        assert "PDF extraction failed" in str(exc_info.value.detail["message"])

    @patch('app.api.endpoints.feedback.get_db')
    @patch('app.api.endpoints.feedback.get_adaptive_learning_coordinator')
    @patch('app.api.endpoints.feedback.get_pdf_processor')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_coordinator_processing_error(self, mock_path, mock_get_processor, 
                                                         mock_get_coordinator, mock_get_db, 
                                                         mock_db_session, mock_coordinator, 
                                                         mock_pdf_processor, sample_feedback_payload):
        """Test feedback submission when coordinator processing fails."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_coordinator.return_value = mock_coordinator
        mock_get_processor.return_value = mock_pdf_processor

        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_document

        # Mock file path operations properly
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        mock_file_path = Mock()
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_upload_dir.glob.return_value = [mock_file_path]  # Fixed: glob on upload_dir, not constructed path

        # Mock coordinator processing failure
        mock_coordinator.process_feedback_and_learn.side_effect = Exception("Processing failed")

        from app.api.endpoints.feedback import submit_feedback

        with pytest.raises(HTTPException) as exc_info:
            submit_feedback(
                payload=sample_feedback_payload,
                coordinator=mock_coordinator,
                pdf_processor=mock_pdf_processor,
                db=mock_db_session
            )

        assert exc_info.value.status_code == 500
        assert "Processing failed" in str(exc_info.value.detail["message"])

    def test_feedback_type_conversion_confirmed_pii(self, sample_feedback_payload):
        """Test feedback type conversion for confirmed PII."""
        item = sample_feedback_payload.feedback_items[0]  # is_correct=True
        
        feedback_type = FeedbackType.CONFIRMED_PII if item.is_correct else FeedbackType.FALSE_POSITIVE
        
        assert feedback_type == FeedbackType.CONFIRMED_PII

    def test_feedback_type_conversion_false_positive(self, sample_feedback_payload):
        """Test feedback type conversion for false positive."""
        item = sample_feedback_payload.feedback_items[1]  # is_correct=False
        
        feedback_type = FeedbackType.CONFIRMED_PII if item.is_correct else FeedbackType.FALSE_POSITIVE
        
        assert feedback_type == FeedbackType.FALSE_POSITIVE

    @patch('app.api.endpoints.feedback.get_db')
    @patch('app.api.endpoints.feedback.get_adaptive_learning_coordinator') 
    @patch('app.api.endpoints.feedback.get_pdf_processor')
    def test_submit_feedback_with_logging(self, mock_get_processor, mock_get_coordinator, 
                                         mock_get_db, mock_db_session, mock_coordinator, 
                                         mock_pdf_processor, sample_feedback_payload):
        """Test that feedback submission logs appropriately."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_coordinator.return_value = mock_coordinator
        mock_get_processor.return_value = mock_pdf_processor

        with patch('app.api.endpoints.feedback.api_logger') as mock_logger:
            mock_coordinator.is_enabled = False  # Trigger disabled path
            
            from app.api.endpoints.feedback import submit_feedback

            result = submit_feedback(
                payload=sample_feedback_payload,
                coordinator=mock_coordinator,
                pdf_processor=mock_pdf_processor,
                db=mock_db_session
            )

            # Verify logging calls
            mock_logger.info.assert_called()
            mock_logger.warning.assert_called_with("Adaptive learning system is disabled. Feedback will not be processed.")

    def test_user_feedback_object_creation(self, sample_feedback_payload):
        """Test UserFeedback object creation from payload."""
        item = sample_feedback_payload.feedback_items[0]
        
        feedback = UserFeedback(
            feedback_id=f"feedback_test_123_{hash(item.text_segment)}",
            document_id=str(sample_feedback_payload.document_id),
            text_segment=item.text_segment,
            detected_category=item.original_category,  # Fixed parameter name
            user_corrected_category=None,
            detected_confidence=0.5,  # Added required field
            user_confidence_rating=None,
            feedback_type=FeedbackType.CONFIRMED_PII if item.is_correct else FeedbackType.FALSE_POSITIVE,
            severity=FeedbackSeverity.MEDIUM,  # Added required field
            user_comment=None,
            context={}  # Added required field
        )
        
        assert feedback.text_segment == "John Doe"
        assert feedback.feedback_type == FeedbackType.CONFIRMED_PII
        assert feedback.detected_category == "names"
        assert feedback.document_id == "123"

    @patch('app.api.endpoints.feedback.get_db')
    @patch('app.api.endpoints.feedback.get_adaptive_learning_coordinator')
    @patch('app.api.endpoints.feedback.get_pdf_processor')
    @patch('app.api.endpoints.feedback.Path')
    def test_submit_feedback_with_multiple_files_found(self, mock_path, mock_get_processor, 
                                                      mock_get_coordinator, mock_get_db, 
                                                      mock_db_session, mock_coordinator, 
                                                      mock_pdf_processor, sample_feedback_payload):
        """Test feedback submission when multiple files match the pattern."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_coordinator.return_value = mock_coordinator
        mock_get_processor.return_value = mock_pdf_processor

        # Mock database document
        mock_document = Mock()
        mock_document.id = 123
        mock_document.original_filename = "test_document.pdf"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_document

        # Mock file path operations - multiple files found
        mock_upload_dir = Mock()
        mock_path.return_value = mock_upload_dir
        mock_file_path1 = Mock()
        mock_file_path2 = Mock()
        mock_file_path_constructed = Mock()
        mock_upload_dir.__truediv__ = Mock(return_value=mock_file_path_constructed)
        mock_upload_dir.glob.return_value = [mock_file_path1, mock_file_path2]  # Fixed: glob on upload_dir

        from app.api.endpoints.feedback import submit_feedback

        result = submit_feedback(
            payload=sample_feedback_payload,
            coordinator=mock_coordinator,
            pdf_processor=mock_pdf_processor,
            db=mock_db_session
        )

        # Should use the first file found
        mock_pdf_processor.extract_text_from_pdf.assert_called_once_with(mock_file_path1)
        assert result["message"] == "Feedback submitted successfully and is being processed."


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_feedback_payload_with_single_item(self):
        """Test feedback payload with single feedback item."""
        item = FeedbackItem(text_segment="single@example.com", original_category="emails", is_correct=True)
        payload = FeedbackPayload(document_id=789, feedback_items=[item])
        
        assert len(payload.feedback_items) == 1
        assert payload.feedback_items[0].text_segment == "single@example.com"

    def test_feedback_item_with_special_characters(self):
        """Test feedback item with special characters."""
        item = FeedbackItem(
            text_segment="Jürgën Müller-Småłł",
            original_category="names",
            is_correct=True
        )
        
        assert item.text_segment == "Jürgën Müller-Småłł"
        assert item.original_category == "names"

    def test_feedback_item_with_long_text(self):
        """Test feedback item with very long text segment."""
        long_text = "A" * 1000  # Very long text
        item = FeedbackItem(
            text_segment=long_text,
            original_category="custom",
            is_correct=False
        )
        
        assert len(item.text_segment) == 1000
        assert item.is_correct is False 