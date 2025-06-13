"""
Tests for UserFeedbackSystem

This file contains tests for the user feedback collection, processing, and
integration with the ML training pipeline.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid

from app.core.feedback_system import (
    UserFeedbackProcessor,
    UserFeedback,
    FeedbackType,
    FeedbackSeverity,
    create_feedback_system,
)
from app.core.data_models import TrainingExample

@pytest.fixture
def mock_processor():
    """Provides a mocked UserFeedbackProcessor."""
    processor = MagicMock(spec=UserFeedbackProcessor)
    processor.process_pending_feedback.return_value = {
        'processed_count': 1, 'training_examples_generated': 1
    }
    return processor

@patch('app.core.feedback_system.UserFeedbackProcessor')
def test_feedback_system_initialization(MockUserFeedbackProcessor, tmp_path):
    """Test that the feedback system initializes its processor correctly."""
    mock_processor_instance = MockUserFeedbackProcessor.return_value
    db_path = tmp_path / "test.db"
    system = create_feedback_system(config={'use_background_thread': False, 'storage_path': str(db_path)})
    
    assert system.processor == mock_processor_instance
    MockUserFeedbackProcessor.assert_called_once()

@patch('app.core.feedback_system.UserFeedbackProcessor')
def test_feedback_submission_and_processing(MockUserFeedbackProcessor, tmp_path):
    """Test the submission flow, ensuring it reaches the processor."""
    mock_processor_instance = MockUserFeedbackProcessor.return_value
    db_path = tmp_path / "test.db"
    
    # Setup a system with background processing disabled for deterministic testing
    system = create_feedback_system(config={'use_background_thread': False, 'storage_path': str(db_path)})
    
    feedback = MagicMock(spec=UserFeedback)
    
    # The system's submit method should call the processor's submit method
    system.submit_feedback(feedback)
    mock_processor_instance.submit_feedback.assert_called_once_with(feedback)
    
    # Manually trigger processing and check that the processor's method is called
    system.processor.process_pending_feedback()
    mock_processor_instance.process_pending_feedback.assert_called_once()

@pytest.fixture
def mock_training_data_collector():
    """Mock TrainingDataCollector and its storage."""
    mock_storage = MagicMock()
    mock_collector = MagicMock()
    mock_collector.storage = mock_storage
    return mock_collector

@pytest.fixture
def feedback_processor(mock_training_data_collector):
    """Provides a UserFeedbackProcessor with a mocked collector."""
    config = {
        'auto_retrain_threshold': 5,
        'analyzer': {'min_feedback_threshold': 1} # Lower threshold for testing
    }
    processor = UserFeedbackProcessor(config=config)
    processor.training_data_collector = mock_training_data_collector
    return processor

def create_dummy_feedback(feedback_type: FeedbackType, detected_category: str = "NAME") -> UserFeedback:
    """Helper to create UserFeedback objects for tests."""
    return UserFeedback(
        feedback_id=str(uuid.uuid4()),
        document_id="doc_123",
        text_segment="John Doe",
        detected_category=detected_category,
        user_corrected_category=None,
        detected_confidence=0.85,
        user_confidence_rating=None,
        feedback_type=feedback_type,
        severity=FeedbackSeverity.MEDIUM,
        user_comment="Test comment",
        context={"surrounding_text": "The person John Doe was here."}
    )

@pytest.mark.unit
class TestUserFeedbackProcessor:
    """Tests for the UserFeedbackProcessor component."""

    def test_submit_and_process_feedback(self, feedback_processor, mock_training_data_collector):
        """Test processing of pending feedback converts to TrainingExamples."""
        feedback = create_dummy_feedback(FeedbackType.CORRECT_DETECTION)
        
        dummy_example = TrainingExample(
            detection_text="John Doe", category="NAME", context="The person John Doe was here.",
            features={"length": 8}, confidence_score=0.95, is_true_positive=True
        )

        with patch.object(feedback_processor, '_feedback_to_training_example', return_value=dummy_example) as mock_converter:
            feedback_processor.submit_feedback(feedback)
            result = feedback_processor.process_pending_feedback()

            mock_converter.assert_called_once_with(feedback)
            
            # Verify the correct method on the storage mock is called
            mock_training_data_collector.storage.save_training_examples.assert_called_once()
            call_args = mock_training_data_collector.storage.save_training_examples.call_args[0]
            assert call_args[0] == [dummy_example] # examples
            assert call_args[1] == 'user_feedback' # source

            assert result['processed_count'] == 1
            assert result['training_examples_created'] == 1

@pytest.mark.integration
class TestFeedbackSystemIntegration:
    """Integration tests for the complete feedback system."""

    def test_feedback_submission_flow_is_deterministic(self, tmp_path):
        """Test the full flow from submission to processing without a background thread."""
        db_path = tmp_path / "test.db"
        test_config = {
            'use_background_thread': False,
            'storage_path': str(db_path),
            'processor': {'auto_retrain_threshold': 5}
        }
        feedback_system = create_feedback_system(config=test_config)
        feedback = create_dummy_feedback(FeedbackType.FALSE_POSITIVE, "ADDRESS")

        with patch.object(feedback_system.processor.training_data_collector, 'storage', autospec=True) as mock_storage:
            feedback_system.submit_feedback(feedback)
            feedback_system.processor.process_pending_feedback()

            mock_storage.save_training_examples.assert_called_once()
            call_args, call_kwargs = mock_storage.save_training_examples.call_args
            
            # call_args[0] is the list of examples
            assert isinstance(call_args[0][0], TrainingExample)
            assert call_args[0][0].is_true_positive is False
            # call_args[1] is the source
            assert call_args[1] == 'user_feedback'


if __name__ == '__main__':
    pytest.main([__file__]) 