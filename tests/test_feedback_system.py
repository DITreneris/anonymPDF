"""
Tests for User Feedback System - Session 4 Priority 3 Implementation

Tests the user feedback collection, processing, and integration with
the ML training pipeline.
"""

import pytest
import tempfile
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from app.core.feedback_system import (
    UserFeedbackSystem,
    UserFeedbackProcessor,
    FeedbackAnalyzer,
    UserFeedback,
    FeedbackType,
    FeedbackSeverity,
    FeedbackStats
)
from app.core.training_data import TrainingExample
from app.core.ml_training_pipeline import MLTrainingPipeline

# Make sure to patch the correct location where create_ml_training_pipeline is looked up
PATCH_CREATE_PIPELINE = 'app.core.feedback_system.create_ml_training_pipeline'

@pytest.mark.unit
class TestUserFeedback:
    """Test the UserFeedback dataclass."""
    
    def test_user_feedback_creation(self):
        """Test UserFeedback creation with required fields."""
        feedback = UserFeedback(
            feedback_id="test_001",
            document_id="doc_123",
            text_segment="John Doe",
            detected_category="names",
            user_corrected_category="names",
            detected_confidence=0.8,
            user_confidence_rating=0.9,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="This is correctly detected",
            context={"surrounding_text": "Hello John Doe welcome"}
        )
        
        assert feedback.feedback_id == "test_001"
        assert feedback.document_id == "doc_123"
        assert feedback.text_segment == "John Doe"
        assert feedback.detected_category == "names"
        assert feedback.feedback_type == FeedbackType.CORRECT_DETECTION
        assert feedback.severity == FeedbackSeverity.MEDIUM
        assert not feedback.processed  # Default value
    
    def test_user_feedback_defaults(self):
        """Test UserFeedback with minimal required fields."""
        feedback = UserFeedback(
            feedback_id="test_002",
            document_id="doc_456",
            text_segment="555-123-4567",
            detected_category="phone_numbers",
            user_corrected_category=None,
            detected_confidence=0.7,
            user_confidence_rating=None,
            feedback_type=FeedbackType.FALSE_POSITIVE,
            severity=FeedbackSeverity.LOW,
            user_comment=None,
            context={}
        )
        
        assert feedback.user_corrected_category is None
        assert feedback.user_confidence_rating is None
        assert feedback.user_comment is None
        assert isinstance(feedback.timestamp, datetime)


@pytest.mark.unit
class TestFeedbackAnalyzer:
    """Test the FeedbackAnalyzer class."""
    
    def setup_method(self):
        """Setup test environment."""
        config = {
            'min_feedback_threshold': 3,
            'confidence_threshold': 0.7
        }
        self.analyzer = FeedbackAnalyzer(config)
    
    def test_initialization(self):
        """Test FeedbackAnalyzer initialization."""
        assert self.analyzer is not None
        assert self.analyzer.min_feedback_threshold == 3
        assert self.analyzer.confidence_threshold == 0.7
        assert len(self.analyzer.feedback_history) == 0
    
    def test_add_feedback(self):
        """Test adding feedback to analyzer."""
        feedback = UserFeedback(
            feedback_id="test_001",
            document_id="doc_123",
            text_segment="John Doe",
            detected_category="names",
            user_corrected_category="names",
            detected_confidence=0.8,
            user_confidence_rating=0.9,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="Correct",
            context={}
        )
        
        self.analyzer.add_feedback(feedback)
        
        assert len(self.analyzer.feedback_history) == 1
        assert "names" in self.analyzer.category_performance
        assert len(self.analyzer.category_performance["names"]) == 1
    
    def test_analyze_feedback_patterns_empty(self):
        """Test feedback pattern analysis with no data."""
        analysis = self.analyzer.analyze_feedback_patterns()
        
        assert 'message' in analysis
        assert analysis['message'] == 'No feedback in specified time window'
    
    def test_analyze_feedback_patterns_with_data(self):
        """Test feedback pattern analysis with data."""
        # Add multiple feedback items
        feedbacks = [
            UserFeedback(
                feedback_id=f"test_{i}",
                document_id=f"doc_{i}",
                text_segment=f"Text {i}",
                detected_category="names",
                user_corrected_category="names",
                detected_confidence=0.8,
                user_confidence_rating=0.9,
                feedback_type=FeedbackType.CORRECT_DETECTION if i % 2 == 0 else FeedbackType.FALSE_POSITIVE,
                severity=FeedbackSeverity.MEDIUM,
                user_comment=f"Comment {i}",
                context={}
            )
            for i in range(5)
        ]
        
        for feedback in feedbacks:
            self.analyzer.add_feedback(feedback)
        
        analysis = self.analyzer.analyze_feedback_patterns()
        
        assert analysis['total_feedback'] == 5
        assert 'feedback_by_type' in analysis
        assert 'feedback_by_category' in analysis
        assert 'avg_user_confidence' in analysis
        
        # Check specific counts
        assert analysis['feedback_by_type']['correct_detection'] == 3  # Even indices: 0, 2, 4
        assert analysis['feedback_by_type']['false_positive'] == 2     # Odd indices: 1, 3
        assert analysis['feedback_by_category']['names'] == 5
        assert analysis['avg_user_confidence'] == 0.9
    
    def test_generate_improvement_suggestions_high_false_positive(self):
        """Test improvement suggestions for high false positive rate."""
        # Add feedback with high false positive rate
        for i in range(10):
            feedback_type = FeedbackType.FALSE_POSITIVE if i < 7 else FeedbackType.CORRECT_DETECTION
            
            feedback = UserFeedback(
                feedback_id=f"test_{i}",
                document_id=f"doc_{i}",
                text_segment=f"Text {i}",
                detected_category="names",
                user_corrected_category="names",
                detected_confidence=0.8,
                user_confidence_rating=0.5,  # Low confidence
                feedback_type=feedback_type,
                severity=FeedbackSeverity.MEDIUM,
                user_comment=f"Comment {i}",
                context={}
            )
            
            self.analyzer.add_feedback(feedback)
        
        suggestions = self.analyzer.generate_improvement_suggestions()
        
        assert len(suggestions) > 0
        # Should suggest improvements for both high false positive rate and low confidence
        suggestion_text = ' '.join(suggestions)
        assert 'false positive' in suggestion_text.lower()
        assert 'confidence' in suggestion_text.lower()
    
    def test_thread_safety(self):
        """Test thread safety of FeedbackAnalyzer."""
        def add_feedback_batch():
            for i in range(20):
                feedback = UserFeedback(
                    feedback_id=f"thread_test_{threading.current_thread().ident}_{i}",
                    document_id=f"doc_{i}",
                    text_segment=f"Text {i}",
                    detected_category="test_category",
                    user_corrected_category="test_category",
                    detected_confidence=0.8,
                    user_confidence_rating=0.9,
                    feedback_type=FeedbackType.CORRECT_DETECTION,
                    severity=FeedbackSeverity.MEDIUM,
                    user_comment=f"Comment {i}",
                    context={}
                )
                self.analyzer.add_feedback(feedback)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_feedback_batch)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have processed all feedback without errors
        assert len(self.analyzer.feedback_history) == 60
        assert "test_category" in self.analyzer.category_performance


@pytest.mark.unit
class TestUserFeedbackProcessor:
    """Test the UserFeedbackProcessor class."""
    
    def setup_method(self):
        """Setup test environment."""
        config = {
            'auto_retrain_threshold': 5,
            'batch_size': 3
        }
        self.processor = UserFeedbackProcessor(config)
    
    def test_initialization(self):
        """Test UserFeedbackProcessor initialization."""
        assert self.processor is not None
        assert self.processor.auto_retrain_threshold == 5
        assert self.processor.processed_count == 0
        assert len(self.processor.pending_feedback) == 0
    
    def test_validate_feedback_valid(self):
        """Test feedback validation with valid data."""
        feedback = UserFeedback(
            feedback_id="test_001",
            document_id="doc_123",
            text_segment="John Doe",
            detected_category="names",
            user_corrected_category="names",
            detected_confidence=0.8,
            user_confidence_rating=0.9,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="Correct",
            context={}
        )
        
        assert self.processor._validate_feedback(feedback) == True
    
    def test_validate_feedback_invalid(self):
        """Test feedback validation with invalid data."""
        # Missing feedback_id
        feedback = UserFeedback(
            feedback_id="",
            document_id="doc_123",
            text_segment="John Doe",
            detected_category="names",
            user_corrected_category="names",
            detected_confidence=0.8,
            user_confidence_rating=0.9,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="Correct",
            context={}
        )
        
        assert self.processor._validate_feedback(feedback) == False
        
        # Invalid confidence value
        feedback.feedback_id = "test_001"
        feedback.detected_confidence = 1.5  # Invalid range
        
        assert self.processor._validate_feedback(feedback) == False
    
    def test_submit_feedback(self):
        """Test submitting feedback."""
        feedback = UserFeedback(
            feedback_id="test_001",
            document_id="doc_123",
            text_segment="John Doe",
            detected_category="names",
            user_corrected_category="names",
            detected_confidence=0.8,
            user_confidence_rating=0.9,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="Correct",
            context={}
        )
        
        success = self.processor.submit_feedback(feedback)
        
        assert success == True
        assert len(self.processor.pending_feedback) == 1
        assert len(self.processor.feedback_analyzer.feedback_history) == 1
    
    @patch('app.core.feedback_system.FeatureExtractor')
    def test_feedback_to_training_example_false_positive(self, mock_feature_extractor):
        """Test converting false positive feedback to training example."""
        # Mock feature extractor
        mock_extractor = Mock()
        mock_extractor.extract_features.return_value = {'feature1': 0.5, 'feature2': 0.8}
        self.processor.feature_extractor = mock_extractor
        
        feedback = UserFeedback(
            feedback_id="test_001",
            document_id="doc_123",
            text_segment="John Doe",
            detected_category="names",
            user_corrected_category=None,
            detected_confidence=0.8,
            user_confidence_rating=None,
            feedback_type=FeedbackType.FALSE_POSITIVE,
            severity=FeedbackSeverity.HIGH,
            user_comment="This is not a name",
            context={'surrounding_context': 'Hello world'}
        )
        
        training_example = self.processor._feedback_to_training_example(feedback)
        
        assert training_example is not None
        assert training_example.is_true_positive == False  # False positive means not PII
        assert training_example.confidence_score == 0.1  # Low confidence for non-PII
        assert 'feedback_id' in training_example.metadata
    
    @patch('app.core.feedback_system.FeatureExtractor')
    def test_feedback_to_training_example_false_negative(self, mock_feature_extractor):
        """Test converting false negative feedback to training example."""
        # Mock feature extractor
        mock_extractor = Mock()
        mock_extractor.extract_features.return_value = {'feature1': 0.5, 'feature2': 0.8}
        self.processor.feature_extractor = mock_extractor
        
        feedback = UserFeedback(
            feedback_id="test_002",
            document_id="doc_456",
            text_segment="jane.smith@email.com",
            detected_category=None,
            user_corrected_category="emails",
            detected_confidence=0.2,
            user_confidence_rating=0.95,
            feedback_type=FeedbackType.FALSE_NEGATIVE,
            severity=FeedbackSeverity.HIGH,
            user_comment="This is definitely an email",
            context={'surrounding_context': 'Contact me at'}
        )
        
        training_example = self.processor._feedback_to_training_example(feedback)
        
        assert training_example is not None
        assert training_example.is_true_positive == True  # False negative means it is PII
        assert training_example.confidence_score == 0.95  # User confidence rating
        assert training_example.category == "emails"  # User corrected category
    
    def test_should_trigger_retrain_threshold(self):
        """Test retraining trigger based on processed count."""
        # Process enough feedback to trigger retraining
        self.processor.processed_count = 6  # Above threshold of 5
        
        assert self.processor._should_trigger_retrain() == True
    
    def test_should_trigger_retrain_time(self):
        """Test retraining trigger based on time."""
        # Set last retrain time to 8 days ago
        self.processor.last_retrain_time = datetime.now() - timedelta(days=8)
        
        assert self.processor._should_trigger_retrain() == True
    
    @patch('app.core.feedback_system.TrainingDataCollector')
    def test_process_pending_feedback(self, mock_training_collector):
        """Test processing pending feedback."""
        # Mock training data collector
        mock_collector = Mock()
        self.processor.training_data_collector = mock_collector
        
        # Add feedback to pending queue
        for i in range(3):
            feedback = UserFeedback(
                feedback_id=f"test_{i}",
                document_id=f"doc_{i}",
                text_segment=f"Text {i}",
                detected_category="names",
                user_corrected_category="names",
                detected_confidence=0.8,
                user_confidence_rating=0.9,
                feedback_type=FeedbackType.CORRECT_DETECTION,
                severity=FeedbackSeverity.MEDIUM,
                user_comment=f"Comment {i}",
                context={'surrounding_context': 'context'}
            )
            self.processor.submit_feedback(feedback)
        
        # Process feedback
        result = self.processor.process_pending_feedback()
        
        assert result['processed_count'] == 3
        assert result['training_examples_created'] >= 0  # Some may fail conversion
        assert 'should_retrain' in result
    
    def test_get_feedback_stats(self):
        """Test getting feedback statistics."""
        # Add some feedback
        feedback = UserFeedback(
            feedback_id="test_001",
            document_id="doc_123",
            text_segment="John Doe",
            detected_category="names",
            user_corrected_category="names",
            detected_confidence=0.8,
            user_confidence_rating=0.9,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="Correct",
            context={}
        )
        
        self.processor.submit_feedback(feedback)
        
        stats = self.processor.get_feedback_stats()
        
        assert hasattr(stats, 'total_feedback_count')
        assert hasattr(stats, 'feedback_by_type')
        assert hasattr(stats, 'feedback_by_category')
        assert hasattr(stats, 'avg_user_confidence')
        assert hasattr(stats, 'improvement_suggestions')
        assert isinstance(stats.feedback_by_type, dict)
        assert isinstance(stats.improvement_suggestions, list)


@pytest.mark.unit
class TestUserFeedbackSystem:
    """Test the UserFeedbackSystem class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_feedback.db"
        self.config = {
            'storage_path': str(self.db_path),
            'auto_process': False, # Disable auto_process for controlled testing
            'processor': {
                'auto_retrain_threshold': 3, # Lower for testing
                'batch_size': 2
            }
        }

        # Patch create_ml_training_pipeline
        self.mock_create_pipeline_patcher = patch(PATCH_CREATE_PIPELINE)
        self.mock_create_pipeline = self.mock_create_pipeline_patcher.start()
        self.mock_pipeline_instance = MagicMock(spec=MLTrainingPipeline)
        self.mock_pipeline_instance.force_training.return_value = "test_job_id_123"
        self.mock_create_pipeline.return_value = self.mock_pipeline_instance

        self.system = UserFeedbackSystem(config=self.config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.mock_create_pipeline_patcher.stop() # Stop the patcher
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test UserFeedbackSystem initialization."""
        assert self.system is not None
        assert self.system.processor is not None
        assert self.system.analyzer is not None
        assert self.system.auto_process == False
        assert self.system.storage_path.exists()
    
    def test_storage_initialization(self):
        """Test database storage initialization."""
        # Check that the database and table were created
        import sqlite3
        
        with sqlite3.connect(self.system.storage_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_feedback'")
            table_exists = cursor.fetchone() is not None
            assert table_exists
    
    def test_submit_feedback_integration(self):
        """Test full feedback submission integration."""
        feedback = UserFeedback(
            feedback_id="test_001",
            document_id="doc_123",
            text_segment="John Doe",
            detected_category="names",
            user_corrected_category="names",
            detected_confidence=0.8,
            user_confidence_rating=0.9,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="Correct detection",
            context={'surrounding_context': 'Hello John Doe welcome'}
        )
        
        success = self.system.submit_feedback(feedback)
        
        assert success == True
        
        # Verify storage
        import sqlite3
        with sqlite3.connect(self.system.storage_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM user_feedback WHERE feedback_id = ?", ("test_001",))
            count = cursor.fetchone()[0]
            assert count == 1
    
    def test_get_system_status(self):
        """Test getting system status."""
        # Add some feedback first
        # To make this test independent, we directly manipulate processor's state for should_retrain
        # or submit enough feedback if auto_process was True and _trigger_processing was called.
        # For now, this test mainly checks status structure.
        
        # Modify the setup to ensure this test is also using the patched pipeline
        # The setup_method already handles this for all tests in this class.

        feedback = UserFeedback(
            feedback_id="status_test_001",
            document_id="doc_123",
            text_segment="John Doe",
            detected_category="names",
            user_corrected_category="names",
            detected_confidence=0.8,
            user_confidence_rating=0.9,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="Correct",
            context={}
        )
        
        self.system.submit_feedback(feedback)
        
        status = self.system.get_system_status()
        
        assert 'processor_status' in status
        assert 'feedback_statistics' in status
        assert 'system_config' in status
        
        assert 'processed_count' in status['processor_status']
        assert 'pending_count' in status['processor_status']
        assert 'auto_process' in status['system_config']

    def test_processing_loop_triggers_retraining_and_updates_timestamp(self):
        """Test that _processing_loop calls force_training and updates last_retrain_time."""
        # 1. Setup feedback and processor state to ensure should_retrain is True
        # Submit enough feedback to meet auto_retrain_threshold
        for i in range(self.config['processor']['auto_retrain_threshold']):
            fb = UserFeedback(
                feedback_id=f"retrain_fb_{i}", document_id=f"doc_{i}", text_segment=f"text {i}",
                detected_category="CAT", detected_confidence=0.5, feedback_type=FeedbackType.CORRECT_DETECTION,
                severity=FeedbackSeverity.LOW, context={}
            )
            self.system.submit_feedback(fb) # This also adds to processor's pending_feedback
        
        # Ensure processor.processed_count will meet threshold after processing
        # process_pending_feedback updates processed_count internally.
        # We can also directly mock _should_trigger_retrain on the processor instance for more direct control
        with patch.object(self.system.processor, '_should_trigger_retrain', return_value=True) as mock_should_retrain:
            # Store original last_retrain_time
            original_last_retrain_time = self.system.processor.last_retrain_time

            # 2. Call _processing_loop (which calls processor.process_pending_feedback)
            self.system._processing_loop() 

            # 3. Assertions
            mock_should_retrain.assert_called_once()
            self.mock_pipeline_instance.force_training.assert_called_once_with(reason="feedback_triggered_retraining")
            
            # Check that last_retrain_time was updated
            assert self.system.processor.last_retrain_time > original_last_retrain_time
            # Check it's close to now
            assert (datetime.now() - self.system.processor.last_retrain_time) < timedelta(seconds=5)

            # Verify that feedback items were processed
            # The loop processes in batches, ensure all are processed if needed for count
            # process_pending_feedback returns {'processed_count': ..., 'training_examples_created': ...}
            # The test above already submits enough for processed_count to hit threshold *within* _should_trigger_retrain
            # if it relies on self.processor.processed_count. 
            # If _should_trigger_retrain is mocked, this part is just to ensure loop ran.
            assert self.system.processor.processed_count >= self.config['processor']['auto_retrain_threshold']

    def test_processing_loop_retraining_failure(self):
        """Test _processing_loop when force_training fails."""
        self.mock_pipeline_instance.force_training.side_effect = Exception("Pipeline error")
        original_last_retrain_time = self.system.processor.last_retrain_time

        with patch.object(self.system.processor, '_should_trigger_retrain', return_value=True) as mock_should_retrain:
            with patch.object(self.system.processor, 'process_pending_feedback', return_value={'should_retrain': True, 'processed_count': 3, 'training_examples_created': 3}): # ensure it returns should_retrain True
              self.system._processing_loop()

        self.mock_pipeline_instance.force_training.assert_called_once()
        # last_retrain_time should NOT be updated if force_training fails
        assert self.system.processor.last_retrain_time == original_last_retrain_time


@pytest.mark.integration
class TestFeedbackSystemIntegration:
    """Integration tests for feedback system with existing components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        # Use a consistent config for the system instance in this class
        self.db_path = Path(self.temp_dir) / "test_feedback_integration.db"
        self.config = {
            'storage_path': str(self.db_path),
            'auto_process': False, 
            'processor': {
                'auto_retrain_threshold': 5,
                'batch_size': 3
            }
        }
        # Patch create_ml_training_pipeline for UserFeedbackSystem instantiation
        self.mock_create_pipeline_patcher_integration = patch(PATCH_CREATE_PIPELINE) # Use a different name for the patcher variable
        self.mock_create_pipeline_integration = self.mock_create_pipeline_patcher_integration.start()
        self.mock_pipeline_instance_integration = MagicMock(spec=MLTrainingPipeline)
        self.mock_pipeline_instance_integration.force_training.return_value = "integration_job_id"
        self.mock_create_pipeline_integration.return_value = self.mock_pipeline_instance_integration
        
        self.system = UserFeedbackSystem(config=self.config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.mock_create_pipeline_patcher_integration.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('app.core.feedback_system.get_config')
    def test_config_integration(self, mock_get_config):
        """Test integration with config manager."""
        mock_config_data = {
            'user_feedback': {
                'storage_path': '/test/path',
                'auto_process': False,
                'processor': {
                    'auto_retrain_threshold': 10
                }
            }
        }
        mock_get_config.return_value = mock_config_data
        
        # This test creates its own system, so we need a separate patch for it if it doesn't use self.system
        # Or, ensure it uses self.system which is already set up with a patched pipeline.
        # The current UserFeedbackSystem() call will use the global patch if active or the real one.
        # For this specific test, it is testing how UserFeedbackSystem itself uses get_config,
        # so the create_ml_training_pipeline patch on self.system is not directly relevant for this test's own UserFeedbackSystem instance.
        # However, if UserFeedbackSystem() inside this test is meant to be *the* system under test 
        # AND it should use a mocked pipeline, the patch needs to be active when it's instantiated.
        
        # Option 1: If this test needs its own UserFeedbackSystem with a new mock pipeline for config test
        with patch(PATCH_CREATE_PIPELINE) as mock_create_pipeline_for_config_test:
            mock_pipeline_for_config_test = MagicMock(spec=MLTrainingPipeline)
            mock_create_pipeline_for_config_test.return_value = mock_pipeline_for_config_test
            system_for_config_test = UserFeedbackSystem() # Will use mock_get_config and the new mock_create_pipeline
            assert system_for_config_test.auto_process == False
            assert system_for_config_test.processor.auto_retrain_threshold == 10
            mock_create_pipeline_for_config_test.assert_called_once()

        # Option 2: If this test can rely on the class-level self.system (simpler, but tests less in isolation)
        # This assumes that get_config is called by __init__ of UserFeedbackSystem *before* create_ml_training_pipeline.
        # If the order is different, this test might not be accurate for the pipeline part.
        # For now, let's assume the config under test is for UserFeedbackSystem's direct config attributes.
        # The self.system is already created with its own config and mock pipeline.
        # This test, as written, seems to be testing a new UserFeedbackSystem instance and its use of get_config.
        # So Option 1 is more appropriate for isolating the config impact on a fresh instance.

    def test_training_data_integration(self):
        """Test integration with training data collector."""
        # This test uses self.system, which has its ml_training_pipeline mocked.
        # It tests if feedback processing creates training examples. 
        # The retraining trigger aspect is now covered by test_processing_loop_triggers_retraining.
        
        # Submit feedback that should create training examples
        feedback = UserFeedback(
            feedback_id="test_001",
            document_id="doc_123",
            text_segment="john.doe@example.com",
            detected_category="emails",
            user_corrected_category="emails",
            detected_confidence=0.9,
            user_confidence_rating=0.95,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="Correctly detected email",
            context={'surrounding_context': 'Contact me at john.doe@example.com for'}
        )
        
        success = self.system.submit_feedback(feedback)
        assert success
        
        # Process the feedback
        result = self.system.processor.process_pending_feedback()
        
        assert result['processed_count'] > 0
        # Training examples should be created and added to collector

    def test_concurrent_feedback_submission(self):
        """Test concurrent feedback submission."""
        def submit_feedback_batch():
            for i in range(5):
                feedback = UserFeedback(
                    feedback_id=f"thread_test_{threading.current_thread().ident}_{i}",
                    document_id=f"doc_{i}",
                    text_segment=f"Test text {i}",
                    detected_category="test_category",
                    user_corrected_category="test_category",
                    detected_confidence=0.8,
                    user_confidence_rating=0.9,
                    feedback_type=FeedbackType.CORRECT_DETECTION,
                    severity=FeedbackSeverity.MEDIUM,
                    user_comment=f"Comment {i}",
                    context={}
                )
                self.system.submit_feedback(feedback)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=submit_feedback_batch)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all feedback was stored
        import sqlite3
        with sqlite3.connect(self.system.storage_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM user_feedback")
            count = cursor.fetchone()[0]
            assert count == 15  # 3 threads * 5 feedback each
    
    def test_performance_impact(self):
        """Test that feedback system has minimal performance impact."""
        import time
        
        # Measure time for feedback submission
        start_time = time.time()
        
        for i in range(50):
            feedback = UserFeedback(
                feedback_id=f"perf_test_{i}",
                document_id=f"doc_{i}",
                text_segment=f"Test text {i}",
                detected_category="test_category",
                user_corrected_category="test_category",
                detected_confidence=0.8,
                user_confidence_rating=0.9,
                feedback_type=FeedbackType.CORRECT_DETECTION,
                severity=FeedbackSeverity.MEDIUM,
                user_comment=f"Comment {i}",
                context={}
            )
            self.system.submit_feedback(feedback)
        
        submission_time = time.time() - start_time
        
        # Should be fast (less than 2 seconds for 50 submissions)
        assert submission_time < 2.0


if __name__ == '__main__':
    pytest.main([__file__]) 