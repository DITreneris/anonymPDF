"""
Tests for ML Integration Layer - Session 4 Priority 3 Implementation

Focuses on testing the MLIntegrationLayer, especially the refactored
add_user_feedback method.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import uuid
from datetime import datetime

from app.core.ml_integration import MLIntegrationLayer, DetectionResult, MLPrediction
from app.core.ml_engine import create_ml_confidence_scorer
from app.core.feature_engineering import create_feature_extractor
from app.core.training_data import create_training_data_collector
from app.core.feedback_system import UserFeedback, FeedbackType, FeedbackSeverity, UserFeedbackSystem
from app.core.analytics_engine import QualityAnalyzer
from app.core.adaptive.coordinator import AdaptiveLearningCoordinator
from app.core.ml_engine import MLConfidenceScorer

# Register custom marks
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")

@pytest.fixture(autouse=True)
def mock_logging():
    """Mock logging to prevent LogRecord conflicts."""
    with patch('app.core.logging.StructuredLogger') as mock_logger:
        mock_logger.return_value.info = MagicMock()
        mock_logger.return_value.error = MagicMock()
        mock_logger.return_value.warning = MagicMock()
        mock_logger.return_value.debug = MagicMock()
        yield mock_logger

@pytest.fixture
def mock_dependencies():
    """
    Provides mocks for dependencies of MLIntegrationLayer.
    Patches are applied where the objects are looked up (in the ml_integration module),
    which is the correct way to patch imported objects.
    """
    with patch('app.core.ml_integration.create_ml_confidence_scorer') as mock_scorer, \
         patch('app.core.ml_integration.create_feature_extractor') as mock_extractor, \
         patch('app.core.ml_integration.create_training_data_collector') as mock_collector, \
         patch('app.core.analytics_engine.QualityAnalyzer') as MockQualityAnalyzer, \
         patch('app.core.ml_integration.UserFeedbackSystem') as MockUserFeedbackSystem, \
         patch('app.core.ml_engine.AdaptiveLearningCoordinator') as MockCoordinator:

        mock_ufs_instance = MockUserFeedbackSystem.return_value
        mock_ufs_instance.submit_feedback.return_value = True

        mock_quality_analyzer_instance = MockQualityAnalyzer.return_value
        mock_quality_analyzer_instance.add_detection_result = MagicMock()

        yield {
            "scorer": mock_scorer,
            "extractor": mock_extractor,
            "collector": mock_collector,
            "user_feedback_system_instance": mock_ufs_instance,
            "quality_analyzer_instance": mock_quality_analyzer_instance,
            "coordinator": MockCoordinator.return_value
        }

@pytest.fixture
def ml_integration_layer(mock_dependencies):
    """Fixture for MLIntegrationLayer with mocked dependencies."""
    # This now correctly receives mocked dependencies because the patches target the right module
    layer = MLIntegrationLayer(config={})
    # Ensure quality_analyzer is set for the detection processing tests
    layer.quality_analyzer = mock_dependencies['quality_analyzer_instance']
    return layer

@pytest.fixture
def sample_detection_result() -> DetectionResult:
    """Provides a sample DetectionResult for testing."""
    # Create MLPrediction with correct fields
    ml_prediction = MLPrediction(
        confidence=0.85,
        probability=0.85,
        features_used=["text_length", "context_similarity", "pattern_match"],
        model_version="1.2.3",
        prediction_time=datetime.now()
    )
    
    # Create DetectionResult with the MLPrediction
    return DetectionResult(
        text="Sample PII text",
        category="EMAIL",  # Category is part of DetectionResult, not MLPrediction
        context="This is context for Sample PII text in an email.",
        position=10,
        ml_confidence=0.85,
        ml_prediction=ml_prediction,
        priority2_confidence=0.70,
        fallback_used=False,
        processing_time_ms=15.0,
        features_extracted=100,
        document_type="test_document_type_123",
        language="en",
        timestamp=datetime.now()
    )

@pytest.mark.unit
class TestMLIntegrationLayerFeedback:
    """Tests the add_user_feedback method of MLIntegrationLayer."""

    def test_add_user_feedback_confirmed_correct(self, ml_integration_layer: MLIntegrationLayer, 
                                                 sample_detection_result: DetectionResult, 
                                                 mock_dependencies):
        """Test feedback for a correctly confirmed detection."""
        mock_ufs_instance = mock_dependencies['user_feedback_system_instance']
        
        ml_integration_layer.add_user_feedback(
            result=sample_detection_result, 
            user_confirmed=True,
            user_comment="Looks good!"
        )

        mock_ufs_instance.submit_feedback.assert_called_once()
        submitted_feedback: UserFeedback = mock_ufs_instance.submit_feedback.call_args[0][0]

        assert isinstance(submitted_feedback, UserFeedback)
        assert submitted_feedback.feedback_type == FeedbackType.CORRECT_DETECTION
        assert submitted_feedback.text_segment == sample_detection_result.text
        assert submitted_feedback.detected_category == sample_detection_result.category
        assert submitted_feedback.user_corrected_category is None
        assert submitted_feedback.detected_confidence == sample_detection_result.ml_confidence
        assert submitted_feedback.user_confidence_rating is None
        assert submitted_feedback.user_comment == "Looks good!"
        assert submitted_feedback.document_id == sample_detection_result.document_type
        assert 'surrounding_text' in submitted_feedback.context
        assert submitted_feedback.context['surrounding_text'] == sample_detection_result.context

    def test_add_user_feedback_false_positive(self, ml_integration_layer: MLIntegrationLayer, 
                                              sample_detection_result: DetectionResult, 
                                              mock_dependencies):
        """Test feedback for a false positive."""
        mock_ufs_instance = mock_dependencies['user_feedback_system_instance']

        ml_integration_layer.add_user_feedback(
            result=sample_detection_result, 
            user_confirmed=False, # Key difference: not confirmed
            user_comment="This was not PII."
        )

        mock_ufs_instance.submit_feedback.assert_called_once()
        submitted_feedback: UserFeedback = mock_ufs_instance.submit_feedback.call_args[0][0]

        assert submitted_feedback.feedback_type == FeedbackType.FALSE_POSITIVE
        assert submitted_feedback.user_comment == "This was not PII."

    def test_add_user_feedback_category_correction(self, ml_integration_layer: MLIntegrationLayer, 
                                                   sample_detection_result: DetectionResult, 
                                                   mock_dependencies):
        """Test feedback for a category correction."""
        mock_ufs_instance = mock_dependencies['user_feedback_system_instance']
        corrected_category = "NAME"

        ml_integration_layer.add_user_feedback(
            result=sample_detection_result, 
            user_confirmed=True, 
            correct_category=corrected_category,
            confidence_rating=0.95,
            user_comment="Actually a name, not an email."
        )

        mock_ufs_instance.submit_feedback.assert_called_once()
        submitted_feedback: UserFeedback = mock_ufs_instance.submit_feedback.call_args[0][0]

        assert submitted_feedback.feedback_type == FeedbackType.CATEGORY_CORRECTION
        assert submitted_feedback.user_corrected_category == corrected_category
        assert submitted_feedback.user_confidence_rating == 0.95
        assert submitted_feedback.user_comment == "Actually a name, not an email."

    def test_add_user_feedback_context_details(self, ml_integration_layer: MLIntegrationLayer, 
                                               sample_detection_result: DetectionResult, 
                                               mock_dependencies):
        """Test that the feedback context is correctly populated."""
        mock_ufs_instance = mock_dependencies['user_feedback_system_instance']
        
        ml_integration_layer.add_user_feedback(result=sample_detection_result, user_confirmed=True)

        mock_ufs_instance.submit_feedback.assert_called_once()
        submitted_feedback: UserFeedback = mock_ufs_instance.submit_feedback.call_args[0][0]

        expected_context_keys = [
            'surrounding_text', 'position', 'original_document_type', 
            'language', 'priority2_confidence', 'ml_prediction_details',
            'original_timestamp'
        ]
        for key in expected_context_keys:
            assert key in submitted_feedback.context
        
        assert submitted_feedback.context['position'] == sample_detection_result.position
        assert submitted_feedback.context['language'] == sample_detection_result.language
        if sample_detection_result.ml_prediction:
            assert submitted_feedback.context['ml_prediction_details'] == sample_detection_result.ml_prediction.to_dict()
        else:
            assert submitted_feedback.context['ml_prediction_details'] is None
        assert submitted_feedback.context['original_timestamp'] == sample_detection_result.timestamp.isoformat()

    def test_add_user_feedback_handles_uuid(self, ml_integration_layer: MLIntegrationLayer, 
                                        sample_detection_result: DetectionResult, 
                                        mock_dependencies):
        """Test that a UUID is generated for feedback_id."""
        mock_ufs_instance = mock_dependencies['user_feedback_system_instance']
        
        with patch('app.core.ml_integration.uuid.uuid4') as mock_uuid:
            test_uuid = uuid.UUID('12345678-1234-5678-1234-567812345678')
            mock_uuid.return_value = test_uuid
            ml_integration_layer.add_user_feedback(result=sample_detection_result, user_confirmed=True)
        
        mock_ufs_instance.submit_feedback.assert_called_once()
        submitted_feedback: UserFeedback = mock_ufs_instance.submit_feedback.call_args[0][0]
        assert submitted_feedback.feedback_id == str(test_uuid)

    def test_add_user_feedback_handles_user_feedback_system_failure(self, ml_integration_layer: MLIntegrationLayer, 
                                                                  sample_detection_result: DetectionResult, 
                                                                  mock_dependencies):
        """Test how add_user_feedback handles failure in UserFeedbackSystem.submit_feedback."""
        mock_ufs_instance = mock_dependencies['user_feedback_system_instance']
        mock_ufs_instance.submit_feedback.return_value = False # Simulate failure
        mock_ufs_instance.submit_feedback.side_effect = None # Clear any previous side effects like exceptions

        # We expect the method to log an error but not raise an exception itself
        try:
            ml_integration_layer.add_user_feedback(result=sample_detection_result, user_confirmed=True)
        except Exception as e:
            pytest.fail(f"add_user_feedback raised an exception unexpectedly: {e}")

        mock_ufs_instance.submit_feedback.assert_called_once()

@pytest.mark.unit
class TestMLIntegrationLayerDetectionProcessing:
    """Tests processing within detect_with_ml_integration, including QualityAnalyzer call."""

    def test_detect_with_ml_integration_calls_quality_analyzer(self, 
                                                              ml_integration_layer: MLIntegrationLayer, 
                                                              sample_detection_result: DetectionResult, 
                                                              mock_dependencies):
        """Test that detect_with_ml_integration calls quality_analyzer.add_detection_result."""
        mock_qa_instance = mock_dependencies['quality_analyzer_instance']

        # Mock the internal detection methods
        with patch.object(ml_integration_layer, '_detect_with_ml', return_value=sample_detection_result) as mock_detect_ml, \
             patch.object(ml_integration_layer, '_detect_with_fallback') as mock_detect_fallback, \
             patch.object(ml_integration_layer, '_should_use_ml', return_value=True):

            # Call the main method
            processed_result = ml_integration_layer.detect_with_ml_integration(
                text=sample_detection_result.text,
                category=sample_detection_result.category,
                context=sample_detection_result.context
            )
            
            mock_detect_ml.assert_called_once()
            mock_detect_fallback.assert_not_called()
            assert processed_result is sample_detection_result

            # Assert QualityAnalyzer was called with the correct result
            mock_qa_instance.add_detection_result.assert_called_once_with(
                sample_detection_result,
                ground_truth=None
            )

    def test_detect_with_ml_integration_calls_quality_analyzer_on_fallback(self, 
                                                                      ml_integration_layer: MLIntegrationLayer, 
                                                                      sample_detection_result: DetectionResult, 
                                                                      mock_dependencies):
        """Test that quality_analyzer is called even when using fallback detection."""
        mock_qa_instance = mock_dependencies['quality_analyzer_instance']
        
        # Mock the internal detection methods to force fallback
        with patch.object(ml_integration_layer, '_detect_with_ml') as mock_detect_ml, \
             patch.object(ml_integration_layer, '_detect_with_fallback', return_value=sample_detection_result) as mock_detect_fallback, \
             patch.object(ml_integration_layer, '_should_use_ml', return_value=False):

            # Call the main method
            processed_result = ml_integration_layer.detect_with_ml_integration(
                text=sample_detection_result.text,
                category=sample_detection_result.category,
                context=sample_detection_result.context
            )
            
            mock_detect_ml.assert_not_called()
            mock_detect_fallback.assert_called_once()
            assert processed_result is sample_detection_result

            # Assert QualityAnalyzer was called with the correct result
            mock_qa_instance.add_detection_result.assert_called_once_with(
                sample_detection_result,
                ground_truth=None
            ) 

@pytest.mark.unit
class TestMLConfidenceScorerIntegration:
    """Tests the integration between MLConfidenceScorer and the AdaptiveLearningCoordinator."""

    def test_adaptive_pattern_overrides_ml_model(self, mock_dependencies):
        """
        Verify that a high-confidence adaptive pattern bypasses the regular ML model.
        """
        # Arrange
        # 1. Setup the mock coordinator to return a specific pattern
        mock_coordinator = mock_dependencies['coordinator']
        test_pattern = {
            "pattern_id": "p_user_123",
            "regex": r"USR-123-ABC",
            "pii_category": "USER_ID",
            "confidence": 0.99
        }
        mock_coordinator.get_adaptive_patterns.return_value = [test_pattern]

        # 2. Create a real MLConfidenceScorer, but inject the mocked coordinator
        # We don't need a real config for this test
        scorer = MLConfidenceScorer(config={}, coordinator=mock_coordinator)
        
        # 3. Patch the scorer's internal model to ensure it's not called
        with patch.object(scorer, '_is_model_trained', return_value=True), \
             patch.object(scorer.model, 'predict_proba', new_callable=MagicMock) as mock_predict:

            # Act
            # Call the scorer with text that matches the adaptive pattern
            prediction = scorer.calculate_ml_confidence(
                detection="found user USR-123-ABC here",
                context="some context",
                features={"feature1": 1.0}
            )

            # Assert
            # 1. The prediction should come from the adaptive pattern
            assert prediction.confidence == 0.99
            assert prediction.model_version == "adaptive_pattern_override"
            assert "adaptive_pattern:p_user_123" in prediction.features_used

            # 2. The underlying ML model should NOT have been called
            mock_predict.assert_not_called() 