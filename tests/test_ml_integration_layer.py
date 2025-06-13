"""
Unit tests for the ML Integration Layer.

Covers:
- add_user_feedback behavior (Session 4, Priority 3)
- detect_with_ml_integration processing and QualityAnalyzer integration
- MLConfidenceScorer integration with AdaptiveLearningCoordinator
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import uuid
from datetime import datetime

from app.core.ml_integration import (
    MLIntegrationLayer,
    DetectionResult,
    MLPrediction
)
from app.core.data_models import MLModel
from app.core.ml_engine import MLConfidenceScorer, create_ml_confidence_scorer
from app.core.feature_engineering import create_feature_extractor
from app.core.training_data import create_training_data_collector
from app.core.feedback_system import (
    UserFeedback,
    FeedbackType,
    UserFeedbackSystem
)
from app.core.analytics_engine import QualityAnalyzer
from app.core.adaptive.coordinator import AdaptiveLearningCoordinator

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")

@pytest.fixture(autouse=True)
def mock_logging():
    """Mock the StructuredLogger to suppress real logging during tests."""
    with patch('app.core.logging.StructuredLogger') as mock_logger:
        mock_logger.return_value.info = MagicMock()
        mock_logger.return_value.error = MagicMock()
        mock_logger.return_value.warning = MagicMock()
        mock_logger.return_value.debug = MagicMock()
        yield mock_logger

@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies of MLIntegrationLayer."""
    MockCoordinator = Mock(spec=AdaptiveLearningCoordinator)

    with patch('app.core.ml_engine.create_ml_confidence_scorer') as mock_scorer, \
         patch('app.core.ml_integration.create_feature_extractor') as mock_extractor, \
         patch('app.core.ml_integration.create_training_data_collector') as mock_collector, \
         patch('app.core.ml_integration.create_feedback_system') as mock_feedback, \
         patch('app.core.ml_integration.MLIntegrationLayer._get_quality_analyzer') as mock_analyzer:

        # configure mocks
        mock_scorer.return_value = Mock(spec=MLConfidenceScorer)
        mock_extractor.return_value = Mock()
        mock_collector.return_value = Mock()
        mock_feedback.return_value = Mock(spec=UserFeedbackSystem)
        mock_analyzer.return_value = Mock(spec=QualityAnalyzer)

        ufs = mock_feedback.return_value
        ufs.submit_feedback.return_value = True

        qa = mock_analyzer.return_value
        qa.add_detection_result = MagicMock()

        yield {
            "scorer": mock_scorer,
            "extractor": mock_extractor,
            "collector": mock_collector,
            "user_feedback_system_instance": ufs,
            "quality_analyzer_instance": qa,
            "coordinator": MockCoordinator,
            "analyzer": mock_analyzer,
        }

@pytest.fixture
def ml_integration_layer(mock_dependencies):
    """Instantiate MLIntegrationLayer with all dependencies mocked out."""
    layer = MLIntegrationLayer(config={})
    layer.quality_analyzer = mock_dependencies['quality_analyzer_instance']
    return layer

@pytest.fixture
def sample_detection_result() -> DetectionResult:
    """Provide a sample DetectionResult with a valid MLPrediction."""
    ml_pred = MLPrediction(
        pii_category="EMAIL",
        confidence=0.85,
        features_used=["text_length", "context_similarity", "pattern_match"],
        model_version="1.2.3"
    )

    return DetectionResult(
        text="Sample PII text",
        category="EMAIL",
        context="This is context for Sample PII text in an email.",
        position=10,
        ml_confidence=0.85,
        ml_prediction=ml_pred,
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
    """Tests for MLIntegrationLayer.add_user_feedback."""

    def test_add_user_feedback_confirmed_correct(
        self,
        ml_integration_layer: MLIntegrationLayer,
        sample_detection_result: DetectionResult,
        mock_dependencies
    ):
        ufs = mock_dependencies['user_feedback_system_instance']

        ml_integration_layer.add_user_feedback(
            result=sample_detection_result,
            user_confirmed=True,
            user_comment="Looks good!"
        )

        ufs.submit_feedback.assert_called_once()
        fb: UserFeedback = ufs.submit_feedback.call_args[0][0]

        assert fb.feedback_type == FeedbackType.CORRECT_DETECTION
        assert fb.text_segment == sample_detection_result.text
        assert fb.detected_category == sample_detection_result.category
        assert fb.user_corrected_category is None
        assert fb.detected_confidence == sample_detection_result.ml_confidence
        assert fb.user_confidence_rating is None
        assert fb.user_comment == "Looks good!"
        assert fb.document_id == sample_detection_result.document_type
        assert fb.context['surrounding_text'] == sample_detection_result.context

    def test_add_user_feedback_false_positive(
        self,
        ml_integration_layer: MLIntegrationLayer,
        sample_detection_result: DetectionResult,
        mock_dependencies
    ):
        ufs = mock_dependencies['user_feedback_system_instance']

        ml_integration_layer.add_user_feedback(
            result=sample_detection_result,
            user_confirmed=False,
            user_comment="This was not PII."
        )

        ufs.submit_feedback.assert_called_once()
        fb: UserFeedback = ufs.submit_feedback.call_args[0][0]

        assert fb.feedback_type == FeedbackType.FALSE_POSITIVE
        assert fb.user_comment == "This was not PII."

    def test_add_user_feedback_category_correction(
        self,
        ml_integration_layer: MLIntegrationLayer,
        sample_detection_result: DetectionResult,
        mock_dependencies
    ):
        ufs = mock_dependencies['user_feedback_system_instance']
        corrected = "NAME"

        ml_integration_layer.add_user_feedback(
            result=sample_detection_result,
            user_confirmed=True,
            correct_category=corrected,
            confidence_rating=0.95,
            user_comment="Actually a name, not an email."
        )

        ufs.submit_feedback.assert_called_once()
        fb: UserFeedback = ufs.submit_feedback.call_args[0][0]

        assert fb.feedback_type == FeedbackType.CATEGORY_CORRECTION
        assert fb.user_corrected_category == corrected
        assert fb.user_confidence_rating == 0.95
        assert fb.user_comment == "Actually a name, not an email."

    def test_add_user_feedback_context_details(
        self,
        ml_integration_layer: MLIntegrationLayer,
        sample_detection_result: DetectionResult,
        mock_dependencies
    ):
        ufs = mock_dependencies['user_feedback_system_instance']

        ml_integration_layer.add_user_feedback(
            result=sample_detection_result,
            user_confirmed=True
        )

        ufs.submit_feedback.assert_called_once()
        fb: UserFeedback = ufs.submit_feedback.call_args[0][0]

        keys = [
            'surrounding_text', 'position', 'original_document_type',
            'language', 'priority2_confidence', 'ml_prediction_details',
            'original_timestamp'
        ]
        for key in keys:
            assert key in fb.context

        assert fb.context['position'] == sample_detection_result.position
        assert fb.context['language'] == sample_detection_result.language
        assert fb.context['ml_prediction_details'] == sample_detection_result.ml_prediction.to_dict()
        assert fb.context['original_timestamp'] == sample_detection_result.timestamp.isoformat()

    def test_add_user_feedback_handles_uuid(
        self,
        ml_integration_layer: MLIntegrationLayer,
        sample_detection_result: DetectionResult,
        mock_dependencies
    ):
        ufs = mock_dependencies['user_feedback_system_instance']

        with patch('app.core.ml_integration.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678')
            ml_integration_layer.add_user_feedback(
                result=sample_detection_result,
                user_confirmed=True
            )

        ufs.submit_feedback.assert_called_once()
        fb: UserFeedback = ufs.submit_feedback.call_args[0][0]
        assert fb.feedback_id == str(mock_uuid.return_value)

    def test_add_user_feedback_handles_user_feedback_system_failure(
        self,
        ml_integration_layer: MLIntegrationLayer,
        sample_detection_result: DetectionResult,
        mock_dependencies
    ):
        ufs = mock_dependencies['user_feedback_system_instance']
        ufs.submit_feedback.return_value = False

        # Should not raise
        ml_integration_layer.add_user_feedback(
            result=sample_detection_result,
            user_confirmed=True
        )

        ufs.submit_feedback.assert_called_once()

@pytest.mark.unit
class TestMLIntegrationLayerDetectionProcessing:
    """Tests for detect_with_ml_integration and QualityAnalyzer calls."""

    def test_detect_with_ml_integration_calls_quality_analyzer(
        self,
        ml_integration_layer: MLIntegrationLayer,
        sample_detection_result: DetectionResult,
        mock_dependencies
    ):
        qa = mock_dependencies['quality_analyzer_instance']

        with patch.object(ml_integration_layer, '_detect_with_ml', return_value=sample_detection_result) as mock_ml, \
             patch.object(ml_integration_layer, '_detect_with_fallback') as mock_fb, \
             patch.object(ml_integration_layer, '_should_use_ml', return_value=True):

            result = ml_integration_layer.detect_with_ml_integration(
                text=sample_detection_result.text,
                category=sample_detection_result.category,
                context=sample_detection_result.context
            )

            mock_ml.assert_called_once()
            mock_fb.assert_not_called()
            assert result is sample_detection_result
            qa.add_detection_result.assert_called_once_with(sample_detection_result, ground_truth=None)

    def test_detect_with_ml_integration_calls_quality_analyzer_on_fallback(
        self,
        ml_integration_layer: MLIntegrationLayer,
        sample_detection_result: DetectionResult,
        mock_dependencies
    ):
        qa = mock_dependencies['quality_analyzer_instance']

        with patch.object(ml_integration_layer, '_detect_with_ml') as mock_ml, \
             patch.object(ml_integration_layer, '_detect_with_fallback', return_value=sample_detection_result) as mock_fb, \
             patch.object(ml_integration_layer, '_should_use_ml', return_value=False):

            result = ml_integration_layer.detect_with_ml_integration(
                text=sample_detection_result.text,
                category=sample_detection_result.category,
                context=sample_detection_result.context
            )

            mock_ml.assert_not_called()
            mock_fb.assert_called_once()
            assert result is sample_detection_result
            qa.add_detection_result.assert_called_once_with(sample_detection_result, ground_truth=None)

@pytest.mark.unit
class TestMLConfidenceScorerIntegration:
    """Tests MLConfidenceScorer integration with AdaptiveLearningCoordinator."""

    def test_adaptive_pattern_overrides_ml_model(self, mock_dependencies):
        coord = mock_dependencies['coordinator']
        pattern = {
            "pattern_id": "p_user_123",
            "regex": r"USR-123-ABC",
            "pii_category": "USER_ID",
            "confidence": 0.99
        }
        coord.get_adaptive_patterns.return_value = [pattern]

        scorer = MLConfidenceScorer(config={}, coordinator=coord)

        with patch.object(scorer, '_is_model_trained', return_value=True), \
             patch.object(scorer.model, 'predict_proba', new_callable=MagicMock) as mock_predict:

            pred = scorer.calculate_ml_confidence(
                detection="found user USR-123-ABC here",
                context="some context",
                features={"feature1": 1.0}
            )

            assert pred.confidence == 0.99
            assert pred.model_version == "adaptive_pattern_override"
            assert "adaptive_pattern:p_user_123" in pred.features_used
            mock_predict.assert_not_called()
