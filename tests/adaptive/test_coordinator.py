"""
Unit tests for the AdaptiveLearningCoordinator.
"""

import pytest
from unittest.mock import MagicMock, patch, ANY

# Mock the database and other dependencies BEFORE importing the modules that use them
@pytest.fixture(autouse=True)
def mock_db_and_dependencies():
    """Auto-mocking fixture for all database and pipeline dependencies."""
    with patch('app.core.adaptive.pattern_db.sqlite3'), \
         patch('app.core.ml_training_pipeline.create_ml_training_pipeline'):
        yield

# Now we can safely import the coordinator
from app.core.adaptive.coordinator import AdaptiveLearningCoordinator
from app.core.adaptive.pattern_learner import ValidatedPattern
from app.core.feedback_system import UserFeedback, FeedbackType, FeedbackSeverity
from app.core.adaptive.ab_testing import ABTestResult

@pytest.fixture
def mock_subsystems():
    """Fixture to mock the subsystems used by the coordinator."""
    with patch('app.core.adaptive.coordinator.PatternDiscovery') as MockPatternDiscovery, \
         patch('app.core.adaptive.coordinator.AdaptivePatternDB') as MockPatternDB, \
         patch('app.core.adaptive.coordinator.OnlineLearner') as MockOnlineLearner, \
         patch('app.core.adaptive.coordinator.create_ab_test_manager') as MockCreateABManager, \
         patch('app.core.adaptive.coordinator.create_feature_extractor') as MockCreateFeatureExtractor, \
         patch('app.core.adaptive.coordinator.QualityAnalyzer') as MockQualityAnalyzer:
        
        mock_discoverer = MockPatternDiscovery.return_value
        mock_db = MockPatternDB.return_value
        mock_learner = MockOnlineLearner.return_value
        mock_ab_manager = MockCreateABManager.return_value
        mock_feature_extractor = MockCreateFeatureExtractor.return_value
        mock_quality_analyzer = MockQualityAnalyzer.return_value
        
        yield {
            "discoverer_class": MockPatternDiscovery,
            "db_class": MockPatternDB,
            "learner_class": MockOnlineLearner,
            "ab_manager_creator": MockCreateABManager,
            "feature_extractor_creator": MockCreateFeatureExtractor,
            "quality_analyzer_class": MockQualityAnalyzer,
            "discoverer_instance": mock_discoverer,
            "db_instance": mock_db,
            "learner_instance": mock_learner,
            "ab_manager_instance": mock_ab_manager,
            "feature_extractor_instance": mock_feature_extractor,
            "quality_analyzer_instance": mock_quality_analyzer
        }

@pytest.fixture
def coordinator(mock_subsystems):
    """Provides an instance of the coordinator with mocked subsystems."""
    return AdaptiveLearningCoordinator(quality_analyzer=mock_subsystems['quality_analyzer_instance'])

def test_initialization(coordinator, mock_subsystems):
    """Test that the coordinator initializes its subsystems."""
    assert coordinator is not None
    assert coordinator.quality_analyzer is not None
    # Assert that the classes were instantiated
    mock_subsystems['discoverer_class'].assert_called_once()
    mock_subsystems['db_class'].assert_called_once()
    mock_subsystems['learner_class'].assert_called_once()
    mock_subsystems['ab_manager_creator'].assert_called_once()
    mock_subsystems['feature_extractor_creator'].assert_called_once()
    mock_subsystems['quality_analyzer_class'].assert_called_once()

def test_process_feedback_and_learn_cycle(coordinator, mock_subsystems):
    """Test the main learning cycle orchestration."""
    mock_discoverer = mock_subsystems['discoverer_instance']
    mock_db = mock_subsystems['db_instance']
    mock_learner = mock_subsystems['learner_instance']
    mock_feature_extractor = mock_subsystems['feature_extractor_instance']
    
    # Arrange: Create sample feedback with valid Enum members
    sample_feedback = [
        UserFeedback(
            feedback_id="fb-1",
            document_id="doc-1",
            text_segment="test@example.com",
            detected_category="EMAIL",
            user_corrected_category=None,
            detected_confidence=0.9,
            user_confidence_rating=1.0,
            feedback_type=FeedbackType.CORRECT_DETECTION,
            severity=FeedbackSeverity.LOW,
            user_comment="Correct!",
            context={'position': 10, 'language': 'en', 'surrounding_text': '...'}
        )
    ]
    discovered_pattern = ValidatedPattern(
        pattern_id="p_123",
        regex="\\btest@example\\.com\\b",
        pii_category="EMAIL",
        confidence=0.95
    )
    mock_discoverer.discover_and_validate_patterns.return_value = [discovered_pattern]
    
    # Mock the feature extractor's output
    mock_features = {'text_length': 17, 'has_digits': 0.0, 'word_count': 2}
    mock_feature_extractor.extract_all_features.return_value.to_dict.return_value = mock_features

    # Act
    coordinator.process_feedback_and_learn(sample_feedback, ["some text corpus"])

    # Assert
    # 1. Check that PII was extracted from feedback and passed to discoverer
    mock_discoverer.discover_and_validate_patterns.assert_called_once_with(
        ["some text corpus"],
        pii_to_discover={"test@example.com": "EMAIL"},
        ground_truth_pii={"test@example.com": "EMAIL"}
    )
    
    # 2. Check that the new pattern was added to the database
    mock_db.add_or_update_pattern.assert_called_once_with(discovered_pattern)

    # 3. Check that the online learner was called with converted training examples
    mock_learner.retrain_model_if_needed.assert_called_once()
    
    # 4. Check that the feature extractor was called correctly
    mock_feature_extractor.extract_all_features.assert_called_once_with(
        detection_text="test@example.com",
        category="EMAIL",
        context="...",
        position=10,
        document_type="doc-1",
        language="en"
    )

    # 5. Check the argument passed to the learner
    args, _ = mock_learner.retrain_model_if_needed.call_args
    training_examples = args[0]
    assert len(training_examples) == 1
    example = training_examples[0]
    assert example.source == "user_feedback"
    assert example.expected_output == mock_features

def test_process_feedback_with_no_new_pii(coordinator, mock_subsystems):
    """Test that pattern discovery is skipped if feedback contains no new PII."""
    mock_discoverer = mock_subsystems['discoverer_instance']
    mock_db = mock_subsystems['db_instance']
    mock_learner = mock_subsystems['learner_instance']
    mock_feature_extractor = mock_subsystems['feature_extractor_instance']

    # Arrange: Feedback for a false positive contains no confirmed PII
    sample_feedback = [
        UserFeedback(
            feedback_id="fb-2",
            document_id="doc-2",
            text_segment="some non-pii text",
            detected_category="PERSON",
            user_corrected_category=None,
            detected_confidence=0.8,
            user_confidence_rating=None,
            feedback_type=FeedbackType.FALSE_POSITIVE,
            severity=FeedbackSeverity.MEDIUM,
            user_comment="Not a person",
            context={}
        )
    ]

    # Act
    coordinator.process_feedback_and_learn(sample_feedback, ["some text corpus"])

    # Assert
    mock_discoverer.discover_and_validate_patterns.assert_not_called()
    mock_db.add_or_update_pattern.assert_not_called()
    # Retraining should also be skipped
    mock_learner.retrain_model_if_needed.assert_not_called()
    # Feature extraction should also not be called if there's no training example to create
    mock_feature_extractor.extract_all_features.assert_not_called()

def test_ab_testing_integration(coordinator, mock_subsystems):
    """Test that the coordinator correctly delegates to the A/B test manager."""
    mock_ab_manager = mock_subsystems['ab_manager_instance']

    # Test create
    coordinator.create_ab_test("Test 1", "A cool test", "v2-model")
    mock_ab_manager.create_test.assert_called_once_with("Test 1", "A cool test", "v2-model", 0.5)

    # Test start
    coordinator.start_ab_test("test_id_123")
    mock_ab_manager.start_test.assert_called_once_with("test_id_123", 7)

    # Test assignment
    mock_ab_manager.get_assignment.return_value = 'variant'
    assignment = coordinator.get_model_assignment_for_request("user_abc", "test_id_123")
    mock_ab_manager.get_assignment.assert_called_once_with("user_abc", "test_id_123")
    assert assignment == 'variant'

def test_evaluate_and_log_ab_test(coordinator, mock_subsystems):
    """Test that A/B test evaluation is called and the result is logged."""
    mock_ab_manager = mock_subsystems['ab_manager_instance']
    mock_quality_analyzer = mock_subsystems['quality_analyzer_instance']

    # Arrange: Setup a mock result from the A/B test manager
    test_result = ABTestResult(
        test_id="test_123",
        winner="variant",
        confidence=0.95,
        summary="Variant was better.",
        metrics_comparison={"accuracy": {"winner": "variant"}}
    )
    mock_ab_manager.evaluate_test.return_value = test_result

    # Act
    coordinator.evaluate_and_log_ab_test("test_123")

    # Assert
    # 1. Ensure the A/B manager's evaluation method was called
    mock_ab_manager.evaluate_test.assert_called_once_with("test_123", 0.05)

    # 2. Ensure the quality analyzer's logging method was called with the result
    mock_quality_analyzer.log_ab_test_result.assert_called_once_with(test_result)

def test_close_method(coordinator, mock_subsystems):
    """Test that the close method calls the db's close method."""
    mock_db = mock_subsystems['db_instance']
    
    coordinator.close()
    
    mock_db.close.assert_called_once() 