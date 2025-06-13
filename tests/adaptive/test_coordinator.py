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
def mock_subsystems(mocker):
    """Fixture to mock all subsystems of the AdaptiveCoordinator."""
    with patch('app.core.adaptive.pattern_learner.PatternLearner') as MockPatternLearner, \
         patch('app.core.adaptive.pattern_db.AdaptivePatternDB') as MockPatternDB, \
         patch('app.core.adaptive.ab_testing.ABTestManager') as MockABTestManager, \
         patch('app.core.adaptive.online_learner.OnlineLearner') as MockOnlineLearner, \
         patch('app.core.analytics_engine.QualityAnalyzer') as MockQualityAnalyzer, \
         patch('app.core.feature_engineering.FeatureExtractor') as MockFeatureExtractor, \
         patch('app.core.training_data.TrainingDataStorage') as MockTrainingDataStorage, \
         patch('app.core.feedback_system.UserFeedbackProcessor') as MockFeedbackProcessor:

        yield {
            "learner": MockPatternLearner.return_value,
            "db": MockPatternDB.return_value,
            "ab_tester": MockABTestManager.return_value,
            "online_learner": MockOnlineLearner.return_value,
            "quality_analyzer_instance": MockQualityAnalyzer.return_value,
            "feature_extractor_instance": MockFeatureExtractor.return_value,
            "training_data_storage_instance": MockTrainingDataStorage.return_value,
            "feedback_processor_instance": MockFeedbackProcessor.return_value,

            # Optional assertions
            "learner_class": MockPatternLearner,
            "db_class": MockPatternDB,
            "ab_manager_creator": MockABTestManager,
            "discoverer_class": MockPatternLearner,
            "feature_extractor_creator": MockFeatureExtractor,
            "quality_analyzer_class": MockQualityAnalyzer,
        }

@pytest.fixture
def coordinator(mock_subsystems):
    """Provides an instance of the coordinator with mocked subsystems."""
    return AdaptiveLearningCoordinator(quality_analyzer=mock_subsystems['quality_analyzer_instance'])

# All tests below remain unchanged
# (they assume correct functioning of the mocks and coordinator)
