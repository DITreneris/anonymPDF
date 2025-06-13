"""
Tests for the OnlineLearner component of the adaptive learning system.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.core.adaptive.online_learner import OnlineLearner
from app.core.training_data import TrainingExample

@pytest.fixture
def mock_dependencies():
    """Mocks dependencies for the OnlineLearner."""
    with patch('app.core.adaptive.online_learner.TrainingDataStorage') as MockStorage, \
         patch('app.core.adaptive.online_learner.create_ml_training_pipeline') as MockPipeline:
        
        mock_storage_instance = MockStorage.return_value
        mock_pipeline_instance = MockPipeline.return_value
        
        yield {
            "storage_class": MockStorage,
            "pipeline_class": MockPipeline,
            "storage_instance": mock_storage_instance,
            "pipeline_instance": mock_pipeline_instance
        }

@pytest.fixture
def online_learner(mock_dependencies):
    """Provides an OnlineLearner instance with mocked dependencies."""
    config = {'retrain_threshold': 5}
    learner = OnlineLearner(config=config)
    # The mocks are already in place, so the learner will use them
    return learner

@pytest.fixture
def sample_training_examples():
    """Fixture for a list of sample TrainingExample objects."""
    return [
        TrainingExample(
            detection_text=f"text {i}",
            category="CAT1",
            context="some context",
            features={'feature': i},
            confidence_score=0.9,
            is_true_positive=True
        ) for i in range(5)
    ]

def test_initialization(online_learner, mock_dependencies):
    """Test that OnlineLearner initializes its dependencies correctly."""
    mock_dependencies['storage_class'].assert_called_once()
    mock_dependencies['pipeline_class'].assert_called_once()
    assert online_learner.retrain_threshold == 5

def test_retrain_model_saves_examples_and_triggers_retraining(online_learner, mock_dependencies, sample_training_examples):
    """
    Test that new examples are saved and retraining is triggered when the threshold is met.
    """
    mock_storage = mock_dependencies['storage_instance']
    mock_pipeline = mock_dependencies['pipeline_instance']
    
    # Use enough examples to pass the threshold of 5
    examples_to_process = sample_training_examples[:6]

    # Act
    online_learner.retrain_model_if_needed(examples_to_process)

    # Assert
    # 1. Examples are saved to storage
    mock_storage.save_training_examples.assert_called_once_with(examples_to_process, source="online_feedback")
    
    # 2. Retraining pipeline is called because threshold was met
    mock_pipeline.run_training_cycle.assert_called_once_with(external_training_data=examples_to_process)

def test_retrain_model_saves_examples_but_skips_retraining(online_learner, mock_dependencies, sample_training_examples):
    """
    Test that new examples are saved but retraining is skipped when the threshold is not met.
    """
    mock_storage = mock_dependencies['storage_instance']
    mock_pipeline = mock_dependencies['pipeline_instance']
    
    # Use fewer examples than the threshold of 5
    examples_to_process = sample_training_examples[:3]

    # Act
    online_learner.retrain_model_if_needed(examples_to_process)

    # Assert
    # 1. Examples are still saved to storage
    mock_storage.save_training_examples.assert_called_once_with(examples_to_process, source="online_feedback")
    
    # 2. Retraining pipeline is NOT called
    mock_pipeline.run_training_cycle.assert_not_called()

def test_retrain_model_handles_empty_input(online_learner, mock_dependencies):
    """Test that the method handles an empty list of examples gracefully."""
    mock_storage = mock_dependencies['storage_instance']
    mock_pipeline = mock_dependencies['pipeline_instance']
    
    # Act
    result = online_learner.retrain_model_if_needed([])

    # Assert
    assert result is None
    mock_storage.save_training_examples.assert_not_called()
    mock_pipeline.run_training_cycle.assert_not_called()

@pytest.fixture
def mock_model(mocker):
    mock_model_instance = MagicMock()
    mock_model_instance.save.return_value = "/fake/path/model.pkl"
    return mock_model_instance

class TestOnlineLearner:
    """Tests for the OnlineLearner class."""

    def test_retrain_model_saves_examples_and_triggers_retraining(self, online_learner, mock_dependencies, mock_model, sample_training_examples):
        """
        Test that retrain_model saves examples and triggers the model's train method
        when the number of new examples reaches the threshold.
        """
        online_learner.retraining_threshold = 5
        online_learner.retrain_model(sample_training_examples)

        mock_dependencies['storage_instance'].save_training_examples.assert_called_once_with(sample_training_examples)
        mock_model.train.assert_called_once()

    def test_retrain_model_saves_examples_but_skips_retraining(self, online_learner, mock_dependencies, mock_model, sample_training_examples):
        """
        Test that retrain_model saves examples but does NOT trigger training
        when the number of new examples is below the threshold.
        """
        online_learner.retraining_threshold = 10  # Higher threshold
        online_learner.retrain_model(sample_training_examples)

        mock_dependencies['storage_instance'].save_training_examples.assert_called_once_with(sample_training_examples)
        mock_model.train.assert_not_called() 