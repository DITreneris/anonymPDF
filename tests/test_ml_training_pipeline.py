"""
Tests for ML Training Pipeline - Priority 3 Implementation

This module tests the ML training pipeline components including scheduling,
evaluation, and orchestration functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from app.core.ml_training_pipeline import (
    TrainingStatus,
    TrainingJob, 
    TrainingMetrics,
    TrainingScheduler,
    ModelEvaluator,
    MLTrainingPipeline,
    create_ml_training_pipeline
)
from app.core.data_models import TrainingExample


class TestTrainingStatus:
    """Test TrainingStatus enum."""
    
    def test_enum_values(self):
        """Test that all enum values are correctly defined."""
        assert TrainingStatus.IDLE.value == "idle"
        assert TrainingStatus.QUEUED.value == "queued"
        assert TrainingStatus.TRAINING.value == "training"
        assert TrainingStatus.EVALUATING.value == "evaluating"
        assert TrainingStatus.DEPLOYING.value == "deploying"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"

    def test_enum_membership(self):
        """Test enum membership operations."""
        assert TrainingStatus.TRAINING in TrainingStatus
        assert "invalid" not in [status.value for status in TrainingStatus]


class TestTrainingJob:
    """Test TrainingJob dataclass."""
    
    def test_job_creation(self):
        """Test creating a TrainingJob with required fields."""
        job = TrainingJob(
            job_id="test_job_001",
            trigger_reason="manual",
            training_data_count=100
        )
        
        assert job.job_id == "test_job_001"
        assert job.trigger_reason == "manual"
        assert job.training_data_count == 100
        assert job.status == TrainingStatus.QUEUED
        assert job.model_type == 'xgboost'
        assert job.max_samples == 5000
        assert job.balance_ratio == 0.5
        assert job.metrics is None
        assert job.model_path is None
        assert job.error_message is None

    def test_job_with_optional_fields(self):
        """Test TrainingJob with optional fields set."""
        metrics = {"accuracy": 0.92, "precision": 0.89}
        
        job = TrainingJob(
            job_id="test_job_002",
            trigger_reason="scheduled",
            training_data_count=250,
            status=TrainingStatus.COMPLETED,
            model_type="random_forest",
            max_samples=1000,
            balance_ratio=0.7,
            metrics=metrics,
            model_path="/models/test_model.pkl",
            error_message=None
        )
        
        assert job.status == TrainingStatus.COMPLETED
        assert job.model_type == "random_forest"
        assert job.max_samples == 1000
        assert job.balance_ratio == 0.7
        assert job.metrics == metrics
        assert job.model_path == "/models/test_model.pkl"

    def test_get_duration_with_timestamps(self):
        """Test get_duration when both timestamps are set."""
        job = TrainingJob(
            job_id="test_job_003",
            trigger_reason="test",
            training_data_count=50
        )
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=30)
        
        job.started_at = start_time
        job.completed_at = end_time
        
        duration = job.get_duration()
        assert duration == timedelta(minutes=30)

    def test_get_duration_without_timestamps(self):
        """Test get_duration when timestamps are missing."""
        job = TrainingJob(
            job_id="test_job_004",
            trigger_reason="test",
            training_data_count=50
        )
        
        # Both None
        assert job.get_duration() is None
        
        # Only start time
        job.started_at = datetime.now()
        assert job.get_duration() is None
        
        # Only end time
        job.started_at = None
        job.completed_at = datetime.now()
        assert job.get_duration() is None


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating TrainingMetrics with defaults."""
        metrics = TrainingMetrics()
        
        assert metrics.total_jobs == 0
        assert metrics.successful_jobs == 0
        assert metrics.failed_jobs == 0
        assert metrics.average_duration_seconds == 0.0
        assert metrics.last_training is None
        assert metrics.model_accuracy_trend == []

    def test_metrics_with_values(self):
        """Test TrainingMetrics with specific values."""
        last_training = datetime.now()
        accuracy_trend = [0.85, 0.87, 0.89, 0.92]
        
        metrics = TrainingMetrics(
            total_jobs=10,
            successful_jobs=8,
            failed_jobs=2,
            average_duration_seconds=125.5,
            last_training=last_training,
            model_accuracy_trend=accuracy_trend
        )
        
        assert metrics.total_jobs == 10
        assert metrics.successful_jobs == 8
        assert metrics.failed_jobs == 2
        assert metrics.average_duration_seconds == 125.5
        assert metrics.last_training == last_training
        assert metrics.model_accuracy_trend == accuracy_trend

    def test_get_success_rate(self):
        """Test success rate calculation."""
        # Zero jobs
        metrics = TrainingMetrics()
        assert metrics.get_success_rate() == 0.0
        
        # Some successful jobs
        metrics.total_jobs = 10
        metrics.successful_jobs = 8
        assert metrics.get_success_rate() == 0.8
        
        # All successful
        metrics.total_jobs = 5
        metrics.successful_jobs = 5
        assert metrics.get_success_rate() == 1.0
        
        # No successful jobs
        metrics.total_jobs = 3
        metrics.successful_jobs = 0
        assert metrics.get_success_rate() == 0.0


class TestTrainingScheduler:
    """Test TrainingScheduler class."""
    
    def test_scheduler_initialization(self):
        """Test TrainingScheduler initialization."""
        config = {
            'base_training_interval_hours': 12,
            'min_samples_for_training': 75,
            'performance_degradation_threshold': 0.1,
            'new_samples_trigger': 150
        }
        
        scheduler = TrainingScheduler(config)
        
        assert scheduler.config == config
        assert scheduler.last_training is None
        assert scheduler.training_frequency == 12
        assert scheduler.min_samples_threshold == 75
        assert scheduler.performance_threshold == 0.1

    def test_scheduler_default_values(self):
        """Test TrainingScheduler with default values."""
        scheduler = TrainingScheduler({})
        
        assert scheduler.training_frequency == 24  # Default
        assert scheduler.min_samples_threshold == 50  # Default from get()
        assert scheduler.performance_threshold == 0.05  # Default from get()

    def test_should_trigger_training_insufficient_samples(self):
        """Test training trigger with insufficient samples."""
        scheduler = TrainingScheduler({'min_samples_for_training': 100})
        
        stats = {'total_samples': 50}
        should_trigger, reason = scheduler.should_trigger_training(stats)
        
        assert should_trigger is False
        assert reason == "insufficient_samples"

    def test_should_trigger_training_initial_training(self):
        """Test training trigger for initial training."""
        scheduler = TrainingScheduler({'min_samples_for_training': 50})
        
        stats = {'total_samples': 100}
        should_trigger, reason = scheduler.should_trigger_training(stats)
        
        assert should_trigger is True
        assert reason == "initial_training"

    @patch('app.core.ml_training_pipeline.datetime')
    def test_should_trigger_training_scheduled(self, mock_datetime):
        """Test training trigger for scheduled training."""
        current_time = datetime(2023, 6, 15, 10, 0, 0)
        last_training_time = datetime(2023, 6, 14, 9, 0, 0)  # 25 hours ago
        
        mock_datetime.now.return_value = current_time
        
        scheduler = TrainingScheduler({'min_samples_for_training': 50})
        scheduler.last_training = last_training_time
        scheduler.training_frequency = 24  # 24 hours
        
        stats = {'total_samples': 100}
        should_trigger, reason = scheduler.should_trigger_training(stats)
        
        assert should_trigger is True
        assert reason == "scheduled_training"

    def test_should_trigger_training_data_accumulation(self):
        """Test training trigger for data accumulation."""
        current_time = datetime.now()
        recent_training = current_time - timedelta(hours=5)  # Recent training
        
        scheduler = TrainingScheduler({
            'min_samples_for_training': 50,
            'new_samples_trigger': 100
        })
        scheduler.last_training = recent_training
        scheduler.training_frequency = 24
        
        stats = {
            'total_samples': 200,
            'new_samples_since_last_training': 120
        }
        should_trigger, reason = scheduler.should_trigger_training(stats)
        
        assert should_trigger is True
        assert reason == "data_accumulation"

    def test_should_trigger_training_performance_degradation(self):
        """Test training trigger for performance degradation."""
        current_time = datetime.now()
        recent_training = current_time - timedelta(hours=5)
        
        scheduler = TrainingScheduler({
            'min_samples_for_training': 50,
            'performance_degradation_threshold': 0.1
        })
        scheduler.last_training = recent_training
        scheduler.performance_degradation_threshold = 0.1  # Explicitly set the attribute
        
        stats = {'total_samples': 100}
        performance_metrics = {'accuracy_estimate': 0.85}  # Below threshold (1.0 - 0.1 = 0.9)
        
        should_trigger, reason = scheduler.should_trigger_training(stats, performance_metrics)
        
        assert should_trigger is True
        assert reason == "performance_degradation"

    def test_should_trigger_training_no_trigger(self):
        """Test when no training trigger conditions are met."""
        current_time = datetime.now()
        recent_training = current_time - timedelta(hours=5)
        
        scheduler = TrainingScheduler({
            'min_samples_for_training': 50,
            'performance_degradation_threshold': 0.05
        })
        scheduler.last_training = recent_training
        scheduler.training_frequency = 24
        scheduler.performance_degradation_threshold = 0.05  # Explicitly set the attribute
        
        stats = {
            'total_samples': 100,
            'new_samples_since_last_training': 10
        }
        performance_metrics = {'accuracy_estimate': 0.95}
        
        should_trigger, reason = scheduler.should_trigger_training(stats, performance_metrics)
        
        assert should_trigger is False
        assert reason == "no_trigger"

    @patch('app.core.ml_training_pipeline.datetime')
    def test_update_last_training(self, mock_datetime):
        """Test updating last training timestamp."""
        current_time = datetime(2023, 6, 15, 12, 0, 0)
        mock_datetime.now.return_value = current_time
        
        scheduler = TrainingScheduler({})
        scheduler.update_last_training()
        
        assert scheduler.last_training == current_time

    def test_adapt_frequency_successful_improvement(self):
        """Test frequency adaptation for successful training with improvement."""
        scheduler = TrainingScheduler({})
        scheduler.training_frequency = 24
        
        scheduler.adapt_frequency(training_success=True, performance_improvement=0.05)
        
        # Should decrease frequency (more frequent training)
        assert scheduler.training_frequency == 24 * 0.9  # 21.6 hours

    def test_adapt_frequency_successful_no_improvement(self):
        """Test frequency adaptation for successful training without improvement."""
        scheduler = TrainingScheduler({})
        scheduler.training_frequency = 24
        
        scheduler.adapt_frequency(training_success=True, performance_improvement=0.005)
        
        # Should not change frequency (improvement too small)
        assert scheduler.training_frequency == 24

    def test_adapt_frequency_failure(self):
        """Test frequency adaptation for failed training."""
        scheduler = TrainingScheduler({})
        scheduler.training_frequency = 24
        
        scheduler.adapt_frequency(training_success=False, performance_improvement=0.0)
        
        # Should increase frequency (less frequent training)
        assert scheduler.training_frequency == 24 * 1.2  # 28.8 hours

    def test_adapt_frequency_bounds(self):
        """Test frequency adaptation bounds."""
        scheduler = TrainingScheduler({})
        
        # Test minimum bound - case where 0.9 * frequency > 6
        scheduler.training_frequency = 7
        scheduler.adapt_frequency(training_success=True, performance_improvement=0.1)
        # Should be 6.3 (7 * 0.9), but max(6, 6.3) = 6.3
        assert scheduler.training_frequency == 6.3
        
        # Test minimum bound - case where 0.9 * frequency < 6
        scheduler.training_frequency = 6
        scheduler.adapt_frequency(training_success=True, performance_improvement=0.1)
        # Should be max(6, 6 * 0.9) = max(6, 5.4) = 6
        assert scheduler.training_frequency == 6
        
        # Test maximum bound
        scheduler.training_frequency = 150
        scheduler.adapt_frequency(training_success=False, performance_improvement=0.0)
        assert scheduler.training_frequency == 168  # Maximum is 168 hours (1 week)


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        config = {'min_deployment_accuracy': 0.8}
        evaluator = ModelEvaluator(config)
        
        assert evaluator.config == config

    def test_evaluate_model_no_validation_data(self):
        """Test model evaluation with no validation data."""
        evaluator = ModelEvaluator({})
        mock_model = Mock()
        
        result = evaluator.evaluate_model(mock_model, [])
        
        assert result == {'error': 'no_validation_data'}

    def test_evaluate_model_success(self):
        """Test successful model evaluation."""
        evaluator = ModelEvaluator({})
        
        # Mock model
        mock_model = Mock()
        mock_prediction = Mock()
        mock_prediction.confidence = 0.85
        mock_model.calculate_ml_confidence.return_value = mock_prediction
        
        # Mock validation data - using correct TrainingExample constructor
        validation_data = [
            TrainingExample(
                detection_text="John Doe",
                category="names",
                context="Name: John Doe",
                features={},
                confidence_score=0.9,
                is_true_positive=True
            ),
            TrainingExample(
                detection_text="123-45-6789", 
                category="ssn",
                context="SSN: 123-45-6789",
                features={},
                confidence_score=0.8,
                is_true_positive=True
            )
        ]
        
        result = evaluator.evaluate_model(mock_model, validation_data)
        
        assert result['accuracy'] == 1.0  # Both predictions correct
        assert result['total_samples'] == 2
        assert result['correct_predictions'] == 2
        assert result['average_confidence'] == 0.85
        assert 'confidence_std' in result

    def test_evaluate_model_with_errors(self):
        """Test model evaluation with some prediction errors."""
        evaluator = ModelEvaluator({})
        
        # Mock model that raises exception for some predictions
        mock_model = Mock()
        mock_model.calculate_ml_confidence.side_effect = [
            Mock(confidence=0.9),  # Success
            Exception("Prediction failed"),  # Error
            Mock(confidence=0.7)   # Success
        ]
        
        validation_data = [
            TrainingExample(
                detection_text="text1", 
                category="category1", 
                context="context1", 
                features={}, 
                confidence_score=0.8, 
                is_true_positive=True
            ),
            TrainingExample(
                detection_text="text2", 
                category="category2", 
                context="context2", 
                features={}, 
                confidence_score=0.9, 
                is_true_positive=True
            ),
            TrainingExample(
                detection_text="text3", 
                category="category3", 
                context="context3", 
                features={}, 
                confidence_score=0.7, 
                is_true_positive=True
            )
        ]
        
        with patch('app.core.ml_training_pipeline.pipeline_logger') as mock_logger:
            result = evaluator.evaluate_model(mock_model, validation_data)
        
        # Should process 2 successful predictions out of 3
        assert result['total_samples'] == 3
        assert result['correct_predictions'] == 2
        mock_logger.warning.assert_called_once()

    def test_calculate_std(self):
        """Test standard deviation calculation."""
        evaluator = ModelEvaluator({})
        
        # Empty list
        assert evaluator._calculate_std([]) == 0.0
        
        # Single value
        assert evaluator._calculate_std([0.5]) == 0.0
        
        # Multiple values
        values = [0.8, 0.9, 0.7, 0.85]
        std = evaluator._calculate_std(values)
        assert std > 0  # Should be positive
        assert abs(std - 0.075) < 0.01  # Approximate expected value

    def test_should_deploy_model_below_accuracy_threshold(self):
        """Test deployment decision with low accuracy."""
        config = {'min_deployment_accuracy': 0.8}
        evaluator = ModelEvaluator(config)
        
        metrics = {'accuracy': 0.75, 'total_samples': 100}
        should_deploy, reason = evaluator.should_deploy_model(metrics)
        
        assert should_deploy is False
        assert reason == "accuracy_below_threshold_0.8"

    def test_should_deploy_model_insufficient_samples(self):
        """Test deployment decision with insufficient evaluation samples."""
        config = {'min_evaluation_samples': 50}
        evaluator = ModelEvaluator(config)
        
        metrics = {'accuracy': 0.9, 'total_samples': 25}
        should_deploy, reason = evaluator.should_deploy_model(metrics)
        
        assert should_deploy is False
        assert reason == "insufficient_evaluation_samples_50"

    def test_should_deploy_model_insufficient_improvement(self):
        """Test deployment decision with insufficient improvement over baseline."""
        config = {'min_improvement_for_deployment': 0.05}
        evaluator = ModelEvaluator(config)
        
        metrics = {'accuracy': 0.82, 'total_samples': 100}
        baseline_metrics = {'accuracy': 0.80}
        
        should_deploy, reason = evaluator.should_deploy_model(metrics, baseline_metrics)
        
        assert should_deploy is False
        assert reason == "insufficient_improvement_0.05"

    def test_should_deploy_model_approved(self):
        """Test deployment decision when model should be deployed."""
        config = {
            'min_deployment_accuracy': 0.7,
            'min_evaluation_samples': 20
        }
        evaluator = ModelEvaluator(config)
        
        metrics = {'accuracy': 0.85, 'total_samples': 50}
        should_deploy, reason = evaluator.should_deploy_model(metrics)
        
        assert should_deploy is True
        assert reason == "deployment_approved"

    def test_should_deploy_model_no_baseline(self):
        """Test deployment decision without baseline metrics."""
        config = {'min_deployment_accuracy': 0.7, 'min_evaluation_samples': 20}
        evaluator = ModelEvaluator(config)
        
        metrics = {'accuracy': 0.85, 'total_samples': 50}
        should_deploy, reason = evaluator.should_deploy_model(metrics)
        
        assert should_deploy is True
        assert reason == "deployment_approved"


class TestMLTrainingPipeline:
    """Test MLTrainingPipeline class."""
    
    @patch('app.core.ml_training_pipeline.create_training_data_collector')
    @patch('app.core.ml_training_pipeline.create_ml_confidence_scorer')
    @patch('app.core.ml_training_pipeline.get_config')
    @patch('app.core.ml_training_pipeline.pipeline_logger')
    def test_pipeline_initialization(self, mock_logger, mock_get_config, mock_create_scorer, mock_create_collector):
        """Test MLTrainingPipeline initialization."""
        mock_config = {'ml_training_pipeline': {'min_training_samples': 20}}
        mock_get_config.return_value = mock_config
        
        mock_collector = Mock()
        mock_scorer = Mock()
        mock_create_collector.return_value = mock_collector
        mock_create_scorer.return_value = mock_scorer
        
        pipeline = MLTrainingPipeline()
        
        assert pipeline.config == {'min_training_samples': 20}
        assert pipeline.training_data_collector == mock_collector
        assert pipeline.model_trainer == mock_scorer

    @patch('app.core.ml_training_pipeline.create_training_data_collector')
    @patch('app.core.ml_training_pipeline.create_ml_confidence_scorer')
    @patch('app.core.ml_training_pipeline.pipeline_logger')
    def test_pipeline_with_custom_config(self, mock_logger, mock_create_scorer, mock_create_collector):
        """Test MLTrainingPipeline with custom config."""
        custom_config = {'min_training_samples': 50}
        
        mock_collector = Mock()
        mock_scorer = Mock()
        mock_create_collector.return_value = mock_collector
        mock_create_scorer.return_value = mock_scorer
        
        pipeline = MLTrainingPipeline(config=custom_config)
        
        assert pipeline.config == custom_config

    @patch('app.core.ml_training_pipeline.create_training_data_collector')
    @patch('app.core.ml_training_pipeline.create_ml_confidence_scorer')
    @patch('app.core.ml_training_pipeline.pipeline_logger')
    def test_run_training_cycle_insufficient_data(self, mock_logger, mock_create_scorer, mock_create_collector):
        """Test training cycle with insufficient data."""
        mock_collector = Mock()
        mock_collector.get_all_data.return_value = []  # No data
        mock_create_collector.return_value = mock_collector
        mock_create_scorer.return_value = Mock()
        
        pipeline = MLTrainingPipeline(config={'min_training_samples': 10})
        result = pipeline.run_training_cycle()
        
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_data"
        mock_logger.warning.assert_called_once()

    @patch('app.core.ml_training_pipeline.create_training_data_collector')
    @patch('app.core.ml_training_pipeline.create_ml_confidence_scorer')
    @patch('app.core.ml_training_pipeline.pipeline_logger')
    def test_run_training_cycle_success(self, mock_logger, mock_create_scorer, mock_create_collector):
        """Test successful training cycle."""
        # Mock training data
        training_data = [Mock() for _ in range(15)]
        
        mock_collector = Mock()
        mock_collector.get_all_data.return_value = training_data
        mock_create_collector.return_value = mock_collector
        
        mock_scorer = Mock()
        mock_training_results = {'accuracy': 0.9, 'model_path': '/tmp/model.pkl'}
        mock_scorer.train_model.return_value = mock_training_results
        mock_create_scorer.return_value = mock_scorer
        
        pipeline = MLTrainingPipeline(config={'min_training_samples': 10})
        result = pipeline.run_training_cycle()
        
        assert result["status"] == "completed"
        assert result["results"] == mock_training_results
        
        mock_scorer.train_model.assert_called_once_with(training_data)
        mock_logger.info.assert_called()


class TestFactoryFunction:
    """Test factory function."""
    
    @patch('app.core.ml_training_pipeline.MLTrainingPipeline')
    def test_create_ml_training_pipeline(self, mock_pipeline_class):
        """Test create_ml_training_pipeline factory function."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        config = {'test': 'config'}
        result = create_ml_training_pipeline(config)
        
        mock_pipeline_class.assert_called_once_with(config=config)
        assert result == mock_pipeline

    @patch('app.core.ml_training_pipeline.MLTrainingPipeline')
    def test_create_ml_training_pipeline_no_config(self, mock_pipeline_class):
        """Test factory function without config."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        result = create_ml_training_pipeline()
        
        mock_pipeline_class.assert_called_once_with(config=None)
        assert result == mock_pipeline


if __name__ == "__main__":
    pytest.main([__file__]) 