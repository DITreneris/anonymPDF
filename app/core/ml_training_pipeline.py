"""
Advanced ML Training Pipeline for Priority 3 Implementation

This module provides real-time training, model updates, and training orchestration
with intelligent scheduling and performance monitoring.
"""

import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future

# Import components
from app.core.config_manager import get_config
from app.core.logging import get_logger
from app.core.data_models import TrainingExample
from app.core.ml_engine import MLConfidenceScorer, create_ml_confidence_scorer
from app.core.training_data import TrainingDataCollector, create_training_data_collector
from app.core.feature_engineering import FeatureExtractor, create_feature_extractor

pipeline_logger = get_logger("ml_training_pipeline")


class TrainingStatus(Enum):
    """Training pipeline status."""
    IDLE = "idle"
    QUEUED = "queued"
    TRAINING = "training"
    EVALUATING = "evaluating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    """Training job configuration and metadata."""
    job_id: str
    trigger_reason: str
    training_data_count: int
    status: TrainingStatus = TrainingStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Training parameters
    model_type: str = 'xgboost'
    max_samples: int = 5000
    balance_ratio: float = 0.5
    
    # Results
    metrics: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    
    def get_duration(self) -> Optional[timedelta]:
        """Get training duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    average_duration_seconds: float = 0.0
    last_training: Optional[datetime] = None
    model_accuracy_trend: List[float] = field(default_factory=list)
    
    def get_success_rate(self) -> float:
        """Get training success rate."""
        return self.successful_jobs / self.total_jobs if self.total_jobs > 0 else 0.0


class TrainingScheduler:
    """Intelligent training scheduler with adaptive intervals."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_training = None
        self.training_frequency = config.get('base_training_interval_hours', 24)  # 24 hours
        self.min_samples_threshold = config.get('min_samples_for_training', 50)
        self.performance_threshold = config.get('performance_degradation_threshold', 0.05)
        
    def should_trigger_training(self, stats: Dict[str, Any], 
                              performance_metrics: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Determine if training should be triggered.
        
        Args:
            stats: Current training data statistics
            performance_metrics: Current model performance metrics
            
        Returns:
            Tuple of (should_trigger, reason)
        """
        current_time = datetime.now()
        
        # Check minimum samples threshold
        if stats.get('total_samples', 0) < self.min_samples_threshold:
            return False, "insufficient_samples"
        
        # Check time-based triggers
        if self.last_training is None:
            return True, "initial_training"
        
        time_since_last = current_time - self.last_training
        if time_since_last.total_seconds() > self.training_frequency * 3600:  # Convert hours to seconds
            return True, "scheduled_training"
        
        # Check data accumulation trigger
        new_samples = stats.get('new_samples_since_last_training', 0)
        if new_samples >= self.config.get('new_samples_trigger', 100):
            return True, "data_accumulation"
        
        # Check performance degradation trigger
        if performance_metrics:
            accuracy = performance_metrics.get('accuracy_estimate', 1.0)
            if accuracy < (1.0 - self.performance_degradation_threshold):
                return True, "performance_degradation"
        
        return False, "no_trigger"
    
    def update_last_training(self):
        """Update last training timestamp."""
        self.last_training = datetime.now()
    
    def adapt_frequency(self, training_success: bool, performance_improvement: float):
        """Adapt training frequency based on success and performance."""
        if training_success and performance_improvement > 0.01:
            # Increase frequency if training is successful and improving
            self.training_frequency = max(6, self.training_frequency * 0.9)  # Min 6 hours
        elif not training_success:
            # Decrease frequency if training fails
            self.training_frequency = min(168, self.training_frequency * 1.2)  # Max 1 week


class ModelEvaluator:
    """Evaluates trained models before deployment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def evaluate_model(self, model: MLConfidenceScorer, 
                      validation_data: List[TrainingExample]) -> Dict[str, float]:
        """
        Evaluate model performance on validation data.
        
        Args:
            model: Trained ML model
            validation_data: Validation dataset
            
        Returns:
            Evaluation metrics
        """
        if not validation_data:
            return {'error': 'no_validation_data'}
        
        correct_predictions = 0
        total_predictions = len(validation_data)
        confidence_scores = []
        
        for example in validation_data:
            try:
                prediction = model.calculate_ml_confidence(
                    detection=example.detection_text,
                    context=example.context,
                    features=example.features,
                    pii_category=example.category
                )
                
                confidence_scores.append(prediction.confidence)
                
                # Simple accuracy check (could be more sophisticated)
                predicted_positive = prediction.confidence > 0.5
                actual_positive = example.is_true_positive
                
                if predicted_positive == actual_positive:
                    correct_predictions += 1
                    
            except Exception as e:
                pipeline_logger.warning(f"Evaluation failed for example: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            'accuracy': accuracy,
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'average_confidence': avg_confidence,
            'confidence_std': self._calculate_std(confidence_scores) if confidence_scores else 0.0
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def should_deploy_model(self, metrics: Dict[str, float], 
                           baseline_metrics: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
        """
        Determine if model should be deployed.
        
        Args:
            metrics: New model evaluation metrics
            baseline_metrics: Current model metrics for comparison
            
        Returns:
            Tuple of (should_deploy, reason)
        """
        min_accuracy = self.config.get('min_deployment_accuracy', 0.7)
        if metrics.get('accuracy', 0.0) < min_accuracy:
            return False, f"accuracy_below_threshold_{min_accuracy}"
        
        min_samples = self.config.get('min_evaluation_samples', 20)
        if metrics.get('total_samples', 0) < min_samples:
            return False, f"insufficient_evaluation_samples_{min_samples}"
        
        # Compare with baseline if available
        if baseline_metrics:
            improvement_threshold = self.config.get('min_improvement_for_deployment', 0.01)
            current_accuracy = baseline_metrics.get('accuracy', 0.0)
            new_accuracy = metrics.get('accuracy', 0.0)
            
            if new_accuracy < current_accuracy + improvement_threshold:
                return False, f"insufficient_improvement_{improvement_threshold}"
        
        return True, "deployment_approved"


class MLTrainingPipeline:
    """Orchestrates the ML model training and update process."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get('ml_training_pipeline', {})
        self.training_data_collector = create_training_data_collector()
        self.model_trainer: MLConfidenceScorer = create_ml_confidence_scorer()
        pipeline_logger.info("MLTrainingPipeline initialized.")

    def run_training_cycle(self) -> Dict[str, Any]:
        """Execute a full training cycle."""
        pipeline_logger.info("Starting new ML training cycle...")
        
        # 1. Collect new training data
        training_data = self.training_data_collector.get_all_data()
        if not training_data or len(training_data) < self.config.get('min_training_samples', 10):
            pipeline_logger.warning("Not enough new training data available. Skipping cycle.")
            return {"status": "skipped", "reason": "insufficient_data"}
        
        # 2. Train the model
        pipeline_logger.info(f"Training model with {len(training_data)} examples.")
        training_results = self.model_trainer.train_model(training_data)
        
        # 3. (Future) Evaluate model against a hold-out test set
        
        # 4. (Future) If evaluation is successful, promote the new model
        
        pipeline_logger.info("ML training cycle completed successfully.")
        return {"status": "completed", "results": training_results}


def create_ml_training_pipeline(config: Optional[Dict[str, Any]] = None) -> MLTrainingPipeline:
    """Factory function to create an MLTrainingPipeline."""
    return MLTrainingPipeline(config=config) 