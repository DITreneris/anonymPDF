"""
Central Coordinator for the Adaptive Learning System.

This module ties together all components of the adaptive learning pipeline,
managing the flow of data from user feedback to pattern discovery, validation,
and model retraining.
"""

from __future__ import annotations
import functools
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from pathlib import Path

# Use TYPE_CHECKING block to avoid circular imports at runtime
if TYPE_CHECKING:
    from .pattern_learner import PatternLearner
    from .pattern_db import AdaptivePatternDB
    from .online_learner import OnlineLearner
    from .ab_testing import ABTestManager
    from app.core.feedback_system import UserFeedback, UserFeedbackProcessor
    from app.core.data_models import TrainingExample
    from app.core.analytics_engine import QualityAnalyzer
    from app.core.feature_engineering import FeatureExtractor
    from app.core.training_data import TrainingDataStorage

from app.core.config_manager import get_config_manager
from app.core.logging import get_logger

logger = get_logger("adaptive_learning.coordinator")


def disabled_on_flag(func):
    """
    Decorator to disable a method if the coordinator's `is_enabled` flag is False.
    It returns a default value (e.g., None, [], 'control') based on the method's return type hint.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_enabled:
            method_name = func.__name__
            logger.warning(f"Adaptive learning is disabled, cannot call '{method_name}'.")
            
            # Provide sensible defaults based on expected return types
            if method_name == "get_adaptive_patterns":
                return []
            if method_name == "get_model_assignment_for_request":
                return "control"
            return None # Default for create_ab_test, start_ab_test, evaluate_and_log_ab_test
        return func(self, *args, **kwargs)
    return wrapper


class AdaptiveLearningCoordinator:
    """
    Orchestrates the entire adaptive learning process.
    """

    def __init__(
        self,
        pattern_db: Optional[AdaptivePatternDB] = None,
        ab_test_manager: Optional[ABTestManager] = None,
        quality_analyzer: Optional[QualityAnalyzer] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        feedback_processor: Optional[UserFeedbackProcessor] = None,
        training_data_storage: Optional[TrainingDataStorage] = None
    ):
        """
        Initializes the coordinator with optional, injectable components.
        This allows for easy testing and flexible component replacement.
        """
        config = get_config_manager().settings.get("adaptive_learning", {})
        self.config = config
        self.is_enabled = config.get("enabled", False)

        if self.is_enabled:
            # Import heavy dependencies only when enabled
            from .pattern_db import AdaptivePatternDB
            from .ab_testing import ABTestManager
            from .pattern_learner import PatternLearner
            from app.core.analytics_engine import QualityAnalyzer
            from app.core.feature_engineering import FeatureExtractor
            from app.core.training_data import TrainingDataStorage
            from app.core.feedback_system import UserFeedbackProcessor

            db_config = config.get("databases", {})
            thresholds = config.get("thresholds", {})

            self.pattern_db = pattern_db or AdaptivePatternDB(db_path=Path(db_config.get("patterns_db")))
            self.ab_test_manager = ab_test_manager or ABTestManager(db_path=Path(db_config.get("ab_tests_db")))
            self.quality_analyzer = quality_analyzer or QualityAnalyzer(db_path=Path(db_config.get("analytics_db")))
            self.feature_extractor = feature_extractor or FeatureExtractor()
            self.feedback_processor = feedback_processor or UserFeedbackProcessor()
            self.training_data_storage = training_data_storage or TrainingDataStorage()
            self.pattern_learner = PatternLearner(
                self.pattern_db,
                min_confidence=thresholds.get("min_confidence_to_validate"),
                min_samples=thresholds.get("min_samples_for_learning")
            )
            logger.info("AdaptiveLearningCoordinator initialized with configuration.")
        else:
            logger.warning("Adaptive learning is disabled in configuration.")
            # Set all components to None if disabled
            self.pattern_db = None
            self.ab_test_manager = None
            self.quality_analyzer = None
            self.feature_extractor = None
            self.feedback_processor = None
            self.training_data_storage = None
            self.pattern_learner = None

    @disabled_on_flag
    def process_feedback_and_learn(self, feedback_list: List[UserFeedback], text_corpus: List[str]):
        ground_truth_pii = self.feedback_processor.extract_pii_from_feedback(feedback_list)
        if not ground_truth_pii:
            logger.info("No new PII confirmed in feedback batch. Skipping pattern discovery.")
            return

        logger.info(f"Starting pattern discovery for {len(ground_truth_pii)} new PII instances.")
        new_patterns = self.pattern_learner.discover_and_validate_patterns(
            text_corpus,
            pii_to_discover=ground_truth_pii,
            ground_truth_pii=ground_truth_pii
        )

        for pattern in new_patterns:
            self.pattern_db.add_or_update_pattern(pattern)

        training_examples = self.feedback_processor.convert_to_training_examples(feedback_list)
        if training_examples:
            self.training_data_storage.save_training_examples(training_examples, source="user_feedback")

    @disabled_on_flag
    def get_adaptive_patterns(self) -> List[Dict[str, Any]]:
        active_patterns = self.pattern_db.get_active_patterns()
        return [pattern.__dict__ for pattern in active_patterns]

    @disabled_on_flag
    def get_model_assignment_for_request(self, request_id: str, test_id: str) -> str:
        return self.ab_test_manager.get_assignment(request_id, test_id)

    @disabled_on_flag
    def create_ab_test(self, name: str, description: str, variant_model_version: str, split: float = 0.5):
        logger.info(f"Received request to create A/B test '{name}'.")
        return self.ab_test_manager.create_test(name, description, variant_model_version, split)

    @disabled_on_flag
    def start_ab_test(self, test_id: str, duration_days: int = 7):
        logger.info(f"Received request to start A/B test {test_id}.")
        self.ab_test_manager.start_test(test_id, duration_days)

    @disabled_on_flag
    def evaluate_and_log_ab_test(self, test_id: str, alpha: float = 0.05):
        logger.info(f"Evaluating and logging result for A/B test {test_id}.")
        try:
            test_result = self.ab_test_manager.evaluate_test(test_id, alpha)
            if self.quality_analyzer:
                self.quality_analyzer.log_ab_test_result(test_result)
            else:
                logger.warning("QualityAnalyzer not available. Skipping logging of A/B test result.")
            return test_result
        except Exception as e:
            logger.error(f"Failed to evaluate or log A/B test {test_id}: {e}", exc_info=True)
            return None

    def close(self):
        """Safely close all database connections managed by the coordinator."""
        if hasattr(self, 'pattern_db') and self.pattern_db:
            self.pattern_db.close()
        if hasattr(self, 'ab_test_manager') and self.ab_test_manager:
            self.ab_test_manager.close()
        if hasattr(self, 'quality_analyzer') and self.quality_analyzer:
            self.quality_analyzer.close()
        logger.info("AdaptiveLearningCoordinator connections shut down.") 