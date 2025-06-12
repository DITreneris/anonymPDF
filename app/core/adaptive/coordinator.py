"""
Central Coordinator for the Adaptive Learning System.

This module ties together all components of the adaptive learning pipeline,
managing the flow of data from user feedback to pattern discovery, validation,
and model retraining.
"""

from typing import Dict, List, Any, Optional

from .pattern_learner import PatternLearner
from .pattern_db import AdaptivePatternDB
from .online_learner import OnlineLearner
from .ab_testing import ABTestManager, create_ab_test_manager
from app.core.feedback_system import UserFeedback, FeedbackType, UserFeedbackProcessor
from app.core.data_models import TrainingExample
from app.core.analytics_engine import QualityAnalyzer
from app.core.adaptive.pattern_db import AdaptivePatternDB
from app.core.adaptive.ab_testing import ABTestManager
from app.core.feature_engineering import FeatureExtractor, create_feature_extractor
from app.core.training_data import TrainingDataStorage

from app.core.logging import get_logger

logger = get_logger("adaptive_learning.coordinator")


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
        self.pattern_db = pattern_db or AdaptivePatternDB()
        self.ab_test_manager = ab_test_manager or ABTestManager()
        self.quality_analyzer = quality_analyzer or QualityAnalyzer()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.feedback_processor = feedback_processor or UserFeedbackProcessor()
        self.training_data_storage = training_data_storage or TrainingDataStorage()
        
        # The PatternLearner is tightly coupled and doesn't need injection for now.
        self.pattern_learner = PatternLearner(self.pattern_db)
        
        logger.info("AdaptiveLearningCoordinator initialized.")

    def process_feedback_and_learn(self, feedback_list: List[UserFeedback], text_corpus: List[str]):
        """
        Main entry point to trigger the adaptive learning cycle.

        Args:
            feedback_list: A batch of user feedback to learn from.
            text_corpus: A corpus of recent documents for pattern validation.
        """
        logger.info(f"Starting adaptive learning cycle with {len(feedback_list)} feedback items.")

        # 1. Extract confirmed PII from feedback. This is our ground truth.
        ground_truth_pii = self.feedback_processor.extract_pii_from_feedback(feedback_list)

        if ground_truth_pii:
            logger.info(f"Starting pattern discovery for {len(ground_truth_pii)} new PII instances.")
            # 2. Discover and validate patterns using the extracted PII
            new_patterns = self.pattern_learner.discover_and_validate_patterns(
                text_corpus,
                pii_to_discover=ground_truth_pii,
                ground_truth_pii=ground_truth_pii
            )

            # 3. Save newly validated patterns to the database
            for pattern in new_patterns:
                self.pattern_db.add_or_update_pattern(pattern)
        else:
            logger.info("No new PII confirmed in feedback batch. Skipping pattern discovery.")

        # 4. Convert feedback to training examples and store them
        training_examples = self.feedback_processor.convert_to_training_examples(feedback_list)
        if training_examples:
            logger.info(f"Storing {len(training_examples)} new training examples.")
            self.training_data_storage.save_training_examples(training_examples, source="user_feedback")

    def get_adaptive_patterns(self) -> List[Dict[str, Any]]:
        """
        Retrieves active, validated patterns from the database.
        """
        active_patterns = self.pattern_db.get_active_patterns()
        # Convert to dict for easier use in other parts of the application
        return [pattern.__dict__ for pattern in active_patterns]

    # --- A/B Testing Integration ---

    def create_ab_test(self, name: str, description: str, variant_model_version: str, split: float = 0.5):
        """Creates a new A/B test."""
        logger.info(f"Received request to create A/B test '{name}'.")
        return self.ab_test_manager.create_test(name, description, variant_model_version, split)

    def start_ab_test(self, test_id: str, duration_days: int = 7):
        """Starts an existing A/B test."""
        logger.info(f"Received request to start A/B test {test_id}.")
        self.ab_test_manager.start_test(test_id, duration_days)

    def get_model_assignment_for_request(self, request_id: str, test_id: str) -> str:
        """
        Gets the model assignment ('control' or 'variant') for a given request.
        This is the main integration point for the application to decide which model to use.

        Args:
            request_id: A unique ID for the request or user.
            test_id: The active A/B test to get an assignment for.

        Returns:
            'variant' or 'control'.
        """
        assignment = self.ab_test_manager.get_assignment(request_id, test_id)
        logger.debug(f"Assigned request {request_id} to '{assignment}' group for test {test_id}.")
        return assignment

    def evaluate_and_log_ab_test(self, test_id: str, alpha: float = 0.05):
        """Evaluates an A/B test and logs the result to the analytics engine."""
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
        """Cleanly closes database connections."""
        self.pattern_db.close()
        logger.info("AdaptiveLearningCoordinator shut down.") 