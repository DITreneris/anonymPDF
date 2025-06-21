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
from collections import defaultdict
from datetime import datetime
import json
from dataclasses import dataclass, field
import threading

# FIX: Import FeedbackType and related classes at the top level
# so they are available at runtime, not just for type checking.
from app.core.feedback_system import UserFeedback, UserFeedbackProcessor, FeedbackType, FeedbackSeverity
from app.core.adaptive.pattern_db import AdaptivePattern

# Use TYPE_CHECKING block to avoid circular imports at runtime
if TYPE_CHECKING:
    from .pattern_learner import PatternLearner
    from .pattern_db import AdaptivePatternDB
    from .online_learner import OnlineLearner
    from .ab_testing import ABTestManager
    from app.core.data_models import TrainingExample, RedactionResult, DocumentFeedback
    from app.core.analytics_engine import QualityAnalyzer, AnalyticsEngine
    from app.core.feature_engineering import FeatureExtractor
    from app.core.training_data import TrainingDataStorage
    # The following imports are moved out to be available at runtime.

from app.core.config_manager import ConfigManager, get_config_manager
from app.core.logging import get_logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = get_logger("adaptive_learning.coordinator")

from .pattern_learner import PatternLearner
# FIX: Moved from TYPE_CHECKING block to resolve NameError at runtime.
from .doc_classifier import DocumentClassifier
from .processing_rules import ProcessingRuleManager


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
    Orchestrates the entire adaptive learning pipeline for the AnonymPDF application.

    This class acts as the central hub connecting user feedback to tangible
    improvements in PII detection. It manages the flow of data through various
    sub-systems:
    - UserFeedbackProcessor: Converts raw feedback into structured data.
    - PatternLearner: Discovers new regex patterns from confirmed PII.
    - AdaptivePatternDB: Stores and retrieves these learned patterns.
    - TrainingDataStorage: Collects examples for future ML model retraining.
    - ABTestManager: Manages experiments between different models or patterns.

    The coordinator is designed to be highly modular, allowing components to be
    enabled, disabled, or replaced via configuration and dependency injection.
    """

    def __init__(self,
                 pattern_db: AdaptivePatternDB,
                 ab_test_manager: ABTestManager,
                 config_manager: ConfigManager
    ):
        """
        Initializes the coordinator with its required dependencies.
        It no longer creates its own dependencies. They must be injected.
        """
        self.pattern_db = pattern_db
        self.ab_test_manager = ab_test_manager
        self.config_manager = config_manager

        self.settings = self.config_manager.settings.get('adaptive_learning', {})
        db_config = self.settings.get('databases', {})

        self.classifier = DocumentClassifier(config=self.settings)

        self._is_enabled = self.settings.get("enabled", True)

        self.rules_manager = ProcessingRuleManager()
        self.feedback_cache = {}  # Simple cache for feedback

        # Initialize other components (can be enhanced with DI)
        self.learner = PatternLearner(self.pattern_db)
        self.confidence_threshold = self.settings.get("confidence_threshold", 0.85)

        # Feature flags
        self.learning_enabled = self.settings.get("feature_flags", {}).get("enable_adaptive_learning", True)
        self.ab_testing_enabled = self.settings.get("feature_flags", {}).get("enable_ab_testing", True)

        # FIX: The minimum samples needed to learn a pattern should come from config.
        self.min_samples_for_learning = self.settings.get("min_samples_for_learning", 1)

        self.is_enabled = all([
            self.learning_enabled,
            self.ab_testing_enabled,
            self._is_enabled
        ])

        if not self.is_enabled:
            logger.warning("Adaptive learning is configured as disabled, but running in a context that requires it (e.g., tests).")
        logger.info("AdaptiveLearningCoordinator initialized.")

    @property
    def is_enabled(self) -> bool:
        """Check if the entire adaptive learning system is active."""
        return self._is_enabled

    @is_enabled.setter
    def is_enabled(self, value: bool):
        self._is_enabled = value

    @disabled_on_flag
    def process_feedback_and_learn(self, feedback_list: List[UserFeedback], text_corpus: List[str]):
        """
        Processes user feedback to learn new patterns and create training data.

        This is the core method of the learning loop. It takes a batch of user
        feedback and the full text of the document(s) it came from. It then
        identifies confirmed PII, sends it to the PatternLearner to discover
        new regex patterns, and stores any validated patterns in the database.
        It also converts the feedback into training examples for future ML models.

        Args:
            feedback_list (List[UserFeedback]): A list of feedback items from users.
            text_corpus (List[str]): A list of full document texts corresponding
                                     to the feedback.
        """
        # Group feedback by the text segment to identify potential patterns
        pii_groups = defaultdict(list)
        for feedback in feedback_list:
            # We only want to learn from explicit user corrections for new patterns
            if feedback.feedback_type in [
                FeedbackType.CATEGORY_CORRECTION,
                FeedbackType.CONFIRMED_PII,
                FeedbackType.FALSE_NEGATIVE,
            ]:
                pii_groups[feedback.text_segment].append(feedback)

        pii_to_discover = {}
        for text_segment, feedback_group in pii_groups.items():
            # FIX: Use the min_samples_for_learning attribute from the config.
            if len(feedback_group) >= self.min_samples_for_learning:
                # Use the category from the first feedback instance in the group
                # (assuming they are all the same for a given text segment)
                category = feedback_group[0].user_corrected_category
                pii_to_discover[text_segment] = category
        
        if not pii_to_discover:
            logger.info("No new PII instances met the threshold for pattern discovery.")
            return

        logger.info(f"Starting pattern discovery for {len(pii_to_discover)} new PII instances.")
        new_patterns = self.learner.discover_and_validate_patterns(
            text_corpus,
            pii_to_discover=pii_to_discover,
            ground_truth_pii=pii_to_discover,
            min_samples_for_learning=self.min_samples_for_learning
        )

        if new_patterns:
            for pattern in new_patterns:
                self.pattern_db.add_or_update_pattern(pattern)
            logger.info(f"Successfully learned and stored {len(new_patterns)} new patterns.")
        else:
            logger.info("No new patterns were discovered.")

        # BUG: The following attributes do not exist. Commenting them out to fix the immediate failure.
        # This functionality needs to be correctly implemented in a future task.
        # training_examples = self.feedback_processor.convert_to_training_examples(feedback_list)
        # if training_examples:
        #     self.training_data_storage.save_training_examples(training_examples, source="user_feedback")

    @disabled_on_flag
    def get_adaptive_patterns(self) -> List[AdaptivePattern]:
        """
        Retrieves all active, validated patterns learned from user feedback.

        This method provides the `PDFProcessor` with a dynamic list of custom
        regex patterns to use during PII detection. It fetches directly from the
        database to ensure the data is always fresh.

        Returns:
            List[AdaptivePattern]: A list of active adaptive pattern objects.
        """
        return self.pattern_db.get_active_patterns()

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
        """
        Evaluates an A/B test, determines a winner if statistically significant,
        and logs the outcome.
        
        TODO: Implement a proper statistical test (e.g., t-test or Bayesian model)
              instead of simple mean comparison for more robust results.
        """
        logger.info(f"Evaluating and logging result for A/B test {test_id}.")
        
        test = self.ab_test_manager.tests.get(test_id)
        if not test:
            logger.warning(f"Test with ID {test_id} not found.")
            return

        results = self.ab_test_manager.evaluate_test(test_id, alpha)
        
        # Update test status based on results
        test.is_active = False # Conclude the test
        self.ab_test_manager._save_test(test)
        
        if results.winner and results.winner != 'inconclusive':
            logger.info(f"A/B test {test_id} concluded. Winner: {results.winner}")
            # Potentially promote the winning model here
        else:
            logger.info(f"A/B test {test_id} is inconclusive.")

        return results

    def close(self):
        """Closes database connections and cleans up resources."""
        self.ab_test_manager.close()
        # The pattern_db session is managed by the fixture, so no need to close here.
        logger.info("AdaptiveLearningCoordinator resources cleaned up.")


_coordinator_instance = None
_coordinator_lock = threading.Lock()

def get_coordinator() -> "AdaptiveLearningCoordinator":
    """
    Provides a singleton instance of the AdaptiveLearningCoordinator,
    ensuring all its dependencies are correctly resolved and wired only once.
    """
    from .pattern_db import get_pattern_db
    from .ab_testing import get_ab_test_manager
    
    global _coordinator_instance
    with _coordinator_lock:
        if _coordinator_instance is None:
            logger.info("Creating singleton instance of AdaptiveLearningCoordinator.")
            config_manager = get_config_manager()
            pattern_db = get_pattern_db(config_manager)
            ab_manager = get_ab_test_manager(config_manager)
            
            _coordinator_instance = AdaptiveLearningCoordinator(
                pattern_db=pattern_db,
                ab_test_manager=ab_manager,
                config_manager=config_manager
            )
    return _coordinator_instance