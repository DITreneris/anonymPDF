"""
Online Learner for the Adaptive Learning System.
"""

from typing import Dict, List, Any, Optional

from app.core.ml_training_pipeline import MLTrainingPipeline, create_ml_training_pipeline
from app.core.training_data import TrainingExample, TrainingDataStorage
from app.core.logging import get_logger

logger = get_logger("adaptive_learning.online_learner")


class OnlineLearner:
    """
    Handles the online retraining of ML models based on new training examples.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # This dependency will be injected by the coordinator or a factory
        self.training_data_storage = TrainingDataStorage()
        # Retraining will be handled by the existing, robust MLTrainingPipeline
        self.training_pipeline: MLTrainingPipeline = create_ml_training_pipeline()
        self.retrain_threshold = self.config.get('retrain_threshold', 50)
        logger.info("OnlineLearner initialized.")

    def retrain_model_if_needed(self, new_examples: List[TrainingExample]) -> Optional[Dict[str, Any]]:
        """
        Triggers a model retraining cycle if enough new examples have been collected.

        Args:
            new_examples: A list of new training examples derived from user feedback.

        Returns:
            A dictionary with the retraining results, or None if retraining was not triggered.
        """
        if not new_examples:
            return None
            
        # First, persist the new examples for long-term storage and batch retraining
        try:
            self.training_data_storage.save_training_examples(new_examples, source="online_feedback")
            logger.info(f"Successfully saved {len(new_examples)} new training examples to persistent storage.")
        except Exception as e:
            logger.error(f"Failed to save new training examples to storage: {e}", exc_info=True)
            # We can decide whether to continue with retraining or not. For now, we'll continue.

        if len(new_examples) < self.retrain_threshold:
            logger.info(f"Not enough new examples ({len(new_examples)}) to trigger retraining. Threshold is {self.retrain_threshold}.")
            return None

        logger.info(f"Threshold of {self.retrain_threshold} met. Starting online retraining with {len(new_examples)} examples.")

        # The MLTrainingPipeline already knows how to handle training, validation, and deployment.
        # We can just call its main execution method.
        try:
            # The 'run_training_cycle' method should be designed to accept external data.
            # Assuming it can for this refactoring.
            training_results = self.training_pipeline.run_training_cycle(
                external_training_data=new_examples
            )

            logger.info("Online model retraining completed successfully.")
            return training_results
        except Exception as e:
            logger.error(f"An error occurred during online model retraining: {e}", exc_info=True)
            return {"error": str(e)} 