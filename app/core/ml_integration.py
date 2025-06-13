"""
ML Integration Layer for Priority 3 Implementation

This module provides seamless integration between the ML engine and existing
Priority 1 & 2 PII detection systems, with fallback mechanisms and performance tracking.
"""

import time
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
import uuid

# Import existing components
if TYPE_CHECKING:
    from app.core.ml_engine import MLConfidenceScorer, create_ml_confidence_scorer
from app.core.data_models import MLPrediction
from app.core.feature_engineering import FeatureExtractor, create_feature_extractor
from app.core.training_data import TrainingDataCollector, create_training_data_collector
from app.core.context_analyzer import ContextualValidator, DetectionContext, ConfidenceLevel
from app.core.config_manager import get_config
from app.core.logging import get_logger
from app.core.feedback_system import UserFeedback, FeedbackType, FeedbackSeverity, create_feedback_system
# Avoid circular import - import QualityAnalyzer lazily when needed

integration_logger = get_logger("ml_integration")


@dataclass
class DetectionResult:
    """Enhanced detection result with ML confidence and fallback information."""
    text: str
    category: str
    context: str
    position: int
    
    # ML-specific fields
    ml_confidence: float
    ml_prediction: Optional["MLPrediction"] = None
    
    # Fallback fields
    priority2_confidence: float = 0.0
    fallback_used: bool = False
    
    # Performance tracking
    processing_time_ms: float = 0.0
    features_extracted: int = 0
    
    # Metadata
    document_type: Optional[str] = None
    language: str = 'lt'
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerformanceMetrics:
    """Performance metrics for ML integration."""
    total_detections: int = 0
    ml_predictions: int = 0
    fallback_predictions: int = 0
    average_processing_time_ms: float = 0.0
    accuracy_estimate: float = 0.0
    confidence_correlation: float = 0.0
    
    def get_ml_usage_ratio(self) -> float:
        """Get ratio of ML predictions vs fallback."""
        return self.ml_predictions / self.total_detections if self.total_detections > 0 else 0.0


class MLIntegrationLayer:
    """Main integration layer between ML engine and existing PII detection systems."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config().get('ml_integration', {})
        
        # Initialize components
        from app.core.ml_engine import create_ml_confidence_scorer
        self.ml_scorer: "MLConfidenceScorer" = create_ml_confidence_scorer()
        self.feature_extractor = create_feature_extractor()
        self.training_collector = create_training_data_collector()
        self.contextual_validator = ContextualValidator()  # Priority 2 fallback
        self.user_feedback_system = create_feedback_system()
        self.quality_analyzer = None  # Lazy import to avoid circular dependencies
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.recent_results = []  # Store recent results for analysis
        self.max_recent_results = self.config.get('max_recent_results', 1000)
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        self._lock = threading.Lock()
        
        # A/B testing configuration
        self.ab_testing_enabled = self.config.get('ab_testing_enabled', False)
        self.ml_traffic_ratio = self.config.get('ml_traffic_ratio', 1.0)  # 1.0 = 100% ML
        
        integration_logger.info("ML Integration Layer initialized")
    
    def _get_quality_analyzer(self):
        """Get QualityAnalyzer instance with lazy import to avoid circular dependencies."""
        if self.quality_analyzer is None:
            from app.core.analytics_engine import QualityAnalyzer
            self.quality_analyzer = QualityAnalyzer()
        return self.quality_analyzer
    
    def detect_with_ml_integration(self, text: str, category: str, context: str,
                                 position: int = -1, document_type: str = None,
                                 language: str = 'lt', force_ml: bool = False) -> DetectionResult:
        """
        Main detection method with ML integration and fallback.
        
        Args:
            text: The detected text
            category: PII category
            context: Surrounding context
            position: Position in document
            document_type: Type of document
            language: Document language
            force_ml: Force ML prediction (ignore A/B testing)
            
        Returns:
            DetectionResult with ML confidence and metadata
        """
        start_time = time.time()
        
        try:
            # Determine if we should use ML (A/B testing logic)
            use_ml = force_ml or self._should_use_ml()
            
            if use_ml:
                result = self._detect_with_ml(text, category, context, position, 
                                            document_type, language)
            else:
                result = self._detect_with_fallback(text, category, context, position,
                                                  document_type, language)
            
            # Set processing time
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Store result for analysis
            self._store_recent_result(result)
            
            if result:
                try:
                    # Attempt to set a pattern_type attribute on the result object
                    # This is a placeholder and should ideally be part of DetectionResult population logic
                    pattern_type_str = "unknown"
                    if result.ml_prediction and result.ml_prediction.model_version:
                        pattern_type_str = f"ml_model:{result.ml_prediction.model_version}"
                    elif result.fallback_used:
                        # Try to infer from category or other context if a specific rule name isn't available
                        pattern_type_str = f"fallback_rule_for:{result.category}"
                    
                    # Dynamically add pattern_type to result for QualityAnalyzer.
                    # A more robust solution is to add 'pattern_type' to the DetectionResult dataclass.
                    setattr(result, 'pattern_type', pattern_type_str) 

                    self._get_quality_analyzer().add_detection_result(result, ground_truth=None)
                    integration_logger.debug(f"DetectionResult for '{result.text}' sent to QualityAnalyzer.")
                except Exception as qa_exc:
                    integration_logger.error(f"Failed to send result to QualityAnalyzer: {qa_exc}", exc_info=True)
            
            return result
            
        except Exception as e:
            integration_logger.error(f"Detection failed, using fallback: {e}")
            return self._detect_with_fallback(text, category, context, position,
                                            document_type, language, error=True)
    
    def _detect_with_ml(self, text: str, category: str, context: str,
                       position: int, document_type: str, language: str) -> DetectionResult:
        """Perform detection using ML engine."""
        # Extract features
        feature_set = self.feature_extractor.extract_all_features(
            detection_text=text,
            category=category,
            context=context,
            position=position,
            document_type=document_type,
            language=language
        )
        
        features = feature_set.to_dict()
        
        # Get ML prediction
        ml_prediction = self.ml_scorer.calculate_ml_confidence(
            detection=text,
            context=context,
            features=features,
            document_type=document_type
        )
        
        # Get Priority 2 confidence for comparison/fallback
        priority2_confidence = self.contextual_validator.calculate_confidence(
            text, category, context
        )
        
        # Use ML confidence as primary
        final_confidence = ml_prediction.confidence
        
        result = DetectionResult(
            text=text,
            category=category,
            context=context,
            position=position,
            ml_confidence=final_confidence,
            ml_prediction=ml_prediction,
            priority2_confidence=priority2_confidence,
            fallback_used=False,
            features_extracted=len(features),
            document_type=document_type,
            language=language
        )
        
        integration_logger.debug(f"ML detection: {text} -> {final_confidence:.3f}")
        return result
    
    def _detect_with_fallback(self, text: str, category: str, context: str,
                            position: int, document_type: str, language: str,
                            error: bool = False) -> DetectionResult:
        """Perform detection using Priority 2 fallback."""
        priority2_confidence = self.contextual_validator.calculate_confidence(
            text, category, context
        )
        
        result = DetectionResult(
            text=text,
            category=category,
            context=context,
            position=position,
            ml_confidence=priority2_confidence,  # Use Priority 2 as ML confidence
            priority2_confidence=priority2_confidence,
            fallback_used=True,
            features_extracted=0,
            document_type=document_type,
            language=language
        )
        
        if error:
            integration_logger.warning(f"Error fallback: {text} -> {priority2_confidence:.3f}")
        else:
            integration_logger.debug(f"A/B fallback: {text} -> {priority2_confidence:.3f}")
        
        return result
    
    def _should_use_ml(self) -> bool:
        """Determine if ML should be used based on A/B testing configuration."""
        if not self.ab_testing_enabled:
            return True
        
        import random
        return random.random() < self.ml_traffic_ratio
    
    def _update_performance_metrics(self, result: DetectionResult):
        """Update performance metrics with new result."""
        with self._lock:
            self.performance_metrics.total_detections += 1
            
            if result.fallback_used:
                self.performance_metrics.fallback_predictions += 1
            else:
                self.performance_metrics.ml_predictions += 1
            
            # Update average processing time
            total_time = (self.performance_metrics.average_processing_time_ms * 
                         (self.performance_metrics.total_detections - 1) + 
                         result.processing_time_ms)
            self.performance_metrics.average_processing_time_ms = (
                total_time / self.performance_metrics.total_detections
            )
    
    def _store_recent_result(self, result: DetectionResult):
        """Store result for recent analysis."""
        with self._lock:
            self.recent_results.append(result)
            
            # Keep only recent results
            if len(self.recent_results) > self.max_recent_results:
                self.recent_results = self.recent_results[-self.max_recent_results:]
    
    def batch_detect(self, detections: List[Dict[str, Any]]) -> List[DetectionResult]:
        """
        Process multiple detections in batch for better performance.
        
        Args:
            detections: List of detection dictionaries with required fields
            
        Returns:
            List of DetectionResult objects
        """
        if not detections:
            return []
        
        start_time = time.time()
        results = []
        
        # Use thread pool for parallel processing
        future_to_detection = {}
        
        for detection in detections:
            future = self.executor.submit(
                self.detect_with_ml_integration,
                detection.get('text', ''),
                detection.get('category', 'unknown'),
                detection.get('context', ''),
                detection.get('position', -1),
                detection.get('document_type'),
                detection.get('language', 'lt')
            )
            future_to_detection[future] = detection
        
        # Collect results
        for future in future_to_detection:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                integration_logger.error(f"Batch detection failed: {e}")
                # Add fallback result
                detection = future_to_detection[future]
                fallback_result = self._detect_with_fallback(
                    detection.get('text', ''),
                    detection.get('category', 'unknown'),
                    detection.get('context', ''),
                    detection.get('position', -1),
                    detection.get('document_type'),
                    detection.get('language', 'lt'),
                    error=True
                )
                results.append(fallback_result)
        
        batch_time = (time.time() - start_time) * 1000
        integration_logger.info(f"Batch processed {len(detections)} detections in {batch_time:.1f}ms")
        
        return results
    
    def add_user_feedback(self, result: DetectionResult, user_confirmed: bool,
                         correct_category: Optional[str] = None, confidence_rating: Optional[float] = None,
                         user_comment: Optional[str] = None):
        """Adds user feedback to the feedback system."""
        
        feedback_type = FeedbackType.CORRECT_DETECTION
        if not user_confirmed:
            feedback_type = FeedbackType.FALSE_POSITIVE
        elif correct_category and correct_category.upper() != result.category.upper():
            feedback_type = FeedbackType.CATEGORY_CORRECTION

        # Populate context for deeper analysis
        context_details = {
            'surrounding_text': result.context,
            'position': result.position,
            'original_document_type': result.document_type,
            'language': result.language,
            'priority2_confidence': result.priority2_confidence,
            'ml_prediction_details': result.ml_prediction.to_dict() if result.ml_prediction else None,
            'original_timestamp': result.timestamp.isoformat()
        }

        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            document_id=result.document_type or "unknown_doc",
            text_segment=result.text,
            detected_category=result.category,
            user_corrected_category=correct_category,
            detected_confidence=result.ml_confidence,
            user_confidence_rating=confidence_rating,
            feedback_type=feedback_type,
            severity=FeedbackSeverity.MEDIUM, # Default severity
            user_comment=user_comment,
            context=context_details,
        )
        
        # Use the feedback system instance to submit
        success = self.user_feedback_system.submit_feedback(feedback)
        
        if not success:
            integration_logger.error(f"Failed to submit feedback: {feedback.feedback_id}")

    def retrain_model_if_needed(self) -> bool:
        """
        Check if model retraining is needed and trigger if necessary.
        
        Returns:
            True if retraining was triggered
        """
        # Get current training data stats
        stats = self.training_collector.storage.get_dataset_stats()
        
        # Check retraining criteria
        should_retrain = (
            stats.total_samples >= self.config.get('min_samples_for_retrain', 100) and
            stats.total_samples % self.config.get('retrain_interval', 500) == 0
        )
        
        if should_retrain:
            integration_logger.info("Triggering model retraining")
            
            # Get balanced training set
            training_examples, _ = self.training_collector.get_balanced_training_set(
                max_samples=self.config.get('max_training_samples', 5000)
            )
            
            # Retrain model
            metrics = self.ml_scorer.train_model(training_examples)
            
            integration_logger.info(f"Model retrained: {metrics}")
            return True
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        with self._lock:
            recent_count = min(len(self.recent_results), 100)
            recent_results = self.recent_results[-recent_count:] if recent_count > 0 else []
            
            # Calculate confidence correlation if we have both ML and Priority 2 results
            ml_confidences = [r.ml_confidence for r in recent_results if not r.fallback_used]
            p2_confidences = [r.priority2_confidence for r in recent_results if not r.fallback_used]
            
            correlation = 0.0
            if len(ml_confidences) > 1 and len(p2_confidences) > 1:
                import numpy as np
                try:
                    correlation = float(np.corrcoef(ml_confidences, p2_confidences)[0, 1])
                except:
                    correlation = 0.0
            
            return {
                'total_detections': self.performance_metrics.total_detections,
                'ml_usage_ratio': self.performance_metrics.get_ml_usage_ratio(),
                'average_processing_time_ms': self.performance_metrics.average_processing_time_ms,
                'confidence_correlation': correlation,
                'recent_sample_size': recent_count,
                'ml_model_version': self.ml_scorer.model_version,
                'features_count': len(self.feature_extractor.feature_names) if hasattr(self.feature_extractor, 'feature_names') else 0
            }
    
    def enable_ab_testing(self, ml_ratio: float = 0.5):
        """Enable A/B testing with specified ML traffic ratio."""
        self.ab_testing_enabled = True
        self.ml_traffic_ratio = ml_ratio
        integration_logger.info(f"A/B testing enabled: {ml_ratio*100}% ML traffic")
    
    def disable_ab_testing(self):
        """Disable A/B testing (use ML for all traffic)."""
        self.ab_testing_enabled = False
        self.ml_traffic_ratio = 1.0
        integration_logger.info("A/B testing disabled: 100% ML traffic")
    
    def shutdown(self):
        """Clean shutdown of integration layer."""
        self.executor.shutdown(wait=True)
        integration_logger.info("ML Integration Layer shut down.")


# Factory function for easy integration
def create_ml_integration_layer(config: Optional[Dict] = None) -> MLIntegrationLayer:
    """Factory function to create an MLIntegrationLayer instance."""
    return MLIntegrationLayer(config=config) 