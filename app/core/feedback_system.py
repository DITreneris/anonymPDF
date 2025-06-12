"""
User Feedback System for Priority 3 Implementation

This module provides comprehensive user feedback collection, processing,
and integration with the ML training pipeline for continuous improvement.
"""

import threading
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
from enum import Enum
import statistics
import uuid

# Import existing components
from app.core.training_data import TrainingDataCollector, TrainingExample
from app.core.feature_engineering import FeatureExtractor
from app.core.config_manager import get_config
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.ml_engine import MLConfidenceScorer
    from app.core.ml_training_pipeline import MLTrainingPipeline

feedback_logger = get_logger("anonympdf.feedback_system")


class FeedbackType(Enum):
    """Types of user feedback."""
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    CORRECT_DETECTION = "correct_detection"
    CATEGORY_CORRECTION = "category_correction"
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"


class FeedbackSeverity(Enum):
    """Severity levels for feedback."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UserFeedback:
    """Represents user feedback on detection results."""
    feedback_id: str
    document_id: str
    text_segment: str
    detected_category: Optional[str]
    user_corrected_category: Optional[str]
    detected_confidence: float
    user_confidence_rating: Optional[float]
    feedback_type: FeedbackType
    severity: FeedbackSeverity
    user_comment: Optional[str]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False


@dataclass
class FeedbackStats:
    """Statistics about user feedback."""
    total_feedback_count: int
    feedback_by_type: Dict[str, int]
    feedback_by_category: Dict[str, int]
    avg_user_confidence: float
    improvement_suggestions: List[str]
    last_updated: datetime = field(default_factory=datetime.now)


class FeedbackAnalyzer:
    """Analyzes user feedback patterns and generates insights."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Analysis parameters
        self.min_feedback_threshold = self.config.get('min_feedback_threshold', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Feedback tracking
        self.feedback_history = deque(maxlen=1000)
        self.category_performance = defaultdict(list)
        self.pattern_issues = defaultdict(int)
        
        self._lock = threading.Lock()
        
        feedback_logger.info("FeedbackAnalyzer initialized")
    
    def add_feedback(self, feedback: UserFeedback):
        """Add user feedback for analysis."""
        with self._lock:
            self.feedback_history.append(feedback)
            
            # Track category performance
            if feedback.detected_category:
                self.category_performance[feedback.detected_category].append({
                    'feedback_type': feedback.feedback_type,
                    'detected_confidence': feedback.detected_confidence,
                    'user_confidence': feedback.user_confidence_rating,
                    'timestamp': feedback.timestamp
                })
            
            # Track pattern issues
            if feedback.feedback_type in [FeedbackType.FALSE_POSITIVE, FeedbackType.FALSE_NEGATIVE]:
                self.pattern_issues[feedback.detected_category or 'unknown'] += 1
    
    def analyze_feedback_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze feedback patterns over a time window."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        with self._lock:
            recent_feedback = [
                fb for fb in self.feedback_history 
                if fb.timestamp >= cutoff_time
            ]
        
        if not recent_feedback:
            return {'message': 'No feedback in specified time window'}
        
        # Analyze by feedback type
        feedback_by_type = defaultdict(int)
        feedback_by_category = defaultdict(int)
        confidence_ratings = []
        
        for feedback in recent_feedback:
            feedback_by_type[feedback.feedback_type.value] += 1
            if feedback.detected_category:
                feedback_by_category[feedback.detected_category] += 1
            if feedback.user_confidence_rating:
                confidence_ratings.append(feedback.user_confidence_rating)
        
        # Calculate statistics
        avg_user_confidence = statistics.mean(confidence_ratings) if confidence_ratings else 0.0
        
        # Identify problematic categories
        problem_categories = []
        for category, feedback_list in self.category_performance.items():
            recent_category_feedback = [
                fb for fb in feedback_list 
                if fb['timestamp'] >= cutoff_time
            ]
            
            if len(recent_category_feedback) >= self.min_feedback_threshold:
                false_positive_rate = sum(
                    1 for fb in recent_category_feedback 
                    if fb['feedback_type'] == FeedbackType.FALSE_POSITIVE
                ) / len(recent_category_feedback)
                
                if false_positive_rate > 0.3:  # 30% false positive rate
                    problem_categories.append({
                        'category': category,
                        'false_positive_rate': false_positive_rate,
                        'feedback_count': len(recent_category_feedback)
                    })
        
        return {
            'time_window_hours': time_window_hours,
            'total_feedback': len(recent_feedback),
            'feedback_by_type': dict(feedback_by_type),
            'feedback_by_category': dict(feedback_by_category),
            'avg_user_confidence': avg_user_confidence,
            'problem_categories': problem_categories,
            'pattern_issues': dict(self.pattern_issues)
        }
    
    def generate_improvement_suggestions(self) -> List[str]:
        """Generate improvement suggestions based on feedback analysis."""
        suggestions = []
        
        analysis = self.analyze_feedback_patterns()
        
        # Check false positive rate
        false_positives = analysis['feedback_by_type'].get('false_positive', 0)
        total_feedback = analysis['total_feedback']
        
        if total_feedback > 0:
            false_positive_rate = false_positives / total_feedback
            
            if false_positive_rate > 0.2:
                suggestions.append(
                    f"High false positive rate ({false_positive_rate:.1%}). "
                    "Consider increasing confidence thresholds or refining pattern matching."
                )
        
        # Check user confidence
        if analysis['avg_user_confidence'] < 0.6:
            suggestions.append(
                f"Low user confidence rating ({analysis['avg_user_confidence']:.2f}). "
                "Review detection quality and consider model retraining."
            )
        
        # Check problem categories
        if analysis['problem_categories']:
            categories = [cat['category'] for cat in analysis['problem_categories']]
            suggestions.append(
                f"Categories with high error rates: {', '.join(categories)}. "
                "Focus training data collection on these categories."
            )
        
        # Check pattern issues
        pattern_issues = analysis['pattern_issues']
        if pattern_issues:
            most_problematic = max(pattern_issues.items(), key=lambda x: x[1])
            suggestions.append(
                f"Pattern '{most_problematic[0]}' has {most_problematic[1]} issues. "
                "Review and update pattern definitions."
            )
        
        return suggestions


class UserFeedbackProcessor:
    """Processes user feedback and integrates with training pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize components
        self.training_data_collector = TrainingDataCollector()
        self.feature_extractor = FeatureExtractor()
        self.feedback_analyzer = FeedbackAnalyzer(self.config.get('analyzer', {}))
        
        # Processing configuration
        self.auto_retrain_threshold = self.config.get('auto_retrain_threshold', 50)
        self.batch_size = self.config.get('batch_size', 10)
        
        # Processing state
        self.pending_feedback = deque()
        self.processed_count = 0
        self.last_retrain_time = datetime.now()
        
        self._lock = threading.Lock()
        
        feedback_logger.info("UserFeedbackProcessor initialized")
    
    def extract_pii_from_feedback(self, feedback: List[UserFeedback]) -> Dict[str, str]:
        """
        Extracts confirmed PII text and categories from user feedback.
        This is used to find ground truth for pattern discovery.
        """
        confirmed_pii = {}
        for fb in feedback:
            # Consider corrections and confirmations as ground truth
            if fb.feedback_type == FeedbackType.CATEGORY_CORRECTION and fb.user_corrected_category:
                confirmed_pii[fb.text_segment] = fb.user_corrected_category
            elif fb.feedback_type == FeedbackType.CORRECT_DETECTION and fb.detected_category:
                confirmed_pii[fb.text_segment] = fb.detected_category
        return confirmed_pii

    def convert_to_training_examples(self, feedback_list: List[UserFeedback]) -> List[TrainingExample]:
        """
        Converts a list of user feedback into a list of training examples.
        """
        examples = []
        for fb in feedback_list:
            example = self._feedback_to_training_example(fb)
            if example:
                examples.append(example)
        return examples
    
    def submit_feedback(self, feedback: UserFeedback) -> bool:
        """Submit user feedback for processing."""
        try:
            # Validate feedback
            if not self._validate_feedback(feedback):
                feedback_logger.warning(f"Invalid feedback submitted: {feedback.feedback_id}")
                return False
            
            # Add to analyzer
            self.feedback_analyzer.add_feedback(feedback)
            
            # Add to processing queue
            with self._lock:
                self.pending_feedback.append(feedback)
            
            feedback_logger.info(
                f"Feedback submitted: {feedback.feedback_type.value} for {feedback.detected_category}",
                feedback_id=feedback.feedback_id,
                document_id=feedback.document_id
            )
            
            return True
            
        except Exception as e:
            feedback_logger.error(f"Error submitting feedback: {e}")
            return False
    
    def _validate_feedback(self, feedback: UserFeedback) -> bool:
        """Validate user feedback data."""
        # Check required fields
        if not feedback.feedback_id or not feedback.document_id:
            return False
        
        if not feedback.text_segment:
            return False
        
        # Check confidence values
        if not (0.0 <= feedback.detected_confidence <= 1.0):
            return False
        
        if feedback.user_confidence_rating and not (0.0 <= feedback.user_confidence_rating <= 1.0):
            return False
        
        return True
    
    def process_pending_feedback(self) -> Dict[str, Any]:
        """Process pending user feedback."""
        processed_feedback = []
        
        with self._lock:
            # Process feedback in batches
            while len(processed_feedback) < self.batch_size and self.pending_feedback:
                feedback = self.pending_feedback.popleft()
                processed_feedback.append(feedback)
        
        if not processed_feedback:
            return {'processed_count': 0, 'training_examples_created': 0}
        
        # Convert feedback to training examples
        training_examples = self.convert_to_training_examples(processed_feedback)
        
        for feedback in processed_feedback:
            feedback.processed = True
            self.processed_count += 1
            
        if training_examples:
            # Add to training data storage
            self.training_data_collector.storage.save_training_examples(training_examples, 'user_feedback')

        # Check if we should trigger retraining
        should_retrain = self._should_trigger_retrain()
        
        feedback_logger.info(
            f"Processed {len(processed_feedback)} feedback items, "
            f"created {len(training_examples)} training examples",
            should_retrain=should_retrain
        )
        
        return {
            'processed_count': len(processed_feedback),
            'training_examples_created': len(training_examples),
            'should_retrain': should_retrain,
            'total_processed': self.processed_count
        }
    
    def _feedback_to_training_example(self, feedback: UserFeedback) -> Optional[TrainingExample]:
        """Convert user feedback to training example."""
        try:
            # Determine the correct label based on feedback
            is_pii = False
            confidence = 0.0
            correct_category = None

            if feedback.feedback_type == FeedbackType.FALSE_POSITIVE:
                is_pii = False
                confidence = feedback.user_confidence_rating or 0.1
                correct_category = feedback.detected_category or 'UNKNOWN'
            elif feedback.feedback_type in [FeedbackType.FALSE_NEGATIVE, FeedbackType.CORRECT_DETECTION, FeedbackType.CATEGORY_CORRECTION]:
                is_pii = True
                confidence = feedback.user_confidence_rating or 0.95
                if feedback.user_corrected_category:
                    correct_category = feedback.user_corrected_category
                else:
                    correct_category = feedback.detected_category
            else:
                # Other feedback types might not be suitable for creating training examples
                return None

            if not correct_category:
                return None

            full_text = feedback.context.get('full_text', '')
            surrounding_context = feedback.context.get('surrounding_text', feedback.context.get('context', ''))

            # Extract features from text
            features = self.feature_extractor.extract_all_features(
                detection_text=feedback.text_segment,
                category=correct_category,
                context=surrounding_context,
                full_text=full_text,
                position=feedback.context.get('position', -1),
                document_type=feedback.document_id,
                language=feedback.context.get('language', 'en')
            ).to_dict()
            
            # Create training example
            training_example = TrainingExample(
                detection_text=feedback.text_segment,
                category=correct_category,
                context=surrounding_context,
                features=features,
                confidence_score=confidence,
                is_true_positive=is_pii,
                document_type=feedback.document_id,
                metadata={
                    'feedback_id': feedback.feedback_id,
                    'feedback_type': feedback.feedback_type.value,
                    'original_category': feedback.detected_category,
                    'original_confidence': feedback.detected_confidence,
                    'user_comment': feedback.user_comment,
                    'source': 'user_feedback'
                }
            )
            
            return training_example
            
        except Exception as e:
            feedback_logger.error(f"Error converting feedback to training example: {e}", exc_info=True)
            return None
    
    def _should_trigger_retrain(self) -> bool:
        """Determine if model retraining should be triggered."""
        # Check if enough feedback has been processed
        if self.processed_count >= self.auto_retrain_threshold:
            return True
        
        # Check time since last retrain
        time_since_retrain = datetime.now() - self.last_retrain_time
        if time_since_retrain > timedelta(days=7):  # Weekly retraining
            return True
        
        # Check feedback quality indicators
        analysis = self.feedback_analyzer.analyze_feedback_patterns(time_window_hours=24)
        if analysis.get('total_feedback', 0) > 10:
            false_positive_rate = analysis['feedback_by_type'].get('false_positive', 0) / analysis['total_feedback']
            if false_positive_rate > 0.3:  # High error rate
                return True
        
        return False
    
    def get_feedback_stats(self) -> FeedbackStats:
        """Get comprehensive feedback statistics."""
        analysis = self.feedback_analyzer.analyze_feedback_patterns(time_window_hours=168)  # 1 week
        suggestions = self.feedback_analyzer.generate_improvement_suggestions()
        
        return FeedbackStats(
            total_feedback_count=analysis.get('total_feedback', 0),
            feedback_by_type=analysis.get('feedback_by_type', {}),
            feedback_by_category=analysis.get('feedback_by_category', {}),
            avg_user_confidence=analysis.get('avg_user_confidence', 0.0),
            improvement_suggestions=suggestions
        )


class UserFeedbackSystem:
    """Main user feedback system coordinating all components."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initializes the feedback system with all its components."""
        from app.core.ml_training_pipeline import create_ml_training_pipeline
        self.config = config or get_config().get('user_feedback', {})
        
        # Initialize components
        self.processor = UserFeedbackProcessor(self.config.get('processor', {}))
        self.analyzer = self.processor.feedback_analyzer
        self.ml_training_pipeline: "MLTrainingPipeline" = create_ml_training_pipeline()
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'data/user_feedback.db'))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_storage()
        
        # Background processing
        self.auto_process = self.config.get('auto_process', True)
        self.processing_interval = self.config.get('processing_interval', 300)  # 5 minutes
        self.processing_thread = None
        self.is_processing = False
        
        feedback_logger.info("UserFeedbackSystem initialized")
    
    def _init_storage(self):
        """Initialize database storage for feedback."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL,
                    text_segment TEXT NOT NULL,
                    detected_category TEXT,
                    user_corrected_category TEXT,
                    detected_confidence REAL,
                    user_confidence_rating REAL,
                    feedback_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_comment TEXT,
                    context TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON user_feedback(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_processed ON user_feedback(processed)')
    
    def submit_feedback(self, feedback: UserFeedback) -> bool:
        """Submit user feedback to the system."""
        try:
            # Store in database
            self._store_feedback(feedback)
            
            # Process through processor
            success = self.processor.submit_feedback(feedback)
            
            if success and self.auto_process:
                # Trigger background processing
                self._trigger_processing()
            
            return success
            
        except Exception as e:
            feedback_logger.error(f"Error in feedback submission: {e}")
            return False
    
    def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO user_feedback 
                (feedback_id, document_id, text_segment, detected_category, 
                 user_corrected_category, detected_confidence, user_confidence_rating,
                 feedback_type, severity, user_comment, context, timestamp, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.feedback_id,
                feedback.document_id,
                feedback.text_segment,
                feedback.detected_category,
                feedback.user_corrected_category,
                feedback.detected_confidence,
                feedback.user_confidence_rating,
                feedback.feedback_type.value,
                feedback.severity.value,
                feedback.user_comment,
                json.dumps(feedback.context),
                feedback.timestamp.isoformat(),
                feedback.processed
            ))
    
    def _trigger_processing(self):
        """Trigger background feedback processing."""
        if not self.is_processing and self.auto_process:
            self.is_processing = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
    
    def _processing_loop(self):
        """Background processing loop."""
        try:
            result = self.processor.process_pending_feedback()
            
            if result['should_retrain']:
                feedback_logger.info("Feedback processing indicates model retraining should be triggered")
                # Here you could trigger actual retraining
                try:
                    job_id = self.ml_training_pipeline.force_training(reason="feedback_triggered_retraining")
                    feedback_logger.info(f"Successfully queued retraining job {job_id} due to feedback.")
                    self.processor.last_retrain_time = datetime.now() # Update last retrain time after successful trigger
                except Exception as e:
                    feedback_logger.error(f"Failed to trigger retraining via MLTrainingPipeline: {e}", exc_info=True)
                
        except Exception as e:
            feedback_logger.error(f"Error in feedback processing loop: {e}")
        finally:
            self.is_processing = False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        stats = self.processor.get_feedback_stats()
        
        return {
            'processor_status': {
                'processed_count': self.processor.processed_count,
                'pending_count': len(self.processor.pending_feedback),
                'last_retrain_time': self.processor.last_retrain_time.isoformat()
            },
            'feedback_statistics': {
                'total_feedback': stats.total_feedback_count,
                'by_type': stats.feedback_by_type,
                'by_category': stats.feedback_by_category,
                'avg_user_confidence': stats.avg_user_confidence
            },
            'improvement_suggestions': stats.improvement_suggestions,
            'system_config': {
                'auto_process': self.auto_process,
                'processing_interval': self.processing_interval,
                'auto_retrain_threshold': self.processor.auto_retrain_threshold
            }
        }

def create_feedback_system(config: Optional[Dict[str, Any]] = None) -> UserFeedbackSystem:
    """Factory function to create an instance of UserFeedbackSystem."""
    return UserFeedbackSystem(config=config) 