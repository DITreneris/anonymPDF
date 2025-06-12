"""
ML Performance Monitoring and A/B Testing for Priority 3 Implementation

This module provides comprehensive monitoring, A/B testing framework,
and real-time analytics for ML model performance.
"""

import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json
import sqlite3
from pathlib import Path

# Import components
from app.core.ml_integration import DetectionResult, PerformanceMetrics
from app.core.config_manager import get_config
from app.core.logging import get_logger

monitoring_logger = get_logger("ml_monitoring")


@dataclass
class AlertThreshold:
    """Configuration for monitoring alerts."""
    metric_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    change_threshold: Optional[float] = None  # Percentage change
    sample_size: int = 100  # Minimum samples before alerting
    
    def check_alert(self, current_value: float, previous_value: Optional[float] = None, 
                   sample_count: int = 0) -> Tuple[bool, str]:
        """Check if alert should be triggered."""
        if sample_count < self.sample_size:
            return False, "insufficient_samples"
        
        # Check absolute thresholds
        if self.min_value is not None and current_value < self.min_value:
            return True, f"{self.metric_name}_below_minimum_{self.min_value}"
        
        if self.max_value is not None and current_value > self.max_value:
            return True, f"{self.metric_name}_above_maximum_{self.max_value}"
        
        # Check change threshold
        if (self.change_threshold is not None and previous_value is not None and 
            previous_value > 0):
            change_percent = abs((current_value - previous_value) / previous_value) * 100
            if change_percent > self.change_threshold:
                return True, f"{self.metric_name}_changed_{change_percent:.1f}%"
        
        return False, "no_alert"


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time_ms: float
    confidence_correlation: float
    sample_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'processing_time_ms': self.processing_time_ms,
            'confidence_correlation': self.confidence_correlation,
            'sample_count': self.sample_count
        }


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    name: str
    description: str
    traffic_split: Dict[str, float]  # variant_name -> traffic_percentage
    start_date: datetime
    end_date: Optional[datetime] = None
    metrics_to_track: List[str] = field(default_factory=lambda: ['accuracy', 'processing_time_ms'])
    min_sample_size: int = 1000
    significance_threshold: float = 0.05
    
    def is_active(self) -> bool:
        """Check if A/B test is currently active."""
        now = datetime.now()
        return (now >= self.start_date and 
                (self.end_date is None or now <= self.end_date))


class MetricsCalculator:
    """Calculate various ML performance metrics."""
    
    def __init__(self):
        self.recent_results = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def add_result(self, result: DetectionResult, ground_truth: Optional[bool] = None):
        """Add detection result for metrics calculation."""
        with self._lock:
            result_data = {
                'timestamp': result.timestamp,
                'ml_confidence': result.ml_confidence,
                'processing_time_ms': result.processing_time_ms,
                'fallback_used': result.fallback_used,
                'ground_truth': ground_truth,
                'text': result.text,
                'category': result.category
            }
            self.recent_results.append(result_data)
    
    def calculate_current_metrics(self, window_minutes: int = 60) -> MetricSnapshot:
        """Calculate current performance metrics."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            # Filter recent results
            recent = [r for r in self.recent_results 
                     if r['timestamp'] >= cutoff_time]
        
        if not recent:
            return MetricSnapshot(
                timestamp=datetime.now(),
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                processing_time_ms=0.0, confidence_correlation=0.0,
                sample_count=0
            )
        
        # Calculate metrics
        total_samples = len(recent)
        
        # Processing time
        processing_times = [r['processing_time_ms'] for r in recent]
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
        
        # Accuracy metrics (only if we have ground truth)
        ground_truth_samples = [r for r in recent if r['ground_truth'] is not None]
        
        if ground_truth_samples:
            # Calculate accuracy, precision, recall
            true_positives = sum(1 for r in ground_truth_samples 
                               if r['ground_truth'] and r['ml_confidence'] > 0.5)
            false_positives = sum(1 for r in ground_truth_samples 
                                if not r['ground_truth'] and r['ml_confidence'] > 0.5)
            true_negatives = sum(1 for r in ground_truth_samples 
                               if not r['ground_truth'] and r['ml_confidence'] <= 0.5)
            false_negatives = sum(1 for r in ground_truth_samples 
                                if r['ground_truth'] and r['ml_confidence'] <= 0.5)
            
            total_gt_samples = len(ground_truth_samples)
            accuracy = (true_positives + true_negatives) / total_gt_samples if total_gt_samples > 0 else 0.0
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            # Use confidence as proxy metrics
            high_conf_ratio = sum(1 for r in recent if r['ml_confidence'] > 0.7) / total_samples
            accuracy = precision = recall = f1_score = high_conf_ratio
        
        # Confidence correlation (simplified)
        confidences = [r['ml_confidence'] for r in recent]
        confidence_correlation = 1.0 - statistics.stdev(confidences) if len(confidences) > 1 else 1.0
        
        return MetricSnapshot(
            timestamp=datetime.now(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            processing_time_ms=avg_processing_time,
            confidence_correlation=confidence_correlation,
            sample_count=total_samples
        )


class ABTestManager:
    """Manages A/B testing for ML models."""
    
    def __init__(self, storage_path: str = "data/ab_tests.db"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.active_tests = {}
        self.test_results = defaultdict(lambda: defaultdict(list))
        
        self._init_storage()
        self._load_active_tests()
    
    def _init_storage(self):
        """Initialize SQLite storage for A/B tests."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES ab_tests (test_id)
                )
            ''')
    
    def _load_active_tests(self):
        """Load active tests from storage."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("SELECT test_id, config_json FROM ab_tests")
            
            for test_id, config_json in cursor.fetchall():
                try:
                    config_data = json.loads(config_json)
                    config = ABTestConfig(
                        test_id=config_data['test_id'],
                        name=config_data['name'],
                        description=config_data['description'],
                        traffic_split=config_data['traffic_split'],
                        start_date=datetime.fromisoformat(config_data['start_date']),
                        end_date=datetime.fromisoformat(config_data['end_date']) if config_data.get('end_date') else None,
                        metrics_to_track=config_data.get('metrics_to_track', ['accuracy', 'processing_time_ms']),
                        min_sample_size=config_data.get('min_sample_size', 1000),
                        significance_threshold=config_data.get('significance_threshold', 0.05)
                    )
                    
                    if config.is_active():
                        self.active_tests[test_id] = config
                        
                except Exception as e:
                    monitoring_logger.error(f"Failed to load A/B test {test_id}: {e}")
    
    def create_test(self, config: ABTestConfig) -> bool:
        """Create new A/B test."""
        try:
            # Validate config
            total_traffic = sum(config.traffic_split.values())
            if abs(total_traffic - 1.0) > 0.001:  # Allow small floating point errors
                raise ValueError(f"Traffic split must sum to 1.0, got {total_traffic}")
            
            # Store in database
            config_json = json.dumps({
                'test_id': config.test_id,
                'name': config.name,
                'description': config.description,
                'traffic_split': config.traffic_split,
                'start_date': config.start_date.isoformat(),
                'end_date': config.end_date.isoformat() if config.end_date else None,
                'metrics_to_track': config.metrics_to_track,
                'min_sample_size': config.min_sample_size,
                'significance_threshold': config.significance_threshold
            })
            
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO ab_tests (test_id, config_json) VALUES (?, ?)",
                    (config.test_id, config_json)
                )
            
            # Add to active tests if currently active
            if config.is_active():
                self.active_tests[config.test_id] = config
            
            monitoring_logger.info(f"A/B test created: {config.test_id}")
            return True
            
        except Exception as e:
            monitoring_logger.error(f"Failed to create A/B test: {e}")
            return False
    
    def assign_variant(self, test_id: str, user_id: str = None) -> Optional[str]:
        """Assign user to A/B test variant."""
        if test_id not in self.active_tests:
            return None
        
        config = self.active_tests[test_id]
        
        # Simple hash-based assignment for consistency
        import hashlib
        hash_input = f"{test_id}_{user_id or 'anonymous'}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 1000000
        
        # Assign based on traffic split
        cumulative_split = 0.0
        threshold = hash_value / 1000000.0
        
        for variant, split in config.traffic_split.items():
            cumulative_split += split
            if threshold <= cumulative_split:
                return variant
        
        # Fallback to first variant
        return list(config.traffic_split.keys())[0]
    
    def record_result(self, test_id: str, variant: str, metrics: Dict[str, float]):
        """Record A/B test results."""
        if test_id not in self.active_tests:
            return
        
        config = self.active_tests[test_id]
        
        # Store in database
        with sqlite3.connect(self.storage_path) as conn:
            for metric_name, metric_value in metrics.items():
                if metric_name in config.metrics_to_track:
                    conn.execute(
                        "INSERT INTO ab_test_results (test_id, variant, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                        (test_id, variant, metric_name, metric_value)
                    )
        
        # Store in memory for quick access
        for metric_name, metric_value in metrics.items():
            if metric_name in config.metrics_to_track:
                self.test_results[test_id][f"{variant}_{metric_name}"].append(metric_value)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results with statistical analysis."""
        if test_id not in self.active_tests:
            return {'error': 'test_not_found'}
        
        config = self.active_tests[test_id]
        results = {}
        
        # Get results from database
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute(
                "SELECT variant, metric_name, metric_value FROM ab_test_results WHERE test_id = ?",
                (test_id,)
            )
            
            variant_metrics = defaultdict(lambda: defaultdict(list))
            for variant, metric_name, metric_value in cursor.fetchall():
                variant_metrics[variant][metric_name].append(metric_value)
        
        # Calculate statistics for each variant and metric
        for variant in config.traffic_split.keys():
            results[variant] = {}
            
            for metric_name in config.metrics_to_track:
                values = variant_metrics[variant][metric_name]
                
                if values:
                    results[variant][metric_name] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values)
                    }
                else:
                    results[variant][metric_name] = {
                        'count': 0, 'mean': 0.0, 'median': 0.0, 
                        'std': 0.0, 'min': 0.0, 'max': 0.0
                    }
        
        # Add test configuration
        results['config'] = {
            'test_id': config.test_id,
            'name': config.name,
            'traffic_split': config.traffic_split,
            'is_active': config.is_active(),
            'min_sample_size': config.min_sample_size
        }
        
        return results


class MLPerformanceMonitor:
    """Enhanced real-time monitoring with integrated quality analysis and A/B testing."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config().get('ml_monitoring', {})
        
        # Initialize components
        self.metrics_calculator = MetricsCalculator()
        self.ab_test_manager = ABTestManager(
            self.config.get('ab_test_storage_path', 'data/ab_tests.db')
        )
        
        # Alert system
        self.alert_thresholds = self._init_alert_thresholds()
        self.alert_callbacks = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'data/monitoring.db'))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_storage()
        
        # Metrics forwarding for real-time monitor integration
        self.metrics_forwarding_callbacks = []
        
        monitoring_logger.info("ML Performance Monitor initialized")
    
    def _init_alert_thresholds(self) -> List[AlertThreshold]:
        """Initialize alert thresholds from configuration."""
        default_thresholds = [
            AlertThreshold('accuracy', min_value=0.7, change_threshold=10.0),
            AlertThreshold('processing_time_ms', max_value=5000.0, change_threshold=50.0),
            AlertThreshold('confidence_correlation', min_value=0.5, change_threshold=20.0)
        ]
        
        # TODO: Load from config if available
        return default_thresholds
    
    def _init_storage(self):
        """Initialize storage for historical metrics."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metric_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    processing_time_ms REAL,
                    confidence_correlation REAL,
                    sample_count INTEGER
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON metric_snapshots(timestamp)
            ''')
    
    def start_monitoring(self, interval_seconds: int = 300):  # 5 minutes default
        """Start continuous monitoring."""
        if self.is_monitoring:
            monitoring_logger.warning("Monitor already running")
            return
        
        self.is_monitoring = True
        self.monitoring_interval = interval_seconds
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self.monitoring_thread.start()
        
        monitoring_logger.info(f"Performance monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)
        
        monitoring_logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop with metrics forwarding."""
        monitoring_logger.info("ML monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Calculate current metrics
                current_metrics = self.metrics_calculator.calculate_current_metrics()
                
                # Store metrics
                self._store_metrics(current_metrics)
                
                # Forward metrics to real-time monitor
                self._forward_metrics(current_metrics)
                
                # Check alerts
                self._check_alerts(current_metrics)
                
                # Sleep
                time.sleep(self.config.get('monitoring_interval', 300))  # 5 minutes default
                
            except Exception as e:
                monitoring_logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait before retrying
        
        monitoring_logger.info("ML monitoring loop stopped")
    
    def _store_metrics(self, metrics: MetricSnapshot):
        """Store metrics to database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                INSERT INTO metric_snapshots 
                (timestamp, accuracy, precision_score, recall, f1_score, 
                 processing_time_ms, confidence_correlation, sample_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.processing_time_ms,
                metrics.confidence_correlation,
                metrics.sample_count
            ))
    
    def _check_alerts(self, current_metrics: MetricSnapshot):
        """Check for alert conditions."""
        metrics_dict = {
            'accuracy': current_metrics.accuracy,
            'processing_time_ms': current_metrics.processing_time_ms,
            'confidence_correlation': current_metrics.confidence_correlation
        }
        
        previous_dict = {}
        if self.last_metrics:
            previous_dict = {
                'accuracy': self.last_metrics.accuracy,
                'processing_time_ms': self.last_metrics.processing_time_ms,
                'confidence_correlation': self.last_metrics.confidence_correlation
            }
        
        for threshold in self.alert_thresholds:
            current_value = metrics_dict.get(threshold.metric_name, 0.0)
            previous_value = previous_dict.get(threshold.metric_name)
            
            should_alert, reason = threshold.check_alert(
                current_value, previous_value, current_metrics.sample_count
            )
            
            if should_alert:
                self._trigger_alert(threshold.metric_name, reason, current_value, current_metrics)
    
    def _trigger_alert(self, metric_name: str, reason: str, value: float, metrics: MetricSnapshot):
        """Trigger alert and notify callbacks."""
        alert_data = {
            'metric_name': metric_name,
            'reason': reason,
            'value': value,
            'timestamp': metrics.timestamp.isoformat(),
            'sample_count': metrics.sample_count
        }
        
        monitoring_logger.warning(f"ALERT: {metric_name} - {reason} (value: {value})")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                monitoring_logger.error(f"Alert callback failed: {e}")
    
    def add_detection_result(self, result: DetectionResult, ground_truth: Optional[bool] = None):
        """Add detection result for monitoring."""
        self.metrics_calculator.add_result(result, ground_truth)
        
        # Record A/B test results if applicable
        if hasattr(result, 'ab_test_variant') and hasattr(result, 'ab_test_id'):
            test_metrics = {
                'accuracy': 1.0 if ground_truth == (result.ml_confidence > 0.5) else 0.0,
                'processing_time_ms': result.processing_time_ms,
                'confidence': result.ml_confidence
            }
            self.ab_test_manager.record_result(
                result.ab_test_id, result.ab_test_variant, test_metrics
            )
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> MetricSnapshot:
        """Get current performance metrics."""
        return self.metrics_calculator.calculate_current_metrics()
    
    def get_historical_metrics(self, hours_back: int = 24) -> List[MetricSnapshot]:
        """Get historical metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute('''
                SELECT timestamp, accuracy, precision_score, recall, f1_score,
                       processing_time_ms, confidence_correlation, sample_count
                FROM metric_snapshots 
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            ''', (cutoff_time.isoformat(),))
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append(MetricSnapshot(
                    timestamp=datetime.fromisoformat(row[0]),
                    accuracy=row[1],
                    precision=row[2],
                    recall=row[3],
                    f1_score=row[4],
                    processing_time_ms=row[5],
                    confidence_correlation=row[6],
                    sample_count=row[7]
                ))
            
            return metrics
    
    def create_ab_test(self, config: ABTestConfig) -> bool:
        """Create new A/B test."""
        return self.ab_test_manager.create_test(config)
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results."""
        return self.ab_test_manager.get_test_results(test_id)

    def add_metrics_callback(self, callback: Callable[[MetricSnapshot], None]):
        """Add callback for real-time metrics forwarding."""
        self.metrics_forwarding_callbacks.append(callback)
        monitoring_logger.info("Metrics forwarding callback added")

    def remove_metrics_callback(self, callback: Callable[[MetricSnapshot], None]):
        """Remove metrics forwarding callback."""
        if callback in self.metrics_forwarding_callbacks:
            self.metrics_forwarding_callbacks.remove(callback)
            monitoring_logger.info("Metrics forwarding callback removed")

    def _forward_metrics(self, metrics: MetricSnapshot):
        """Forward metrics to registered callbacks."""
        for callback in self.metrics_forwarding_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                monitoring_logger.error(f"Metrics forwarding callback failed: {e}")


# Factory function for easy integration
def create_ml_performance_monitor(config: Optional[Dict] = None) -> MLPerformanceMonitor:
    """Create and return ML performance monitor instance."""
    return MLPerformanceMonitor(config) 