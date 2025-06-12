"""
Real-Time Monitoring Enhancement for Priority 3 Implementation

This module provides advanced real-time monitoring, anomaly detection,
and alert system for the ML-powered PII detection system.
"""

import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json
import sqlite3
from pathlib import Path
from enum import Enum
import numpy as np

# Import existing components
from app.core.ml_monitoring import MLPerformanceMonitor, MetricSnapshot, AlertThreshold
from app.core.performance import PerformanceMonitor
from app.core.analytics_engine import QualityAnalyzer
from app.core.config_manager import get_config
from app.core.logging import get_logger

monitor_logger = get_logger("anonympdf.real_time_monitor")


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    OUTLIER = "outlier"
    PATTERN_BREAK = "pattern_break"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_type: AnomalyType
    metric_name: str
    current_value: float
    expected_value: float
    deviation_score: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistical information for a metric."""
    name: str
    mean: float
    std: float
    min_value: float
    max_value: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    sample_count: int
    last_updated: datetime = field(default_factory=datetime.now)


class AnomalyDetector:
    """Statistical anomaly detection for performance metrics."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Detection parameters
        self.sensitivity = self.config.get('sensitivity', 'medium')
        self.window_size = self.config.get('window_size', 50)
        self.std_threshold = self._get_std_threshold()
        
        # Alert thresholds integration
        self.alert_thresholds = self.config.get('alert_thresholds', [])
        # Convert dict configs to AlertThreshold objects if needed
        if self.alert_thresholds and isinstance(self.alert_thresholds[0], dict):
            self.alert_thresholds = [
                AlertThreshold(
                    metric_name=th.get('metric_name', ''),
                    min_value=th.get('min_value'),
                    max_value=th.get('max_value'),
                    change_threshold=th.get('change_threshold'),
                    sample_size=th.get('sample_size', 10)
                )
                for th in self.alert_thresholds
            ]
        
        # Metric history for statistical analysis
        self.metric_history = defaultdict(lambda: deque(maxlen=self.window_size * 2))
        self.metric_stats = {}
        
        # Previous values for threshold change detection
        self.previous_values = {}
        
        # Anomaly tracking
        self.detected_anomalies = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        monitor_logger.info(f"AnomalyDetector initialized with sensitivity: {self.sensitivity}, alert_thresholds: {len(self.alert_thresholds)}")
    
    def _get_std_threshold(self) -> float:
        """Get standard deviation threshold based on sensitivity."""
        thresholds = {
            'low': 3.0,      # Very conservative
            'medium': 2.5,   # Balanced
            'high': 2.0,     # More sensitive
            'very_high': 1.5 # Very sensitive
        }
        return thresholds.get(self.sensitivity, 2.5)
    
    def add_metric_value(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add a metric value for anomaly detection."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.metric_history[metric_name].append((timestamp, value))
            self._update_metric_stats(metric_name)
    
    def _update_metric_stats(self, metric_name: str):
        """Update statistical information for a metric."""
        history = self.metric_history[metric_name]
        
        if len(history) < 3:  # Need at least 3 points for stats
            return
        
        values = [point[1] for point in history]
        
        # Calculate basic statistics
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        min_val = min(values)
        max_val = max(values)
        
        # Determine trend (simple linear trend)
        if len(values) >= 10:
            recent_half = values[-len(values)//2:]
            older_half = values[:len(values)//2]
            
            recent_mean = statistics.mean(recent_half)
            older_mean = statistics.mean(older_half)
            
            change_threshold = std_val * 0.5  # Half std as threshold
            
            if recent_mean > older_mean + change_threshold:
                trend = 'increasing'
            elif recent_mean < older_mean - change_threshold:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        self.metric_stats[metric_name] = MetricStats(
            name=metric_name,
            mean=mean_val,
            std=std_val,
            min_value=min_val,
            max_value=max_val,
            trend=trend,
            sample_count=len(values)
        )
    
    def detect_anomalies(self, metric_name: str, current_value: float) -> List[Anomaly]:
        """Detect anomalies for a given metric value."""
        anomalies = []
        
        with self._lock:
            if metric_name not in self.metric_stats:
                # Check alert thresholds even without statistical data
                threshold_anomalies = self._check_alert_thresholds(metric_name, current_value)
                anomalies.extend(threshold_anomalies)
                return anomalies
            
            stats = self.metric_stats[metric_name]
            
            if stats.sample_count < 10:  # Need enough data for reliable statistical detection
                # Check alert thresholds even with limited statistical data
                threshold_anomalies = self._check_alert_thresholds(metric_name, current_value)
                anomalies.extend(threshold_anomalies)
                return anomalies
        
        # Z-score based anomaly detection
        if stats.std > 0:
            z_score = abs(current_value - stats.mean) / stats.std
            
            if z_score > self.std_threshold:
                severity = self._calculate_severity(z_score)
                anomaly_type = AnomalyType.SPIKE if current_value > stats.mean else AnomalyType.DROP
                
                anomaly = Anomaly(
                    anomaly_type=anomaly_type,
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_value=stats.mean,
                    deviation_score=z_score,
                    severity=severity,
                    description=f"{metric_name} {anomaly_type.value}: {current_value:.2f} (expected: {stats.mean:.2f}, z-score: {z_score:.2f})",
                    context={'std': stats.std, 'trend': stats.trend, 'detection_method': 'statistical'}
                )
                anomalies.append(anomaly)
        
        # Range-based anomaly detection
        expected_range = (stats.min_value, stats.max_value)
        if current_value < expected_range[0] * 0.5 or current_value > expected_range[1] * 1.5:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.OUTLIER,
                metric_name=metric_name,
                current_value=current_value,
                expected_value=stats.mean,
                deviation_score=abs(current_value - stats.mean) / (stats.std + 1e-6),
                severity='high',
                description=f"{metric_name} outlier: {current_value:.2f} outside expected range {expected_range}",
                context={'range': expected_range, 'trend': stats.trend, 'detection_method': 'range_based'}
            )
            anomalies.append(anomaly)
        
        # Alert threshold-based detection
        threshold_anomalies = self._check_alert_thresholds(metric_name, current_value)
        anomalies.extend(threshold_anomalies)
        
        # Store detected anomalies
        for anomaly in anomalies:
            self.detected_anomalies.append(anomaly)
        
        return anomalies
    
    def _check_alert_thresholds(self, metric_name: str, current_value: float) -> List[Anomaly]:
        """Check alert thresholds for anomalies."""
        anomalies = []
        
        for threshold in self.alert_thresholds:
            if threshold.metric_name != metric_name:
                continue
            
            previous_value = self.previous_values.get(metric_name)
            
            # Get current sample count for this metric
            sample_count = len(self.metric_history.get(metric_name, []))
            
            should_alert, reason = threshold.check_alert(
                current_value=current_value,
                previous_value=previous_value,
                sample_count=sample_count
            )
            
            if should_alert and reason != "insufficient_samples":
                # Determine anomaly type from reason
                if "below_minimum" in reason:
                    anomaly_type = AnomalyType.DROP
                elif "above_maximum" in reason:
                    anomaly_type = AnomalyType.SPIKE
                elif "changed" in reason:
                    anomaly_type = AnomalyType.TREND_CHANGE
                else:
                    anomaly_type = AnomalyType.PATTERN_BREAK
                
                # Calculate severity based on threshold violation severity
                if threshold.min_value is not None and current_value < threshold.min_value:
                    deviation = abs(current_value - threshold.min_value) / (threshold.min_value + 1e-6)
                elif threshold.max_value is not None and current_value > threshold.max_value:
                    deviation = abs(current_value - threshold.max_value) / (threshold.max_value + 1e-6)
                else:
                    deviation = 1.0
                
                # Map deviation to severity
                if deviation > 0.5:
                    severity = 'critical'
                elif deviation > 0.3:
                    severity = 'high'
                elif deviation > 0.1:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                anomaly = Anomaly(
                    anomaly_type=anomaly_type,
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_value=threshold.min_value or threshold.max_value or previous_value or current_value,
                    deviation_score=deviation,
                    severity=severity,
                    description=f"{metric_name} threshold violation: {reason}",
                    context={
                        'threshold_config': {
                            'min_value': threshold.min_value,
                            'max_value': threshold.max_value,
                            'change_threshold': threshold.change_threshold
                        },
                        'detection_method': 'alert_threshold'
                    }
                )
                anomalies.append(anomaly)
        
        # Update previous value for change detection
        self.previous_values[metric_name] = current_value
        
        return anomalies
    
    def add_alert_threshold(self, threshold: AlertThreshold):
        """Add an alert threshold for monitoring."""
        self.alert_thresholds.append(threshold)
        monitor_logger.info(f"Added alert threshold for {threshold.metric_name}")

    def remove_alert_threshold(self, metric_name: str):
        """Remove alert thresholds for a specific metric."""
        original_count = len(self.alert_thresholds)
        self.alert_thresholds = [th for th in self.alert_thresholds if th.metric_name != metric_name]
        removed_count = original_count - len(self.alert_thresholds)
        if removed_count > 0:
            monitor_logger.info(f"Removed {removed_count} alert thresholds for {metric_name}")

    def _calculate_severity(self, z_score: float) -> str:
        """Calculate severity based on z-score."""
        if z_score > 4.0:
            return 'critical'
        elif z_score > 3.0:
            return 'high'
        elif z_score > 2.5:
            return 'medium'
        else:
            return 'low'
    
    def get_metric_stats(self, metric_name: Optional[str] = None) -> Dict[str, MetricStats]:
        """Get statistical information for metrics."""
        with self._lock:
            if metric_name:
                return {metric_name: self.metric_stats.get(metric_name)}
            return dict(self.metric_stats)
    
    def get_recent_anomalies(self, hours_back: int = 24) -> List[Anomaly]:
        """Get recent anomalies within specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            return [anomaly for anomaly in self.detected_anomalies 
                   if anomaly.timestamp >= cutoff_time]


class PerformanceAggregator:
    """Aggregate performance metrics from various sources."""
    
    def __init__(self, update_interval: float = 30.0):
        self.update_interval = update_interval
        self.aggregated_metrics = {}
        self._lock = threading.Lock()
        
        # Rolling windows for different time periods
        self.metric_windows = {
            '1min': deque(maxlen=60),    # 60 seconds
            '5min': deque(maxlen=300),   # 5 minutes  
            '1hour': deque(maxlen=3600)  # 1 hour
        }
        
        monitor_logger.info("PerformanceAggregator initialized")
    
    def add_metrics(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """Add metrics for aggregation."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            metric_entry = {
                'timestamp': timestamp,
                'metrics': metrics.copy()
            }
            
            # Add to all time windows
            for window in self.metric_windows.values():
                window.append(metric_entry)
            
            # Update aggregated metrics
            self._update_aggregated_metrics()
    
    def _update_aggregated_metrics(self):
        """Update aggregated metrics for different time windows."""
        now = datetime.now()
        
        for window_name, window_data in self.metric_windows.items():
            if not window_data:
                continue
            
            # Filter data based on window duration
            window_duration = self._get_window_duration(window_name)
            cutoff_time = now - window_duration
            
            recent_data = [
                entry for entry in window_data 
                if entry['timestamp'] >= cutoff_time
            ]
            
            if not recent_data:
                continue
            
            # Aggregate metrics
            aggregated = defaultdict(list)
            for entry in recent_data:
                for metric_name, value in entry['metrics'].items():
                    aggregated[metric_name].append(value)
            
            # Calculate statistics
            window_stats = {}
            for metric_name, values in aggregated.items():
                if values:
                    window_stats[metric_name] = {
                        'mean': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'count': len(values),
                        'last_value': values[-1]
                    }
            
            self.aggregated_metrics[window_name] = window_stats
    
    def _get_window_duration(self, window_name: str) -> timedelta:
        """Get duration for time window."""
        durations = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '1hour': timedelta(hours=1)
        }
        return durations.get(window_name, timedelta(minutes=5))
    
    def get_aggregated_metrics(self, window: str = '5min') -> Dict[str, Any]:
        """Get aggregated metrics for specified time window."""
        with self._lock:
            return self.aggregated_metrics.get(window, {})
    
    def get_all_windows(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated metrics for all time windows."""
        with self._lock:
            return dict(self.aggregated_metrics)


class RealTimeMonitor:
    """Enhanced real-time monitoring with anomaly detection and alerting."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config().get('real_time_monitoring', {})
        
        # Initialize components
        self.anomaly_detector = AnomalyDetector(self.config.get('anomaly_detection', {}))
        self.performance_aggregator = PerformanceAggregator(
            self.config.get('aggregation_interval', 30.0)
        )
        
        # Integrate with existing monitoring
        self.ml_monitor = None  # Will be set externally
        self.performance_monitor = PerformanceMonitor()
        self.quality_analyzer = None  # Will be set externally
        
        # Alert system
        self.alert_callbacks = []
        self.alert_history = deque(maxlen=1000)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.last_metrics_collection = datetime.now()
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'data/real_time_monitor.db'))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_storage()
        
        monitor_logger.info("RealTimeMonitor initialized")
    
    def _init_storage(self):
        """Initialize storage for monitoring data."""
        with sqlite3.connect(self.storage_path) as conn:
            # Anomalies table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL,
                    expected_value REAL,
                    deviation_score REAL,
                    severity TEXT,
                    description TEXT,
                    context TEXT
                )
            ''')
            
            # Real-time metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS real_time_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL,
                    window_type TEXT,
                    aggregation_type TEXT
                )
            ''')
            
            # Create indices
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomalies(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON real_time_metrics(timestamp)')
    
    def set_ml_monitor(self, ml_monitor: MLPerformanceMonitor):
        """Set the ML performance monitor for integration."""
        self.ml_monitor = ml_monitor
        
        # Register as metrics callback for automatic forwarding
        if hasattr(ml_monitor, 'add_metrics_callback'):
            ml_monitor.add_metrics_callback(self._on_ml_metrics_received)
            monitor_logger.info("Registered as ML metrics callback for automatic forwarding")

    def _on_ml_metrics_received(self, ml_metrics: 'MetricSnapshot'):
        """Callback for receiving ML metrics from MLPerformanceMonitor."""
        try:
            # Convert MetricSnapshot to metrics dict
            metrics_dict = {
                'ml_accuracy': ml_metrics.accuracy,
                'ml_precision': ml_metrics.precision,
                'ml_recall': ml_metrics.recall,
                'ml_f1_score': ml_metrics.f1_score,
                'ml_processing_time_ms': ml_metrics.processing_time_ms,
                'ml_confidence_correlation': ml_metrics.confidence_correlation,
                'ml_sample_count': ml_metrics.sample_count
            }
            
            # Add to performance aggregator
            self.performance_aggregator.add_metrics(metrics_dict)
            
            # Detect anomalies for ML metrics
            anomalies = []
            for metric_name, value in metrics_dict.items():
                # Add to anomaly detector
                self.anomaly_detector.add_metric_value(metric_name, value)
                
                # Detect anomalies
                metric_anomalies = self.anomaly_detector.detect_anomalies(metric_name, value)
                anomalies.extend(metric_anomalies)
            
            # Process any anomalies
            if anomalies:
                self._process_anomaly_alerts(anomalies)
                
            monitor_logger.debug(f"Processed ML metrics via callback: {len(metrics_dict)} metrics, {len(anomalies)} anomalies")
                
        except Exception as e:
            monitor_logger.error(f"Error processing ML metrics callback: {e}")

    def set_quality_analyzer(self, quality_analyzer: QualityAnalyzer):
        """Set the quality analyzer for integration."""
        self.quality_analyzer = quality_analyzer
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start real-time monitoring."""
        if self.is_monitoring:
            monitor_logger.warning("Real-time monitor already running")
            return
        
        self.is_monitoring = True
        self.monitoring_interval = interval_seconds
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        monitor_logger.info(f"Real-time monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)
        
        monitor_logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()
                
                # Add to aggregator
                self.performance_aggregator.add_metrics(current_metrics)
                
                # Detect anomalies
                anomalies = self._detect_anomalies(current_metrics)
                
                # Process alerts
                if anomalies:
                    self._process_anomaly_alerts(anomalies)
                
                # Store metrics
                self._store_real_time_metrics(current_metrics)
                
                # Update last collection time
                self.last_metrics_collection = datetime.now()
                
                # Sleep until next iteration
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                monitor_logger.error(f"Real-time monitoring error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current metrics from all sources."""
        metrics = {}
        
        # System metrics from performance monitor
        system_metrics = self.performance_monitor.get_system_metrics()
        metrics.update({
            'process_memory_mb': system_metrics.get('process_memory_mb', 0),
            'process_cpu_percent': system_metrics.get('process_cpu_percent', 0),
            'system_memory_percent': system_metrics.get('system_memory_percent', 0)
        })
        
        # ML performance metrics
        if self.ml_monitor:
            ml_metrics = self.ml_monitor.get_current_metrics()
            metrics.update({
                'ml_accuracy': ml_metrics.accuracy,
                'ml_precision': ml_metrics.precision,
                'ml_recall': ml_metrics.recall,
                'ml_processing_time_ms': ml_metrics.processing_time_ms,
                'ml_confidence_correlation': ml_metrics.confidence_correlation
            })
        
        # Quality analyzer metrics
        if self.quality_analyzer:
            quality_metrics = self.quality_analyzer.analyze_detection_quality(time_window_hours=1)
            if quality_metrics:
                avg_confidence = statistics.mean([m.avg_confidence for m in quality_metrics])
                avg_processing_time = statistics.mean([m.processing_time_ms for m in quality_metrics])
                
                metrics.update({
                    'quality_avg_confidence': avg_confidence,
                    'quality_avg_processing_time_ms': avg_processing_time,
                    'quality_categories_count': len(quality_metrics)
                })
        
        return metrics
    
    def _detect_anomalies(self, current_metrics: Dict[str, float]) -> List[Anomaly]:
        """Detect anomalies in current metrics."""
        all_anomalies = []
        
        for metric_name, value in current_metrics.items():
            # Add to anomaly detector
            self.anomaly_detector.add_metric_value(metric_name, value)
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(metric_name, value)
            all_anomalies.extend(anomalies)
        
        return all_anomalies
    
    def _process_anomaly_alerts(self, anomalies: List[Anomaly]):
        """Process anomaly alerts and trigger callbacks."""
        for anomaly in anomalies:
            # Store anomaly
            self._store_anomaly(anomaly)
            
            # Add to history
            self.alert_history.append(anomaly)
            
            # Log alert
            monitor_logger.warning(
                f"ANOMALY DETECTED: {anomaly.description}",
                anomaly_type=anomaly.anomaly_type.value,
                severity=anomaly.severity,
                metric=anomaly.metric_name,
                current_value=anomaly.current_value,
                expected_value=anomaly.expected_value
            )
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    monitor_logger.error(f"Alert callback failed: {e}")
    
    def _store_anomaly(self, anomaly: Anomaly):
        """Store anomaly to database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                INSERT INTO anomalies 
                (timestamp, anomaly_type, metric_name, current_value, expected_value,
                 deviation_score, severity, description, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                anomaly.timestamp.isoformat(),
                anomaly.anomaly_type.value,
                anomaly.metric_name,
                anomaly.current_value,
                anomaly.expected_value,
                anomaly.deviation_score,
                anomaly.severity,
                anomaly.description,
                json.dumps(anomaly.context)
            ))
    
    def _store_real_time_metrics(self, metrics: Dict[str, float]):
        """Store real-time metrics to database."""
        timestamp = datetime.now()
        
        with sqlite3.connect(self.storage_path) as conn:
            for metric_name, value in metrics.items():
                conn.execute('''
                    INSERT INTO real_time_metrics 
                    (timestamp, metric_name, value, window_type, aggregation_type)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    timestamp.isoformat(),
                    metric_name,
                    value,
                    'real_time',
                    'current'
                ))
    
    def add_alert_callback(self, callback: Callable[[Anomaly], None]):
        """Add callback for anomaly alerts."""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'is_monitoring': self.is_monitoring,
            'last_metrics_collection': self.last_metrics_collection.isoformat(),
            'monitoring_interval': getattr(self, 'monitoring_interval', 0),
            'anomaly_detector_metrics': len(self.anomaly_detector.metric_stats),
            'recent_anomalies_count': len(self.anomaly_detector.get_recent_anomalies(hours_back=1)),
            'alert_callbacks_count': len(self.alert_callbacks)
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time dashboard."""
        # Current aggregated metrics
        current_metrics = self.performance_aggregator.get_all_windows()
        
        # Recent anomalies
        recent_anomalies = self.anomaly_detector.get_recent_anomalies(hours_back=24)
        
        # Metric statistics
        metric_stats = self.anomaly_detector.get_metric_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': self.get_monitoring_status(),
            'aggregated_metrics': current_metrics,
            'recent_anomalies': [
                {
                    'type': a.anomaly_type.value,
                    'metric': a.metric_name,
                    'severity': a.severity,
                    'description': a.description,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in recent_anomalies[-10:]  # Last 10 anomalies
            ],
            'metric_statistics': {
                name: {
                    'mean': stats.mean,
                    'std': stats.std,
                    'trend': stats.trend,
                    'sample_count': stats.sample_count
                }
                for name, stats in metric_stats.items()
            }
        } 