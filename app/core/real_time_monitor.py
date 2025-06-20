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
import psutil
from logging import getLogger

# Import existing components
from app.core.ml_monitoring import MLPerformanceMonitor, MetricSnapshot, AlertThreshold
from app.core.performance import PerformanceMonitor
from app.core.analytics_engine import QualityAnalyzer
from app.core.config_manager import get_config

# Set up a specific logger for this module
system_logger = getLogger("anonympdf.monitor")

DATABASE_FILE = "data/monitoring.db"

class RealTimeMonitor:
    """
    A thread-safe singleton monitor for collecting and storing application performance metrics.

    This class provides a centralized way to track key performance indicators (KPIs)
    such as event durations, CPU usage, and memory consumption. It uses a dedicated
    SQLite database for persistence, making the data available for real-time
    and historical analysis.

    Attributes:
        db_path (str): The path to the SQLite database file.
        process (psutil.Process): The current process object for metric collection.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = DATABASE_FILE):
        """
        Initializes the RealTimeMonitor instance.

        As a singleton, this method ensures that initialization only occurs once.
        It sets up the database connection and the psutil process handle.

        Args:
            db_path (str): The path to the monitoring database. Defaults to DATABASE_FILE.
        """
        if hasattr(self, 'initialized') and self.initialized:
            return

        with self._lock:
            if hasattr(self, 'initialized') and self.initialized:
                return

            self.db_path = db_path
            self.db_write_lock = threading.Lock()
            # Use a short timeout to prevent blocking on a locked DB
            self.conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
            self._init_db()
            self.process = psutil.Process()
            self.initialized = True
            system_logger.info("RealTimeMonitor initialized with db at %s.", self.db_path)

    def _init_db(self):
        """
        Initializes the SQLite database and creates the performance_metrics table.

        This method sets up the necessary table schema if it does not already exist.
        It is designed to be safe to call on every startup.
        
        Raises:
            sqlite3.Error: If there is an issue with database initialization.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_name TEXT NOT NULL,
                duration REAL,
                cpu_percent REAL,
                memory_mb REAL,
                document_id TEXT,
                details TEXT
            )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            system_logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            system_logger.info("RealTimeMonitor database connection closed.")

    def log_event(self, event_name: str, duration: float = None, document_id: str = None, details: Dict[str, Any] = None):
        """
        Logs a specific performance event to the database.

        This is the primary method for logging performance data. It captures a snapshot
        of the current CPU and memory usage along with the provided event details.

        Args:
            event_name (str): The name of the event (e.g., 'text_extraction_finished').
            duration (float, optional): The duration of the event in seconds. Defaults to None.
            document_id (str, optional): The ID of the document being processed. Defaults to None.
            details (Dict[str, Any], optional): A dictionary for any additional context. Defaults to None.
        """
        timestamp = time.time()
        cpu_percent = self.process.cpu_percent()
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        details_str = json.dumps(details) if details else None

        try:
            with self.db_write_lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "INSERT INTO performance_metrics (timestamp, event_name, duration, cpu_percent, memory_mb, document_id, details) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, event_name, duration, cpu_percent, memory_mb, document_id, details_str)
                )
                self.conn.commit()
        except sqlite3.Error as e:
            # This should not crash the main application
            system_logger.error(f"Failed to log event '{event_name}': {e}", exc_info=True)

    def get_latest_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent performance metrics from the database.

        Args:
            limit (int): The maximum number of metric records to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                 represents a recorded performance event.
        """
        try:
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            system_logger.error(f"Failed to retrieve latest metrics: {e}", exc_info=True)
            return []

    def get_summary(self) -> Dict[str, Any]:
        """
        Provides a summarized view of the most recent performance data.

        This method calculates aggregate statistics (averages) for the last 500 events
        to provide a high-level overview of application health.

        Returns:
            Dict[str, Any]: A dictionary containing performance summary statistics.
        """
        metrics = self.get_latest_metrics(limit=500)
        if not metrics:
            return {"status": "No data available."}

        # Filter out None values before calculating averages
        valid_durations = [m['duration'] for m in metrics if m.get('duration') is not None]
        valid_cpus = [m['cpu_percent'] for m in metrics if m.get('cpu_percent') is not None]
        valid_mems = [m['memory_mb'] for m in metrics if m.get('memory_mb') is not None]

        avg_duration = sum(valid_durations) / len(valid_durations) if valid_durations else 0
        avg_cpu = sum(valid_cpus) / len(valid_cpus) if valid_cpus else 0
        avg_mem = sum(valid_mems) / len(valid_mems) if valid_mems else 0
        
        latest_event = metrics[0] if metrics else {}

        return {
            "total_events": len(metrics),
            "average_duration_ms": round(avg_duration * 1000, 2),
            "average_cpu_percent": round(avg_cpu, 2),
            "average_memory_mb": round(avg_mem, 2),
            "latest_event_timestamp": time.ctime(latest_event.get('timestamp')) if latest_event else None,
            "latest_cpu_percent": round(latest_event.get('cpu_percent', 0), 2),
            "latest_memory_mb": round(latest_event.get('memory_mb', 0), 2),
        }

    def shutdown(self):
        """
        Cleanly shuts down the monitor.
        This is primarily for testing environments to release file handles.
        """
        self.close()
        system_logger.info("RealTimeMonitor shutdown signal received.")
        # We can also reset the instance to allow for re-initialization in tests.
        RealTimeMonitor._instance = None

    @staticmethod
    def reset_for_testing():
        """
        A static method to forcefully reset the singleton instance.
        This is essential for ensuring test isolation.
        """
        with RealTimeMonitor._lock:
            if RealTimeMonitor._instance:
                # Attempt a clean shutdown first if the instance exists
                RealTimeMonitor._instance.shutdown()
                RealTimeMonitor._instance = None

# Global variable to hold the singleton instance, initialized to None
_monitor_lock = threading.Lock()

def get_real_time_monitor(db_path: Optional[str] = None) -> RealTimeMonitor:
    """
    Dependency injector that creates and returns the RealTimeMonitor singleton.
    This function ensures that the same instance of the monitor is used throughout
    the application, including in Celery workers.
    Args:
        db_path (Optional[str]): Path to the database. If None, uses default.
                                 This is mainly for tests.
    """
    if RealTimeMonitor._instance is None:
        with RealTimeMonitor._lock:
            if RealTimeMonitor._instance is None:
                # Creates the instance only if it doesn't exist
                # Pass db_path if provided, otherwise the default will be used.
                if db_path:
                    RealTimeMonitor(db_path=db_path)
                else:
                    RealTimeMonitor()
    # In a multi-threaded context, the instance might have been created
    # by another thread while this thread was waiting for the lock.
    # So we return the class instance directly.
    return RealTimeMonitor._instance

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
        
        system_logger.info(f"AnomalyDetector initialized with sensitivity: {self.sensitivity}, alert_thresholds: {len(self.alert_thresholds)}")
    
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
        system_logger.info(f"Added alert threshold for {threshold.metric_name}")

    def remove_alert_threshold(self, metric_name: str):
        """Remove alert thresholds for a specific metric."""
        original_count = len(self.alert_thresholds)
        self.alert_thresholds = [th for th in self.alert_thresholds if th.metric_name != metric_name]
        removed_count = original_count - len(self.alert_thresholds)
        if removed_count > 0:
            system_logger.info(f"Removed {removed_count} alert thresholds for {metric_name}")

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
        
        system_logger.info("PerformanceAggregator initialized")
    
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


