"""
Tests for Real-Time Monitor - Session 4 Priority 3 Implementation

Tests the enhanced real-time monitoring, anomaly detection,
and alert system functionality.
"""

import pytest
import tempfile
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.core.real_time_monitor import (
    RealTimeMonitor,
    AnomalyDetector, 
    PerformanceAggregator,
    AnomalyType,
    Anomaly,
    MetricStats
)
from app.core.ml_monitoring import MLPerformanceMonitor, MetricSnapshot, AlertThreshold


@pytest.mark.unit
class TestAnomalyDetector:
    """Test the AnomalyDetector class."""
    
    def setup_method(self):
        """Setup test environment."""
        config = {
            'sensitivity': 'medium',
            'window_size': 20
        }
        self.detector = AnomalyDetector(config)
    
    def test_initialization(self):
        """Test AnomalyDetector initialization."""
        assert self.detector is not None
        assert self.detector.sensitivity == 'medium'
        assert self.detector.window_size == 20
        assert self.detector.std_threshold == 2.5
    
    def test_sensitivity_thresholds(self):
        """Test different sensitivity settings."""
        low_detector = AnomalyDetector({'sensitivity': 'low'})
        assert low_detector.std_threshold == 3.0
        
        high_detector = AnomalyDetector({'sensitivity': 'high'})
        assert high_detector.std_threshold == 2.0
    
    def test_add_metric_value(self):
        """Test adding metric values."""
        self.detector.add_metric_value('test_metric', 100.0)
        
        assert 'test_metric' in self.detector.metric_history
        assert len(self.detector.metric_history['test_metric']) == 1
    
    def test_metric_stats_calculation(self):
        """Test metric statistics calculation."""
        # Add multiple values
        values = [100, 102, 98, 101, 99, 103, 97]
        for value in values:
            self.detector.add_metric_value('test_metric', value)
        
        stats = self.detector.get_metric_stats('test_metric')
        assert 'test_metric' in stats
        
        metric_stats = stats['test_metric']
        assert metric_stats.sample_count == len(values)
        assert 95 < metric_stats.mean < 105  # Should be around 100
        assert metric_stats.std > 0
    
    def test_anomaly_detection_no_data(self):
        """Test anomaly detection with insufficient data."""
        anomalies = self.detector.detect_anomalies('new_metric', 100.0)
        assert len(anomalies) == 0
    
    def test_anomaly_detection_spike(self):
        """Test spike anomaly detection."""
        # Add baseline values with some variation to create non-zero std
        baseline_values = [100, 101, 99, 100, 102, 98, 100, 101, 99, 100, 102, 98, 100, 101, 99]
        for value in baseline_values:
            self.detector.add_metric_value('test_metric', value)
        
        # Add a spike
        anomalies = self.detector.detect_anomalies('test_metric', 200.0)  # Clear spike
        
        assert len(anomalies) > 0
        # Could be either SPIKE or OUTLIER depending on detection method
        anomaly = anomalies[0]
        assert anomaly.anomaly_type in [AnomalyType.SPIKE, AnomalyType.OUTLIER]
        assert anomaly.current_value == 200.0
        assert anomaly.severity in ['medium', 'high', 'critical']
    
    def test_anomaly_detection_drop(self):
        """Test drop anomaly detection."""
        # Add baseline values with some variation to create non-zero std
        baseline_values = [100, 101, 99, 100, 102, 98, 100, 101, 99, 100, 102, 98, 100, 101, 99]
        for value in baseline_values:
            self.detector.add_metric_value('test_metric', value)
        
        # Add a drop
        anomalies = self.detector.detect_anomalies('test_metric', 20.0)  # Clear drop
        
        assert len(anomalies) > 0
        # Could be either DROP or OUTLIER depending on detection method
        anomaly = anomalies[0]
        assert anomaly.anomaly_type in [AnomalyType.DROP, AnomalyType.OUTLIER]
        assert anomaly.current_value == 20.0
    
    def test_no_false_positives_normal_data(self):
        """Test that normal variations don't trigger anomalies."""
        # Add normal variations around 100
        import random
        random.seed(42)  # For reproducible tests
        
        for _ in range(20):
            value = 100 + random.uniform(-5, 5)  # Small variations
            self.detector.add_metric_value('test_metric', value)
        
        # Test a normal value
        anomalies = self.detector.detect_anomalies('test_metric', 103.0)
        assert len(anomalies) == 0  # Should not trigger anomaly
    
    def test_get_recent_anomalies(self):
        """Test getting recent anomalies."""
        # Add baseline and trigger anomaly
        for _ in range(15):
            self.detector.add_metric_value('test_metric', 100.0)
        
        self.detector.detect_anomalies('test_metric', 200.0)  # Trigger anomaly
        
        recent = self.detector.get_recent_anomalies(hours_back=1)
        assert len(recent) > 0
        assert recent[0].metric_name == 'test_metric'
    
    def test_alert_threshold_integration(self):
        """Test AlertThreshold integration with AnomalyDetector."""
        # Create an alert threshold
        threshold = AlertThreshold(
            metric_name='test_metric',
            min_value=50.0,
            max_value=150.0,
            sample_size=1
        )
        
        # Add threshold to detector
        self.detector.add_alert_threshold(threshold)
        assert len(self.detector.alert_thresholds) == 1
        
        # Test threshold violation (above max)
        anomalies = self.detector.detect_anomalies('test_metric', 200.0)
        assert len(anomalies) > 0
        threshold_anomaly = next((a for a in anomalies if a.context.get('detection_method') == 'alert_threshold'), None)
        assert threshold_anomaly is not None
        assert threshold_anomaly.anomaly_type == AnomalyType.SPIKE
        assert 'above_maximum' in threshold_anomaly.description
        
        # Test threshold violation (below min)
        anomalies = self.detector.detect_anomalies('test_metric', 25.0)
        assert len(anomalies) > 0
        threshold_anomaly = next((a for a in anomalies if a.context.get('detection_method') == 'alert_threshold'), None)
        assert threshold_anomaly is not None
        assert threshold_anomaly.anomaly_type == AnomalyType.DROP
        assert 'below_minimum' in threshold_anomaly.description
        
        # Test normal value (no violation)
        anomalies = self.detector.detect_anomalies('test_metric', 100.0)
        threshold_anomalies = [a for a in anomalies if a.context.get('detection_method') == 'alert_threshold']
        assert len(threshold_anomalies) == 0  # Should not trigger threshold violation

    def test_remove_alert_threshold(self):
        """Test removing alert thresholds."""
        from app.core.ml_monitoring import AlertThreshold
        
        threshold1 = AlertThreshold(metric_name='metric1', max_value=100.0, sample_size=1)
        threshold2 = AlertThreshold(metric_name='metric2', max_value=200.0, sample_size=1)
        
        self.detector.add_alert_threshold(threshold1)
        self.detector.add_alert_threshold(threshold2)
        assert len(self.detector.alert_thresholds) == 2
        
        self.detector.remove_alert_threshold('metric1')
        assert len(self.detector.alert_thresholds) == 1
        assert self.detector.alert_thresholds[0].metric_name == 'metric2'
    
    def test_thread_safety(self):
        """Test thread safety of anomaly detector."""
        def add_values():
            for i in range(50):
                self.detector.add_metric_value('concurrent_metric', 100 + i % 10)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_values)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have processed all values without errors
        stats = self.detector.get_metric_stats('concurrent_metric')
        assert 'concurrent_metric' in stats
        # Note: deque maxlen=window_size*2=40, so only last 40 values are kept
        assert stats['concurrent_metric'].sample_count == 40
        assert stats['concurrent_metric'].sample_count > 0


@pytest.mark.unit
class TestPerformanceAggregator:
    """Test the PerformanceAggregator class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.aggregator = PerformanceAggregator(update_interval=1.0)
    
    def test_initialization(self):
        """Test PerformanceAggregator initialization."""
        assert self.aggregator is not None
        assert self.aggregator.update_interval == 1.0
        assert len(self.aggregator.metric_windows) == 3  # 1min, 5min, 1hour
    
    def test_add_metrics(self):
        """Test adding metrics."""
        metrics = {
            'cpu_percent': 50.0,
            'memory_mb': 1024.0,
            'processing_time_ms': 100.0
        }
        
        self.aggregator.add_metrics(metrics)
        
        # Check that metrics were added to all windows
        for window in self.aggregator.metric_windows.values():
            assert len(window) == 1
            assert window[0]['metrics'] == metrics
    
    def test_aggregated_metrics_calculation(self):
        """Test aggregated metrics calculation."""
        # Add multiple metric entries
        for i in range(5):
            metrics = {
                'cpu_percent': 50.0 + i,
                'memory_mb': 1000.0 + i * 100
            }
            self.aggregator.add_metrics(metrics, timestamp=datetime.now())
        
        # Get aggregated metrics
        aggregated = self.aggregator.get_aggregated_metrics('1min')
        
        assert 'cpu_percent' in aggregated
        assert 'memory_mb' in aggregated
        
        cpu_stats = aggregated['cpu_percent']
        assert 'mean' in cpu_stats
        assert 'min' in cpu_stats
        assert 'max' in cpu_stats
        assert cpu_stats['count'] == 5
    
    def test_all_windows(self):
        """Test getting all time windows."""
        metrics = {'test_metric': 100.0}
        self.aggregator.add_metrics(metrics)
        
        all_windows = self.aggregator.get_all_windows()
        
        assert '1min' in all_windows
        assert '5min' in all_windows
        assert '1hour' in all_windows


@pytest.mark.unit
class TestRealTimeMonitor:
    """Test the RealTimeMonitor class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        config = {
            'storage_path': f"{self.temp_dir}/test_monitor.db",
            'anomaly_detection': {
                'sensitivity': 'medium'
            }
        }
        self.monitor = RealTimeMonitor(config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test RealTimeMonitor initialization."""
        assert self.monitor is not None
        assert isinstance(self.monitor.anomaly_detector, AnomalyDetector)
        assert isinstance(self.monitor.performance_aggregator, PerformanceAggregator)
        assert not self.monitor.is_monitoring

    def test_set_ml_monitor(self):
        """Test setting ML monitor."""
        mock_ml_monitor = Mock()
        mock_ml_monitor.add_metrics_callback = Mock()
        
        self.monitor.set_ml_monitor(mock_ml_monitor)
        
        assert self.monitor.ml_monitor is mock_ml_monitor
        # Should register callback if add_metrics_callback method exists
        mock_ml_monitor.add_metrics_callback.assert_called_once()

    def test_ml_monitor_callback_functionality(self):
        """Test ML monitor integration with automatic callback."""
        # Mock MLPerformanceMonitor with metrics callback support
        mock_ml_monitor = Mock()
        mock_ml_monitor.add_metrics_callback = Mock()
        
        # Set ML monitor
        self.monitor.set_ml_monitor(mock_ml_monitor)
        
        # Get the registered callback function
        callback_function = mock_ml_monitor.add_metrics_callback.call_args[0][0]
        assert callable(callback_function)
        
        # Test the callback with mock metrics
        mock_metrics = Mock()
        mock_metrics.accuracy = 0.95
        mock_metrics.precision = 0.92
        mock_metrics.recall = 0.88
        mock_metrics.f1_score = 0.90
        mock_metrics.processing_time_ms = 150.0
        mock_metrics.confidence_correlation = 0.85
        mock_metrics.sample_count = 100
        
        # Mock the anomaly detector to avoid actual anomaly detection in test
        with patch.object(self.monitor.anomaly_detector, 'add_metric_value') as mock_add_metric, \
             patch.object(self.monitor.anomaly_detector, 'detect_anomalies', return_value=[]) as mock_detect, \
             patch.object(self.monitor.performance_monitor, 'get_system_metrics') as mock_system:
            
            # Call the callback
            callback_function(mock_metrics)
            
            # Verify metrics were processed
            assert mock_add_metric.call_count == 7  # 7 metrics from MetricSnapshot
            assert mock_detect.call_count == 7
            assert mock_system.call_count == 1

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        assert not self.monitor.is_monitoring
        
        self.monitor.start_monitoring(interval_seconds=1)
        assert self.monitor.is_monitoring
        assert self.monitor.monitoring_thread is not None
        
        self.monitor.stop_monitoring()
        assert not self.monitor.is_monitoring

    def test_collect_current_metrics(self):
        """Test collecting current metrics."""
        # Mock performance monitor
        with patch.object(self.monitor.performance_monitor, 'get_system_metrics') as mock_system:
            mock_system.return_value = {
                'process_memory_mb': 100.0,
                'process_cpu_percent': 50.0,
                'system_memory_percent': 75.0
            }
            
            metrics = self.monitor._collect_current_metrics()
            
            assert 'process_memory_mb' in metrics
            assert 'process_cpu_percent' in metrics
            assert 'system_memory_percent' in metrics
            assert metrics['process_memory_mb'] == 100.0

    def test_collect_metrics_with_ml_monitor(self):
        """Test metric collection with ML monitor."""
        mock_ml_monitor = Mock()
        mock_metrics = Mock()
        mock_metrics.accuracy = 0.95
        mock_metrics.precision = 0.92
        mock_metrics.recall = 0.88
        mock_metrics.processing_time_ms = 150.0
        mock_metrics.confidence_correlation = 0.85
        
        mock_ml_monitor.get_current_metrics.return_value = mock_metrics
        # Don't trigger callback registration for this test
        mock_ml_monitor.add_metrics_callback = Mock()
        self.monitor.set_ml_monitor(mock_ml_monitor)
        
        # Mock system metrics
        with patch.object(self.monitor.performance_monitor, 'get_system_metrics') as mock_system:
            mock_system.return_value = {
                'process_memory_mb': 100.0,
                'process_cpu_percent': 50.0,
                'system_memory_percent': 75.0
            }
            
            metrics = self.monitor._collect_current_metrics()
            
            # Should include both system and ML metrics
            assert 'ml_accuracy' in metrics
            assert 'ml_precision' in metrics
            assert 'ml_processing_time_ms' in metrics
            assert metrics['ml_accuracy'] == 0.95

    def test_anomaly_detection_integration(self):
        """Test anomaly detection integration."""
        # Mock some metrics
        test_metrics = {
            'test_metric': 100.0,
            'another_metric': 50.0
        }
        
        # Mock anomaly detector to return specific anomalies
        mock_anomaly = Mock()
        mock_anomaly.severity = 'high'
        
        with patch.object(self.monitor.anomaly_detector, 'detect_anomalies', return_value=[mock_anomaly]):
            anomalies = self.monitor._detect_anomalies(test_metrics)
            
            assert len(anomalies) == 2  # Should detect for each metric
            assert all(a is mock_anomaly for a in anomalies)

    def test_alert_callback_system(self):
        """Test alert callback system."""
        callback_called = []
        
        def test_callback(anomaly):
            callback_called.append(anomaly)
        
        self.monitor.add_alert_callback(test_callback)
        
        # Create mock anomaly
        mock_anomaly = Mock()
        mock_anomaly.severity = 'high'
        mock_anomaly.description = 'Test anomaly'
        mock_anomaly.anomaly_type = Mock()
        mock_anomaly.anomaly_type.value = 'spike'
        mock_anomaly.metric_name = 'test_metric'
        mock_anomaly.current_value = 100.0
        mock_anomaly.expected_value = 50.0
        
        # Process anomaly alert
        self.monitor._process_anomaly_alerts([mock_anomaly])
        
        # Verify callback was called
        assert len(callback_called) == 1
        assert callback_called[0] is mock_anomaly

    def test_monitoring_status(self):
        """Test monitoring status reporting."""
        status = self.monitor.get_monitoring_status()
        
        assert 'is_monitoring' in status
        assert 'last_metrics_collection' in status
        assert 'monitoring_interval' in status
        assert 'anomaly_detector_metrics' in status
        assert 'recent_anomalies_count' in status
        assert 'alert_callbacks_count' in status
        assert status['is_monitoring'] == False
        assert status['alert_callbacks_count'] == 0

    def test_dashboard_data(self):
        """Test dashboard data generation."""
        dashboard_data = self.monitor.get_dashboard_data()
        
        assert 'timestamp' in dashboard_data
        assert 'monitoring_status' in dashboard_data
        assert 'aggregated_metrics' in dashboard_data
        assert 'recent_anomalies' in dashboard_data
        assert 'metric_statistics' in dashboard_data
        
        # Should be valid structure
        assert isinstance(dashboard_data['aggregated_metrics'], dict)
        assert isinstance(dashboard_data['recent_anomalies'], list)

    def test_storage_integration(self):
        """Test storage integration."""
        # Add some metrics and anomalies
        test_metrics = {'test_metric': 100.0}
        
        # Store metrics
        self.monitor._store_real_time_metrics(test_metrics)
        
        # Create and store anomaly
        anomaly = Anomaly(
            anomaly_type=AnomalyType.SPIKE,
            metric_name='test_metric',
            current_value=200.0,
            expected_value=100.0,
            deviation_score=2.0,
            severity='high',
            description='Test anomaly'
        )
        self.monitor._store_anomaly(anomaly)
        
        # Verify data was stored
        import sqlite3
        with sqlite3.connect(self.monitor.storage_path) as conn:
            # Check metrics
            cursor = conn.execute("SELECT COUNT(*) FROM real_time_metrics")
            metrics_count = cursor.fetchone()[0]
            assert metrics_count > 0
            
            # Check anomalies
            cursor = conn.execute("SELECT COUNT(*) FROM anomalies")
            anomalies_count = cursor.fetchone()[0]
            assert anomalies_count > 0


@pytest.mark.integration
class TestRealTimeMonitorIntegration:
    """Integration tests for real-time monitor with existing components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        config = {'storage_path': f"{self.temp_dir}/test_monitor.db"}
        self.monitor = RealTimeMonitor(config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('app.core.real_time_monitor.get_config')
    def test_config_integration(self, mock_get_config):
        """Test integration with config manager."""
        mock_config = {
            'real_time_monitoring': {
                'storage_path': '/test/path',
                'anomaly_detection': {
                    'sensitivity': 'high'
                }
            }
        }
        mock_get_config.return_value = mock_config
        
        monitor = RealTimeMonitor()
        
        assert monitor.anomaly_detector.sensitivity == 'high'
    
    def test_ml_monitor_integration(self):
        """Test integration with existing ML monitoring."""
        from app.core.ml_monitoring import MLPerformanceMonitor
        
        # Create a mock ML monitor
        ml_monitor = Mock(spec=MLPerformanceMonitor)
        mock_metrics = Mock()
        mock_metrics.accuracy = 0.95
        mock_metrics.precision = 0.90
        mock_metrics.recall = 0.85
        mock_metrics.processing_time_ms = 120.0
        mock_metrics.confidence_correlation = 0.88
        
        ml_monitor.get_current_metrics.return_value = mock_metrics
        
        # Set the ML monitor
        self.monitor.set_ml_monitor(ml_monitor)
        
        # Test metrics collection
        metrics = self.monitor._collect_current_metrics()
        
        assert 'ml_accuracy' in metrics
        assert metrics['ml_accuracy'] == 0.95
    
    def test_performance_impact(self):
        """Test that real-time monitoring has minimal performance impact."""
        import time
        
        # Measure time for anomaly detection operations
        start_time = time.time()
        
        for i in range(100):
            metrics = {
                'cpu_percent': 50.0 + i % 10,
                'memory_mb': 1000.0 + i,
                'processing_time_ms': 100.0 + i % 20
            }
            
            self.monitor._detect_anomalies(metrics)
        
        operation_time = time.time() - start_time
        
        # Should be fast (less than 1 second for 100 operations)
        assert operation_time < 1.0
    
    def test_concurrent_monitoring(self):
        """Test concurrent monitoring operations."""
        def simulate_metrics():
            for i in range(10):
                metrics = {
                    'test_metric': 100.0 + i,
                    'cpu_percent': 50.0
                }
                self.monitor._detect_anomalies(metrics)
                time.sleep(0.01)
        
        # Run concurrent operations
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=simulate_metrics)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        status = self.monitor.get_monitoring_status()
        assert status['anomaly_detector_metrics'] > 0


if __name__ == '__main__':
    pytest.main([__file__]) 