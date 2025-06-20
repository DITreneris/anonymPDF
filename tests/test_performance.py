import pytest
import time
from pathlib import Path
from unittest.mock import patch

from app.core.performance import (
    PerformanceMonitor,
    FileProcessingMetrics,
    performance_monitor,
    get_performance_report,
    get_optimized_processor,
)

@pytest.fixture(autouse=True)
def clean_monitor():
    """Ensures the global monitor is clean for each test."""
    processor = get_optimized_processor()
    processor.monitor.clear_metrics()
    # Also reset the file metrics by creating a new instance
    processor.file_metrics = FileProcessingMetrics()
    yield

@pytest.mark.unit
class TestPerformanceMonitor:
    """Tests for the PerformanceMonitor class."""

    def test_track_operation(self):
        """Tests that an operation is tracked and its metrics are recorded."""
        monitor = PerformanceMonitor()
        tracking = monitor.track_operation("test_op")
        time.sleep(0.01)
        metrics = tracking["end_tracking"]()

        assert "duration_seconds" in metrics
        assert metrics["duration_seconds"] > 0
        
        stats = monitor.get_operation_stats("test_op")
        assert stats is not None
        assert stats["count"] == 1

    def test_get_operation_stats(self):
        """Tests the statistics calculation for tracked operations."""
        monitor = PerformanceMonitor()
        monitor.track_operation("stats_op")["end_tracking"]()
        monitor.track_operation("stats_op")["end_tracking"]()
        
        stats = monitor.get_operation_stats("stats_op")
        assert stats["count"] == 2
        assert "avg_duration_seconds" in stats

    def test_get_system_metrics(self):
        """Tests the retrieval of system-level metrics."""
        monitor = PerformanceMonitor()
        metrics = monitor.get_system_metrics()
        
        assert "process_memory_mb" in metrics
        assert "system_memory_percent" in metrics
        assert metrics["process_memory_mb"] > 0

@pytest.mark.unit
class TestFileProcessingMetrics:
    """Tests for the FileProcessingMetrics class."""

    def test_track_file_processing_legacy(self, tmp_path):
        """Tests the legacy method for tracking file processing."""
        metrics_tracker = FileProcessingMetrics()
        metrics_tracker.track_file_processing_legacy("test_proc", 1024, 0.1)
        stats = metrics_tracker.get_throughput_stats("test_proc")
        
        assert stats["count"] == 1
        assert stats["avg_throughput_mbps"] > 0

    def test_new_file_tracking_context(self, tmp_path):
        """Tests the new context-based file tracking."""
        metrics_tracker = FileProcessingMetrics()
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")
        
        tracking_context = metrics_tracker.track_file_processing(test_file, test_file.stat().st_size)
        time.sleep(0.01)
        metrics = tracking_context['end_tracking']()

        assert 'throughput' in metrics
        assert metrics['file_size_mb'] > 0
        
        stats = metrics_tracker.get_throughput_stats("pdf_processing")
        assert stats['count'] == 1

@pytest.mark.unit
class TestPerformanceDecorator:
    """Tests for the @performance_monitor decorator."""

    def test_decorator_tracks_operation(self):
        """Tests that the decorator correctly wraps a function and tracks it."""
        
        @performance_monitor("decorated_op")
        def my_func():
            pass

        my_func()
        
        stats = get_optimized_processor().monitor.get_operation_stats("decorated_op")
        assert stats["count"] == 1

    def test_decorator_uses_function_name_by_default(self):
        """Tests that the decorator uses the function name if no name is provided."""
        
        @performance_monitor()
        def another_func():
            pass
            
        another_func()
        
        stats = get_optimized_processor().monitor.get_operation_stats("another_func")
        assert stats["count"] == 1

@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for the performance module."""

    def test_get_performance_report(self):
        """Tests that the performance report can be generated."""
        # Run a decorated function to generate some stats
        @performance_monitor("report_test_op")
        def op_to_report():
            time.sleep(0.01)
        
        op_to_report()
        
        report = get_performance_report()
        
        # The report structure has changed to be simpler
        assert "report_test_op" in report
        assert report["report_test_op"]["count"] == 1
        assert "avg_duration_seconds" in report["report_test_op"] 