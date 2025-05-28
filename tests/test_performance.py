import pytest
import time
from pathlib import Path
from app.core.performance import (
    PerformanceMonitor,
    FileProcessingMetrics,
    performance_monitor,
    get_performance_report,
)


@pytest.mark.unit
class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        
        assert isinstance(monitor.metrics, dict)
        assert len(monitor.metrics) == 0

    def test_track_operation(self):
        """Test operation tracking."""
        monitor = PerformanceMonitor()
        
        # Start tracking
        tracking = monitor.track_operation("test_operation", test_param="value")
        
        # Simulate some work
        time.sleep(0.1)
        
        # End tracking
        metrics = tracking["end_tracking"]()
        
        assert metrics["operation"] == "test_operation"
        assert metrics["duration_seconds"] >= 0.1
        assert "memory_start_mb" in metrics
        assert "memory_end_mb" in metrics
        assert "memory_delta_mb" in metrics
        assert "test_param" in metrics
        assert metrics["test_param"] == "value"

    def test_get_operation_stats(self):
        """Test operation statistics calculation."""
        monitor = PerformanceMonitor()
        
        # Track multiple operations
        for i in range(3):
            tracking = monitor.track_operation("test_stats", iteration=i)
            time.sleep(0.05)  # Small delay
            tracking["end_tracking"]()
        
        stats = monitor.get_operation_stats("test_stats")
        
        assert stats is not None
        assert stats["operation"] == "test_stats"
        assert stats["count"] == 3
        assert stats["avg_duration_seconds"] > 0
        assert stats["min_duration_seconds"] > 0
        assert stats["max_duration_seconds"] > 0
        assert stats["total_duration_seconds"] > 0

    def test_get_operation_stats_nonexistent(self):
        """Test getting stats for non-existent operation."""
        monitor = PerformanceMonitor()
        stats = monitor.get_operation_stats("nonexistent")
        assert stats is None

    def test_clear_metrics(self):
        """Test clearing metrics."""
        monitor = PerformanceMonitor()
        
        # Add some metrics
        tracking = monitor.track_operation("test_clear")
        tracking["end_tracking"]()
        
        assert len(monitor.metrics) > 0
        
        # Clear metrics
        monitor.clear_metrics()
        
        assert len(monitor.metrics) == 0

    def test_get_system_metrics(self):
        """Test system metrics retrieval."""
        monitor = PerformanceMonitor()
        metrics = monitor.get_system_metrics()
        
        required_keys = [
            "process_memory_mb",
            "process_virtual_memory_mb",
            "process_cpu_percent",
            "system_memory_percent",
            "system_memory_available_mb",
            "timestamp",
        ]
        
        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))


@pytest.mark.unit
class TestFileProcessingMetrics:
    """Test file processing metrics functionality."""

    def test_file_processing_metrics_initialization(self):
        """Test file processing metrics initialization."""
        metrics = FileProcessingMetrics()
        assert isinstance(metrics.monitor, PerformanceMonitor)

    def test_track_file_processing(self, tmp_path):
        """Test file processing tracking."""
        metrics = FileProcessingMetrics()
        
        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_content = b"Test PDF content"
        test_file.write_bytes(test_content)
        
        # Track file processing
        tracking = metrics.track_file_processing(
            test_file, len(test_content), "test_processing"
        )
        
        # Simulate processing time
        time.sleep(0.1)
        
        # End tracking
        result = tracking["end_tracking"]()
        
        assert result["operation"] == "test_processing"
        assert result["file_size_mb"] == len(test_content) / 1024 / 1024
        assert result["file_extension"] == ".pdf"
        assert result["operation_type"] == "test_processing"
        assert "throughput_mb_per_sec" in result

    def test_get_throughput_stats(self, tmp_path):
        """Test throughput statistics calculation."""
        metrics = FileProcessingMetrics()
        
        # Create test files and track processing
        for i in range(2):
            test_file = tmp_path / f"test_{i}.pdf"
            test_content = b"Test PDF content" * (i + 1)  # Different sizes
            test_file.write_bytes(test_content)
            
            tracking = metrics.track_file_processing(
                test_file, len(test_content), "throughput_test"
            )
            time.sleep(0.05)
            tracking["end_tracking"]()
        
        stats = metrics.get_throughput_stats()
        
        assert "throughput_test" in stats
        operation_stats = stats["throughput_test"]
        assert "avg_throughput_mb_per_sec" in operation_stats
        assert "min_throughput_mb_per_sec" in operation_stats
        assert "max_throughput_mb_per_sec" in operation_stats


@pytest.mark.unit
class TestPerformanceDecorator:
    """Test performance monitoring decorator."""

    def test_performance_decorator(self):
        """Test performance monitoring decorator."""
        
        @performance_monitor("test_function")
        def test_function(x, y):
            time.sleep(0.05)
            return x + y
        
        result = test_function(1, 2)
        assert result == 3

    def test_performance_decorator_with_exception(self):
        """Test performance monitoring decorator with exception."""
        
        @performance_monitor("test_function_error")
        def test_function_error():
            time.sleep(0.05)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function_error()

    def test_performance_decorator_auto_name(self):
        """Test performance monitoring decorator with auto-generated name."""
        
        @performance_monitor()
        def test_auto_name():
            time.sleep(0.05)
            return "success"
        
        result = test_auto_name()
        assert result == "success"


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for performance monitoring."""

    def test_get_performance_report(self):
        """Test comprehensive performance report generation."""
        # Generate some metrics first
        monitor = PerformanceMonitor()
        tracking = monitor.track_operation("integration_test")
        time.sleep(0.05)
        tracking["end_tracking"]()
        
        report = get_performance_report()
        
        required_keys = [
            "system_metrics",
            "operation_stats",
            "file_processing_stats",
            "report_timestamp",
        ]
        
        for key in required_keys:
            assert key in report
        
        # Check system metrics structure
        system_metrics = report["system_metrics"]
        assert "process_memory_mb" in system_metrics
        assert "timestamp" in system_metrics
        
        # Check that timestamp is recent
        assert abs(report["report_timestamp"] - time.time()) < 1.0

    def test_performance_monitoring_persistence(self):
        """Test that performance metrics persist across operations."""
        from app.core.performance import global_performance_monitor
        
        # Clear any existing metrics
        global_performance_monitor.clear_metrics()
        
        # Add some operations
        for i in range(3):
            tracking = global_performance_monitor.track_operation(
                "persistence_test", iteration=i
            )
            time.sleep(0.02)
            tracking["end_tracking"]()
        
        # Get stats
        stats = global_performance_monitor.get_operation_stats("persistence_test")
        assert stats is not None
        assert stats["count"] == 3
        
        # Add more operations
        for i in range(2):
            tracking = global_performance_monitor.track_operation(
                "persistence_test", iteration=i + 3
            )
            time.sleep(0.02)
            tracking["end_tracking"]()
        
        # Check updated stats
        updated_stats = global_performance_monitor.get_operation_stats("persistence_test")
        assert updated_stats["count"] == 5
        assert updated_stats["total_duration_seconds"] > stats["total_duration_seconds"] 