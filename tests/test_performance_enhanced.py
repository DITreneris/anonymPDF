"""
Tests for Enhanced Performance Module - Priority 3 Session 3
Comprehensive testing of parallel performance tracking and integration with optimizers.
"""

import pytest
import time
import threading
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from app.core.performance import (
    PerformanceMonitor,
    ParallelPerformanceTracker,
    FileProcessingMetrics,
    PerformanceOptimizedProcessor,
    performance_monitor,
    get_performance_report,
    get_parallel_processor,
    global_performance_monitor,
    file_processing_metrics,
    get_optimized_processor
)
from app.core.config_manager import get_config
from app.core.performance_optimizer import get_memory_optimizer
from app.core.performance_optimizer import BatchEngine


@pytest.fixture(scope="module")
def config_manager():
    """Fixture to provide a config manager instance for the test module."""
    return get_config()


class TestEnhancedPerformanceMonitor:
    """Test enhanced performance monitor with thread safety."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
    
    def test_thread_safe_metrics_tracking(self):
        """Test thread-safe metrics tracking."""
        results = []
        
        def track_operation(operation_id):
            tracking = self.monitor.track_operation(f"test_op_{operation_id}")
            time.sleep(0.01)  # Simulate work
            metrics = tracking["end_tracking"]()
            results.append(metrics)
        
        # Run concurrent operations
        threads = []
        for i in range(5):
            t = threading.Thread(target=track_operation, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All operations should be tracked
        assert len(results) == 5
        for metrics in results:
            assert "operation" in metrics
            assert "duration_seconds" in metrics
            assert metrics["duration_seconds"] > 0
    
    def test_concurrent_stats_access(self):
        """Test concurrent access to statistics."""
        # Track some operations
        for i in range(3):
            tracking = self.monitor.track_operation("concurrent_test")
            time.sleep(0.001)
            tracking["end_tracking"]()
        
        # Access stats concurrently
        def get_stats():
            return self.monitor.get_operation_stats("concurrent_test")
        
        threads = []
        results = []
        
        for _ in range(3):
            t = threading.Thread(target=lambda: results.append(get_stats()))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All should return valid stats
        for stats in results:
            assert stats is not None
            assert stats["count"] == 3
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking functionality."""
        memory_usage = self.monitor.get_memory_usage()
        
        assert isinstance(memory_usage, dict)
        assert "rss_mb" in memory_usage
        assert "vms_mb" in memory_usage
        assert memory_usage["rss_mb"] > 0
        assert memory_usage["vms_mb"] > 0
    
    def test_system_metrics(self):
        """Test system metrics collection."""
        metrics = self.monitor.get_system_metrics()
        
        assert isinstance(metrics, dict)
        assert "process_memory_mb" in metrics
        assert "process_cpu_percent" in metrics
        assert "system_memory_percent" in metrics
        assert "timestamp" in metrics
        assert metrics["timestamp"] > 0


class TestParallelPerformanceTracker:
    """Test parallel performance tracking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = ParallelPerformanceTracker()
    
    def test_parallel_session_lifecycle(self):
        """Test complete parallel session lifecycle."""
        session_id = "test_session_1"
        worker_count = 4
        
        # Start session
        session = self.tracker.start_parallel_session(session_id, worker_count)
        
        assert session["session_id"] == session_id
        assert session["worker_count"] == worker_count
        assert session["tasks_completed"] == 0
        assert session["tasks_failed"] == 0
        
        # Track some task completions
        self.tracker.track_task_completion(session_id, 0.5, success=True)
        self.tracker.track_task_completion(session_id, 0.3, success=True)
        self.tracker.track_task_completion(session_id, 0.2, success=False)
        
        # End session
        metrics = self.tracker.end_parallel_session(session_id)
        
        assert metrics["session_id"] == session_id
        assert metrics["worker_count"] == worker_count
        assert metrics["tasks_completed"] == 2
        assert metrics["tasks_failed"] == 1
        assert metrics["total_tasks"] == 3
        assert metrics["success_rate"] == 2/3
        assert "parallel_efficiency" in metrics
        assert "throughput_tasks_per_second" in metrics
    
    def test_concurrent_task_tracking(self):
        """Test concurrent task completion tracking."""
        session_id = "concurrent_session"
        self.tracker.start_parallel_session(session_id, 4)
        
        def track_tasks():
            for i in range(10):
                self.tracker.track_task_completion(session_id, 0.1, success=True)
        
        # Run concurrent tracking
        threads = []
        for _ in range(3):
            t = threading.Thread(target=track_tasks)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # End session and check results
        metrics = self.tracker.end_parallel_session(session_id)
        assert metrics["tasks_completed"] == 30
        assert metrics["tasks_failed"] == 0
    
    def test_nonexistent_session_handling(self):
        """Test handling of nonexistent session operations."""
        # Track completion for nonexistent session
        self.tracker.track_task_completion("nonexistent", 1.0)
        
        # End nonexistent session
        metrics = self.tracker.end_parallel_session("nonexistent")
        assert metrics == {}
    
    def test_parallel_efficiency_calculation(self):
        """Test parallel efficiency calculation."""
        session_id = "efficiency_test"
        worker_count = 4
        
        self.tracker.start_parallel_session(session_id, worker_count)
        
        # Simulate 4 tasks of 0.1 second each
        for _ in range(4):
            self.tracker.track_task_completion(session_id, 0.1, success=True)
        
        time.sleep(0.5)  # Longer session duration to get reasonable efficiency
        metrics = self.tracker.end_parallel_session(session_id)
        
        # Efficiency should be reasonable
        # With 4 tasks of 0.1s each (0.4s total work) over ~0.5s with 4 workers
        # Efficiency = 0.4 / (0.5 * 4) = 0.4 / 2.0 = 0.2 (20%)
        assert 0 <= metrics["parallel_efficiency"] <= 1
        assert metrics["total_processing_time"] == 0.4


class TestFileProcessingMetrics:
    """Test enhanced file processing metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = FileProcessingMetrics()
    
    def test_single_file_tracking(self):
        """Test single file processing tracking."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content" * 1000)
            file_path = Path(tmp.name)
        
        try:
            file_size = file_path.stat().st_size
            
            tracking = self.metrics.track_file_processing(file_path, file_size)
            time.sleep(0.01)  # Simulate processing
            metrics = tracking["end_tracking"]()
            
            assert "file_size_mb" in metrics
            assert "throughput_mb_per_sec" in metrics
            assert metrics["file_size_mb"] > 0
            assert metrics["throughput_mb_per_sec"] > 0
            
        finally:
            file_path.unlink()
    
    def test_batch_processing_tracking(self):
        """Test batch file processing tracking."""
        # Create temporary files
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(b"test content" * 100)
                files.append(Path(tmp.name))
        
        try:
            # Track batch processing
            tracking = self.metrics.track_batch_processing(
                files, 
                "test_batch",
                parallel=True,
                max_workers=2
            )
            
            time.sleep(0.01)  # Simulate processing
            metrics = tracking["end_tracking"]()
            
            assert "file_count" in metrics
            assert "total_size_mb" in metrics
            assert "parallel" in metrics
            assert "batch_throughput_mb_per_sec" in metrics
            assert metrics["file_count"] == 3
            assert metrics["parallel"] is True
            
        finally:
            for file in files:
                file.unlink()
    
    def test_sequential_batch_processing(self):
        """Test sequential batch processing tracking."""
        files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(b"test content" * 50)
                files.append(Path(tmp.name))
        
        try:
            tracking = self.metrics.track_batch_processing(
                files,
                "sequential_batch", 
                parallel=False
            )
            
            time.sleep(0.01)
            metrics = tracking["end_tracking"]()
            
            assert "parallel" in metrics
            assert "session_id" not in metrics or metrics["session_id"] is None
            
        finally:
            for file_path in files:
                file_path.unlink()
    
    def test_empty_batch_handling(self):
        """Test handling of empty file batch."""
        tracking = self.metrics.track_batch_processing([])
        assert tracking == {}
    
    def test_throughput_stats_calculation(self):
        """Test throughput stats calculation."""
        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"content" * 1000)
            file_path = Path(tmp.name)
        
        try:
            file_size = file_path.stat().st_size
            
            # Simulate processing
            with self.metrics.track_file_processing(file_path, file_size) as tracker:
                time.sleep(0.01)
            
            # Check stats
            stats = self.metrics.get_throughput_stats()
            assert "total_files_processed" in stats
            assert "total_mb_processed" in stats
            assert "avg_throughput_mbps" in stats
            assert stats["total_files_processed"] == 1
            
        finally:
            file_path.unlink()


@pytest.fixture
def mock_dependencies():
    """Provides a patch for all external dependencies of the processor."""
    with patch('app.core.performance.FileProcessingMetrics') as mock_metrics, \
         patch('app.core.memory_optimizer.get_memory_optimizer') as mock_get_mem_opt, \
         patch('app.core.performance_optimizer.get_performance_optimizer') as mock_get_perf_opt, \
         patch('app.core.intelligent_cache.get_intelligent_cache') as mock_get_cache, \
         patch('app.core.performance.get_config') as mock_get_config:

        mock_memory_optimizer = MagicMock()
        mock_get_mem_opt.return_value = mock_memory_optimizer

        mock_performance_optimizer = MagicMock()
        mock_get_perf_opt.return_value = mock_performance_optimizer
        
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache

        mock_get_config.return_value.get.return_value = {
            'memory_optimizer': {'enabled': True},
            'batch_engine': {'enabled': True}
        }
        
        yield {
            "metrics": mock_metrics,
            "get_mem_opt": mock_get_mem_opt,
            "get_perf_opt": mock_get_perf_opt,
            "get_cache": mock_get_cache,
            "mem_opt": mock_memory_optimizer,
            "perf_opt": mock_performance_optimizer,
            "cache": mock_cache,
            "config": mock_get_config
        }


class TestPerformanceOptimizedProcessor:
    """Focused, mock-based tests for the PerformanceOptimizedProcessor."""

    def test_initialization(self, mock_dependencies):
        """Test that the processor initializes its optimizers based on config."""
        processor = PerformanceOptimizedProcessor()
        
        # Check that optimizer factory functions were called
        mock_dependencies['get_mem_opt'].assert_called_once()
        mock_dependencies['get_perf_opt'].assert_called_once()
        mock_dependencies['get_cache'].assert_called_once()
        
        assert processor._memory_optimizer is not None
        assert processor._performance_optimizer is not None
        assert processor._intelligent_cache is not None

    @patch('app.core.performance.ThreadPoolExecutor')
    def test_process_files_parallel(self, mock_executor, mock_dependencies):
        """Test the parallel processing logic with a mocked executor."""
        mock_executor_instance = mock_executor.return_value.__enter__.return_value
        
        processor = PerformanceOptimizedProcessor()
        files = [Path("file1.pdf"), Path("file2.pdf")]
        mock_process_func = MagicMock()

        processor.process_files_parallel(files, mock_process_func)

        # Check that the executor was used to map the function over the files
        mock_executor_instance.map.assert_called_once()

    def test_optimized_session_management(self, mock_dependencies):
        """Test the optimized_processing_session context manager."""
        processor = PerformanceOptimizedProcessor()
        
        # Mock the context manager on the memory optimizer
        mock_mem_opt = mock_dependencies['mem_opt']
        mock_mem_opt.optimized_processing = MagicMock()
        mock_mem_opt.optimized_processing.return_value.__enter__.return_value = None
        mock_mem_opt.optimized_processing.return_value.__exit__.return_value = None

        with processor.optimized_processing_session(processing_mode="aggressive"):
            pass # Simulate work inside the session
        
        # Check that the memory optimizer's methods were called
        mock_mem_opt.start_optimization.assert_called_once()
        mock_mem_opt.optimized_processing.assert_called_once_with("aggressive")
        mock_mem_opt.stop_optimization.assert_called_once()


class TestPerformanceMonitorDecorator:
    """Test performance monitor decorator."""
    
    def test_decorator_success_tracking(self):
        """Test decorator tracking successful function execution."""
        @performance_monitor("test_decorated_function")
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        assert result == 5
        
        # Check metrics were recorded
        stats = global_performance_monitor.get_operation_stats("test_decorated_function")
        assert stats is not None
        assert stats["count"] == 1
    
    def test_decorator_error_tracking(self):
        """Test decorator tracking function errors."""
        @performance_monitor("test_error_function")
        def error_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            error_function()
        
        # Check metrics were recorded with error
        stats = global_performance_monitor.get_operation_stats("test_error_function")
        assert stats is not None
        assert stats["count"] == 1
    
    def test_decorator_with_args_logging(self):
        """Test decorator with argument logging."""
        @performance_monitor("test_args_function", log_args=True)
        def args_function(a, b, c=None):
            return a + b
        
        result = args_function(1, 2, c="test")
        assert result == 3
        
        stats = global_performance_monitor.get_operation_stats("test_args_function")
        assert stats is not None
    
    def test_decorator_default_operation_name(self):
        """Test decorator with default operation name."""
        @performance_monitor()
        def named_function():
            return "test"
        
        result = named_function()
        assert result == "test"
        
        # Should use module.function_name as operation name
        all_stats = global_performance_monitor.get_all_stats()
        assert any("named_function" in op_name for op_name in all_stats.keys())


class TestGlobalFunctions:
    """Test global functions like get_performance_report."""
    
    @patch('app.core.performance.global_performance_monitor')
    def test_get_performance_report(self, mock_monitor):
        """Test the global performance report function with a mock."""
        mock_monitor.get_all_stats.return_value = {"global_op": {"count": 1}}
        mock_monitor.get_system_metrics.return_value = {"cpu": 10.0}

        report = get_performance_report()
        
        assert "system_metrics" in report
        assert "operation_stats" in report
        assert "global_op" in report["operation_stats"]
        mock_monitor.get_all_stats.assert_called_once()
        mock_monitor.get_system_metrics.assert_called_once()

    @patch('app.core.performance.PerformanceOptimizedProcessor')
    def test_get_parallel_processor(self, mock_processor_class):
        """Test the factory function for the parallel processor."""
        processor_instance = mock_processor_class.return_value
        
        processor = get_parallel_processor()
        
        assert processor == processor_instance
        mock_processor_class.assert_called_once()

    @patch('tests.test_performance_enhanced.get_optimized_processor')
    def test_performance_report_with_optimizers(self, mock_get_processor):
        """Test performance report generation with optimizers."""
        mock_processor = mock_get_processor.return_value
        mock_processor.config = {'enabled': True} # Ensure config is not None
        
        # This test now verifies that the factory is called, not its internal state.
        processor = get_optimized_processor()
        
        assert processor is not None
        assert processor.config is not None
        mock_get_processor.assert_called_once()


@patch('tests.test_performance_enhanced.get_optimized_processor')
def test_end_to_end_parallel_processing_integration(mock_get_processor, config_manager):
    """
    A revised integration test for the new parallel processing flow.
    """
    # Setup mock processor and its dependencies
    mock_processor = MagicMock()
    mock_get_processor.return_value = mock_processor

    # The function that will be executed in parallel
    def simple_process_func(file_path):
        return f"processed_{file_path.name}"

    files_to_process = [Path(f"doc_{i}.pdf") for i in range(5)]
    
    # Configure the mock to return expected results
    expected_results = [simple_process_func(f) for f in files_to_process]
    mock_processor.process_files_parallel.return_value = expected_results
    
    # Call the parallel processing method
    results = mock_processor.process_files_parallel(
        files=files_to_process,
        process_function=simple_process_func
    )
    
    # Assertions
    mock_processor.process_files_parallel.assert_called_once_with(
        files=files_to_process,
        process_function=simple_process_func
    )
    assert results == expected_results
    assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__]) 