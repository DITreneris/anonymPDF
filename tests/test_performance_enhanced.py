"""
Tests for Enhanced Performance Module
"""

import pytest
import time
import threading
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.core.performance import (
    PerformanceMonitor,
    ParallelPerformanceTracker,
    FileProcessingMetrics,
    PerformanceOptimizedProcessor,
    performance_monitor
)
from app.core.config_manager import get_config


def top_level_mock_process_func(file_path):
    """A top-level function that can be pickled for multiprocessing tests."""
    return f"processed_{file_path.name}"


@pytest.fixture(scope="module")
def config_manager():
    """Fixture to provide a config manager instance for the test module."""
    return get_config()


class TestEnhancedPerformanceMonitor:
    """Tests for the enhanced, thread-safe PerformanceMonitor."""

    @pytest.fixture
    def monitor(self):
        """Provides a clean PerformanceMonitor instance for each test."""
        return PerformanceMonitor()

    def test_thread_safe_metrics_tracking(self, monitor):
        """Ensures that tracking operations from multiple threads works correctly."""
        def track_in_thread(op_name):
            tracker = monitor.track_operation(op_name)
            time.sleep(0.01)
            tracker['end_tracking']()

        threads = [threading.Thread(target=track_in_thread, args=(f"op_{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_stats = monitor.get_all_stats()
        assert len(all_stats) == 5
        assert "op_0" in all_stats
        assert all_stats["op_1"]["count"] == 1

    def test_system_metrics_collection(self, monitor):
        """Tests that system metrics can be collected."""
        metrics = monitor.get_system_metrics()
        assert "process_memory_mb" in metrics
        assert "system_memory_percent" in metrics
        assert metrics["process_memory_mb"] > 0


class TestParallelPerformanceTracker:
    """Tests for the ParallelPerformanceTracker."""

    @pytest.fixture
    def tracker(self):
        return ParallelPerformanceTracker()

    def test_parallel_session_lifecycle(self, tracker):
        """Tests the full lifecycle of a parallel session."""
        session_id = "test_session"
        tracker.start_parallel_session(session_id, worker_count=4)
        time.sleep(0.01)
        tracker.track_task_completion(session_id, success=True)
        tracker.track_task_completion(session_id, success=False)
        metrics = tracker.end_parallel_session(session_id)

        assert metrics['tasks_completed'] == 1
        assert metrics['tasks_failed'] == 1
        assert metrics['throughput_tasks_per_second'] > 0


class TestFileProcessingMetrics:
    """Tests for the FileProcessingMetrics tracker."""

    @pytest.fixture
    def file_tracker(self):
        return FileProcessingMetrics()

    def test_file_tracking_and_stats(self, file_tracker):
        """Tests tracking a file and getting stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("data")
            
            tracker = file_tracker.track_file_processing(test_file, test_file.stat().st_size)
            time.sleep(0.01)
            tracker['end_tracking']()

            stats = file_tracker.get_throughput_stats("pdf_processing")
            assert stats['count'] == 1
            assert stats['avg_throughput_mbps'] > 0

    def test_empty_stats_handling(self, file_tracker):
        """Tests that getting stats for an untracked operation returns a default dict."""
        stats = file_tracker.get_throughput_stats("nonexistent_op")
        assert stats == {'count': 0, 'avg_throughput_mbps': 0}


class TestPerformanceOptimizedProcessor:
    """Tests for the main processor class."""

    @pytest.fixture
    def processor(self):
        return PerformanceOptimizedProcessor()

    def test_parallel_processing_with_processes(self, processor):
        """Tests the parallel file processing with process pool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / "test1.txt", Path(tmpdir) / "test2.txt"]
            for f in files:
                f.write_text(f"content for {f.name}")
            
            results = processor.process_files_parallel(
                files, 
                top_level_mock_process_func, 
                use_processes=True,
                max_workers=2
            )
            
            assert sorted(results) == ["processed_test1.txt", "processed_test2.txt"]
            
    def test_parallel_processing_with_threads(self, processor):
        """Tests the parallel file processing with thread pool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / "test1.txt", Path(tmpdir) / "test2.txt"]
            for f in files:
                f.write_text(f"content for {f.name}")
            
            results = processor.process_files_parallel(
                files, 
                top_level_mock_process_func, 
                use_processes=False,
                max_workers=2
            )
            
            assert sorted(results) == ["processed_test1.txt", "processed_test2.txt"]