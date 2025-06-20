# This file is being rewritten to align with the refactored RealTimeMonitor API.

"""
Tests for the refactored Real-Time Monitor.

This suite tests the current, simplified implementation of the RealTimeMonitor,
focusing on its core responsibilities: event logging, metric retrieval,
and summary generation.
"""

import pytest
import tempfile
import time
import threading
from pathlib import Path

# The new, simplified monitor is the primary target for testing.
# The other classes are not directly used by it and are tested elsewhere.
from app.core.real_time_monitor import RealTimeMonitor

@pytest.mark.unit
class TestRealTimeMonitor:
    """
    Unit tests for the RealTimeMonitor's core functionality.
    
    These tests verify that the monitor can be initialized, log events to a
    database, and retrieve data correctly in a controlled, single-threaded
    environment.
    """

    def setup_method(self):
        """Set up a test environment before each test."""
        # Forcefully reset the singleton to ensure test isolation.
        RealTimeMonitor.reset_for_testing()
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_monitor_unit.db"
        self.monitor = RealTimeMonitor(db_path=str(self.db_path))

    def teardown_method(self):
        """Clean up the test environment after each test."""
        self.monitor.shutdown()
        self.temp_dir.cleanup()

    def test_initialization_creates_database(self):
        """Verify that the monitor initializes correctly and creates the DB file."""
        assert self.db_path.exists(), "Database file should be created on initialization."

    def test_log_single_event(self):
        """Test logging a single, complete performance event."""
        self.monitor.log_event("test_event", duration=0.5, document_id="doc1", details={"status": "success"})
        
        metrics = self.monitor.get_latest_metrics(limit=1)
        assert len(metrics) == 1
        
        event = metrics[0]
        assert event['event_name'] == "test_event"
        assert event['duration'] == 0.5
        assert event['document_id'] == "doc1"
        assert '"status": "success"' in event['details']

    def test_get_summary_with_data(self):
        """Test the summary generation logic with multiple events."""
        self.monitor.log_event("event1", duration=0.1)
        self.monitor.log_event("event2", duration=0.2)
        
        summary = self.monitor.get_summary()
        
        assert "total_events" in summary
        assert summary["total_events"] == 2
        assert "average_duration_ms" in summary
        assert summary["average_duration_ms"] == 150.0
        assert "average_cpu_percent" in summary
        assert "average_memory_mb" in summary

    def test_get_summary_no_data(self):
        """Test that the summary returns a clean status when no data is available."""
        summary = self.monitor.get_summary()
        assert summary.get('status') == "No data available."

    def test_get_latest_metrics_ordering_and_limit(self):
        """Verify that get_latest_metrics respects the limit and order."""
        for i in range(5):
            self.monitor.log_event(f"event_{i}")
            time.sleep(0.01)

        metrics = self.monitor.get_latest_metrics(limit=3)
        assert len(metrics) == 3
        assert metrics[0]['event_name'] == 'event_4'
        assert metrics[1]['event_name'] == 'event_3'
        assert metrics[2]['event_name'] == 'event_2'


@pytest.mark.integration
class TestRealTimeMonitorIntegration:
    """
    Integration tests for the RealTimeMonitor, focusing on concurrency.
    """

    def setup_method(self):
        """Set up a test environment before each test."""
        # Forcefully reset the singleton to ensure test isolation.
        RealTimeMonitor.reset_for_testing()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_monitor_integration.db"
        self.monitor = RealTimeMonitor(db_path=str(self.db_path))

    def teardown_method(self):
        """Clean up the test environment after each test."""
        self.monitor.shutdown()
        self.temp_dir.cleanup()

    def test_singleton_instance(self):
        """Verify that the class maintains a singleton instance."""
        monitor1 = RealTimeMonitor(db_path=str(self.db_path))
        monitor2 = RealTimeMonitor(db_path=str(self.db_path))
        assert monitor1 is self.monitor
        assert monitor2 is self.monitor

    def test_concurrent_logging_is_thread_safe(self):
        """Test that logging from multiple threads is handled correctly without data loss or corruption."""
        
        def logger_thread(thread_id: int, events_to_log: int):
            for i in range(events_to_log):
                self.monitor.log_event(f"event_thread_{thread_id}", duration=i*0.01)
        
        num_threads = 5
        events_per_thread = 20
        threads = []
        
        for i in range(num_threads):
            thread = threading.Thread(target=logger_thread, args=(i, events_per_thread))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify that all events were logged correctly.
        metrics = self.monitor.get_latest_metrics(limit=200)
        total_expected_events = num_threads * events_per_thread
        assert len(metrics) == total_expected_events, f"Expected {total_expected_events} events, but found {len(metrics)}."

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 