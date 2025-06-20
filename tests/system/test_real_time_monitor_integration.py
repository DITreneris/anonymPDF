import pytest
import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock
import time

from app.services.pdf_processor import PDFProcessor
from app.core.config_manager import ConfigManager
from app.core.real_time_monitor import RealTimeMonitor

# This test file should be self-contained and rely only on fixtures
# provided by pytest from conftest.py files.

@pytest.fixture(scope="function")
def isolated_monitor(tmp_path: Path) -> RealTimeMonitor:
    """
    Provides a fresh, isolated RealTimeMonitor instance for each system test
    to prevent state leakage between tests.
    """
    db_path = tmp_path / f"test_monitor_{hash(time.time())}.db"
    monitor = RealTimeMonitor(db_path=str(db_path))
    yield monitor
    monitor.shutdown()


@pytest.mark.system
def test_monitoring_end_to_end(
    isolated_monitor: RealTimeMonitor, 
    caplog: pytest.LogCaptureFixture, 
    config_manager: ConfigManager
):
    """
    A full end-to-end test of the real-time monitoring system.
    This test verifies that when the PDFProcessor runs, its activities
    are correctly logged by the monitoring system.
    """
    caplog.set_level(logging.INFO)
    
    # 1. Setup: Create a PDFProcessor and manually inject the isolated monitor.
    # In a real application, this injection would be handled by the FastAPI dependency system.
    processor_under_test = PDFProcessor(config_manager=config_manager)
    processor_under_test.monitor = isolated_monitor

    # 2. Action: Perform an action that triggers monitored events.
    # We use a known sample document for this system test.
    test_document_path = str(Path("tests/samples/simple_pii_document.txt"))
    
    # The process_pdf method is the main entry point we need to test.
    asyncio.run(processor_under_test.process_pdf(test_document_path))

    # 3. Verification: Check the database via the monitor to ensure events were logged.
    logged_metrics = isolated_monitor.get_latest_metrics(limit=20)
    
    assert len(logged_metrics) > 0, "No metrics were logged to the real-time monitor database."
    
    # Create a set of event names for easy lookup.
    logged_event_names = {log["event_name"] for log in logged_metrics}
    
    # We expect specific events to be logged during the processing pipeline.
    assert "file_processing_completed" in logged_event_names
    assert "pii_detection_completed" in logged_event_names
    
    # 4. Detailed Verification: Inspect the content of a specific log entry.
    pii_log_entry = next((log for log in logged_metrics if log["event_name"] == "pii_detection_completed"), None)
    
    assert pii_log_entry is not None, "The 'pii_detection_completed' event was not found."
    assert "details" in pii_log_entry
    assert "duration_ms" in pii_log_entry["details"]
    assert pii_log_entry["details"]["document_id"] is not None
    
    # 5. Summary Verification: Check that the monitor's summary function works.
    summary = isolated_monitor.get_summary()
    
    assert summary["total_events"] > 0
    assert summary["average_cpu_percent"] is not None
    assert summary["average_memory_mb"] is not None 