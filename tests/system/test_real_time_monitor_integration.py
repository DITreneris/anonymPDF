import pytest
import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock
import time

from app.services.pdf_processor import PDFProcessor
from app.core.config_manager import ConfigManager
from app.core.real_time_monitor import RealTimeMonitor
from app.core.adaptive.coordinator import AdaptiveLearningCoordinator

# This test file should be self-contained and rely only on fixtures
# provided by pytest from conftest.py files.

@pytest.fixture(scope="function")
def isolated_monitor(tmp_path: Path) -> RealTimeMonitor:
    """
    Provides a fresh, isolated RealTimeMonitor instance for each system test
    to prevent state leakage between tests.
    """
    db_path = tmp_path / f"test_monitor_{hash(time.time())}.db"
    monitor = RealTimeMonitor(db_path=db_path)
    yield monitor
    monitor.shutdown()


@pytest.mark.system
def test_monitoring_end_to_end(
    isolated_monitor: RealTimeMonitor, 
    caplog: pytest.LogCaptureFixture, 
    test_pdf_processor: PDFProcessor,
    adaptive_coordinator: AdaptiveLearningCoordinator
):
    """
    A full end-to-end test of the real-time monitoring system.
    This test verifies that when the PDFProcessor runs, its activities
    are correctly logged by the monitoring system.
    """
    caplog.set_level(logging.INFO)
    
    # 1. Setup: Use the fixture-provided PDFProcessor and manually inject the isolated monitor.
    # In a real application, this injection would be handled by the FastAPI dependency system.
    processor_under_test = test_pdf_processor
    processor_under_test.monitor = isolated_monitor

    # 2. Action: Perform an action that triggers monitored events.
    # Create a temporary PDF for this system test.
    import tempfile
    import fitz
    import time
    
    # Create temporary directory and file manually for better control
    temp_dir = Path(tempfile.mkdtemp())
    test_document_path = temp_dir / "test_monitor.pdf"
    
    # Create a simple PDF with PII content
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "This is a test PDF with the name John Doe. Email: john.doe@test.com")
    doc.save(test_document_path)
    doc.close()
    
    try:
        # The process_pdf method is the main entry point we need to test.
        result = asyncio.run(processor_under_test.process_pdf(test_document_path))
        
        # Brief pause to ensure all file handles are released
        time.sleep(0.1)
    finally:
        # Robust cleanup with retry logic for Windows
        import shutil
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.2)  # Wait before retry
                # On final attempt, just ignore the error to not fail the test

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
    assert pii_log_entry["document_id"] is not None  # document_id is a separate column, not in details
    
    # 5. Summary Verification: Check that the monitor's summary function works.
    summary = isolated_monitor.get_summary()
    
    assert summary["total_events"] > 0
    assert summary["average_cpu_percent"] is not None
    assert summary["average_memory_mb"] is not None 