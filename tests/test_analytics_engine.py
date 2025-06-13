"""
Unit tests for the refactored Analytics Engine.
"""

import pytest
import sqlite3
import tempfile
import shutil
import gc
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

from app.core.analytics_engine import QualityAnalyzer, QualityInsightsGenerator

@pytest.fixture
def temp_db():
    """Provides a temporary directory and a path to a test database."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "analytics_test.db"
    yield db_path
    try:
        # Ensure any open connections are closed
        gc.collect()
        # Add a longer delay to allow file handles to be released
        time.sleep(0.5)
        # Try to remove the directory multiple times if needed
        for _ in range(3):
            try:
                shutil.rmtree(temp_dir)
                break
            except PermissionError:
                gc.collect()
                time.sleep(0.5)
    except Exception as e:
        print(f"Warning: Failed to clean up temp directory: {e}")

@pytest.fixture
def analyzer(temp_db):
    """
    Provides a QualityAnalyzer instance with a temporary database.
    This fixture ensures the database connection is closed and the object
    is garbage-collected to prevent file locking issues on Windows.
    """
    a = QualityAnalyzer(db_path=temp_db)
    try:
        yield a
    finally:
        # Forcefully close the connection and release handles.
        try:
            if a and a.conn:
                a.conn.close()
        except Exception as e:
            print(f"Error closing DB connection in fixture: {e}")
        
        a = None  # De-reference the object.
        gc.collect() # Force garbage collection.
        time.sleep(0.2) # Give Windows time to release the file lock.

def create_mock_detection_result(category: str, confidence: float) -> dict:
    """Helper to create a mock detection result dictionary."""
    return {
        "text": "some pii", "category": category, "context": "surrounding text",
        "position": 10, "ml_confidence": confidence, "ml_prediction": None,
        "priority2_confidence": 0.0, "fallback_used": False, "processing_time_ms": 20.0,
        "features_extracted": 5, "document_type": "test_doc", "language": "en",
        "timestamp": datetime.now().isoformat(), "pattern_type": "test_pattern"
    }

class TestQualityAnalyzer:
    """Tests the core functionality of the QualityAnalyzer class."""

    def test_initialization(self, temp_db):
        """Test that the analyzer initializes correctly and creates the DB file."""
        assert not Path(temp_db).exists()
        analyzer = QualityAnalyzer(db_path=temp_db)
        try:
            assert Path(temp_db).exists()
        finally:
            analyzer.close()
            # Force garbage collection
            analyzer = None
            gc.collect()
            time.sleep(0.1)

    def test_add_detection_results(self, analyzer, temp_db):
        """Test adding detection results to the database."""
        results = [
            create_mock_detection_result("EMAIL", 0.95),
            create_mock_detection_result("PHONE", 0.88)
        ]
        analyzer.add_detection_results("doc_123", results)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM detection_results")
            assert cursor.fetchone()[0] == 2
            cursor = conn.execute("SELECT category FROM detection_results WHERE category = 'EMAIL'")
            assert cursor.fetchone() is not None

    def test_analyze_detection_quality_empty(self, analyzer):
        """Test analysis on an empty database."""
        assert analyzer.analyze_detection_quality() == {}

    def test_analyze_detection_quality_with_data(self, analyzer):
        """Test analysis with some data."""
        analyzer.add_detection_results("doc_1", [create_mock_detection_result("EMAIL", 0.95)])
        analyzer.add_detection_results("doc_2", [create_mock_detection_result("PHONE", 0.80), create_mock_detection_result("PHONE", 0.90)])
        
        metrics = analyzer.analyze_detection_quality()
        
        assert "EMAIL" in metrics and "PHONE" in metrics
        assert metrics["EMAIL"]["total_detections"] == 1
        assert metrics["PHONE"]["total_detections"] == 2
        assert metrics["PHONE"]["avg_confidence"] == pytest.approx(0.85)

    def test_close_connection(self, analyzer):
        """Test that the close method closes the connection."""
        assert analyzer.conn is not None
        analyzer.close()
        assert analyzer.conn is None

class TestQualityInsightsGenerator:
    """Tests the QualityInsightsGenerator."""

    def test_generate_report_with_insights(self, insights_generator, mock_analyzer):
        """Test generating a report with insights."""
        # Mock the analyzer's methods to return predictable data
        summary_data = {
            "overall": {"avg_confidence": 0.8, "total_detections": 10},
            "by_category": {
                "EMAIL": {"avg_confidence": 0.9, "total_detections": 5},
                "PHONE": {"avg_confidence": 0.7, "total_detections": 5}
            }
        }
        mock_analyzer.get_quality_summary.return_value = summary_data
        
        # Act
        report = insights_generator.generate_report()

        # Assert
        assert "quality_summary" in report, "The key 'quality_summary' should be in the report"
        assert "insights" in report
        assert report["quality_summary"] == summary_data
        assert isinstance(report["insights"], list)
        # Check if an insight was generated for low confidence on PHONE
        assert any("low confidence" in s.lower() and "phone" in s.lower() for s in report["insights"])

@pytest.fixture
def mock_analyzer():
    """Provides a mocked QualityAnalyzer."""
    return Mock(spec=QualityAnalyzer)

@pytest.fixture
def insights_generator(mock_analyzer):
    """Provides a QualityInsightsGenerator instance with a mocked analyzer."""
    return QualityInsightsGenerator(analyzer=mock_analyzer)
