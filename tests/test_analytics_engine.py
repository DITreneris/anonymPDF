"""
Tests for Analytics Engine - Session 4 Priority 3 Implementation

Tests the comprehensive quality analysis, detection assessment,
and improvement suggestions functionality.
"""

import pytest
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sqlite3

from app.core.analytics_engine import (
    QualityAnalyzer, 
    QualityInsightsGenerator,
    QualityIssueType,
    QualityIssue,
    DetectionQualityMetrics
)
from app.core.ml_integration import DetectionResult


@pytest.mark.unit
class TestQualityAnalyzer:
    """Test the QualityAnalyzer class."""
    
    def setup_method(self):
        """Setup test environment."""
        # Use temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        config = {
            'storage_path': f"{self.temp_dir}/test_analytics.db",
            'quality_thresholds': {
                'min_confidence': 0.7,
                'max_false_positive_rate': 0.05,
                'min_precision': 0.9,
                'min_recall': 0.85,
                'max_processing_time_ms': 2000
            }
        }
        self.analyzer = QualityAnalyzer(config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_detection_result(self, category: str = "names", 
                                   confidence: float = 0.8,
                                   processing_time_ms: float = 100,
                                   pattern_type: str = "person_name") -> Mock:
        """Create a mock detection result for testing."""
        result = Mock()
        result.timestamp = datetime.now()
        result.category = category
        result.text = f"test_{category}"
        result.ml_confidence = confidence
        result.processing_time_ms = processing_time_ms
        result.fallback_used = False
        result.pattern_type = pattern_type
        return result
    
    def test_initialization(self):
        """Test QualityAnalyzer initialization."""
        assert self.analyzer is not None
        assert self.analyzer.quality_thresholds['min_confidence'] == 0.7
        assert len(self.analyzer.detection_history) == 0
        assert len(self.analyzer.quality_issues) == 0
    
    def test_add_detection_result(self):
        """Test adding detection results."""
        result = self.create_mock_detection_result()
        
        self.analyzer.add_detection_result(result)
        
        assert len(self.analyzer.detection_history) == 1
        assert len(self.analyzer.category_metrics['names']) == 1
        assert len(self.analyzer.pattern_performance['person_name']) == 1
    
    def test_add_detection_result_with_ground_truth(self):
        """Test adding detection results with ground truth."""
        result = self.create_mock_detection_result()
        
        self.analyzer.add_detection_result(result, ground_truth=True)
        
        detection_data = list(self.analyzer.detection_history)[0]
        assert detection_data['ground_truth'] is True
    
    def test_analyze_detection_quality_empty(self):
        """Test quality analysis with no data."""
        metrics = self.analyzer.analyze_detection_quality()
        
        assert len(metrics) == 0
    
    def test_analyze_detection_quality_single_category(self):
        """Test quality analysis for single category."""
        # Add multiple detection results
        for i in range(5):
            result = self.create_mock_detection_result(
                confidence=0.8 + i * 0.01,
                processing_time_ms=100 + i * 10
            )
            self.analyzer.add_detection_result(result)
        
        metrics = self.analyzer.analyze_detection_quality()
        
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric.category == 'names'
        assert metric.total_detections == 5
        assert 0.8 <= metric.avg_confidence <= 0.85
        assert metric.processing_time_ms > 0
    
    def test_analyze_detection_quality_multiple_categories(self):
        """Test quality analysis for multiple categories."""
        categories = ['names', 'addresses', 'phone_numbers']
        
        for category in categories:
            for i in range(3):
                result = self.create_mock_detection_result(category=category)
                self.analyzer.add_detection_result(result)
        
        metrics = self.analyzer.analyze_detection_quality()
        
        assert len(metrics) == 3
        category_names = [m.category for m in metrics]
        assert set(category_names) == set(categories)
    
    def test_analyze_detection_quality_with_document_type_filter(self):
        """Test quality analysis with document type filter."""
        # Add detections with different document types
        result_doc_A_cat1 = self.create_mock_detection_result(category="CAT1")
        setattr(result_doc_A_cat1, 'document_type', "doc_type_A") # Use setattr for mock object
        self.analyzer.add_detection_result(result_doc_A_cat1)

        result_doc_B_cat1 = self.create_mock_detection_result(category="CAT1")
        setattr(result_doc_B_cat1, 'document_type', "doc_type_B")
        self.analyzer.add_detection_result(result_doc_B_cat1)

        result_doc_A_cat2 = self.create_mock_detection_result(category="CAT2")
        setattr(result_doc_A_cat2, 'document_type', "doc_type_A")
        self.analyzer.add_detection_result(result_doc_A_cat2)

        # Analyze for CAT1, doc_type_A
        metrics_cat1_docA = self.analyzer.analyze_detection_quality(category="CAT1", document_type_filter="doc_type_A")
        assert len(metrics_cat1_docA) == 1
        assert metrics_cat1_docA[0].category == "CAT1"
        assert metrics_cat1_docA[0].total_detections == 1 
        # We need to ensure that the internal detection_data in QualityAnalyzer has document_type
        # The add_detection_result method needs to store it. Let's check.
        # It seems detection_data['document_type'] = getattr(result, 'document_type', 'unknown') 
        # is missing in QualityAnalyzer.add_detection_result. That's a bug in the source if not added!
        # For the test, let's assume it is stored, or the mock needs to reflect how it is stored.
        # The create_mock_detection_result doesn't set document_type on the mock object in a way add_detection_result would pick it up by default.
        # The setattr above handles this for the mock passed to add_detection_result.

        # Let's make the mock more explicit for document_type if needed for the test of analyze_detection_quality.
        # The `add_detection_result` in QualityAnalyzer does: `'document_type': getattr(result, 'document_type', 'unknown')`
        # So setattr on the mock is correct.

        # Analyze for CAT1, all doc types
        metrics_cat1_all = self.analyzer.analyze_detection_quality(category="CAT1")
        assert len(metrics_cat1_all) == 1
        assert metrics_cat1_all[0].total_detections == 2

        # Analyze all categories, doc_type_A
        metrics_all_docA = self.analyzer.analyze_detection_quality(document_type_filter="doc_type_A")
        assert len(metrics_all_docA) == 2 # CAT1 and CAT2 from doc_type_A
        found_cat1 = any(m.category == "CAT1" and m.total_detections == 1 for m in metrics_all_docA)
        found_cat2 = any(m.category == "CAT2" and m.total_detections == 1 for m in metrics_all_docA)
        assert found_cat1 and found_cat2
    
    def test_analyze_detection_quality_with_ground_truth(self):
        """Test quality analysis with ground truth data."""
        # Add true positives
        for i in range(3):
            result = self.create_mock_detection_result(confidence=0.9)
            self.analyzer.add_detection_result(result, ground_truth=True)
        
        # Add false positives
        for i in range(1):
            result = self.create_mock_detection_result(confidence=0.8)
            self.analyzer.add_detection_result(result, ground_truth=False)
        
        metrics = self.analyzer.analyze_detection_quality()
        
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric.precision_estimate > 0.5  # Should be 3/4 = 0.75
        assert metric.false_positive_rate < 0.5
    
    def test_detect_quality_issues_low_confidence(self):
        """Test detection of low confidence issues."""
        # Add low confidence detections
        for i in range(5):
            result = self.create_mock_detection_result(confidence=0.5)  # Below threshold
            self.analyzer.add_detection_result(result)
        
        issues = self.analyzer.detect_quality_issues()
        
        assert len(issues) > 0
        low_conf_issues = [i for i in issues if i.issue_type == QualityIssueType.LOW_CONFIDENCE]
        assert len(low_conf_issues) > 0
        assert low_conf_issues[0].severity in ['medium', 'high']
    
    def test_detect_quality_issues_performance_degradation(self):
        """Test detection of performance degradation."""
        # Add slow processing detections
        for i in range(5):
            result = self.create_mock_detection_result(processing_time_ms=3000)  # Above threshold
            self.analyzer.add_detection_result(result)
        
        issues = self.analyzer.detect_quality_issues()
        
        perf_issues = [i for i in issues if i.issue_type == QualityIssueType.PERFORMANCE_DEGRADATION]
        assert len(perf_issues) > 0
        assert perf_issues[0].severity == 'medium'
    
    def test_detect_quality_issues_high_false_positive(self):
        """Test detection of high false positive rate."""
        # Add false positives
        for i in range(8):
            result = self.create_mock_detection_result(confidence=0.9)
            self.analyzer.add_detection_result(result, ground_truth=False)  # False positive
        
        # Add few true positives
        for i in range(2):
            result = self.create_mock_detection_result(confidence=0.9)
            self.analyzer.add_detection_result(result, ground_truth=True)
        
        issues = self.analyzer.detect_quality_issues()
        
        fp_issues = [i for i in issues if i.issue_type == QualityIssueType.HIGH_FALSE_POSITIVE]
        assert len(fp_issues) > 0
    
    def test_get_quality_report_empty(self):
        """Test quality report with no data."""
        report = self.analyzer.get_quality_report()
        
        assert 'timestamp' in report
        assert 'summary' in report
        assert report['summary']['total_categories_analyzed'] == 0
        assert report['summary']['total_quality_issues'] == 0
        assert len(report['category_metrics']) == 0
        assert len(report['quality_issues']) == 0
    
    def test_get_quality_report_with_data(self):
        """Test quality report with data."""
        # Add some detection results
        for i in range(5):
            result = self.create_mock_detection_result()
            self.analyzer.add_detection_result(result)
        
        # Trigger quality analysis
        self.analyzer.analyze_detection_quality()
        
        report = self.analyzer.get_quality_report()
        
        assert report['summary']['total_categories_analyzed'] == 1
        assert len(report['category_metrics']) == 1
        assert report['category_metrics'][0]['category'] == 'names'
        assert report['category_metrics'][0]['total_detections'] == 5
    
    def test_get_improvement_suggestions_no_issues(self):
        """Test improvement suggestions with no issues."""
        suggestions = self.analyzer.get_improvement_suggestions()
        
        assert len(suggestions) == 0
    
    def test_get_improvement_suggestions_with_issues(self):
        """Test improvement suggestions with quality issues."""
        # Create multiple similar issues
        for i in range(3):
            issue = QualityIssue(
                issue_type=QualityIssueType.LOW_CONFIDENCE,
                severity='medium',
                description=f"Test issue {i}",
                affected_patterns=[f'category_{i}'],
                metrics={'confidence': 0.5},
                suggested_actions=['Test action']
            )
            self.analyzer.quality_issues.append(issue)
        
        suggestions = self.analyzer.get_improvement_suggestions()
        
        assert len(suggestions) > 0
        assert suggestions[0]['category'] == 'systematic_issue'
        assert 'low_confidence' in suggestions[0]['description']
    
    def test_storage_integration(self):
        """Test that data is properly stored and retrieved."""
        # Add a detection result
        result1 = self.create_mock_detection_result(category="DB_CAT1", confidence=0.9)
        setattr(result1, 'document_type', "db_doc_type_1")
        self.analyzer.add_detection_result(result1)
        
        # Analyze and store metrics for it
        metrics1 = self.analyzer.analyze_detection_quality(category="DB_CAT1")
        assert len(metrics1) == 1

        # Create a new analyzer instance with the same DB path
        # to ensure data was persisted and can be loaded (implicitly tested by get_quality_trends later)
        # For now, check if the metrics table has entries
        with sqlite3.connect(self.analyzer.storage_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM quality_metrics WHERE category = ?", ("DB_CAT1",))
            count = cursor.fetchone()[0]
            assert count > 0 # Should be at least 1 from the analyze_detection_quality call above

    def test_get_quality_trends_insufficient_data(self):
        """Test get_quality_trends when there are not enough data points."""
        trend = self.analyzer.get_quality_trends(category="CAT_TREND", metric_name="avg_confidence")
        assert trend is not None
        assert trend.trend_direction == "insufficient_data"
        assert trend.num_data_points == 0

    def test_get_quality_trends_valid_metric(self):
        """Test get_quality_trends with some data producing a trend."""
        category_name = "TREND_CAT_1"
        metric_to_trend = "avg_confidence"

        # Simulate storing daily metrics over several days
        # Need to mock the database call or insert data directly for this test
        # Let's insert data directly into the test DB for QualityAnalyzer instance
        conn = sqlite3.connect(self.analyzer.storage_path)
        cursor = conn.cursor()
        today = datetime.now()
        
        # Day 1: Low confidence
        cursor.execute("INSERT INTO quality_metrics (timestamp, category, avg_confidence) VALUES (?, ?, ?)", 
                       ((today - timedelta(days=2)).isoformat(), category_name, 0.6))
        # Day 2: Medium confidence
        cursor.execute("INSERT INTO quality_metrics (timestamp, category, avg_confidence) VALUES (?, ?, ?)", 
                       ((today - timedelta(days=1)).isoformat(), category_name, 0.7))
        # Day 3: High confidence (today)
        cursor.execute("INSERT INTO quality_metrics (timestamp, category, avg_confidence) VALUES (?, ?, ?)", 
                       (today.isoformat(), category_name, 0.85))
        conn.commit()
        conn.close()

        trend = self.analyzer.get_quality_trends(category=category_name, metric_name=metric_to_trend, days_back=3)
        
        assert trend is not None
        assert trend.metric_name == metric_to_trend
        assert trend.category == category_name
        assert trend.num_data_points == 3
        assert trend.trend_direction == "improving" 
        assert trend.start_value == 0.6
        assert trend.end_value == 0.85
        assert trend.change_percentage == pytest.approx(((0.85 - 0.6) / 0.6) * 100, 0.01)
        assert len(trend.data_points) == 3
        assert trend.data_points[0][1] == 0.6
        assert trend.data_points[2][1] == 0.85

    def test_get_quality_trends_declining(self):
        """Test get_quality_trends for a declining metric (e.g., false_positive_rate)."""
        category_name = "TREND_CAT_FPR"
        metric_to_trend = "false_positive_rate" # Lower is better

        conn = sqlite3.connect(self.analyzer.storage_path)
        cursor = conn.cursor()
        today = datetime.now()
        cursor.execute("INSERT INTO quality_metrics (timestamp, category, false_positive_rate) VALUES (?, ?, ?)", 
                       ((today - timedelta(days=2)).isoformat(), category_name, 0.20))
        cursor.execute("INSERT INTO quality_metrics (timestamp, category, false_positive_rate) VALUES (?, ?, ?)", 
                       ((today - timedelta(days=1)).isoformat(), category_name, 0.15))
        cursor.execute("INSERT INTO quality_metrics (timestamp, category, false_positive_rate) VALUES (?, ?, ?)", 
                       (today.isoformat(), category_name, 0.10))
        conn.commit()
        conn.close()

        trend = self.analyzer.get_quality_trends(category=category_name, metric_name=metric_to_trend, days_back=3)
        assert trend is not None
        assert trend.trend_direction == "improving" # Lower FPR is improving
        assert trend.start_value == 0.20
        assert trend.end_value == 0.10

    def test_get_quality_trends_invalid_metric(self):
        """Test get_quality_trends with an invalid metric name."""
        trend = self.analyzer.get_quality_trends(category="CAT_ANY", metric_name="non_existent_metric")
        assert trend is None # Should return None for invalid metric name
    
    def test_thread_safety(self):
        """Test thread safety of quality analyzer."""
        import threading
        
        def add_results():
            for i in range(10):
                result = self.create_mock_detection_result()
                self.analyzer.add_detection_result(result)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_results)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all results were added
        assert len(self.analyzer.detection_history) == 30


@pytest.mark.unit
class TestQualityInsightsGenerator:
    """Test the QualityInsightsGenerator class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        config = {'storage_path': f"{self.temp_dir}/test_analytics.db"}
        self.analyzer = QualityAnalyzer(config)
        self.insights_generator = QualityInsightsGenerator(self.analyzer)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_detection_result(self, pattern_type: str = "person_name", 
                                   confidence: float = 0.8,
                                   processing_time_ms: float = 100) -> Mock:
        """Create a mock detection result for testing."""
        result = Mock()
        result.timestamp = datetime.now()
        result.category = "names"
        result.text = f"test_{pattern_type}"
        result.ml_confidence = confidence
        result.processing_time_ms = processing_time_ms
        result.fallback_used = False
        result.pattern_type = pattern_type
        return result
    
    def test_generate_pattern_insights_empty(self):
        """Test pattern insights with no data."""
        insights = self.insights_generator.generate_pattern_insights()
        
        assert 'best_performing_patterns' in insights
        assert 'underperforming_patterns' in insights
        assert 'optimization_opportunities' in insights
        assert len(insights['best_performing_patterns']) == 0
        assert len(insights['underperforming_patterns']) == 0
    
    def test_generate_pattern_insights_with_data(self):
        """Test pattern insights with pattern data."""
        # Add best performing pattern
        for i in range(10):
            result = self.create_mock_detection_result(
                pattern_type="high_perf_pattern",
                confidence=0.9,
                processing_time_ms=200
            )
            self.analyzer.add_detection_result(result)
        
        # Add underperforming pattern
        for i in range(10):
            result = self.create_mock_detection_result(
                pattern_type="low_perf_pattern",
                confidence=0.5,
                processing_time_ms=3000
            )
            self.analyzer.add_detection_result(result)
        
        insights = self.insights_generator.generate_pattern_insights()
        
        assert len(insights['best_performing_patterns']) > 0
        assert len(insights['underperforming_patterns']) > 0
        
        best_pattern = insights['best_performing_patterns'][0]
        assert best_pattern['pattern_type'] == 'high_perf_pattern'
        assert best_pattern['avg_confidence'] > 0.85
        
        worst_pattern = insights['underperforming_patterns'][0]
        assert worst_pattern['pattern_type'] == 'low_perf_pattern'
        assert worst_pattern['avg_confidence'] < 0.6
    
    def test_generate_model_insights_insufficient_data(self):
        """Test model insights with insufficient data."""
        insights = self.insights_generator.generate_model_insights()
        
        assert insights['status'] == 'insufficient_data'
    
    def test_generate_model_insights_excellent_model(self):
        """Test model insights with excellent model performance."""
        # Add high quality detections
        for i in range(10):
            result = self.create_mock_detection_result(confidence=0.95)
            self.analyzer.add_detection_result(result, ground_truth=True)
        
        # Trigger quality analysis
        self.analyzer.analyze_detection_quality()
        
        insights = self.insights_generator.generate_model_insights()
        
        assert insights['model_health'] == 'excellent'
        assert insights['overall_precision'] > 0.9
        assert insights['overall_confidence'] > 0.9
        assert len(insights['recommendations']) > 0
    
    def test_generate_model_insights_needs_improvement(self):
        """Test model insights with poor model performance."""
        # Add low quality detections
        for i in range(5):
            result = self.create_mock_detection_result(confidence=0.6)
            self.analyzer.add_detection_result(result, ground_truth=True)
        
        # Add false positives
        for i in range(5):
            result = self.create_mock_detection_result(confidence=0.7)
            self.analyzer.add_detection_result(result, ground_truth=False)
        
        # Trigger quality analysis
        self.analyzer.analyze_detection_quality()
        
        insights = self.insights_generator.generate_model_insights()
        
        assert insights['model_health'] == 'needs_improvement'
        assert len(insights['recommendations']) > 0
        assert any('precision' in rec for rec in insights['recommendations'])


@pytest.mark.integration
class TestAnalyticsEngineIntegration:
    """Integration tests for analytics engine with existing components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        config = {'storage_path': f"{self.temp_dir}/test_analytics.db"}
        self.analyzer = QualityAnalyzer(config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('app.core.analytics_engine.get_config')
    def test_config_integration(self, mock_get_config):
        """Test integration with config manager."""
        mock_config = {
            'analytics': {
                'storage_path': '/test/path',
                'quality_thresholds': {
                    'min_confidence': 0.8
                }
            }
        }
        mock_get_config.return_value = mock_config
        
        analyzer = QualityAnalyzer()
        
        assert analyzer.quality_thresholds['min_confidence'] == 0.8
    
    def test_ml_integration_compatibility(self):
        """Test compatibility with existing ML integration components."""
        from app.core.ml_integration import DetectionResult
        
        # Create a real DetectionResult-like object
        result = Mock(spec=DetectionResult)
        result.timestamp = datetime.now()
        result.category = "test_category"
        result.text = "test text"
        result.ml_confidence = 0.85
        result.processing_time_ms = 150.0
        result.fallback_used = False
        
        # Should not raise any exceptions
        self.analyzer.add_detection_result(result)
        metrics = self.analyzer.analyze_detection_quality()
        
        assert len(metrics) == 1
        assert metrics[0].category == "test_category"
    
    def test_performance_impact(self):
        """Test that analytics engine has minimal performance impact."""
        import time
        
        # Measure time to add many detection results
        start_time = time.time()
        
        for i in range(100):
            result = Mock()
            result.timestamp = datetime.now()
            result.category = f"category_{i % 5}"
            result.text = f"test_{i}"
            result.ml_confidence = 0.8
            result.processing_time_ms = 100.0
            result.fallback_used = False
            result.pattern_type = "test_pattern"
            
            self.analyzer.add_detection_result(result)
        
        add_time = time.time() - start_time
        
        # Measure time to analyze quality
        start_time = time.time()
        metrics = self.analyzer.analyze_detection_quality()
        analysis_time = time.time() - start_time
        
        # Should be reasonably fast (less than 1 second for 100 items)
        assert add_time < 1.0
        assert analysis_time < 2.0
        assert len(metrics) == 5  # 5 categories


if __name__ == '__main__':
    pytest.main([__file__]) 