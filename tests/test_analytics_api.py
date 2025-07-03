"""
Tests for Analytics API Endpoints - Priority 3 Implementation

This module tests the analytics API endpoints for dashboard integration,
including quality analysis, real-time monitoring, and trend analysis.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

from app.main import app
from app.core.analytics_engine import DetectionQualityMetrics, QualityIssue, QualityTrend, QualityIssueType
from app.core.real_time_monitor import Anomaly, AnomalyType
from app.core.ml_monitoring import MetricSnapshot
from app.core.ml_integration import MLIntegrationLayer


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_quality_analyzer():
    """Create a mock QualityAnalyzer for testing."""
    analyzer = Mock()
    
    # Mock quality metrics
    sample_metric = DetectionQualityMetrics(
        category="names",
        total_detections=100,
        avg_confidence=0.85,
        confidence_std=0.15,
        precision_estimate=0.92,
        recall_estimate=0.88,
        false_positive_rate=0.08,
        processing_time_ms=45.5,
        pattern_diversity=3,
        timestamp=datetime.now()
    )
    analyzer.analyze_detection_quality.return_value = [sample_metric]
    
    # Mock quality issues
    sample_issue = QualityIssue(
        issue_type=QualityIssueType.LOW_CONFIDENCE,
        severity="medium",
        description="Low confidence patterns detected",
        affected_patterns=["pattern1", "pattern2"],
        metrics={"avg_confidence": 0.65},
        suggested_actions=["Review patterns", "Gather more training data"],
        timestamp=datetime.now()
    )
    analyzer.detect_quality_issues.return_value = [sample_issue]
    
    # Mock improvement suggestions
    analyzer.get_improvement_suggestions.return_value = {
        "high_priority": ["Improve pattern X", "Add more training data"],
        "medium_priority": ["Review threshold Y"]
    }
    
    # Mock quality report
    analyzer.get_quality_report.return_value = {
        "summary": {
            "overall_score": 85.5,
            "total_categories": 5,
            "improvement_areas": 2
        }
    }
    
    # Mock quality trends
    sample_trend = QualityTrend(
        category="names",
        metric_name="avg_confidence",
        time_period_days=30,
        trend_direction="improving",
        change_percentage=12.5,
        start_value=0.75,
        end_value=0.85,
        num_data_points=15,
        data_points=[(datetime.now() - timedelta(days=i), 0.75 + i*0.01) for i in range(15)]
    )
    analyzer.get_quality_trends.return_value = sample_trend
    
    return analyzer


@pytest.fixture
def mock_real_time_monitor():
    """Create a mock RealTimeMonitor for testing."""
    monitor = Mock()
    
    # Mock dashboard data
    sample_anomaly = Anomaly(
        anomaly_type=AnomalyType.SPIKE,
        metric_name="processing_time",
        current_value=150.0,
        expected_value=100.0,
        deviation_score=2.5,
        severity="medium",
        description="Processing time exceeded threshold",
        timestamp=datetime.now(),
        context={"threshold": 120.0}
    )
    
    monitor.get_dashboard_data.return_value = {
        "timestamp": datetime.now().isoformat(),
        "recent_anomalies": [sample_anomaly],
        "metric_statistics": {
            "processing_time": {"avg": 95.5, "std": 12.3},
            "accuracy": {"avg": 0.92, "std": 0.05}
        }
    }
    
    # Mock monitoring status
    monitor.get_monitoring_status.return_value = {
        "is_active": True,
        "metrics_collected": 1500,
        "last_collection": datetime.now().isoformat()
    }
    
    # Mock performance aggregator
    monitor.performance_aggregator = Mock()
    monitor.performance_aggregator.get_aggregated_metrics.return_value = {
        "5min": {
            "processing_time": {"avg": 95.5, "min": 80.0, "max": 120.0},
            "accuracy": {"avg": 0.92, "min": 0.88, "max": 0.96}
        }
    }
    
    # Mock anomaly detector
    monitor.anomaly_detector = Mock()
    monitor.anomaly_detector.get_recent_anomalies.return_value = [sample_anomaly]
    monitor.anomaly_detector.add_alert_threshold.return_value = None
    monitor.anomaly_detector.remove_alert_threshold.return_value = None
    
    return monitor


@pytest.fixture
def mock_ml_performance_monitor():
    """Create a mock MLPerformanceMonitor for testing."""
    monitor = Mock()
    
    # Mock current metrics
    current_metrics = MetricSnapshot(
        accuracy=0.92,
        precision=0.89,
        recall=0.94,
        f1_score=0.915,
        processing_time_ms=85.5,
        confidence_correlation=0.87,
        sample_count=1000,
        timestamp=datetime.now()
    )
    monitor.get_current_metrics.return_value = current_metrics
    
    # Mock historical metrics
    historical_metrics = [
        MetricSnapshot(
            accuracy=0.90 + i*0.005,
            precision=0.87 + i*0.003,
            recall=0.92 + i*0.004,
            f1_score=0.895 + i*0.004,
            processing_time_ms=90.0 - i*0.5,
            confidence_correlation=0.85 + i*0.003,
            sample_count=1000 + i*10,
            timestamp=datetime.now() - timedelta(hours=24-i)
        )
        for i in range(24)
    ]
    monitor.get_historical_metrics.return_value = historical_metrics
    
    return monitor


@pytest.fixture
def mock_ml_integration_layer():
    """Create a mock MLIntegrationLayer for testing."""
    layer = Mock()
    
    # Mock performance summary
    layer.get_performance_summary.return_value = {
        "ml_usage_ratio": 0.75,
        "fallback_usage_ratio": 0.25,
        "total_predictions": 5000,
        "avg_processing_time": 85.5,
        "success_rate": 0.98
    }
    
    return layer


class TestAnalyticsEndpoints:
    """Test class for analytics API endpoints."""

    @patch('app.api.endpoints.analytics.get_quality_analyzer')
    def test_get_quality_overview_success(self, mock_get_analyzer, client, mock_quality_analyzer):
        """Test successful quality overview retrieval."""
        mock_get_analyzer.return_value = mock_quality_analyzer
        
        response = client.get("/api/v1/analytics/quality/overview")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "time_window_hours" in data
        assert "metrics" in data
        assert "issues" in data
        assert "suggestions" in data
        assert "overall_health" in data
        
        # Verify metrics structure
        assert len(data["metrics"]) > 0
        metric = data["metrics"][0]
        assert "category" in metric
        assert "avg_confidence" in metric

    @patch('app.api.endpoints.analytics.get_quality_analyzer')
    def test_get_quality_overview_with_filters(self, mock_get_analyzer, client, mock_quality_analyzer):
        """Test quality overview with query filters."""
        mock_get_analyzer.return_value = mock_quality_analyzer
        
        response = client.get("/api/v1/analytics/quality/overview?time_window_hours=48&category=names&document_type=invoice")
        assert response.status_code == 200
        
        data = response.json()
        assert data["time_window_hours"] == 48
        assert data["filters"]["category"] == "names"
        assert data["filters"]["document_type"] == "invoice"

    @patch('app.api.endpoints.analytics.get_quality_analyzer')
    def test_get_quality_trends_success(self, mock_get_analyzer, client, mock_quality_analyzer):
        """Test successful quality trends retrieval."""
        mock_get_analyzer.return_value = mock_quality_analyzer
        
        response = client.get("/api/v1/analytics/quality/trends/names/avg_confidence")
        assert response.status_code == 200
        
        data = response.json()
        assert data["category"] == "names"
        assert data["metric_name"] == "avg_confidence"
        assert "trend_direction" in data
        assert "data_points" in data
        assert "analysis" in data

    @patch('app.api.endpoints.analytics.get_quality_analyzer')
    def test_get_quality_trends_invalid_metric(self, mock_get_analyzer, client, mock_quality_analyzer):
        """Test quality trends with invalid metric name."""
        mock_get_analyzer.return_value = mock_quality_analyzer
        mock_quality_analyzer.get_quality_trends.return_value = None
        
        response = client.get("/api/v1/analytics/quality/trends/names/invalid_metric")
        assert response.status_code == 400
        assert "Invalid metric name" in response.json()["detail"]

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_get_realtime_metrics_success(self, mock_get_monitor, client, mock_real_time_monitor):
        """Test successful real-time metrics retrieval."""
        mock_get_monitor.return_value = mock_real_time_monitor
        
        response = client.get("/api/v1/analytics/monitoring/realtime")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "monitoring_status" in data
        assert "metrics" in data
        assert "recent_anomalies" in data
        assert "system_health" in data

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_get_anomaly_history_success(self, mock_get_monitor, client, mock_real_time_monitor):
        """Test successful anomaly history retrieval."""
        mock_get_monitor.return_value = mock_real_time_monitor
        
        response = client.get("/api/v1/analytics/monitoring/anomalies")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)

    @patch('app.api.endpoints.analytics.get_anomaly_history')
    def test_get_anomaly_history_with_filters(self, mock_get_history, client):
        """Test anomaly history with query filters."""
        mock_get_history.return_value = []
        
        response = client.get("/api/v1/analytics/monitoring/anomalies?hours_back=12&severity=high&metric_name=accuracy")
        assert response.status_code == 200

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_add_alert_threshold_success(self, mock_get_monitor, client, mock_real_time_monitor):
        """Test successful alert threshold addition."""
        mock_get_monitor.return_value = mock_real_time_monitor
        
        threshold_config = {
            "metric_name": "processing_time",
            "max_value": 150.0,
            "sample_size": 10
        }
        
        response = client.post("/api/v1/analytics/monitoring/alerts/threshold", json=threshold_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "processing_time" in data["message"]

    def test_add_alert_threshold_missing_field(self, client):
        """Test alert threshold addition with missing required field."""
        threshold_config = {
            "max_value": 150.0
            # Missing metric_name
        }
        
        response = client.post("/api/v1/analytics/monitoring/alerts/threshold", json=threshold_config)
        assert response.status_code == 400
        assert "Missing required field: metric_name" in response.json()["detail"]

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_remove_alert_threshold_success(self, mock_get_monitor, client, mock_real_time_monitor):
        """Test successful alert threshold removal."""
        mock_get_monitor.return_value = mock_real_time_monitor
        
        response = client.delete("/api/v1/analytics/monitoring/alerts/threshold/processing_time")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "processing_time" in data["message"]

    @patch('app.api.endpoints.analytics.get_ml_performance_monitor')
    @patch('app.api.endpoints.analytics.get_ml_integration_layer')
    def test_get_ml_performance_success(self, mock_get_layer, mock_get_monitor, client, 
                                       mock_ml_performance_monitor, mock_ml_integration_layer):
        """Test successful ML performance metrics retrieval."""
        mock_get_monitor.return_value = mock_ml_performance_monitor
        mock_get_layer.return_value = mock_ml_integration_layer
        
        response = client.get("/api/v1/analytics/ml/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "current_metrics" in data
        assert "historical_data" in data
        assert "performance_summary" in data
        assert "model_health" in data
        
        # Verify current metrics structure
        current = data["current_metrics"]
        assert "accuracy" in current
        assert "precision" in current
        assert "recall" in current

    @patch('app.api.endpoints.analytics.get_quality_overview')
    @patch('app.api.endpoints.analytics.get_realtime_metrics')
    @patch('app.api.endpoints.analytics.get_ml_performance')
    @patch('app.api.endpoints.analytics.get_anomaly_history')
    def test_get_dashboard_data_success(self, mock_anomaly, mock_ml, mock_realtime, mock_quality, client):
        """Test successful dashboard data retrieval."""
        # Mock all endpoint responses
        mock_quality.return_value = {"overall_health": {"score": 85}}
        mock_realtime.return_value = {"system_health": {"score": 90}}
        mock_ml.return_value = {"model_health": {"score": 88}}
        mock_anomaly.return_value = {"summary": {"by_severity": {"critical": 0}}}
        
        response = client.get("/api/v1/analytics/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert "quality_overview" in data
        assert "realtime_metrics" in data
        assert "ml_performance" in data
        assert "recent_anomalies" in data
        assert "system_status" in data
        
        # Verify system status calculation
        system_status = data["system_status"]
        assert "overall_score" in system_status
        assert "status" in system_status
        assert "component_scores" in system_status

    @patch('app.api.endpoints.analytics.get_quality_analyzer')
    def test_analytics_error_handling(self, mock_get_analyzer, client):
        """Test error handling in analytics endpoints."""
        # Simulate an exception in quality analyzer
        mock_get_analyzer.side_effect = Exception("Database connection failed")
        
        response = client.get("/api/v1/analytics/quality/overview")
        assert response.status_code == 500
        assert "Failed to generate quality overview" in response.json()["detail"]

    # Additional tests for better coverage

    @patch('app.api.endpoints.analytics.get_quality_analyzer')
    def test_get_quality_trends_with_parameters(self, mock_get_analyzer, client, mock_quality_analyzer):
        """Test quality trends with custom parameters."""
        mock_get_analyzer.return_value = mock_quality_analyzer
        
        response = client.get("/api/v1/analytics/quality/trends/emails/precision?days_back=60&window_days=2")
        assert response.status_code == 200
        
        # Verify the analyzer was called with correct parameters
        mock_quality_analyzer.get_quality_trends.assert_called_with(
            category="emails",
            metric_name="precision",
            days_back=60
        )

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_get_realtime_metrics_different_windows(self, mock_get_monitor, client, mock_real_time_monitor):
        """Test real-time metrics with different time windows."""
        mock_get_monitor.return_value = mock_real_time_monitor
        
        # Test different window parameters
        for window in ["1min", "1hour"]:
            response = client.get(f"/api/v1/analytics/monitoring/realtime?window={window}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["window"] == window

    @patch('app.api.endpoints.analytics.get_quality_analyzer')
    def test_quality_trends_error_handling(self, mock_get_analyzer, client):
        """Test error handling in quality trends endpoint."""
        mock_get_analyzer.side_effect = Exception("Trend calculation failed")
        
        response = client.get("/api/v1/analytics/quality/trends/names/confidence")
        assert response.status_code == 500
        assert "Failed to generate quality trends" in response.json()["detail"]

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_realtime_metrics_error_handling(self, mock_get_monitor, client):
        """Test error handling in real-time metrics endpoint."""
        mock_get_monitor.side_effect = Exception("Monitor unavailable")
        
        response = client.get("/api/v1/analytics/monitoring/realtime")
        assert response.status_code == 500
        assert "Failed to generate real-time metrics" in response.json()["detail"]

    def test_anomaly_history_current_implementation(self, client):
        """Test anomaly history endpoint current implementation (returns empty list)."""
        # The current implementation returns an empty list directly
        response = client.get("/api/v1/analytics/monitoring/anomalies")
        assert response.status_code == 200
        assert response.json() == []

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_add_alert_threshold_error_handling(self, mock_get_monitor, client):
        """Test error handling in alert threshold addition."""
        mock_get_monitor.side_effect = Exception("Threshold addition failed")
        
        threshold_config = {"metric_name": "accuracy", "min_value": 0.8}
        
        response = client.post("/api/v1/analytics/monitoring/alerts/threshold", json=threshold_config)
        assert response.status_code == 500
        assert "Failed to add alert threshold" in response.json()["detail"]

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_remove_alert_threshold_error_handling(self, mock_get_monitor, client):
        """Test error handling in alert threshold removal."""
        mock_get_monitor.side_effect = Exception("Threshold removal failed")
        
        response = client.delete("/api/v1/analytics/monitoring/alerts/threshold/accuracy")
        assert response.status_code == 500
        assert "Failed to remove alert threshold" in response.json()["detail"]

    @patch('app.api.endpoints.analytics.get_ml_performance_monitor')
    @patch('app.api.endpoints.analytics.get_ml_integration_layer')
    def test_ml_performance_error_handling(self, mock_get_layer, mock_get_monitor, client):
        """Test error handling in ML performance endpoint."""
        mock_get_monitor.side_effect = Exception("ML monitoring failed")
        
        response = client.get("/api/v1/analytics/ml/performance")
        assert response.status_code == 500
        assert "Failed to generate ML performance metrics" in response.json()["detail"]

    @patch('app.api.endpoints.analytics.get_quality_overview')
    def test_dashboard_data_error_handling(self, mock_quality_overview, client):
        """Test error handling in dashboard endpoint."""
        mock_quality_overview.side_effect = Exception("Dashboard generation failed")
        
        response = client.get("/api/v1/analytics/dashboard")
        assert response.status_code == 500
        assert "Failed to generate dashboard data" in response.json()["detail"]


class TestAnalyticsSingletonInitialization:
    """Test singleton initialization functions."""

    def test_get_quality_analyzer_initialization(self):
        """Test QualityAnalyzer singleton initialization."""
        from app.api.endpoints.analytics import get_quality_analyzer
        
        with patch('app.api.endpoints.analytics.QualityAnalyzer') as mock_analyzer_class:
            with patch('app.api.endpoints.analytics._quality_analyzer', None):
                # First call should create new instance
                analyzer = get_quality_analyzer()
                mock_analyzer_class.assert_called_once()

    def test_get_real_time_monitor_initialization(self):
        """Test RealTimeMonitor singleton initialization."""
        from app.api.endpoints.analytics import get_real_time_monitor
        
        with patch('app.api.endpoints.analytics.RealTimeMonitor') as mock_monitor_class:
            with patch('app.api.endpoints.analytics._real_time_monitor', None):
                # First call should create new instance
                monitor = get_real_time_monitor()
                mock_monitor_class.assert_called_once()

    def test_get_ml_performance_monitor_initialization(self):
        """Test MLPerformanceMonitor singleton initialization."""
        from app.api.endpoints.analytics import get_ml_performance_monitor
        
        with patch('app.api.endpoints.analytics.create_ml_performance_monitor') as mock_create:
            with patch('app.api.endpoints.analytics._ml_performance_monitor', None):
                # First call should create new instance
                monitor = get_ml_performance_monitor()
                mock_create.assert_called_once()

    def test_get_ml_integration_layer_initialization(self):
        """Test MLIntegrationLayer singleton initialization."""
        from app.api.endpoints.analytics import get_ml_integration_layer
        
        with patch('app.api.endpoints.analytics.create_ml_integration_layer') as mock_create:
            with patch('app.api.endpoints.analytics._ml_integration_layer', None):
                # First call should create new instance
                layer = get_ml_integration_layer()
                mock_create.assert_called_once()


class TestAnalyticsHelperFunctions:
    """Test class for analytics helper functions."""

    def test_calculate_overall_health_with_metrics(self):
        """Test overall health calculation with valid metrics."""
        from app.api.endpoints.analytics import _calculate_overall_health
        
        # Create sample metrics
        metrics = [
            Mock(avg_confidence=0.85, precision_estimate=0.90, recall_estimate=0.88),
            Mock(avg_confidence=0.82, precision_estimate=0.92, recall_estimate=0.86)
        ]
        
        # Create sample issues
        issues = [
            Mock(severity="medium"),
            Mock(severity="high")
        ]
        
        result = _calculate_overall_health(metrics, issues)
        
        assert "score" in result
        assert "status" in result
        assert "details" in result
        assert 0 <= result["score"] <= 100

    def test_calculate_overall_health_no_metrics(self):
        """Test overall health calculation with no metrics."""
        from app.api.endpoints.analytics import _calculate_overall_health
        
        result = _calculate_overall_health([], [])
        
        assert result["score"] == 0
        assert result["status"] == "insufficient_data"

    def test_calculate_overall_health_different_statuses(self):
        """Test overall health calculation for different status levels."""
        from app.api.endpoints.analytics import _calculate_overall_health
        
        # Test excellent status (score >= 85)
        excellent_metrics = [Mock(avg_confidence=0.95, precision_estimate=0.96, recall_estimate=0.94)]
        result = _calculate_overall_health(excellent_metrics, [])
        assert result["status"] == "excellent"
        
        # Test good status (70 <= score < 85)
        good_metrics = [Mock(avg_confidence=0.75, precision_estimate=0.78, recall_estimate=0.76)]
        result = _calculate_overall_health(good_metrics, [])
        assert result["status"] == "good"
        
        # Test fair status (50 <= score < 70)
        fair_metrics = [Mock(avg_confidence=0.60, precision_estimate=0.62, recall_estimate=0.58)]
        result = _calculate_overall_health(fair_metrics, [])
        assert result["status"] == "fair"
        
        # Test poor status (score < 50)
        poor_metrics = [Mock(avg_confidence=0.40, precision_estimate=0.42, recall_estimate=0.38)]
        result = _calculate_overall_health(poor_metrics, [])
        assert result["status"] == "poor"

    def test_calculate_overall_health_with_critical_issues(self):
        """Test overall health calculation with critical issues."""
        from app.api.endpoints.analytics import _calculate_overall_health
        
        metrics = [Mock(avg_confidence=0.95, precision_estimate=0.96, recall_estimate=0.94)]
        critical_issues = [
            Mock(severity="critical"),
            Mock(severity="critical"),
            Mock(severity="high")
        ]
        
        result = _calculate_overall_health(metrics, critical_issues)
        
        # Score should be reduced due to critical issues
        assert result["score"] < 95  # Should be lower than base score
        assert result["details"]["critical_issues"] == 2
        assert result["details"]["high_issues"] == 1

    def test_analyze_trend_improving(self):
        """Test trend analysis for improving trend."""
        from app.api.endpoints.analytics import _analyze_trend
        
        trend = Mock(
            trend_direction="improving",
            change_percentage=15.5,
            num_data_points=20
        )
        
        result = _analyze_trend(trend)
        
        assert "improving" in result["interpretation"]
        assert "15.5%" in result["interpretation"]
        assert result["confidence"] == "high"
        assert "Continue current practices" in result["recommendation"]

    def test_analyze_trend_declining(self):
        """Test trend analysis for declining trend."""
        from app.api.endpoints.analytics import _analyze_trend
        
        trend = Mock(
            trend_direction="declining",
            change_percentage=-12.3,
            num_data_points=8
        )
        
        result = _analyze_trend(trend)
        
        assert "declining" in result["interpretation"]
        assert "12.3%" in result["interpretation"]
        assert result["confidence"] == "medium"
        assert "Investigate causes" in result["recommendation"]

    def test_analyze_trend_stable(self):
        """Test trend analysis for stable trend."""
        from app.api.endpoints.analytics import _analyze_trend
        
        trend = Mock(
            trend_direction="stable",
            change_percentage=2.1,
            num_data_points=12
        )
        
        result = _analyze_trend(trend)
        
        assert "stable" in result["interpretation"]
        assert "minimal variation" in result["interpretation"]
        assert result["confidence"] == "medium"
        assert "Monitor for any changes" in result["recommendation"]

    def test_analyze_trend_insufficient_data(self):
        """Test trend analysis for insufficient data."""
        from app.api.endpoints.analytics import _analyze_trend
        
        trend = Mock(
            trend_direction="insufficient_data",
            change_percentage=0,
            num_data_points=2
        )
        
        result = _analyze_trend(trend)
        
        assert "Not enough historical data" in result["interpretation"]
        assert result["confidence"] == "low"
        assert "Continue monitoring" in result["recommendation"]

    def test_calculate_system_health_healthy(self):
        """Test system health calculation for healthy system."""
        from app.api.endpoints.analytics import _calculate_system_health
        
        metrics = {"processing_time": {"avg": 95.5}}
        anomalies = []
        
        result = _calculate_system_health(metrics, anomalies)
        
        assert result["status"] == "healthy"
        assert result["score"] == 100
        assert result["critical_anomalies"] == 0

    def test_calculate_system_health_critical(self):
        """Test system health calculation for critical system."""
        from app.api.endpoints.analytics import _calculate_system_health
        
        metrics = {"processing_time": {"avg": 95.5}}
        anomalies = [
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "medium"}
        ]
        
        result = _calculate_system_health(metrics, anomalies)
        
        assert result["status"] == "critical"
        assert result["score"] < 100
        assert result["critical_anomalies"] == 1

    def test_calculate_system_health_warning(self):
        """Test system health calculation for warning status."""
        from app.api.endpoints.analytics import _calculate_system_health
        
        metrics = {"processing_time": {"avg": 95.5}}
        anomalies = [
            {"severity": "high"},
            {"severity": "high"}
        ]
        
        result = _calculate_system_health(metrics, anomalies)
        
        assert result["status"] == "warning"
        assert result["score"] == 80  # 100 - 2*10
        assert result["critical_anomalies"] == 0
        assert result["high_anomalies"] == 2

    def test_calculate_system_health_no_metrics(self):
        """Test system health calculation with no metrics."""
        from app.api.endpoints.analytics import _calculate_system_health
        
        result = _calculate_system_health({}, [])
        
        assert result["status"] == "unknown"
        assert result["score"] == 0

    def test_calculate_system_health_with_anomaly_objects(self):
        """Test system health calculation with Anomaly objects."""
        from app.api.endpoints.analytics import _calculate_system_health
        
        metrics = {"processing_time": {"avg": 95.5}}
        anomaly_objects = [
            Mock(severity="critical"),
            Mock(severity="high")
        ]
        
        result = _calculate_system_health(metrics, anomaly_objects)
        
        assert result["status"] == "critical"
        assert result["critical_anomalies"] == 1
        assert result["high_anomalies"] == 1

    def test_analyze_anomalies_with_data(self):
        """Test anomaly analysis with data."""
        from app.api.endpoints.analytics import _analyze_anomalies
        
        anomalies = [
            Mock(severity="high", anomaly_type=Mock(value="threshold_exceeded"), metric_name="accuracy"),
            Mock(severity="medium", anomaly_type=Mock(value="outlier_detected"), metric_name="processing_time"),
            Mock(severity="high", anomaly_type=Mock(value="threshold_exceeded"), metric_name="accuracy")
        ]
        
        result = _analyze_anomalies(anomalies)
        
        assert result["total"] == 3
        assert result["by_severity"]["high"] == 2
        assert result["by_severity"]["medium"] == 1
        assert result["by_type"]["threshold_exceeded"] == 2
        assert len(result["most_affected_metrics"]) == 2

    def test_analyze_anomalies_empty(self):
        """Test anomaly analysis with no data."""
        from app.api.endpoints.analytics import _analyze_anomalies
        
        result = _analyze_anomalies([])
        
        assert result["total"] == 0
        assert result["by_severity"] == {}
        assert result["by_type"] == {}
        assert result["most_affected_metrics"] == []

    def test_calculate_ml_health_excellent(self):
        """Test ML health calculation for excellent performance."""
        from app.api.endpoints.analytics import _calculate_ml_health
        
        metrics = Mock(
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            processing_time_ms=50.0
        )
        
        performance_summary = {"ml_usage_ratio": 0.85}
        
        result = _calculate_ml_health(metrics, performance_summary)
        
        assert result["status"] == "excellent"
        assert result["score"] >= 85
        assert "factors" in result

    def test_calculate_ml_health_different_statuses(self):
        """Test ML health calculation for different status levels."""
        from app.api.endpoints.analytics import _calculate_ml_health
        
        # Test good status
        good_metrics = Mock(accuracy=0.80, precision=0.78, recall=0.82, processing_time_ms=80.0)
        result = _calculate_ml_health(good_metrics, {"ml_usage_ratio": 0.70})
        assert result["status"] == "good"
        
        # Test fair status
        fair_metrics = Mock(accuracy=0.65, precision=0.63, recall=0.67, processing_time_ms=120.0)
        result = _calculate_ml_health(fair_metrics, {"ml_usage_ratio": 0.50})
        assert result["status"] == "fair"
        
        # Test poor status
        poor_metrics = Mock(accuracy=0.40, precision=0.38, recall=0.42, processing_time_ms=200.0)
        result = _calculate_ml_health(poor_metrics, {"ml_usage_ratio": 0.30})
        assert result["status"] == "poor"

    def test_calculate_overall_system_status_healthy(self):
        """Test overall system status calculation for healthy system."""
        from app.api.endpoints.analytics import _calculate_overall_system_status
        
        quality_overview = {"overall_health": {"score": 90}}
        realtime_metrics = {"system_health": {"score": 88}}
        ml_performance = {"model_health": {"score": 92}}
        recent_anomalies = {"summary": {"by_severity": {"critical": 0}}}
        
        result = _calculate_overall_system_status(
            quality_overview, realtime_metrics, ml_performance, recent_anomalies
        )
        
        assert result["status"] == "healthy"
        assert result["overall_score"] >= 85
        assert result["critical_issues"] == 0
        
        components = result["component_scores"]
        assert components["quality"] == 90
        assert components["system"] == 88
        assert components["ml_model"] == 92

    def test_calculate_overall_system_status_warning(self):
        """Test overall system status calculation for warning status."""
        from app.api.endpoints.analytics import _calculate_overall_system_status
        
        quality_overview = {"overall_health": {"score": 75}}
        realtime_metrics = {"system_health": {"score": 72}}
        ml_performance = {"model_health": {"score": 78}}
        recent_anomalies = {"summary": {"by_severity": {"critical": 0}}}
        
        result = _calculate_overall_system_status(
            quality_overview, realtime_metrics, ml_performance, recent_anomalies
        )
        
        assert result["status"] == "warning"
        assert 70 <= result["overall_score"] < 85

    def test_calculate_overall_system_status_degraded(self):
        """Test overall system status calculation for degraded status."""
        from app.api.endpoints.analytics import _calculate_overall_system_status
        
        quality_overview = {"overall_health": {"score": 60}}
        realtime_metrics = {"system_health": {"score": 55}}
        ml_performance = {"model_health": {"score": 65}}
        recent_anomalies = {"summary": {"by_severity": {"critical": 0}}}
        
        result = _calculate_overall_system_status(
            quality_overview, realtime_metrics, ml_performance, recent_anomalies
        )
        
        assert result["status"] == "degraded"
        assert result["overall_score"] < 70

    def test_calculate_overall_system_status_critical_with_anomalies(self):
        """Test overall system status calculation with critical anomalies."""
        from app.api.endpoints.analytics import _calculate_overall_system_status
        
        quality_overview = {"overall_health": {"score": 90}}
        realtime_metrics = {"system_health": {"score": 88}}
        ml_performance = {"model_health": {"score": 92}}
        recent_anomalies = {"summary": {"by_severity": {"critical": 2}}}
        
        result = _calculate_overall_system_status(
            quality_overview, realtime_metrics, ml_performance, recent_anomalies
        )
        
        assert result["status"] == "critical"
        assert result["critical_issues"] == 2

    def test_calculate_overall_system_status_missing_data(self):
        """Test overall system status calculation with missing data."""
        from app.api.endpoints.analytics import _calculate_overall_system_status
        
        # Test with missing health scores
        quality_overview = {}
        realtime_metrics = {}
        ml_performance = {}
        recent_anomalies = {}
        
        result = _calculate_overall_system_status(
            quality_overview, realtime_metrics, ml_performance, recent_anomalies
        )
        
        assert "overall_score" in result
        assert "status" in result
        assert result["component_scores"]["quality"] == 0
        assert result["component_scores"]["system"] == 0
        assert result["component_scores"]["ml_model"] == 0


if __name__ == "__main__":
    pytest.main([__file__])