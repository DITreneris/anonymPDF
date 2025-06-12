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
        
        # Test basic request
        response = client.get("/api/v1/analytics/quality/overview")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "metrics" in data
        assert "issues" in data
        assert "suggestions" in data
        assert "overall_health" in data
        
        # Verify metrics structure
        assert len(data["metrics"]) == 1
        metric = data["metrics"][0]
        assert metric["category"] == "names"
        assert metric["total_detections"] == 100
        assert "avg_confidence" in metric
        
        # Verify issues structure
        assert len(data["issues"]) == 1
        issue = data["issues"][0]
        assert issue["type"] == "low_confidence"
        assert issue["severity"] == "medium"
        
        # Verify overall health
        health = data["overall_health"]
        assert "score" in health
        assert "status" in health

    @patch('app.api.endpoints.analytics.get_quality_analyzer')
    def test_get_quality_overview_with_filters(self, mock_get_analyzer, client, mock_quality_analyzer):
        """Test quality overview with filters."""
        mock_get_analyzer.return_value = mock_quality_analyzer
        
        response = client.get("/api/v1/analytics/quality/overview?time_window_hours=12&category=names&document_type=legal")
        assert response.status_code == 200
        
        data = response.json()
        assert data["time_window_hours"] == 12
        assert data["filters"]["category"] == "names"
        assert data["filters"]["document_type"] == "legal"
        
        # Verify analyzer was called with correct parameters
        mock_quality_analyzer.analyze_detection_quality.assert_called_with(
            category="names",
            time_window_hours=12,
            document_type_filter="legal"
        )

    @patch('app.api.endpoints.analytics.get_quality_analyzer')
    def test_get_quality_trends_success(self, mock_get_analyzer, client, mock_quality_analyzer):
        """Test successful quality trends retrieval."""
        mock_get_analyzer.return_value = mock_quality_analyzer
        
        response = client.get("/api/v1/analytics/quality/trends/names/avg_confidence")
        assert response.status_code == 200
        
        data = response.json()
        assert data["category"] == "names"
        assert data["metric_name"] == "avg_confidence"
        assert data["trend_direction"] == "improving"
        assert data["change_percentage"] == 12.5
        assert "data_points" in data
        assert "analysis" in data
        
        # Verify data points structure
        assert len(data["data_points"]) == 15
        point = data["data_points"][0]
        assert "timestamp" in point
        assert "value" in point

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
        
        response = client.get("/api/v1/analytics/monitoring/realtime?window=5min")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "monitoring_status" in data
        assert data["window"] == "5min"
        assert "metrics" in data
        assert "recent_anomalies" in data
        assert "system_health" in data
        
        # Verify monitoring status
        status = data["monitoring_status"]
        assert status["is_active"] is True
        assert "metrics_collected" in status

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_get_anomaly_history_success(self, mock_get_monitor, client, mock_real_time_monitor):
        """Test successful anomaly history retrieval."""
        mock_get_monitor.return_value = mock_real_time_monitor
        
        response = client.get("/api/v1/analytics/monitoring/anomalies?hours_back=24")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_anomalies" in data
        assert "anomalies" in data
        assert "summary" in data
        assert data["filters"]["hours_back"] == 24
        
        # Verify anomaly structure
        assert len(data["anomalies"]) == 1
        anomaly = data["anomalies"][0]
        assert anomaly["type"] == "spike"
        assert anomaly["metric_name"] == "processing_time"
        assert anomaly["severity"] == "medium"

    @patch('app.api.endpoints.analytics.get_anomaly_history')
    def test_get_anomaly_history_with_filters(self, mock_get_history, client):
        """Test anomaly history with severity and metric filters."""
        # Mock the response
        mock_response = {
            "total_anomalies": 1,
            "anomalies": [],
            "summary": {"total": 1, "by_severity": {"high": 1}},
            "filters": {"hours_back": 12, "severity": "high", "metric_name": "accuracy"}
        }
        mock_get_history.return_value = mock_response
        
        response = client.get("/api/v1/analytics/monitoring/anomalies?hours_back=12&severity=high&metric_name=accuracy")
        assert response.status_code == 200

    @patch('app.api.endpoints.analytics.get_real_time_monitor')
    def test_add_alert_threshold_success(self, mock_get_monitor, client, mock_real_time_monitor):
        """Test successful alert threshold addition."""
        mock_get_monitor.return_value = mock_real_time_monitor
        
        threshold_config = {
            "metric_name": "processing_time",
            "max_value": 150.0,
            "min_value": 50.0,
            "sample_size": 10
        }
        
        response = client.post("/api/v1/analytics/monitoring/alerts/threshold", json=threshold_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "processing_time" in data["message"]
        
        # Verify threshold was added
        mock_real_time_monitor.anomaly_detector.add_alert_threshold.assert_called_once()

    def test_add_alert_threshold_missing_field(self, client):
        """Test alert threshold addition with missing required field."""
        threshold_config = {
            "max_value": 150.0,
            "min_value": 50.0
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
        
        # Verify threshold was removed
        mock_real_time_monitor.anomaly_detector.remove_alert_threshold.assert_called_once_with("processing_time")

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
        assert current["accuracy"] == 0.92
        assert current["precision"] == 0.89
        assert current["recall"] == 0.94
        assert "processing_time_ms" in current
        
        # Verify historical data
        assert len(data["historical_data"]) == 24
        
        # Verify model health
        health = data["model_health"]
        assert "score" in health
        assert "status" in health
        assert "factors" in health

    @patch('app.api.endpoints.analytics.get_quality_overview')
    @patch('app.api.endpoints.analytics.get_realtime_metrics')
    @patch('app.api.endpoints.analytics.get_ml_performance')
    @patch('app.api.endpoints.analytics.get_anomaly_history')
    def test_get_dashboard_data_success(self, mock_anomaly, mock_ml, mock_realtime, mock_quality, client):
        """Test successful dashboard data retrieval combining all analytics."""
        # Mock all the component responses
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


if __name__ == "__main__":
    pytest.main([__file__])