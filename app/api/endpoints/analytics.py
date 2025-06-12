"""
Analytics API Endpoints for Priority 3 Implementation

This module provides comprehensive analytics endpoints for dashboard integration,
including quality analysis, real-time monitoring, and trend analysis.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import asyncio

from app.database import get_db
from app.core.analytics_engine import QualityAnalyzer, QualityTrend
from app.core.real_time_monitor import RealTimeMonitor, AnomalyDetector, PerformanceAggregator
from app.core.ml_monitoring import MLPerformanceMonitor, create_ml_performance_monitor, AlertThreshold
from app.core.ml_integration import MLIntegrationLayer, create_ml_integration_layer
from app.core.logging import api_logger

router = APIRouter()

# Initialize analytics components - these would typically be singletons or dependency injected
_quality_analyzer = None
_real_time_monitor = None
_ml_performance_monitor = None
_ml_integration_layer = None

def get_quality_analyzer() -> QualityAnalyzer:
    """Get or create QualityAnalyzer instance."""
    global _quality_analyzer
    if _quality_analyzer is None:
        _quality_analyzer = QualityAnalyzer()
        api_logger.info("QualityAnalyzer initialized for analytics API")
    return _quality_analyzer

def get_real_time_monitor() -> RealTimeMonitor:
    """Get or create RealTimeMonitor instance."""
    global _real_time_monitor
    if _real_time_monitor is None:
        _real_time_monitor = RealTimeMonitor()
        # Integrate with other components
        _real_time_monitor.set_quality_analyzer(get_quality_analyzer())
        _real_time_monitor.set_ml_monitor(get_ml_performance_monitor())
        api_logger.info("RealTimeMonitor initialized for analytics API")
    return _real_time_monitor

def get_ml_performance_monitor() -> MLPerformanceMonitor:
    """Get or create MLPerformanceMonitor instance."""
    global _ml_performance_monitor
    if _ml_performance_monitor is None:
        _ml_performance_monitor = create_ml_performance_monitor()
        api_logger.info("MLPerformanceMonitor initialized for analytics API")
    return _ml_performance_monitor

def get_ml_integration_layer() -> MLIntegrationLayer:
    """Get or create MLIntegrationLayer instance."""
    global _ml_integration_layer
    if _ml_integration_layer is None:
        _ml_integration_layer = create_ml_integration_layer()
        api_logger.info("MLIntegrationLayer initialized for analytics API")
    return _ml_integration_layer


@router.get("/quality/overview")
async def get_quality_overview(
    time_window_hours: int = Query(24, description="Time window in hours for analysis"),
    category: Optional[str] = Query(None, description="Specific category to analyze"),
    document_type: Optional[str] = Query(None, description="Filter by document type")
) -> Dict[str, Any]:
    """Get comprehensive quality analysis overview."""
    api_logger.info("Quality overview requested", time_window_hours=time_window_hours, category=category)
    
    try:
        analyzer = get_quality_analyzer()
        
        # Get quality metrics
        quality_metrics = analyzer.analyze_detection_quality(
            category=category,
            time_window_hours=time_window_hours,
            document_type_filter=document_type
        )
        
        # Get quality issues
        quality_issues = analyzer.detect_quality_issues(time_window_hours=time_window_hours)
        
        # Get improvement suggestions
        suggestions = analyzer.get_improvement_suggestions()
        
        # Get quality report
        quality_report = analyzer.get_quality_report(include_trends=True)
        
        # Format response
        response = {
            "timestamp": datetime.now().isoformat(),
            "time_window_hours": time_window_hours,
            "filters": {
                "category": category,
                "document_type": document_type
            },
            "metrics": [
                {
                    "category": m.category,
                    "total_detections": m.total_detections,
                    "avg_confidence": round(m.avg_confidence, 3),
                    "confidence_std": round(m.confidence_std, 3),
                    "precision_estimate": round(m.precision_estimate, 3),
                    "recall_estimate": round(m.recall_estimate, 3),
                    "false_positive_rate": round(m.false_positive_rate, 3),
                    "processing_time_ms": round(m.processing_time_ms, 2),
                    "pattern_diversity": m.pattern_diversity,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in quality_metrics
            ],
            "issues": [
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity,
                    "description": issue.description,
                    "affected_patterns": issue.affected_patterns,
                    "metrics": issue.metrics,
                    "suggested_actions": issue.suggested_actions,
                    "timestamp": issue.timestamp.isoformat()
                }
                for issue in quality_issues
            ],
            "suggestions": suggestions,
            "summary": quality_report.get("summary", {}),
            "overall_health": _calculate_overall_health(quality_metrics, quality_issues)
        }
        
        api_logger.info("Quality overview generated successfully", 
                       metrics_count=len(quality_metrics), 
                       issues_count=len(quality_issues))
        
        return response
        
    except Exception as e:
        api_logger.error(f"Error generating quality overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate quality overview: {str(e)}")


@router.get("/quality/trends/{category}/{metric_name}")
async def get_quality_trends(
    category: str,
    metric_name: str,
    days_back: int = Query(30, description="Number of days to analyze"),
    window_days: int = Query(1, description="Aggregation window in days")
) -> Dict[str, Any]:
    """Get quality trends for specific category and metric."""
    api_logger.info("Quality trends requested", category=category, metric_name=metric_name, days_back=days_back)
    
    try:
        analyzer = get_quality_analyzer()
        
        # Get quality trend
        trend = analyzer.get_quality_trends(
            category=category,
            metric_name=metric_name,
            days_back=days_back
        )
        
        if trend is None:
            raise HTTPException(status_code=400, detail=f"Invalid metric name: {metric_name}")
        
        response = {
            "category": trend.category,
            "metric_name": trend.metric_name,
            "time_period_days": trend.time_period_days,
            "trend_direction": trend.trend_direction,
            "change_percentage": trend.change_percentage,
            "start_value": trend.start_value,
            "end_value": trend.end_value,
            "num_data_points": trend.num_data_points,
            "data_points": [
                {
                    "timestamp": point[0].isoformat(),
                    "value": round(point[1], 4)
                }
                for point in trend.data_points
            ],
            "analysis": _analyze_trend(trend)
        }
        
        api_logger.info("Quality trends generated successfully", 
                       category=category, 
                       metric_name=metric_name,
                       data_points=len(trend.data_points))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error generating quality trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate quality trends: {str(e)}")


@router.get("/monitoring/realtime")
async def get_realtime_metrics(
    window: str = Query("5min", description="Time window: 1min, 5min, 1hour")
) -> Dict[str, Any]:
    """Get real-time monitoring metrics."""
    api_logger.info("Real-time metrics requested", window=window)
    
    try:
        monitor = get_real_time_monitor()
        
        # Get dashboard data
        dashboard_data = monitor.get_dashboard_data()
        
        # Get monitoring status
        status = monitor.get_monitoring_status()
        
        # Get aggregated metrics for specific window
        aggregated_metrics = monitor.performance_aggregator.get_aggregated_metrics(window)
        
        response = {
            "timestamp": dashboard_data["timestamp"],
            "monitoring_status": status,
            "window": window,
            "metrics": aggregated_metrics,
            "recent_anomalies": dashboard_data["recent_anomalies"],
            "metric_statistics": dashboard_data["metric_statistics"],
            "system_health": _calculate_system_health(aggregated_metrics, dashboard_data["recent_anomalies"])
        }
        
        api_logger.info("Real-time metrics generated successfully", window=window)
        
        return response
        
    except Exception as e:
        api_logger.error(f"Error generating real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate real-time metrics: {str(e)}")


@router.get("/monitoring/anomalies")
async def get_anomaly_history(
    hours_back: int = Query(24, description="Hours to look back"),
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical"),
    metric_name: Optional[str] = Query(None, description="Filter by metric name")
) -> Dict[str, Any]:
    """Get anomaly detection history."""
    api_logger.info("Anomaly history requested", hours_back=hours_back, severity=severity)
    
    try:
        monitor = get_real_time_monitor()
        
        # Get recent anomalies
        anomalies = monitor.anomaly_detector.get_recent_anomalies(hours_back=hours_back)
        
        # Filter by severity if specified
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        
        # Filter by metric name if specified
        if metric_name:
            anomalies = [a for a in anomalies if a.metric_name == metric_name]
        
        # Group anomalies by type and severity for analysis
        anomaly_summary = _analyze_anomalies(anomalies)
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "filters": {
                "hours_back": hours_back,
                "severity": severity,
                "metric_name": metric_name
            },
            "total_anomalies": len(anomalies),
            "anomalies": [
                {
                    "type": anomaly.anomaly_type.value,
                    "metric_name": anomaly.metric_name,
                    "current_value": round(anomaly.current_value, 4),
                    "expected_value": round(anomaly.expected_value, 4),
                    "deviation_score": round(anomaly.deviation_score, 2),
                    "severity": anomaly.severity,
                    "description": anomaly.description,
                    "timestamp": anomaly.timestamp.isoformat(),
                    "context": anomaly.context
                }
                for anomaly in anomalies
            ],
            "summary": anomaly_summary
        }
        
        api_logger.info("Anomaly history generated successfully", 
                       total_anomalies=len(anomalies))
        
        return response
        
    except Exception as e:
        api_logger.error(f"Error generating anomaly history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate anomaly history: {str(e)}")


@router.post("/monitoring/alerts/threshold")
async def add_alert_threshold(threshold_config: Dict[str, Any]) -> Dict[str, str]:
    """Add a new alert threshold for monitoring."""
    api_logger.info("Adding alert threshold", config=threshold_config)
    
    try:
        # Validate required fields
        required_fields = ["metric_name"]
        for field in required_fields:
            if field not in threshold_config:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create AlertThreshold object
        threshold = AlertThreshold(
            metric_name=threshold_config["metric_name"],
            min_value=threshold_config.get("min_value"),
            max_value=threshold_config.get("max_value"),
            change_threshold=threshold_config.get("change_threshold"),
            sample_size=threshold_config.get("sample_size", 10)
        )
        
        # Add to anomaly detector
        monitor = get_real_time_monitor()
        monitor.anomaly_detector.add_alert_threshold(threshold)
        
        api_logger.info("Alert threshold added successfully", metric=threshold.metric_name)
        
        return {"status": "success", "message": f"Alert threshold added for {threshold.metric_name}"}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error adding alert threshold: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add alert threshold: {str(e)}")


@router.delete("/monitoring/alerts/threshold/{metric_name}")
async def remove_alert_threshold(metric_name: str) -> Dict[str, str]:
    """Remove alert threshold for a specific metric."""
    api_logger.info("Removing alert threshold", metric_name=metric_name)
    
    try:
        monitor = get_real_time_monitor()
        monitor.anomaly_detector.remove_alert_threshold(metric_name)
        
        api_logger.info("Alert threshold removed successfully", metric_name=metric_name)
        
        return {"status": "success", "message": f"Alert threshold removed for {metric_name}"}
        
    except Exception as e:
        api_logger.error(f"Error removing alert threshold: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove alert threshold: {str(e)}")


@router.get("/ml/performance")
async def get_ml_performance() -> Dict[str, Any]:
    """Get ML model performance metrics."""
    api_logger.info("ML performance metrics requested")
    
    try:
        ml_monitor = get_ml_performance_monitor()
        ml_integration = get_ml_integration_layer()
        
        # Get current metrics
        current_metrics = ml_monitor.get_current_metrics()
        
        # Get historical metrics
        historical_metrics = ml_monitor.get_historical_metrics(hours_back=24)
        
        # Get performance summary from integration layer
        performance_summary = ml_integration.get_performance_summary()
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "accuracy": round(current_metrics.accuracy, 4),
                "precision": round(current_metrics.precision, 4),
                "recall": round(current_metrics.recall, 4),
                "f1_score": round(current_metrics.f1_score, 4),
                "processing_time_ms": round(current_metrics.processing_time_ms, 2),
                "confidence_correlation": round(current_metrics.confidence_correlation, 4),
                "sample_count": current_metrics.sample_count,
                "timestamp": current_metrics.timestamp.isoformat()
            },
            "historical_data": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "accuracy": round(m.accuracy, 4),
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "processing_time_ms": round(m.processing_time_ms, 2)
                }
                for m in historical_metrics[-24:]  # Last 24 data points
            ],
            "performance_summary": performance_summary,
            "model_health": _calculate_ml_health(current_metrics, performance_summary)
        }
        
        api_logger.info("ML performance metrics generated successfully")
        
        return response
        
    except Exception as e:
        api_logger.error(f"Error generating ML performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate ML performance metrics: {str(e)}")


@router.get("/dashboard")
async def get_dashboard_data() -> Dict[str, Any]:
    """Get comprehensive dashboard data combining all analytics."""
    api_logger.info("Dashboard data requested")
    
    try:
        # Get data from all components
        quality_overview = await get_quality_overview(time_window_hours=24)
        realtime_metrics = await get_realtime_metrics(window="5min")
        ml_performance = await get_ml_performance()
        recent_anomalies = await get_anomaly_history(hours_back=6)
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "quality_overview": quality_overview,
            "realtime_metrics": realtime_metrics,
            "ml_performance": ml_performance,
            "recent_anomalies": recent_anomalies,
            "system_status": _calculate_overall_system_status(
                quality_overview, realtime_metrics, ml_performance, recent_anomalies
            )
        }
        
        api_logger.info("Dashboard data generated successfully")
        
        return response
        
    except Exception as e:
        api_logger.error(f"Error generating dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard data: {str(e)}")


# Helper functions for analysis and health calculations

def _calculate_overall_health(quality_metrics, quality_issues) -> Dict[str, Any]:
    """Calculate overall quality health score."""
    if not quality_metrics:
        return {"score": 0, "status": "insufficient_data", "details": "No quality metrics available"}
    
    # Calculate average scores
    avg_confidence = sum(m.avg_confidence for m in quality_metrics) / len(quality_metrics)
    avg_precision = sum(m.precision_estimate for m in quality_metrics) / len(quality_metrics)
    avg_recall = sum(m.recall_estimate for m in quality_metrics) / len(quality_metrics)
    
    # Weight different aspects
    confidence_score = min(avg_confidence * 100, 100)
    precision_score = avg_precision * 100
    recall_score = avg_recall * 100
    
    # Penalty for issues
    issue_penalty = len([i for i in quality_issues if i.severity in ["high", "critical"]]) * 5
    
    overall_score = max(0, (confidence_score + precision_score + recall_score) / 3 - issue_penalty)
    
    if overall_score >= 85:
        status = "excellent"
    elif overall_score >= 70:
        status = "good"
    elif overall_score >= 50:
        status = "fair"
    else:
        status = "poor"
    
    return {
        "score": round(overall_score, 1),
        "status": status,
        "details": {
            "avg_confidence": round(avg_confidence, 3),
            "avg_precision": round(avg_precision, 3),
            "avg_recall": round(avg_recall, 3),
            "critical_issues": len([i for i in quality_issues if i.severity == "critical"]),
            "high_issues": len([i for i in quality_issues if i.severity == "high"])
        }
    }

def _analyze_trend(trend: QualityTrend) -> Dict[str, Any]:
    """Analyze trend and provide insights."""
    analysis = {
        "interpretation": "",
        "recommendation": "",
        "confidence": "medium"
    }
    
    if trend.trend_direction == "insufficient_data":
        analysis["interpretation"] = "Not enough historical data to determine trend"
        analysis["recommendation"] = "Continue monitoring to gather more data"
        analysis["confidence"] = "low"
    elif trend.trend_direction == "improving":
        analysis["interpretation"] = f"Metric is improving by {abs(trend.change_percentage):.1f}%"
        analysis["recommendation"] = "Continue current practices"
        analysis["confidence"] = "high" if trend.num_data_points > 10 else "medium"
    elif trend.trend_direction == "declining":
        analysis["interpretation"] = f"Metric is declining by {abs(trend.change_percentage):.1f}%"
        analysis["recommendation"] = "Investigate causes and implement improvements"
        analysis["confidence"] = "high" if trend.num_data_points > 10 else "medium"
    else:  # stable
        analysis["interpretation"] = "Metric is stable with minimal variation"
        analysis["recommendation"] = "Monitor for any changes"
        analysis["confidence"] = "medium"
    
    return analysis

def _calculate_system_health(aggregated_metrics, recent_anomalies) -> Dict[str, Any]:
    """Calculate overall system health from real-time metrics."""
    if not aggregated_metrics:
        return {"status": "unknown", "score": 0}
    
    # Handle both Anomaly objects and dictionaries
    critical_anomalies = 0
    high_anomalies = 0
    
    for anomaly in recent_anomalies:
        if hasattr(anomaly, 'severity'):
            # Anomaly object
            severity = anomaly.severity
        else:
            # Dictionary
            severity = anomaly.get("severity")
        
        if severity == "critical":
            critical_anomalies += 1
        elif severity == "high":
            high_anomalies += 1
    
    # Base score
    score = 100
    
    # Penalties
    score -= critical_anomalies * 20
    score -= high_anomalies * 10
    
    score = max(0, score)
    
    # Status determination - critical if any critical anomalies exist
    if critical_anomalies > 0:
        status = "critical"
    elif score >= 90:
        status = "healthy"
    elif score >= 70:
        status = "warning"
    else:
        status = "critical"
    
    return {
        "status": status,
        "score": score,
        "critical_anomalies": critical_anomalies,
        "high_anomalies": high_anomalies
    }

def _analyze_anomalies(anomalies) -> Dict[str, Any]:
    """Analyze anomaly patterns."""
    if not anomalies:
        return {"total": 0, "by_severity": {}, "by_type": {}, "most_affected_metrics": []}
    
    # Group by severity
    by_severity = {}
    for anomaly in anomalies:
        severity = anomaly.severity
        by_severity[severity] = by_severity.get(severity, 0) + 1
    
    # Group by type
    by_type = {}
    for anomaly in anomalies:
        anomaly_type = anomaly.anomaly_type.value
        by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1
    
    # Most affected metrics
    metric_counts = {}
    for anomaly in anomalies:
        metric = anomaly.metric_name
        metric_counts[metric] = metric_counts.get(metric, 0) + 1
    
    most_affected = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "total": len(anomalies),
        "by_severity": by_severity,
        "by_type": by_type,
        "most_affected_metrics": [{"metric": m[0], "count": m[1]} for m in most_affected]
    }

def _calculate_ml_health(current_metrics, performance_summary) -> Dict[str, Any]:
    """Calculate ML model health score."""
    # Use multiple factors to determine health
    accuracy_score = current_metrics.accuracy * 100
    precision_score = current_metrics.precision * 100
    recall_score = current_metrics.recall * 100
    
    # Processing time factor (lower is better)
    time_factor = max(0, 100 - (current_metrics.processing_time_ms / 10))
    
    # ML usage ratio factor
    ml_ratio = performance_summary.get("ml_usage_ratio", 0) * 100
    
    overall_score = (accuracy_score + precision_score + recall_score + time_factor + ml_ratio) / 5
    
    if overall_score >= 85:
        status = "excellent"
    elif overall_score >= 70:
        status = "good"
    elif overall_score >= 50:
        status = "fair"
    else:
        status = "poor"
    
    return {
        "score": round(overall_score, 1),
        "status": status,
        "factors": {
            "accuracy": round(accuracy_score, 1),
            "precision": round(precision_score, 1),
            "recall": round(recall_score, 1),
            "processing_efficiency": round(time_factor, 1),
            "ml_usage_ratio": round(ml_ratio, 1)
        }
    }

def _calculate_overall_system_status(quality_overview, realtime_metrics, ml_performance, recent_anomalies) -> Dict[str, Any]:
    """Calculate overall system status combining all factors."""
    quality_health = quality_overview.get("overall_health", {})
    system_health = realtime_metrics.get("system_health", {})
    ml_health = ml_performance.get("model_health", {})
    
    quality_score = quality_health.get("score", 0)
    system_score = system_health.get("score", 0)
    ml_score = ml_health.get("score", 0)
    
    # Weight the scores
    overall_score = (quality_score * 0.4 + system_score * 0.3 + ml_score * 0.3)
    
    # Check for critical issues
    critical_anomalies = recent_anomalies.get("summary", {}).get("by_severity", {}).get("critical", 0)
    
    if critical_anomalies > 0:
        status = "critical"
    elif overall_score >= 85:
        status = "healthy"
    elif overall_score >= 70:
        status = "warning"
    else:
        status = "degraded"
    
    return {
        "overall_score": round(overall_score, 1),
        "status": status,
        "component_scores": {
            "quality": quality_score,
            "system": system_score,
            "ml_model": ml_score
        },
        "critical_issues": critical_anomalies
    }