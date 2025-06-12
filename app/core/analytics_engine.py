"""
Advanced Analytics Engine for Priority 3 Implementation

This module provides comprehensive quality analysis, detection assessment,
and improvement suggestions for the ML-powered PII detection system.
"""

import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json
import sqlite3
from pathlib import Path
from enum import Enum

# Import existing components
from app.core.ml_monitoring import MetricSnapshot, MLPerformanceMonitor
from app.core.ml_integration import DetectionResult
from app.core.adaptive.ab_testing import ABTestResult
from app.core.config_manager import get_config
from app.core.logging import get_logger

logger = get_logger(__name__)

analytics_logger = get_logger("anonympdf.analytics")


class QualityIssueType(Enum):
    """Types of quality issues that can be detected."""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_FALSE_POSITIVE = "high_false_positive"
    INCONSISTENT_DETECTION = "inconsistent_detection"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PATTERN_ACCURACY_DROP = "pattern_accuracy_drop"


@dataclass
class QualityIssue:
    """Represents a detected quality issue."""
    issue_type: QualityIssueType
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_patterns: List[str]
    metrics: Dict[str, float]
    suggested_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DetectionQualityMetrics:
    """Comprehensive quality metrics for detection analysis."""
    category: str
    total_detections: int
    avg_confidence: float
    confidence_std: float
    precision_estimate: float
    recall_estimate: float
    false_positive_rate: float
    processing_time_ms: float
    pattern_diversity: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityTrend:
    """Quality trend analysis over time."""
    metric_name: str
    category: str
    time_period_days: int
    trend_direction: str
    change_percentage: float
    start_value: Optional[float] = None
    end_value: Optional[float] = None
    num_data_points: int = 0
    data_points: List[Tuple[datetime, float]] = field(default_factory=list)


class QualityAnalyzer:
    """Comprehensive quality analysis for ML detection system."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initializes the QualityAnalyzer.

        Args:
            storage_path: Optional path for the analytics database. Uses config if None.
        """
        self.config = get_config().get('analytics', {}) # ADD THIS LINE
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # THIS LINE BELOW IS WRONG in your current file, but will be correct after the fix
            self.storage_path = Path(self.config.get('storage_path', 'data/analytics.db'))

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_db()
        logger.info(f"QualityAnalyzer initialized with database at {self.storage_path}")
        
        # This line is what crashes because self.config doesn't exist
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'min_confidence': 0.7,
            'max_false_positive_rate': 0.05,
            'min_precision': 0.9,
            'min_recall': 0.85,
            'max_processing_time_ms': 2000
        })
        
    
    
        # Detection tracking
        self.detection_history = deque(maxlen=10000)
        self.category_metrics = defaultdict(list)
        self.pattern_performance = defaultdict(list)
        
        # Quality issues tracking
        self.quality_issues = []
        self.issue_history = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance logging
        self.performance_metrics: Dict[str, MetricSnapshot] = {}
        self.metrics_lock = threading.Lock()
        self._init_performance_monitor()
    
    def _init_db(self):
        """Initialize storage for analytics data."""
        with sqlite3.connect(self.storage_path) as conn:
            # Quality metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    category TEXT NOT NULL,
                    total_detections INTEGER,
                    avg_confidence REAL,
                    confidence_std REAL,
                    precision_estimate REAL,
                    recall_estimate REAL,
                    false_positive_rate REAL,
                    processing_time_ms REAL,
                    pattern_diversity INTEGER
                )
            ''')
            
            # Quality issues table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    issue_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    affected_patterns TEXT,
                    metrics TEXT,
                    suggested_actions TEXT
                )
            ''')
            
            # Table for A/B test results
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    test_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    winner TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    summary TEXT,
                    metrics_comparison_json TEXT
                )
            ''')

            # Create indices
            conn.execute('CREATE INDEX IF NOT EXISTS idx_quality_timestamp ON quality_metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_issues_timestamp ON quality_issues(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_ab_test_timestamp ON ab_test_results(timestamp)')
    
    def add_detection_result(self, result: DetectionResult, ground_truth: Optional[bool] = None):
        """Add detection result for quality analysis."""
        with self._lock:
            detection_data = {
                'timestamp': result.timestamp,
                'category': result.category,
                'text': result.text,
                'ml_confidence': result.ml_confidence,
                'processing_time_ms': result.processing_time_ms,
                'fallback_used': result.fallback_used,
                'ground_truth': ground_truth,
                'pattern_type': getattr(result, 'pattern_type', 'unknown')
            }
            
            self.detection_history.append(detection_data)
            self.category_metrics[result.category].append(detection_data)
            
            if hasattr(result, 'pattern_type'):
                self.pattern_performance[result.pattern_type].append(detection_data)
    
    def log_ab_test_result(self, test_result: ABTestResult):
        """Logs the result of a completed A/B test to the analytics database."""
        with self._lock:
            analytics_logger.info(f"Logging A/B test result for test ID: {test_result.test_id}")
            try:
                with sqlite3.connect(self.storage_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO ab_test_results 
                        (test_id, timestamp, winner, confidence, summary, metrics_comparison_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            test_result.test_id,
                            datetime.now(),
                            test_result.winner,
                            test_result.confidence,
                            test_result.summary,
                            json.dumps(test_result.metrics_comparison)
                        ),
                    )
                analytics_logger.info(f"Successfully logged result for A/B test {test_result.test_id}.")
            except Exception as e:
                analytics_logger.error(f"Failed to log A/B test result for {test_result.test_id}: {e}", exc_info=True)
    
    def analyze_detection_quality(self, category: Optional[str] = None, 
                                 time_window_hours: int = 24,
                                 document_type_filter: Optional[str] = None) -> List[DetectionQualityMetrics]:
        """Analyze detection quality for categories, optionally filtered by document type."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        with self._lock:
            if category:
                categories_to_analyze = [category] if category in self.category_metrics else []
            else:
                categories_to_analyze = list(self.category_metrics.keys())
        
        quality_metrics_list = []
        
        for cat in categories_to_analyze:
            # Get recent detections for this category
            recent_detections_for_category = [
                d for d in self.category_metrics[cat]
                if d['timestamp'] >= cutoff_time
            ]

            # Apply document_type_filter if provided
            if document_type_filter:
                final_recent_detections = [
                    d for d in recent_detections_for_category
                    if d.get('document_type') == document_type_filter
                ]
            else:
                final_recent_detections = recent_detections_for_category
            
            if not final_recent_detections:
                continue
            
            # Calculate quality metrics
            confidences = [d['ml_confidence'] for d in final_recent_detections]
            processing_times = [d['processing_time_ms'] for d in final_recent_detections]
            
            # Pattern diversity
            patterns = set(d.get('pattern_type', 'unknown') for d in final_recent_detections)
            
            # Estimate precision/recall from confidence and ground truth
            with_ground_truth = [d for d in final_recent_detections if d.get('ground_truth') is not None]
            
            precision = 0.0
            recall = 0.0
            false_positive_rate_val = 0.0

            if with_ground_truth:
                # Calculate actual precision/recall
                true_positives = sum(1 for d in with_ground_truth 
                                   if d['ground_truth'] and d['ml_confidence'] > 0.5)
                false_positives = sum(1 for d in with_ground_truth 
                                    if not d['ground_truth'] and d['ml_confidence'] > 0.5)
                false_negatives = sum(1 for d in with_ground_truth 
                                    if d['ground_truth'] and d['ml_confidence'] <= 0.5)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                # Calculate false_positive_rate based on all samples with ground truth
                total_gt_samples = len(with_ground_truth)
                false_positive_rate_val = false_positives / total_gt_samples if total_gt_samples > 0 else 0.0

            elif final_recent_detections: # Ensure final_recent_detections is not empty for proxy metrics
                # Estimate from confidence distribution
                high_conf_ratio = sum(1 for c in confidences if c > 0.8) / len(confidences) if confidences else 0.0
                precision = recall = high_conf_ratio
                false_positive_rate_val = 1.0 - high_conf_ratio
            
            current_metrics = DetectionQualityMetrics(
                category=cat,
                total_detections=len(final_recent_detections),
                avg_confidence=statistics.mean(confidences) if confidences else 0.0,
                confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                precision_estimate=precision,
                recall_estimate=recall,
                false_positive_rate=false_positive_rate_val,
                processing_time_ms=statistics.mean(processing_times) if processing_times else 0.0,
                pattern_diversity=len(patterns)
            )
            
            quality_metrics_list.append(current_metrics)
            
            # Store in database - Note: current DB schema does not store document_type for metrics.
            # If filtering by document_type, storing these filtered metrics might need schema change or careful interpretation.
            # For now, we store the overall category metric or the filtered one if a filter is applied.
            self._store_quality_metrics(current_metrics)
        
        return quality_metrics_list
    
    def _store_quality_metrics(self, metrics_data: DetectionQualityMetrics):
        """Store quality metrics to database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                INSERT INTO quality_metrics 
                (timestamp, category, total_detections, avg_confidence, confidence_std,
                 precision_estimate, recall_estimate, false_positive_rate, 
                 processing_time_ms, pattern_diversity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics_data.timestamp.isoformat(),
                metrics_data.category,
                metrics_data.total_detections,
                metrics_data.avg_confidence,
                metrics_data.confidence_std,
                metrics_data.precision_estimate,
                metrics_data.recall_estimate,
                metrics_data.false_positive_rate,
                metrics_data.processing_time_ms,
                metrics_data.pattern_diversity
            ))
    
    def get_quality_trends(
        self, 
        category: str, 
        metric_name: str, 
        days_back: int = 30,
        min_data_points_for_trend: int = 3
    ) -> Optional[QualityTrend]:
        """Calculate quality trend for a specific metric and category over a period."""
        if metric_name not in DetectionQualityMetrics.__annotations__:
            analytics_logger.warning(f"Invalid metric_name '{metric_name}' for trend analysis.")
            return None

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        query = f"""
            SELECT date(timestamp) as day, AVG({metric_name}) as avg_metric
            FROM quality_metrics
            WHERE category = ? AND timestamp >= ? AND timestamp <= ?
            GROUP BY day
            ORDER BY day ASC
        """
        
        daily_avg_metrics: List[Tuple[datetime, float]] = []
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(query, (category, start_date.isoformat(), end_date.isoformat()))
                for row in cursor.fetchall():
                    day_str, avg_val = row
                    if day_str and avg_val is not None:
                        daily_avg_metrics.append((datetime.strptime(day_str, '%Y-%m-%d'), float(avg_val)))
        except sqlite3.Error as e:
            analytics_logger.error(f"Database error during trend calculation for {category} - {metric_name}: {e}")
            return None

        if len(daily_avg_metrics) < min_data_points_for_trend:
            analytics_logger.info(f"Not enough data points ({len(daily_avg_metrics)}) for trend analysis for {category} - {metric_name}.")
            return QualityTrend(
                metric_name=metric_name,
                category=category,
                time_period_days=days_back,
                trend_direction='insufficient_data',
                change_percentage=0.0,
                num_data_points=len(daily_avg_metrics),
                data_points=daily_avg_metrics
            )

        first_val = daily_avg_metrics[0][1]
        last_val = daily_avg_metrics[-1][1]

        change_percentage = 0.0
        if first_val != 0: # Avoid division by zero
            change_percentage = ((last_val - first_val) / abs(first_val)) * 100
        elif last_val != 0: # If first_val is 0 and last_val is not, it's an infinite percentage change
            change_percentage = float('inf') if last_val > 0 else float('-inf')

        trend_direction = 'stable'
        # Define stability threshold, e.g., +/- 5% change
        stability_threshold = self.config.get('trend_stability_threshold_percent', 5.0)
        
        # Higher is better for: avg_confidence, precision_estimate, recall_estimate, pattern_diversity
        # Lower is better for: false_positive_rate, processing_time_ms, confidence_std
        higher_is_better_metrics = ['avg_confidence', 'precision_estimate', 'recall_estimate', 'pattern_diversity']
        
        is_higher_better = metric_name in higher_is_better_metrics

        if change_percentage > stability_threshold:
            trend_direction = 'improving' if is_higher_better else 'declining'
        elif change_percentage < -stability_threshold:
            trend_direction = 'declining' if is_higher_better else 'improving'
        
        return QualityTrend(
            metric_name=metric_name,
            category=category,
            time_period_days=days_back,
            trend_direction=trend_direction,
            change_percentage=round(change_percentage, 2),
            start_value=round(first_val, 4),
            end_value=round(last_val, 4),
            num_data_points=len(daily_avg_metrics),
            data_points=daily_avg_metrics
        )
    
    def detect_quality_issues(self, time_window_hours: int = 24) -> List[QualityIssue]:
        """Detect quality issues based on current metrics."""
        quality_metrics = self.analyze_detection_quality(time_window_hours=time_window_hours)
        issues = []
        
        for metrics in quality_metrics:
            # Check for low confidence
            if metrics.avg_confidence < self.quality_thresholds['min_confidence']:
                issue = QualityIssue(
                    issue_type=QualityIssueType.LOW_CONFIDENCE,
                    severity='medium' if metrics.avg_confidence > 0.5 else 'high',
                    description=f"Low average confidence ({metrics.avg_confidence:.2f}) for category {metrics.category}",
                    affected_patterns=[metrics.category],
                    metrics={'avg_confidence': metrics.avg_confidence},
                    suggested_actions=[
                        f"Review training data for {metrics.category}",
                        "Consider model retraining with more examples",
                        "Adjust confidence thresholds"
                    ]
                )
                issues.append(issue)
            
            # Check for high false positive rate
            if metrics.false_positive_rate > self.quality_thresholds['max_false_positive_rate']:
                issue = QualityIssue(
                    issue_type=QualityIssueType.HIGH_FALSE_POSITIVE,
                    severity='high' if metrics.false_positive_rate > 0.1 else 'medium',
                    description=f"High false positive rate ({metrics.false_positive_rate:.2f}) for category {metrics.category}",
                    affected_patterns=[metrics.category],
                    metrics={'false_positive_rate': metrics.false_positive_rate},
                    suggested_actions=[
                        f"Review {metrics.category} detection patterns",
                        "Improve pattern specificity",
                        "Add negative examples to training data"
                    ]
                )
                issues.append(issue)
            
            # Check for performance degradation
            if metrics.processing_time_ms > self.quality_thresholds['max_processing_time_ms']:
                issue = QualityIssue(
                    issue_type=QualityIssueType.PERFORMANCE_DEGRADATION,
                    severity='medium',
                    description=f"Slow processing time ({metrics.processing_time_ms:.0f}ms) for category {metrics.category}",
                    affected_patterns=[metrics.category],
                    metrics={'processing_time_ms': metrics.processing_time_ms},
                    suggested_actions=[
                        f"Optimize {metrics.category} detection algorithms",
                        "Review pattern complexity",
                        "Consider caching strategies"
                    ]
                )
                issues.append(issue)
        
        # Store issues
        for issue in issues:
            self._store_quality_issue(issue)
        
        with self._lock:
            self.quality_issues.extend(issues)
            self.issue_history.extend(issues)
        
        return issues
    
    def _store_quality_issue(self, issue: QualityIssue):
        """Store quality issue to database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute('''
                INSERT INTO quality_issues 
                (timestamp, issue_type, severity, description, affected_patterns, metrics, suggested_actions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                issue.timestamp.isoformat(),
                issue.issue_type.value,
                issue.severity,
                issue.description,
                json.dumps(issue.affected_patterns),
                json.dumps(issue.metrics),
                json.dumps(issue.suggested_actions)
            ))
    
    def get_quality_report(self, include_trends: bool = True) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        # Get current quality metrics
        current_metrics = self.analyze_detection_quality(time_window_hours=24)
        
        # Get recent issues
        recent_issues = [issue for issue in self.quality_issues 
                        if issue.timestamp >= datetime.now() - timedelta(hours=24)]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_categories_analyzed': len(current_metrics),
                'total_quality_issues': len(recent_issues),
                'avg_confidence': statistics.mean([m.avg_confidence for m in current_metrics]) if current_metrics else 0.0,
                'avg_processing_time_ms': statistics.mean([m.processing_time_ms for m in current_metrics]) if current_metrics else 0.0
            },
            'category_metrics': [
                {
                    'category': m.category,
                    'total_detections': m.total_detections,
                    'avg_confidence': m.avg_confidence,
                    'precision_estimate': m.precision_estimate,
                    'recall_estimate': m.recall_estimate,
                    'processing_time_ms': m.processing_time_ms
                }
                for m in current_metrics
            ],
            'quality_issues': [
                {
                    'type': issue.issue_type.value,
                    'severity': issue.severity,
                    'description': issue.description,
                    'suggested_actions': issue.suggested_actions
                }
                for issue in recent_issues
            ]
        }
        
        return report
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        # Analyze recent issues for patterns
        recent_issues = [issue for issue in self.quality_issues 
                        if issue.timestamp >= datetime.now() - timedelta(hours=48)]
        
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in recent_issues:
            issues_by_type[issue.issue_type].append(issue)
        
        # Generate suggestions based on issue patterns
        for issue_type, issues in issues_by_type.items():
            if len(issues) >= 2:  # Multiple similar issues
                affected_categories = set()
                for issue in issues:
                    affected_categories.update(issue.affected_patterns)
                
                suggestion = {
                    'priority': 'high' if len(issues) > 3 else 'medium',
                    'category': 'systematic_issue',
                    'description': f"Multiple {issue_type.value} issues detected across categories",
                    'affected_categories': list(affected_categories),
                    'recommended_actions': [
                        f"Systematic review of {issue_type.value} patterns",
                        "Consider retraining models with updated data",
                        "Review threshold configurations"
                    ],
                    'impact_assessment': f"Affects {len(affected_categories)} categories"
                }
                suggestions.append(suggestion)
        
        return suggestions

    def _init_performance_monitor(self):
        """Initializes the performance monitor component."""
        self.performance_monitor = MLPerformanceMonitor(config=self.config)
        logger.info("MLPerformanceMonitor initialized within QualityAnalyzer.")

    def _get_connection(self):
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(self.storage_path)
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database: {e}")
                self._conn = None
        return self._conn

    def close(self):
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            logger.info(f"Database connection to {self.storage_path} closed.")


class QualityInsightsGenerator:
    """Generate insights and recommendations from quality analysis."""
    
    def __init__(self, quality_analyzer: QualityAnalyzer):
        self.analyzer = quality_analyzer
        
    def generate_pattern_insights(self) -> Dict[str, Any]:
        """Generate insights about pattern performance."""
        insights = {
            'best_performing_patterns': [],
            'underperforming_patterns': [],
            'optimization_opportunities': []
        }
        
        # Analyze pattern performance
        for pattern_type, detections in self.analyzer.pattern_performance.items():
            if not detections:
                continue
                
            recent_detections = [
                d for d in detections
                if d['timestamp'] >= datetime.now() - timedelta(hours=24)
            ]
            
            if len(recent_detections) < 5:  # Skip patterns with low volume
                continue
            
            avg_confidence = statistics.mean([d['ml_confidence'] for d in recent_detections])
            avg_time = statistics.mean([d['processing_time_ms'] for d in recent_detections])
            
            pattern_info = {
                'pattern_type': pattern_type,
                'detection_count': len(recent_detections),
                'avg_confidence': avg_confidence,
                'avg_processing_time_ms': avg_time
            }
            
            # Categorize patterns
            if avg_confidence > 0.85 and avg_time < 500:
                insights['best_performing_patterns'].append(pattern_info)
            elif avg_confidence < 0.6 or avg_time > 2000:
                insights['underperforming_patterns'].append(pattern_info)
        
        return insights
    
    def generate_model_insights(self) -> Dict[str, Any]:
        """Generate insights about model performance."""
        quality_metrics = self.analyzer.analyze_detection_quality(time_window_hours=24)
        
        if not quality_metrics:
            return {'status': 'insufficient_data'}
        
        # Calculate overall model health
        avg_precision = statistics.mean([m.precision_estimate for m in quality_metrics])
        avg_recall = statistics.mean([m.recall_estimate for m in quality_metrics])
        avg_confidence = statistics.mean([m.avg_confidence for m in quality_metrics])
        
        model_health = 'excellent' if avg_precision > 0.9 and avg_recall > 0.85 else \
                      'good' if avg_precision > 0.8 and avg_recall > 0.75 else \
                      'needs_improvement'
        
        return {
            'model_health': model_health,
            'overall_precision': avg_precision,
            'overall_recall': avg_recall,
            'overall_confidence': avg_confidence,
            'categories_analyzed': len(quality_metrics),
            'recommendations': self._generate_model_recommendations(model_health, quality_metrics)
        }
    
    def _generate_model_recommendations(self, health: str, metrics: List[DetectionQualityMetrics]) -> List[str]:
        """Generate model improvement recommendations."""
        recommendations = []
        
        if health == 'needs_improvement':
            low_precision_categories = [m.category for m in metrics if m.precision_estimate < 0.7]
            if low_precision_categories:
                recommendations.append(f"Focus on improving precision for: {', '.join(low_precision_categories)}")
            
            low_recall_categories = [m.category for m in metrics if m.recall_estimate < 0.7]
            if low_recall_categories:
                recommendations.append(f"Add more training examples for: {', '.join(low_recall_categories)}")
        
        elif health == 'good':
            recommendations.append("Model performing well, consider fine-tuning for edge cases")
        
        else:  # excellent
            recommendations.append("Model performing excellently, focus on maintaining performance")
        
        return recommendations 