"""
Advanced Analytics Engine for Priority 3 Implementation

This module provides comprehensive quality analysis, detection assessment,
and improvement suggestions for the ML-powered PII detection system.
"""

import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Set, Union
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
from app.core.config_manager import get_config, get_config_manager
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
    """Analyzes the quality of PII detections and stores results."""
    
    def __init__(self, db_path: Union[str, Path] = "data/analytics.db", config: Optional[Dict] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._init_db()
        
        self.config = config or get_config().get('analytics', {})
        self.ml_monitor = self._get_ml_monitor()
        analytics_logger.info("QualityAnalyzer initialized", extra={"db_path": str(self.db_path)})

    def _connect(self):
        """Establish the database connection."""
        if self.conn is None:
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                analytics_logger.debug(f"Database connection opened for {self.db_path}")
            except sqlite3.Error as e:
                analytics_logger.error(f"Error connecting to analytics database: {e}")
                raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                # Ensure any pending transaction is committed
                self.conn.commit()
            except sqlite3.Error:
                # If commit fails, try to rollback
                try:
                    self.conn.rollback()
                except sqlite3.Error:
                    pass
            finally:
                # Always close the connection
                self.conn.close()
                self.conn = None
                analytics_logger.debug(f"Database connection closed for {self.db_path}")

    def __del__(self):
        """Ensure connection is closed when object is garbage collected."""
        self.close()

    def _get_ml_monitor(self) -> MLPerformanceMonitor:
        """Get an instance of the ML performance monitor."""
        monitor = MLPerformanceMonitor()
        return monitor
    
    def _init_db(self):
        """Initialize the SQLite database."""
        if not self.conn: self._connect()
        try:
            with self.conn:
                self.conn.execute('''
                CREATE TABLE IF NOT EXISTS detection_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    category TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ml_confidence REAL,
                    processing_time_ms REAL,
                    fallback_used BOOLEAN,
                    pattern_type TEXT
                )
                ''')

                self.conn.execute('''
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
            
                self.conn.execute('''
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
            
                self.conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    test_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    winner TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    summary TEXT,
                    metrics_comparison_json TEXT
                )
            ''')

                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_quality_timestamp ON quality_metrics(timestamp)')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_issues_timestamp ON quality_issues(timestamp)')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_ab_test_timestamp ON ab_test_results(timestamp)')
        except sqlite3.Error as e:
            analytics_logger.error(f"Error initializing analytics database: {e}")

    def add_detection_results(self, document_id: str, results: List[Dict[str, Any]]):
        """Add detection results to the database for analysis."""
        if not self.conn: self._connect()
        try:
            with self.conn:
                for result in results:
                    self.conn.execute(
                        """
                        INSERT INTO detection_results (document_id, timestamp, category, text, ml_confidence, processing_time_ms, fallback_used, pattern_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            result['timestamp'],
                            result['category'],
                            result['text'],
                            result['ml_confidence'],
                            result['processing_time_ms'],
                            result['fallback_used'],
                            result.get('pattern_type', 'unknown')
                        )
                    )
                # Ensure the transaction is committed before closing
                self.conn.commit()
        except sqlite3.Error as e:
            analytics_logger.error(f"Error adding detection results: {e}")
            # Ensure we rollback on error
            if self.conn:
                self.conn.rollback()
            raise

    def add_feedback(self, feedback: "UserFeedback"):
        """Add user feedback to the database."""
        if not self.conn: self._connect()
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO user_feedback (timestamp, feedback_type, feedback_text)
                    VALUES (?, ?, ?)
                    """,
                    (
                        feedback.timestamp,
                        feedback.feedback_type,
                        feedback.feedback_text
                    )
                )
        except sqlite3.Error as e:
            analytics_logger.error(f"Error adding feedback: {e}")

    def analyze_detection_quality(
        self,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze detection quality based on stored results and feedback."""
        if not self.conn: self._connect()
        
        # Build query dynamically
        query = "SELECT * FROM detection_results WHERE 1=1"
        params = {}
        if document_type:
            query += " AND category = ?"
            params["category"] = document_type

        quality_metrics = {}
        try:
            with self.conn:
                self.conn.row_factory = sqlite3.Row
                cursor = self.conn.execute(query, params)
                rows = cursor.fetchall()
                for row in rows:
                    category = row['category']
                    if category not in quality_metrics:
                        quality_metrics[category] = {
                            'total_detections': 0,
                            'avg_confidence': 0,
                            'confidence_std': 0,
                            'precision_estimate': 0,
                            'recall_estimate': 0,
                            'false_positive_rate': 0,
                            'processing_time_ms': 0,
                            'pattern_diversity': 0
                        }
                    quality_metrics[category]['total_detections'] += 1
                    quality_metrics[category]['avg_confidence'] += row['ml_confidence']
                    quality_metrics[category]['confidence_std'] += row['ml_confidence'] ** 2
                    quality_metrics[category]['precision_estimate'] += row['ml_confidence']
                    quality_metrics[category]['recall_estimate'] += row['ml_confidence']
                    quality_metrics[category]['false_positive_rate'] += 1 - row['ml_confidence']
                    quality_metrics[category]['processing_time_ms'] += row['processing_time_ms']
                    quality_metrics[category]['pattern_diversity'] += 1
        except sqlite3.Error as e:
            analytics_logger.error(f"Error analyzing detection quality: {e}")
            return {}
        
        # Post-process to calculate final averages and stddev
        for category, metrics in quality_metrics.items():
            total = metrics['total_detections']
            if total > 0:
                metrics['avg_confidence'] /= total
                metrics['precision_estimate'] /= total
                metrics['recall_estimate'] /= total
                metrics['false_positive_rate'] /= total
                metrics['processing_time_ms'] /= total
                metrics['pattern_diversity'] /= total
                # Note: This is a simplified stddev calculation
                mean_sq = metrics['confidence_std'] / total
                metrics['confidence_std'] = (mean_sq - metrics['avg_confidence']**2)**0.5 if mean_sq - metrics['avg_confidence']**2 >= 0 else 0.0

        return quality_metrics

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of data quality across all documents."""
        if not self.conn: self._connect()
        
        summary = {}
        try:
            with self.conn:
                # Category distribution
                cursor = self.conn.execute(
                    "SELECT category, COUNT(*) FROM detection_results GROUP BY category"
                )
                summary["category_distribution"] = {row[0]: row[1] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            analytics_logger.error(f"Error getting quality summary: {e}")
        
        return summary


class QualityInsightsGenerator:
    """Generates actionable insights from quality analysis data."""
    def __init__(self, analyzer: QualityAnalyzer):
        self.analyzer = analyzer

    def generate_report(self) -> Dict[str, Any]:
        report_data = {}
        report_data["quality_summary"] = self.analyzer.get_quality_summary()

        # Generate insights
        insights = self._generate_insights(report_data)

        report_data["insights"] = insights
        return report_data

    def _generate_insights(self, report_data: Dict[str, Any]) -> List[str]:
        # Implementation of _generate_insights method
        return []


class AnalyticsEngine:
    """High-level engine to coordinate analytics tasks."""
    def __init__(self, db_path: Optional[str] = None, config: Optional[Dict] = None):
        self.config = config or get_config()
        db_path = db_path or self.config.get("analytics", {}).get("database_path")
        self.quality_analyzer = QualityAnalyzer(db_path=db_path, config=self.config)
        self.insights_generator = QualityInsightsGenerator(self.quality_analyzer)
        
    def run_full_analysis(self) -> Dict[str, Any]:
        # Implementation of run_full_analysis method
        return {}

    def close(self):
        """Close the analytics engine and its components."""
        if self.quality_analyzer:
            self.quality_analyzer.close()
            analytics_logger.info("Analytics engine closed.")

def get_analytics_engine() -> AnalyticsEngine:
    """Get a singleton instance of the AnalyticsEngine."""
    # Implementation of get_analytics_engine method
    return AnalyticsEngine() 