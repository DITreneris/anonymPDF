"""
A/B Testing Framework for Adaptive Learning

This module provides a framework for running A/B tests to compare the
performance of different models, patterns, or configurations.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import random
import json
from pathlib import Path
import sqlite3
from collections import defaultdict
from scipy.stats import ttest_ind
import statistics
import threading

from app.core.logging import get_logger
from app.core.config_manager import ConfigManager, get_config_manager

ab_test_logger = get_logger("ab_testing")

@dataclass
class ABTest:
    """Represents a single A/B test configuration."""
    test_id: str = field(default_factory=lambda: f"ab_{uuid.uuid4().hex[:8]}")
    name: str = "Unnamed Test"
    description: str = ""
    control_model_version: str = "current"
    variant_model_version: str = "candidate"
    traffic_split_ratio: float = 0.5  # 50% to variant, 50% to control
    is_active: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_db_tuple(self):
        """Serializes the object to a tuple for database insertion."""
        return (
            self.test_id,
            self.name,
            self.description,
            self.control_model_version,
            self.variant_model_version,
            self.traffic_split_ratio,
            self.is_active,
            self.start_time.isoformat() if self.start_time else None,
            self.end_time.isoformat() if self.end_time else None,
        )

    @classmethod
    def from_db_row(cls, row: Tuple) -> "ABTest":
        """Deserializes a database row into an ABTest object."""
        return cls(
            test_id=row[0],
            name=row[1],
            description=row[2],
            control_model_version=row[3],
            variant_model_version=row[4],
            traffic_split_ratio=row[5],
            is_active=bool(row[6]),
            start_time=datetime.fromisoformat(row[7]) if row[7] else None,
            end_time=datetime.fromisoformat(row[8]) if row[8] else None,
        )
    
@dataclass
class ABTestResult:
    """Stores the result and conclusion of an A/B test."""
    test_id: str
    winner: str # 'control', 'variant', or 'inconclusive'
    confidence: float
    summary: str
    metrics_comparison: Dict[str, Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
class ABTestManager:
    """
    Manages the lifecycle of A/B tests.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initializes the A/B Test Manager.

        Args:
            db_path: Optional path to the database. Uses config if not provided.
        """
        if db_path:
            self.db_path = Path(db_path)  # Ensure db_path is a Path object
        else:
            config = get_config_manager().settings.get('adaptive_learning', {})
            db_config = config.get('databases', {})
            self.db_path = Path(db_config.get('ab_tests_db', 'data/adaptive/ab_tests.db'))

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))  # Convert Path to string for sqlite3
        self.conn.row_factory = sqlite3.Row
        self.tests: Dict[str, ABTest] = {}
        self._create_schema()
        self._load_tests()
        ab_test_logger.info(f"ABTestManager initialized with database at {self.db_path}.")

    def _create_schema(self):
        """Creates the necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        # Main table for test configurations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                control_model_version TEXT NOT NULL,
                variant_model_version TEXT NOT NULL,
                traffic_split_ratio REAL NOT NULL,
                is_active BOOLEAN NOT NULL,
                start_time TEXT,
                end_time TEXT
            )
        """)
        # Table for storing individual metric data points
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_test_metrics (
                metric_id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                group_name TEXT NOT NULL, -- 'control' or 'variant'
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (test_id) REFERENCES ab_tests (test_id)
            )
        """)
        self.conn.commit()

    def _load_tests(self):
        """Loads all tests from the database into memory."""
        ab_test_logger.info("Loading A/B tests from database.")
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM ab_tests")
        rows = cursor.fetchall()
        for row in rows:
            test = ABTest.from_db_row(row)
            self.tests[test.test_id] = test
        ab_test_logger.info(f"Loaded {len(self.tests)} tests.")

    def _save_test(self, test: ABTest):
        """Saves a single test to the database (INSERT or REPLACE)."""
        ab_test_logger.info(f"Saving test {test.test_id} to database.")
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO ab_tests (test_id, name, description, control_model_version, variant_model_version, traffic_split_ratio, is_active, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, test.to_db_tuple())
        self.conn.commit()

    def create_test(self, name: str, description: str, variant_model_version: str, split: float = 0.5) -> ABTest:
        """
        Creates and registers a new A/B test.

        Args:
            name: The name of the test.
            description: A description of what is being tested.
            variant_model_version: The version identifier for the variant model.
            split: The ratio of traffic to send to the variant (0.0 to 1.0).

        Returns:
            The newly created ABTest object.
        """
        test = ABTest(
            name=name,
            description=description,
            variant_model_version=variant_model_version,
            traffic_split_ratio=split
        )
        self.tests[test.test_id] = test
        self._save_test(test)
        ab_test_logger.info(f"Created new A/B test '{name}' with ID {test.test_id}.")
        return test

    def start_test(self, test_id: str, duration_days: int = 7):
        """Starts an active A/B test."""
        if test_id not in self.tests:
            raise ValueError(f"Test with ID {test_id} not found.")
        
        test = self.tests[test_id]
        test.is_active = True
        test.start_time = datetime.now()
        test.end_time = test.start_time + timedelta(days=duration_days)
        self._save_test(test)
        ab_test_logger.info(f"Started A/B test '{test.name}' (ID: {test_id}).")

    def get_assignment(self, user_id: str, test_id: str) -> str:
        """
        Assigns a user to either the 'control' or 'variant' group.

        Args:
            user_id: A unique identifier for the user/request.
            test_id: The ID of the test to assign for.

        Returns:
            'variant' or 'control'.
        """
        if test_id not in self.tests or not self.tests[test_id].is_active:
            return 'control' # Default to control if test is not active
            
        test = self.tests[test_id]
        
        # Simple deterministic assignment based on user_id hash
        # A more robust implementation might use a dedicated bucketing service.
        assignment_value = (hash(user_id) % 100) / 100.0
        
        if assignment_value < test.traffic_split_ratio:
            return 'variant'
        else:
            return 'control'

    def record_metrics(self, test_id: str, group: str, metrics: Dict[str, float]):
        """Records performance metrics for a specific group in a test."""
        if test_id not in self.tests:
            ab_test_logger.warning(f"Attempted to record metrics for non-existent test {test_id}.")
            return

        ab_test_logger.debug(f"Recording {len(metrics)} metric(s) for test {test_id}, group {group}.")
        cursor = self.conn.cursor()
        now_iso = datetime.now().isoformat()
        
        metric_data = []
        for name, value in metrics.items():
            metric_data.append((
                f"m_{uuid.uuid4().hex[:12]}",
                test_id,
                group,
                name,
                value,
                now_iso
            ))

        cursor.executemany("""
            INSERT INTO ab_test_metrics (metric_id, test_id, group_name, metric_name, metric_value, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, metric_data)
        self.conn.commit()

    def evaluate_test(self, test_id: str, alpha: float = 0.05) -> ABTestResult:
        """
        Evaluates the results of a test using an independent t-test.

        Args:
            test_id: The ID of the test to evaluate.
            alpha: The significance level. A p-value below this threshold
                   is considered statistically significant.

        Returns:
            An ABTestResult with the winner and statistical summary.
        """
        if test_id not in self.tests:
            raise ValueError(f"Test with ID {test_id} not found.")

        # Load configuration for confidence level
        config = get_config_manager().settings.get('adaptive_learning', {})
        thresholds = config.get('thresholds', {})
        confidence_level = thresholds.get('ab_test_confidence_level', 0.95)
        alpha = 1 - confidence_level

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT group_name, metric_name, metric_value
            FROM ab_test_metrics
            WHERE test_id = ?
        """, (test_id,))
        
        # Group metrics by group and metric name
        metrics_by_group = defaultdict(lambda: defaultdict(list))
        for row in cursor.fetchall():
            metrics_by_group[row['group_name']][row['metric_name']].append(row['metric_value'])
        
        # Compare metrics between groups
        comparison = {}
        for metric_name in metrics_by_group['control'].keys():
            control_values = metrics_by_group['control'][metric_name]
            variant_values = metrics_by_group['variant'][metric_name]
            
            if not control_values or not variant_values:
                comparison[metric_name] = {"summary": "Insufficient data for comparison."}
                continue
                
            # Perform t-test
            t_stat, p_value = ttest_ind(control_values, variant_values)
            is_significant = p_value < alpha
            
            # Determine winner for this specific metric
            metric_winner = 'inconclusive'
            if is_significant:
                if statistics.mean(variant_values) > statistics.mean(control_values):
                    metric_winner = 'variant'
                else:
                    metric_winner = 'control'
            
            # Store comparison results
            comparison[metric_name] = {
                'control_mean': statistics.mean(control_values),
                'variant_mean': statistics.mean(variant_values),
                'p_value': p_value,
                'is_significant': bool(is_significant),
                'winner': metric_winner
            }
            
        # Determine overall winner based on the primary metric (e.g., 'accuracy')
        overall_winner = comparison.get('accuracy', {}).get('winner', 'inconclusive')

        summary = f"Overall Winner: {overall_winner.capitalize()} based on 'accuracy'."
            
        return ABTestResult(
            test_id=test_id,
            winner=overall_winner,
            confidence=1 - comparison.get('accuracy', {}).get('p_value', 1.0),
            summary=summary,
            metrics_comparison=comparison
        )

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            ab_test_logger.info(f"Database connection to {self.db_path} closed.")

_ab_test_manager_instance = None
_ab_test_manager_lock = threading.Lock()

def get_ab_test_manager(config_manager: ConfigManager) -> "ABTestManager":
    """
    Provides a singleton instance of the ABTestManager.
    """
    global _ab_test_manager_instance
    with _ab_test_manager_lock:
        if _ab_test_manager_instance is None:
            settings = config_manager.settings.get('adaptive_learning', {})
            db_config = settings.get('databases', {})
            db_path = db_config.get('ab_tests_db_path', 'data/adaptive/ab_tests.db')
            
            _ab_test_manager_instance = ABTestManager(db_path=db_path)
    return _ab_test_manager_instance