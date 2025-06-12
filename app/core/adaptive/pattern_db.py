"""
Adaptive Pattern Database for Priority 3 ML Implementation

This module manages the persistence of validated PII patterns discovered
by the adaptive learning system. It uses a SQLite database for storage.
"""

import sqlite3
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json

from app.core.logging import get_logger
from app.core.adaptive.pattern_learner import ValidatedPattern
from app.core.config_manager import get_config

db_logger = get_logger("adaptive_pattern_db")

class AdaptivePatternDB:
    """
    Manages a database of adaptively learned PII patterns.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the database connection.

        Args:
            db_path: Optional path to the database file. If None, uses default from config.
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Fallback to config if no path is provided
            config = get_config().get('adaptive_learning', {})
            self.db_path = Path(config.get('pattern_db_path', 'data/anonymized_pii.db'))

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
        db_logger.info(f"AdaptivePatternDB initialized with database at {self.db_path}")

    def _create_schema(self):
        """Create database schema if it doesn't exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    regex TEXT NOT NULL,
                    pii_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    created_at TEXT NOT NULL,
                    validated_at TEXT NOT NULL,
                    positive_matches INTEGER,
                    negative_matches INTEGER,
                    version INTEGER NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    metadata_json TEXT
                )
            """)
            self.conn.commit()
            db_logger.info("Table 'adaptive_patterns' is ready.")
        except sqlite3.Error as e:
            db_logger.error(f"Failed to create table: {e}")

    def add_or_update_pattern(self, pattern: ValidatedPattern) -> bool:
        """
        Adds a new validated pattern to the database or updates it if the new
        pattern has a higher confidence score.
        """
        cursor = self.conn.cursor()

        # Check if a pattern with the same regex already exists
        cursor.execute("SELECT pattern_id, confidence FROM adaptive_patterns WHERE regex = ?", (pattern.regex,))
        existing = cursor.fetchone()

        if existing:
            # If new pattern is not better, do nothing.
            if pattern.confidence <= existing['confidence']:
                db_logger.info(f"Skipping update for pattern with regex '{pattern.regex}'. New confidence ({pattern.confidence}) is not higher than existing ({existing['confidence']}).")
                return True # Operation is successful in the sense that no action was needed.

            # If new pattern is better, update the existing one
            pattern.pattern_id = existing['pattern_id'] # Use the existing ID
            sql = """
                UPDATE adaptive_patterns SET
                    pii_category=?, confidence=?, accuracy=?, precision=?, recall=?,
                    validated_at=?, positive_matches=?, negative_matches=?, version=version + 1, is_active=TRUE
                WHERE pattern_id=?
            """
            params = (
                pattern.pii_category, pattern.confidence, pattern.accuracy, pattern.precision, pattern.recall,
                pattern.validated_at.isoformat(), pattern.positive_matches, pattern.negative_matches,
                pattern.pattern_id
            )
        else:
            # If no pattern with this regex exists, insert a new one
            sql = """
                INSERT INTO adaptive_patterns (
                    pattern_id, regex, pii_category, confidence, accuracy, precision, recall,
                    created_at, validated_at, positive_matches, negative_matches, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                pattern.pattern_id, pattern.regex, pattern.pii_category, pattern.confidence,
                pattern.accuracy, pattern.precision, pattern.recall,
                pattern.created_at.isoformat(), pattern.validated_at.isoformat(),
                pattern.positive_matches, pattern.negative_matches, pattern.version
            )

        try:
            cursor.execute(sql, params)
            self.conn.commit()
            db_logger.info(f"Successfully added/updated pattern with ID: {pattern.pattern_id}")
            return True
        except sqlite3.Error as e:
            db_logger.error(f"Failed to add/update pattern {pattern.pattern_id}: {e}")
            return False

    def get_active_patterns(self) -> List[ValidatedPattern]:
        """
        Retrieves all active, high-confidence patterns from the database.

        Returns:
            A list of active ValidatedPattern objects.
        """
        patterns = []
        sql = "SELECT * FROM adaptive_patterns WHERE is_active = TRUE ORDER BY confidence DESC"
        try:
            cursor = self.conn.cursor()
            for row in cursor.execute(sql):
                patterns.append(self._row_to_pattern(row))
            db_logger.info(f"Retrieved {len(patterns)} active patterns from the database.")
        except sqlite3.Error as e:
            db_logger.error(f"Failed to retrieve active patterns: {e}")
        return patterns

    def deactivate_pattern(self, pattern_id: str) -> bool:
        """
        Deactivates a pattern in the database, e.g., due to performance degradation.

        Args:
            pattern_id: The ID of the pattern to deactivate.

        Returns:
            True if successful, False otherwise.
        """
        sql = "UPDATE adaptive_patterns SET is_active = FALSE WHERE pattern_id = ?"
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, (pattern_id,))
            self.conn.commit()
            if cursor.rowcount > 0:
                db_logger.info(f"Deactivated pattern: {pattern_id}")
                return True
            else:
                db_logger.warning(f"Attempted to deactivate non-existent pattern: {pattern_id}")
                return False
        except sqlite3.Error as e:
            db_logger.error(f"Failed to deactivate pattern {pattern_id}: {e}")
            return False

    def _row_to_pattern(self, row: sqlite3.Row) -> ValidatedPattern:
        """Converts a database row to a ValidatedPattern object."""
        return ValidatedPattern(
            pattern_id=row['pattern_id'],
            regex=row['regex'],
            pii_category=row['pii_category'],
            confidence=row['confidence'],
            accuracy=row['accuracy'],
            precision=row['precision'],
            recall=row['recall'],
            created_at=datetime.fromisoformat(row['created_at']),
            validated_at=datetime.fromisoformat(row['validated_at']),
            positive_matches=row['positive_matches'],
            negative_matches=row['negative_matches'],
            version=row['version']
        )

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            db_logger.info(f"Database connection to {self.db_path} closed.")

# Factory function for easy integration
def create_pattern_db(db_path: Optional[str] = None) -> AdaptivePatternDB:
    """Creates and returns an AdaptivePatternDB instance."""
    if db_path:
        return AdaptivePatternDB(db_path=db_path)
    return AdaptivePatternDB() 