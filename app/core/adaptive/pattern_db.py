"""
Adaptive Pattern Database for Priority 3 ML Implementation

This module manages the persistence of validated PII patterns discovered
by the adaptive learning system. It uses a SQLite database for storage.
"""
from __future__ import annotations
import sqlite3
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass, field

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from app.core.logging import get_logger
from app.core.config_manager import get_config_manager

# New imports needed for standalone session management
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

db_logger = get_logger("adaptive_pattern_db")

@dataclass
class AdaptivePattern:
    """Represents a pattern stored in the adaptive pattern database."""
    pattern_id: str
    regex: str
    pii_category: str
    confidence: float
    positive_matches: int
    negative_matches: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    version: int = 1
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_validated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the pattern to a dictionary, handling datetime conversion."""
        data = self.__dict__.copy()
        data['created_at'] = self.created_at.isoformat()
        if self.last_validated_at:
            data['last_validated_at'] = self.last_validated_at.isoformat()
        return data

    @classmethod
    def from_row(cls, row) -> "AdaptivePattern":
        """Creates an AdaptivePattern from a database row."""
        data = dict(row._mapping)
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_validated_at') and isinstance(data['last_validated_at'], str):
            data['last_validated_at'] = datetime.fromisoformat(data['last_validated_at'])
        
        # Remove metadata if it's not a field in the dataclass
        data.pop('metadata_json', None)

        return cls(**data)

class AdaptivePatternDB:
    """
    Manages a database of adaptively learned PII patterns.
    Can operate with an injected session or create its own connection.
    """

    def __init__(self, db_session: Optional[Session] = None, db_path: Optional[Path] = None):
        """
        Initializes the database connection.

        Args:
            db_session: An optional active SQLAlchemy Session.
            db_path: An optional path to the database file. One of db_session or db_path must be provided.
        """
        if db_session:
            self.db = db_session
            self.manages_session = False
        elif db_path:
            db_url = f"sqlite:///{db_path.resolve()}"
            engine = create_engine(db_url)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self.db = self.SessionLocal()
            self.manages_session = True
        else:
            raise ValueError("Either db_session or db_path must be provided.")

        self._create_schema()
        db_logger.info(f"AdaptivePatternDB initialized (manages session: {self.manages_session}).")


    def _create_schema(self):
        """Create database schema if it doesn't exist."""
        try:
            self.db.execute(text("""
                CREATE TABLE IF NOT EXISTS adaptive_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    regex TEXT NOT NULL,
                    pii_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    created_at TEXT NOT NULL,
                    last_validated_at TEXT,
                    positive_matches INTEGER,
                    negative_matches INTEGER,
                    version INTEGER NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    metadata_json TEXT
                )
            """))
            self.db.commit()
            db_logger.info("Table 'adaptive_patterns' is ready.")
        except SQLAlchemyError as e:
            db_logger.error(f"Failed to create table: {e}")
            self.db.rollback()

    def close(self):
        """Closes the database session if this instance is managing it."""
        if self.manages_session and self.db:
            self.db.close()
            db_logger.info("Closed self-managed database session.")

    def add_or_update_pattern(self, pattern: AdaptivePattern) -> bool:
        """
        Adds a new validated pattern to the database or updates it if the new
        pattern has a higher confidence score.
        """
        # Check if a pattern with the same regex already exists
        existing_row = self.db.execute(
            text("SELECT pattern_id, confidence FROM adaptive_patterns WHERE regex = :regex"),
            {"regex": pattern.regex}
        ).first()

        if existing_row:
            existing = dict(existing_row._mapping)
            # If new pattern is not better, do nothing.
            if pattern.confidence <= existing['confidence']:
                db_logger.info(f"Skipping update for pattern with regex '{pattern.regex}'. New confidence ({pattern.confidence}) is not higher than existing ({existing['confidence']}).")
                return True # Operation is successful in the sense that no action was needed.

            # If new pattern is better, update the existing one
            pattern.pattern_id = existing['pattern_id'] # Use the existing ID
            sql = text("""
                UPDATE adaptive_patterns SET
                    pii_category=:pii_category, confidence=:confidence, accuracy=:accuracy, 
                    precision=:precision, recall=:recall, last_validated_at=:last_validated_at, 
                    positive_matches=:positive_matches, negative_matches=:negative_matches, 
                    version=version + 1, is_active=:is_active
                WHERE pattern_id=:pattern_id
            """)
            params = {
                "pii_category": pattern.pii_category,
                "confidence": pattern.confidence,
                "accuracy": pattern.accuracy,
                "precision": pattern.precision,
                "recall": pattern.recall,
                "last_validated_at": pattern.last_validated_at.isoformat() if pattern.last_validated_at else None,
                "positive_matches": pattern.positive_matches,
                "negative_matches": pattern.negative_matches,
                "pattern_id": pattern.pattern_id,
                "is_active": pattern.is_active
            }
        else:
            # If no pattern with this regex exists, insert a new one
            sql = text("""
                INSERT INTO adaptive_patterns (
                    pattern_id, regex, pii_category, confidence, accuracy, precision, recall,
                    created_at, last_validated_at, positive_matches, negative_matches, version, is_active
                ) VALUES (
                    :pattern_id, :regex, :pii_category, :confidence, :accuracy, :precision, :recall,
                    :created_at, :last_validated_at, :positive_matches, :negative_matches, :version, :is_active
                )
            """)
            params = {
                "pattern_id": pattern.pattern_id,
                "regex": pattern.regex,
                "pii_category": pattern.pii_category,
                "confidence": pattern.confidence,
                "accuracy": pattern.accuracy,
                "precision": pattern.precision,
                "recall": pattern.recall,
                "created_at": pattern.created_at.isoformat(),
                "last_validated_at": pattern.last_validated_at.isoformat() if pattern.last_validated_at else None,
                "positive_matches": pattern.positive_matches,
                "negative_matches": pattern.negative_matches,
                "version": pattern.version,
                "is_active": pattern.is_active
            }

        try:
            self.db.execute(sql, params)
            self.db.commit()
            db_logger.info(f"Successfully added/updated pattern with ID: {pattern.pattern_id}")
            return True
        except SQLAlchemyError as e:
            db_logger.error(f"Failed to add/update pattern {pattern.pattern_id}: {e}")
            self.db.rollback()
            return False

    def get_active_patterns(self) -> List[AdaptivePattern]:
        """
        Retrieves all active, high-confidence patterns from the database.

        Returns:
            A list of active ValidatedPattern objects.
        """
        patterns = []
        sql = text("SELECT * FROM adaptive_patterns WHERE is_active = TRUE ORDER BY confidence DESC")
        try:
            result = self.db.execute(sql)
            for row in result:
                patterns.append(AdaptivePattern.from_row(row))
            db_logger.info(f"Retrieved {len(patterns)} active patterns from the database.")
        except SQLAlchemyError as e:
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
        sql = text("UPDATE adaptive_patterns SET is_active = FALSE WHERE pattern_id = :pattern_id")
        try:
            result = self.db.execute(sql, {"pattern_id": pattern_id})
            self.db.commit()
            if result.rowcount > 0:
                db_logger.info(f"Deactivated pattern: {pattern_id}")
                return True
            else:
                db_logger.warning(f"Attempted to deactivate non-existent pattern: {pattern_id}")
                return False
        except SQLAlchemyError as e:
            db_logger.error(f"Failed to deactivate pattern {pattern_id}: {e}")
            self.db.rollback()
            return False

    def _row_to_pattern(self, row: sqlite3.Row) -> AdaptivePattern:
        """Converts a database row to a ValidatedPattern object."""
        return AdaptivePattern.from_row(row)

# Factory function for easy integration - This function is now outdated and should be removed or refactored.
# For now, we will leave it to avoid breaking other parts of the code that might use it,
# but it will need to be addressed.
def create_pattern_db(db_path: Optional[Path] = None) -> AdaptivePatternDB:
    """Creates and returns an AdaptivePatternDB instance."""
    if not db_path:
        config_manager = get_config_manager()
        db_path_str = config_manager.settings.get('adaptive_learning', {}).get('databases', {}).get('patterns_db', 'data/adaptive/patterns.db')
        db_path = Path(db_path_str)
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return AdaptivePatternDB(db_path=db_path) 