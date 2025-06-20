"""
Tests for the Adaptive Pattern Database.
"""

import pytest
from datetime import datetime
from dataclasses import asdict
from app.core.adaptive.pattern_db import AdaptivePatternDB, AdaptivePattern
from sqlalchemy.orm import Session
from sqlalchemy import text

@pytest.fixture
def pattern_db(db_session: Session) -> AdaptivePatternDB:
    """Provides a clean instance of AdaptivePatternDB for each test."""
    return AdaptivePatternDB(db_session=db_session)

@pytest.fixture
def sample_pattern1() -> AdaptivePattern:
    """A sample pattern for testing."""
    return AdaptivePattern(
        pattern_id="test_pattern_1",
        regex="\\b\\d{5}\\b",
        pii_category="ZIP_CODE",
        confidence=0.95,
        positive_matches=19,
        negative_matches=1,
        last_validated_at=datetime.now()
    )

@pytest.fixture
def sample_pattern2() -> AdaptivePattern:
    """Another sample pattern for testing."""
    return AdaptivePattern(
        pattern_id="test_pattern_2",
        regex="[A-Z]{3}-\\d{3}",
        pii_category="PRODUCT_CODE",
        confidence=0.88,
        positive_matches=22,
        negative_matches=3,
        last_validated_at=datetime.now()
    )

class TestAdaptivePatternDB:
    def test_add_and_get_pattern(self, pattern_db: AdaptivePatternDB, sample_pattern1: AdaptivePattern):
        """Test adding a pattern and retrieving it."""
        # A fresh database should be empty.
        initial_patterns = pattern_db.get_active_patterns()
        assert len(initial_patterns) == 0

        # Now, add a pattern and verify it's there.
        pattern_db.add_or_update_pattern(sample_pattern1)
        
        final_patterns = pattern_db.get_active_patterns()
        assert len(final_patterns) == 1
        assert final_patterns[0].pii_category == sample_pattern1.pii_category
        assert final_patterns[0].pattern_id == sample_pattern1.pattern_id

    def test_update_pattern(self, pattern_db: AdaptivePatternDB, sample_pattern1: AdaptivePattern):
        """Test updating an existing pattern with higher confidence."""
        # First, add the initial pattern.
        pattern_db.add_or_update_pattern(sample_pattern1)

        # Modify the pattern with a higher confidence and update it.
        updated_details = asdict(sample_pattern1)
        updated_details['confidence'] = 0.98
        updated_details['positive_matches'] = 20
        updated_pattern_obj = AdaptivePattern(**updated_details)
        
        pattern_db.add_or_update_pattern(updated_pattern_obj)
        
        # We must query the DB directly since no getter for a single pattern exists
        result = pattern_db.db.execute(
            text("SELECT * FROM adaptive_patterns WHERE pattern_id = :id"),
            {"id": sample_pattern1.pattern_id}
        ).first()
        
        fetched_pattern = AdaptivePattern.from_row(result)
        assert fetched_pattern is not None
        assert fetched_pattern.confidence == 0.98
        assert fetched_pattern.version == 2 # Version should increment on update

    def test_get_active_patterns(self, pattern_db: AdaptivePatternDB, sample_pattern1, sample_pattern2):
        """Test retrieving only active patterns."""
        # Add two patterns, one active and one inactive.
        sample_pattern1.is_active = True
        sample_pattern2.is_active = False
        pattern_db.add_or_update_pattern(sample_pattern1)
        pattern_db.add_or_update_pattern(sample_pattern2)

        active_patterns = pattern_db.get_active_patterns()
        assert len(active_patterns) == 1
        assert active_patterns[0].pattern_id == sample_pattern1.pattern_id

    def test_deactivate_pattern(self, pattern_db: AdaptivePatternDB, sample_pattern1: AdaptivePattern):
        """Test deactivating a pattern."""
        # Add the pattern first.
        pattern_db.add_or_update_pattern(sample_pattern1)

        pattern_db.deactivate_pattern(sample_pattern1.pattern_id)
        
        active_patterns = pattern_db.get_active_patterns()
        assert len(active_patterns) == 0
        
        # Verify by querying the DB directly
        result = pattern_db.db.execute(
            text("SELECT * FROM adaptive_patterns WHERE pattern_id = :id"),
            {"id": sample_pattern1.pattern_id}
        ).first()
        deactivated_pattern = AdaptivePattern.from_row(result)
        assert deactivated_pattern is not None
        assert not deactivated_pattern.is_active

    def test_update_pattern_fields(self, pattern_db: AdaptivePatternDB, sample_pattern1: AdaptivePattern):
        """Test that various fields are updated correctly."""
        # Add the pattern first.
        pattern_db.add_or_update_pattern(sample_pattern1)

        sample_pattern1.confidence = 0.98 # Higher confidence to ensure update
        sample_pattern1.pii_category = "UPDATED_CATEGORY"
        sample_pattern1.positive_matches += 5
        
        pattern_db.add_or_update_pattern(sample_pattern1)
        
        result = pattern_db.db.execute(
            text("SELECT * FROM adaptive_patterns WHERE pattern_id = :id"),
            {"id": sample_pattern1.pattern_id}
        ).first()
        updated_pattern = AdaptivePattern.from_row(result)

        assert updated_pattern.pii_category == "UPDATED_CATEGORY"
        assert updated_pattern.positive_matches == 24
        assert updated_pattern.version == 2

if __name__ == '__main__':
    pytest.main() 