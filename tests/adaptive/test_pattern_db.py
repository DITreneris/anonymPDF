"""
Tests for the Adaptive Pattern Database.
"""

import unittest
import os
from datetime import datetime
from app.core.adaptive.pattern_db import AdaptivePatternDB
from app.core.adaptive.pattern_learner import ValidatedPattern

class TestAdaptivePatternDB(unittest.TestCase):
    def setUp(self):
        """Set up an in-memory SQLite database for testing."""
        self.db_path = ":memory:"
        self.db = AdaptivePatternDB(db_path=self.db_path)
        self.pattern1 = ValidatedPattern(
            pattern_id="test_pattern_1",
            regex="\\b\\d{5}\\b",
            pii_category="ZIP_CODE",
            confidence=0.95,
            accuracy=0.9,
            precision=0.95,
            recall=0.85,
            created_at=datetime.now(),
            positive_matches=19,
            negative_matches=1
        )
        self.pattern2 = ValidatedPattern(
            pattern_id="test_pattern_2",
            regex="[A-Z]{3}-\\d{3}",
            pii_category="PRODUCT_CODE",
            confidence=0.88,
            accuracy=0.85,
            precision=0.88,
            recall=0.82,
            created_at=datetime.now(),
            positive_matches=22,
            negative_matches=3
        )

    def tearDown(self):
        """Close the database connection."""
        self.db.close()

    def test_add_and_get_pattern(self):
        """Test adding a pattern and retrieving it."""
        self.assertTrue(self.db.add_or_update_pattern(self.pattern1))
        
        active_patterns = self.db.get_active_patterns()
        self.assertEqual(len(active_patterns), 1)
        
        retrieved_pattern = active_patterns[0]
        self.assertEqual(retrieved_pattern.pattern_id, self.pattern1.pattern_id)
        self.assertEqual(retrieved_pattern.regex, self.pattern1.regex)
        self.assertAlmostEqual(retrieved_pattern.confidence, self.pattern1.confidence)

    def test_update_pattern(self):
        """Test updating an existing pattern."""
        self.db.add_or_update_pattern(self.pattern1)
        
        # Now update the pattern
        self.pattern1.confidence = 0.98
        self.pattern1.positive_matches = 25
        self.assertTrue(self.db.add_or_update_pattern(self.pattern1))
        
        active_patterns = self.db.get_active_patterns()
        self.assertEqual(len(active_patterns), 1)
        self.assertAlmostEqual(active_patterns[0].confidence, 0.98)
        self.assertEqual(active_patterns[0].positive_matches, 25)
        self.assertEqual(active_patterns[0].version, 2) # Version should be incremented

    def test_get_active_patterns(self):
        """Test retrieving only active patterns."""
        self.db.add_or_update_pattern(self.pattern1)
        self.db.add_or_update_pattern(self.pattern2)
        
        active_patterns = self.db.get_active_patterns()
        self.assertEqual(len(active_patterns), 2)

    def test_deactivate_pattern(self):
        """Test deactivating a pattern."""
        self.db.add_or_update_pattern(self.pattern1)
        self.db.add_or_update_pattern(self.pattern2)
        
        self.assertTrue(self.db.deactivate_pattern(self.pattern1.pattern_id))
        
        active_patterns = self.db.get_active_patterns()
        self.assertEqual(len(active_patterns), 1)
        self.assertEqual(active_patterns[0].pattern_id, self.pattern2.pattern_id)

    def test_deactivate_non_existent_pattern(self):
        """Test that deactivating a non-existent pattern fails gracefully."""
        self.assertFalse(self.db.deactivate_pattern("non_existent_id"))

if __name__ == '__main__':
    unittest.main() 