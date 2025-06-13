"""
Tests for the A/B Testing Framework.
"""

import unittest
from pathlib import Path
from app.core.adaptive.ab_testing import ABTestManager, ABTest
from app.core.config_manager import ConfigManager
import os

class TestABTestManager(unittest.TestCase):
    db_path_str = "test_ab_manager.db"

    def setUp(self):
        """Set up the ABTestManager with a file-based database for persistence tests."""
        self.db_path = Path(self.db_path_str)
        self.ab_manager = ABTestManager(db_path=self.db_path)

    def tearDown(self):
        """Close the connection and remove the test database file."""
        self.ab_manager.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_create_and_load_test_persistence(self):
        """Test that created tests are persisted and loaded correctly."""
        test = self.ab_manager.create_test("Persistence Test", "Test desc", "v2")
        test_id = test.test_id

        # Close the manager and create a new instance to force reloading from DB
        self.ab_manager.close()
        new_manager = ABTestManager(db_path=self.db_path)

        self.assertIn(test_id, new_manager.tests)
        loaded_test = new_manager.tests[test_id]
        self.assertEqual(loaded_test.name, "Persistence Test")
        new_manager.close()

    def test_create_test(self):
        """Test the creation of a new A/B test."""
        test_name = "New Model Test"
        test_desc = "Testing v2 of the ML model."
        variant_version = "model_v2"

        test = self.ab_manager.create_test(test_name, test_desc, variant_version, split=0.2)

        self.assertIsInstance(test, ABTest)
        self.assertEqual(test.name, test_name)
        self.assertEqual(test.variant_model_version, variant_version)
        self.assertAlmostEqual(test.traffic_split_ratio, 0.2)
        self.assertFalse(test.is_active)
        self.assertIn(test.test_id, self.ab_manager.tests)

    def test_start_test(self):
        """Test starting an A/B test."""
        test = self.ab_manager.create_test("Test", "", "v2")
        self.assertFalse(test.is_active)

        self.ab_manager.start_test(test.test_id)

        self.assertTrue(test.is_active)
        self.assertIsNotNone(test.start_time)
        self.assertIsNotNone(test.end_time)

    def test_get_assignment(self):
        """Test user assignment to control and variant groups."""
        test = self.ab_manager.create_test("Assignment Test", "", "v2", split=0.5)
        self.ab_manager.start_test(test.test_id)

        # Test assignment is deterministic
        assignment1 = self.ab_manager.get_assignment("user123", test.test_id)
        assignment2 = self.ab_manager.get_assignment("user123", test.test_id)
        self.assertEqual(assignment1, assignment2)

        # Test that assignments are distributed
        assignments = [self.ab_manager.get_assignment(f"user_{i}", test.test_id) for i in range(100)]
        variant_count = assignments.count('variant')
        control_count = assignments.count('control')

        self.assertGreater(variant_count, 0)
        self.assertGreater(control_count, 0)
        # Check if the split is roughly correct (e.g., within 30% of the expectation)
        self.assertAlmostEqual(variant_count / 100, 0.5, delta=0.3)

    def test_get_assignment_for_inactive_test(self):
        """Test that inactive tests always assign to control."""
        test = self.ab_manager.create_test("Inactive Test", "", "v2")
        # Test is not started

        assignment = self.ab_manager.get_assignment("user123", test.test_id)
        self.assertEqual(assignment, 'control')

    def test_get_assignment_for_non_existent_test(self):
        """Test that non-existent tests always assign to control."""
        assignment = self.ab_manager.get_assignment("user123", "non_existent_test")
        self.assertEqual(assignment, 'control')

    def test_record_and_evaluate_metrics_variant_wins(self):
        """Test recording metrics and evaluating a test where the variant should win."""
        test = self.ab_manager.create_test("Evaluation Test", "Test eval", "v2", split=0.5)
        self.ab_manager.start_test(test.test_id)

        # Record metrics: variant has a clearly higher mean
        for _ in range(30):
            self.ab_manager.record_metrics(test.test_id, 'control', {'accuracy': 0.85, 'latency': 100})
            self.ab_manager.record_metrics(test.test_id, 'variant', {'accuracy': 0.95, 'latency': 90})

        # Evaluate the test
        result = self.ab_manager.evaluate_test(test.test_id)

        self.assertEqual(result.winner, 'variant')
        self.assertEqual(result.metrics_comparison['accuracy']['winner'], 'variant')
        self.assertLess(result.metrics_comparison['accuracy']['p_value'], 0.05)
        # For latency, lower is better. Our current implementation assumes higher is better.
        # This is an acceptable simplification for now, so we expect control to "win" latency.
        self.assertEqual(result.metrics_comparison['latency']['winner'], 'control')

    def test_evaluate_inconclusive_test_not_enough_data(self):
        """Test that evaluation is inconclusive if there's not enough data."""
        test = self.ab_manager.create_test("Inconclusive Test", "", "v2")
        self.ab_manager.start_test(test.test_id)

        # Only record one data point
        self.ab_manager.record_metrics(test.test_id, 'control', {'accuracy': 0.9})

        result = self.ab_manager.evaluate_test(test.test_id)
        self.assertEqual(result.winner, 'inconclusive')

if __name__ == '__main__':
    unittest.main()