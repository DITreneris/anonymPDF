"""
Tests for Training Data Collector and Storage - Session 4 Priority 3 Implementation

Tests the functionality of TrainingDataStorage and TrainingDataCollector
for managing TrainingExample objects.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any

# Assuming TrainingExample might be defined elsewhere or has a simple structure for testing
# If it's complex, we might need its actual definition or a mock.
from app.core.ml_engine import TrainingExample
from app.core.training_data import TrainingDataStorage, TrainingDataCollector, DatasetStats
from app.core.feature_engineering import FeatureExtractor # For TrainingExample creation
from app.core.config_manager import get_config # For default configs if needed

# Helper to create dummy TrainingExample instances
def create_dummy_example(
    text: str, 
    category: str, 
    is_positive: bool, 
    doc_type: str = "test_doc",
    features: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> TrainingExample:
    if features is None:
        features = {"feature1": 1.0, "feature2": "test"}
    if metadata is None:
        metadata = {"source_id": "dummy_source"}
    return TrainingExample(
        detection_text=text,
        category=category,
        context=f"Some context around {text}",
        features=features,
        confidence_score=0.9 if is_positive else 0.1,
        is_true_positive=is_positive,
        document_type=doc_type,
        metadata=metadata
    )

@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for the SQLite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def training_data_storage(temp_db_dir):
    """Fixture for TrainingDataStorage using a temporary database."""
    return TrainingDataStorage(data_dir=str(temp_db_dir))

@pytest.fixture
def training_data_collector(training_data_storage):
    """Fixture for TrainingDataCollector."""
    # Mock parts of config if TrainingDataCollector relies on it heavily for initialization
    # For now, assume default or it handles missing config gracefully
    collector_config = get_config().get('training_data', {})
    collector = TrainingDataCollector(config=collector_config)
    collector.storage = training_data_storage # Override storage with our test fixture
    return collector


@pytest.mark.unit
class TestTrainingDataStorage:
    """Tests for the TrainingDataStorage class."""

    def test_storage_initialization(self, training_data_storage: TrainingDataStorage):
        """Test that the database and training_examples table are created."""
        assert training_data_storage.db_path.exists()
        with sqlite3.connect(training_data_storage.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_examples'")
            table_exists = cursor.fetchone() is not None
            assert table_exists, "'training_examples' table was not created"

    def test_save_and_load_training_examples(self, training_data_storage: TrainingDataStorage):
        """Test saving and loading TrainingExample objects."""
        examples_to_save = [
            create_dummy_example("John Doe", "NAME", True, "doc1"),
            create_dummy_example("123 Main St", "ADDRESS", True, "doc1", source="test_source_1"),
            create_dummy_example("Not PII", "OTHER", False, "doc2", source="test_source_2"),
        ]
        
        training_data_storage.save_training_examples(examples_to_save[:2], source="test_source_1")
        training_data_storage.save_training_examples([examples_to_save[2]], source="test_source_2")

        # Load all
        loaded_examples = training_data_storage.load_training_examples()
        assert len(loaded_examples) == 3
        # Simple check, more thorough checks would compare all fields
        loaded_texts = sorted([ex.detection_text for ex in loaded_examples])
        expected_texts = sorted([ex.detection_text for ex in examples_to_save])
        assert loaded_texts == expected_texts

        # Load by source
        loaded_source1 = training_data_storage.load_training_examples(source="test_source_1")
        assert len(loaded_source1) == 2
        assert all(ex.metadata.get('source_id') != 'dummy_source' or ex.detection_text in ["John Doe", "123 Main St"] for ex in loaded_source1)


        loaded_source2 = training_data_storage.load_training_examples(source="test_source_2")
        assert len(loaded_source2) == 1
        assert loaded_source2[0].detection_text == "Not PII"

        # Load by category
        loaded_name = training_data_storage.load_training_examples(category="NAME")
        assert len(loaded_name) == 1
        assert loaded_name[0].detection_text == "John Doe"
        
        # Load with limit
        loaded_limited = training_data_storage.load_training_examples(limit=1)
        assert len(loaded_limited) == 1

    def test_get_dataset_stats_empty(self, training_data_storage: TrainingDataStorage):
        """Test get_dataset_stats on an empty database."""
        stats = training_data_storage.get_dataset_stats()
        assert stats.total_samples == 0
        assert stats.positive_samples == 0
        assert stats.negative_samples == 0
        assert not stats.categories
        assert not stats.document_types

    def test_get_dataset_stats_with_data(self, training_data_storage: TrainingDataStorage):
        """Test get_dataset_stats with some data."""
        examples_to_save = [
            create_dummy_example("Text1", "CAT1", True, "doc_A"),
            create_dummy_example("Text2", "CAT1", False, "doc_A"),
            create_dummy_example("Text3", "CAT2", True, "doc_B"),
        ]
        training_data_storage.save_training_examples(examples_to_save, source="stats_test")

        stats = training_data_storage.get_dataset_stats()
        assert stats.total_samples == 3
        assert stats.positive_samples == 2
        assert stats.negative_samples == 1
        assert stats.categories == {"CAT1": 2, "CAT2": 1}
        assert stats.document_types == {"doc_A": 2, "doc_B": 1}
        # Basic check for date range, more specific checks might be needed
        assert isinstance(stats.date_range[0], datetime)
        assert isinstance(stats.date_range[1], datetime)


@pytest.mark.unit
class TestTrainingDataCollector:
    """Tests for the TrainingDataCollector class."""

    def test_collector_initialization(self, training_data_collector: TrainingDataCollector):
        """Test basic initialization of the collector."""
        assert training_data_collector is not None
        assert isinstance(training_data_collector.storage, TrainingDataStorage)

    def test_collect_all_training_data_empty_storage(self, training_data_collector: TrainingDataCollector):
        """Test collecting data when storage is empty (and no synthetic)."""
        # Assuming synthetic data generation is handled or mocked if complex
        examples = training_data_collector.collect_all_training_data(include_synthetic=False)
        assert len(examples) == 0

    def test_collect_all_training_data_with_examples_in_storage(self, training_data_collector: TrainingDataCollector):
        """Test collecting data when examples exist in storage (no synthetic)."""
        stored_examples = [
            create_dummy_example("Stored1", "STORED_CAT", True),
            create_dummy_example("Stored2", "STORED_CAT", False),
        ]
        training_data_collector.storage.save_training_examples(stored_examples, source="from_storage")
        
        collected_examples = training_data_collector.collect_all_training_data(include_synthetic=False)
        assert len(collected_examples) == 2
        # Ensure the collected examples are the ones from storage
        collected_texts = sorted([ex.detection_text for ex in collected_examples])
        expected_texts = sorted([ex.detection_text for ex in stored_examples])
        assert collected_texts == expected_texts

    def test_get_balanced_training_set_empty(self, training_data_collector: TrainingDataCollector):
        """Test getting a balanced set when no data is available."""
        examples, stats = training_data_collector.get_balanced_training_set()
        assert len(examples) == 0
        assert stats.total_samples == 0

    def test_get_balanced_training_set_with_data(self, training_data_collector: TrainingDataCollector):
        """Test getting a balanced set with some data."""
        # Populate with more positive than negative
        examples_data = []
        for i in range(10): # 10 positive
            examples_data.append(create_dummy_example(f"Positive {i}", "POS_CAT", True))
        for i in range(3): # 3 negative
            examples_data.append(create_dummy_example(f"Negative {i}", "NEG_CAT", False))
        
        training_data_collector.storage.save_training_examples(examples_data, source="balancing_test")

        # Request a balanced set
        balanced_examples, stats = training_data_collector.get_balanced_training_set(max_samples=10, balance_ratio=0.5)
        
        assert len(balanced_examples) <= 10 # Should be at most max_samples
        # Check balance (might not be perfect if few samples of one class)
        # For this test, we expect 3 negative (all available) and 3 positive (to match negatives for 0.5 goal if total is 6, or more if max_samples allows)
        # Actual number will depend on the sampling logic.
        # Here, we'll get all 3 negatives and min(5, 3) = 3 positives = 6 examples.
        
        num_pos = sum(1 for ex in balanced_examples if ex.is_true_positive)
        num_neg = sum(1 for ex in balanced_examples if not ex.is_true_positive)

        assert num_neg == 3 # All negative samples should be included
        assert num_pos >= 0 # Some positive samples
        
        # More precise check if we know the sampling: if target_pos = 5, target_neg = 5
        # We have 10 pos, 3 neg. Will take min(5,10)=5 pos, min(5,3)=3 neg. So 8 examples, 5 pos, 3 neg.
        # If max_samples = 6, balance_ratio = 0.5 -> target_pos=3, target_neg=3
        # Will take min(3,10)=3 pos, min(3,3)=3 neg. So 6 examples.
        if max_samples == 10 and balance_ratio == 0.5: # Original params for test logic
             # Default sampling strategy might take all of the smaller class, and then try to match from larger class
             # Given 3 negative, it might aim for 3 positive for a 50/50 split from the available data.
             # Or it might try to get 5 positive and 3 negative to reach closer to max_samples.
             # The implementation takes min(target_positive, len(positive_examples)) and min(target_negative, len(negative_examples))
             # target_positive = 10 * 0.5 = 5. target_negative = 10 - 5 = 5
             # sampled_positive = min(5, 10) = 5
             # sampled_negative = min(5, 3) = 3
             # Total = 8. 5 positive, 3 negative.
            assert num_pos == 5
            assert num_neg == 3
            assert len(balanced_examples) == 8

        assert stats.total_samples == 13 # From storage before balancing
        assert stats.positive_samples == 10
        assert stats.negative_samples == 3 