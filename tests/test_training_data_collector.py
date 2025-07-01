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
import gc
from unittest.mock import Mock, patch, MagicMock

# Assuming TrainingExample might be defined elsewhere or has a simple structure for testing
# If it's complex, we might need its actual definition or a mock.
from app.core.ml_engine import TrainingExample
from app.core.training_data import (
    TrainingDataStorage, TrainingDataCollector, DatasetStats,
    Priority2DataCollector, SyntheticDataGenerator, create_training_data_collector
)
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
    """Fixture for TrainingDataStorage that ensures the database is closed after tests."""
    storage = TrainingDataStorage(data_dir=str(temp_db_dir))
    try:
        yield storage
    finally:
        storage.close()
        # Force garbage collection to release file handles on Windows
        gc.collect()

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
class TestDatasetStats:
    """Tests for the DatasetStats dataclass."""

    def test_get_class_balance_normal(self):
        """Test class balance calculation with normal data."""
        stats = DatasetStats(
            total_samples=100,
            positive_samples=60,
            negative_samples=40,
            categories={'PII': 60, 'NON_PII': 40},
            document_types={'doc1': 50, 'doc2': 50},
            date_range=(datetime.now(), datetime.now()),
            quality_score=0.8
        )
        assert stats.get_class_balance() == 0.6

    def test_get_class_balance_zero_total(self):
        """Test class balance calculation with zero total samples."""
        stats = DatasetStats(
            total_samples=0,
            positive_samples=0,
            negative_samples=0,
            categories={},
            document_types={},
            date_range=(datetime.now(), datetime.now()),
            quality_score=0.0
        )
        assert stats.get_class_balance() == 0.0

    def test_get_class_balance_edge_cases(self):
        """Test class balance calculation with edge cases."""
        # All positive
        stats = DatasetStats(
            total_samples=50,
            positive_samples=50,
            negative_samples=0,
            categories={'PII': 50},
            document_types={'doc1': 50},
            date_range=(datetime.now(), datetime.now()),
            quality_score=1.0
        )
        assert stats.get_class_balance() == 1.0
        
        # All negative
        stats = DatasetStats(
            total_samples=30,
            positive_samples=0,
            negative_samples=30,
            categories={'NON_PII': 30},
            document_types={'doc1': 30},
            date_range=(datetime.now(), datetime.now()),
            quality_score=0.5
        )
        assert stats.get_class_balance() == 0.0


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
            create_dummy_example("123 Main St", "ADDRESS", True, "doc1", metadata={"source": "test_source_1"}),
            create_dummy_example("Not PII", "OTHER", False, "doc2", metadata={"source": "test_source_2"}),
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
        # A more robust check on what was loaded
        loaded_source1_texts = {ex.detection_text for ex in loaded_source1}
        assert "John Doe" in loaded_source1_texts
        assert "123 Main St" in loaded_source1_texts


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
        assert stats is None

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

    def test_database_connection_handling(self, temp_db_dir):
        """Test database connection and reconnection."""
        storage = TrainingDataStorage(data_dir=str(temp_db_dir))
        try:
            # Test initial connection
            assert storage.conn is not None
            
            # Test close and reconnect
            storage.close()
            assert storage.conn is None
            
            # Test reconnection when needed
            storage._connect()
            assert storage.conn is not None
        finally:
            storage.close()

    def test_save_examples_database_error(self, training_data_storage):
        """Test handling of database errors during save."""
        # Close the connection to force an error
        training_data_storage.close()
        
        examples = [create_dummy_example("Test", "TEST", True)]
        
        # This should handle the error gracefully (reconnect and retry)
        training_data_storage.save_training_examples(examples, "test_error")
        
        # Verify the data was still saved after reconnection
        loaded = training_data_storage.load_training_examples()
        assert len(loaded) == 1
        assert loaded[0].detection_text == "Test"


@pytest.mark.unit
class TestPriority2DataCollector:
    """Tests for the Priority2DataCollector class."""

    @patch('app.core.training_data.get_config_manager')
    @patch('app.core.training_data.ContextualValidator')
    @patch('app.core.training_data.create_feature_extractor')
    def test_initialization(self, mock_feature_extractor, mock_validator, mock_config_manager):
        """Test Priority2DataCollector initialization."""
        mock_config = Mock()
        mock_config.cities = ['Vilnius', 'Kaunas']
        mock_config.brand_names = ['Brand1', 'Brand2']
        mock_config_manager.return_value = mock_config
        
        collector = Priority2DataCollector()
        
        assert collector.contextual_validator is not None
        assert collector.feature_extractor is not None
        mock_validator.assert_called_once_with(
            cities=['Vilnius', 'Kaunas'],
            brand_names=['Brand1', 'Brand2']
        )

    @patch('app.core.training_data.get_config_manager')
    @patch('app.core.training_data.ContextualValidator')
    @patch('app.core.training_data.create_feature_extractor')
    def test_collect_from_priority2_logs_nonexistent_dir(self, mock_feature_extractor, mock_validator, mock_config_manager):
        """Test collecting from non-existent log directory."""
        mock_config_manager.return_value = Mock(cities=[], brand_names=[])
        
        collector = Priority2DataCollector()
        examples = collector.collect_from_priority2_logs("nonexistent_dir")
        
        assert examples == []

    @patch('app.core.training_data.get_config_manager')
    @patch('app.core.training_data.ContextualValidator')
    @patch('app.core.training_data.create_feature_extractor')
    def test_collect_from_priority2_logs_with_files(self, mock_feature_extractor, mock_validator, mock_config_manager, temp_db_dir):
        """Test collecting from log directory with files."""
        mock_config_manager.return_value = Mock(cities=[], brand_names=[])
        
        # Create mock log files
        log_dir = temp_db_dir / "logs"
        log_dir.mkdir()
        (log_dir / "processing_test.log").write_text("test log content")
        (log_dir / "other.log").write_text("other content")
        
        collector = Priority2DataCollector()
        examples = collector.collect_from_priority2_logs(str(log_dir))
        
        # Should return empty list since _parse_log_file is not implemented
        assert examples == []

    @patch('app.core.training_data.get_config_manager')
    @patch('app.core.training_data.ContextualValidator')
    @patch('app.core.training_data.create_feature_extractor')
    def test_collect_from_detection_results(self, mock_feature_extractor, mock_validator, mock_config_manager):
        """Test collecting training data from detection results."""
        mock_config_manager.return_value = Mock(cities=[], brand_names=[])
        
        # Mock feature extractor
        mock_features = Mock()
        mock_features.to_dict.return_value = {"feature1": 0.8, "feature2": "test"}
        mock_feature_extractor.return_value.extract_all_features.return_value = mock_features
        
        collector = Priority2DataCollector()
        
        detection_results = [
            {
                'text': 'Jonas Jonaitis',
                'category': 'person_name',
                'context': 'Vardas: Jonas Jonaitis',
                'confidence': 0.85,
                'position': 10,
                'full_text': 'Document text here',
                'document_type': 'contract',
                'language': 'lt'
            },
            {
                'text': 'Not PII',
                'category': 'other',
                'context': 'Some text',
                'confidence': 0.3
            }
        ]
        
        examples = collector.collect_from_detection_results(detection_results)
        
        assert len(examples) == 2
        assert examples[0].detection_text == 'Jonas Jonaitis'
        assert examples[0].category == 'person_name'
        assert examples[0].is_true_positive == True  # confidence > 0.7
        assert examples[1].is_true_positive == False  # confidence < 0.7

    @patch('app.core.training_data.get_config_manager')
    @patch('app.core.training_data.ContextualValidator')
    @patch('app.core.training_data.create_feature_extractor')
    def test_collect_validated_examples(self, mock_feature_extractor, mock_validator_class, mock_config_manager):
        """Test collecting validated training examples."""
        mock_config_manager.return_value = Mock(cities=[], brand_names=[])
        
        # Mock contextual validator
        mock_validator = Mock()
        mock_validator.validate.side_effect = [
            (True, "Valid context"),
            (False, "Invalid context")
        ]
        mock_validator_class.return_value = mock_validator
        
        collector = Priority2DataCollector()
        
        input_examples = [
            create_dummy_example("Valid PII", "NAME", True),
            create_dummy_example("Invalid PII", "NAME", True)
        ]
        
        validated = collector.collect_validated_examples(input_examples)
        
        assert len(validated) == 2
        assert validated[0].context == "Valid context"
        assert validated[0].is_true_positive == True
        assert validated[1].context == "Invalid context" 
        assert validated[1].is_true_positive == False  # Invalid examples become negative
        assert validated[1].confidence_score == 0.5


@pytest.mark.unit
class TestSyntheticDataGenerator:
    """Tests for the SyntheticDataGenerator class."""

    @patch('app.core.training_data.create_feature_extractor')
    def test_initialization(self, mock_feature_extractor):
        """Test SyntheticDataGenerator initialization."""
        generator = SyntheticDataGenerator()
        
        assert generator.feature_extractor is not None
        assert generator.pii_templates is not None
        assert 'person_name' in generator.pii_templates
        assert 'emails' in generator.pii_templates
        assert 'phone' in generator.pii_templates

    @patch('app.core.training_data.create_feature_extractor')
    def test_initialize_pii_templates(self, mock_feature_extractor):
        """Test PII template initialization."""
        generator = SyntheticDataGenerator()
        templates = generator._initialize_pii_templates()
        
        # Check that all expected categories are present
        expected_categories = ['person_name', 'organization', 'emails', 'phone', 'address']
        for category in expected_categories:
            assert category in templates
            assert len(templates[category]) > 0
            
        # Check template structure
        person_templates = templates['person_name']
        assert all('template' in t and 'confidence' in t for t in person_templates)

    @patch('app.core.training_data.create_feature_extractor')
    def test_generate_synthetic_examples_default(self, mock_feature_extractor):
        """Test generating synthetic examples with default parameters."""
        # Mock feature extractor
        mock_features = Mock()
        mock_features.to_dict.return_value = {"feature1": 0.8}
        mock_feature_extractor.return_value.extract_all_features.return_value = mock_features
        
        generator = SyntheticDataGenerator()
        
        # Set a seed for reproducible tests
        import random
        random.seed(42)
        
        examples = generator.generate_synthetic_examples(count=50)
        
        # Should generate examples for all categories plus negatives
        assert len(examples) > 0
        
        # Check that we have both positive and negative examples
        positive_count = sum(1 for ex in examples if ex.is_true_positive)
        negative_count = sum(1 for ex in examples if not ex.is_true_positive)
        assert positive_count > 0
        assert negative_count > 0

    @patch('app.core.training_data.create_feature_extractor')
    def test_generate_synthetic_examples_specific_categories(self, mock_feature_extractor):
        """Test generating synthetic examples for specific categories."""
        mock_features = Mock()
        mock_features.to_dict.return_value = {"feature1": 0.8}
        mock_feature_extractor.return_value.extract_all_features.return_value = mock_features
        
        generator = SyntheticDataGenerator()
        
        examples = generator.generate_synthetic_examples(
            count=20, 
            categories=['person_name', 'emails']
        )
        
        # Should have examples for person_name and emails categories
        categories_found = set(ex.category for ex in examples if ex.is_true_positive)
        assert 'person_name' in categories_found or 'emails' in categories_found

    @patch('app.core.training_data.create_feature_extractor')
    def test_fill_template(self, mock_feature_extractor):
        """Test template filling with synthetic data."""
        generator = SyntheticDataGenerator()
        
        # Test with a simple template
        template = "{first_name} {last_name}"
        result = generator._fill_template(template, 'person_name')
        
        # Should replace placeholders with actual names
        assert '{first_name}' not in result
        assert '{last_name}' not in result
        assert len(result.split()) == 2  # Should have two parts

    @patch('app.core.training_data.create_feature_extractor')
    def test_generate_context(self, mock_feature_extractor):
        """Test context generation for synthetic text."""
        generator = SyntheticDataGenerator()
        
        # Test context generation for different categories
        context = generator._generate_context("Jonas Jonaitis", "person_name")
        assert "Jonas Jonaitis" in context
        
        context = generator._generate_context("test@email.com", "emails")
        assert "test@email.com" in context

    @patch('app.core.training_data.create_feature_extractor')
    def test_generate_negative_examples(self, mock_feature_extractor):
        """Test generation of negative (non-PII) examples."""
        mock_features = Mock()
        mock_features.to_dict.return_value = {"feature1": 0.2}
        mock_feature_extractor.return_value.extract_all_features.return_value = mock_features
        
        generator = SyntheticDataGenerator()
        
        negative_examples = generator._generate_negative_examples(10)
        
        assert len(negative_examples) == 10
        for example in negative_examples:
            assert example.is_true_positive == False
            assert example.confidence_score < 0.5
            assert example.category == 'non_pii'


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

    @patch('app.core.training_data.SyntheticDataGenerator')
    def test_collect_all_training_data_with_synthetic(self, mock_generator_class, training_data_collector):
        """Test collecting data including synthetic examples."""
        # Mock the synthetic generator
        mock_generator = Mock()
        mock_synthetic_examples = [
            create_dummy_example("Synthetic1", "SYNTH_CAT", True, "synthetic"),
            create_dummy_example("Synthetic2", "SYNTH_CAT", False, "synthetic")
        ]
        mock_generator.generate_synthetic_examples.return_value = mock_synthetic_examples
        mock_generator_class.return_value = mock_generator
        
        # Create a new collector to use the mocked generator
        collector = TrainingDataCollector()
        collector.storage = training_data_collector.storage
        
        examples = collector.collect_all_training_data(include_synthetic=True, synthetic_count=2)
        
        assert len(examples) == 2  # Only synthetic since storage is empty
        mock_generator.generate_synthetic_examples.assert_called_once_with(count=2)

    def test_get_balanced_training_set_empty(self, training_data_collector: TrainingDataCollector):
        """Test getting a balanced set when no data is available."""
        examples, stats = training_data_collector.get_balanced_training_set()
        assert len(examples) == 0
        assert stats is None

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
        balanced_examples, stats = training_data_collector.get_balanced_training_set(max_samples=6, balance_ratio=0.5)
        
        # With max_samples=6 and ratio=0.5, it should aim for 3 positive and 3 negative.
        # It has 10 pos and 3 neg available.
        # It should take all 3 negatives, and 3 positives.
        
        num_pos = sum(1 for ex in balanced_examples if ex.is_true_positive)
        num_neg = sum(1 for ex in balanced_examples if not ex.is_true_positive)

        assert len(balanced_examples) == 6
        assert num_pos == 3
        assert num_neg == 3

        assert stats.total_samples == 6
        assert stats.positive_samples == 3
        assert stats.negative_samples == 3

    def test_validate_training_data_empty(self, training_data_collector: TrainingDataCollector):
        """Test validation with empty training data."""
        report = training_data_collector.validate_training_data([])
        
        assert report['status'] == 'error'
        assert 'No training examples' in report['message']

    def test_validate_training_data_good_quality(self, training_data_collector: TrainingDataCollector):
        """Test validation with good quality training data."""
        examples = []
        # Create balanced, diverse dataset
        for i in range(60):  # 60 positive
            examples.append(create_dummy_example(f"PII {i}", f"CAT{i % 5}", True))
        for i in range(40):  # 40 negative  
            examples.append(create_dummy_example(f"Non-PII {i}", "OTHER", False))
        
        report = training_data_collector.validate_training_data(examples)
        
        assert report['status'] == 'good'
        assert report['total_examples'] == 100
        assert report['positive_examples'] == 60
        assert report['negative_examples'] == 40
        assert len(report['categories']) >= 5  # At least 5 categories
        assert len(report['quality_issues']) == 0

    def test_validate_training_data_quality_issues(self, training_data_collector: TrainingDataCollector):
        """Test validation with quality issues."""
        # Create small, imbalanced dataset
        examples = []
        for i in range(8):  # Only 8 positive examples
            examples.append(create_dummy_example(f"PII {i}", "NAME", True))
        for i in range(2):  # Only 2 negative examples
            examples.append(create_dummy_example(f"Non-PII {i}", "NAME", False))
        
        report = training_data_collector.validate_training_data(examples)
        
        assert report['status'] == 'warning'
        assert len(report['quality_issues']) > 0
        assert any('imbalance' in issue.lower() for issue in report['quality_issues'])
        assert any('insufficient' in issue.lower() for issue in report['quality_issues'])
        assert any('diversity' in issue.lower() for issue in report['quality_issues'])


@pytest.mark.unit
class TestFactoryFunction:
    """Tests for the create_training_data_collector factory function."""

    def test_create_training_data_collector_default(self):
        """Test creating collector with default config."""
        collector = create_training_data_collector()
        
        assert isinstance(collector, TrainingDataCollector)
        assert collector.storage is not None
        assert collector.synthetic_generator is not None
        assert collector.priority2_collector is not None

    def test_create_training_data_collector_custom_config(self, temp_db_dir):
        """Test creating collector with custom config."""
        custom_config = {
            'db_path': str(temp_db_dir),
            'synthetic_count': 1000
        }
        
        collector = create_training_data_collector(custom_config)
        
        try:
            assert isinstance(collector, TrainingDataCollector)
            assert collector.config == custom_config
        finally:
            # Ensure database connection is properly closed
            if hasattr(collector, 'storage') and collector.storage:
                collector.storage.close()
            # Force garbage collection to release file handles on Windows
            import gc
            gc.collect()


@pytest.mark.integration
class TestIntegrationWorkflow:
    """Integration tests for complete training data workflow."""

    def test_end_to_end_workflow(self, temp_db_dir):
        """Test complete training data collection and processing workflow."""
        # Create collector with temporary storage
        config = {'db_path': str(temp_db_dir)}
        collector = create_training_data_collector(config)
        
        # Step 1: Collect some initial data (mock detection results)
        detection_results = [
            {
                'text': 'Jonas Jonaitis',
                'category': 'person_name', 
                'context': 'Sutartį pasirašė Jonas Jonaitis',
                'confidence': 0.9,
                'document_type': 'contract'
            }
        ]
        
        # Mock the feature extractor for this test
        with patch.object(collector.priority2_collector, 'feature_extractor') as mock_extractor:
            mock_features = Mock()
            mock_features.to_dict.return_value = {"feature1": 0.8}
            mock_extractor.extract_all_features.return_value = mock_features
            
            examples_from_detection = collector.priority2_collector.collect_from_detection_results(detection_results)
            collector.storage.save_training_examples(examples_from_detection, 'priority2')
        
        # Step 2: Collect all training data including synthetic
        with patch.object(collector.synthetic_generator, 'feature_extractor') as mock_synth_extractor:
            mock_features = Mock()
            mock_features.to_dict.return_value = {"feature1": 0.7}
            mock_synth_extractor.extract_all_features.return_value = mock_features
            
            all_examples = collector.collect_all_training_data(include_synthetic=True, synthetic_count=10)
        
        # Step 3: Get balanced training set
        balanced_examples, stats = collector.get_balanced_training_set(max_samples=50)
        
        # Step 4: Validate training data
        validation_report = collector.validate_training_data(balanced_examples)
        
        # Assertions
        assert len(all_examples) > 10  # Should have detection + synthetic examples
        assert len(balanced_examples) > 0
        assert stats is not None
        assert validation_report['status'] in ['good', 'warning']
        
        # Clean up
        collector.storage.close() 