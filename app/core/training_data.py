"""
Training Data Collection and Management for Priority 3 ML Implementation

This module collects, validates, and manages training data for ML confidence scoring
from existing Priority 2 results, user feedback, and synthetic examples.
"""

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import random
import numpy as np
import pandas as pd

# Import existing components
from app.core.data_models import TrainingExample
from app.core.feature_engineering import FeatureExtractor, create_feature_extractor
from app.core.context_analyzer import ContextualValidator, DetectionContext
from app.core.config_manager import get_config
from app.core.logging import get_logger

training_logger = get_logger(__name__)


@dataclass
class DatasetStats:
    """Statistics about the training dataset."""
    total_samples: int
    positive_samples: int
    negative_samples: int
    categories: Dict[str, int]
    document_types: Dict[str, int]
    date_range: Tuple[datetime, datetime]
    quality_score: float
    
    def get_class_balance(self) -> float:
        """Get class balance ratio (positive/total)."""
        return self.positive_samples / self.total_samples if self.total_samples > 0 else 0.0


class TrainingDataStorage:
    """Manages storage and retrieval of training data."""
    
    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "training_data.db"
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._init_database()
        
        # File paths for different data types
        self.examples_file = self.data_dir / "training_examples.jsonl"
        self.synthetic_file = self.data_dir / "synthetic_examples.jsonl"
        
    def _connect(self):
        """Establish the database connection."""
        if self.conn is None:
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                training_logger.debug(f"Database connection opened for {self.db_path}")
            except sqlite3.Error as e:
                training_logger.error(f"Error connecting to database {self.db_path}: {e}")
                raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            training_logger.debug(f"Database connection closed for {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database for training data."""
        if not self.conn:
            self._connect()
        try:
            with self.conn:
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS training_examples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        detection_text TEXT NOT NULL,
                        category TEXT NOT NULL,
                        context TEXT,
                        confidence_score REAL,
                        is_true_positive BOOLEAN,
                        document_type TEXT,
                        language TEXT,
                        features_json TEXT,
                        metadata_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data_source TEXT  -- 'priority2', 'feedback', 'synthetic'
                    )
                ''')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_examples_category ON training_examples(category)')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_examples_source ON training_examples(data_source)')
        except sqlite3.Error as e:
            training_logger.error(f"Error initializing database: {e}")

    def save_training_examples(self, examples: List[TrainingExample], source: str = 'unknown'):
        """Save training examples to database."""
        if not self.conn:
            self._connect()
        try:
            with self.conn:
                for example in examples:
                    self.conn.execute('''
                        INSERT INTO training_examples 
                        (detection_text, category, context, confidence_score, is_true_positive,
                         document_type, features_json, metadata_json, data_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        example.detection_text,
                        example.category,
                        example.context,
                        example.confidence_score,
                        example.is_true_positive,
                        example.document_type,
                        json.dumps(example.features),
                        json.dumps(example.metadata) if example.metadata else None,
                        source
                    ))
            training_logger.debug(f"Saved {len(examples)} training examples from {source}")
        except sqlite3.Error as e:
            training_logger.error(f"Error saving training examples: {e}")

    def load_training_examples(self, source: Optional[str] = None, 
                             category: Optional[str] = None,
                             limit: Optional[int] = None) -> List[TrainingExample]:
        """Load training examples from database."""
        if not self.conn:
            self._connect()
        
        query = "SELECT * FROM training_examples WHERE 1=1"
        params = []
        
        if source:
            query += " AND data_source = ?"
            params.append(source)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY created_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        examples = []
        try:
            with self.conn:
                self.conn.row_factory = sqlite3.Row
                cursor = self.conn.execute(query, params)
                for row in cursor.fetchall():
                    features = json.loads(row['features_json']) if row['features_json'] else {}
                    metadata = json.loads(row['metadata_json']) if row['metadata_json'] else None
                    example = TrainingExample(
                        detection_text=row['detection_text'],
                        category=row['category'],
                        context=row['context'] or '',
                        features=features,
                        confidence_score=row['confidence_score'],
                        is_true_positive=bool(row['is_true_positive']),
                        document_type=row['document_type'],
                        metadata=metadata
                    )
                    examples.append(example)
        except sqlite3.Error as e:
            training_logger.error(f"Error loading training examples: {e}")
        
        return examples
    
    def get_dataset_stats(self) -> Optional[DatasetStats]:
        """Get statistics about the training dataset."""
        if not self.conn:
            self._connect()
        
        try:
            with self.conn:
                # Total counts
                total_count = self.conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
                if total_count == 0:
                    return None
                
                positive_count = self.conn.execute(
                    "SELECT COUNT(*) FROM training_examples WHERE is_true_positive = 1"
                ).fetchone()[0]
                
                # Category distribution
                category_counts = dict(self.conn.execute(
                    "SELECT category, COUNT(*) FROM training_examples GROUP BY category"
                ).fetchall())
                
                # Document type distribution
                doc_type_counts = dict(self.conn.execute(
                    "SELECT document_type, COUNT(*) FROM training_examples WHERE document_type IS NOT NULL GROUP BY document_type"
                ).fetchall())
                
                # Date range
                min_date_str, max_date_str = self.conn.execute(
                    "SELECT MIN(created_at), MAX(created_at) FROM training_examples"
                ).fetchone()
                
                min_date = datetime.fromisoformat(min_date_str) if min_date_str else datetime.now()
                max_date = datetime.fromisoformat(max_date_str) if max_date_str else datetime.now()
                
                quality_score = min(1.0, total_count / 1000)
                
                return DatasetStats(
                    total_samples=total_count,
                    positive_samples=positive_count,
                    negative_samples=total_count - positive_count,
                    categories=category_counts,
                    document_types=doc_type_counts,
                    date_range=(min_date, max_date),
                    quality_score=quality_score
                )
        except sqlite3.Error as e:
            training_logger.error(f"Error getting dataset stats: {e}")
            return None


class Priority2DataCollector:
    """Collects training data from existing Priority 2 results."""
    
    def __init__(self):
        self.feature_extractor = create_feature_extractor()
        self.contextual_validator = ContextualValidator()
        
    def collect_from_priority2_logs(self, log_dir: str = "logs") -> List[TrainingExample]:
        """
        Collect training data from Priority 2 processing logs.
        
        Args:
            log_dir: Directory containing log files
            
        Returns:
            List of training examples
        """
        examples = []
        log_path = Path(log_dir)
        
        if not log_path.exists():
            training_logger.warning(f"Log directory not found: {log_dir}")
            return examples
        
        # Look for processing log files
        for log_file in log_path.glob("*processing*.log"):
            try:
                file_examples = self._parse_log_file(log_file)
                examples.extend(file_examples)
                training_logger.debug(f"Collected {len(file_examples)} examples from {log_file}")
            except Exception as e:
                training_logger.error(f"Failed to parse log file {log_file}: {e}")
        
        training_logger.info(f"Collected {len(examples)} examples from Priority 2 logs")
        return examples
    
    def _parse_log_file(self, log_file: Path) -> List[TrainingExample]:
        """Parse individual log file for training examples."""
        examples = []
        
        # This is a simplified parser - in practice, you'd parse actual log format
        # For now, return empty list as we don't have actual log structure
        training_logger.debug(f"Log file parsing not yet implemented for {log_file}")
        return examples
    
    def collect_from_detection_results(self, detection_results: List[Dict]) -> List[TrainingExample]:
        """
        Convert detection results to training examples.
        
        Args:
            detection_results: List of detection result dictionaries
            
        Returns:
            List of training examples
        """
        examples = []
        
        for result in detection_results:
            try:
                # Extract features
                feature_set = self.feature_extractor.extract_all_features(
                    detection_text=result.get('text', ''),
                    category=result.get('category', 'unknown'),
                    context=result.get('context', ''),
                    position=result.get('position', -1),
                    full_text=result.get('full_text'),
                    document_type=result.get('document_type'),
                    language=result.get('language', 'lt')
                )
                
                # Determine ground truth (simplified heuristic)
                confidence = result.get('confidence', 0.5)
                is_true_positive = confidence > 0.7  # Threshold-based classification
                
                example = TrainingExample(
                    detection_text=result.get('text', ''),
                    category=result.get('category', 'unknown'),
                    context=result.get('context', ''),
                    features=feature_set.to_dict(),
                    confidence_score=confidence,
                    is_true_positive=is_true_positive,
                    document_type=result.get('document_type'),
                    metadata={
                        'source': 'priority2_detection',
                        'original_confidence': confidence,
                        'position': result.get('position', -1)
                    }
                )
                
                examples.append(example)
                
            except Exception as e:
                training_logger.error(f"Failed to process detection result: {e}")
                continue
        
        training_logger.info(f"Created {len(examples)} training examples from detection results")
        return examples


class SyntheticDataGenerator:
    """Generates synthetic training examples for data augmentation."""
    
    def __init__(self):
        self.feature_extractor = create_feature_extractor()
        self.pii_templates = self._initialize_pii_templates()
        
    def _initialize_pii_templates(self) -> Dict[str, List[Dict]]:
        """Initialize templates for generating synthetic PII data."""
        return {
            'person_name': [
                {'template': '{first_name} {last_name}', 'confidence': 0.9},
                {'template': 'Mr. {last_name}', 'confidence': 0.8},
                {'template': 'Dr. {first_name} {last_name}', 'confidence': 0.95},
            ],
            'organization': [
                {'template': '{company_name} UAB', 'confidence': 0.85},
                {'template': '{company_name} Ltd.', 'confidence': 0.8},
                {'template': '{company_name} Corporation', 'confidence': 0.75},
            ],
            'email': [
                {'template': '{name}@{domain}.com', 'confidence': 0.95},
                {'template': '{name}.{surname}@{company}.lt', 'confidence': 0.9},
            ],
            'phone': [
                {'template': '+370 6{phone_digits}', 'confidence': 0.9},
                {'template': '8{phone_digits}', 'confidence': 0.85},
            ],
            'address': [
                {'template': '{street_name} g. {number}', 'confidence': 0.8},
                {'template': '{number} {street_name} street', 'confidence': 0.75},
            ]
        }
    
    def generate_synthetic_examples(self, count: int = 100, 
                                  categories: Optional[List[str]] = None) -> List[TrainingExample]:
        """
        Generate synthetic training examples.
        
        Args:
            count: Number of examples to generate
            categories: PII categories to generate (None for all)
            
        Returns:
            List of synthetic training examples
        """
        if categories is None:
            categories = list(self.pii_templates.keys())
        
        examples = []
        examples_per_category = count // len(categories)
        
        for category in categories:
            category_examples = self._generate_category_examples(
                category, examples_per_category
            )
            examples.extend(category_examples)
        
        # Generate some negative examples (non-PII)
        negative_examples = self._generate_negative_examples(count // 10)
        examples.extend(negative_examples)
        
        # Shuffle the examples
        random.shuffle(examples)
        
        training_logger.info(f"Generated {len(examples)} synthetic examples")
        return examples
    
    def _generate_category_examples(self, category: str, count: int) -> List[TrainingExample]:
        """Generate examples for a specific PII category."""
        examples = []
        templates = self.pii_templates.get(category, [])
        
        if not templates:
            return examples
        
        for i in range(count):
            template_data = random.choice(templates)
            template = template_data['template']
            base_confidence = template_data['confidence']
            
            # Generate synthetic text
            synthetic_text = self._fill_template(template, category)
            
            # Generate context
            context = self._generate_context(synthetic_text, category)
            
            # Extract features
            try:
                feature_set = self.feature_extractor.extract_all_features(
                    detection_text=synthetic_text,
                    category=category,
                    context=context,
                    document_type='synthetic',
                    language='lt'
                )
                
                # Add some noise to confidence
                confidence = base_confidence + random.uniform(-0.1, 0.1)
                confidence = max(0.1, min(0.99, confidence))
                
                example = TrainingExample(
                    detection_text=synthetic_text,
                    category=category,
                    context=context,
                    features=feature_set.to_dict(),
                    confidence_score=confidence,
                    is_true_positive=True,  # All synthetic examples are positive
                    document_type='synthetic',
                    metadata={
                        'source': 'synthetic',
                        'template': template,
                        'generation_index': i
                    }
                )
                
                examples.append(example)
                
            except Exception as e:
                training_logger.error(f"Failed to generate synthetic example: {e}")
                continue
        
        return examples
    
    def _fill_template(self, template: str, category: str) -> str:
        """Fill template with synthetic data."""
        # Simple synthetic data generation
        replacements = {
            'first_name': random.choice(['Jonas', 'Petras', 'Antanas', 'Vytautas', 'Mindaugas']),
            'last_name': random.choice(['Jonaitis', 'Petraitis', 'Antanaitis', 'Kazlauskas', 'Vaičiūnas']),
            'company_name': random.choice(['Vilnius', 'Kaunas', 'Klaipėda', 'Baltic', 'Lietuva']),
            'domain': random.choice(['gmail', 'yahoo', 'hotmail', 'company']),
            'name': random.choice(['jonas', 'petras', 'ana', 'rita']),
            'surname': random.choice(['jonaitis', 'petraitis', 'kazlauskas']),
            'company': random.choice(['vilnius', 'kaunas', 'baltic']),
            'phone_digits': ''.join([str(random.randint(0, 9)) for _ in range(7)]),
            'street_name': random.choice(['Gedimino', 'Vilniaus', 'Kauno', 'Laisvės', 'Taikos']),
            'number': str(random.randint(1, 200))
        }
        
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(f'{{{placeholder}}}', value)
        
        return result
    
    def _generate_context(self, text: str, category: str) -> str:
        """Generate realistic context for synthetic text."""
        context_templates = {
            'person_name': [
                f"Vardas: {text}",
                f"Kontaktinis asmuo: {text}",
                f"Sutartį pasirašė {text} dėl",
                f"Pacientas {text} kreipėsi"
            ],
            'organization': [
                f"Įmonė: {text}",
                f"Darbdavys: {text}",
                f"Partneris {text} teikia",
                f"Bendrovė {text} garantuoja"
            ],
            'email': [
                f"El. paštas: {text}",
                f"Kreiptis: {text}",
                f"Atsiųsti į {text}",
                f"Kontaktai: {text}"
            ],
            'phone': [
                f"Telefonas: {text}",
                f"Mob. tel.: {text}",
                f"Skambinti {text}",
                f"Kontaktinis tel.: {text}"
            ],
            'address': [
                f"Adresas: {text}",
                f"Gyvenamoji vieta: {text}",
                f"Registracijos adresas {text}",
                f"Pristatymas į {text}"
            ]
        }
        
        templates = context_templates.get(category, [f"Duomenys: {text}"])
        return random.choice(templates)
    
    def _generate_negative_examples(self, count: int) -> List[TrainingExample]:
        """Generate negative examples (non-PII text)."""
        examples = []
        
        negative_texts = [
            "Skyrius 1",
            "Puslapis 5",
            "2024 metai",
            "Dokumentas Nr.",
            "Priedas A",
            "Lentelė 3.1",
            "Pastaba:",
            "Žiūrėti taip pat",
            "Literatūros sąrašas",
            "Turinys"
        ]
        
        for i in range(count):
            text = random.choice(negative_texts)
            context = f"Dokumento {text} aprašymas ir papildoma informacija."
            
            try:
                feature_set = self.feature_extractor.extract_all_features(
                    detection_text=text,
                    category='non_pii',
                    context=context,
                    document_type='synthetic',
                    language='lt'
                )
                
                example = TrainingExample(
                    detection_text=text,
                    category='non_pii',
                    context=context,
                    features=feature_set.to_dict(),
                    confidence_score=random.uniform(0.1, 0.3),  # Low confidence for non-PII
                    is_true_positive=False,
                    document_type='synthetic',
                    metadata={
                        'source': 'synthetic_negative',
                        'generation_index': i
                    }
                )
                
                examples.append(example)
                
            except Exception as e:
                training_logger.error(f"Failed to generate negative example: {e}")
                continue
        
        return examples


class TrainingDataCollector:
    """Main training data collection coordinator."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config().get('training_data', {})
        self.storage = TrainingDataStorage()
        self.priority2_collector = Priority2DataCollector()
        self.synthetic_generator = SyntheticDataGenerator()
        
    def collect_all_training_data(self, include_synthetic: bool = True,
                                 synthetic_count: int = 500) -> List[TrainingExample]:
        """
        Collect training data from all sources.
        TrainingExamples from 'user_feedback' source are expected to be populated 
        by the UserFeedbackSystem.
        
        Args:
            include_synthetic: Whether to include synthetic examples
            synthetic_count: Number of synthetic examples to generate
            
        Returns:
            Combined list of training examples
        """
        all_examples = []
        
        # Collect from existing sources (including those from feedback saved by UserFeedbackProcessor)
        existing_examples = self.storage.load_training_examples()
        all_examples.extend(existing_examples)
        training_logger.info(f"Loaded {len(existing_examples)} existing examples (may include 'feedback' source)")
        
        # User feedback is now processed by UserFeedbackSystem and directly saved as TrainingExamples.
        # Thus, direct conversion here is removed.
        
        # Generate synthetic examples if requested
        if include_synthetic:
            synthetic_examples = self.synthetic_generator.generate_synthetic_examples(
                count=synthetic_count
            )
            all_examples.extend(synthetic_examples)
            
            # Save synthetic examples
            self.storage.save_training_examples(synthetic_examples, 'synthetic')
            training_logger.info(f"Generated {len(synthetic_examples)} synthetic examples")
        
        training_logger.info(f"Total training data collected: {len(all_examples)} examples")
        return all_examples
    
    def get_balanced_training_set(self, max_samples: int = 10000,
                                 balance_ratio: float = 0.5) -> Tuple[List[TrainingExample], DatasetStats]:
        """
        Get a balanced training set with specified positive/negative ratio.
        
        Args:
            max_samples: Maximum number of samples to return
            balance_ratio: Desired ratio of positive samples
            
        Returns:
            Tuple of (balanced examples, dataset statistics)
        """
        all_examples = self.storage.load_training_examples()
        
        # Separate positive and negative examples
        positive_examples = [ex for ex in all_examples if ex.is_true_positive]
        negative_examples = [ex for ex in all_examples if not ex.is_true_positive]
        
        # Calculate target counts
        target_positive = int(max_samples * balance_ratio)
        target_negative = max_samples - target_positive
        
        # Sample examples
        sampled_positive = random.sample(
            positive_examples, 
            min(target_positive, len(positive_examples))
        )
        sampled_negative = random.sample(
            negative_examples,
            min(target_negative, len(negative_examples))
        )
        
        balanced_examples = sampled_positive + sampled_negative
        random.shuffle(balanced_examples)
        
        # Calculate statistics
        stats = DatasetStats(
            total_samples=len(balanced_examples),
            positive_samples=len(sampled_positive),
            negative_samples=len(sampled_negative),
            categories={},
            document_types={},
            date_range=(datetime.now(), datetime.now()),
            quality_score=1.0
        )
        
        training_logger.info(f"Created balanced training set: {len(balanced_examples)} examples")
        return balanced_examples, stats
    
    def validate_training_data(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """
        Validate quality of training data.
        
        Args:
            examples: Training examples to validate
            
        Returns:
            Validation report
        """
        if not examples:
            return {'status': 'error', 'message': 'No training examples provided'}
        
        report = {
            'total_examples': len(examples),
            'positive_examples': sum(1 for ex in examples if ex.is_true_positive),
            'negative_examples': sum(1 for ex in examples if not ex.is_true_positive),
            'categories': {},
            'quality_issues': [],
            'recommendations': []
        }
        
        # Category distribution
        category_counts = {}
        for example in examples:
            category_counts[example.category] = category_counts.get(example.category, 0) + 1
        report['categories'] = category_counts
        
        # Quality checks
        class_balance = report['positive_examples'] / report['total_examples']
        if class_balance < 0.3 or class_balance > 0.7:
            report['quality_issues'].append(f"Class imbalance: {class_balance:.2f}")
            report['recommendations'].append("Consider balancing positive/negative examples")
        
        if report['total_examples'] < 100:
            report['quality_issues'].append("Insufficient training data")
            report['recommendations'].append("Collect more training examples")
        
        if len(category_counts) < 3:
            report['quality_issues'].append("Limited category diversity")
            report['recommendations'].append("Include more PII categories")
        
        report['status'] = 'good' if not report['quality_issues'] else 'warning'
        return report


# Factory function for easy integration
def create_training_data_collector(config: Optional[Dict] = None) -> TrainingDataCollector:
    """Create and return training data collector instance."""
    return TrainingDataCollector(config) 