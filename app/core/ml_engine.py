"""
Machine Learning Engine for Priority 3 Implementation

This module provides ML-powered confidence scoring, adaptive pattern learning,
and document type classification to enhance PII detection accuracy.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import joblib
import re

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Import existing components
from app.core.context_analyzer import ContextualValidator, DetectionContext, ConfidenceLevel
from app.core.config_manager import get_config
from app.core.logging import get_logger
if TYPE_CHECKING:
    from app.core.adaptive.coordinator import AdaptiveLearningCoordinator
from app.core.analytics_engine import QualityAnalyzer
from app.core.feature_engineering import FeatureExtractor
from .data_models import MLModel, MLPrediction, TrainingExample

ml_logger = get_logger(__name__)


class MLModelManager:
    """Manages ML model lifecycle - training, saving, loading, versioning."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        
    def save_model(self, model: Any, model_name: str, version: str = None) -> str:
        """Save ML model with versioning."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.model_dir / f"{model_name}_v{version}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'feature_names': self.feature_names
        }
        
        metadata_path = self.model_dir / f"{model_name}_v{version}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        ml_logger.info(f"Model saved: {model_path}")
        return str(model_path)
    
    def load_model(self, model_name: str, version: str = None) -> Optional[Any]:
        """Load ML model by name and version."""
        if version is None:
            # Load latest version
            pattern = f"{model_name}_v*.joblib"
            model_files = list(self.model_dir.glob(pattern))
            if not model_files:
                return None
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        else:
            model_path = self.model_dir / f"{model_name}_v{version}.joblib"
        
        if not model_path.exists():
            ml_logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            model = joblib.load(model_path)
            ml_logger.info(f"Model loaded: {model_path}")
            return model
        except Exception as e:
            ml_logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def list_models(self) -> List[Dict]:
        """List all available models with metadata."""
        models = []
        for model_file in self.model_dir.glob("*.joblib"):
            if "metadata" in model_file.name:
                continue
            
            # Try to load metadata
            metadata_file = model_file.with_suffix('.json').with_name(
                model_file.name.replace('.joblib', '_metadata.json')
            )
            
            model_info = {
                'name': model_file.stem,
                'path': str(model_file),
                'size_mb': model_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime)
            }
            
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    model_info.update(metadata)
                except Exception as e:
                    ml_logger.warning(f"Failed to load metadata for {model_file}: {e}")
            
            models.append(model_info)
        
        return sorted(models, key=lambda x: x.get('modified', datetime.min), reverse=True)


class MLConfidenceScorer:
    """ML-powered confidence scoring for PII detections."""
    
    def __init__(self, config: Optional[Dict] = None, coordinator: Optional["AdaptiveLearningCoordinator"] = None):
        self.config = config or get_config().get('ml_engine', {})
        self.model_manager = MLModelManager()
        self.contextual_validator = ContextualValidator()  # Fallback to Priority 2
        
        # Integration with Adaptive Learning System
        self.coordinator = coordinator
        
        # Initialize model and scaler
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.feature_names = []
        self.model_version = "none"
        
        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        
        # Load existing model if available
        self._load_or_initialize_model()
    
    def _load_or_initialize_model(self):
        """Load existing model or initialize a new one."""
        try:
            # Try to load existing model
            self.model = self.model_manager.load_model("confidence_scorer")
            self.scaler = self.model_manager.load_model("confidence_scaler")
            self.calibrator = self.model_manager.load_model("confidence_calibrator")
            
            if self.model is not None:
                ml_logger.info("Loaded existing ML confidence model")
                self.model_version = "loaded"
                return
        except Exception as e:
            ml_logger.warning(f"Failed to load existing model: {e}")
        
        # Initialize new model if loading failed
        self._initialize_new_model()
    
    def _initialize_new_model(self):
        """Initialize a new ML model with default parameters."""
        model_type = self.config.get('model_type', 'xgboost')
        model_params = self.config.get('model_params', {})
        
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 6),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=model_params.get('random_state', 42),
                eval_metric='logloss'
            )
        else:  # Default to Random Forest
            self.model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                random_state=model_params.get('random_state', 42)
            )
        
        self.scaler = StandardScaler()
        self.calibrator = CalibratedClassifierCV(
            self.model, 
            method=self.config.get('confidence_calibration', {}).get('method', 'isotonic'),
            cv=self.config.get('confidence_calibration', {}).get('cv_folds', 3)
        )
        
        ml_logger.info(f"Initialized new {model_type} model")
        self.model_version = "new"
    
    def _apply_adaptive_patterns(self, detection: str) -> Optional[MLPrediction]:
        """
        Check against high-confidence adaptive patterns first.
        If a match is found, return a high-confidence prediction immediately.
        """
        if not self.coordinator:
            return None

        adaptive_patterns = self.coordinator.get_adaptive_patterns()
        if not adaptive_patterns:
            return None

        for pattern in adaptive_patterns:
            try:
                # Assuming pattern['regex'] is the key for the regex string
                if re.search(pattern['regex'], detection):
                    ml_logger.info(f"Detection matched high-confidence adaptive pattern ID: {pattern.get('pattern_id')}")
                    return MLPrediction(
                        confidence=pattern.get('confidence', 0.98), # Assign high confidence
                        probability=pattern.get('confidence', 0.98),
                        features_used=[f"adaptive_pattern:{pattern.get('pattern_id')}"],
                        model_version="adaptive_pattern_override",
                        prediction_time=datetime.now()
                    )
            except re.error as e:
                ml_logger.warning(f"Invalid regex in adaptive pattern {pattern.get('pattern_id')}: {e}")
                continue
        
        return None
    
    def calculate_ml_confidence(self, detection: str, context: str, 
                              features: Dict[str, float], 
                              document_type: Optional[str] = None) -> MLPrediction:
        """
        Calculate confidence using trained ML model.
        
        Args:
            detection: The detected PII text
            context: Surrounding context
            features: Extracted features
            document_type: Type of document
            
        Returns:
            MLPrediction with confidence and metadata
        """
        start_time = datetime.now()
        
        # --- Adaptive Pattern Integration ---
        adaptive_prediction = self._apply_adaptive_patterns(detection)
        if adaptive_prediction:
            return adaptive_prediction
        # ------------------------------------
        
        try:
            # If model is not trained, fall back to Priority 2 confidence
            if not self._is_model_trained():
                fallback_confidence = self.contextual_validator.calculate_confidence(
                    detection, 'unknown', context
                )
                return MLPrediction(
                    confidence=fallback_confidence,
                    probability=fallback_confidence,
                    features_used=['fallback'],
                    model_version='priority2_fallback',
                    prediction_time=start_time
                )
            
            # Prepare features for prediction
            feature_vector = self._prepare_feature_vector(features)
            
            # Scale features
            if self.scaler is not None:
                feature_vector = self.scaler.transform([feature_vector])
            else:
                feature_vector = [feature_vector]
            
            # Get prediction
            if self.calibrator is not None:
                # Use calibrated probabilities for better confidence estimation
                probability = self.calibrator.predict_proba(feature_vector)[0][1]
                confidence = probability
            else:
                # Fallback to raw model prediction
                probability = self.model.predict_proba(feature_vector)[0][1]
                confidence = probability
            
            # Adjust confidence based on document type
            if document_type:
                confidence = self._adjust_confidence_for_document_type(
                    confidence, document_type
                )
            
            prediction = MLPrediction(
                confidence=float(confidence),
                probability=float(probability),
                features_used=self.feature_names,
                model_version=self.model_version,
                prediction_time=start_time
            )
            
            # Update performance tracking
            self.prediction_count += 1
            self.total_prediction_time += (datetime.now() - start_time).total_seconds()
            
            return prediction
            
        except Exception as e:
            ml_logger.error(f"ML confidence calculation failed: {e}")
            # Fallback to Priority 2 confidence
            fallback_confidence = self.contextual_validator.calculate_confidence(
                detection, 'unknown', context
            )
            return MLPrediction(
                confidence=fallback_confidence,
                probability=fallback_confidence,
                features_used=['error_fallback'],
                model_version='priority2_fallback',
                prediction_time=start_time
            )
    
    def _prepare_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Prepare feature vector for model prediction."""
        if not self.feature_names:
            # If feature names not set, use all features
            self.feature_names = sorted(features.keys())
        
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0.0))
        
        return feature_vector
    
    def _adjust_confidence_for_document_type(self, confidence: float, 
                                           document_type: str) -> float:
        """Adjust confidence based on document type."""
        # Document type specific adjustments
        adjustments = {
            'insurance': 1.05,  # Insurance documents tend to have cleaner PII
            'legal': 0.95,      # Legal documents may have complex language
            'financial': 1.02,  # Financial documents usually have structured PII
            'medical': 1.03,    # Medical documents have standardized formats
            'personal': 0.98,   # Personal documents may be less structured
        }
        
        adjustment = adjustments.get(document_type.lower(), 1.0)
        adjusted_confidence = min(1.0, confidence * adjustment)
        
        return adjusted_confidence
    
    def _is_model_trained(self) -> bool:
        """Check if the model has been trained."""
        if self.model is None:
            return False
        
        # Check if model has been fitted
        try:
            # XGBoost models have different attribute names
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_ is not None
            elif hasattr(self.model, '_Booster'):
                return self.model._Booster is not None
            else:
                # Generic check for sklearn models
                return hasattr(self.model, 'tree_') or hasattr(self.model, 'coef_')
        except:
            return False
    
    def train_model(self, training_data: List[TrainingExample]) -> Dict[str, float]:
        """
        Train the ML confidence model.
        
        Args:
            training_data: List of training examples
            
        Returns:
            Training metrics
        """
        if len(training_data) < 10:
            ml_logger.warning("Insufficient training data for ML model")
            return {'error': 'insufficient_data', 'samples': len(training_data)}
        
        ml_logger.info(f"Training ML model with {len(training_data)} samples")
        
        # Prepare training data
        X, y, feature_names = self._prepare_training_data(training_data)
        self.feature_names = feature_names
        
        # Split data
        test_size = self.config.get('training', {}).get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Train calibrator
        self.calibrator = CalibratedClassifierCV(
            self.model, 
            method=self.config.get('confidence_calibration', {}).get('method', 'isotonic'),
            cv=self.config.get('confidence_calibration', {}).get('cv_folds', 3)
        )
        self.calibrator.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = 0.5
        
        accuracy = np.mean(y_pred == y_test)
        
        # Cross-validation
        cv_folds = self.config.get('training', {}).get('cv_folds', 5)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)
        
        metrics = {
            'accuracy': float(accuracy),
            'auc_score': float(auc_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'training_samples': len(training_data),
            'features_count': len(feature_names)
        }
        
        # Save trained model
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_manager.feature_names = feature_names
        self.model_manager.save_model(self.model, "confidence_scorer", self.model_version)
        self.model_manager.save_model(self.scaler, "confidence_scaler", self.model_version)
        self.model_manager.save_model(self.calibrator, "confidence_calibrator", self.model_version)
        
        ml_logger.info(f"Model training completed: {metrics}")
        return metrics
    
    def _prepare_training_data(self, training_data: List[TrainingExample]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for ML model."""
        # Extract features and labels
        feature_dicts = [example.features for example in training_data]
        labels = [1 if example.is_true_positive else 0 for example in training_data]
        
        # Get all unique feature names
        all_features = set()
        for feature_dict in feature_dicts:
            all_features.update(feature_dict.keys())
        
        feature_names = sorted(list(all_features))
        
        # Create feature matrix
        X = []
        for feature_dict in feature_dicts:
            feature_vector = [feature_dict.get(name, 0.0) for name in feature_names]
            X.append(feature_vector)
        
        return np.array(X), np.array(labels), feature_names
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics."""
        if self.prediction_count == 0:
            return {'status': 'no_predictions_made'}
        
        avg_prediction_time = self.total_prediction_time / self.prediction_count
        
        return {
            'model_version': self.model_version,
            'prediction_count': self.prediction_count,
            'avg_prediction_time_ms': avg_prediction_time * 1000,
            'is_trained': self._is_model_trained(),
            'feature_count': len(self.feature_names),
            'model_type': type(self.model).__name__ if self.model else None
        }


# Factory function for easy integration
def create_ml_confidence_scorer(config: Optional[Dict] = None) -> MLConfidenceScorer:
    """Factory function for MLConfidenceScorer."""
    # This factory now needs to be aware of the adaptive learning system and analytics.
    from app.core.adaptive.coordinator import AdaptiveLearningCoordinator
    from app.core.analytics_engine import QualityAnalyzer
    
    ml_logger.info("Instantiating QualityAnalyzer for MLConfidenceScorer.")
    quality_analyzer = QualityAnalyzer()

    ml_logger.info("Instantiating AdaptiveLearningCoordinator for MLConfidenceScorer.")
    adaptive_coordinator = AdaptiveLearningCoordinator(quality_analyzer=quality_analyzer)
    
    return MLConfidenceScorer(config, coordinator=adaptive_coordinator) 