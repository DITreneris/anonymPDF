from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

@dataclass
class MLModel:
    """Represents a trained machine learning model and its metadata."""
    model_path: str
    version: str
    metadata: Dict[str, Any]

@dataclass
class MLPrediction:
    """Represents the output of an ML confidence score prediction."""
    pii_category: str
    confidence: float
    model_version: str
    features_used: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the dataclass to a dictionary."""
        return asdict(self)

@dataclass
class TrainingExample:
    """Represents a single training example for the ML model."""
    detection_text: str
    category: str
    context: str
    features: Dict[str, Any]
    confidence_score: float
    is_true_positive: bool
    document_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None 