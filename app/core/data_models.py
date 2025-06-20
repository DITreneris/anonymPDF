from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

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
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AdaptivePattern:
    """Represents an adaptive pattern for detecting PII."""
    pattern: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_validated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the pattern to a dictionary for database storage."""
        data = self.__dict__.copy()
        data['created_at'] = self.created_at.isoformat()
        if self.last_validated_at:
            data['last_validated_at'] = self.last_validated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptivePattern":
        """Deserializes a dictionary to a pattern object."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_validated_at'):
            data['last_validated_at'] = datetime.fromisoformat(data['last_validated_at'])
        return cls(**data) 