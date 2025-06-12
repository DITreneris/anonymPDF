"""
Document Classifier for Adaptive Learning

This module classifies incoming documents into predefined categories to allow
for context-specific PII detection rules and processing pipelines.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.core.config_manager import get_config
from .processing_rules import ProcessingRuleManager, RuleSet

logger = get_logger("adaptive_learning.doc_classifier")

@dataclass
class DocumentClassification:
    """Represents the result of a document classification."""
    doc_type: str
    confidence: float
    model_version: str = "keyword_v1.0"

class DocumentClassifier:
    """
    Classifies documents into predefined types to apply specific
    processing rules.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get('adaptive_learning', {})
        self.rule_manager = ProcessingRuleManager(config=self.config)
        self.default_doc_type = self.config.get('default_doc_type', "general_text")
        self.classification_threshold = self.config.get('classification_threshold', 0.1)
        logger.info("DocumentClassifier initialized.")

    def classify_document(self, text: str) -> DocumentClassification:
        """
        Classifies the document based on keywords defined in the rule sets.
        Calculates a confidence score for the classification.
        """
        scores = {}
        text_lower = text.lower()
        
        all_rulesets = self.rule_manager.get_all_rulesets()

        for ruleset in all_rulesets:
            if not ruleset.keywords:
                continue

            # Calculate score based on keyword matches
            score = sum(1 for keyword in ruleset.keywords if keyword in text_lower)
            
            if score > 0:
                # Normalize the score to get a preliminary confidence
                confidence = score / len(ruleset.keywords)
                scores[ruleset.doc_type] = confidence

        if not scores:
            logger.debug("No keywords matched, defaulting to general_text.")
            return DocumentClassification(doc_type=self.default_doc_type, confidence=0.1)

        # Determine the best match
        best_doc_type = max(scores, key=scores.get)
        best_confidence = scores[best_doc_type]

        if best_confidence < self.classification_threshold:
            logger.debug(f"Best confidence {best_confidence:.2f} is below threshold, defaulting.")
            return DocumentClassification(doc_type=self.default_doc_type, confidence=best_confidence)

        logger.info(f"Classified document as '{best_doc_type}' with confidence {best_confidence:.2f}")
        return DocumentClassification(doc_type=best_doc_type, confidence=best_confidence)

    def get_processing_rules(self, doc_type: str) -> Optional[RuleSet]:
        """
        Retrieves the set of processing rules for a given document type.
        """
        logger.info(f"Fetching processing rules for document type: '{doc_type}'")
        return self.rule_manager.get_rules_for_doc_type(doc_type)

# Factory function for easy integration
def create_doc_classifier(config: Optional[Dict[str, Any]] = None) -> DocumentClassifier:
    """Creates and returns a DocumentClassifier instance."""
    return DocumentClassifier(config) 