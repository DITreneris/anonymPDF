"""
Processing Rules Manager for Adaptive Learning.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from app.core.logging import get_logger
import yaml
from pathlib import Path
from app.core.config_manager import get_config

logger = get_logger("adaptive_learning.processing_rules")


@dataclass
class Rule:
    """A single processing rule."""
    action: str  # e.g., 'REDACT', 'WARN', 'SKIP_CATEGORY'
    target_category: str
    confidence_threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RuleSet:
    """A set of rules for a specific document type."""
    doc_type: str
    keywords: List[str] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)

class ProcessingRuleManager:
    """
    Manages the creation, storage, and retrieval of processing rules
    that are tailored to specific document types.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get('adaptive_learning', {})
        self.rules_config_path = Path(self.config.get('rules_config_path', 'config/processing_rules.yaml'))
        self._rule_store: Dict[str, RuleSet] = self._load_rules_from_config()
        logger.info(f"ProcessingRuleManager initialized with {len(self._rule_store)} rule sets.")

    def _load_rules_from_config(self) -> Dict[str, RuleSet]:
        """Loads processing rules from the YAML configuration file."""
        rule_store = {}
        if not self.rules_config_path.exists():
            logger.error(f"Rules configuration file not found at: {self.rules_config_path}")
            return rule_store

        try:
            with open(self.rules_config_path, 'r', encoding='utf-8') as f:
                rules_data = yaml.safe_load(f)

            for doc_type_data in rules_data.get('document_types', []):
                doc_type = doc_type_data['doc_type']
                rules = [Rule(**r) for r in doc_type_data.get('rules', [])]
                keywords = doc_type_data.get('keywords', [])
                
                rule_store[doc_type] = RuleSet(
                    doc_type=doc_type,
                    keywords=keywords,
                    rules=rules
                )
            logger.info(f"Successfully loaded {len(rule_store)} rule sets from {self.rules_config_path}")
        except (yaml.YAMLError, FileNotFoundError, KeyError) as e:
            logger.error(f"Failed to load or parse rule configuration: {e}", exc_info=True)
        
        return rule_store

    def get_rules_for_doc_type(self, doc_type: str) -> Optional[RuleSet]:
        """Returns the RuleSet for a given document type."""
        return self._rule_store.get(doc_type)

    def get_all_rulesets(self) -> List[RuleSet]:
        """Returns all loaded RuleSets."""
        return list(self._rule_store.values())

    def apply_rules(self, text: str, doc_type: str) -> Dict:
        """
        Applies the rules for a given doc type to the text.
        This is a placeholder for the actual rule application logic.
        """
        ruleset = self.get_rules_for_doc_type(doc_type)
        if not ruleset:
            logger.warning(f"No rules found for document type: {doc_type}")
            return {"actions_taken": 0}

        logger.info(f"Applying {len(ruleset.rules)} rules for document type: {doc_type}")
        # Placeholder for real implementation
        return {"actions_taken": len(ruleset.rules), "doc_type": doc_type}

# Factory function for easy integration
def create_rule_manager(config: Optional[Dict[str, Any]] = None) -> ProcessingRuleManager:
    """Creates and returns a ProcessingRuleManager instance."""
    return ProcessingRuleManager(config) 