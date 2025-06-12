import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from app.core.adaptive.processing_rules import ProcessingRuleManager, RuleSet, Rule

@pytest.fixture
def mock_rules_config_content():
    return """
document_types:
  - doc_type: test_invoice
    keywords: ["invoice", "payment"]
    rules:
      - action: 'REDACT'
        target_category: 'BANK_ACCOUNT'
        confidence_threshold: 0.85
  - doc_type: test_medical
    keywords: ["patient", "doctor"]
    rules:
      - action: 'REDACT'
        target_category: 'PATIENT_NAME'
"""

@pytest.fixture
def mock_config_path(tmp_path: Path, mock_rules_config_content: str):
    config_file = tmp_path / "test_rules.yaml"
    config_file.write_text(mock_rules_config_content)
    return config_file

def test_load_rules_from_config_success(mock_config_path: Path):
    """
    Tests that the ProcessingRuleManager correctly loads rules from a valid YAML file.
    """
    config = {'rules_config_path': str(mock_config_path)}
    manager = ProcessingRuleManager(config=config)
    
    # Assert that rule sets were loaded
    assert len(manager.get_all_rulesets()) == 2
    
    # Assert invoice rules are correct
    invoice_ruleset = manager.get_rules_for_doc_type("test_invoice")
    assert isinstance(invoice_ruleset, RuleSet)
    assert invoice_ruleset.doc_type == "test_invoice"
    assert "invoice" in invoice_ruleset.keywords
    assert len(invoice_ruleset.rules) == 1
    assert invoice_ruleset.rules[0].action == 'REDACT'
    assert invoice_ruleset.rules[0].target_category == 'BANK_ACCOUNT'
    assert invoice_ruleset.rules[0].confidence_threshold == 0.85

    # Assert medical rules are correct
    medical_ruleset = manager.get_rules_for_doc_type("test_medical")
    assert isinstance(medical_ruleset, RuleSet)
    assert medical_ruleset.doc_type == "test_medical"
    assert "patient" in medical_ruleset.keywords
    assert len(medical_ruleset.rules) == 1
    assert medical_ruleset.rules[0].action == 'REDACT'

def test_load_rules_file_not_found():
    """
    Tests that the manager handles a missing configuration file gracefully.
    """
    config = {'rules_config_path': 'non_existent_file.yaml'}
    manager = ProcessingRuleManager(config=config)
    assert len(manager.get_all_rulesets()) == 0

def test_load_rules_bad_yaml(tmp_path: Path):
    """
    Tests that the manager handles a malformed YAML file gracefully.
    """
    bad_yaml_content = "doc_type: - malformed"
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(bad_yaml_content)
    
    config = {'rules_config_path': str(config_file)}
    manager = ProcessingRuleManager(config=config)
    assert len(manager.get_all_rulesets()) == 0

def test_get_all_rulesets(mock_config_path: Path):
    """
    Tests retrieving all loaded rulesets.
    """
    config = {'rules_config_path': str(mock_config_path)}
    manager = ProcessingRuleManager(config=config)
    all_rulesets = manager.get_all_rulesets()
    assert isinstance(all_rulesets, list)
    assert len(all_rulesets) == 2
    assert all(isinstance(rs, RuleSet) for rs in all_rulesets) 