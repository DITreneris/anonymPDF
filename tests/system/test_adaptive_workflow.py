import pytest
import uuid
from pathlib import Path
from unittest.mock import Mock
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from typing import Tuple
from unittest.mock import MagicMock

from app.core.adaptive.coordinator import AdaptiveLearningCoordinator
from app.core.feedback_system import UserFeedback, FeedbackType, FeedbackSeverity
from app.services.pdf_processor import PDFProcessor
from app.core.adaptive.pattern_db import AdaptivePatternDB
from app.core.adaptive.ab_testing import ABTestManager
from app.core.config_manager import get_config_manager, ConfigManager
from app.core.adaptive.pattern_learner import PatternLearner

# Mark all tests in this file as system-level tests
pytestmark = pytest.mark.system

@pytest.fixture(scope="function")
def adaptive_learning_system(db_session: Session):
    """
    Provides a fully wired adaptive learning system for a single test function.
    This is the single source of truth for creating the system in tests.
    """
    config = get_config_manager()
    pattern_db = AdaptivePatternDB(db_session)
    pattern_learner = PatternLearner(pattern_db=pattern_db)
    ab_manager = Mock(spec=ABTestManager)
    
    coordinator = AdaptiveLearningCoordinator(
        pattern_db=pattern_db,
        ab_test_manager=ab_manager,
        config_manager=config
    )
    # The real pattern learner needs to be on the coordinator.
    coordinator.learner = pattern_learner

    # The PDFProcessor now gets its dependencies via the ConfigManager
    pdf_processor = PDFProcessor(config_manager=config)
    
    return coordinator, pdf_processor, pattern_db

def test_fixture_creation(adaptive_learning_system):
    """Fixture should initialize all components properly."""
    coordinator, pdf_processor, pattern_db = adaptive_learning_system
    assert coordinator is not None
    assert pdf_processor is not None
    assert pattern_db is not None

@pytest.mark.parametrize(
    "text_segment, corrected_category, language", [
        ("EMP-ID-98765", "EMPLOYEE_ID", "en"),
        ("PROJ-SECRET-ALPHA", "PROJECT_CODE", "en"),
        ("MIN-KORT-98765", "LT_MINISTRY_CARD", "lt"),
    ]
)
def test_feedback_learns_and_discovers_new_pattern(adaptive_learning_system, text_segment, corrected_category, language):
    """
    Tests that feedback teaches the system a new pattern, which is then used for discovery.
    """
    coordinator, pdf_processor, pattern_db = adaptive_learning_system
    test_text = f"The secret value is {text_segment}, please handle with care."

    # 1. Initial detection should NOT find the custom PII.
    initial_detections = pdf_processor.find_personal_info(test_text, language=language)
    # The returned structure is a dict of lists of tuples. We need to check the categories.
    initial_categories = {cat for cat, detections in initial_detections.items() if detections}
    assert corrected_category not in initial_categories, \
        f"Category '{corrected_category}' should not be detected before learning."

    # 2. Provide feedback to teach the system.
    feedback = UserFeedback(
        feedback_id=f"fb_{uuid.uuid4().hex}",
        document_id="doc_test_discover",
        text_segment=text_segment,
        detected_category="UNKNOWN",
        user_corrected_category=corrected_category,
        detected_confidence=0.0,
        user_confidence_rating=1.0,
        feedback_type=FeedbackType.CATEGORY_CORRECTION,
        severity=FeedbackSeverity.HIGH,
        user_comment="Test case for discovery",
        context={"full_text": test_text},
    )

    # Configure the coordinator to learn from a single sample via its config.
    # This is a more robust way to test than direct manipulation.
    coordinator.min_samples_for_learning = 1
    
    # Run the learning process
    coordinator.process_feedback_and_learn([feedback], [test_text])

    # 3. Verify the pattern was created and is active.
    active_patterns = pattern_db.get_active_patterns()
    assert any(p.pii_category == corrected_category for p in active_patterns), \
        "Adaptive pattern was not found in the active patterns after learning."

    # 4. Final detection SHOULD now find the custom PII.
    final_detections = pdf_processor.find_personal_info(test_text, language=language)
    final_categories = {cat for cat, detections in final_detections.items() if detections}
    assert corrected_category in final_categories, \
        "Learned category was not found in final detections."
    
    # The final detections are tuples of (text, confidence)
    detected_texts = [item[0] for item in final_detections.get(corrected_category, [])]
    assert text_segment in detected_texts, \
        "The specific PII text was not found in the final detections."

def test_feedback_api_is_disabled_for_now(client: TestClient):
    """
    This is a placeholder to confirm the API test is recognized but skipped.
    The real test depends on an endpoint that is not yet fully implemented.
    """
    assert client is not None
    pytest.skip("Skipping API test until '/api/v1/pdf/process' is implemented.")

@pytest.mark.parametrize(
    "sample_pii_document, new_pattern_test_case",
    [
        (
            """
            Vardas: Jonas Petraitis
            Asmens kodas: 38901234567
            El. paštas: jonas.petraitis@example.com
            Naujas sutarties numeris yra SUTARTIS-12345.
            Telefonas: +370 600 12345
            Adresas: Gedimino pr. 25, LT-01103, Vilnius
            """,
            ("SUTARTIS-12345", "CONTRACT_ID", "lt")
        )
    ]
)
@pytest.mark.system
def test_adaptive_workflow_learns_new_pattern(
    adaptive_coordinator: AdaptiveLearningCoordinator,
    adaptive_pattern_db: AdaptivePatternDB,
    config_manager: ConfigManager,
    sample_pii_document: str,
    new_pattern_test_case: Tuple[str, str, str]
):
    """
    A full end-to-end system test of the adaptive learning workflow.
    It verifies that feedback about a missed PII leads to the creation
    of a new, effective pattern.
    """
    # 1. Setup: Use the provided fixtures to create the main processor.
    pii_to_find, pattern_category, lang = new_pattern_test_case
    processor = PDFProcessor(config_manager=config_manager)

    # 2. Initial State: Verify the PII is NOT detected initially.
    initial_detections = processor.find_personal_info(sample_pii_document, language=lang)
    initial_texts = {item[0] for sublist in initial_detections.values() for item in sublist}
    assert pii_to_find not in initial_texts, "PII should not be detected before learning."

    # 3. Learning Step: Simulate user feedback to teach the system.
    feedback = UserFeedback(
        feedback_id=f"fb_{uuid.uuid4().hex}",
        document_id="doc_system_test",
        text_segment=pii_to_find,
        detected_category=None,
        user_corrected_category=pattern_category,
        detected_confidence=None,
        feedback_type=FeedbackType.MISSING_PII,
        user_confidence_rating=1.0,
        user_comment="System test feedback for new pattern."
    )
    
    # Process the feedback through the coordinator.
    adaptive_coordinator.process_feedback_and_learn([feedback], [sample_pii_document])

    # 4. Verification: Check that a new, active pattern was created.
    active_patterns = adaptive_pattern_db.get_active_patterns()
    new_pattern = next((p for p in active_patterns if p.pii_category == pattern_category), None)
    
    assert new_pattern is not None, "A new adaptive pattern was not created after feedback."
    assert new_pattern.is_active, "The new pattern should be active."
    assert new_pattern.confidence > 0.5, "The new pattern should have a reasonable confidence score."

    # 5. Final State: Verify the PII IS NOW detected by the processor.
    final_detections = processor.find_personal_info(sample_pii_document, language=lang)
    final_texts = {item[0] for sublist in final_detections.values() for item in sublist}
    assert pii_to_find in final_texts, "The new PII was not detected after the learning cycle."
