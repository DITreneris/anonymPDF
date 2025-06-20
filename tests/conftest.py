import pytest
import tempfile
from pathlib import Path
from typing import Generator
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from starlette.testclient import TestClient

from app.main import app
from app.core.config_manager import ConfigManager, get_config_manager
from app.services.pdf_processor import PDFProcessor
from app.core.context_analyzer import ContextualValidator
from app.database import Base
from app.core.adaptive.pattern_db import AdaptivePatternDB
from app.core.adaptive.ab_testing import ABTestManager
from app.core.adaptive.coordinator import AdaptiveLearningCoordinator


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """
    Create a temporary, in-memory SQLite database session for each test function.
    This ensures test isolation.
    """
    # Use a file-based db for better cross-process/thread visibility if needed,
    # but for most unit/integration tests, in-memory is faster.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()  # Rollback any uncommitted transactions
        db.close()


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """
    Get a TestClient instance that reads/writes to the test database.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def adaptive_pattern_db(db_session: Session) -> Generator[AdaptivePatternDB, None, None]:
    """Fixture for the adaptive pattern database, using the test session."""
    pattern_db = AdaptivePatternDB(db_session=db_session)
    yield pattern_db


@pytest.fixture(scope="session")
def ab_test_manager(tmp_path_factory: pytest.TempPathFactory) -> Generator[ABTestManager, None, None]:
    """Fixture for the A/B test manager, using a temporary database file."""
    db_path = tmp_path_factory.mktemp("ab_test_data") / "ab_tests.db"
    manager = ABTestManager(db_path=db_path)
    yield manager
    manager.close()


@pytest.fixture(scope="session")
def adaptive_coordinator(
    adaptive_pattern_db: AdaptivePatternDB,
    ab_test_manager: ABTestManager,
    config_manager: ConfigManager
) -> AdaptiveLearningCoordinator:
    """Fixture for the Adaptive Learning Coordinator."""
    # This coordinator is now properly initialized with injected, session-scoped dependencies.
    coordinator = AdaptiveLearningCoordinator(
        pattern_db=adaptive_pattern_db,
        ab_test_manager=ab_test_manager,
        config_manager=config_manager
    )
    return coordinator


@pytest.fixture(scope="session")
def test_config_dir() -> Generator[Path, None, None]:
    """Create a temporary config directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "test_config"
        config_dir.mkdir(exist_ok=True)
        yield config_dir


@pytest.fixture(scope="session")
def test_config_manager(test_config_dir: Path) -> ConfigManager:
    """Create a test configuration manager with a valid, complete config."""
    # Create a settings file with all required sections to ensure validation passes
    settings_content = {
        "patterns": {}, "cities": {}, "settings": {}, "performance": {},
        "ml_integration": {}, "feedback_system": {}, "adaptive_learning": {}, "analytics": {}
    }
    settings_file = test_config_dir / "settings.yaml"
    with open(settings_file, 'w') as f:
        yaml.dump(settings_content, f)
        
    return ConfigManager(config_dir=test_config_dir)


@pytest.fixture(scope="session")
def config_manager(test_config_manager) -> ConfigManager:
    """Alias for test_config_manager for easier use in system tests."""
    return test_config_manager


@pytest.fixture(scope="session")
def test_pdf_processor(test_config_manager: ConfigManager) -> PDFProcessor:
    """Create a test PDF processor with test configuration."""
    # The constructor now requires the config manager.
    processor = PDFProcessor(config_manager=test_config_manager)
    return processor


@pytest.fixture
def sample_lithuanian_text() -> str:
    """Sample Lithuanian text with various PII patterns."""
    return """
    Vardas: Jonas Petraitis
    Asmens kodas: 38901234567
    El. paštas: jonas.petraitis@example.com
    Telefonas: +370 600 12345
    Tel. nr.: +370 600 55678
    Adresas: Gedimino pr. 25, LT-01103, Vilnius
    PVM kodas: LT100001738313
    IBAN: LT123456789012345678
    Gimimo data: 1989-01-23
    Automobilio numeris: ABC-123
    """


@pytest.fixture
def sample_english_text() -> str:
    """Sample English text with various PII patterns."""
    return """
    Name: John Smith
    Email: john.smith@example.com
    Phone: +1 555 123 4567
    SSN: 123-45-6789
    Credit Card: 4111 1111 1111 1111
    Date: 2024-01-15
    Address: 123 Main St, New York, NY 10001
    """


@pytest.fixture
def test_patterns() -> dict:
    """Test PII patterns for validation."""
    return {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "lithuanian_personal_code": r"\b[3-6]\d{10}\b",
        "lithuanian_phone_generic": r"\+370\s+\d{3}\s+\d{5}\b",
        "lithuanian_vat_code": r"\bLT\d{9,12}\b",
        "date_yyyy_mm_dd": r"\b\d{4}-\d{2}-\d{2}\b",
        "invalid_pattern": r"[unclosed_bracket",  # Invalid pattern for testing
    }


@pytest.fixture
def test_cities() -> list:
    """Test Lithuanian cities list."""
    return [
        "Vilnius",
        "Kaunas",
        "Klaipėda",
        "Šiauliai",
        "Panevėžys",
        "Alytus",
        "Marijampolė",
        "Utena",
        "Telšiai",
        "Tauragė",
    ]


@pytest.fixture
def temp_pdf_dir() -> Generator[Path, None, None]:
    """Create temporary directory for PDF test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_dir = Path(temp_dir) / "test_pdfs"
        pdf_dir.mkdir(exist_ok=True)
        yield pdf_dir


@pytest.fixture(autouse=True)
def cleanup_logs():
    """Clean up log files after each test."""
    yield
    # Clean up any test log files
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("test_*.log"):
            try:
                log_file.unlink()
            except Exception:
                pass


@pytest.fixture
def mock_spacy_models(monkeypatch):
    """Mock spaCy models for testing without requiring actual model files."""

    class MockDoc:
        def __init__(self, text):
            self.text = text
            self.ents = []

    class MockNLP:
        def __init__(self, name):
            self.name = name
            self.pipe_names = ["tagger", "parser", "ner"]

        def __call__(self, text):
            return MockDoc(text)

    def mock_load(model_name):
        if model_name in ["en_core_web_sm", "lt_core_news_sm"]:
            return MockNLP(model_name)
        else:
            raise OSError(f"Model {model_name} not found")

    monkeypatch.setattr("spacy.load", mock_load)
    return mock_load
