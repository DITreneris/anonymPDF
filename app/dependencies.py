from fastapi import Depends
from sqlalchemy.orm import Session
import functools
from functools import lru_cache

from app.core.adaptive.coordinator import AdaptiveLearningCoordinator
from app.core.real_time_monitor import get_real_time_monitor, RealTimeMonitor
from app.database import get_db, SessionLocal
from app.core.adaptive.pattern_db import AdaptivePatternDB
from app.services.pdf_processor import PDFProcessor
from app.core.config_manager import ConfigManager


@functools.lru_cache(maxsize=1)
def get_adaptive_learning_coordinator() -> AdaptiveLearningCoordinator:
    """
    Dependency to get a singleton instance of the AdaptiveLearningCoordinator.
    Using lru_cache ensures it's created only once.
    """
    return AdaptiveLearningCoordinator()


@lru_cache()
def get_config_manager() -> ConfigManager:
    """Dependency injector for ConfigManager."""
    return ConfigManager()


@lru_cache()
def get_pdf_processor() -> PDFProcessor:
    """
    Dependency injector for the PDFProcessor.
    Ensures that all dependencies are properly initialized and passed.
    """
    config_manager = get_config_manager()
    return PDFProcessor(config_manager=config_manager)


def get_db():
    """Dependency injector for the database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()