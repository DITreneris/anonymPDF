"""
Singleton Factory Functions for Core Application Components.

This module provides centralized, thread-safe factory functions for creating
and accessing singleton instances of major application components like the
PDFProcessor, AdaptiveLearningCoordinator, and their dependencies.

Using these factories ensures that only one instance of each component exists
per application lifecycle, which is crucial for managing state, resources
(like database connections), and performance.
"""
from functools import lru_cache
from typing import Optional
import threading

# Import core components
from app.core.config_manager import ConfigManager, get_config_manager
from app.core.adaptive.pattern_db import AdaptivePatternDB
from app.core.adaptive.ab_testing import ABTestManager
from app.core.adaptive.coordinator import AdaptiveLearningCoordinator
from app.services.pdf_processor import PDFProcessor
from app.database import SessionLocal

# Thread-safe locks for singleton creation
_db_lock = threading.Lock()
_coordinator_lock = threading.Lock()
_processor_lock = threading.Lock()
_ab_test_lock = threading.Lock()

# Global singleton instances
_pattern_db_instance: Optional[AdaptivePatternDB] = None
_ab_test_manager_instance: Optional[ABTestManager] = None
_coordinator_instance: Optional[AdaptiveLearningCoordinator] = None
_pdf_processor_instance: Optional[PDFProcessor] = None


@lru_cache(maxsize=1)
def get_pattern_db() -> AdaptivePatternDB:
    """
    Factory function to get a singleton instance of the AdaptivePatternDB.
    Uses a lock to ensure thread-safe instantiation.
    """
    global _pattern_db_instance
    with _db_lock:
        if _pattern_db_instance is None:
            # The pattern DB needs a database session to operate.
            # We create a new session here for its lifecycle.
            db_session = SessionLocal()
            _pattern_db_instance = AdaptivePatternDB(db_session=db_session)
    return _pattern_db_instance


@lru_cache(maxsize=1)
def get_ab_test_manager() -> ABTestManager:
    """
    Factory function to get a singleton instance of the ABTestManager.
    Uses a lock to ensure thread-safe instantiation.
    """
    global _ab_test_manager_instance
    with _ab_test_lock:
        if _ab_test_manager_instance is None:
            # ABTestManager manages its own database connection internally
            _ab_test_manager_instance = ABTestManager()
    return _ab_test_manager_instance


@lru_cache(maxsize=1)
def get_coordinator(config_manager: Optional[ConfigManager] = None) -> AdaptiveLearningCoordinator:
    """

    Factory function to get a singleton instance of the AdaptiveLearningCoordinator.
    Uses a lock to ensure thread-safe instantiation.
    """
    global _coordinator_instance
    with _coordinator_lock:
        if _coordinator_instance is None:
            # The coordinator depends on the pattern DB, A/B test manager, and config manager
            pattern_db = get_pattern_db()
            ab_test_manager = get_ab_test_manager()
            # If no config manager is passed, get the default singleton
            cfg_manager = config_manager or get_config_manager()

            _coordinator_instance = AdaptiveLearningCoordinator(
                pattern_db=pattern_db,
                ab_test_manager=ab_test_manager,
                config_manager=cfg_manager
            )
    return _coordinator_instance


@lru_cache(maxsize=1)
def get_pdf_processor() -> PDFProcessor:
    """
    Factory function to get a singleton instance of the PDFProcessor.
    Uses a lock to ensure thread-safe instantiation.
    """
    global _pdf_processor_instance
    with _processor_lock:
        if _pdf_processor_instance is None:
            # The processor depends on the config manager and the adaptive coordinator
            config_manager = get_config_manager()
            coordinator = get_coordinator(config_manager)

            _pdf_processor_instance = PDFProcessor(
                config_manager=config_manager,
                coordinator=coordinator
            )
    return _pdf_processor_instance 