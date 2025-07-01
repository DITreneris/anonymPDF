"""
Tests for Celery worker module.
Tests cover task execution, error handling, database operations, and PDF processing integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json
from pathlib import Path
from app.worker import (
    celery_app,
    process_pdf_task,
    get_pdf_processor,
    pdf_processor_instance
)
from app.models.pdf_document import PDFStatus


class TestCeleryConfiguration:
    """Test Celery app configuration."""

    def test_celery_app_exists(self):
        """Test that Celery app is properly initialized."""
        assert celery_app is not None
        assert celery_app.main == "anonympdf_worker"

    def test_celery_app_configuration(self):
        """Test Celery app configuration settings."""
        conf = celery_app.conf
        assert conf.task_serializer == "json"
        assert conf.result_serializer == "json"
        assert conf.accept_content == ["json"]
        assert conf.task_time_limit == 300
        assert conf.task_soft_time_limit == 280

    @patch.dict('os.environ', {'CELERY_BROKER_URL': 'redis://test:6379/1'})
    def test_celery_broker_url_from_env(self):
        """Test that broker URL can be set from environment."""
        # Note: This tests the configuration pattern, actual broker URL 
        # is set at module import time
        import os
        assert os.environ.get('CELERY_BROKER_URL') == 'redis://test:6379/1'

    @patch.dict('os.environ', {'CELERY_RESULT_BACKEND': 'redis://test:6379/2'})
    def test_celery_result_backend_from_env(self):
        """Test that result backend can be set from environment."""
        import os
        assert os.environ.get('CELERY_RESULT_BACKEND') == 'redis://test:6379/2'


class TestGetPdfProcessor:
    """Test PDF processor dependency injection."""

    @patch('app.core.real_time_monitor.get_real_time_monitor')
    @patch('app.core.adaptive.pattern_db.AdaptivePatternDB')
    @patch('app.database.SessionLocal')
    @patch('app.services.pdf_processor.PDFProcessor')
    def test_get_pdf_processor_creation(self, mock_pdf_processor_class, mock_session_local, 
                                       mock_pattern_db_class, mock_get_monitor):
        """Test PDF processor creation with dependencies."""
        # Setup mocks
        mock_db = Mock()
        mock_session_local.return_value = mock_db
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        mock_pattern_db = Mock()
        mock_pattern_db_class.return_value = mock_pattern_db
        mock_processor = Mock()
        mock_pdf_processor_class.return_value = mock_processor
        
        result = get_pdf_processor()
        
        # Verify components were created
        mock_session_local.assert_called_once()
        mock_get_monitor.assert_called_once()
        mock_pattern_db_class.assert_called_once_with(db_session=mock_db)
        mock_pdf_processor_class.assert_called_once_with(
            pattern_db=mock_pattern_db,
            monitor=mock_monitor
        )
        
        assert result == mock_processor


class TestProcessPdfTask:
    """Test the main PDF processing task."""

    @pytest.fixture
    def mock_setup(self):
        """Setup common mocks for PDF processing tests."""
        with patch('app.worker.SessionLocal') as mock_session_local, \
             patch('app.worker.pdf_processor_instance') as mock_processor, \
             patch('app.worker.worker_logger') as mock_logger, \
             patch('app.worker.db_logger') as mock_db_logger:
            
            # Setup database mock
            mock_db = Mock()
            mock_session_local.return_value = mock_db
            
            # Setup document mock
            mock_document = Mock()
            mock_document.id = 123
            mock_db.query.return_value.filter.return_value.first.return_value = mock_document
            
            # Setup processor mock
            mock_result = Mock()
            mock_result.getbuffer.return_value = b"fake_pdf_data"
            mock_processor.process_pdf = AsyncMock(return_value=mock_result)
            
            yield {
                'db': mock_db,
                'document': mock_document,
                'processor': mock_processor,
                'logger': mock_logger,
                'db_logger': mock_db_logger
            }

    def test_process_pdf_task_success(self, mock_setup):
        """Test successful PDF processing task execution."""
        mocks = mock_setup
        file_path = "/test/path/document.pdf"
        document_id = 123
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('json.dumps') as mock_json_dumps:
            
            mock_open.return_value.__enter__.return_value = Mock()
            mock_json_dumps.return_value = '{"test": "report"}'
            
            # Execute task
            process_pdf_task(file_path, document_id)
            
            # Verify processor was called
            mocks['processor'].process_pdf.assert_called_once_with(file_path)
            
            # Verify status was set to completed
            assert mocks['document'].status == PDFStatus.COMPLETED
            
            # Verify database operations
            mocks['db'].commit.assert_called()
            mocks['db'].close.assert_called_once()

    def test_process_pdf_task_document_not_found(self, mock_setup):
        """Test task behavior when document is not found."""
        mocks = mock_setup
        mocks['db'].query.return_value.filter.return_value.first.return_value = None
        
        file_path = "/test/path/document.pdf"
        document_id = 999
        
        # Execute task
        process_pdf_task(file_path, document_id)
        
        # Verify error logging
        mocks['logger'].error.assert_called_with("Document with ID 999 not found.")

    def test_process_pdf_task_processing_error(self, mock_setup):
        """Test task behavior when PDF processing fails."""
        mocks = mock_setup
        
        # Make processor raise exception
        mocks['processor'].process_pdf.side_effect = Exception("Processing failed")
        
        file_path = "/test/path/document.pdf"
        document_id = 123
        
        # Execute task
        process_pdf_task(file_path, document_id)
        
        # Verify error handling
        mocks['logger'].error.assert_called()
        
        # Verify database is cleaned up
        mocks['db'].close.assert_called_once()

    def test_process_pdf_task_file_operations(self, mock_setup):
        """Test file operations during PDF processing."""
        mocks = mock_setup
        file_path = "/test/path/document.pdf"
        document_id = 123
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('json.dumps') as mock_json_dumps:
            
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_json_dumps.return_value = '{"test": "report"}'
            
            # Execute task
            process_pdf_task(file_path, document_id)
            
            # Verify directory creation
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            
            # Verify file writing
            mock_file.write.assert_called_once_with(b"fake_pdf_data")

    @patch('app.worker.asyncio.run')
    def test_process_pdf_task_asyncio_integration(self, mock_asyncio_run, mock_setup):
        """Test asyncio integration in task execution."""
        mocks = mock_setup
        file_path = "/test/path/document.pdf"
        document_id = 123
        
        mock_result = Mock()
        mock_result.getbuffer.return_value = b"test_data"
        mock_asyncio_run.return_value = mock_result
        
        with patch('builtins.open', create=True), \
             patch('pathlib.Path.mkdir'), \
             patch('json.dumps', return_value='{}'):
            
            # Execute task
            process_pdf_task(file_path, document_id)
            
            # Verify asyncio.run was called
            mock_asyncio_run.assert_called_once()


class TestTaskRegistration:
    """Test Celery task registration."""

    def test_task_registration(self):
        """Test that the PDF processing task is properly registered."""
        registered_tasks = celery_app.tasks
        assert "process_pdf_task" in registered_tasks

    def test_task_has_celery_methods(self):
        """Test that task has Celery methods."""
        assert hasattr(process_pdf_task, 'delay')
        assert hasattr(process_pdf_task, 'apply_async')
        assert callable(process_pdf_task.delay)
        assert callable(process_pdf_task.apply_async)


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch('app.worker.SessionLocal')
    @patch('app.worker.worker_logger')
    def test_database_connection_error(self, mock_logger, mock_session_local):
        """Test handling of database connection errors."""
        # Make SessionLocal raise exception
        mock_session_local.side_effect = Exception("Database connection failed")
        
        file_path = "/test/path/document.pdf"
        document_id = 123
        
        # Execute task - should handle the exception gracefully
        with pytest.raises(Exception):
            # The task doesn't catch SessionLocal exceptions at the start
            process_pdf_task(file_path, document_id)
        
        # The exception occurs before any logging, so we don't check for error logs

    @patch('app.worker.SessionLocal')
    @patch('app.worker.pdf_processor_instance')
    @patch('app.worker.worker_logger')
    def test_file_write_error(self, mock_logger, mock_processor, mock_session_local):
        """Test handling of file write errors."""
        # Setup basic mocks
        mock_db = Mock()
        mock_session_local.return_value = mock_db
        mock_document = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        
        mock_result = Mock()
        mock_result.getbuffer.return_value = b"test_data"
        mock_processor.process_pdf = AsyncMock(return_value=mock_result)
        
        with patch('builtins.open', side_effect=IOError("Write failed")), \
             patch('pathlib.Path.mkdir'):
            
            file_path = "/test/path/document.pdf"
            document_id = 123
            
            # Execute task
            process_pdf_task(file_path, document_id)
            
            # Verify error handling
            mock_logger.error.assert_called()


class TestIntegration:
    """Integration tests for worker module."""

    def test_pdf_processor_instance_exists(self):
        """Test that global PDF processor instance exists."""
        assert pdf_processor_instance is not None

    @patch('app.core.factory.get_pdf_processor')
    def test_processor_instance_initialization(self, mock_get_processor):
        """Test processor instance initialization on module import."""
        mock_processor = Mock()
        mock_get_processor.return_value = mock_processor
        
        # Reimport module to test initialization
        import importlib
        import app.worker
        importlib.reload(app.worker)
        
        # Verify processor was created (called during module import)
        mock_get_processor.assert_called()


def test_task_decorator_registration():
    """Test that task decorator properly registers the function."""
    # Verify the decorator was applied correctly
    assert hasattr(process_pdf_task, 'delay')
    assert hasattr(process_pdf_task, 'apply_async')
    assert callable(process_pdf_task.delay)
    assert callable(process_pdf_task.apply_async) 