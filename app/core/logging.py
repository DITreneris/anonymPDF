import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from datetime import datetime


class StructuredLogger:
    """Structured logger for AnonymPDF application with file rotation and contextual information."""

    def __init__(self, name: str, log_dir: str = "logs"):
        self.logger = logging.getLogger(name)

        # Create logs directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Configure file handler with rotation
        log_file = log_path / "anonympdf.log"
        handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)  # 10MB

        # Create formatter for structured logging
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Configure logger
        self.logger.addHandler(handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate logs
        self.logger.propagate = False

    def log_processing(self, filename: str, status: str, **kwargs):
        """Log PDF processing events with structured data."""
        extra_data = {
            "file_name": filename,  # Changed from 'filename' to avoid conflict
            "processing_status": status,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }

        message = f"PDF Processing: {status} - {filename}"
        if kwargs:
            message += f" - {json.dumps(kwargs, default=str)}"

        self.logger.info(message, extra=extra_data)

    def log_error(self, operation: str, error: Exception, **kwargs):
        """Log errors with full context and traceback."""
        safe_kwargs = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs,
        }
        extra_data = self._safe_extra_data(**safe_kwargs)

        message = f"Error in {operation}: {type(error).__name__} - {str(error)}"
        if kwargs:
            message += f" - {json.dumps(kwargs, default=str)}"

        self.logger.error(message, extra=extra_data, exc_info=True)

    def log_dependency_check(self, dependency: str, status: str, details: Optional[str] = None):
        """Log dependency validation results."""
        safe_kwargs = {"dependency": dependency, "dependency_status": status}

        if details:
            safe_kwargs["details"] = details

        extra_data = self._safe_extra_data(**safe_kwargs)

        message = f"Dependency Check: {dependency} - {status}"
        if details:
            message += f" - {details}"

        if status == "missing" or status == "failed":
            self.logger.error(message, extra=extra_data)
        else:
            self.logger.info(message, extra=extra_data)

    def log_database_operation(
        self, operation: str, table: str, record_id: Optional[int] = None, **kwargs
    ):
        """Log database operations with context."""
        safe_kwargs = {"db_operation": operation, "table": table, **kwargs}

        if record_id:
            safe_kwargs["record_id"] = record_id

        extra_data = self._safe_extra_data(**safe_kwargs)

        message = f"Database {operation}: {table}"
        if record_id:
            message += f" (ID: {record_id})"
        if kwargs:
            message += f" - {json.dumps(kwargs, default=str)}"

        self.logger.info(message, extra=extra_data)

    def info(self, message: str, **kwargs):
        """General info logging with optional structured data."""
        if kwargs:
            extra_data = self._safe_extra_data(**kwargs)
            message += f" - {json.dumps(kwargs, default=str)}"
            self.logger.info(message, extra=extra_data)
        else:
            self.logger.info(message)

    def _safe_extra_data(self, **kwargs):
        """Create safe extra data dict avoiding LogRecord reserved fields."""
        reserved_fields = {
            "filename",
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "module",
        }
        extra_data = {"log_timestamp": datetime.now().isoformat()}

        for key, value in kwargs.items():
            safe_key = f"custom_{key}" if key in reserved_fields else key
            extra_data[safe_key] = value

        return extra_data

    def warning(self, message: str, **kwargs):
        """Warning logging with optional structured data."""
        if kwargs:
            extra_data = self._safe_extra_data(**kwargs)
            message += f" - {json.dumps(kwargs, default=str)}"
            self.logger.warning(message, extra=extra_data)
        else:
            self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """Error logging with optional structured data."""
        if kwargs:
            extra_data = self._safe_extra_data(**kwargs)
            message += f" - {json.dumps(kwargs, default=str)}"
            self.logger.error(message, extra=extra_data)
        else:
            self.logger.error(message)

    def debug(self, message: str, **kwargs):
        """Debug logging with optional structured data."""
        if kwargs:
            extra_data = self._safe_extra_data(**kwargs)
            message += f" - {json.dumps(kwargs, default=str)}"
            self.logger.debug(message, extra=extra_data)
        else:
            self.logger.debug(message)


# Global logger instances for different modules
pdf_logger = StructuredLogger("anonympdf.pdf")
api_logger = StructuredLogger("anonympdf.api")
db_logger = StructuredLogger("anonympdf.database")
dependency_logger = StructuredLogger("anonympdf.dependencies")
