from celery import Celery
from app.core.logging import worker_logger
import os
import json
from pathlib import Path
from app.database import SessionLocal
from app.models.pdf_document import PDFDocument, PDFStatus
from app.core.logging import db_logger
from app.core.factory import get_pdf_processor
import asyncio

# Configure Celery
# It's recommended to use a more robust broker like RabbitMQ for production
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "anonympdf_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Add a timeout for tasks to prevent them from running indefinitely
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=280,
)

# Create a global PDFProcessor instance for the worker
# This is recommended for performance to avoid re-initializing on every task.
# Use the new central factory to get the singleton instance.
pdf_processor_instance = get_pdf_processor()

def get_pdf_processor():
    """
    Dependency provider for the PDFProcessor.
    This allows us to override it during testing.
    """
    from app.core.real_time_monitor import get_real_time_monitor
    from app.core.adaptive.pattern_db import AdaptivePatternDB
    from app.services.pdf_processor import PDFProcessor
    
    # In a real scenario, you might get the db session from a pool
    # For now, we create it as needed, which is acceptable for the worker
    from app.database import SessionLocal
    db = SessionLocal()

    monitor = get_real_time_monitor()
    pattern_db = AdaptivePatternDB(db_session=db)
    return PDFProcessor(pattern_db=pattern_db, monitor=monitor)


@celery_app.task(name="process_pdf_task")
def process_pdf_task(file_path_str: str, document_id: int):
    """
    Celery task to process a PDF file in the background.
    """
    worker_logger.info(f"Worker received task for document {document_id} at path {file_path_str}")
    
    # Each task gets its own DB session and PDF processor instance
    db = SessionLocal()
    pdf_processor = None
    try:
        # Use the global processor instance
        result = asyncio.run(pdf_processor_instance.process_pdf(file_path_str))
        
        # Step 1: Get the document record from the database
        document = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
        if not document:
            worker_logger.error(f"Document with ID {document_id} not found.")
            return

        # Step 2: Update status to 'processing'
        document.status = PDFStatus.PROCESSING
        db.commit()
        db_logger.log_database_operation(
            "update_status", "pdf_documents", record_id=document.id, new_status=PDFStatus.PROCESSING
        )

        # Step 3: Save the redacted file
        anonymized_filename = f"redacted_{Path(file_path_str).name}"
        anonymized_path = Path("processed") / anonymized_filename
        anonymized_path.parent.mkdir(parents=True, exist_ok=True)
        with open(anonymized_path, "wb") as f:
            f.write(result.getbuffer())

        # Step 4: Update the document record with the results
        document.status = PDFStatus.COMPLETED
        document.anonymized_filename = anonymized_filename
        document.redaction_report = json.dumps(result)
        
        db_logger.log_database_operation(
            "update_on_completion", "pdf_documents", record_id=document.id, new_status=PDFStatus.COMPLETED
        )

    except Exception as e:
        worker_logger.error(f"Error processing document {document_id}: {e}", exc_info=True)
        # Step 6: Update status to 'failed' in case of error
        # Re-query the document within this session to ensure it's attached
        document_to_fail = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
        if document_to_fail:
            document_to_fail.status = PDFStatus.FAILED
            document_to_fail.error_message = str(e)
            db_logger.log_database_operation(
                "update_on_error", "pdf_documents", record_id=document_to_fail.id, new_status=PDFStatus.FAILED
            )
    finally:
        db.commit()
        db.close()
        worker_logger.info(f"Finished task for document {document_id}") 