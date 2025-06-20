from typing import Dict, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
import os
from pathlib import Path
import uuid
from fastapi.responses import FileResponse
import json

from app.database import get_db
from app.models.pdf_document import PDFDocument, PDFStatus
from app.schemas.pdf import PDFDocument as PDFDocumentSchema
from app.services.pdf_processor import PDFProcessor
from app.core.logging import api_logger, db_logger
from app.core.performance import get_performance_report
from app.version import __version__
from app.worker import process_pdf_task
from app.core.intelligent_cache import cache_manager
from app.dependencies import get_pdf_processor

router = APIRouter()


@router.get("/version")
def get_version():
    """Returns the application version."""
    return {"version": __version__}


@router.post("/upload", response_model=PDFDocumentSchema, status_code=202)
def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts a PDF file, saves it, creates a DB record, and dispatches
    a background task for processing. Returns immediately.
    """
    api_logger.info(
        "Received PDF upload request", filename=file.filename, content_type=file.content_type
    )

    if not file.filename.lower().endswith(".pdf"):
        api_logger.warning(
            "Invalid file type uploaded", filename=file.filename, content_type=file.content_type
        )
        raise HTTPException(
            status_code=400, 
            detail={"message": "Only PDF files are allowed", "code": "INVALID_FILE_TYPE"}
        )

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    upload_path = Path("uploads") / unique_filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the file locally first
    try:
        with open(upload_path, "wb") as buffer:
            # Not reading the whole file into memory, chunk it
            for chunk in file.file:
                buffer.write(chunk)
        file_size = os.path.getsize(upload_path)
        api_logger.info("File saved to temporary location", path=str(upload_path), size=file_size)
    except Exception as e:
        api_logger.log_error("file_save_error", e, filename=file.filename)
        raise HTTPException(
            status_code=500,
            detail={"message": f"Could not save file: {e}", "code": "FILE_SAVE_ERROR"}
        )

    # Create database entry with 'queued' status
    db_document = PDFDocument(
        original_filename=file.filename, 
        anonymized_filename=None, 
        file_size=file_size, 
        status=PDFStatus.QUEUED
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # Convert the SQLAlchemy model to a Pydantic schema immediately
    # This loads all required attributes while the session is still active
    response_data = PDFDocumentSchema.from_orm(db_document)

    db_logger.log_database_operation(
        operation="create_for_processing",
        table="pdf_documents",
        record_id=response_data.id,
        filename=file.filename,
        status=PDFStatus.QUEUED,
    )

    # Dispatch the background task with the document ID
    process_pdf_task.delay(str(upload_path), response_data.id)
    api_logger.info(
        "Dispatched PDF processing task to worker",
        document_id=response_data.id,
        task_name=process_pdf_task.name
    )

    return response_data


@router.get("/documents", response_model=List[PDFDocumentSchema])
def list_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all processed PDF documents."""
    documents = db.query(PDFDocument).offset(skip).limit(limit).all()
    return documents


@router.get("/documents/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get a specific PDF document by ID, with downloadUrl if available."""
    document = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
    if document is None:
        raise HTTPException(
            status_code=404, 
            detail={"message": "Document not found", "code": "DOCUMENT_NOT_FOUND"}
        )
    data = document.__dict__.copy()
    if document.anonymized_filename and document.status == "completed":
        data["downloadUrl"] = f"/api/v1/pdf/download/{document.anonymized_filename}"
    else:
        data["downloadUrl"] = None
    return data


@router.get("/documents/{document_id}/report")
def get_document_report(document_id: int, db: Session = Depends(get_db)):
    """Get the redaction report for a specific document."""
    api_logger.info("Fetching document report", document_id=document_id)

    document = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
    if document is None:
        api_logger.warning("Document not found for report request", document_id=document_id)
        raise HTTPException(
            status_code=404,
            detail={"message": "Document not found", "code": "DOCUMENT_NOT_FOUND"}
        )

    if document.status != "completed":
        api_logger.warning(
            "Report requested for incomplete document",
            document_id=document_id,
            status=document.status,
        )
        raise HTTPException(
            status_code=400,
            detail={"message": "Document processing not completed", "code": "DOCUMENT_NOT_COMPLETED"}
        )

    # Parse the JSON report from the redaction_report field
    report_data = None
    if document.redaction_report:
        try:
            report_data = json.loads(document.redaction_report)
        except json.JSONDecodeError as e:
            api_logger.error(
                "Failed to parse redaction report JSON", document_id=document_id, error=str(e)
            )
            report_data = {"error": "Report data is corrupted"}

    api_logger.info("Document report retrieved successfully", document_id=document_id)
    return {"report": report_data}


@router.get("/health")
async def health_check() -> Dict:
    """
    Check PDF processing service health
    """
    return {"status": "healthy"}


@router.get("/download/{filename}")
def download_pdf(filename: str):
    file_path = Path("processed") / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"message": "File not found", "code": "FILE_NOT_FOUND"}
        )
    return FileResponse(str(file_path), media_type="application/pdf", filename=filename)


@router.get("/performance")
def get_performance_metrics() -> Dict:
    """Get performance metrics and system information."""
    api_logger.info("Performance metrics requested")
    
    try:
        performance_data = get_performance_report()
        
        api_logger.info(
            "Performance metrics retrieved successfully",
            operations_tracked=len(performance_data.get("operation_stats", {})),
        )
        
        return performance_data
    except Exception as e:
        api_logger.log_error("get_performance_metrics", e)
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to retrieve performance metrics", "code": "PERFORMANCE_ERROR"}
        )
