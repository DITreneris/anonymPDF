from typing import Dict, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import os
from pathlib import Path
import uuid
from fastapi.responses import FileResponse
import json

from app.database import get_db
from app.models.pdf_document import PDFDocument
from app.schemas.pdf import PDFDocument as PDFDocumentSchema
from app.services.pdf_processor import PDFProcessor
from app.core.logging import api_logger, db_logger
from app.core.performance import get_performance_report

router = APIRouter()
pdf_processor = PDFProcessor()


@router.post("/upload", response_model=PDFDocumentSchema)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload and process a PDF file for anonymization."""
    api_logger.info(
        "Received PDF upload request", filename=file.filename, content_type=file.content_type
    )

    if not file.filename.lower().endswith(".pdf"):
        api_logger.warning(
            "Invalid file type uploaded", filename=file.filename, content_type=file.content_type
        )
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    upload_path = Path("uploads") / unique_filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    # Create database entry
    db_document = PDFDocument(
        original_filename=file.filename, anonymized_filename=None, file_size=0, status="pending"
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    db_logger.log_database_operation(
        operation="create",
        table="pdf_documents",
        record_id=db_document.id,
        filename=file.filename,
        status="pending",
    )

    try:
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        api_logger.info(
            "File saved successfully",
            filename=file.filename,
            path=str(upload_path),
            size_bytes=len(content),
        )

        db_document.file_size = os.path.getsize(upload_path)
        db_document.status = "processing"
        db.commit()

        db_logger.log_database_operation(
            operation="update",
            table="pdf_documents",
            record_id=db_document.id,
            status="processing",
            file_size=db_document.file_size,
        )

        # Process the PDF
        result = await pdf_processor.process_pdf(str(upload_path))

        api_logger.log_processing(
            filename=file.filename,
            status=result.get("status", "unknown"),
            processing_time=result.get("processing_time"),
            redactions_count=result.get("redactions_count"),
        )

        if result.get("status") == "processed":
            db_document.status = "completed"
            db_document.anonymized_filename = f"anonymized_{unique_filename}"
            # Store report in the new redaction_report field
            if result.get("report"):
                db_document.redaction_report = json.dumps(result.get("report"))
            # Store processing metadata
            if result.get("metadata"):
                db_document.processing_metadata = json.dumps(result.get("metadata"))
        else:
            db_document.status = "failed"
            db_document.error_message = result.get("error", "Processing failed")

        db.commit()
        db.refresh(db_document)

        db_logger.log_database_operation(
            operation="update",
            table="pdf_documents",
            record_id=db_document.id,
            status=db_document.status,
            final_status=True,
        )

        api_logger.info(
            "PDF processing completed successfully",
            filename=file.filename,
            document_id=db_document.id,
            status=db_document.status,
        )

        return db_document

    except Exception as e:
        api_logger.log_error(
            "pdf_upload_processing",
            e,
            filename=file.filename,
            document_id=db_document.id if "db_document" in locals() else None,
        )

        if "db_document" in locals():
            db_document.status = "failed"
            db_document.error_message = str(e)
            db.commit()

            db_logger.log_database_operation(
                operation="update",
                table="pdf_documents",
                record_id=db_document.id,
                status="failed",
                error=str(e),
            )

        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=404, detail="Document not found")
    data = document.__dict__.copy()
    if document.anonymized_filename and document.status == "completed":
        data["downloadUrl"] = f"/api/v1/download/{document.anonymized_filename}"
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
        raise HTTPException(status_code=404, detail="Document not found")

    if document.status != "completed":
        api_logger.warning(
            "Report requested for incomplete document",
            document_id=document_id,
            status=document.status,
        )
        raise HTTPException(status_code=400, detail="Document processing not completed")

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
        raise HTTPException(status_code=404, detail="File not found")
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
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")
