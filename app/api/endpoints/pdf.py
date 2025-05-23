from typing import Dict, List, Optional, Union
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
import os
from pathlib import Path
import uuid
import traceback
from fastapi.responses import FileResponse

from app.database import get_db
from app.models.pdf_document import PDFDocument
from app.schemas.pdf import PDFDocument as PDFDocumentSchema
from app.services.pdf_processor import PDFProcessor

router = APIRouter()
pdf_processor = PDFProcessor()

@router.post("/upload", response_model=PDFDocumentSchema)
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a PDF file for anonymization."""
    print("Received upload request")
    if not file.filename.lower().endswith('.pdf'):
        print("File is not a PDF")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    upload_path = Path("uploads") / unique_filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    # Create database entry
    db_document = PDFDocument(
        original_filename=file.filename,
        anonymized_filename=None,
        file_size=0,
        status="pending"
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    print("Database entry created")
    
    try:
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        print(f"File saved to {upload_path}")
        
        db_document.file_size = os.path.getsize(upload_path)
        db_document.status = "processing"
        db.commit()
        print("File size updated in DB")
        
        # Process the PDF
        result = await pdf_processor.process_pdf(str(upload_path))
        print("PDF processed:", result)
        
        if result.get("status") == "processed":
            db_document.status = "completed"
            db_document.anonymized_filename = f"anonymized_{unique_filename}"
            db_document.error_message = result.get("report", "")
        else:
            db_document.status = "failed"
            db_document.error_message = "Processing failed"
        
        db.commit()
        db.refresh(db_document)
        print("Returning DB document")
        return db_document
        
    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
        db_document.status = "failed"
        db_document.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=List[PDFDocumentSchema])
def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all processed PDF documents."""
    documents = db.query(PDFDocument).offset(skip).limit(limit).all()
    return documents

@router.get("/documents/{document_id}")
def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
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
def get_document_report(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get the redaction report for a specific document."""
    document = db.query(PDFDocument).filter(PDFDocument.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.status != "completed":
        raise HTTPException(status_code=400, detail="Document processing not completed")
    
    return {"report": document.error_message}  # Using error_message field for now

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