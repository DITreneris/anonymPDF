"""
API endpoint for receiving and processing user feedback.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db
from app.models.pdf_document import PDFDocument
from app.services.pdf_processor import PDFProcessor
from app.core.adaptive.coordinator import AdaptiveLearningCoordinator
from app.core.feedback_system import UserFeedback, FeedbackType, FeedbackSeverity
from app.dependencies import get_adaptive_learning_coordinator, get_pdf_processor
from app.core.logging import api_logger

router = APIRouter()

class FeedbackItem(BaseModel):
    text_segment: str
    original_category: str
    is_correct: bool
    # user_corrected_category: Optional[str] = None # For future use

class FeedbackPayload(BaseModel):
    document_id: int
    feedback_items: List[FeedbackItem]

@router.post("/feedback", status_code=202)
def submit_feedback(
    payload: FeedbackPayload,
    coordinator: AdaptiveLearningCoordinator = Depends(get_adaptive_learning_coordinator),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    db: Session = Depends(get_db)
):
    """
    Receives user feedback, loads the original document text,
    and passes everything to the adaptive learning system for processing.
    """
    api_logger.info(f"Received feedback for document {payload.document_id}", extra={"item_count": len(payload.feedback_items)})

    if not coordinator.is_enabled:
        api_logger.warning("Adaptive learning system is disabled. Feedback will not be processed.")
        return {"message": "Feedback received, but adaptive learning is disabled."}

    try:
        # Step 1: Retrieve the document record from the database
        db_document = db.query(PDFDocument).filter(PDFDocument.id == payload.document_id).first()
        if not db_document or not db_document.original_filename:
            raise HTTPException(status_code=404, detail={"message": "Original document not found", "code": "DOCUMENT_NOT_FOUND"})

        # Step 2: Construct the path and extract the full text
        # Assumes 'uploads' is the directory where original files are stored
        upload_dir = Path("uploads")
        file_path = upload_dir / db_document.original_filename.split('_', 1)[-1] # Strip the UUID
        
        # A more robust way would be to store the unique path in the DB
        # For now, we reconstruct it.
        # Let's find the actual unique file.
        files = list(upload_dir.glob(f"*_{db_document.original_filename}"))
        if not files:
             raise HTTPException(status_code=404, detail={"message": f"File {db_document.original_filename} not found in uploads directory", "code": "FILE_NOT_FOUND_ON_DISK"})
        
        actual_file_path = files[0]
        text_corpus = pdf_processor.extract_text_from_pdf(actual_file_path)

        # Step 3: Convert feedback payload to internal format
        feedback_list: List[UserFeedback] = []
        for item in payload.feedback_items:
            feedback_type = FeedbackType.CONFIRMED_PII if item.is_correct else FeedbackType.FALSE_POSITIVE
            
            feedback = UserFeedback(
                feedback_id=f"feedback_{payload.document_id}_{hash(item.text_segment)}",
                document_id=str(payload.document_id),
                text_segment=item.text_segment,
                detected_category=item.original_category,
                user_corrected_category=None, # Placeholder for future enhancement
                detected_confidence=0.5,  # Default confidence since not provided
                user_confidence_rating=None,
                feedback_type=feedback_type,
                severity=FeedbackSeverity.MEDIUM,  # Default severity
                user_comment=None,
                context={}  # Empty context for now
            )
            feedback_list.append(feedback)
        
        # Step 4: Pass the feedback and the context to the coordinator
        coordinator.process_feedback_and_learn(feedback_list, [text_corpus])
        
        api_logger.info("Feedback processed and sent to learning coordinator.", extra={"document_id": payload.document_id})

        return {"message": "Feedback submitted successfully and is being processed."}
    except HTTPException:
        # Re-raise HTTP exceptions (404, 422, etc.) as-is
        raise
    except Exception as e:
        api_logger.error("Failed to process feedback payload", exc_info=True)
        raise HTTPException(status_code=500, detail={"message": str(e), "code": "FEEDBACK_PROCESSING_ERROR"}) 