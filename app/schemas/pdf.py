from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class PDFDocumentBase(BaseModel):
    original_filename: str
    file_size: int


class PDFDocumentCreate(PDFDocumentBase):
    pass


class PDFDocument(PDFDocumentBase):
    id: int
    anonymized_filename: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    redaction_report: Optional[str] = None

    class Config:
        from_attributes = True
