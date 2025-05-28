from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.database import Base


class PDFDocument(Base):
    __tablename__ = "pdf_documents"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, nullable=False)
    anonymized_filename = Column(String, nullable=True)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    status = Column(String, nullable=False)  # pending, processing, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    error_message = Column(String, nullable=True)  # Only for actual errors
    redaction_report = Column(Text, nullable=True)  # Only for redaction reports (JSON string)
    processing_metadata = Column(JSON, nullable=True)  # Additional processing data
