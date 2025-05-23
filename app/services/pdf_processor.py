from typing import List, Optional, Tuple, Dict
import PyPDF2
from pdfminer.high_level import extract_text
from fastapi import UploadFile, HTTPException
import os
from pathlib import Path
import spacy
import re
from datetime import datetime
from langdetect import detect, LangDetectException
from app.pdf_processor import redact_pdf
import json

class PDFProcessor:
    def __init__(self):
        # Load English model by default
        self.nlp_en = spacy.load("en_core_web_sm")
        # Attempt to load Lithuanian model, handle if not found
        try:
            self.nlp_lt = spacy.load("lt_core_news_sm")
        except IOError:
            print("Warning: Lithuanian spaCy model 'lt_core_news_sm' not found. Lithuanian name detection will be limited.")
            self.nlp_lt = None # Fallback or use a multilingual model if preferred

        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        self.processed_dir = Path("processed")
        self.processed_dir.mkdir(exist_ok=True)
        
        # Define patterns for sensitive information
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', # Generic phone
            # Lithuanian Mobile Phone with Prefix "Tel.:" or "Tel. nr.:"
            'lithuanian_mobile_prefixed': r'Tel\.\s*(?:nr\.\s*)?\+370\s+\d\s+\d{3}\s+\d{4}\b',
            # Lithuanian Address with Prefix "Adresas:" and common street types (g., pr., al.)
            'lithuanian_address_prefixed': r'Adresas:\s*[^,]+\s*(?:g|pr|al)\.\s*[^,]+,\s*LT-\d{5}\s*,\s*[^\n\r]+',
            'lithuanian_personal_code': r'\b[3-6]\d{10}\b',
            'lithuanian_vat_code': r'\bLT\d{9}\b',
            'eleven_digit_numeric': r'\b\d{11}\b',
            'date_yyyy_mm_dd': r'\b\d{4}-\d{2}-\d{2}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        }

    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        try:
            # Use a sample of the text for detection
            sample = text[:1000] if len(text) > 1000 else text
            return detect(sample)
        except LangDetectException:
            return "unknown"

    def find_personal_info(self, text: str, language: str = 'en') -> Dict[str, List[Tuple[str, str]]]:
        """Find personal information in text using spaCy NER and regex patterns.
           Uses language-specific NLP model if available.
        """
        
        nlp_to_use = self.nlp_en # Default to English
        if language == 'lt' and self.nlp_lt:
            nlp_to_use = self.nlp_lt
        elif language == 'lt' and not self.nlp_lt:
            print("Warning: Lithuanian model not loaded, using English NLP for Lithuanian text.")

        doc = nlp_to_use(text)
        personal_info = {
            'names': [],
            'locations': [], # SpaCy GPE can also find locations/cities
            'organizations': [],
            'emails': [],
            'phones': [], # For generic phone numbers
            'mobile_phones_prefixed': [], # For Tel. nr.: ...
            'addresses_prefixed': [], # For Adresas: ...
            'lithuanian_personal_codes': [],
            'lithuanian_vat_codes': [], # Added for VAT codes
            'eleven_digit_numerics': [],
            'dates_yyyy_mm_dd': [],
            'ssns': [],
            'credit_cards': []
        }
        
        # Extract entities using spaCy (PERSON, GPE, ORG)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                personal_info['names'].append((ent.text, ent.label_))
            elif ent.label_ == "GPE":
                personal_info['locations'].append((ent.text, ent.label_))
            elif ent.label_ == "ORG":
                personal_info['organizations'].append((ent.text, ent.label_))
        
        # Extract sensitive information using regex patterns
        for pattern_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                if pattern_type == 'email':
                    personal_info['emails'].append((match.group(), 'EMAIL'))
                elif pattern_type == 'phone':
                    personal_info['phones'].append((match.group(), 'PHONE'))
                elif pattern_type == 'lithuanian_mobile_prefixed':
                    personal_info['mobile_phones_prefixed'].append((match.group(), 'MOBILE_PHONE_PREFIXED'))
                elif pattern_type == 'lithuanian_address_prefixed':
                    personal_info['addresses_prefixed'].append((match.group(), 'ADDRESS_PREFIXED'))
                elif pattern_type == 'lithuanian_personal_code':
                    personal_info['lithuanian_personal_codes'].append((match.group(), 'LITHUANIAN_PERSONAL_CODE'))
                elif pattern_type == 'lithuanian_vat_code':
                    personal_info['lithuanian_vat_codes'].append((match.group(), 'LITHUANIAN_VAT_CODE'))
                elif pattern_type == 'eleven_digit_numeric':
                    # Avoid double-counting if already caught as a personal code
                    is_already_personal_code = False
                    for pc_match, _ in personal_info['lithuanian_personal_codes']:
                        if match.group() == pc_match:
                            is_already_personal_code = True
                            break
                    if not is_already_personal_code:
                        personal_info['eleven_digit_numerics'].append((match.group(), 'ELEVEN_DIGIT_NUMERIC'))
                elif pattern_type == 'date_yyyy_mm_dd':
                    personal_info['dates_yyyy_mm_dd'].append((match.group(), 'DATE_YYYY_MM_DD'))
                elif pattern_type == 'ssn':
                    personal_info['ssns'].append((match.group(), 'SSN'))
                elif pattern_type == 'credit_card':
                    personal_info['credit_cards'].append((match.group(), 'CREDIT_CARD'))
        
        return personal_info

    def anonymize_pdf(self, input_path: Path, output_path: Path) -> Tuple[bool, Dict]:
        """Anonymize PDF by redacting personal information."""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(input_path)
            
            # Detect language
            language = self.detect_language(text)
            print(f"Detected language: {language}") # Logging detected language
            
            # Find personal information using the detected language
            personal_info = self.find_personal_info(text, language=language)
            
            # Collect all sensitive words
            sensitive_words = []
            for category_items in personal_info.values():
                for item_text, _ in category_items:
                    sensitive_words.append(item_text)
            
            # Redact the PDF using PyMuPDF
            redaction_successful = redact_pdf(str(input_path), str(output_path), sensitive_words)
            
            if not redaction_successful:
                # Return a structured error message if redaction fails
                return False, {"error": "PDF redaction failed", "details": "redact_pdf returned False"}

            # Generate report of redacted information
            report = self.generate_redaction_report(personal_info, language)
            
            return True, report
            
        except Exception as e:
            # Return a structured error message for other exceptions
            return False, {"error": "An exception occurred during anonymization", "details": str(e)}

    def generate_redaction_report(self, personal_info: Dict[str, List[Tuple[str, str]]], language: str) -> Dict:
        """Generate a structured report of redacted information."""
        total_redactions = 0
        categories = {}
        
        for category, items in personal_info.items():
            if items:
                category_name = category.upper()
                categories[category_name] = len(items)
                total_redactions += len(items)
        
        report_data = {
            "title": f"Redaction Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "detectedLanguage": language,
            "totalRedactions": total_redactions,
            "categories": categories,
            "details": personal_info # Include raw details for potential future use or deeper inspection
        }
        return report_data

    async def process_pdf(self, file_path: str) -> dict:
        """Process PDF file at file_path and extract text."""
        if not str(file_path).lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        file_path_obj = Path(file_path)
        temp_path = self.temp_dir / file_path_obj.name
        try:
            # Copy the file to temp_path for processing
            with open(file_path_obj, "rb") as src, open(temp_path, "wb") as dst:
                dst.write(src.read())

            # Process the PDF
            success, report_data = self.anonymize_pdf(temp_path, self.processed_dir / f"anonymized_{file_path_obj.name}")
            
            if success:
                return {
                    "filename": file_path_obj.name,
                    "status": "processed",
                    "report": json.dumps(report_data)
                }
            else:
                error_detail = report_data.get("details", "Unknown error during PDF processing")
                if isinstance(report_data, dict) and "error" in report_data:
                    error_detail = json.dumps(report_data)
                raise HTTPException(status_code=500, detail=f"Failed to process PDF: {error_detail}")

        except Exception as e:
            error_report = {"error": "An exception occurred in process_pdf", "details": str(e)}
            raise HTTPException(status_code=500, detail=json.dumps(error_report))
        
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    def cleanup(self):
        """Clean up temporary files."""
        for file in self.temp_dir.glob("*"):
            if file.is_file():
                file.unlink()

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        return extract_text(str(pdf_path))

    def process_pdf_for_anonymization(self, file_path: Path) -> Tuple[bool, str]:
        """Process a PDF file for anonymization."""
        try:
            # Generate output filename
            output_filename = f"anonymized_{file_path.name}"
            output_path = self.processed_dir / output_filename
            
            # Process the PDF
            success, report_data = self.anonymize_pdf(file_path, output_path)
            
            if success:
                return True, str(output_path)
            else:
                return False, report_data.get("error", "Failed to anonymize PDF")
                
        except Exception as e:
            return False, str(e) 