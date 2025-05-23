import fitz  # PyMuPDF

def redact_pdf(input_path: str, output_path: str, sensitive_words: list):
    """Redacts specified sensitive words from a PDF file using PyMuPDF.

    Args:
        input_path: Path to the original PDF file.
        output_path: Path to save the redacted PDF file.
        sensitive_words: A list of words to redact from the PDF.

    Returns:
        True if redaction was successful, False otherwise.
    """
    try:
        doc = fitz.open(input_path)
        for page in doc:
            for word in sensitive_words:
                areas = page.search_for(word)
                for area in areas:
                    page.add_redact_annot(area, fill=(0, 0, 0)) # Black out
            page.apply_redactions()
        doc.save(output_path, garbage=4, deflate=True, clean=True) # Save with options for smaller size and clean
        return True
    except Exception as e:
        print(f"Error during PDF redaction: {e}") # Added basic error logging
        return False 