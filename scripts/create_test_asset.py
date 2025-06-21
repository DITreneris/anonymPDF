import fitz  # PyMuPDF
from pathlib import Path

def create_test_pdf(txt_path: Path, pdf_path: Path):
    """
    Creates a simple PDF from a text file to be used in tests.
    Using a basic font and layout to ensure reliable text extraction.
    """
    try:
        # Read text content from the source file
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Create a new PDF document
        doc = fitz.open()
        page = doc.new_page()

        # Use a reliable, basic font like Courier
        # Insert text into a specified rectangle (bbox)
        page.insert_textbox(page.rect, text, fontname="cour", fontsize=11)

        # Save the document
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(pdf_path))
        doc.close()
        print(f"Successfully created test PDF: {pdf_path}")

    except Exception as e:
        print(f"Error creating test PDF: {e}")

def create_lithuanian_pii_pdf():
    """Creates a simple PDF with Lithuanian PII for testing."""
    
    output_dir = Path(__file__).parent.parent / "tests" / "samples"
    output_dir.mkdir(exist_ok=True)
    
    file_path = output_dir / "lithuanian_pii_document.pdf"

    if file_path.exists():
        print(f"File already exists: {file_path}")
        return

    doc = fitz.open()
    page = doc.new_page()

    # Add some text with common Lithuanian PII
    text = """
    Vardenis Pavardenis
    Asmens kodas: 38801011234
    El. paštas: vardenis.pavardenis@email.com
    Adresas: Gedimino pr. 9, Vilnius, Lietuva
    Telefonas: +370 688 12345
    """
    
    page.insert_text((72, 72), text, fontsize=11)
    
    doc.save(file_path)
    doc.close()
    
    print(f"Created test PDF: {file_path}")

if __name__ == "__main__":
    # This allows running the script directly to generate the asset
    txt_file = Path("tests/samples/simple_pii_document.txt")
    pdf_file = Path("tests/samples/simple_pii_document.pdf")
    create_test_pdf(txt_file, pdf_file)
    create_lithuanian_pii_pdf() 