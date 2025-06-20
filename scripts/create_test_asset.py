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

if __name__ == "__main__":
    # This allows running the script directly to generate the asset
    txt_file = Path("tests/samples/simple_pii_document.txt")
    pdf_file = Path("tests/samples/simple_pii_document.pdf")
    create_test_pdf(txt_file, pdf_file) 