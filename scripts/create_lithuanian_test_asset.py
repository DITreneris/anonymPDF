import fitz
from pathlib import Path

def create_lithuanian_pii_pdf():
    """Creates a simple PDF with Lithuanian PII for testing."""
    
    # Correctly locate the 'tests/samples' directory relative to the project root
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "tests" / "samples"
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
    
    doc.save(str(file_path))
    doc.close()
    
    print(f"Created test PDF: {file_path}")

if __name__ == "__main__":
    create_lithuanian_pii_pdf() 